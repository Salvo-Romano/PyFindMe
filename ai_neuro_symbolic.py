import torch
import torch.nn as nn
import torch.nn.functional as F
import random as r
import threading
from concurrent.futures import ThreadPoolExecutor
from CTL_bridge import build_horizon_tree, generate_vitamin_model, get_state_signature
from pyFighters import Action

r.seed(42)

class MetaControllerLSTM(nn.Module):
    def __init__(self, state_dim, vocab_size, hidden_dim=64, embedding_dim=16):
        super(MetaControllerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(state_dim + embedding_dim, hidden_dim, batch_first=True)
        
        # ACTOR HEAD: genera le probabilità per il vocabolario CTL
        self.actor_head = nn.Linear(hidden_dim, vocab_size)
        
        # CRITIC HEAD: stima il valore tattico dello stato fisico corrente (V(s))
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, game_state_tensor, token_id, hidden):
        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
        embeds = self.embedding(token_tensor)  
        state_expanded = game_state_tensor.unsqueeze(1) 
        lstm_input = torch.cat((state_expanded, embeds), dim=2)
        out, new_hidden = self.lstm(lstm_input, hidden)
        
        logits = self.actor_head(out.squeeze(1))
        return logits, new_hidden

    def evaluate_state(self, game_state_tensor):
        """Metodo esclusivo del Critic per calcolare il valore dello stato"""
        return self.critic_head(game_state_tensor)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


class NeuroSymbolicAI:
    def __init__(self, vitamin_client, user_predictor, vocab, masker, state_dim=8):
        self.vitamin = vitamin_client
        self.user_predictor = user_predictor
        self.vocab = vocab
        self.masker = masker
        
        self.policy_net = MetaControllerLSTM(state_dim=state_dim, vocab_size=vocab.get_size())
        
        self.current_plan = []
        self.expected_current_signature = None
        self.last_expected_player_action = None
        self.last_prob = 0.0
        self.horizon_tree = None
        
        # Variabili di stato per il PPO
        self.last_log_probs = []
        self.last_formula = None
        self.last_state_value = None # <--- Nuovo contenitore per il calcolo dell'Advantage
        
        self.is_thinking = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="NeuroSym_Worker")


    def _get_state_tensor(self, game_state):

        sig = get_state_signature(game_state, 0)
        tensor = torch.tensor(list(sig[1:]), dtype=torch.float32)
        return (tensor / 3.0).unsqueeze(0)


    def _generate_formula(self, game_state):
        self.masker.reset()
        formula_tokens = []
        saved_log_probs = []
        
        state_tensor = self._get_state_tensor(game_state)
        
        # Calcolo istantaneo del Valore di Stato tramite il Critic
        state_value = self.policy_net.evaluate_state(state_tensor)
        
        hidden = self.policy_net.init_hidden()
        current_token_id = 0 
        
        max_tokens = 10 
        reached_eos = False
        
        for _ in range(max_tokens): 
            logits, hidden = self.policy_net(state_tensor, current_token_id, hidden)
            
            valid_mask = torch.tensor(self.masker.get_valid_mask(), dtype=torch.float32)
            masked_logits = logits.masked_fill(valid_mask == 0, -float('inf'))
            
            if not self.policy_net.training:
                temperature = 0.5
                scaled_logits = masked_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                chosen_token_id = torch.multinomial(probs, 1).item()
            else:
                probs = F.softmax(masked_logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                chosen_token_id_tensor = m.sample()
                chosen_token_id = chosen_token_id_tensor.item()
                saved_log_probs.append(m.log_prob(chosen_token_id_tensor))
                
            token_str = self.masker.update_state(chosen_token_id)
            if token_str == '<EOS>':
                reached_eos = True
                break
                
            formula_tokens.append(token_str)
            current_token_id = chosen_token_id
            
        if not reached_eos:
            return "INVALID_TRUNCATED", saved_log_probs, state_value
            
        return " ".join(formula_tokens), saved_log_probs, state_value


    def _evaluate_transition(self, current_state, next_state):

        damage_dealt = current_state.player.hp - next_state.player.hp
        damage_taken = current_state.npc.hp - next_state.npc.hp
        sp_gained = next_state.npc.sp - current_state.npc.sp

        score = (damage_dealt * 5.0) + (sp_gained * 1.5) - (damage_taken * 3.0)

        if current_state.npc.sp >= current_state.npc.stats["sp_threshold"] and next_state.npc.sp < current_state.npc.sp:
            score += 50.0  
   
        if next_state.player.hp <= 0:
            score += 1000.0  

        return score


    def _extract_plan_from_trace(self, response, root_node, nodes_list):

        plan = []
        valid_state_names = response.get("verification", {}).get("satisfying_states", [])

        if not valid_state_names:
            return plan
           
        safe_nodes = set()

        for name in valid_state_names:
            try:
                idx = int(name.replace("s", ""))
                safe_nodes.add(nodes_list[idx])
            except (ValueError, IndexError):
                pass   

        current = root_node

        while current.transitions:
            action_evals = {}
            for p_act, n_act, prob, child in current.transitions:

                if n_act not in action_evals:
                    action_evals[n_act] = {
                        'expected_utility': 0.0,
                        'max_p_prob': -1.0,
                        'best_p_act_for_logs': None,
                        'target_node': None
                    }

                score = self._evaluate_transition(current.state, child.state)
                action_evals[n_act]['expected_utility'] += prob * score

                if prob > action_evals[n_act]['max_p_prob']:
                    action_evals[n_act]['max_p_prob'] = prob
                    action_evals[n_act]['best_p_act_for_logs'] = p_act
                    action_evals[n_act]['target_node'] = child

            best_n_act = None
            best_ev = -float('inf')
            best_data = None

            for n_act, data in action_evals.items():
                if data['target_node'] in safe_nodes and data['expected_utility'] > best_ev:
                    best_ev = data['expected_utility']
                    best_n_act = n_act
                    best_data = data

            if best_n_act and best_data:
                plan.append({
                    "npc_action": best_n_act,
                    "expected_player_action": best_data['best_p_act_for_logs'],
                    "probability": best_data['max_p_prob'],
                    "expected_signature": get_state_signature(best_data['target_node'].state, best_data['target_node'].depth),
                    "expected_props": list(best_data['target_node'].atomic_props)
                })
                current = best_data['target_node']
            else:
                break
        return plan


    def _async_compute(self, game_state, last_p_act, last_n_act):
        # Estraiamo anche lo state_value e lo salviamo nella classe
        generated_formula, saved_log_probs, state_value = self._generate_formula(game_state)
        
        self.last_log_probs = saved_log_probs
        self.last_formula = generated_formula
        self.last_state_value = state_value 
        
        if generated_formula == "INVALID_TRUNCATED":
            print("    [META-CONTROLLER] ⚠️ Formula troncata. Scartata in locale.")
            self.current_plan = [] 
            self.is_thinking = False
            return "TRUNCATED"
        
        p_ratio = game_state.player.hp / game_state.player.stats["max_hp"]
        n_ratio = game_state.npc.hp / game_state.npc.stats["max_hp"]
        target_depth = 3 if (p_ratio <= 0.5 or n_ratio <= 0.5) else 2

        self.horizon_tree = build_horizon_tree(
            game_state, self.user_predictor,
            max_depth=target_depth, prune_threshold=0.15,
            use_dag=True, last_p_act=last_p_act, last_n_act=last_n_act
        )

        model_text = generate_vitamin_model(self.horizon_tree)
        nodes = []


        def collect_nodes(node):
            if node not in nodes:
                nodes.append(node)
                for _, _, _, child in node.transitions:
                    collect_nodes(child)

        collect_nodes(self.horizon_tree)

        is_satisfied, response = self.vitamin.verify_formula(model_text, generated_formula, generate_trace=True)


        if is_satisfied:
            safe_states = response.get("verification", {}).get("satisfying_states", [])
            print(f"    [VITAMIN] ✅ Formula '{generated_formula}' SODDISFATTA! Trovati {len(safe_states)} stati sicuri.")

            self.current_plan = self._extract_plan_from_trace(response, self.horizon_tree, nodes)

            if self.current_plan:
                self.is_thinking = False
                return "SUCCESS"
            else:
                print(f"    [VITAMIN] ⚠️ Formula soddisfatta, ma l'Expectimax non ha estratto un piano valido dal DAG.")
                return "REJECTED"
        else:
            error_msg = response.get("error", "Proprietà falsa per tutti gli stati iniziali o sintassi non valida.")
            print(f"    [VITAMIN] ❌ Formula '{generated_formula}' RIFIUTATA. Motivo: {error_msg}")
            return "REJECTED"


    def evaluate_post_turn(self, game_state, last_player_action, last_npc_action):
       
        if self.is_thinking:
            return
        
        current_signature = get_state_signature(game_state, current_depth=0)

        if self.expected_current_signature:
            if current_signature[1:] != self.expected_current_signature[1:]:
                self.current_plan = []
                self.expected_current_signature = None

        if not self.current_plan:
            self.is_thinking = True
            self.executor.submit(self._async_compute, game_state, last_player_action, last_npc_action)



    def decide_action(self, game_state, last_player_action=None, last_npc_action=None):

        if self.is_thinking:
            return "THINKING"

        if self.current_plan:
            step = self.current_plan.pop(0)
            self.expected_current_signature = step["expected_signature"]
            self.last_expected_player_action = step.get("expected_player_action")
            self.last_prob = step.get("probability", 0.0)
            action = step["npc_action"]

            return action

        self.is_thinking = True
        self.executor.submit(self._async_compute, game_state, last_player_action, last_npc_action)

        return "THINKING" 


class DummyAI:
    """
    Fase 1: Sacco da boxe. 
    Non fa danni, carica solo la speciale o si buffa. 
    Insegna all'agente che la tartaruga porta allo stallo e bisogna attaccare per vincere.
    """
    def decide_action(self,game_state, npc_role="player"):
        
        action = Action.BUFF if game_state.is_buff_allowed(game_state.player) else Action.ATTACK
        # Assumendo che esista Action.CHARGE o similare nel tuo enum Action
        return action if r.randint(1,100) > 50 else Action.DEFEND