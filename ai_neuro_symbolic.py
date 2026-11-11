import torch
import torch.nn as nn
import torch.nn.functional as F
import random as r
import threading
from concurrent.futures import ThreadPoolExecutor
from CTL_bridge import build_horizon_tree, generate_vitamin_model, get_state_signature
from pyFighters import Action

r.seed(42)

# Portafoglio fisso di formule canoniche per l'Ablazione 1 (FIXED_FORMULAS)
FIXED_PORTFOLIO = [
    "E F Dead_P",
    "E F CanKill_N",
    "E F LowHP_P",
    "E F HP_Advantage_N",
    "E F SpecialReady_N",
    "A G CounterReady_N"
]


class MetaControllerLSTM(nn.Module):
    def __init__(self, state_dim, vocab_size, hidden_dim=64, embedding_dim=16, portfolio_size=len(FIXED_PORTFOLIO)):
        super(MetaControllerLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(state_dim + embedding_dim, hidden_dim, batch_first=True)
        
        # ACTOR HEAD 1 (Token-level): genera le probabilità per il vocabolario CTL (FULL e FIXED_TIMING)
        self.actor_head = nn.Linear(hidden_dim, vocab_size)
        
        # ACTOR HEAD 2 (Portfolio-level): sceglie una formula fissa dal portafoglio (FIXED_FORMULAS)
        self.portfolio_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, portfolio_size)
        )
        
        # CRITIC HEAD: stima il valore tattico dello stato corrente V(s)
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

    def forward_portfolio(self, game_state_tensor):
        """Usato in modalità FIXED_FORMULAS per selezionare un'intera formula predefinita."""
        return self.portfolio_head(game_state_tensor)

    def evaluate_state(self, game_state_tensor):
        """Metodo esclusivo del Critic per calcolare il valore dello stato"""
        return self.critic_head(game_state_tensor)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))


class NeuroSymbolicAI:
    def __init__(self, vitamin_client, user_predictor, vocab, masker, state_dim=14, ablation_mode="FULL"):
        self.vitamin = vitamin_client
        self.user_predictor = user_predictor
        self.vocab = vocab
        self.masker = masker
        self.ablation_mode = ablation_mode.upper()  # "FULL", "FIXED_FORMULAS", "FIXED_TIMING"
        
        self.policy_net = MetaControllerLSTM(
            state_dim=state_dim, 
            vocab_size=vocab.get_size(),
            portfolio_size=len(FIXED_PORTFOLIO)
        )
        
        self.current_plan = []
        self.expected_current_signature = None
        self.last_expected_player_action = None
        self.last_prob = 0.0
        self.horizon_tree = None
        
        # Variabili di stato per il PPO
        self.last_log_probs = []
        self.last_formula = None
        self.last_state_value = None
        self.last_portfolio_idx = None
        self.excluded_portfolio_indices = set()
        
        self.is_thinking = False
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="NeuroSym_Worker")


    def _get_state_tensor(self, game_state):
        classes = ["Warrior", "Mage", "Ranger"]
        p_onehot = [1.0 if game_state.player.char_class == c else 0.0 for c in classes]
        n_onehot = [1.0 if game_state.npc.char_class == c else 0.0 for c in classes]
        class_features = torch.tensor(p_onehot + n_onehot, dtype=torch.float32)

        if self.ablation_mode == "FIXED_TIMING":
            # In FIXED_TIMING il "quando" è scandito solo dal cronometro fisso del match (Early / Mid / Late),
            # mascherando HP, SP e Buff istantanei per testare la pura sintesi su calendario fisso.
            turn = getattr(game_state, "turn", 1)
            early = 1.0 if turn <= 4 else 0.0
            mid = 1.0 if 5 <= turn <= 9 else 0.0
            late = 1.0 if turn >= 10 else 0.0
            turn_norm = min(1.0, turn / 35.0)
            base_features = torch.tensor([early, mid, late, turn_norm, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            sig = get_state_signature(game_state, 0)
            scales = torch.tensor([7.0, 3.0, 1.0, 1.0, 7.0, 3.0, 1.0, 1.0], dtype=torch.float32)
            base_features = torch.tensor(list(sig[1:]), dtype=torch.float32) / scales
        
        return torch.cat((base_features, class_features), dim=0).unsqueeze(0)


    def _generate_formula(self, game_state):
        state_tensor = self._get_state_tensor(game_state)
        state_value = self.policy_net.evaluate_state(state_tensor)

        # --- MODALITÀ ABLAZIONE 1: PORTAFOGLIO DI FORMULE FISSE ---
        if self.ablation_mode == "FIXED_FORMULAS":
            logits = self.policy_net.forward_portfolio(state_tensor)
            
            # Se nei tentativi precedenti dello stesso turno alcune formule fisse sono state
            # rifiutate da ViTAMiN, le mascheriamo per non ripeterle identiche nei 3 tentativi
            if self.excluded_portfolio_indices and len(self.excluded_portfolio_indices) < len(FIXED_PORTFOLIO):
                mask = torch.ones_like(logits)
                for idx in self.excluded_portfolio_indices:
                    mask[0, idx] = 0.0
                logits = logits.masked_fill(mask == 0.0, -float('inf'))

            if not self.policy_net.training:
                probs = F.softmax(logits / 0.5, dim=-1)
                chosen_idx_tensor = torch.multinomial(probs, 1).squeeze(0)
            else:
                probs = F.softmax(logits, dim=-1)
                m = torch.distributions.Categorical(probs)
                chosen_idx_tensor = m.sample()

            chosen_idx = chosen_idx_tensor.item()
            self.last_portfolio_idx = chosen_idx
            
            dist_full = torch.distributions.Categorical(F.softmax(logits, dim=-1))
            saved_log_probs = [dist_full.log_prob(chosen_idx_tensor)]
            return FIXED_PORTFOLIO[chosen_idx], saved_log_probs, state_value

        # --- MODALITÀ FULL e FIXED_TIMING: SINTESI TOKEN-PER-TOKEN CON LSTM ---
        self.masker.reset()
        formula_tokens = []
        saved_log_probs = []
        self.last_portfolio_idx = None
        
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
        form = self.last_formula or ""

        # Allineato a tutte le nuove proposizioni offensive e tattiche di CTL_bridge.py
        has_off = any(p in form for p in ["Dead_P", "CanKill_N", "LowHP_P", "HP_Advantage_N"])
        has_spec = any(p in form for p in ["SpecialReady_N", "Fortified_N"])

        if has_off:
            # Piano offensivo: massimizza i danni e l'uso della Special
            score = (damage_dealt * 6.0) + (sp_gained * 2.0) - (damage_taken * 2.0)
            if current_state.npc.sp >= current_state.npc.stats["sp_threshold"] and next_state.npc.sp < current_state.npc.sp:
                score += 60.0
            if next_state.player.hp <= 0:
                score += 1000.0
        elif has_spec:
            # Piano di preparazione/potenziamento: massimizza SP e attivazione Buff
            score = (sp_gained * 8.0) + (damage_dealt * 2.0) - (damage_taken * 3.0)
            if (next_state.npc.def_buff_turns > current_state.npc.def_buff_turns or 
                next_state.npc.atk_buff_turns > current_state.npc.atk_buff_turns):
                score += 25.0
        else:
            # Piano conservativo/controllo (es. sola CounterReady_N): minimizza i danni subiti e guadagna SP
            score = (damage_dealt * 3.0) - (damage_taken * 6.0) + (sp_gained * 2.0)
            if next_state.npc.def_buff_turns > current_state.npc.def_buff_turns:
                score += 15.0

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
                        'safe_prob': 0.0,
                        'max_safe_p_prob': -1.0,
                        'best_safe_score': -float('inf'),
                        'best_p_act_for_logs': None,
                        'target_node': None
                    }

                score = self._evaluate_transition(current.state, child.state)
                action_evals[n_act]['expected_utility'] += prob * score

                if child in safe_nodes:
                    action_evals[n_act]['safe_prob'] += prob
                    if (prob > action_evals[n_act]['max_safe_p_prob'] or 
                       (abs(prob - action_evals[n_act]['max_safe_p_prob']) < 1e-6 and score > action_evals[n_act]['best_safe_score'])):
                        action_evals[n_act]['max_safe_p_prob'] = prob
                        action_evals[n_act]['best_safe_score'] = score
                        action_evals[n_act]['best_p_act_for_logs'] = p_act
                        action_evals[n_act]['target_node'] = child

            best_n_act = None
            best_combined_score = -float('inf')
            best_data = None

            for n_act, data in action_evals.items():
                if data['target_node'] is not None and data['safe_prob'] > 0.0:
                    combined = (data['safe_prob'] * 25.0) + data['expected_utility']
                    if combined > best_combined_score:
                        best_combined_score = combined
                        best_n_act = n_act
                        best_data = data

            if best_n_act is not None and best_data is not None:
                plan.append({
                    "npc_action": best_n_act,
                    "expected_player_action": best_data['best_p_act_for_logs'],
                    "probability": best_data['safe_prob'],
                    "expected_signature": get_state_signature(best_data['target_node'].state, best_data['target_node'].depth),
                    "expected_props": list(best_data['target_node'].atomic_props)
                })
                current = best_data['target_node']
            else:
                break
                
        return plan


    def _async_compute(self, game_state, last_p_act, last_n_act):
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
                self.excluded_portfolio_indices.clear()
                self.is_thinking = False
                return "SUCCESS"
            else:
                if self.ablation_mode == "FIXED_FORMULAS" and self.last_portfolio_idx is not None:
                    self.excluded_portfolio_indices.add(self.last_portfolio_idx)
                print(f"    [VITAMIN] ⚠️ Formula soddisfatta, ma l'Expectimax non ha estratto un piano valido dal DAG.")
                self.is_thinking = False
                return "REJECTED"
        else:
            if self.ablation_mode == "FIXED_FORMULAS" and self.last_portfolio_idx is not None:
                self.excluded_portfolio_indices.add(self.last_portfolio_idx)
            error_msg = response.get("error", "Proprietà falsa per tutti gli stati iniziali o sintassi non valida.")
            print(f"    [VITAMIN] ❌ Formula '{generated_formula}' RIFIUTATA. Motivo: {error_msg}")
            self.is_thinking = False
            return "REJECTED"


    def _async_compute_gui(self, game_state, last_p_act, last_n_act):
        """
        Wrapper per Pygame (main.py): prova fino a 3 formule e, se tutte vengono rifiutate,
        attiva un fallback Expectimax sul DAG per non bloccare mai la partita.
        """
        try:
            self.excluded_portfolio_indices.clear()
            for _ in range(3):
                status = self._async_compute(game_state, last_p_act, last_n_act)
                if status == "SUCCESS" and self.current_plan:
                    return
                    
            if self.horizon_tree and self.horizon_tree.transitions:
                nodes = []
                def collect(n):
                    if n not in nodes:
                        nodes.append(n)
                        for _, _, _, c in n.transitions:
                            collect(c)
                collect(self.horizon_tree)
                fake_resp = {"verification": {"satisfying_states": [f"s{i}" for i in range(len(nodes))]}}
                self.current_plan = self._extract_plan_from_trace(fake_resp, self.horizon_tree, nodes)
        finally:
            self.excluded_portfolio_indices.clear()
            self.is_thinking = False


    def evaluate_post_turn(self, game_state, last_player_action, last_npc_action):
        if self.is_thinking:
            return
        
        # In FIXED_TIMING l'agente non adatta la durata del piano allo stato, ma ricalcola a cadenza fissa
        if self.ablation_mode == "FIXED_TIMING":
            self.current_plan = []
            self.expected_current_signature = None
        else:
            current_signature = get_state_signature(game_state, current_depth=0)
            if self.expected_current_signature:
                if current_signature[1:] != self.expected_current_signature[1:]:
                    self.current_plan = []
                    self.expected_current_signature = None

        if not self.current_plan:
            self.is_thinking = True
            self.executor.submit(self._async_compute_gui, game_state, last_player_action, last_npc_action)


    def decide_action(self, game_state, last_player_action=None, last_npc_action=None):
        if self.is_thinking:
            return "THINKING"

        if self.current_plan:
            step = self.current_plan.pop(0)
            self.expected_current_signature = step["expected_signature"]
            self.last_expected_player_action = step.get("expected_player_action")
            self.last_prob = step.get("probability", 0.0)
            return step["npc_action"]

        self.is_thinking = True
        self.executor.submit(self._async_compute_gui, game_state, last_player_action, last_npc_action)
        return "THINKING" 


class DummyAI:
    """
    Fase 1: Avversario base per il Curriculum Learning.
    """
    def decide_action(self, game_state, npc_role="player"):
        action = Action.BUFF if game_state.is_buff_allowed(game_state.player) else Action.ATTACK
        return action if r.randint(1, 100) > 50 else Action.DEFEND