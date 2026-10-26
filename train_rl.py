import torch
import torch.optim as optim
import torch.nn.functional as F

from pyFighters import Action, LogicFightersState
from CTL_bridge import UserPredictor
from vitamin_client import VitaminClient
from neuro_symbolic_env import CTLVocabulary, GrammarMasker
from ai_neuro_symbolic import NeuroSymbolicAI, DummyAI
from ai_behavior_tree import BehaviorTreeAI 

atomic_props = ["Alive_N", "Dead_P", "LowHP_P", "SpecialReady_N", "HP_Advantage_N", "Horizon"]
vocab = CTLVocabulary(atomic_props)
masker = GrammarMasker(vocab)

vitamin_client = VitaminClient(api_key="821b5bd238df54433364778326920866cca018890e4b4ccc2391853eefedc56e")
user_predictor = UserPredictor()

bt_ai = BehaviorTreeAI()
dummy_ai = DummyAI()

# Iperparametri PPO e RL
gamma = 0.99
eps_clip = 0.2
vf_coef = 0.5
entropy_coef = 0.01
lr = 0.0005 # Learning rate ridotto per stabilizzare il PPO

num_episodes = 500
MAX_TURNS = 35

classes_to_train = ["Warrior"]

for char_class in classes_to_train:
    print(f"\n==================================================")
    print(f"🚀 AVVIO PPO CURRICULUM TRAINING: {char_class.upper()}")
    print(f"==================================================")
    
    ai = NeuroSymbolicAI(vitamin_client, user_predictor, vocab, masker)
    optimizer = optim.Adam(ai.policy_net.parameters(), lr=lr)
    
    class_stats = {
        "wins": 0, "losses": 0, "stalls": 0, "total_turns": 0,
        "first_loss": None, "last_loss": None
    }
    
    # Variabili Curriculum Learning
    recent_outcomes = [] 
    phase = 1
    current_opponent = dummy_ai
    print("🎓 Fase 1: Sparring contro DummyAI (Solo Attacco)")
    
    for episode in range(num_episodes):
        state = LogicFightersState("Warrior", "Warrior")
        state.player.char_class = "Warrior"
        state.npc.char_class = "Warrior"
        
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        turns_played = 0
        
        user_predictor.action_history = []
        user_predictor.class_transitions = {}
        
        while not state.is_finished() and turns_played < MAX_TURNS:
            turns_played += 1
            enemy_action = current_opponent.decide_action(state, npc_role="player")
            
            plan_found = False
            max_retries = 5
            
            # --- CICLO DI RAGIONAMENTO (BUDGET LIMITATO) ---
            for attempt in range(max_retries):
                is_valid_formula = ai._async_compute(state, None, enemy_action)
                
                # Calcolo penalità intermedia per l'errore logico
                reward = 0.0
                # DISTINZIONE CHIRURGICA DEGLI ERRORI
                if is_valid_formula == "SUCCESS":
                    reward += 2.0 
                    
                    if hasattr(ai, 'last_formula'):
                        formula = ai.last_formula
                        
                        # 1. Premio se sceglie l'Esplorazione Ottimistica (E) invece del Pessimismo (A)
                        if formula.startswith("E F") or formula.startswith("E G"):
                            reward += 2.0
                            
                        # 2. Premiamo la preparazione realistica a inizio partita
                        if "SpecialReady_N" in formula:
                            reward += 4.0
                        elif "HP_Advantage_N" in formula:
                            reward += 3.0
                        elif "Dead_P" in formula:
                            # Diamo un bonus enorme SOLO se il nemico ha HP bassi
                            # Altrimenti la rete proverà a usarlo a HP pieni e verrà bocciata
                            if state.player.hp < (state.player.stats["max_hp"] * 0.3):
                                reward += 10.0
                                
                elif is_valid_formula == "REJECTED":
                    # Punizione leggerissima: la formula aveva senso, ma non in questo turno
                    reward -= 1.0 
                elif is_valid_formula == "TRUNCATED":
                    reward -= 10.0 # Punizione brutale: la sintassi deve essere pulita
                
                # Registriamo la penalità per questo specifico tentativo
                if hasattr(ai, 'last_log_probs') and ai.last_log_probs:
                    num_tokens = len(ai.last_log_probs)
                    episode_log_probs.extend(ai.last_log_probs)
                    episode_rewards.extend([reward] * num_tokens)
                    val = ai.last_state_value if ai.last_state_value is not None else torch.tensor([[0.0]])
                    episode_values.extend([val.squeeze()] * num_tokens)
                
                if ai.current_plan:
                    my_action = ai.current_plan[0]["npc_action"]
                    plan_found = True
                    break # Esce dal ciclo retries, la mossa è decisa!

            # --- GESTIONE DELLA SCONFITTA TECNICA ---
            if not plan_found:
                # L'IA non ha saputo elaborare una strategia valida. Fine immediata.
                terminal_penalty = -100.0
                # Aggiungiamo il malus all'ultimo tentativo fallito
                if len(episode_rewards) > 0:
                    episode_rewards[-1] += terminal_penalty 
                
                class_stats["losses"] += 1
                recent_outcomes.append(0)
                break # Interrompe il match istantaneamente

            # --- APPLICAZIONE AZIONE FISICA VALIDATA ---
            result = state.apply_action_resolution(enemy_action, my_action) 
            
            # Calcolo reward fisico (danni)
            phys_reward = -1.0 
            phys_reward += (result.get('player_damage', 0) * 2.0) 
            phys_reward -= (result.get('npc_damage', 0) * 1.5)    
            if state.npc.sp > 0: phys_reward += 0.5
            
            # Sommiamo il reward fisico all'ultimo token generato (che ha chiuso il piano)
            if len(episode_rewards) > 0:
                episode_rewards[-1] += phys_reward

            # Controllo fine partita naturale
            if state.is_finished():
                if state.npc.hp > 0 and state.player.hp <= 0:
                    episode_rewards[-1] += 300.0 
                    class_stats["wins"] += 1
                    recent_outcomes.append(1)
                else:
                    episode_rewards[-1] -= 50.0
                    class_stats["losses"] += 1
                    recent_outcomes.append(0)
            elif turns_played >= MAX_TURNS:
                episode_rewards[-1] -= 80.0 
                class_stats["stalls"] += 1
                recent_outcomes.append(0)
            # Allineamento Tensori PPO
            if hasattr(ai, 'last_log_probs') and ai.last_log_probs:
                num_tokens = len(ai.last_log_probs)
                episode_log_probs.extend(ai.last_log_probs)
                episode_rewards.extend([reward] * num_tokens)
                
                # Salviamo il valore stimato dal Critic (duplicato per ogni token generato nel turno)
                val = ai.last_state_value if ai.last_state_value is not None else torch.tensor([[0.0]])
                episode_values.extend([val.squeeze()] * num_tokens)
                
        class_stats["total_turns"] += turns_played
        
        # --- GESTIONE CURRICULUM LEARNING ---
        if len(recent_outcomes) > 50:
            recent_outcomes.pop(0)
            
        if phase == 1 and len(recent_outcomes) == 50:
            win_rate_recent = sum(recent_outcomes) / 50
            if win_rate_recent >= 0.65:
                phase = 2
                current_opponent = bt_ai
                print("\n" + "🔥"*20)
                print(f"🎓 CURRICULUM AVANZATO! Win-rate {win_rate_recent*100:.0f}% raggiunto.")
                print("🔥 Fase 2: Sparring contro BehaviorTreeAI (Nemico Completo)")
                print("🔥"*20 + "\n")
        
        # --- BACKPROPAGATION PPO ---
        if episode_log_probs and episode_rewards:
            discounted_rewards = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
                
            returns = torch.tensor(discounted_rewards, dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
                
            old_log_probs = torch.stack(episode_log_probs).detach()
            values = torch.stack(episode_values).squeeze()
            
            # Calcolo Advantage (R - V)
            advantages = returns - values.detach()
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
            
            # Simuliamo il ratio PPO (in un setup full-PPO ci sarebbero epochs multiple)
            # Qui applichiamo il clipping direttamente sui log_probs dell'episodio per limitare il passo
            ratios = torch.exp(old_log_probs - old_log_probs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = vf_coef * F.mse_loss(values, returns)
            
            # Approssimazione entropia per incentivare l'esplorazione
            entropy_bonus = entropy_coef * -(old_log_probs.mean())
            
            total_loss = actor_loss + critic_loss - entropy_bonus
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            loss_val = total_loss.item()
            if class_stats["first_loss"] is None:
                class_stats["first_loss"] = loss_val
            class_stats["last_loss"] = loss_val
            
            esito = "Vittoria 🏆" if state.npc.hp > 0 and state.player.hp <= 0 else "Sconfitta/Stallo 💀"
            print(f"✅ Ep {episode+1} | Fase {phase} | Loss: {loss_val:.4f} | {esito} | Turni: {turns_played}")
            if episode % 5 == 0:
                ai.vitamin.print_cache_stats()

    # Salvataggio Pesi e Report finale
    model_filename = f"neuro_symbolic_{char_class.lower()}.pth"
    torch.save(ai.policy_net.state_dict(), model_filename)

    ai.vitamin.save_cache()

    wins = class_stats["wins"]
    win_rate = (wins / max(1, num_episodes)) * 100
    print(f"\n💾 Addestramento {char_class} completato! (Vittorie totali: {win_rate:.1f}%) Pesi in {model_filename}")