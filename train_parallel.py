import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import concurrent.futures

from pyFighters import Action, LogicFightersState
from CTL_bridge import UserPredictor
from vitamin_client import VitaminClient
from neuro_symbolic_env import CTLVocabulary, GrammarMasker
from ai_neuro_symbolic import NeuroSymbolicAI, DummyAI
from ai_behavior_tree import BehaviorTreeAI 

# ===========================================
# Variabili Globali per i Lavoratori (Worker)
# ===========================================
worker_ai = None
worker_bt = None
worker_dummy = None

def init_worker():
    """Inizializza l'ambiente locale del singolo core e silenzia le print."""
    global worker_ai, worker_bt, worker_dummy
    
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    
    atomic_props = ["Alive_N", "Dead_P", "LowHP_P", "SpecialReady_N", "HP_Advantage_N", "Horizon"]
    vocab = CTLVocabulary(atomic_props)
    masker = GrammarMasker(vocab)
    
    vitamin_client = VitaminClient(api_key="821b5bd238df54433364778326920866cca018890e4b4ccc2391853eefedc56e")
    user_predictor = UserPredictor()
    
    worker_ai = NeuroSymbolicAI(vitamin_client, user_predictor, vocab, masker)
    worker_bt = BehaviorTreeAI()
    worker_dummy = DummyAI()


# ==========================================
# Task Parallelo: Il Singolo Episodio
# ==========================================
def run_parallel_episode(global_state_dict, phase, max_turns=35, gamma=0.98, gae_lambda=0.95, vf_coef=0.5):
    global worker_ai, worker_bt, worker_dummy
    
    worker_ai.policy_net.load_state_dict(global_state_dict)
    worker_ai.policy_net.train()
    worker_ai.vitamin.unsaved_changes = 0 
    worker_ai.current_plan = []
    
    state = LogicFightersState("Warrior", "Warrior")
    state.player.char_class = "Warrior"
    state.npc.char_class = "Warrior"
    
    current_opponent = worker_dummy if phase == 1 else worker_bt
    
    step_log_prob_sums = []
    step_values = []
    step_rewards = []
    
    turns_played = 0
    generated_formulas = []
    outcome = 0
    
    max_hp_p = float(state.player.stats["max_hp"])
    max_hp_n = float(state.npc.stats["max_hp"])
    
    # --- CICLO DI GIOCO ---
    while not state.is_finished() and turns_played < max_turns:
        turns_played += 1
        enemy_action = current_opponent.decide_action(state, npc_role="player")
        
        phi_old = (state.npc.hp / max_hp_n) - (state.player.hp / max_hp_p)
        my_action = None
        
        # 1. CONTROLLO DEL PIANO ESISTENTE
        if worker_ai.current_plan:
            next_step = worker_ai.current_plan[0]
            expected_enemy = next_step.get("player_action", enemy_action)
            if expected_enemy == enemy_action:
                my_action = worker_ai.current_plan.pop(0)["npc_action"]
            else:
                worker_ai.current_plan = []

        # 2. GENERAZIONE NUOVA FORMULA (Solo se il piano è vuoto)
        if my_action is None:
            plan_found = False
            for attempt in range(3): 
                is_valid_formula = worker_ai._async_compute(state, None, enemy_action)
                form = worker_ai.last_formula if hasattr(worker_ai, 'last_formula') else ""
                
                if form:
                    generated_formulas.append((form, is_valid_formula))
                
                reward = 0.0
                if is_valid_formula == "SUCCESS":
                    hp_ratio_n = state.npc.hp / max_hp_n
                    hp_ratio_p = state.player.hp / max_hp_p
                    
                    has_offensive = any(p in form for p in ["Dead_P", "LowHP_P", "HP_Advantage_N"])
                    has_special = "SpecialReady_N" in form
                    has_alive = "Alive_N" in form
                    
                    # CASO 1: Formula puramente difensiva/passiva (contiene Alive_N ma NESSUN obiettivo offensivo)
                    if has_alive and not has_offensive and not has_special:
                        if hp_ratio_n > 0.50:
                            # STAI BENE! Vietato usare Alive_N per pigrizia quando hai più di metà vita.
                            reward -= 0.30
                        else:
                            # Sei davvero ferito (<50% HP): ora ha senso cercare un piano di sopravvivenza!
                            reward += 0.25
                            
                    # CASO 2: Formula sul caricamento Speciale
                    elif has_special and not has_offensive:
                        if state.npc.sp >= 3:
                            reward -= 0.30  # Hai già la speciale pronta, smettila di chiederla!
                        else:
                            reward += 0.20  # Buon piano preparatorio
                            
                    # CASO 3: Formule Offensive (HP_Advantage_N, LowHP_P, Dead_P)
                    elif has_offensive:
                        reward += 0.30  # Premio netto per l'iniziativa offensiva
                        if "and" in form:
                            reward += 0.20  # Bonus extra se combina due concetti (es. HP_Advantage_N and Alive_N)
                        if ("Dead_P" in form or "LowHP_P" in form) and hp_ratio_p <= 0.50:
                            reward += 0.30  # Istinto del killer a nemico ferito

                elif is_valid_formula == "REJECTED":
                    reward -= 0.15
                elif is_valid_formula == "TRUNCATED":
                    reward -= 0.60
                    
                if hasattr(worker_ai, 'last_log_probs') and worker_ai.last_log_probs:
                    tok_log_probs = torch.stack(worker_ai.last_log_probs).squeeze()
                    if tok_log_probs.dim() == 0:
                        tok_log_probs = tok_log_probs.unsqueeze(0)
                    
                    step_log_prob_sums.append(tok_log_probs.mean())
                    
                    val = worker_ai.last_state_value if worker_ai.last_state_value is not None else torch.tensor(0.0)
                    step_values.append(val.squeeze())
                    step_rewards.append(reward)
                
                if worker_ai.current_plan:
                    my_action = worker_ai.current_plan.pop(0)["npc_action"]
                    plan_found = True
                    break
                    
            if not plan_found:
                if len(step_rewards) > 0: 
                    step_rewards[-1] -= 1.20
                outcome = 0
                break
            
        # --- 3. APPLICAZIONE AZIONE FISICA ---
        result = state.apply_action_resolution(enemy_action, my_action)
        
        phi_new = (state.npc.hp / max_hp_n) - (state.player.hp / max_hp_p)
        shaping_reward = (phi_new - phi_old) * 3.0
        
        dmg_dealt = result.get('player_damage', 0) / max_hp_p
        time_penalty = -0.04
        
        phys_reward = time_penalty + shaping_reward + (dmg_dealt * 2.5)
        
        if len(step_rewards) > 0: 
            step_rewards[-1] += phys_reward

        # Condizioni di fine partita
        if state.is_finished():
            if state.npc.hp > 0 and state.player.hp <= 0:
                speed_bonus = max(0.0, (max_turns - turns_played) / max_turns) * 1.5
                step_rewards[-1] += (4.0 + speed_bonus)
                outcome = 1
            else:
                step_rewards[-1] -= 1.5
                outcome = 0
            break
        elif turns_played >= max_turns:
            step_rewards[-1] -= 2.5 
            outcome = 0
            break

    if not step_log_prob_sums:
        return (None, 0, turns_played, 0.0, worker_ai.vitamin.cache, generated_formulas)

    # --- CALCOLO GAE (Generalized Advantage Estimation) ---
    values_tensor = torch.stack(step_values)
    if values_tensor.dim() == 0:
        values_tensor = values_tensor.unsqueeze(0)
        
    log_probs_tensor = torch.stack(step_log_prob_sums)
    
    T = len(step_rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    returns = torch.zeros(T, dtype=torch.float32)
    
    gae = 0.0
    next_val = 0.0
    
    for t in reversed(range(T)):
        delta = step_rewards[t] + gamma * next_val - values_tensor[t].detach().item()
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values_tensor[t].detach().item()
        next_val = values_tensor[t].detach().item()
    
    if T > 1:
        adv_scale = advantages.std() + 1e-5
        scaled_advantages = torch.clamp(advantages / adv_scale, -2.5, 2.5)
    else:
        scaled_advantages = torch.clamp(advantages, -2.5, 2.5)
    
    # Piccolo termine di regolarizzazione sull'entropia per non far collassare la rete su 1 sola formula
    entropy_reg = 0.01 * (log_probs_tensor.mean()) # log_probs è negativo, minimizzare +0.01*log_p evita che p -> 1.0 assoluto
    
    actor_loss = -(log_probs_tensor * scaled_advantages.detach()).mean() + entropy_reg
    critic_loss = vf_coef * F.smooth_l1_loss(values_tensor, returns.detach())
    
    total_loss = actor_loss + critic_loss
    
    worker_ai.policy_net.zero_grad()
    total_loss.backward()
    
    nn.utils.clip_grad_norm_(worker_ai.policy_net.parameters(), max_norm=0.5)
    
    grads = [p.grad.clone().cpu() if p.grad is not None else torch.zeros_like(p).cpu() for p in worker_ai.policy_net.parameters()]
    
    return (grads, outcome, turns_played, total_loss.item(), worker_ai.vitamin.cache, generated_formulas)


# ==========================================
# PROCESSO PRINCIPALE (Il Direttore Asincrono)
# ==========================================
if __name__ == '__main__':
    atomic_props = ["Alive_N", "Dead_P", "LowHP_P", "SpecialReady_N", "HP_Advantage_N", "Horizon"]
    global_vocab = CTLVocabulary(atomic_props)
    global_masker = GrammarMasker(global_vocab)
    global_vitamin = VitaminClient()
    
    global_ai = NeuroSymbolicAI(global_vitamin, UserPredictor(), global_vocab, global_masker)
    
    if os.path.exists("neuro_symbolic_warrior.pth"):
        try:
            global_ai.policy_net.load_state_dict(torch.load("neuro_symbolic_warrior.pth"))
            print("🧠 [RESUME] Pesi caricati da 'neuro_symbolic_warrior.pth'! Continuiamo l'addestramento.")
        except Exception as e:
            print(f"⚠️ Impossibile caricare il vecchio modello ({e}), si parte da zero.")
            
    num_episodes = 800
    
    max_workers = os.cpu_count()-4 or 8
    batch_size = max(4, max_workers // 2)
    total_batches = max(1, num_episodes // batch_size)
    
    # Learning Rate più morbido + Cosine Annealing per evitare l'oscillazione 80% -> 35% -> 80%
    optimizer = optim.Adam(global_ai.policy_net.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=0.00005)
    
    phase = 1
    recent_outcomes = []
    best_win_rate = 0.0
    total_episodes_played = 0
    
    batch_grads = []
    batch_loss = 0.0
    batch_wins = 0
    batch_turns = []
    all_formulas = []
    
    print(f"🚀 Avvio Addestramento Neuro-Simbolico ({max_workers} Workers | Batch Size: {batch_size})")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        
        def submit_new_task():
            global_state_dict = {k: v.cpu() for k, v in global_ai.policy_net.state_dict().items()}
            return executor.submit(run_parallel_episode, global_state_dict, phase)

        active_futures = set([submit_new_task() for _ in range(max_workers)])
        
        while total_episodes_played < num_episodes:
            done, active_futures = concurrent.futures.wait(active_futures, return_when=concurrent.futures.FIRST_COMPLETED)
            
            for future in done:
                if total_episodes_played + len(active_futures) < num_episodes:
                    active_futures.add(submit_new_task())

                grads, outcome, turns, loss_val, worker_cache, formulas = future.result()
                total_episodes_played += 1
                
                global_vitamin.cache.update(worker_cache)
                global_vitamin.unsaved_changes += 1
                
                recent_outcomes.append(outcome)
                if len(recent_outcomes) > 50:
                    recent_outcomes.pop(0)
                    
                if grads is not None:
                    batch_grads.append(grads)
                    batch_loss += loss_val
                
                batch_wins += outcome
                batch_turns.append(turns)
                all_formulas.extend(formulas)
                
                if len(batch_grads) >= batch_size:
                    optimizer.zero_grad()
                    for i, param in enumerate(global_ai.policy_net.parameters()):
                        mean_grad = torch.stack([g[i] for g in batch_grads]).mean(dim=0)
                        param.grad = mean_grad
                    
                    nn.utils.clip_grad_norm_(global_ai.policy_net.parameters(), max_norm=0.5)
                    optimizer.step()
                    scheduler.step()

                    from collections import Counter
                    
                    avg_loss = batch_loss / len(batch_grads)
                    avg_turns = sum(batch_turns) / len(batch_turns)
                    rolling_wr = (sum(recent_outcomes) / len(recent_outcomes)) * 100
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Contiamo la distribuzione REALE delle formule invece di stamparne una sola!
                    succ_counts = Counter([f for f, s in all_formulas if s == "SUCCESS"])
                    rej_counts = Counter([f for f, s in all_formulas if s == "REJECTED"])
                    
                    top_succ_str = ", ".join([f"'{k}' ({v})" for k, v in succ_counts.most_common(3)])
                    top_rej_str = ", ".join([f"'{k}' ({v})" for k, v in rej_counts.most_common(2)])
                    
                    print(f"📦 BATCH | Ep: {total_episodes_played}/{num_episodes} | Fase {phase} | Vittorie: {batch_wins}/{len(batch_turns)} | WR(50): {rolling_wr:.1f}% | Cache: {len(global_vitamin.cache)}")
                    
                    if top_rej_str: print(f"   ❌ Top Rifiutate: {top_rej_str}")
                    if top_succ_str: print(f"   ✅ Top Validate : {top_succ_str}")
                    print("-" * 75)
                    
                    torch.save(global_ai.policy_net.state_dict(), "neuro_symbolic_warrior.pth")
                    
                    # Salviamo il Best Model (dando priorità assoluta alla Fase 2!)
                    if len(recent_outcomes) >= 30 and rolling_wr >= best_win_rate:
                        best_win_rate = rolling_wr
                        torch.save(global_ai.policy_net.state_dict(), "neuro_symbolic_warrior_best.pth")
                    
                    batch_grads = []
                    batch_loss = 0.0
                    batch_wins = 0
                    batch_turns = []
                    all_formulas = []
                
                if global_vitamin.unsaved_changes >= 50:
                    global_vitamin.save_cache()
                    
                if phase == 1 and len(recent_outcomes) >= 36:
                    win_rate = sum(recent_outcomes) / len(recent_outcomes)
                    if win_rate >= 0.65:
                        phase = 2
                        recent_outcomes = []
                        best_win_rate = 0.0 
                        print("\n" + "🔥"*25)
                        print(f"🎓 CURRICULUM AVANZATO! Win-rate {win_rate*100:.0f}% raggiunto.")
                        print("🔥 Fase 2: Sparring contro BehaviorTreeAI attivo in tutti i core!")
                        print("🔥"*25 + "\n")

    global_vitamin.save_cache()
    
    if os.path.exists("neuro_symbolic_warrior_best.pth"):
        best_weights = torch.load("neuro_symbolic_warrior_best.pth")
        torch.save(best_weights, "neuro_symbolic_warrior.pth")
        torch.save(best_weights, "neuro_symbolic_warrior_parallel.pth")
    else:
        torch.save(global_ai.policy_net.state_dict(), "neuro_symbolic_warrior.pth")
        torch.save(global_ai.policy_net.state_dict(), "neuro_symbolic_warrior_parallel.pth")
    
    final_win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0.0
    print(f"\n🎉 ADDESTRAMENTO COMPLETATO! Win-rate finale Fase {phase}: {final_win_rate:.1f}% (Best Fase {phase} salvato: {best_win_rate:.1f}%)")
    