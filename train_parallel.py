import os
import sys
import io
import time
import shutil
import random
import torch  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import concurrent.futures
from collections import Counter

from pyFighters import Action, LogicFightersState
from CTL_bridge import UserPredictor
from vitamin_client import VitaminClient
from neuro_symbolic_env import CTLVocabulary, GrammarMasker
from ai_neuro_symbolic import NeuroSymbolicAI, DummyAI, FIXED_PORTFOLIO
from ai_behavior_tree import BehaviorTreeAI 

# ==============================================================================
# CONFIGURAZIONE ESPERIMENTO E ABLATION STUDY
# Valori ammessi: "FULL" | "FIXED_FORMULAS" | "FIXED_TIMING"
# ==============================================================================
ABLATION_MODE = "FIXED_TIMING"

ATOMIC_PROPS = [
    "Dead_P", "CanKill_N", "LowHP_P", "HP_Advantage_N", 
    "SpecialReady_N", "Fortified_N", "Threat_P", "CounterReady_N", "Horizon"
]

# File di salvataggio dedicati per la modalità corrente (non sovrascrivono le altre run)
MODE_TAG = ABLATION_MODE.lower()
CKPT_LATEST = f"neuro_symbolic_{MODE_TAG}.pth"
CKPT_BEST   = f"neuro_symbolic_{MODE_TAG}_best.pth"

worker_ai = None
worker_bt = None
worker_dummy = None


def safe_torch_save(state_dict, filename, retries=8, delay=0.25):
    """
    Serializza i pesi prima in RAM tramite io.BytesIO (evitando il crash C++ Error 1224
    di PyTorchFileWriter) e poi sostituisce atomicamente il file su disco.
    """
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    data = buffer.getvalue()
    
    temp_file = f"{filename}.tmp.{os.getpid()}.{random.randint(1000, 9999)}"
    for attempt in range(retries):
        try:
            with open(temp_file, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, filename)
            return True
        except OSError as e:
            if attempt == retries - 1:
                print(f"⚠️ [WARNING] Impossibile salvare '{filename}' dopo {retries} tentativi ({e}).")
            else:
                time.sleep(delay * (attempt + 1))
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
    return False


def safe_torch_load(filename, map_location="cpu"):
    """Legge il file in RAM prima di passarlo a torch.load per non lasciare handle aperti su Windows."""
    with open(filename, "rb") as f:
        data = f.read()
    return torch.load(io.BytesIO(data), map_location=map_location)


def formula_str_to_token_ids(formula_str, vocab):
    """Converte la formula generata negli indici numerici per il ricalcolo PPO."""
    if not formula_str or formula_str == "INVALID_TRUNCATED":
        return []
    if ABLATION_MODE == "FIXED_FORMULAS":
        return [FIXED_PORTFOLIO.index(formula_str)] if formula_str in FIXED_PORTFOLIO else []
    tokens = formula_str.strip().split()
    ids = [vocab.token2id[t] for t in tokens if t in vocab.token2id]
    ids.append(vocab.token2id[vocab.eos_token])
    return ids


def evaluate_step_for_ppo(policy_net, masker, state_tensor, token_ids):
    """Ricalcola log_prob, valore V(s) ed entropia sotto i pesi correnti durante le epoche PPO."""
    if ABLATION_MODE == "FIXED_FORMULAS":
        logits = policy_net.forward_portfolio(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        target_tensor = torch.tensor([token_ids[0]], dtype=torch.long)
        return (
            dist.log_prob(target_tensor).squeeze(),
            policy_net.evaluate_state(state_tensor).squeeze(),
            dist.entropy().squeeze()
        )

    masker.reset()
    hidden = policy_net.init_hidden()
    current_token_id = 0
    
    log_probs = []
    entropies = []
    
    for target_tid in token_ids:
        logits, hidden = policy_net(state_tensor, current_token_id, hidden)
        valid_mask = torch.tensor(masker.get_valid_mask(), dtype=torch.float32)
        masked_logits = logits.masked_fill(valid_mask == 0, -float('inf'))
        
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        target_tensor = torch.tensor([target_tid], dtype=torch.long)
        log_probs.append(dist.log_prob(target_tensor).squeeze())
        entropies.append(dist.entropy().squeeze())
        
        tok_str = masker.update_state(target_tid)
        if tok_str == '<EOS>':
            break
        current_token_id = target_tid
        
    new_log_prob_mean = torch.stack(log_probs).mean()
    entropy_mean = torch.stack(entropies).mean()
    new_value = policy_net.evaluate_state(state_tensor).squeeze()
    
    return new_log_prob_mean, new_value, entropy_mean


def init_worker():
    """Inizializza l'ambiente locale del singolo core."""
    global worker_ai, worker_bt, worker_dummy
    
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    
    vocab = CTLVocabulary(ATOMIC_PROPS)
    masker = GrammarMasker(vocab)
    
    vitamin_client = VitaminClient(api_key="821b5bd238df54433364778326920866cca018890e4b4ccc2391853eefedc56e")
    user_predictor = UserPredictor()
    
    worker_ai = NeuroSymbolicAI(
        vitamin_client, user_predictor, vocab, masker, 
        ablation_mode=ABLATION_MODE
    )
    worker_bt = BehaviorTreeAI()
    worker_dummy = DummyAI()


def run_parallel_episode(global_state_dict, phase, max_turns=35, gamma=0.98, gae_lambda=0.95):
    global worker_ai, worker_bt, worker_dummy
    
    worker_ai.policy_net.load_state_dict(global_state_dict)
    worker_ai.policy_net.train()
    worker_ai.vitamin.unsaved_changes = 0 
    worker_ai.current_plan = []
    worker_ai.user_predictor = UserPredictor()

    classes = ["Warrior", "Mage", "Ranger"]
    p_class = random.choice(classes)
    n_class = random.choice(classes)

    state = LogicFightersState(p_class, n_class)
    state.player.char_class = p_class
    state.npc.char_class = n_class
    
    current_opponent = worker_dummy if phase == 1 else worker_bt
    
    step_state_tensors = []
    step_token_seqs = []
    step_old_log_probs = []
    step_values = []
    step_rewards = []
    
    turns_played = 0
    generated_formulas = []
    outcome = 0
    current_plan_credit = 1.0
    
    last_p_act = None
    last_n_act = None
    
    max_hp_p = float(state.player.stats["max_hp"])
    max_hp_n = float(state.npc.stats["max_hp"])
    
    while not state.is_finished() and turns_played < max_turns:
        turns_played += 1
        enemy_action = current_opponent.decide_action(state, npc_role="player")
        
        phi_old = (state.npc.hp / max_hp_n) - (state.player.hp / max_hp_p)
        my_action = None
        
        # In FIXED_TIMING il ricalcolo avviene a cadenza fissa ad ogni turno
        if ABLATION_MODE == "FIXED_TIMING":
            worker_ai.current_plan = []
            
        # 1. CONTROLLO DEL PIANO ESISTENTE
        if worker_ai.current_plan:
            next_step = worker_ai.current_plan[0]
            expected_enemy = next_step.get("expected_player_action", enemy_action)
            if expected_enemy == enemy_action:
                my_action = worker_ai.current_plan.pop(0)["npc_action"]
            else:
                worker_ai.current_plan = []

        # 2. GENERAZIONE NUOVA FORMULA (Solo se il piano è vuoto)
        if my_action is None:
            plan_found = False
            worker_ai.excluded_portfolio_indices.clear()
            
            for attempt in range(3): 
                current_s_tensor = worker_ai._get_state_tensor(state).detach().cpu()
                is_valid_formula = worker_ai._async_compute(state, last_p_act, last_n_act)
                form = worker_ai.last_formula if hasattr(worker_ai, 'last_formula') else ""
                
                if form:
                    generated_formulas.append((form, is_valid_formula))
                
                reward = 0.0
                if is_valid_formula == "SUCCESS":
                    hp_ratio_p = state.player.hp / max_hp_p
                    
                    has_offensive = any(p in form for p in ["Dead_P", "CanKill_N", "LowHP_P", "HP_Advantage_N"])
                    has_special = "SpecialReady_N" in form
                    has_tactical = any(p in form for p in ["Threat_P", "CounterReady_N", "Fortified_N"])
                    
                    if has_offensive:
                        reward += 0.40
                        current_plan_credit = 1.0
                        if "and" in form:
                            reward += 0.30
                        if ("Dead_P" in form or "CanKill_N" in form or "LowHP_P" in form) and hp_ratio_p <= 0.50:
                            reward += 0.25
                    elif has_special:
                        if state.npc.sp >= state.npc.stats["sp_threshold"]:
                            reward -= 0.15
                            current_plan_credit = 0.5
                        else:
                            reward += 0.35
                            current_plan_credit = 1.0
                    elif has_tactical:
                        reward += 0.25
                        current_plan_credit = 0.8
                    else:
                        reward += 0.20
                        current_plan_credit = 1.0
                            
                elif is_valid_formula == "REJECTED":
                    reward -= 0.25
                elif is_valid_formula == "TRUNCATED":
                    reward -= 0.60
                    
                token_ids = formula_str_to_token_ids(form, worker_ai.vocab)
                if hasattr(worker_ai, 'last_log_probs') and worker_ai.last_log_probs and token_ids:
                    tok_log_probs = torch.stack(worker_ai.last_log_probs).squeeze()
                    if tok_log_probs.dim() == 0:
                        tok_log_probs = tok_log_probs.unsqueeze(0)
                    
                    step_state_tensors.append(current_s_tensor)
                    step_token_seqs.append(token_ids)
                    step_old_log_probs.append(tok_log_probs.mean().detach().cpu())
                    
                    val = worker_ai.last_state_value if worker_ai.last_state_value is not None else torch.tensor(0.0)
                    step_values.append(val.squeeze().detach().cpu())
                    step_rewards.append(reward)
                
                if worker_ai.current_plan:
                    my_action = worker_ai.current_plan.pop(0)["npc_action"]
                    plan_found = True
                    break
                    
            if not plan_found:
                if len(step_rewards) > 0: 
                    step_rewards[-1] -= 2.00
                outcome = 0
                break
            
        # --- 3. AGGIORNAMENTO PREDITTORE E RISOLUZIONE FISICA ---
        worker_ai.user_predictor.record_action(state.player, enemy_action)
        result = state.apply_action_resolution(enemy_action, my_action)
        
        last_p_act = enemy_action
        last_n_act = my_action
        
        phi_new = (state.npc.hp / max_hp_n) - (state.player.hp / max_hp_p)
        shaping_reward = (phi_new - phi_old) * 3.0
        
        dmg_dealt_by_npc = result.get('npc_damage', 0) / max_hp_p
        dmg_taken_by_npc = result.get('player_damage', 0) / max_hp_n
        time_penalty = -0.04
        
        phys_reward = time_penalty + shaping_reward + (dmg_dealt_by_npc * 2.5) - (dmg_taken_by_npc * 1.0)
        
        if len(step_rewards) > 0:
            if phys_reward > 0:
                step_rewards[-1] += (phys_reward * current_plan_credit)
            else:
                step_rewards[-1] += phys_reward

        if state.is_finished():
            if state.npc.hp > 0 and state.player.hp <= 0:
                speed_bonus = max(0.0, (max_turns - turns_played) / max_turns) * 1.5
                win_bonus = 4.0 + speed_bonus
                if len(step_rewards) > 0:
                    step_rewards[-1] += (win_bonus * current_plan_credit)
                outcome = 1
            else:
                if len(step_rewards) > 0:
                    step_rewards[-1] -= 1.5
                outcome = 0
            break
        elif turns_played >= max_turns:
            if len(step_rewards) > 0:
                step_rewards[-1] -= 2.5 
            outcome = 0
            break

    if not step_old_log_probs:
        return (None, 0, turns_played, worker_ai.vitamin.cache, generated_formulas)

    values_tensor = torch.stack(step_values)
    if values_tensor.dim() == 0:
        values_tensor = values_tensor.unsqueeze(0)
        
    old_log_probs_tensor = torch.stack(step_old_log_probs)
    
    T = len(step_rewards)
    advantages = torch.zeros(T, dtype=torch.float32)
    returns = torch.zeros(T, dtype=torch.float32)
    
    gae = 0.0
    next_val = 0.0
    
    for t in reversed(range(T)):
        delta = step_rewards[t] + gamma * next_val - values_tensor[t].item()
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values_tensor[t].item()
        next_val = values_tensor[t].item()
    
    trajectory_data = {
        "states": step_state_tensors,
        "token_seqs": step_token_seqs,
        "old_log_probs": old_log_probs_tensor,
        "advantages": advantages,
        "returns": returns
    }
    
    return (trajectory_data, outcome, turns_played, worker_ai.vitamin.cache, generated_formulas)


if __name__ == '__main__':
    # Backup automatico del modello FULL al 92% prima di fare qualunque altra cosa!
    if os.path.exists("neuro_symbolic_warrior_best.pth") and not os.path.exists("neuro_symbolic_full_best.pth"):
        try:
            shutil.copy2("neuro_symbolic_warrior_best.pth", "neuro_symbolic_full_best.pth")
            print("🛡️ [BACKUP] Salvata copia protetta del modello 92% in 'neuro_symbolic_full_best.pth'!")
        except OSError:
            pass

    RESET_CURRENT_MODE = False
    if RESET_CURRENT_MODE:
        for f_name in [CKPT_LATEST, CKPT_BEST]:
            if os.path.exists(f_name):
                try:
                    os.remove(f_name)
                    print(f"🧹 Rimosso '{f_name}'.")
                except OSError:
                    pass

    global_vocab = CTLVocabulary(ATOMIC_PROPS)
    global_masker = GrammarMasker(global_vocab)
    global_vitamin = VitaminClient()
    
    global_ai = NeuroSymbolicAI(
        global_vitamin, UserPredictor(), global_vocab, global_masker,
        ablation_mode=ABLATION_MODE
    )
    
    if os.path.exists(CKPT_LATEST):
        try:
            global_ai.policy_net.load_state_dict(safe_torch_load(CKPT_LATEST))
            print(f"🧠 [RESUME] Pesi caricati da '{CKPT_LATEST}' (Modalità: {ABLATION_MODE})!")
        except Exception as e:
            print(f"⚠️ Punto di ripresa '{CKPT_LATEST}' incompatibile ({e}). Si parte da zero per {ABLATION_MODE}.")
            
    num_episodes = 800
    
    max_workers = max(4, (os.cpu_count() or 8) - 4)
    batch_size = max(6, max_workers // 2)
    total_batches = max(1, num_episodes // batch_size)
    
    ppo_epochs = 4
    clip_eps = 0.20
    vf_coef = 0.50
    
    optimizer = optim.Adam(global_ai.policy_net.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=0.00008)
    
    phase = 1
    recent_outcomes = []
    best_win_rate = 0.0
    total_episodes_played = 0
    
    batch_trajectories = []
    batch_wins = 0
    batch_turns = []
    all_formulas = []
    
    print(f"🚀 Avvio Addestramento [{ABLATION_MODE}] ({max_workers} Workers | Batch: {batch_size} | Save: '{CKPT_BEST}')")
    
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

                traj, outcome, turns, worker_cache, formulas = future.result()
                total_episodes_played += 1
                
                global_vitamin.cache.update(worker_cache)
                global_vitamin.unsaved_changes += 1
                
                recent_outcomes.append(outcome)
                if len(recent_outcomes) > 50:
                    recent_outcomes.pop(0)
                    
                if traj is not None:
                    batch_trajectories.append(traj)
                
                batch_wins += outcome
                batch_turns.append(turns)
                all_formulas.extend(formulas)
                
                if len(batch_trajectories) >= batch_size:
                    all_states = []
                    all_token_seqs = []
                    all_old_lps = []
                    all_advs = []
                    all_rets = []
                    
                    for t_data in batch_trajectories:
                        all_states.extend(t_data["states"])
                        all_token_seqs.extend(t_data["token_seqs"])
                        all_old_lps.append(t_data["old_log_probs"])
                        all_advs.append(t_data["advantages"])
                        all_rets.append(t_data["returns"])
                        
                    old_lps_t = torch.cat(all_old_lps).detach()
                    advs_t = torch.cat(all_advs).detach()
                    rets_t = torch.cat(all_rets).detach()
                    
                    if advs_t.numel() > 1:
                        advs_t = torch.clamp((advs_t - advs_t.mean()) / (advs_t.std() + 1e-5), -2.5, 2.5)
                    else:
                        advs_t = torch.clamp(advs_t, -2.5, 2.5)
                        
                    ent_coef = max(0.002, 0.025 * (1.0 - total_episodes_played / num_episodes))
                    
                    global_ai.policy_net.train()
                    total_ppo_loss = 0.0
                    
                    for _ in range(ppo_epochs):
                        new_lps = []
                        new_vals = []
                        entropies = []
                        
                        for s_ten, tok_seq in zip(all_states, all_token_seqs):
                            lp, val, ent = evaluate_step_for_ppo(global_ai.policy_net, global_masker, s_ten, tok_seq)
                            new_lps.append(lp)
                            new_vals.append(val)
                            entropies.append(ent)
                            
                        new_lps_t = torch.stack(new_lps)
                        new_vals_t = torch.stack(new_vals)
                        entropy_t = torch.stack(entropies).mean()
                        
                        ratios = torch.exp(torch.clamp(new_lps_t - old_lps_t, -10.0, 10.0))
                        
                        surr1 = ratios * advs_t
                        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advs_t
                        
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = vf_coef * F.smooth_l1_loss(new_vals_t, rets_t)
                        loss = actor_loss + critic_loss - (ent_coef * entropy_t)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(global_ai.policy_net.parameters(), max_norm=0.5)
                        optimizer.step()
                        
                        total_ppo_loss += loss.item()
                        
                    scheduler.step()

                    avg_loss = total_ppo_loss / ppo_epochs
                    avg_turns = sum(batch_turns) / len(batch_turns)
                    rolling_wr = (sum(recent_outcomes) / len(recent_outcomes)) * 100
                    
                    succ_counts = Counter([f for f, s in all_formulas if s == "SUCCESS"])
                    rej_counts = Counter([f for f, s in all_formulas if s == "REJECTED"])
                    
                    top_succ_str = ", ".join([f"'{k}' ({v})" for k, v in succ_counts.most_common(3)])
                    top_rej_str = ", ".join([f"'{k}' ({v})" for k, v in rej_counts.most_common(2)])
                    
                    print(f"📦 [{ABLATION_MODE}] Ep: {total_episodes_played}/{num_episodes} | Fase {phase} | Win: {batch_wins}/{len(batch_turns)} | WR(50): {rolling_wr:.1f}% | Turni: {avg_turns:.1f} | Loss: {avg_loss:.2f} | Cache: {len(global_vitamin.cache)}")
                    
                    if top_rej_str: print(f"   ❌ Top Rifiutate: {top_rej_str}")
                    if top_succ_str: print(f"   ✅ Top Validate : {top_succ_str}")
                    print("-" * 75)
                    
                    # Salvataggio atomico in RAM + Disco sul file specifico della modalità
                    safe_torch_save(global_ai.policy_net.state_dict(), CKPT_LATEST)
                    
                    if len(recent_outcomes) >= 30 and rolling_wr >= best_win_rate:
                        best_win_rate = rolling_wr
                        safe_torch_save(global_ai.policy_net.state_dict(), CKPT_BEST)
                        if ABLATION_MODE == "FULL":
                            safe_torch_save(global_ai.policy_net.state_dict(), "neuro_symbolic_warrior_best.pth")
                    
                    batch_trajectories = []
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
                        print(f"🎓 CURRICULUM AVANZATO ({ABLATION_MODE})! Win-rate {win_rate*100:.0f}% raggiunto.")
                        print("🔥 Fase 2: Sparring contro BehaviorTreeAI attivo in tutti i core!")
                        print("🔥"*25 + "\n")

    global_vitamin.save_cache()
    
    if os.path.exists(CKPT_BEST):
        best_weights = safe_torch_load(CKPT_BEST)
        safe_torch_save(best_weights, CKPT_LATEST)
        if ABLATION_MODE == "FULL":
            safe_torch_save(best_weights, "neuro_symbolic_warrior.pth")
            safe_torch_save(best_weights, "neuro_symbolic_warrior_parallel.pth")
    
    final_win_rate = (sum(recent_outcomes) / len(recent_outcomes) * 100) if recent_outcomes else 0.0
    print(f"\n🎉 ADDESTRAMENTO [{ABLATION_MODE}] COMPLETATO! Win-rate finale Fase {phase}: {final_win_rate:.1f}% (Best salvato in '{CKPT_BEST}': {best_win_rate:.1f}%)")