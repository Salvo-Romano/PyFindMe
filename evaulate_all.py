import os
import io
import torch
from collections import Counter

from pyFighters import Action, LogicFightersState
from CTL_bridge import UserPredictor
from vitamin_client import VitaminClient
from neuro_symbolic_env import CTLVocabulary, GrammarMasker
from ai_neuro_symbolic import NeuroSymbolicAI
from ai_logic import PureLogicAI
from ai_behavior_tree import BehaviorTreeAI

ATOMIC_PROPS = [
    "Dead_P", "CanKill_N", "LowHP_P", "HP_Advantage_N", 
    "SpecialReady_N", "Fortified_N", "Threat_P", "CounterReady_N", "Horizon"
]

CLASSES = ["Warrior", "Mage", "Ranger"]


def safe_load(path):
    with open(path, "rb") as f:
        return torch.load(io.BytesIO(f.read()), map_location="cpu")


def evaluate_agent(agent_name, mode, weight_path, vitamin_client, matches_per_pair=20, max_turns=35):
    vocab = CTLVocabulary(ATOMIC_PROPS)
    masker = GrammarMasker(vocab)
    bt_opponent = BehaviorTreeAI()

    if mode == "PURE_LOGIC":
        agent = PureLogicAI(vitamin_client, UserPredictor())
    else:
        agent = NeuroSymbolicAI(vitamin_client, UserPredictor(), vocab, masker, ablation_mode=mode)
        if os.path.exists(weight_path):
            agent.policy_net.load_state_dict(safe_load(weight_path), strict=False)
            agent.policy_net.eval()
        else:
            print(f"⚠️ File '{weight_path}' non trovato per {agent_name}! Salto.")
            return None

    wins, losses, draws = 0, 0, 0
    total_turns = 0
    total_mc_calls = 0
    total_rejected = 0
    formula_counter = Counter()
    class_wins = {c: 0 for c in CLASSES}
    class_ totals = {c: 0 for c in CLASSES}

    total_games = len(CLASSES) * len(CLASSES) * matches_per_pair

    for p_cls in CLASSES:
        for n_cls in CLASSES:
            for _ in range(matches_per_pair):
                state = LogicFightersState(p_cls, n_cls)
                state.player.char_class = p_cls
                state.npc.char_class = n_cls

                agent.user_predictor = UserPredictor()
                agent.current_plan = []
                if hasattr(agent, "pure_logic_ai"):
                    agent.user_predictor = UserPredictor()

                turns = 0
                last_p, last_n = None, None
                class_totals[n_cls] += 1

                while not state.is_finished() and turns < max_turns:
                    turns += 1
                    enemy_act = bt_opponent.decide_action(state, npc_role="player")
                    my_act = None

                    if mode == "FIXED_TIMING":
                        agent.current_plan = []

                    if agent.current_plan:
                        step = agent.current_plan[0]
                        if step.get("expected_player_action", enemy_act) == enemy_act:
                            my_act = agent.current_plan.pop(0)["npc_action"]
                        else:
                            agent.current_plan = []

                    if my_act is None:
                        if mode == "PURE_LOGIC":
                            agent._async_compute(state, last_p, last_n)
                            total_mc_calls += 1
                            if getattr(agent, "last_formula", None):
                                formula_counter[agent.last_formula] += 1
                            if agent.current_plan:
                                my_act = agent.current_plan.pop(0)["npc_action"]
                            else:
                                my_act = Action.DEFEND
                        else:
                            agent.excluded_portfolio_indices.clear()
                            for _ in range(3):
                                total_mc_calls += 1
                                status = agent._async_compute(state, last_p, last_n)
                                form = getattr(agent, "last_formula", "")
                                if status == "SUCCESS" and agent.current_plan:
                                    formula_counter[form] += 1
                                    my_act = agent.current_plan.pop(0)["npc_action"]
                                    break
                                else:
                                    total_rejected += 1
                            if my_act is None:
                                my_act = Action.DEFEND

                    agent.user_predictor.record_action(state.player, enemy_act)
                    state.apply_action_resolution(enemy_act, my_act)
                    last_p, last_n = enemy_act, my_act

                total_turns += turns
                if state.is_finished() and state.npc.hp > 0 and state.player.hp <= 0:
                    wins += 1
                    class_wins[n_cls] += 1
                elif state.is_finished() and state.player.hp > 0 and state.npc.hp <= 0:
                    losses += 1
                else:
                    draws += 1

    wr = (wins / total_games) * 100
    rej_rate = (total_rejected / max(1, total_mc_calls)) * 100
    calls_per_game = total_mc_calls / total_games
    avg_t = total_turns / total_games
    compound_pct = sum(v for k, v in formula_counter.items() if " and " in k or " or " in k) / max(1, sum(formula_counter.values())) * 100

    return {
        "name": agent_name,
        "wr": wr,
        "avg_turns": avg_t,
        "calls_per_game": calls_per_game,
        "rej_rate": rej_rate,
        "compound_pct": compound_pct,
        "class_wr": {c: (class_wins[c] / class_totals[c]) * 100 for c in CLASSES},
        "top_formulas": formula_counter.most_common(3)
    }


if __name__ == "__main__":
    import sys
    real_stdout = sys.stdout
    vitamin = VitaminClient()

    configs = [
        ("PureLogicAI (Baseline)", "PURE_LOGIC", ""),
        ("Ablation 1: Fixed Formulas", "FIXED_FORMULAS", "neuro_symbolic_fixed_formulas_best.pth"),
        ("Ablation 2: Fixed Timing", "FIXED_TIMING", "neuro_symbolic_fixed_timing_best.pth"),
        ("NeuroSymbolic FULL", "FULL", "neuro_symbolic_full_best.pth"),
    ]

    results = []
    for name, mode, path in configs:
        real_stdout.write(f"⏳ Valutazione di {name} su 180 match bilanciati...\n")
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        res = evaluate_agent(name, mode, path, vitamin, matches_per_pair=20)
        sys.stdout = real_stdout
        if res:
            results.append(res)

    vitamin.save_cache()

    print("\n" + "=" * 105)
    print(f"{'MODELLO':<28} | {'WIN RATE':<9} | {'TURNI':<6} | {'MC CALLS/MATCH':<14} | {'REJECT %':<9} | {'COMPOUND %':<10}")
    print("-" * 105)
    for r in results:
        print(f"{r['name']:<28} | {r['wr']:>6.1f}%  | {r['avg_turns']:>5.1f}  | {r['calls_per_game']:>12.1f}  | {r['rej_rate']:>7.1f}%  | {r['compound_pct']:>8.1f}%")
    print("=" * 105)

    for r in results:
        c_str = " | ".join([f"{c}: {r['class_wr'][c]:.1f}%" for c in CLASSES])
        f_str = ", ".join([f"'{k}' ({v})" for k, v in r['top_formulas']])
        print(f"🔹 {r['name']} -> Class WR [{c_str}]")
        print(f"   Top Formule: {f_str}\n")