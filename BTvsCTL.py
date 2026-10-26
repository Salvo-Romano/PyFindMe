import time
import os
from pyFighters import Action, LogicFightersState
from CTL_bridge import UserPredictor
from vitamin_client import VitaminClient
from ai_behavior_tree import BehaviorTreeAI
from ai_logic import PureLogicAI

print("⚙️ Inizializzazione Torneo Headless (BT vs Pure Logic)...")

# 1. SETUP DEI CLIENT E DELLE IA
# Inserisci la tua API Key di Vitamin
vitamin_client = VitaminClient(api_key="821b5bd238df54433364778326920866cca018890e4b4ccc2391853eefedc56e")
user_predictor = UserPredictor()

bt_ai = BehaviorTreeAI()
logic_ai = PureLogicAI(vitamin_client, user_predictor)

num_matches = 10 # 10 match per girone (90 match totali)
max_turns = 30
classes = ["Warrior", "Mage", "Ranger"]

tournament_results = []
report_lines = []
report_lines.append("==================================================")
report_lines.append("🏆 TORNEO HEADLESS: BEHAVIOR TREE vs PURE LOGIC AI")
report_lines.append("==================================================\n")

print(f"Avvio torneo: 9 gironi, {num_matches} match per girone.\n")

for bt_class in classes:
    for logic_class in classes:
        print(f"⚔️ GIRONE: BT [{bt_class}] VS Pure Logic [{logic_class}]")
        
        stats = {
            "bt_wins": 0, 
            "logic_wins": 0, 
            "draws": 0, 
            "total_turns": 0
        }
        
        for match in range(num_matches):
            state = LogicFightersState(bt_class, logic_class)
            # FIX: Assegniamo esplicitamente le classi per il BT
            state.player.char_class = bt_class
            state.npc.char_class = logic_class
            
            turns = 0
            last_bt_action = None
            last_logic_action = None
            
            # Reset del Predictor per evitare che si porti dietro la "memoria" del match precedente
            user_predictor.action_history = []
            user_predictor.class_transitions = {}
            
            # Innesco Pure Logic
            logic_ai.evaluate_post_turn(state, None, None)
            
            while not state.is_finished() and turns < max_turns:
                turns += 1
                
                # BT (Player)
                bt_action = bt_ai.decide_action(state, npc_role="player") 
                
                # Pure Logic (NPC)
                logic_action = logic_ai.decide_action(state, last_player_action=last_bt_action, last_npc_action=last_logic_action)
                
                if logic_action == "THINKING":
                    while logic_action == "THINKING":
                        time.sleep(0.05) 
                        logic_action = logic_ai.decide_action(state, last_player_action=last_bt_action)
                        
                result = state.apply_action_resolution(bt_action, logic_action)
                
                last_bt_action = bt_action
                last_logic_action = logic_action
                
                logic_ai.evaluate_post_turn(state, last_bt_action, last_logic_action)
                
            stats["total_turns"] += turns
            
            if state.is_finished():
                if state.player.hp > 0:
                    stats["bt_wins"] += 1
                elif state.npc.hp > 0:
                    stats["logic_wins"] += 1
                else:
                    stats["draws"] += 1
            else:
                stats["draws"] += 1
                
        # Salvataggio statistiche del girone
        avg_turns = stats["total_turns"] / num_matches
        bt_winrate = (stats["bt_wins"] / num_matches) * 100
        logic_winrate = (stats["logic_wins"] / num_matches) * 100
        
        line = f"BT {bt_class} vs LOGIC {logic_class} -> BT: {stats['bt_wins']} ({bt_winrate:.0f}%) | LOGIC: {stats['logic_wins']} ({logic_winrate:.0f}%) | Stalli: {stats['draws']} | Turni Medi: {avg_turns:.1f}"
        print("  " + line + "\n")
        report_lines.append(line)
        
        tournament_results.append({
            "bt_class": bt_class,
            "logic_class": logic_class,
            "bt_wins": stats["bt_wins"],
            "logic_wins": stats["logic_wins"]
        })

# Generazione Report File
report_lines.append("\n==================================================")
report_lines.append("📊 RIASSUNTO GLOBALE")
report_lines.append("==================================================")

total_bt = sum(r["bt_wins"] for r in tournament_results)
total_logic = sum(r["logic_wins"] for r in tournament_results)
total_matches_played = num_matches * 9

report_lines.append(f"Match Totali Giocati : {total_matches_played}")
report_lines.append(f"Vittorie TOTALI BT   : {total_bt} ({(total_bt/total_matches_played)*100:.1f}%)")
report_lines.append(f"Vittorie TOTALI LOGIC: {total_logic} ({(total_logic/total_matches_played)*100:.1f}%)")

with open("report_torneo_bt_vs_logic.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
    
print("✅ Torneo concluso! Risultati salvati in 'report_torneo_bt_vs_logic.txt'.")