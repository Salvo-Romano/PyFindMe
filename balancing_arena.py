from ai_behavior_tree import BehaviorTreeAI
from pyFighters import Action, LogicFightersState
# Assicurati di importare anche la tua BehaviorTreeAI

def run_balancing_arena(num_matches=1000, max_turns=100):
    classes = ["Warrior", "Mage", "Ranger"]
    ai = BehaviorTreeAI()
    
    print(f"Avvio simulazione: {num_matches} match per ogni matchup...\n")
    print(f"{'MATCHUP':<20} | {'VITTORIE P1':<12} | {'VITTORIE P2':<12} | {'PAREGGI':<8}")
    print("-" * 60)
    
    for c1 in classes:
        for c2 in classes:
            matchup_name = f"{c1} vs {c2}"
            wins_p1 = 0
            wins_p2 = 0
            draws = 0
            
            for _ in range(num_matches):
                # Inizializziamo lo stato per il match corrente
                state = LogicFightersState(player_class=c1, npc_class=c2)
                
                # Setup di sicurezza per i cooldown nel caso non siano nell'__init__
                state.player.counter_cooldown = 0
                state.npc.counter_cooldown = 0
                
                while not state.is_finished() and state.turn < max_turns:
                    # Chiediamo all'albero di decidere per entrambi
                    action_p1 = ai.decide_action(state, npc_role="player")
                    action_p2 = ai.decide_action(state, npc_role="npc")
                    
                    # Risoluzione del turno
                    state.apply_action_resolution(action_p1, action_p2)
                    
                    # Gestione manuale dei cooldown del counter per la simulazione
                    if action_p1 == Action.COUNTER:
                        state.player.counter_cooldown = 1
                    else:
                        state.player.counter_cooldown = max(0, state.player.counter_cooldown - 1)
                        
                    if action_p2 == Action.COUNTER:
                        state.npc.counter_cooldown = 1
                    else:
                        state.npc.counter_cooldown = max(0, state.npc.counter_cooldown - 1)
                
                # Valutazione risultato
                winner = state.winner()
                if winner == state.player.name:
                    wins_p1 += 1
                elif winner == state.npc.name:
                    wins_p2 += 1
                else:
                    draws += 1
                    
            # Calcolo percentuali
            p1_pct = (wins_p1 / num_matches) * 100
            p2_pct = (wins_p2 / num_matches) * 100
            draw_pct = (draws / num_matches) * 100
            
            print(f"{matchup_name:<20} | {c1}: {p1_pct:>5.1f}% | {c2}: {p2_pct:>5.1f}% | {draw_pct:>5.1f}%")

# Lancia l'arena
run_balancing_arena(num_matches=1000)