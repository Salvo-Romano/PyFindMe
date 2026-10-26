import threading
from concurrent.futures import ThreadPoolExecutor 
from CTL_bridge import build_horizon_tree, generate_vitamin_model, get_state_signature
from pyFighters import Action
class PureLogicAI:
    def __init__(self, vitamin_client, user_predictor):
        self.vitamin = vitamin_client
        self.user_predictor = user_predictor
        
        self.current_plan = []
        self.expected_current_signature = None
        self.last_expected_player_action = None
        self.last_prob = 0.0
        
        # Esposto per il debugging grafico esterno
        self.horizon_tree = None
        
        self.is_thinking = False
        
        # --- NUOVO: IL WORKER PERSISTENTE ---
        # Creiamo un singolo thread dedicato che rimarrà vivo per tutta la partita
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="AI_Logic_Worker")
        # ------------------------------------
        
        self.formulas = [
            "E(Alive_N U Dead_P)",                                      
            "E(Alive_N U (Horizon and LowHP_P))",                         
            "E(Alive_N U (Horizon and SpecialReady_N and HP_Advantage_N))", 
            "E(Alive_N U Horizon)"                                      
        ]

    def shutdown(self):
        """Metodo di pulizia da chiamare quando la partita finisce (opzionale ma consigliato)"""
        self.executor.shutdown(wait=False)

    # ... [ I METODI _evaluate_transition E _extract_plan_from_trace RIMANGONO IDENTICI A PRIMA ] ...
    
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

            print("\n  [EXPECTIMAX EVALUATION]")
            for n_act, data in action_evals.items():
                is_safe = data['target_node'] in safe_nodes
                safe_str = "✅ SICURA (in CTL)" if is_safe else "❌ RISCHIOSA"
                print(f"    ├─ Valuto {n_act}: EV = {data['expected_utility']:>6.2f} | {safe_str}")

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
        # Calcolo dell'orizzonte dinamico basato sui quartili di salute
        p_ratio = game_state.player.hp / game_state.player.stats["max_hp"]
        n_ratio = game_state.npc.hp / game_state.npc.stats["max_hp"]
        
        target_depth = 3 if (p_ratio <= 0.5 or n_ratio <= 0.5) else 2
        
        print(f"\n[GENERAZIONE DAG...] Costruzione albero (Profondità Dinamica: {target_depth})")
        
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
        
        for formula in self.formulas:
            is_satisfied, response = self.vitamin.verify_formula(model_text, formula, generate_trace=True)
            
            if is_satisfied:
                print(f"\n[ORACOLO VITAMIN] ✨ Traccia garantita trovata per la proprietà: {formula}")
                self.current_plan = self._extract_plan_from_trace(response, self.horizon_tree, nodes)
                
                if not self.current_plan:
                    print(f" ⚠️ ATTENZIONE: Traccia vuota per {formula}. Passo alla prossima euristica.")
                    continue
                
                print("\n[PIANO TATTICO ESTRATTO]")
                for i, step in enumerate(self.current_plan):
                    print(f"  STEP {i+1}:")
                    print(f"    ├─ Se il Player usa: {step['expected_player_action']} (Prob: {step['probability']:.2f})")
                    print(f"    ├─ L'NPC risponderà : {step['npc_action']}")
                    print(f"    ├─ Per arrivare in : {step['expected_signature'][1:]}")
                    print(f"    └─ Proprietà attive: {step['expected_props']}")
                
                self.is_thinking = False
                return

        print("\n ⚠️ [ALLARME] Nessuna dimostrazione sicura trovata in tutto il DAG!")
        print(" 🛡️ Esecuzione manovra di emergenza (DEFEND).")
        self.current_plan = [{
            "npc_action": Action.DEFEND, 
            "expected_signature": get_state_signature(game_state, 0),
            "expected_player_action": "UNKNOWN",
            "probability": 1.0,
            "expected_props": []
        }]
        self.is_thinking = False

    def evaluate_post_turn(self, game_state, last_player_action, last_npc_action):
        if self.is_thinking:
            return

        current_signature = get_state_signature(game_state, current_depth=0)
        
        print("\n" + "="*60)
        print("                TELEMETRIA DI LOGICA FORMALE               ")
        print("="*60)
        
        if self.expected_current_signature:
            print("\n[ANALISI RISULTATO TURNO PRECEDENTE]")
            print(f" > L'IA si aspettava che il Player facesse: {self.last_expected_player_action} (Probabilità: {self.last_prob:.2f})")
            print(f" > Il Player ha REALMENTE fatto         : {last_player_action}")
            print(f" > Astrazione Attesa (Firma)            : {self.expected_current_signature[1:]}")
            print(f" > Astrazione Reale  (Firma)            : {current_signature[1:]}")
            
            if current_signature[1:] == self.expected_current_signature[1:]:
                print("\n ✅ MATCH! Il giocatore si è comportato in modo conforme all'astrazione.")
            else:
                print("\n ❌ MISS! Disallineamento rilevato nello stato astratto.")
                print(" 🔄 Piano invalidato. Svuoto la cache.")
                self.current_plan = []
                self.expected_current_signature = None
        else:
            print("\n[NESSUN PIANO DA VALIDARE]")

        if not self.current_plan:
            print("\n 🧠 [IDLE PLANNING] Delego calcolo al Worker in background...")
            self.is_thinking = True
            # --- MODIFICA: Invio il task all'Executor invece di creare un nuovo thread ---
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
            
            print(f"\n 🚀 Eseguo mossa dalla cache: {action}")
            print("="*60 + "\n")
            return action
            
        self.is_thinking = True
        # --- MODIFICA: Invio il task all'Executor invece di creare un nuovo thread ---
        self.executor.submit(self._async_compute, game_state, last_player_action, last_npc_action)
        return "THINKING"