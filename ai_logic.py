from CTL_bridge import build_horizon_tree, generate_vitamin_model, get_state_signature
from pyFighters import Action

class PureLogicAI:
    def __init__(self, vitamin_client, user_predictor):
        self.vitamin = vitamin_client
        self.user_predictor = user_predictor
        self.current_plan = [] 
        
        # Le formule usano E(Safe U Goal) per ottenere un witness path garantendo la sopravvivenza
        self.formulas = [
            "E(Alive_N U Dead_P)",                                  # 1. Uccidi il giocatore restando vivo
            "E(Alive_N U (SpecialReady_N & HP_Advantage_N))",       # 2. Carica la special mantenendo il vantaggio
            "E(Alive_N U DefBuff_N)",                               # 3. Arriva ad attivare un buff in sicurezza
            "EF Alive_N"                                            # 4. Sopravvivenza base
        ]

    def _extract_plan_from_trace(self, response, root_node, nodes_list):
        trace_path = response.get("trace", {}).get("path", [])
        plan = []
        
        if not trace_path or len(trace_path) < 2:
            return plan

        for i in range(len(trace_path) - 1):
            curr_s, next_s = trace_path[i], trace_path[i+1]
            try:
                curr_idx = int(curr_s.replace("s", ""))
                next_idx = int(next_s.replace("s", ""))
                curr_node = nodes_list[curr_idx]
                target_node = nodes_list[next_idx]
            except (ValueError, IndexError):
                break
                
            best_action = None
            best_prob = -1
            # Cerchiamo l'azione dell'NPC che ha la probabilità più alta di portarci nel target previsto
            for p_act, n_act, prob, child in curr_node.transitions:
                if child == target_node and prob > best_prob:
                    best_prob = prob
                    best_action = n_act
                    
            if best_action:
                plan.append({
                    "action": best_action,
                    "expected_signature": target_node.signature
                })
            else:
                break
        return plan

    def decide_action(self, game_state):
        current_signature = get_state_signature(game_state, current_depth=0)
        
        # 1. Verifica di conformità: il piano è ancora valido?
        if self.current_plan:
            expected_signature = self.current_plan[0]["expected_signature"]
            
            # Tagliamo l'indice 0 (la depth) dalla tupla per confrontare solo i macro-valori di stato
            if current_signature[1:] == expected_signature[1:]:
                print("[PURE LOGIC] Stato conforme al piano atteso! Eseguo mossa in cache.")
                step = self.current_plan.pop(0)
                return step["action"]
            else:
                print("[PURE LOGIC] Disallineamento rilevato (il giocatore ha deviato). Ricalcolo...")
                self.current_plan = []

        # 2. Ricalcolo dell'albero e chiamata a VITAMIN
        horizon_tree = build_horizon_tree(game_state, self.user_predictor, max_depth=3, use_dag=True)
        model_text = generate_vitamin_model(horizon_tree)
        
        # Rigeneriamo la mappa dei nodi per associare l'output testuale agli oggetti in memoria
        nodes = []
        def collect_nodes(node):
            if node not in nodes:
                nodes.append(node)
                for _, _, _, child in node.transitions:
                    collect_nodes(child)
        collect_nodes(horizon_tree)
        
        for formula in self.formulas:
            is_satisfied, response = self.vitamin.verify_formula(model_text, formula, generate_trace=True)
            
            if is_satisfied:
                print(f"[PURE LOGIC] Traccia trovata per la formula: {formula}")
                self.current_plan = self._extract_plan_from_trace(response, horizon_tree, nodes)
                
                if self.current_plan:
                    step = self.current_plan.pop(0)
                    return step["action"]
                else:
                    # Fallback logico se la formula è soddisfatta ma la traccia è troppo corta (es. obiettivo già raggiunto)
                    return Action.ATTACK if game_state.npc.sp < game_state.npc.stats["sp_threshold"] else Action.SPECIAL
                
        print("[PURE LOGIC] Nessuna dimostrazione sicura trovata. Mossa difensiva di emergenza.")
        return Action.DEFEND