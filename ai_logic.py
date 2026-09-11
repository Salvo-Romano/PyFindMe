from CTL_bridge import build_horizon_tree, generate_vitamin_model, parse_vitamin_trace
from pyFighters import Action

class PureLogicAI:
    def __init__(self, vitamin_client, user_predictor):
        self.vitamin = vitamin_client
        self.user_predictor = user_predictor
        self.expected_next_states = None
        self.formulas = [
            "EF (Dead_P & Alive_N)",            # 1. Chiudi la partita se il giocatore è a un colpo dalla morte
            "EF (LowHP_P & Alive_N)",           # 2. Manda il giocatore in stato critico
            "EF (SpecialReady_N & Alive_N)",    # 3. Carica la tua mossa speciale in sicurezza
            "EF (DefBuff_N & Alive_N)",         # 4. Attiva un buff difensivo per prepararti all'impatto
            "EF Alive_N"                        # 5. Sopravvivi e basta
        ]

    def update_plan(self, game_state):
        horizon_tree = build_horizon_tree(game_state, self.user_predictor, max_depth=2, use_dag=True)
        model_text = generate_vitamin_model(horizon_tree)
        
        for formula in self.formulas:
            is_satisfied, response = self.vitamin.verify_formula(model_text, formula, generate_trace=True)
            
            if is_satisfied:
                print(f"[PURE LOGIC] Strategia trovata con formula: {formula}")
                best_action, safe_nodes = parse_vitamin_trace(response, horizon_tree)
                self.expected_next_states = safe_nodes
                return best_action
                
        print("[PURE LOGIC] Nessun percorso sicuro trovato. Mossa difensiva di fallback.")
        self.expected_next_states = None
        return Action.DEFEND

    def decide_action(self, game_state):
        if self.expected_next_states is None or game_state not in self.expected_next_states:
            return self.update_plan(game_state)
        
        print("[PURE LOGIC] Piano in corso. Aggiornamento transizione...")
        return self.update_plan(game_state)