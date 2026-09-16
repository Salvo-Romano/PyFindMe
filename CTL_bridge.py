import copy
from pyFighters import Action, LogicFightersState

# --- UTILITIES DI ASTRAZIONE ---
def discretize_hp(hp, max_hp):
    if hp <= 0:
        return 0  # MORTO
    
    ratio = hp / max_hp
    if ratio <= 0.30:
        return 1  # CRITICO 
    elif ratio <= 0.70:
        return 2  # MEDIO
    else:
        return 3  # ALTO

def discretize_special(special, max_sp):
    if special >= max_sp:
        return 2  # PRONTO 
    elif special > 0:
        return 1  # IN CARICA
    else:
        return 0  # VUOTO


class UserPredictor:
    def __init__(self):
        self.actions = [Action.ATTACK, Action.DEFEND, Action.COUNTER, Action.BUFF]
        self.counts = {
            "neutral": {a: 1.0 for a in self.actions},
            "special_ready": {Action.ATTACK: 4.0, Action.SPECIAL: 6.0, Action.DEFEND: 1.0, Action.COUNTER: 1.0, Action.BUFF: 1.0},
            "critical_hp": {Action.ATTACK: 1.0, Action.DEFEND: 4.0, Action.COUNTER: 3.0, Action.BUFF: 1.0}
        }

    def _get_context(self, player):
        if player.sp >= player.stats["sp_threshold"]:
            return "special_ready"
        if player.hp <= int(player.stats["max_hp"] * 0.35):
            return "critical_hp"
        return "neutral"

    def record_action(self, player_before, chosen_action):
        ctx = self._get_context(player_before)
        if chosen_action not in self.counts[ctx]:
            self.counts[ctx][chosen_action] = 0.0
        self.counts[ctx][chosen_action] += 1.0

    def get_distribution(self, player):
        ctx = self._get_context(player)
        table = self.counts[ctx]
        
        avail = [Action.DEFEND, Action.COUNTER, Action.BUFF]
        if player.sp >= player.stats["sp_threshold"]:
            avail.append(Action.SPECIAL)
        else:
            avail.append(Action.ATTACK)

        total = sum(table.get(a, 1.0) for a in avail)
        return {a: table.get(a, 1.0) / total for a in avail}


class StateTreeNode:
    def __init__(self, state_snapshot, depth=0, parent_prob=1.0):
        self.state = state_snapshot
        self.depth = depth
        self.prob = parent_prob
        self.transitions = []
        self.atomic_props = self._evaluate_atomic_props()

    def _evaluate_atomic_props(self):
        p, n = self.state.player, self.state.npc
        props = set()
        
        # Le proposizioni ora possono usare direttamente le stesse regole di discretizzazione per coerenza
        if p.hp <= 0: props.add("Dead_P")
        if n.hp <= 0: props.add("Dead_N")
        if n.hp > 0: props.add("Alive_N") 
        if discretize_hp(p.hp, p.stats["max_hp"]) == 1: props.add("LowHP_P")
        if discretize_hp(n.hp, n.stats["max_hp"]) == 1: props.add("LowHP_N")
        
        if n.hp > p.hp: props.add("HP_Advantage_N")
        
        if p.sp >= p.stats["sp_threshold"]: props.add("SpecialReady_P")
        elif p.sp >= p.stats["sp_threshold"] - 2: props.add("SpecialDanger_P")
        
        if n.sp >= n.stats["sp_threshold"]: props.add("SpecialReady_N")
        elif n.sp >= n.stats["sp_threshold"] - 2: props.add("SpecialDanger_N")
        
        if p.def_buff_turns > 0: props.add("DefBuff_P")
        if p.atk_buff_turns > 0: props.add("AtkBuff_P")
        if n.def_buff_turns > 0: props.add("DefBuff_N")
        if n.atk_buff_turns > 0: props.add("AtkBuff_N")
        
        if getattr(p, "counter_cooldown", 0) == 0: props.add("CounterReady_P")
        if getattr(n, "counter_cooldown", 0) == 0: props.add("CounterReady_N")
        
        return props


def clone_state(base_state):
    new_s = LogicFightersState(base_state.player.char_class, base_state.npc.char_class)
    new_s.turn = base_state.turn
    new_s.player = copy.deepcopy(base_state.player)
    new_s.npc = copy.deepcopy(base_state.npc)
    return new_s

def get_state_signature(state, current_depth):
    """
    Comprime lo stato in una firma a bassa cardinalità per massimizzare
    i cache hit (Match) ed evitare lo State Space Explosion, 
    mantenendo 4 fasce di salute per preservare i dettagli tattici.
    """
    def abstract_hp(hp, max_hp):
        ratio = hp / max_hp if max_hp > 0 else 0
        if ratio > 0.75: return 3  # Ottima salute
        if ratio > 0.50: return 2  # Buona salute
        if ratio > 0.25: return 1  # Ferito / Attenzione
        return 0                   # Critico

    def abstract_sp(sp, threshold):
        if sp >= threshold: return 2 # Special Pronta
        if sp > 0: return 1          # In Carica
        return 0                     # Vuoto

    p1, p2 = state.player, state.npc
    
    return (
        current_depth,
        
        # Astrazione Player
        abstract_hp(p1.hp, p1.stats["max_hp"]),
        abstract_sp(p1.sp, p1.stats["sp_threshold"]),
        1 if p1.atk_buff_turns > 0 else 0,
        1 if p1.def_buff_turns > 0 else 0,
        
        # Astrazione NPC
        abstract_hp(p2.hp, p2.stats["max_hp"]),
        abstract_sp(p2.sp, p2.stats["sp_threshold"]),
        1 if p2.atk_buff_turns > 0 else 0,
        1 if p2.def_buff_turns > 0 else 0
    )

def build_horizon_tree(current_state, user_predictor, max_depth=3, cur_depth=0, current_prob=1.0, prune_threshold=0.15, memo=None, use_dag=True, last_p_act=None, last_n_act=None):
    if use_dag and memo is None:
        memo = {}
        
    if use_dag:
        signature = get_state_signature(current_state, cur_depth)
        memo_key = (signature, last_p_act, last_n_act)
        
        if memo_key in memo:
            return memo[memo_key]
            
    node = StateTreeNode(current_state, depth=cur_depth, parent_prob=current_prob)
    
    if use_dag:
        memo[memo_key] = node
        
    if cur_depth >= max_depth or current_state.is_finished():
        return node

    p_dist = user_predictor.get_distribution(current_state.player)
    
    # --- 1. APPLICAZIONE REGOLE E NORMALIZZAZIONE PLAYER ---
    COOLDOWN_ACTIONS = [Action.COUNTER]
    adjusted_p_dist = {}
    
    # Check Buff Player: Il buff dura finché "def_buff_turns" > 0 (adatta al nome della tua variabile)
    # Assumo che esista un attributo simile a 'def_buff_turns' basandomi sulla tua logica
    p_has_buff = current_state.player.def_buff_turns > 0 or current_state.player.atk_buff_turns > 0
    
    for a_p, prob in p_dist.items():
        if last_p_act in COOLDOWN_ACTIONS and a_p == last_p_act:
            continue
            
        if p_has_buff and a_p == Action.BUFF:
            continue
            
        if a_p == last_p_act:
            prob *= 0.70 
            
        adjusted_p_dist[a_p] = adjusted_p_dist.get(a_p, 0.0) + prob
        
    # Regola A (Migliorata): Trasferimento e Boost per la Special
    if current_state.player.sp >= current_state.player.stats["sp_threshold"]:
        attack_prob = adjusted_p_dist.pop(Action.ATTACK, 0.0)
        special_prob = adjusted_p_dist.get(Action.SPECIAL, 0.0)
        
        # Uniamo i valori e diamo un bonus psicologico del 50% alla voglia di usare la special
        boosted_special = (attack_prob + special_prob) * 1.5 
        adjusted_p_dist[Action.SPECIAL] = boosted_special

    # Normalizzazione finale Player
    total_prob = sum(adjusted_p_dist.values())
    if total_prob > 0:
        adjusted_p_dist = {k: v / total_prob for k, v in adjusted_p_dist.items()}

    # --- 2. APPLICAZIONE REGOLE NPC ---
    npc_actions = [Action.DEFEND, Action.COUNTER, Action.BUFF]
    if current_state.npc.sp >= current_state.npc.stats["sp_threshold"]:
        npc_actions.append(Action.SPECIAL)
    else:
        npc_actions.append(Action.ATTACK)
        
    if last_n_act in COOLDOWN_ACTIONS and last_n_act in npc_actions:
        npc_actions.remove(last_n_act)
        
    # Check Buff NPC
    n_has_buff = current_state.npc.def_buff_turns > 0 or current_state.npc.atk_buff_turns > 0
    if n_has_buff and Action.BUFF in npc_actions:
        npc_actions.remove(Action.BUFF)
            
    # --- 3. CREAZIONE RAMI ---
    for a_p, prob_p in adjusted_p_dist.items():
        if prob_p < prune_threshold:
            continue
            
        for a_n in npc_actions:
            next_s = clone_state(current_state)
            next_s.apply_action_resolution(a_p, a_n)
            
            branch_prob = current_prob * prob_p
            
            child_node = build_horizon_tree(
                next_s, user_predictor,
                max_depth=max_depth,
                cur_depth=cur_depth + 1,
                current_prob=branch_prob,
                prune_threshold=prune_threshold,
                memo=memo,
                use_dag=use_dag,
                last_p_act=a_p,  # Propaga la mossa del nodo corrente al livello successivo
                last_n_act=a_n
            )
            node.transitions.append((a_p, a_n, prob_p, child_node))

    return node
def generate_vitamin_model(root_node):
    nodes = []
    def collect_nodes(node):
        if node not in nodes:
            nodes.append(node)
            for _, _, _, child in node.transitions:
                collect_nodes(child)
                
    collect_nodes(root_node)
    
    for node in nodes:
        if not node.transitions:
            node.atomic_props.add("Horizon")
            
    state_names = [f"s{i}" for i in range(len(nodes))]
    all_aps = set()
    for n in nodes:
        all_aps.update(n.atomic_props)
    all_aps = sorted(list(all_aps))
    
    if not all_aps:
        all_aps = ["neutral"]
        for n in nodes:
            n.atomic_props.add("neutral")

    matrix = [["0" for _ in nodes] for _ in nodes]
    for i, node in enumerate(nodes):
        if not node.transitions:
            matrix[i][i] = "1" # Anche il loop deve essere 1, non la stringa "loop"
        else:
            for ap_action, npc_action, _, child in node.transitions:
                j = nodes.index(child)
                matrix[i][j] = "1" # CRITICO: Solo "1", nessuna etichetta stringa!

    lines = []
    
    lines.append("Transition")
    for row in matrix:
        lines.append(" ".join(row))
        
    lines.append("Name_State")
    lines.append(" ".join(state_names))
    
    lines.append("Initial_State")
    lines.append(state_names[0])
    
    lines.append("Atomic_propositions")
    lines.append(" ".join(all_aps))
    
    lines.append("Labelling")
    for node in nodes:
        row = []
        for ap in all_aps:
            row.append("1" if ap in node.atomic_props else "0")
        lines.append(" ".join(row))
        
    lines.append("Number_of_agents")
    lines.append("1")
    
    return "\n".join(lines)

def parse_vitamin_trace(response_json, root_node):
    verification = response_json.get("verification", {})
    trace_states = verification.get("satisfying_states", [])
    
    if not trace_states:
        return Action.DEFEND, set()

    nodes = []
    def collect_nodes(node):
        if node not in nodes:
            nodes.append(node)
            for _, _, _, child in node.transitions:
                collect_nodes(child)
                
    collect_nodes(root_node)

    safe_nodes = set()
    for state_name in trace_states:
        try:
            idx = int(state_name.replace("s", ""))
            safe_nodes.add(nodes[idx])
        except (ValueError, IndexError):
            continue

    action_scores = {}
    for p_act, n_act, prob, child in root_node.transitions:
        if n_act not in action_scores:
            action_scores[n_act] = 0.0
        
        if child in safe_nodes:
            action_scores[n_act] += prob

    if not action_scores:
        return Action.DEFEND, set()
        
    best_action = max(action_scores.items(), key=lambda x: x[1])[0]
    return best_action, safe_nodes