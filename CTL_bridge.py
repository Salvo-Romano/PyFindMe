import copy
from pyFighters import Action, LogicFightersState

# --- UTILITIES DI ASTRAZIONE CALIBRATE SUI DANNI REALI (Warrior 70 HP, Base 5, Buffed 3) ---
def discretize_hp(hp, max_hp):
    if hp <= 0:
        return 0  # MORTO
    
    ratio = hp / max_hp if max_hp > 0 else 0.0
    if ratio <= 0.15: return 1
    if ratio <= 0.30: return 2  # Soglia critica LowHP
    if ratio <= 0.45: return 3
    if ratio <= 0.60: return 4
    if ratio <= 0.75: return 5
    if ratio <= 0.90: return 6
    return 7                    # Quasi intatto (64-70 HP)

def discretize_special(special, max_sp):
    if special >= max_sp:
        return 3  # PRONTO (6/6 SP)
    elif special >= max_sp - 2:
        return 2  # A UN PASSO DALLA SPECIALE (4/6 SP)
    elif special > 0:
        return 1  # INIZIO CARICA (2/6 SP)
    else:
        return 0  # VUOTO (0/6 SP)


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
    def __init__(self, state_snapshot, depth=0, parent_prob=1.0, root_state=None):
        self.state = state_snapshot
        self.depth = depth
        self.prob = parent_prob
        self.root_state = root_state if root_state is not None else state_snapshot
        self.transitions = []
        self.atomic_props = self._evaluate_atomic_props()

    def _evaluate_atomic_props(self):
        p, n = self.state.player, self.state.npc
        rp, rn = self.root_state.player, self.root_state.npc
        props = set()
        
        dmg_dealt = rp.hp - p.hp
        
        # 1. Sopravvivenza e Letalità (Dead_P si attiva sul KO o su danno pesante a nemico ferito)
        if p.hp <= 0 or (self.depth > 0 and p.hp <= p.stats["max_hp"] * 0.40 and dmg_dealt >= 5):
            props.add("Dead_P")
        if n.hp <= 0:
            props.add("Dead_N")
        if n.hp > 0:
            props.add("Alive_N") 
        
        # 2. LowHP_P: Vera se il Player è <= 30% HP OPPURE se nel ramo gli abbiamo inflitto
        # almeno 3 danni (1 ATTACK contro Warrior buffato fa esattamente 5 - 2 = 3 danni; DEFEND e BUFF fanno 0!)
        if discretize_hp(p.hp, p.stats["max_hp"]) <= 2 or (self.depth > 0 and dmg_dealt >= 3):
            props.add("LowHP_P")
            
        if discretize_hp(n.hp, n.stats["max_hp"]) <= 2:
            props.add("LowHP_N")
        
        # 3. HP_Advantage_N: Vera se l'NPC è in vantaggio di HP o migliora/mantiene il differenziale attaccando
        if self.depth == 0:
            if n.hp > p.hp:
                props.add("HP_Advantage_N")
        else:
            if n.hp > p.hp or ((n.hp - p.hp) >= (rn.hp - rp.hp) and dmg_dealt >= 3):
                props.add("HP_Advantage_N")
        
        # 4. SpecialReady_N: Vera se gli SP sono al massimo OPPURE se nel ramo abbiamo caricato SP / attivato Buff
        if p.sp >= p.stats["sp_threshold"]: props.add("SpecialReady_P")
        elif p.sp >= p.stats["sp_threshold"] - 2: props.add("SpecialDanger_P")
        
        sp_gained = n.sp > rn.sp
        buff_gained = (n.def_buff_turns > rn.def_buff_turns) or (n.atk_buff_turns > rn.atk_buff_turns)
        if n.sp >= n.stats["sp_threshold"] or (self.depth > 0 and (sp_gained or buff_gained)):
            props.add("SpecialReady_N")
        elif n.sp >= n.stats["sp_threshold"] - 2:
            props.add("SpecialDanger_N")
        
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


def get_state_signature(state, current_depth=0):
    s = state.state if hasattr(state, "state") else state
    p1, p2 = s.player, s.npc
    
    return (
        current_depth,
        discretize_hp(p1.hp, p1.stats["max_hp"]),
        discretize_special(p1.sp, p1.stats["sp_threshold"]),
        1 if p1.atk_buff_turns > 0 else 0,
        1 if p1.def_buff_turns > 0 else 0,
        discretize_hp(p2.hp, p2.stats["max_hp"]),
        discretize_special(p2.sp, p2.stats["sp_threshold"]),
        1 if p2.atk_buff_turns > 0 else 0,
        1 if p2.def_buff_turns > 0 else 0
    )


def build_horizon_tree(current_state, user_predictor, max_depth=3, cur_depth=0, current_prob=1.0, prune_threshold=0.15, memo=None, use_dag=True, last_p_act=None, last_n_act=None, root_state=None):
    if root_state is None:
        root_state = current_state
        
    if use_dag and memo is None:
        memo = {}
        
    COOLDOWN_ACTIONS = [Action.COUNTER]
    p_cd_locked = (last_p_act in COOLDOWN_ACTIONS)
    n_cd_locked = (last_n_act in COOLDOWN_ACTIONS)
    
    node = StateTreeNode(current_state, depth=cur_depth, parent_prob=current_prob, root_state=root_state)
    
    if use_dag:
        signature = get_state_signature(current_state, cur_depth)
        memo_key = (signature, p_cd_locked, n_cd_locked, frozenset(node.atomic_props))
        if memo_key in memo:
            return memo[memo_key]
        memo[memo_key] = node
        
    if cur_depth >= max_depth or current_state.is_finished():
        return node

    p_dist = user_predictor.get_distribution(current_state.player)
    
    adjusted_p_dist = {}
    p_has_buff = (not current_state.is_buff_allowed(current_state.player)) or (
        current_state.player.char_class == "Mage" and current_state.player.sp >= current_state.player.stats["sp_threshold"]
    )
    
    for a_p, prob in p_dist.items():
        if p_cd_locked and a_p == last_p_act:
            continue
        if p_has_buff and a_p == Action.BUFF:
            continue
        if a_p == last_p_act:
            prob *= 0.70 
            
        adjusted_p_dist[a_p] = adjusted_p_dist.get(a_p, 0.0) + prob
        
    if current_state.player.sp >= current_state.player.stats["sp_threshold"]:
        attack_prob = adjusted_p_dist.pop(Action.ATTACK, 0.0)
        special_prob = adjusted_p_dist.get(Action.SPECIAL, 0.0)
        adjusted_p_dist[Action.SPECIAL] = (attack_prob + special_prob) * 1.5

    total_prob = sum(adjusted_p_dist.values())
    if total_prob > 0:
        adjusted_p_dist = {k: v / total_prob for k, v in adjusted_p_dist.items()}

    npc_actions = []
    if current_state.npc.sp >= current_state.npc.stats["sp_threshold"]:
        npc_actions.append(Action.SPECIAL)
    else:
        npc_actions.append(Action.ATTACK)
    npc_actions.extend([Action.COUNTER, Action.BUFF, Action.DEFEND])
        
    if n_cd_locked and last_n_act in npc_actions:
        npc_actions.remove(last_n_act)

    n_buff_useless = (not current_state.is_buff_allowed(current_state.npc)) or (
                current_state.npc.char_class == "Mage" and current_state.npc.sp >= current_state.npc.stats["sp_threshold"]
            )
    
    if (n_buff_useless or not current_state.is_buff_allowed(current_state.npc)) and Action.BUFF in npc_actions:
        npc_actions.remove(Action.BUFF)
            
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
                last_p_act=a_p,
                last_n_act=a_n,
                root_state=root_state
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
            matrix[i][i] = "1"
        else:
            for ap_action, npc_action, _, child in node.transitions:
                j = nodes.index(child)
                matrix[i][j] = "1"

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

    root_diff = root_node.state.npc.hp - root_node.state.player.hp
    action_scores = {}
    
    for p_act, n_act, prob, child in root_node.transitions:
        if n_act not in action_scores:
            action_scores[n_act] = 0.0
        
        if child in safe_nodes:
            child_diff = child.state.npc.hp - child.state.player.hp
            dmg_dealt = root_node.state.player.hp - child.state.player.hp
            progress_bonus = 0.002 * dmg_dealt + 0.001 * (child_diff - root_diff)
            action_scores[n_act] += prob * (1.0 + progress_bonus)

    if not action_scores or max(action_scores.values()) <= 0.0:
        return Action.DEFEND, set()
        
    best_action = max(action_scores.items(), key=lambda x: x[1])[0]
    return best_action, safe_nodes