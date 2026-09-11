import copy
from pyFighters import Action, LogicFightersState

class UserPredictor:
    def __init__(self):
        self.actions = [Action.ATTACK, Action.DEFEND, Action.COUNTER, Action.BUFF]
        # Tabelle di frequenza condizionate dallo stato
        # Contesti: 'neutral', 'special_ready', 'critical_hp'
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
        
        # Filtra azioni disponibili
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
        # transizioni: lista di tuple (a_P, a_N, prob_transizione, child_node)
        self.transitions = []
        self.atomic_props = self._evaluate_atomic_props()

    '''def _evaluate_atomic_props(self):
        p, n = self.state.player, self.state.npc
        props = set()
        
        # --- Sopravvivenza ---
        if p.hp <= 0: props.add("Dead_P")
        if n.hp <= 0: props.add("Dead_N")
        if n.hp > 0: props.add("Alive_N") 
        if p.hp <= int(p.stats["max_hp"] * 0.3): props.add("LowHP_P")
        if n.hp <= int(n.stats["max_hp"] * 0.3): props.add("LowHP_N")
        
        
        # --- Vantaggio ---
        if n.hp > p.hp: props.add("HP_Advantage_N")
        
        # --- Speciali e Minacce ---
        # Player
        if p.sp >= p.stats["sp_threshold"]: props.add("SpecialReady_P")
        elif p.sp >= p.stats["sp_threshold"] - 2: props.add("SpecialDanger_P")
        # NPC
        if n.sp >= n.stats["sp_threshold"]: props.add("SpecialReady_N")
        elif n.sp >= n.stats["sp_threshold"] - 2: props.add("SpecialDanger_N")
        
        # --- Stato dei Buff ---
        if p.def_buff_turns > 0: props.add("DefBuff_P")
        if p.atk_buff_turns > 0: props.add("AtkBuff_P")
        if n.def_buff_turns > 0: props.add("DefBuff_N")
        if n.atk_buff_turns > 0: props.add("AtkBuff_N")
        
        # --- Cooldowns ---
        # Uso getattr per evitare crash nel caso in cui counter_cooldown non fosse stato inizializzato
        if getattr(p, "counter_cooldown", 0) == 0: props.add("CounterReady_P")
        if getattr(n, "counter_cooldown", 0) == 0: props.add("CounterReady_N")
        
        return props'''
        
    def _evaluate_atomic_props(self):
        p, n = self.state.player, self.state.npc
        props = set()
        
        # Modello dummy minimale per testare la pipeline
        if n.hp > 0: 
            props.add("Alive_N")
            
        return props


def clone_state(base_state):
    """Copia profonda dello stato per la simulazione a passi in avanti."""
    new_s = LogicFightersState(base_state.player.char_class, base_state.npc.char_class)
    new_s.turn = base_state.turn
    new_s.player = copy.deepcopy(base_state.player)
    new_s.npc = copy.deepcopy(base_state.npc)
    return new_s

def get_state_signature(state, current_depth):
    p = state.player
    n = state.npc
    
    # La tupla rappresenta l'impronta digitale univoca dello stato
    return (
        current_depth,
        p.hp, p.sp, getattr(p, "atk_buff_turns", 0), getattr(p, "def_buff_turns", 0), getattr(p, "counter_cooldown", 0),
        n.hp, n.sp, getattr(n, "atk_buff_turns", 0), getattr(n, "def_buff_turns", 0), getattr(n, "counter_cooldown", 0)
    )
    
def get_action_code(action):
    # Mappa l'enum Action al singolo carattere
    mapping = {
        Action.ATTACK: "A",
        Action.DEFEND: "D",
        Action.COUNTER: "C",
        Action.BUFF: "B",
        Action.SPECIAL: "S"
    }
    return mapping.get(action, "U")

def format_joint_action(action_p1, action_p2):
    # Unisce i codici. Esempio: ATTACK e DEFEND diventano "AD"
    code_p1 = get_action_code(action_p1)
    code_p2 = get_action_code(action_p2)
    return f"{code_p1}{code_p2}"

def build_horizon_tree(current_state, user_predictor, max_depth=3, cur_depth=0, current_prob=1.0, prune_threshold=0.08, memo=None, use_dag=True):
    # Inizializziamo il dizionario solo se stiamo usando il DAG e siamo alla radice
    if use_dag and memo is None:
        memo = {}
        
    # 1. Se usiamo il DAG, calcoliamo la firma e cerchiamo nel dizionario
    if use_dag:
        signature = get_state_signature(current_state, cur_depth)
        if signature in memo:
            return memo[signature]
            
    # 2. Creazione del nuovo nodo (sia per Tree che per DAG)
    node = StateTreeNode(current_state, depth=cur_depth, parent_prob=current_prob)
    
    # 3. Se usiamo il DAG, salviamo il nodo appena creato per i futuri rami
    if use_dag:
        memo[signature] = node
    
    # Condizione di uscita dalla ricorsione
    if cur_depth >= max_depth or current_state.is_finished():
        return node

    p_dist = user_predictor.get_distribution(current_state.player)
    
    npc_actions = [Action.DEFEND, Action.COUNTER, Action.BUFF]
    if current_state.npc.sp >= current_state.npc.stats["sp_threshold"]:
        npc_actions.append(Action.SPECIAL)
    else:
        npc_actions.append(Action.ATTACK)

    num_npc_actions = len(npc_actions)
    
    for a_p, prob_p in p_dist.items():
        if prob_p < prune_threshold:
            continue
            
        for a_n in npc_actions:
            next_s = clone_state(current_state)
            next_s.apply_action_resolution(a_p, a_n)
            
            joint_prob = prob_p / num_npc_actions
            branch_prob = current_prob * joint_prob
            
            # Passiamo il flag use_dag e il dizionario memo ai figli
            child_node = build_horizon_tree(
                next_s, user_predictor,
                max_depth=max_depth,
                cur_depth=cur_depth + 1,
                current_prob=branch_prob,
                prune_threshold=prune_threshold,
                memo=memo,
                use_dag=use_dag # <--- Propagazione del flag
            )
            node.transitions.append((a_p, a_n, joint_prob, child_node))

    return node


def generate_vitamin_model(root_node):
    # 1. Raccogli tutti i nodi unici nell'albero (Visita in profondità/DFS)
    nodes = []
    def collect_nodes(node):
        if node not in nodes:
            nodes.append(node)
            for _, _, _, child in node.transitions:
                collect_nodes(child)
                
    collect_nodes(root_node)
    
    # 2. Assegna nomi agli stati e raccogli tutte le Proposizioni Atomiche (AP)
    state_names = [f"s{i}" for i in range(len(nodes))]
    all_aps = set()
    for n in nodes:
        all_aps.update(n.atomic_props)
    all_aps = sorted(list(all_aps))
    
    # Fallback di sicurezza: VITAMIN fallisce se non c'è almeno una AP
    if not all_aps:
        all_aps = ["neutral"]
        for n in nodes:
            n.atomic_props.add("neutral")

    # 3. Costruisci la Matrice di Transizione (|S| x |S|)
    matrix = [["0" for _ in nodes] for _ in nodes]
    for i, node in enumerate(nodes):
        if not node.transitions:
            # I model checker CTL richiedono transizioni totali (nessun vicolo cieco)
            # Aggiungiamo un auto-anello (self-loop) sugli stati foglia
            matrix[i][i] = "loop"
        else:
            for ap_action, npc_action, _, child in node.transitions:
                j = nodes.index(child)
                # Inseriamo un'etichetta per l'azione (es. Atk_Def) per debug visivo
                action_label = format_joint_action(ap_action, npc_action)
                matrix[i][j] = action_label

    # 4. Genera il file di testo seguendo rigorosamente la sintassi
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
    # Prendiamo direttamente tutti i nodi considerati validi dal solver
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