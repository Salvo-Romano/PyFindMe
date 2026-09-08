from pyFighters import Action

class NodeStatus:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"

class BTNode:
    def tick(self, state, npc_role="npc"):
        raise NotImplementedError

class Selector(BTNode):
    def __init__(self, children):
        self.children = children

    def tick(self, state, npc_role="npc"):
        for child in self.children:
            status, action = child.tick(state, npc_role)
            if status == NodeStatus.SUCCESS:
                return NodeStatus.SUCCESS, action
        return NodeStatus.FAILURE, None

class Sequence(BTNode):
    def __init__(self, children):
        self.children = children

    def tick(self, state, npc_role="npc"):
        last_action = None
        for child in self.children:
            status, action = child.tick(state, npc_role)
            if status != NodeStatus.SUCCESS:
                return status, None
            if action is not None:
                last_action = action
        return NodeStatus.SUCCESS, last_action

# --- Condizioni Tattiche Avanzate ---

class CheckSpecialReady(BTNode):
    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        if npc.sp >= npc.stats["sp_threshold"]:
            return NodeStatus.SUCCESS, None
        return NodeStatus.FAILURE, None

class CheckHealthRatio(BTNode):
    def __init__(self, threshold, mode="below"):
        self.threshold = threshold
        self.mode = mode

    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        ratio = npc.hp / npc.stats["max_hp"]
        condition = (ratio <= self.threshold) if self.mode == "below" else (ratio >= self.threshold)
        return NodeStatus.SUCCESS if condition else NodeStatus.FAILURE, None

class CheckOpponentThreat(BTNode):
    def tick(self, state, npc_role="npc"):
        player = state.player if npc_role == "npc" else state.npc
        # Minaccia critica se l'avversario ha la Special pronta
        if player.sp >= player.stats["sp_threshold"]:
            return NodeStatus.SUCCESS, None
        return NodeStatus.FAILURE, None

class CheckSPThreshold(BTNode):
    def __init__(self, sp_val, mode="at_least"):
        self.sp_val = sp_val
        self.mode = mode

    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        if self.mode == "at_least":
            cond = npc.sp >= self.sp_val
        elif self.mode == "less_than":
            cond = npc.sp < self.sp_val
        return NodeStatus.SUCCESS if cond else NodeStatus.FAILURE, None

class CheckNeedsSP(BTNode):
    """Evita lo spam del Buff se si è già vicini alla Special."""
    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        threshold = npc.stats["sp_threshold"]
        # Ha senso buffarsi solo se mancano almeno 2 SP al traguardo
        if npc.sp <= threshold - 2:
            return NodeStatus.SUCCESS, None
        return NodeStatus.FAILURE, None

class CheckCounterAvailable(BTNode):
    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        # Usa getattr per sicurezza, nel caso la variabile non sia ancora inizializzata a monte
        if getattr(npc, "counter_cooldown", 0) == 0:
            return NodeStatus.SUCCESS, None
        return NodeStatus.FAILURE, None

class CheckBuffActive(BTNode):
    def tick(self, state, npc_role="npc"):
        npc = state.npc if npc_role == "npc" else state.player
        # Fallisce se il buff specifico della classe è già attivo
        if npc.char_class == "Warrior" and getattr(npc, "def_buff_turns", 0) > 0:
            return NodeStatus.FAILURE, None
        if npc.char_class == "Ranger" and getattr(npc, "atk_buff_turns", 0) > 0:
            return NodeStatus.FAILURE, None
        return NodeStatus.SUCCESS, None

class ActionLeaf(BTNode):
    def __init__(self, action):
        self.action = action

    def tick(self, state, npc_role="npc"):
        return NodeStatus.SUCCESS, self.action


# --- Costruttori Alberi Ribilanciati ---

def build_warrior_tree():
    return Selector([
        Sequence([
            CheckSpecialReady(),
            ActionLeaf(Action.SPECIAL)
        ]),
        Sequence([
            CheckOpponentThreat(),
            ActionLeaf(Action.DEFEND)
        ]),
        Sequence([
            CheckSPThreshold(1, mode="at_least"),
            CheckHealthRatio(0.40, mode="above"),
            CheckCounterAvailable(),
            ActionLeaf(Action.COUNTER)
        ]),
        Sequence([
            CheckHealthRatio(0.25, mode="below"),
            ActionLeaf(Action.DEFEND)
        ]),
        Sequence([
            CheckSPThreshold(0, mode="at_least"),
            CheckSPThreshold(2, mode="less_than"),
            CheckHealthRatio(0.70, mode="above"),
            CheckBuffActive(),
            ActionLeaf(Action.BUFF)
        ]),
        ActionLeaf(Action.ATTACK)
    ])

def build_mage_tree():
    return Selector([
        Sequence([
            CheckSpecialReady(),
            ActionLeaf(Action.SPECIAL)
        ]),
        Sequence([
            CheckOpponentThreat(),
            ActionLeaf(Action.DEFEND)
        ]),
        Sequence([
            CheckHealthRatio(0.35, mode="below"),
            ActionLeaf(Action.DEFEND)
        ]),
        Sequence([
            CheckSPThreshold(2, mode="at_least"),
            CheckCounterAvailable(),
            ActionLeaf(Action.COUNTER)
        ]),
        Sequence([
            CheckHealthRatio(0.45, mode="above"),
            CheckNeedsSP(),
            ActionLeaf(Action.BUFF)
        ]),
        ActionLeaf(Action.ATTACK)
    ])

def build_ranger_tree():
    return Selector([
        Sequence([
            CheckSpecialReady(),
            ActionLeaf(Action.SPECIAL)
        ]),
        Sequence([
            CheckOpponentThreat(),
            ActionLeaf(Action.DEFEND)
        ]),
        Sequence([
            CheckSPThreshold(2, mode="at_least"),
            CheckCounterAvailable(),
            ActionLeaf(Action.COUNTER)
        ]),
        Sequence([
            CheckHealthRatio(0.50, mode="above"),
            CheckNeedsSP(),
            CheckBuffActive(),
            ActionLeaf(Action.BUFF)
        ]),
        ActionLeaf(Action.ATTACK)
    ])


class BehaviorTreeAI:
    def __init__(self):
        self.trees = {
            "Warrior": build_warrior_tree(),
            "Mage": build_mage_tree(),
            "Ranger": build_ranger_tree()
        }

    def decide_action(self, state, npc_role="npc"):
        char = state.npc if npc_role == "npc" else state.player
        char_class = getattr(char, "char_class", None) or getattr(state, f"selected_{npc_role}_class", "Warrior")
        tree = self.trees.get(char_class, self.trees["Warrior"])
        _, action = tree.tick(state, npc_role)
        return action if action else Action.ATTACK