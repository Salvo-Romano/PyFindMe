from enum import Enum

CLASS_STATS = {
    "Warrior": {"max_hp": 70, "base_dmg": 8, "special_dmg": 18, "counter_crit": 14, "counter_weak": 5, "buff_def_val": 4, "sp_threshold": 6},
    "Mage":    {"max_hp": 50, "base_dmg": 4,  "special_dmg": 22, "counter_crit": 10, "counter_weak": 3, "buff_sp_val": 2,  "sp_threshold": 8},
    "Ranger":  {"max_hp": 60, "base_dmg": 6,  "special_dmg": 14, "counter_crit": 12, "counter_weak": 4, "buff_atk_val": 4, "sp_threshold": 6}
}


class Action(Enum):
    ATTACK = 0
    DEFEND = 1
    COUNTER = 2
    BUFF = 3
    SPECIAL = 4


class Fighter:
    def __init__(self, name, char_class):
        self.name = name
        self.char_class = char_class
        self.stats = CLASS_STATS[char_class]

        self.hp = self.stats["max_hp"]
        self.sp = 0
        self.atk_buff_turns = 0
        self.def_buff_turns = 0
        self.counter_cooldown = 0


class LogicFightersState:
    # Nessun default: le classi arrivano dalla UI di Pygame
    def __init__(self, player_class, npc_class):
        self.turn = 1
        self.player = Fighter("Player", player_class)
        self.npc = Fighter("NPC", npc_class)

    @staticmethod
    def _normalize_action(action):
        if isinstance(action, Action):
            return action
        return Action(action)

    def _gain_sp_for_actions(self, action_p1, action_p2):
        p1, p2 = self.player, self.npc

        if action_p1 == Action.ATTACK and p1.sp < p1.stats["sp_threshold"]:
            p1.sp += 2
        if action_p2 == Action.ATTACK and p2.sp < p2.stats["sp_threshold"]:
            p2.sp += 2

        if action_p1 == Action.DEFEND and action_p2 in (Action.ATTACK, Action.SPECIAL) and p1.sp < p1.stats["sp_threshold"]:
            p1.sp += 2
        if action_p2 == Action.DEFEND and action_p1 in (Action.ATTACK, Action.SPECIAL) and p2.sp < p2.stats["sp_threshold"]:
            p2.sp += 2

        if action_p1 == Action.COUNTER and action_p2 == Action.DEFEND and p1.sp < p1.stats["sp_threshold"]:
            p1.sp += 2
        if action_p2 == Action.COUNTER and action_p1 == Action.DEFEND and p2.sp < p2.stats["sp_threshold"]:
            p2.sp += 2

    def _resolve_damage(self, attacker, action, opponent_action):
        if action == Action.SPECIAL and attacker.sp >= attacker.stats["sp_threshold"]:
            damage = attacker.stats["special_dmg"]
            attacker.sp = 0
            return damage

        if action == Action.ATTACK:
            base_damage = attacker.stats["base_dmg"]
            if attacker.atk_buff_turns > 0:
                base_damage += attacker.stats.get("buff_atk_val", 0)
            return base_damage

        if action == Action.COUNTER:
            attacker.counter_cooldown = 1
            return attacker.stats["counter_crit"] if opponent_action == Action.DEFEND else attacker.stats["counter_weak"]

        return 0

    def _apply_defense_and_buff(self, attacker, defender, action, opponent_action, damage):
        if opponent_action == Action.DEFEND and action != Action.COUNTER:
            damage //= 2
        if defender.def_buff_turns > 0:
            damage = max(0, damage - defender.stats.get("buff_def_val", 0))
        return max(0, damage)

    def _activate_buff(self, fighter, action):
        if action != Action.BUFF or fighter.atk_buff_turns > 0 or fighter.def_buff_turns > 0:
            return False

        if fighter.char_class == "Mage" and fighter.sp < fighter.stats["sp_threshold"]:
            fighter.sp += fighter.stats["buff_sp_val"]
            return False

        if fighter.char_class == "Warrior":
            fighter.def_buff_turns = 2
            return True

        if fighter.char_class == "Ranger":
            fighter.atk_buff_turns = 2
            return True

        return False

    def _reduce_buffs(self, fighter):
        fighter.atk_buff_turns = max(0, fighter.atk_buff_turns - 1)
        fighter.def_buff_turns = max(0, fighter.def_buff_turns - 1)


    def _reduce_counter_cooldown(self, fighter, action):
        if action == Action.COUNTER:
            fighter.counter_cooldown = max(0, fighter.counter_cooldown - 1)

    def is_finished(self):
        return self.player.hp <= 0 or self.npc.hp <= 0

    def winner(self):
        if self.player.hp > self.npc.hp:
            return self.player.name
        if self.npc.hp > self.player.hp:
            return self.npc.name
        return "Draw"

    def apply_action_resolution(self, action_p1, action_p2):
        action_p1 = self._normalize_action(action_p1)
        action_p2 = self._normalize_action(action_p2)

        p1, p2 = self.player, self.npc
        dmg_p1_to_p2 = 0
        dmg_p2_to_p1 = 0

        self._gain_sp_for_actions(action_p1, action_p2)

        dmg_p1_to_p2 = self._resolve_damage(p1, action_p1, action_p2)
        dmg_p2_to_p1 = self._resolve_damage(p2, action_p2, action_p1)

    
        new_buff_p1 = self._activate_buff(p1, action_p1)
        new_buff_p2 = self._activate_buff(p2, action_p2)

        dmg_p1_to_p2 = self._apply_defense_and_buff(p1, p2, action_p1, action_p2, dmg_p1_to_p2)
        dmg_p2_to_p1 = self._apply_defense_and_buff(p2, p1, action_p2, action_p1, dmg_p2_to_p1)

        p2.hp = max(0, p2.hp - dmg_p1_to_p2)
        p1.hp = max(0, p1.hp - dmg_p2_to_p1)

        self._reduce_buffs(p1)
        self._reduce_buffs(p2)

        self._reduce_counter_cooldown(p1, action_p1)
        self._reduce_counter_cooldown(p2, action_p2)

        if new_buff_p1:
            if p1.char_class == "Warrior":
                p1.def_buff_turns = 2
            if p1.char_class == "Ranger":
                p1.atk_buff_turns = 2

        if new_buff_p2:
            if p2.char_class == "Warrior":
                p2.def_buff_turns = 2
            if p2.char_class == "Ranger":
                p2.atk_buff_turns = 2

        self.turn += 1

        return {
            "player_damage": dmg_p1_to_p2,
            "npc_damage": dmg_p2_to_p1,
            "player_hp": p1.hp,
            "npc_hp": p2.hp,
            "turn": self.turn,
        }