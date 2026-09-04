import pygame
import random
from CTL_bridge import UserPredictor, build_horizon_tree
from tree_viewer import TreeViewer
from pyFighters import Action, LogicFightersState
from animator import build_animator
from ui_elements import draw_button, draw_character_hud
from vfx import draw_pixel_shatter, draw_clash_overlay
from projectiles import ArrowManager
from ai_behavior_tree import BehaviorTreeAI
from background_manager import BackgroundManager

ACTION_COLORS = {
    Action.ATTACK: (220, 60, 60),
    Action.DEFEND: (60, 150, 220),
    Action.COUNTER: (220, 180, 40),
    Action.BUFF: (150, 80, 210),
    Action.SPECIAL: (255, 30, 30),
}

ANIM_STATE_MAP = {
    Action.ATTACK: "attack",
    Action.DEFEND: "defend",
    Action.COUNTER: "counter",
    Action.BUFF: "buff",
    Action.SPECIAL: "special",
}

CLASS_OFFSETS = {
    "Warrior": {"ground_offset": 90, "projectile_y_ratio": 0.5, "bow_x_ratio": 0.5},
    "Mage":    {"ground_offset": 180, "projectile_y_ratio": 0.55, "bow_x_ratio": 0.5},
    "Ranger":  {"ground_offset": 180, "projectile_y_ratio": 0.62, "bow_x_ratio": 0.65},
}


class GameGUI:
    def __init__(self):
        pygame.init()
        self.width, self.height = 1280, 720
        self.fullscreen = False
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("LogicFighters - Neuro-Symbolic Prototype")
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("verdana", 48)
        self.font_sub = pygame.font.SysFont("verdana", 22)
        self.font_small = pygame.font.SysFont("verdana", 18)

        self.selected_player_class = "Warrior"
        self.selected_npc_class = "Mage"
        self.selected_ai = "Behavior Tree"
        
        self.current_scene = "menu"
        self.state = None
        self.last_result = None
        self.player_action = Action.ATTACK
        self.is_paused = False

        self.anim_delays = {"player": {"state": None, "timer": 0}, "npc": {"state": None, "timer": 0}}
        
        self.shake_timer = 0
        self.clash_timer = 0
        self.fx_delay_timer = 0
        self.break_timer = {"player": 0, "npc": 0}
        self.break_duration = 400
        self.break_delay_timer = 0
        
        self.pending_break = None
        self.target_break = None
        self.pending_clash = False
        self.pending_arrows = []

        self.arrow_manager = ArrowManager()
        self.bt_ai = BehaviorTreeAI()
        self.bg_manager = BackgroundManager("Backgrounds", (self.width, self.height))

        # Modello utente e ispezione albero
        self.user_predictor = UserPredictor()
        self.tree_viewer = TreeViewer()
        self.horizon_tree = None
        self.show_tree_overlay = False
        self.update_ui_layout()

    def update_ui_layout(self):
        cx, cy = self.width // 2, self.height // 2
        
        self.arena_layout = {
            "floor_y": 500,
            "player_x": int(self.width * 0.25),
            "npc_x": int(self.width * 0.75),
            "hit_y": 400
        }

        self.menu_buttons = {
            "play": pygame.Rect(cx - 100, cy - 80, 200, 55),
            "options": pygame.Rect(cx - 100, cy, 200, 55),
            "exit": pygame.Rect(cx - 100, cy + 80, 200, 55),
        }
        
        self.play_buttons = {
            "start": pygame.Rect(cx - 120, self.height - 90, 240, 55),
            "back": pygame.Rect(40, 40, 130, 40),
        }
        self.options_buttons = {
            "toggle_fullscreen": pygame.Rect(cx - 150, cy - 27, 300, 55),
            "back": pygame.Rect(cx - 100, cy + 80, 200, 55),
        }
        
        self.pause_buttons = {
            "resume": pygame.Rect(cx - 100, cy - 60, 200, 55),
            "quit_battle": pygame.Rect(cx - 100, cy + 10, 200, 55),
            "quit_game": pygame.Rect(cx - 100, cy + 80, 200, 55),
        }
        
        self.sel_panels = {
            "player": pygame.Rect(cx - 400, cy - 210, 300, 300),
            "npc": pygame.Rect(cx + 100, cy - 210, 300, 300),
            "ai": pygame.Rect(cx - 400, cy + 120, 800, 100)
        }
        
        self.sel_buttons = {
            "p_warrior": pygame.Rect(cx - 350, cy - 130, 200, 45),
            "p_mage":    pygame.Rect(cx - 350, cy - 70,  200, 45),
            "p_ranger":  pygame.Rect(cx - 350, cy - 10,  200, 45),
            
            "n_warrior": pygame.Rect(cx + 150, cy - 130, 200, 45),
            "n_mage":    pygame.Rect(cx + 150, cy - 70,  200, 45),
            "n_ranger":  pygame.Rect(cx + 150, cy - 10,  200, 45),
            
            "ai_bt":     pygame.Rect(cx - 350, cy + 145, 200, 45),
            "ai_logic":  pygame.Rect(cx - 100, cy + 145, 200, 45),
            "ai_neuro":  pygame.Rect(cx + 150, cy + 145, 200, 45),
        }
        
        self.victory_button = pygame.Rect(cx - 120, cy + 60, 240, 50)

    def start_battle(self):
        # Genera una combinazione casuale ad ogni match
        self.bg_manager.generate_random_scene()
        self.pending_arrows.clear()
        self.state = LogicFightersState(self.selected_player_class, self.selected_npc_class)
        self.current_scene = "battle"
        self.last_result = None
        self.player_action = Action.ATTACK
        self.is_paused = False
        self.arrow_manager.arrows.clear()

        self.horizon_tree = build_horizon_tree(self.state, self.user_predictor, max_depth=3)
        
        self.player_animator = build_animator(self.selected_player_class, scale=3.0, facing_right=True)
        self.npc_animator = build_animator(self.selected_npc_class, scale=3.0, facing_right=False)

    def get_character_rect(self, role, base_x, floor_y):
        char_class = self.selected_player_class if role == "player" else self.selected_npc_class
        animator = self.player_animator if role == "player" else self.npc_animator
        offset = CLASS_OFFSETS.get(char_class, {}).get("ground_offset", 0)
        img = animator.get_image()
        if img:
            return img.get_rect(midbottom=(base_x, floor_y + offset))
        return pygame.Rect(base_x - 30, floor_y - 120, 60, 120)

    def get_projectile_spawn_point(self, role, rect):
        char_class = self.selected_player_class if role == "player" else self.selected_npc_class
        conf = CLASS_OFFSETS.get(char_class, {})
        ratio_y = conf.get("projectile_y_ratio", 0.5)
        ratio_x = conf.get("bow_x_ratio", 0.5)
        
        spawn_y = rect.bottom - (rect.height * ratio_y)
        
        if role == "player":
            spawn_x = rect.left + (rect.width * ratio_x)
        else:
            spawn_x = rect.right - (rect.width * ratio_x)
            
        return (spawn_x, spawn_y)

    def resolve_turn(self):
        # 1. Registra l'azione reale dell'utente per aggiornare P(a_P | s)
        self.user_predictor.record_action(self.state.player, self.player_action)

        if self.selected_ai == "Behavior Tree":
            self.state.npc.char_class = self.selected_npc_class
            npc_action = self.bt_ai.decide_action(self.state, npc_role="npc")
        else:
            npc_action = Action.DEFEND
            
        self.last_result = self.state.apply_action_resolution(self.player_action, npc_action)

        # 2. Ricostruisce l'albero di transizione per il monitoraggio a orizzonte finito
        self.horizon_tree = build_horizon_tree(self.state, self.user_predictor, max_depth=3)
        
        p_state = ANIM_STATE_MAP.get(self.player_action, "idle")
        n_state = ANIM_STATE_MAP.get(npc_action, "idle")
        p_delay, n_delay = 0, 0
        self.break_timer = {"player": 0, "npc": 0}
        
        clash_actions = [Action.ATTACK, Action.SPECIAL, Action.COUNTER]
        
        if self.player_action in clash_actions and npc_action in clash_actions:
            if self.player_action != Action.SPECIAL:
                p_state = "clash" if self.player_action == Action.ATTACK else "counter_clash"
            if npc_action != Action.SPECIAL:
                n_state = "clash" if npc_action == Action.ATTACK else "counter_clash"
            self.fx_delay_timer = 320 
            self.pending_clash = True
            self.pending_break = None

        elif self.player_action == Action.COUNTER and npc_action == Action.DEFEND:
            p_delay, n_delay = 0, 190
            n_state = "hit"
            self.fx_delay_timer = 190 
            self.pending_clash = False
            self.pending_break = "npc"
            
        elif self.player_action == Action.DEFEND and npc_action == Action.COUNTER:
            p_delay, n_delay = 190, 0
            p_state = "hit"
            self.fx_delay_timer = 190 
            self.pending_clash = False
            self.pending_break = "player"

        else:
            self.pending_break = None
            if npc_action in clash_actions:
                p_delay = 120
                if self.player_action == Action.DEFEND:
                    p_state = "block"
                elif self.player_action == Action.BUFF:
                    p_state = "buff"
                    self.fx_delay_timer = 120
                    self.pending_clash = True
                elif self.last_result.get('npc_damage', 0) > 0:
                    p_state = "hit"
                    
            if self.player_action in clash_actions:
                n_delay = 120
                if npc_action == Action.DEFEND:
                    n_state = "block"
                elif npc_action == Action.BUFF:
                    n_state = "buff"
                    self.fx_delay_timer = 120
                    self.pending_clash = True
                elif self.last_result.get('player_damage', 0) > 0:
                    n_state = "hit"

        p_rect = self.get_character_rect("player", self.arena_layout["player_x"], self.arena_layout["floor_y"])
        n_rect = self.get_character_rect("npc", self.arena_layout["npc_x"], self.arena_layout["floor_y"])
        p_origin = self.get_projectile_spawn_point("player", p_rect)
        n_origin = self.get_projectile_spawn_point("npc", n_rect)

        # Sostituisci le chiamate dirette spawn_shot con l'accodamento
        arrow_release_delay = 250

        if self.selected_player_class == "Ranger":
            if self.player_action in [Action.ATTACK, Action.COUNTER]:
                self.pending_arrows.append({"timer": arrow_release_delay, "type": "shot", "start": p_origin, "target": n_rect.center})
            elif self.player_action == Action.SPECIAL:
                self.pending_arrows.append({"timer": arrow_release_delay + 150, "type": "rain", "x": n_rect.centerx, "y": self.arena_layout["floor_y"]})

        if self.selected_npc_class == "Ranger":
            if npc_action in [Action.ATTACK, Action.COUNTER]:
                self.pending_arrows.append({"timer": arrow_release_delay, "type": "shot", "start": n_origin, "target": p_rect.center})
            elif npc_action == Action.SPECIAL:
                self.pending_arrows.append({"timer": arrow_release_delay + 150, "type": "rain", "x": p_rect.centerx, "y": self.arena_layout["floor_y"]})

        if p_delay > 0:
            self.anim_delays["player"] = {"state": p_state, "timer": p_delay}
        else:
            self.player_animator.set_state(p_state)

        if n_delay > 0:
            self.anim_delays["npc"] = {"state": n_state, "timer": n_delay}
        else:
            self.npc_animator.set_state(n_state)

        if self.player_action not in self.get_active_buttons():
            self.player_action = Action.ATTACK

    def get_active_buttons(self):
        buttons = {}
        p1 = self.state.player
        primary_action = Action.SPECIAL if p1.sp >= p1.stats["sp_threshold"] else Action.ATTACK
        actions = [primary_action, Action.DEFEND, Action.COUNTER, Action.BUFF]
        
        btn_width, btn_height, spacing = 150, 50, 20
        total_width = (len(actions) * btn_width) + ((len(actions) - 1) * spacing)
        start_x = (self.width // 2) - (total_width // 2)
        y_pos = self.height - 120
        
        for i, action in enumerate(actions):
            x = start_x + (i * (btn_width + spacing))
            buttons[action] = pygame.Rect(x, y_pos, btn_width, btn_height)
        return buttons

    def draw_menu(self):
        self.screen.fill((15, 18, 25))
        title = self.font_main.render("LogicFighters", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.width // 2, self.height // 4)))
        for key, rect in self.menu_buttons.items():
            draw_button(self.screen, self.font_sub, rect, key.capitalize())
        pygame.display.flip()
        
    def draw_options(self):
        self.screen.fill((15, 18, 25))
        title = self.font_main.render("Options", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(self.width // 2, self.height // 4)))
        
        fs_label = "Fullscreen: ON" if self.fullscreen else "Fullscreen: OFF"
        fs_color = (40, 180, 90) if self.fullscreen else (180, 70, 70)
        draw_button(self.screen, self.font_sub, self.options_buttons["toggle_fullscreen"], fs_label, fill=fs_color)
        draw_button(self.screen, self.font_sub, self.options_buttons["back"], "Back", fill=(100, 100, 110))
        pygame.display.flip()

    def handle_options_events(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        
        if self.options_buttons["toggle_fullscreen"].collidepoint(event.pos):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                info = pygame.display.Info()
                self.width, self.height = info.current_w, info.current_h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
            else:
                self.width, self.height = 1280, 720
                self.screen = pygame.display.set_mode((self.width, self.height))
            self.update_ui_layout() 
            
        elif self.options_buttons["back"].collidepoint(event.pos):
            self.current_scene = "menu"

    def draw_character_selection(self):
        self.screen.fill((15, 18, 25))
        draw_button(self.screen, self.font_sub, self.play_buttons["back"], "Back")
        cx, cy = self.width // 2, self.height // 2
        
        title = self.font_main.render("Match Setup", True, (255, 255, 255))
        self.screen.blit(title, title.get_rect(center=(cx, 70)))
        
        for panel in self.sel_panels.values():
            pygame.draw.rect(self.screen, (32, 38, 48), panel, border_radius=8)
            pygame.draw.rect(self.screen, (50, 60, 80), panel, width=1, border_radius=8)
            
        titles = [("Player", cx - 250, cy - 180), ("NPC", cx + 250, cy - 180), ("AI Core", cx, cy + 90)]
        for text, tx, ty in titles:
            lbl = self.font_sub.render(text, True, (180, 190, 200))
            self.screen.blit(lbl, lbl.get_rect(center=(tx, ty)))
            
        btn_config = {
            "p_warrior": ("Warrior", self.selected_player_class == "Warrior"),
            "p_mage":    ("Mage", self.selected_player_class == "Mage"),
            "p_ranger":  ("Ranger", self.selected_player_class == "Ranger"),
            "n_warrior": ("Warrior", self.selected_npc_class == "Warrior"),
            "n_mage":    ("Mage", self.selected_npc_class == "Mage"),
            "n_ranger":  ("Ranger", self.selected_npc_class == "Ranger"),
            "ai_bt":     ("Behavior Tree", self.selected_ai == "Behavior Tree"),
            "ai_logic":  ("Pure Logic", self.selected_ai == "Pure Logic"),
            "ai_neuro":  ("Neuro-Symbolic", self.selected_ai == "Neuro-Symbolic"),
        }
        
        for key, rect in self.sel_buttons.items():
            label, is_sel = btn_config[key]
            draw_button(self.screen, self.font_sub, rect, label, fill=(40, 120, 210) if is_sel else (55, 65, 80), selected=is_sel)

        draw_button(self.screen, self.font_sub, self.play_buttons["start"], "Start Battle", fill=(40, 180, 90))
        pygame.display.flip()

    def handle_selection_events(self, event):
        if event.type != pygame.MOUSEBUTTONDOWN or event.button != 1:
            return
        
        if self.play_buttons["back"].collidepoint(event.pos):
            self.current_scene = "menu"
        elif self.play_buttons["start"].collidepoint(event.pos):
            self.start_battle()

        actions = {
            "p_warrior": lambda: setattr(self, "selected_player_class", "Warrior"),
            "p_mage":    lambda: setattr(self, "selected_player_class", "Mage"),
            "p_ranger":  lambda: setattr(self, "selected_player_class", "Ranger"),
            "n_warrior": lambda: setattr(self, "selected_npc_class", "Warrior"),
            "n_mage":    lambda: setattr(self, "selected_npc_class", "Mage"),
            "n_ranger":  lambda: setattr(self, "selected_npc_class", "Ranger"),
            "ai_bt":     lambda: setattr(self, "selected_ai", "Behavior Tree"),
            "ai_logic":  lambda: setattr(self, "selected_ai", "Pure Logic"),
            "ai_neuro":  lambda: setattr(self, "selected_ai", "Neuro-Symbolic"),
        }
        
        for key, rect in self.sel_buttons.items():
            if rect.collidepoint(event.pos):
                actions[key]()

    def draw_battle(self):
        # Rendering Sfondo
        self.bg_manager.draw(self.screen)

        # HUD Player (Sinistra)
        draw_character_hud(
            self.screen, self.font_sub, self.font_small,
            pos=(60, 40),
            current_hp=self.state.player.hp, max_hp=self.state.player.stats["max_hp"],
            current_sp=self.state.player.sp, max_sp=self.state.player.stats["sp_threshold"],
            title="HERO", subtitle=self.selected_player_class,
            is_right_aligned=False
        )

        # HUD NPC (Destra - a specchio)
        draw_character_hud(
            self.screen, self.font_sub, self.font_small,
            pos=(self.width - 60, 40),
            current_hp=self.state.npc.hp, max_hp=self.state.npc.stats["max_hp"],
            current_sp=self.state.npc.sp, max_sp=self.state.npc.stats["sp_threshold"],
            title="EVIL GUY", subtitle=self.selected_npc_class,
            is_right_aligned=True
        )
        
        # Fascia di contrasto inferiore per i controlli
        bar_h = 130
        bottom_bar = pygame.Surface((self.width, bar_h), pygame.SRCALPHA)
        bottom_bar.fill((10, 12, 18, 200))
        pygame.draw.line(bottom_bar, (40, 50, 70), (0, 0), (self.width, 0), 2)
        self.screen.blit(bottom_bar, (0, self.height - bar_h))
        
        # Bottoni azioni
        for action, rect in self.get_active_buttons().items():
            draw_button(self.screen, self.font_sub, rect, action.name, fill=ACTION_COLORS[action], selected=(self.player_action == action))

        # Info turno con outline nero a 4 direzioni per massima leggibilità
        if self.last_result:
            info_str = f"Turn {self.last_result['turn']-1} | Player: {self.last_result['player_damage']} dmg | NPC: {self.last_result['npc_damage']} dmg"
            center_pos = (self.width // 2, self.height // 2)
            
            # Rendering del bordo nero
            outline_surf = self.font_sub.render(info_str, True, (0, 0, 0))
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                self.screen.blit(outline_surf, outline_surf.get_rect(center=(center_pos[0] + dx, center_pos[1] + dy)))
            
            # Testo bianco pieno sopra
            info_surf = self.font_sub.render(info_str, True, (245, 245, 245))
            self.screen.blit(info_surf, info_surf.get_rect(center=center_pos))
            
        sx, sy = 0, 0
        if self.shake_timer > 0:
            intensity = int((self.shake_timer / 250) * 8)
            sx = random.randint(-intensity, intensity)
            sy = random.randint(-intensity, intensity)

        render_floor_y = self.arena_layout["floor_y"] + sy
        p_base_x = self.arena_layout["player_x"] + sx
        n_base_x = self.arena_layout["npc_x"] + sx

        p_rect = self.get_character_rect("player", p_base_x, render_floor_y)
        p_img = self.player_animator.get_image()
        if p_img:
            self.screen.blit(p_img, p_rect)

        n_rect = self.get_character_rect("npc", n_base_x, render_floor_y)
        n_img = self.npc_animator.get_image()
        if n_img:
            self.screen.blit(n_img, n_rect)

        draw_clash_overlay(self.screen, self.width, self.height, self.clash_timer)
        draw_pixel_shatter(self.screen, (p_base_x, self.arena_layout["hit_y"] + sy), self.break_timer["player"], self.break_duration)
        draw_pixel_shatter(self.screen, (n_base_x, self.arena_layout["hit_y"] + sy), self.break_timer["npc"], self.break_duration)

        self.arrow_manager.draw(self.screen)
        
        if self.state.is_finished():
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((10, 12, 18, 190))
            self.screen.blit(overlay, (0, 0))
            
            cx, cy = self.width // 2, self.height // 2
            banner_rect = pygame.Rect(cx - 250, cy - 150, 500, 300)
            is_player_win = self.state.winner() == "Player"
            
            border_color = (220, 180, 40) if is_player_win else (150, 40, 40)
            pygame.draw.rect(self.screen, (25, 28, 35), banner_rect, border_radius=12)
            pygame.draw.rect(self.screen, border_color, banner_rect, width=3, border_radius=12)
            
            win_text = "VICTORY" if is_player_win else "DEFEAT"
            win_color = (255, 215, 0) if is_player_win else (220, 80, 80)
            msg = self.font_main.render(win_text, True, win_color)
            self.screen.blit(msg, msg.get_rect(center=(cx, cy - 70)))
            
            survivor_hp = self.state.player.hp if is_player_win else self.state.npc.hp
            turns = self.last_result['turn'] if self.last_result else 0
            stats_msg = self.font_sub.render(f"HP Rimanenti: {survivor_hp}  |  Turni Giocati: {turns}", True, (180, 190, 200))
            self.screen.blit(stats_msg, stats_msg.get_rect(center=(cx, cy + 10)))
            
            draw_button(self.screen, self.font_sub, self.victory_button, "Return to Menu", fill=(40, 100, 180))
        else:
            hint = self.font_small.render("Press SPACE to resolve turn (ESC for Pause)", True, (120, 130, 140))
            self.screen.blit(hint, hint.get_rect(center=(self.width // 2, self.height - 40)))
            
        if self.is_paused:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            self.screen.blit(overlay, (0, 0))
            
            pause_title = self.font_main.render("PAUSED", True, (255, 255, 255))
            self.screen.blit(pause_title, pause_title.get_rect(center=(self.width // 2, self.height // 2 - 150)))
            
            draw_button(self.screen, self.font_sub, self.pause_buttons["resume"], "Resume", fill=(40, 150, 255))
            draw_button(self.screen, self.font_sub, self.pause_buttons["quit_battle"], "Quit Battle", fill=(180, 120, 40))
            draw_button(self.screen, self.font_sub, self.pause_buttons["quit_game"], "Quit Game", fill=(180, 40, 40))

        # Disegna il pulsante di ispezione nell'angolo inferiore
        inspect_btn = pygame.Rect(self.width - 200, self.height - 50, 180, 36)
        draw_button(self.screen, self.font_small, inspect_btn, "Inspect Tree [T]", fill=(45, 60, 85))

        # Se attivo, disegna l'albero in sovraimpressione
        if self.show_tree_overlay and self.horizon_tree:
            self.tree_viewer.draw(
                self.screen, self.horizon_tree,
                self.font_sub, self.font_small,
                self.width, self.height
            )

        pygame.display.flip()

    def handle_battle_events(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.is_paused = not self.is_paused
            return

        if self.is_paused:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.pause_buttons["resume"].collidepoint(event.pos):
                    self.is_paused = False
                elif self.pause_buttons["quit_battle"].collidepoint(event.pos):
                    self.is_paused = False
                    self.current_scene = "menu"
                elif self.pause_buttons["quit_game"].collidepoint(event.pos):
                    pygame.quit()
                    raise SystemExit
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and not self.state.is_finished():
            self.resolve_turn()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_t:
            self.show_tree_overlay = not self.show_tree_overlay
            return

        if self.show_tree_overlay:
            self.tree_viewer.handle_event(event)
            if event.type == pygame.KEYDOWN and event.key in [pygame.K_t, pygame.K_ESCAPE]:
                self.show_tree_overlay = False
            return        

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.state.is_finished() and getattr(self, 'victory_button', self.pause_buttons["resume"]).collidepoint(event.pos):
                self.current_scene = "menu"
                return

            for action, rect in self.get_active_buttons().items():
                if rect.collidepoint(event.pos):
                    self.player_action = action
                    break

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(30)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif self.current_scene == "menu":
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.menu_buttons["play"].collidepoint(event.pos):
                            self.current_scene = "selection"
                        elif self.menu_buttons["options"].collidepoint(event.pos):
                            self.current_scene = "options"
                        elif self.menu_buttons["exit"].collidepoint(event.pos):
                            running = False
                elif self.current_scene == "selection":
                    self.handle_selection_events(event)
                elif self.current_scene == "options":
                    self.handle_options_events(event)
                elif self.current_scene == "battle":
                    self.handle_battle_events(event)

            if self.current_scene == "battle" and not self.is_paused:
                self.arrow_manager.update(dt)
                
                for p in self.pending_arrows:
                    p["timer"] -= dt
                    if p["timer"] <= 0:
                        if p["type"] == "shot":
                            self.arrow_manager.spawn_shot(p["start"], p["target"])
                        elif p["type"] == "rain":
                            self.arrow_manager.spawn_rain(p["x"], p["y"])
                self.pending_arrows = [p for p in self.pending_arrows if p["timer"] > 0]
                
                
                if self.fx_delay_timer > 0:
                    self.fx_delay_timer -= dt
                    if self.fx_delay_timer <= 0:
                        self.shake_timer = 250
                        if getattr(self, "pending_clash", False):
                            self.clash_timer = 200
                            self.pending_clash = False
                        if getattr(self, "pending_break", None):
                            self.break_delay_timer = 150
                            self.target_break = self.pending_break
                            self.pending_break = None

                if self.break_delay_timer > 0:
                    self.break_delay_timer -= dt
                    if self.break_delay_timer <= 0 and getattr(self, "target_break", None):
                        self.break_timer[self.target_break] = self.break_duration
                        self.target_break = None

                if self.shake_timer > 0:
                    self.shake_timer -= dt
                if self.clash_timer > 0:
                    self.clash_timer -= dt
                if self.break_timer["player"] > 0:
                    self.break_timer["player"] -= dt
                if self.break_timer["npc"] > 0:
                    self.break_timer["npc"] -= dt

                for key, animator in [("player", self.player_animator), ("npc", self.npc_animator)]:
                    if self.anim_delays[key]["timer"] > 0:
                        self.anim_delays[key]["timer"] -= dt
                        if self.anim_delays[key]["timer"] <= 0:
                            new_state = self.anim_delays[key]["state"]
                            animator.set_state(new_state)
                            if new_state in ["block", "hit"]:
                                self.shake_timer = 150

                self.player_animator.update(dt)
                self.npc_animator.update(dt)
                
                p_done = (self.player_animator.is_finished() or self.player_animator.current_state == "idle") and self.anim_delays["player"]["timer"] <= 0
                n_done = (self.npc_animator.is_finished() or self.npc_animator.current_state == "idle") and self.anim_delays["npc"]["timer"] <= 0
                
                if p_done and n_done:
                    if self.state.player.hp <= 0:
                        self.player_animator.set_state("death")
                    elif self.player_animator.current_state != "death":
                        self.player_animator.set_state("idle")
                        
                    if self.state.npc.hp <= 0:
                        self.npc_animator.set_state("death")
                    elif self.npc_animator.current_state != "death":
                        self.npc_animator.set_state("idle")

            if self.current_scene == "menu":
                self.draw_menu()
            elif self.current_scene == "selection":
                self.draw_character_selection()
            elif self.current_scene == "options":
                self.draw_options()
            elif self.current_scene == "battle":
                self.draw_battle()

        pygame.quit()


if __name__ == "__main__":
    GameGUI().run()