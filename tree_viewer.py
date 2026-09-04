import pygame

ACTION_NAMES = {
    "ATTACK": "Atk", "DEFEND": "Def", "COUNTER": "Ctr", "BUFF": "Buf", "SPECIAL": "Spc"
}

class TreeViewer:
    def __init__(self):
        self.selected_node = None
        self.scroll_y = 0
        self.clickable_rects = []

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Wheel up
                self.scroll_y = min(0, self.scroll_y + 35)
            elif event.button == 5:  # Wheel down
                self.scroll_y -= 35
            elif event.button == 1:
                for rect, node in self.clickable_rects:
                    if rect.collidepoint(event.pos):
                        self.selected_node = node
                        break

    def draw(self, surface, root_node, font_title, font_body, screen_w, screen_h):
        # 1. Sfondo oscurato al 95% per eliminare le interferenze dell'arena
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((12, 15, 22, 245))
        surface.blit(overlay, (0, 0))

        # Intestazione superiore
        pygame.draw.rect(surface, (20, 25, 38), (0, 0, screen_w, 65))
        pygame.draw.line(surface, (50, 65, 90), (0, 65), (screen_w, 65), 2)
        
        t_surf = font_title.render("KRIPKE HORIZON INSPECTOR [CGS EXPLORER]", True, (240, 240, 255))
        surface.blit(t_surf, (30, 18))
        h_surf = font_body.render("Usa la rotellina per scorrere | Click sui nodi per ispezionare | Premi [T] per chiudere", True, (130, 145, 170))
        surface.blit(h_surf, (screen_w - h_surf.get_width() - 30, 22))

        if not root_node:
            return

        if not self.selected_node:
            self.selected_node = root_node

        self.clickable_rects = []
        
        # 2. Area Albero (Sinistra, larghezza 65% schermo)
        tree_w = int(screen_w * 0.65)
        clip_rect = pygame.Rect(20, 75, tree_w, screen_h - 95)
        surface.set_clip(clip_rect)

        # Tracciamento gerarchico
        col_w = 230
        x_depths = [40, 40 + col_w + 50, 40 + (col_w + 50) * 2]
        
        # Posizionamento lineare dei nodi per evitare sovrapposizioni
        y_cursor = 85 + self.scroll_y
        node_h = 55
        
        # Disegno Radice (t)
        root_rect = pygame.Rect(x_depths[0], y_cursor, col_w, node_h)
        self._draw_node(surface, root_rect, root_node, "Root (t)", font_body, is_selected=(self.selected_node == root_node))
        self.clickable_rects.append((root_rect, root_node))
        
        # Disegno Livello 1 e 2
        l1_y = y_cursor
        for ap_1, an_1, prob_1, child_1 in root_node.transitions:
            c1_rect = pygame.Rect(x_depths[1], l1_y, col_w, node_h)
            lbl_1 = f"[{ACTION_NAMES.get(ap_1.name, '?')}, {ACTION_NAMES.get(an_1.name, '?')}] {prob_1*100:.1f}%"
            
            # Linea di collegamento
            pygame.draw.line(surface, (55, 70, 95), (root_rect.right, root_rect.centery), (c1_rect.left, c1_rect.centery), 1)
            self._draw_node(surface, c1_rect, child_1, lbl_1, font_body, is_selected=(self.selected_node == child_1))
            self.clickable_rects.append((c1_rect, child_1))
            
            # Espansione Livello 2 (foglie di questo ramo)
            l2_y = l1_y
            for ap_2, an_2, prob_2, child_2 in child_1.transitions[:3]:  # Mostra le 3 risposte più probabili
                c2_rect = pygame.Rect(x_depths[2], l2_y, col_w, node_h)
                lbl_2 = f"[{ACTION_NAMES.get(ap_2.name, '?')}, {ACTION_NAMES.get(an_2.name, '?')}] {prob_2*100:.1f}%"
                
                pygame.draw.line(surface, (45, 55, 75), (c1_rect.right, c1_rect.centery), (c2_rect.left, c2_rect.centery), 1)
                self._draw_node(surface, c2_rect, child_2, lbl_2, font_body, is_selected=(self.selected_node == child_2))
                self.clickable_rects.append((c2_rect, child_2))
                l2_y += node_h + 10

            l1_y = max(l1_y + node_h + 15, l2_y + 10)

        surface.set_clip(None)

        # 3. Pannello Ispettore (Destra, larghezza 35%)
        panel_x = tree_w + 30
        panel_w = screen_w - panel_x - 30
        panel_rect = pygame.Rect(panel_x, 80, panel_w, screen_h - 100)
        
        pygame.draw.rect(surface, (18, 22, 32), panel_rect, border_radius=10)
        pygame.draw.rect(surface, (60, 75, 100), panel_rect, width=2, border_radius=10)
        
        self._draw_inspector(surface, panel_rect, self.selected_node, font_title, font_body)

    def _draw_node(self, surface, rect, node, label, font, is_selected=False):
        bg = (30, 40, 58) if is_selected else (20, 24, 34)
        border = (255, 215, 0) if is_selected else (50, 65, 85)
        if "Dead_N" in node.atomic_props:
            border = (220, 60, 60)
        elif "Dead_P" in node.atomic_props:
            border = (40, 200, 100)

        pygame.draw.rect(surface, bg, rect, border_radius=6)
        pygame.draw.rect(surface, border, rect, width=2 if is_selected else 1, border_radius=6)
        
        # Etichetta azione
        lbl_s = font.render(label, True, (240, 200, 80) if "Root" not in label else (100, 200, 255))
        surface.blit(lbl_s, (rect.x + 8, rect.y + 6))
        
        # Stato compatto
        p, n = node.state.player, node.state.npc
        st_str = f"P:{p.hp}hp/{p.sp}sp  N:{n.hp}hp/{n.sp}sp"
        st_s = font.render(st_str, True, (180, 190, 205))
        surface.blit(st_s, (rect.x + 8, rect.y + 28))

    def _draw_inspector(self, surface, rect, node, font_h, font_t):
        px, py = rect.x + 20, rect.y + 20
        
        head = font_h.render("STATE DETAILS", True, (240, 240, 255))
        surface.blit(head, (px, py))
        py += 40
        
        p, n = node.state.player, node.state.npc
        lines = [
            f"Depth Level: {node.depth}",
            f"Cumulative Prob: {node.prob * 100:.1f}%",
            "---------------------------",
            f"PLAYER ({p.name}):",
            f"  HP: {p.hp} / {p.stats['max_hp']}",
            f"  SP: {p.sp} / {p.stats['sp_threshold']}",
            "---------------------------",
            f"NPC ({n.name}):",
            f"  HP: {n.hp} / {n.stats['max_hp']}",
            f"  SP: {n.sp} / {n.stats['sp_threshold']}",
            "---------------------------",
            "ATOMIC PROPOSITIONS (AP):",
            f"  {sorted(list(node.atomic_props)) if node.atomic_props else '{ }'}",
            "---------------------------",
            f"Branches out: {len(node.transitions)}"
        ]
        
        for line in lines:
            col = (255, 215, 0) if "PROPOSITIONS" in line else (190, 205, 225)
            s = font_t.render(line, True, col)
            surface.blit(s, (px, py))
            py += 24