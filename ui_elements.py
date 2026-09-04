import pygame

def draw_button(screen, font, rect, label, fill=(70, 70, 80), text_color=(255, 255, 255), selected=False):
    outline = (255, 255, 255) if selected else (40, 40, 50)
    pygame.draw.rect(screen, fill, rect, border_radius=6)
    pygame.draw.rect(screen, outline, rect, width=2 if not selected else 3, border_radius=6)
    text = font.render(label, True, text_color)
    screen.blit(text, text.get_rect(center=rect.center))

def draw_text_with_outline(surface, font, text, color, outline_color, pos, align="center"):
    text_surf = font.render(text, True, color)
    outline_surf = font.render(text, True, outline_color)
    rect = text_surf.get_rect()
    setattr(rect, align, pos)
    
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
        surface.blit(outline_surf, (rect.x + dx, rect.y + dy))
    surface.blit(text_surf, rect)

def draw_character_hud(surface, font_title, font_stats, pos, current_hp, max_hp, current_sp, max_sp, title="PLAYER", subtitle="Warrior", is_right_aligned=False):
    # Dimensioni coerenti
    bar_w, bar_h = 240, 18
    diamond_size = 14
    diamond_spacing = 6
    padding = 16

    sp_total_w = max_sp * (diamond_size + diamond_spacing) - diamond_spacing
    content_w = max(bar_w, sp_total_w + 40)
    box_w = content_w + padding * 2
    box_h = 92
    
    x, y = pos
    if is_right_aligned:
        x -= box_w

    # 1. Card arrotondata elegante con bordo rifinito
    panel_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    pygame.draw.rect(panel_surf, (15, 18, 26, 210), (0, 0, box_w, box_h), border_radius=14)
    pygame.draw.rect(panel_surf, (55, 68, 90, 240), (0, 0, box_w, box_h), width=2, border_radius=14)
    surface.blit(panel_surf, (x, y))

    # 2. Header: Titolo ironico a contrasto
    title_color = (100, 200, 255) if not is_right_aligned else (255, 90, 90)
    title_pos = (x + box_w - padding, y + 14) if is_right_aligned else (x + padding, y + 14)
    align_side = "topright" if is_right_aligned else "topleft"
    
    full_title = f"{title} [{subtitle}]"
    draw_text_with_outline(surface, font_title, full_title, title_color, (5, 5, 10), title_pos, align=align_side)

    # 3. Barra HP con bordo metallico e sfumatura
    bar_x = x + (box_w - bar_w - padding) if is_right_aligned else x + padding
    bar_y = y + 42
    bar_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
    
    pygame.draw.rect(surface, (60, 20, 20), bar_rect, border_radius=5)
    
    ratio = max(0.0, min(1.0, current_hp / max_hp)) if max_hp > 0 else 0
    fill_w = int(bar_w * ratio)
    if fill_w > 0:
        fill_rect = pygame.Rect(bar_x, bar_y, fill_w, bar_h)
        col = (46, 204, 113) if ratio > 0.5 else (241, 196, 15) if ratio > 0.25 else (231, 76, 60)
        pygame.draw.rect(surface, col, fill_rect, border_radius=5)
        pygame.draw.rect(surface, (255, 255, 255, 70), (bar_x, bar_y, fill_w, 4), border_radius=2)
        
    pygame.draw.rect(surface, (80, 95, 120), bar_rect, width=2, border_radius=5)
    
    # Valore HP numerico al centro
    hp_text = f"{max(0, current_hp)}/{max_hp}"
    draw_text_with_outline(surface, font_stats, hp_text, (255, 255, 255), (0, 0, 0), bar_rect.center, align="center")

    # 4. Rombi SP perfettamente allineati
    sp_y = y + 68
    lbl_pos = (x + box_w - padding, sp_y) if is_right_aligned else (x + padding, sp_y)
    draw_text_with_outline(surface, font_stats, "SP", (180, 195, 215), (0, 0, 0), lbl_pos, align=align_side)

    diamonds_start_x = (x + box_w - padding - 28 - sp_total_w) if is_right_aligned else (x + padding + 28)
    is_ready = current_sp >= max_sp and max_sp > 0
    
    for i in range(max_sp):
        dx = diamonds_start_x + i * (diamond_size + diamond_spacing)
        cy = sp_y + 8
        half = diamond_size // 2
        pts = [
            (dx, cy),
            (dx + half, cy - half),
            (dx + diamond_size, cy),
            (dx + half, cy + half)
        ]
        
        if i < current_sp:
            f_col = (255, 215, 0) if is_ready else (0, 220, 255)
            b_col = (255, 245, 180) if is_ready else (180, 245, 255)
            pygame.draw.polygon(surface, f_col, pts)
            pygame.draw.polygon(surface, b_col, pts, width=1)
        else:
            pygame.draw.polygon(surface, (25, 30, 40), pts)
            pygame.draw.polygon(surface, (60, 70, 85), pts, width=1)

def draw_hp_bar(surface, font, x, y, current_hp, max_hp, label="Player", width=280, height=22):
    # 1. Nome del combattente con outline per stacco assoluto
    name_pos = (x + 2, y - 14)
    draw_text_with_outline(surface, font, label.upper(), (235, 235, 245), (10, 10, 15), name_pos, align="bottomleft")
    
    # 2. Cornice esterna e fondo ferite
    bar_rect = pygame.Rect(x, y, width, height)
    border_rect = pygame.Rect(x - 2, y - 2, width + 4, height + 4)
    
    pygame.draw.rect(surface, (18, 20, 28), border_rect, border_radius=4)
    pygame.draw.rect(surface, (80, 20, 20), bar_rect, border_radius=3)  # Fondo rosso scuro (danno)

    # 3. Calcolo ampiezza e sfumatura dinamica
    ratio = max(0.0, min(1.0, current_hp / max_hp)) if max_hp > 0 else 0
    fill_width = int(width * ratio)
    
    if fill_width > 0:
        fill_rect = pygame.Rect(x, y, fill_width, height)
        
        # Colore reattivo: Verde brillante -> Giallo/Ambra -> Rosso allarme
        if ratio > 0.55:
            base_col = (46, 204, 113)
            light_col = (115, 230, 160)
        elif ratio > 0.25:
            base_col = (241, 196, 15)
            light_col = (247, 220, 111)
        else:
            base_col = (231, 76, 60)
            light_col = (241, 148, 138)
            
        pygame.draw.rect(surface, base_col, fill_rect, border_radius=3)
        # Riflesso lucido superiore (effetto 3D arcade)
        highlight_rect = pygame.Rect(x, y, fill_width, max(2, height // 3))
        pygame.draw.rect(surface, light_col, highlight_rect, border_radius=2)
        
    # Bordo metallico rifinito
    pygame.draw.rect(surface, (70, 80, 100), border_rect, width=2, border_radius=4)
    
    # 4. Numeri HP dentro la barra al centro
    hp_text = f"{max(0, current_hp)} / {max_hp}"
    draw_text_with_outline(surface, font, hp_text, (255, 255, 255), (0, 0, 0), bar_rect.center, align="center")

def draw_sp_indicators(surface, font, x, y, current_sp, max_sp, block_size=18, spacing=6):
    # Etichetta SP
    draw_text_with_outline(surface, font, "SP", (210, 220, 235), (10, 10, 15), (x, y + 2), align="topleft")
    
    start_x = x + 38
    is_fully_charged = current_sp >= max_sp and max_sp > 0
    
    for i in range(max_sp):
        diamond_x = start_x + (i * (block_size + spacing))
        cy = y + 10
        half = block_size // 2
        
        points = [
            (diamond_x, cy),
            (diamond_x + half, cy - half),
            (diamond_x + block_size, cy),
            (diamond_x + half, cy + half)
        ]
        
        if i < current_sp:
            # Pieno: Oro se pronto alla Special, altrimenti Ciano magico intenso
            fill_col = (255, 215, 0) if is_fully_charged else (0, 210, 255)
            border_col = (255, 245, 180) if is_fully_charged else (180, 245, 255)
            
            pygame.draw.polygon(surface, fill_col, points)
            pygame.draw.polygon(surface, border_col, points, width=1)
            
            # Punto luce interno per dare brillantezza
            core_pts = [
                (diamond_x + 3, cy),
                (diamond_x + half, cy - half + 3),
                (diamond_x + block_size - 3, cy),
                (diamond_x + half, cy + half - 3)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), core_pts)
        else:
            # Vuoto: intaglio scuro con profilo visibile
            pygame.draw.polygon(surface, (20, 24, 32), points)
            pygame.draw.polygon(surface, (55, 65, 80), points, width=1)