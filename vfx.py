import pygame
import math

def draw_pixel_shatter(screen, center_pos, timer_val, total_time):
    if timer_val <= 0: return
    progress = 1.0 - (timer_val / total_time)
    alpha = max(0, int(255 * (1 - progress)))
    
    base_size = 40
    tiny_surf = pygame.Surface((base_size, base_size), pygame.SRCALPHA)
    
    r = int(4 + progress * 16)
    cx, cy = base_size // 2, base_size // 2
    
    r_col = int(180 + (75 * progress))  
    g_col = int(50 - (30 * progress))   
    b_col = int(255 - (100 * progress)) 
    
    for angle in range(0, 360, 15):
        rad = math.radians(angle)
        dist = r + (angle % 7) * progress * 4
        x = cx + math.cos(rad) * dist
        y = cy + math.sin(rad) * dist
        
        size = max(1, int(3 - progress * 2))
        pygame.draw.rect(tiny_surf, (r_col, g_col, b_col, alpha), (x, y, size, size))
        
        if progress < 0.5 and angle % 30 == 0:
            inner_x = cx + math.cos(rad) * (dist * 0.4)
            inner_y = cy + math.sin(rad) * (dist * 0.4)
            pygame.draw.rect(tiny_surf, (255, 255, 255, alpha), (inner_x, inner_y, 1, 1))

    final_size = base_size * 8
    pixel_surf = pygame.transform.scale(tiny_surf, (final_size, final_size))
    screen.blit(pixel_surf, pixel_surf.get_rect(center=center_pos))

def draw_clash_overlay(screen, width, height, timer_val):
    if timer_val <= 0: return
    # Aumentato l'alpha massimo per un flash più violento
    alpha = int((timer_val / 200) * 160)
    white_overlay = pygame.Surface((width, height), pygame.SRCALPHA)
    white_overlay.fill((255, 255, 255, alpha))
    screen.blit(white_overlay, (0, 0))