from PIL import Image

'''src = Image.open("Buff.png").convert("RGBA")
w, h = src.size
frames = 7
frame_w = w // frames

# Inserisci qui lo spostamento (x, y) per ogni frame
# Esempio: se il frame 3 è scivolato di 2 pixel a destra, scrivi (-2, 0)
offsets = [(0, 0), (0, 0), (-3, 0), (-6, 0), (-9, 0), (-12, 0), (-18, 0)]

fixed_sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))

for i in range(frames):
    box = (i * frame_w, 0, (i + 1) * frame_w, h)
    frame = src.crop(box)
    
    aligned_frame = Image.new("RGBA", (frame_w, h), (0, 0, 0, 0))
    
    dx, dy = offsets[i]
    aligned_frame.paste(frame, (dx, dy))
    
    fixed_sheet.paste(aligned_frame, (i * frame_w, 0))

fixed_sheet.save("Buff_Fixed.png")
print("Spritesheet riallineato salvato con successo.")
'''

import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Offset Alignment Tool - ZOOM 4X")
clock = pygame.time.Clock()

sheet = pygame.image.load("Idle.png").convert_alpha()
w, h = sheet.get_width(), sheet.get_height()
frames = 11
fw = w // frames

ZOOM = 4 

frame_surfs = []
for i in range(frames):
    raw_surf = sheet.subsurface((i * fw, 0, fw, h))
    scaled_surf = pygame.transform.scale(raw_surf, (fw * ZOOM, h * ZOOM))
    frame_surfs.append(scaled_surf)

offsets = [(0, 0) for _ in range(frames)]
current_frame = 1

ghost = frame_surfs[0].copy()
ghost.set_alpha(120)

running = True
while running:
    screen.fill((30, 30, 40))
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            x, y = offsets[current_frame]
            if event.key == pygame.K_d and current_frame < frames - 1:
                current_frame += 1
            elif event.key == pygame.K_a and current_frame > 1:
                current_frame -= 1
            elif event.key == pygame.K_UP:
                offsets[current_frame] = (x, y - 1)
            elif event.key == pygame.K_DOWN:
                offsets[current_frame] = (x, y + 1)
            elif event.key == pygame.K_LEFT:
                offsets[current_frame] = (x - 1, y)
            elif event.key == pygame.K_RIGHT:
                offsets[current_frame] = (x + 1, y)
            elif event.key == pygame.K_RETURN:
                print(f"offsets = {offsets}")
                
    cx, cy = 400, 300
    pygame.draw.line(screen, (50, 150, 50), (cx, 0), (cx, 600), 1)
    pygame.draw.line(screen, (50, 150, 50), (0, cy), (800, cy), 1)

    screen.blit(ghost, (cx - (fw * ZOOM) // 2, cy - (h * ZOOM) // 2))
    
    ox, oy = offsets[current_frame]
    screen.blit(frame_surfs[current_frame], (cx - (fw * ZOOM) // 2 + (ox * ZOOM), cy - (h * ZOOM) // 2 + (oy * ZOOM)))
    
    font = pygame.font.SysFont(None, 24)
    info = font.render(f"Frame {current_frame}/{frames-1} | Offset Reale: {offsets[current_frame]}", True, (255, 255, 255))
    screen.blit(info, (10, 10))
    
    controls = font.render("A/D: Cambia | Frecce: Sposta | INVIO: Stampa array", True, (150, 150, 150))
    screen.blit(controls, (10, 560))
    
    pygame.display.flip()
    clock.tick(30)
pygame.quit()
