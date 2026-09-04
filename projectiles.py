import pygame
import math
import random

class Arrow:
    def __init__(self, start_pos, target_pos, speed=1200, is_special=False):
        self.x, self.y = float(start_pos[0]), float(start_pos[1])
        self.target_x, self.target_y = float(target_pos[0]), float(target_pos[1])
        self.speed = speed
        self.is_special = is_special
        self.hit = False
        
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        self.vx = (dx / dist) * self.speed if dist > 0 else 0
        self.vy = (dy / dist) * self.speed if dist > 0 else 0
        
        raw_img = pygame.image.load("SpriteSheets/Ranger/ArrowMove.png").convert_alpha()
        scale = 3.5
        w = int(raw_img.get_width() * scale)
        h = int(raw_img.get_height() * scale)
        scaled_img = pygame.transform.scale(raw_img, (w, h))
        
        if self.is_special:
            tinted = scaled_img.copy()
            tinted.fill((80, 255, 120, 0), special_flags=pygame.BLEND_RGBA_ADD)
            scaled_img = tinted

        angle = math.degrees(math.atan2(-dy, dx))
        self.image = pygame.transform.rotate(scaled_img, angle)
        self.rect = self.image.get_rect(center=(int(self.x), int(self.y)))

    def update(self, dt_sec):
        if self.hit:
            return
        
        self.x += self.vx * dt_sec
        self.y += self.vy * dt_sec
        self.rect.center = (int(self.x), int(self.y))
        
        if (self.vx > 0 and self.x >= self.target_x) or (self.vx < 0 and self.x <= self.target_x):
            self.hit = True
        elif self.vy > 0 and self.y >= self.target_y:
            self.hit = True

    def draw(self, surface):
        if not self.hit:
            surface.blit(self.image, self.rect)


class ArrowManager:
    def __init__(self):
        self.arrows = []

    def spawn_shot(self, shooter_pos, target_pos):
        self.arrows.append(Arrow(shooter_pos, target_pos, speed=1600, is_special=False))

    def spawn_rain(self, target_center_x, target_floor_y, count=10):
        for _ in range(count):
            offset_x = random.randint(-60, 60)
            fall_target = (target_center_x + offset_x, target_floor_y - random.randint(10, 60))
            sky_start = (target_center_x + offset_x - random.randint(50, 150), -random.randint(40, 200))
            self.arrows.append(Arrow(sky_start, fall_target, speed=random.randint(1400, 1900), is_special=True))

    def update(self, dt_ms):
        dt_sec = dt_ms / 1000.0
        for arrow in self.arrows:
            arrow.update(dt_sec)
        self.arrows = [a for a in self.arrows if not a.hit]

    def draw(self, surface):
        for arrow in self.arrows:
            arrow.draw(surface)