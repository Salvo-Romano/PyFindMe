import pygame
import os

class SpriteSheet:
    def __init__(self, filename):
        self.sheet = pygame.image.load(filename).convert_alpha()

    def get_frame(self, x, y, width, height, scale=1):
        image = pygame.Surface((width, height), pygame.SRCALPHA)
        image.blit(self.sheet, (0, 0), (x, y, width, height))
        if scale != 1:
            image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        return image

    def load_strip(self, num_frames, scale=1):
        """Taglia automaticamente una striscia orizzontale in frame di uguali dimensioni."""
        sheet_w, sheet_h = self.sheet.get_size()
        frame_w = sheet_w // num_frames
        frame_h = sheet_h
        
        frames = []
        for i in range(num_frames):
            frames.append(self.get_frame(i * frame_w, 0, frame_w, frame_h, scale))
        return frames

class Animation:
    def __init__(self, frames, frame_duration, loop=True):
        self.frames = frames
        self.frame_duration = frame_duration
        self.loop = loop
        self.current_idx = 0
        self.timer = 0
        self.finished = False

    def update(self, dt):
        if self.finished: return

        self.timer += dt
        if self.timer >= self.frame_duration:
            self.timer = 0
            self.current_idx += 1
            
            if self.current_idx >= len(self.frames):
                if self.loop:
                    self.current_idx = 0
                else:
                    self.current_idx = len(self.frames) - 1
                    self.finished = True

    def get_current_image(self):
        return self.frames[self.current_idx]

    def reset(self):
        self.current_idx = 0
        self.timer = 0
        self.finished = False

class FighterAnimator:
    def __init__(self):
        self.animations = {}
        self.current_state = "idle"
        self.is_facing_right = True

    def add_animation(self, state_name, animation):
        self.animations[state_name] = animation

    def set_state(self, state_name):
        if self.current_state != state_name and state_name in self.animations:
            self.current_state = state_name
            self.animations[self.current_state].reset()

    def update(self, dt):
        if self.current_state in self.animations:
            self.animations[self.current_state].update(dt)

    def get_image(self):
        if self.current_state in self.animations:
            img = self.animations[self.current_state].get_current_image()
            if not self.is_facing_right:
                img = pygame.transform.flip(img, True, False)
            return img
        return None
    
    def is_finished(self):
        if self.current_state in self.animations:
            return self.animations[self.current_state].finished
        return True

def build_animator(char_class, scale=3.0, facing_right=True):
    animator = FighterAnimator()
    animator.is_facing_right = facing_right
    
    base_path = os.path.join("SpriteSheets", char_class)
    
    
    if char_class == "Warrior":
        class_scale = scale
        config = {
            "idle":   (os.path.join(base_path, "Idle.png"), 6, True, 80),
            "attack": (os.path.join(base_path, "Attack.png"), 4, False, 80),
            "counter": (os.path.join(base_path, "Counter.png"), 4, False, 80),
            "clash": (os.path.join(base_path, "AttackAttack.png"), 4, False, 80),
            "counter_clash": (os.path.join(base_path, "CounterCounter.png"), 4, False, 80),
            "defend": (os.path.join(base_path, "Defend.png"), 4, False, 120),
            "hit":    (os.path.join(base_path, "Hit.png"), 3, False, 100),
            "death":  (os.path.join(base_path, "Death.png"), 9, False, 150),
            "buff": (os.path.join(base_path, "Buff.png"), 6, False, 120),
            "block":  (os.path.join(base_path, "Block.png"), 4, False, 100),
            "special": (os.path.join(base_path, "Special.png"), 4, False, 140)
        }
    elif char_class == "Mage":
        class_scale = scale * 0.9
        config = {
            "idle":   (os.path.join(base_path, "Idle.png"), 6, True, 80),
            "attack": (os.path.join(base_path, "Attack.png"), 8, False, 80),
            "counter": (os.path.join(base_path, "Counter.png"), 8, False, 80),
            "clash": (os.path.join(base_path, "AttackAttack.png"), 8, False, 80),
            "counter_clash": (os.path.join(base_path, "CounterCounter.png"), 8, False, 80),
            "defend": (os.path.join(base_path, "Defend.png"), 8, False, 120),
            "hit":    (os.path.join(base_path, "Hit.png"), 4, False, 140),
            "death":  (os.path.join(base_path, "Death.png"), 7, False, 150),
            "buff": (os.path.join(base_path, "Buff.png"), 6, False, 80),
            "block":  (os.path.join(base_path, "Block.png"), 8, False, 100),
            "special": (os.path.join(base_path, "Special.png"), 8, False, 120)
        }
    elif char_class == "Ranger":
        class_scale = scale * 1.6
        config = {
            "idle":   (os.path.join(base_path, "Idle.png"), 10, True, 100),
            "attack": (os.path.join(base_path, "Attack.png"), 6, False, 80),
            "hit":    (os.path.join(base_path, "Hit.png"), 3, False, 140),
            "death":  (os.path.join(base_path, "Death.png"), 10, False, 150),
            "clash": (os.path.join(base_path, "AttackAttack.png"), 6, False, 80),
            "counter_clash": (os.path.join(base_path, "CounterCounter.png"), 6, False, 80),
            "defend": (os.path.join(base_path, "Defend.png"), 5, False, 150),
            "block":  (os.path.join(base_path, "Block.png"), 5, False, 140),
            "counter": (os.path.join(base_path, "Counter.png"), 6, False, 80),
            "buff": (os.path.join(base_path, "Buff.png"), 10, False, 100),
            "special": (os.path.join(base_path, "Special.png"), 6, False, 50)
        }
    
    for state, (filename, frames, loop, duration) in config.items():
        if os.path.exists(filename):
            sheet = SpriteSheet(filename)
            frame_images = sheet.load_strip(frames, class_scale)
            animator.add_animation(state, Animation(frame_images, duration, loop))
        else:
            print(f"Warning: {filename} mancante. Animazione '{state}' saltata.")
            
    animator.set_state("idle")
    return animator