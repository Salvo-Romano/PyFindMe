import os
import random
import pygame

class BackgroundManager:
    def __init__(self, base_dir, screen_size):
        self.base_dir = base_dir
        self.screen_w, self.screen_h = screen_size
        
        # Ordine rigoroso di sovrapposizione dal retro al primo piano
        self.layer_folders = ["Sky", "Ground", "Trees", "Foreground"]
        self.layer_files = {layer: [] for layer in self.layer_folders}
        self.current_surface = None
        
        self._scan_layers()

    def _scan_layers(self):
        for layer in self.layer_folders:
            folder_path = os.path.join(self.base_dir, layer)
            if os.path.exists(folder_path):
                files = [
                    os.path.join(folder_path, f)
                    for f in os.listdir(folder_path)
                    if f.lower().endswith(".png")
                ]
                self.layer_files[layer] = files
            else:
                print(f"Warning: Cartella non trovata: {folder_path}")

    def generate_random_scene(self):
        surf = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        
        for layer in self.layer_folders:
            available = self.layer_files[layer]
            if available:
                chosen_file = random.choice(available)
                raw_img = pygame.image.load(chosen_file).convert_alpha()
                scaled_img = pygame.transform.scale(raw_img, (self.screen_w, self.screen_h))
                surf.blit(scaled_img, (0, 0))

        # Converte a RGB normale per ottimizzare il blit finale su schermo
        self.current_surface = surf.convert()

    def draw(self, target_surface):
        if self.current_surface:
            target_surface.blit(self.current_surface, (0, 0))