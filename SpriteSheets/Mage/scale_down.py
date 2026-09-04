import os
from PIL import Image

# Fattore di degradazione. Se il guerriero era ingrandito di 1.5, 
# riduciamo il mago di un fattore simile per pareggiare i macro-pixel.
degrade_factor = 1.5 

for filename in os.listdir("."):
    if filename.endswith(".png"):
        img = Image.open(filename).convert("RGBA")
        w, h = img.size
        
        # Calcola la risoluzione distrutta
        small_w = max(1, int(w / degrade_factor))
        small_h = max(1, int(h / degrade_factor))
        
        # Downscale per perdere informazione
        small_img = img.resize((small_w, small_h), Image.Resampling.NEAREST)
        
        # Upscale per ritornare alle dimensioni originali con pixel giganti
        pixelated_img = small_img.resize((w, h), Image.Resampling.NEAREST)
        
        pixelated_img.save(filename)
        print(f"Processato e pixelato: {filename}")

print("Tutti gli sprite del Mago sono stati degradati con successo.")