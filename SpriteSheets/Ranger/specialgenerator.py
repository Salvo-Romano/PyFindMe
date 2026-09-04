from PIL import Image

src = Image.open("Attack.png").convert("RGBA")
w, h = src.size
frames = 6
frame_w = w // frames

special_sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))
pixels = src.load()
out_pixels = special_sheet.load()

for y in range(h):
    for x in range(w):
        r, g, b, a = pixels[x, y]
        if a > 0:
            current_frame = x // frame_w
            
            # Curva non lineare: il bagliore accelera verso gli ultimi frame
            intensity = (current_frame / (frames - 1)) ** 1.5
            
            luma = (0.299 * r + 0.587 * g + 0.114 * b)
            
            # I pixel chiari diventano verde-bianco, gli scuri verde neon
            if luma > 90:
                target_r, target_g, target_b = 80, 255, 120
            else:
                target_r, target_g, target_b = 0, 180, 30
            
            new_r = int((r * (1 - intensity)) + (target_r * intensity))
            new_g = int((g * (1 - intensity)) + (target_g * intensity))
            new_b = int((b * (1 - intensity)) + (target_b * intensity))
            
            out_pixels[x, y] = (new_r, new_g, new_b, a)
        else:
            out_pixels[x, y] = (0, 0, 0, 0)

special_sheet.save("Special.png")
print("Special.png generato: Caricamento Smeraldo Neon totale.")