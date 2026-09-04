from PIL import Image

src = Image.open("Attack.png").convert("RGBA")
w, h = src.size
frames = 6
frame_w = w // frames

counter_sheet = Image.new("RGBA", (w, h), (0, 0, 0, 0))
pixels = src.load()
out_pixels = counter_sheet.load()

for y in range(h):
    for x in range(w):
        r, g, b, a = pixels[x, y]
        if a > 0:
            current_frame = x // frame_w
            intensity = current_frame / (frames - 1)
            
            # Isola i verdi (tunica) e i marroni (arco/stivali)
            if g > r + 10 and g > b + 10:
                # Da verde a viola brillante
                target_r, target_g, target_b = int(g * 1.4), int(r * 0.5), int(g * 1.8)
            else:
                # Il resto si satura di un riflesso velenoso
                target_r, target_g, target_b = min(255, int(r * 1.2)), int(g * 0.5), min(255, int(b * 1.5))
                
            new_r = int((r * (1 - intensity)) + (target_r * intensity))
            new_g = int((g * (1 - intensity)) + (target_g * intensity))
            new_b = int((b * (1 - intensity)) + (target_b * intensity))
            
            out_pixels[x, y] = (new_r, new_g, new_b, a)
        else:
            out_pixels[x, y] = (0, 0, 0, 0)

counter_sheet.save("Counter.png")
print("Counter.png generato: Tiro infuso al veleno.")