from PIL import Image

src = Image.open("Attack.png").convert("RGBA")
w, h = src.size
num_frames = 6
frame_w = w // num_frames

clash_strip = Image.new("RGBA", (w, h), (0, 0, 0, 0))

for i in range(num_frames):
    frame = src.crop((i * frame_w, 0, (i + 1) * frame_w, h))
    
    # L'impatto dell'attacco avviene solitamente nei frame centrali (es. 4, 5, 6)
    if 2 <= i <= 4:
        # Sovrapponiamo un flash rosso scuro per evidenziare il clash
        flash = Image.new("RGBA", frame.size, (220, 20, 40, 255))
        blended = Image.blend(frame, flash, alpha=0.6)
        
        # Ripristina i bordi trasparenti
        blended.putalpha(frame.getchannel('A'))
        clash_strip.paste(blended, (i * frame_w, 0))
    else:
        clash_strip.paste(frame, (i * frame_w, 0))

clash_strip.save("AttackAttack.png")
print("AttackAttack.png generato con successo (6 frame).")