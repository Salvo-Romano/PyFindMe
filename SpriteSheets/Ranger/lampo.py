from PIL import Image

src = Image.open("Defend.png").convert("RGBA")
w, h = src.size
num_frames = 5
frame_w = w // num_frames

block_strip = Image.new("RGBA", (w, h), (0, 0, 0, 0))

for i in range(num_frames):
    frame = src.crop((i * frame_w, 0, (i + 1) * frame_w, h))
    
    if i < 1:
        # Preparazione: immutata
        block_strip.paste(frame, (i * frame_w, 0))
    else:
        # Impatto: flash ciano sovrapposto
        flash = Image.new("RGBA", frame.size, (50, 200, 255, 255))
        blended = Image.blend(frame, flash, alpha=0.5) # 50% di intensità del flash
        
        # Ripristina i bordi trasparenti della pixel art
        blended.putalpha(frame.getchannel('A'))
        block_strip.paste(blended, (i * frame_w, 0))

block_strip.save("Block.png")
print("Block.png generato (5 frame) con flash visivo di impatto.")