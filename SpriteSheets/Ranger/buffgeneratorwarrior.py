from PIL import Image
import math

jump_sheet = Image.open("Jump.png").convert("RGBA")
w, h = jump_sheet.size
frame_w = w // 4

crouch_frame = jump_sheet.crop((0, 0, frame_w, h))

buff_frames = 6
buff_sheet = Image.new("RGBA", (frame_w * buff_frames, h), (0, 0, 0, 0))

pixels = crouch_frame.load()

for i in range(buff_frames):
    temp_frame = Image.new("RGBA", (frame_w, h), (0, 0, 0, 0))
    out_pixels = temp_frame.load()
    
    pulse = math.sin(i / (buff_frames - 1) * math.pi) * 0.65
    
    for y in range(h):
        for x in range(frame_w):
            r, g, b, a = pixels[x, y]
            if a > 0:
                new_r = int((r * (1 - pulse)) + (40 * pulse))
                new_g = int((g * (1 - pulse)) + (150 * pulse))
                new_b = int((b * (1 - pulse)) + (255 * pulse))
                out_pixels[x, y] = (new_r, new_g, new_b, a)
    
    buff_sheet.paste(temp_frame, (i * frame_w, 0))

buff_sheet.save("Buff.png")
print("Buff.png generato: postura difensiva piantata a terra con aura azzurra.")