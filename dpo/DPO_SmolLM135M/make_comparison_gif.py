"""
Animated SFT-vs-DPO typing GIF for the same prompt.

Reads sft_vs_dpo.json ([{prompt, sft, dpo}, ...]) and renders a two-column
typing animation: same prompt at the top, SFT (left) and DPO (right) generating
their responses side by side.

Requires: pillow  (uv run python make_comparison_gif.py)
"""
import json
from PIL import Image, ImageDraw, ImageFont

INPUT = "sft_vs_dpo.json"
OUTPUT = "sft_vs_dpo.gif"

W, H = 1200, 340
COLS = 2
PAD = 18
COL_W = (W - PAD * (COLS + 1)) // COLS
HEADER_H = 92
FRAME_MS = 40
HOLD_FRAMES = 45
TYPE_STEPS = 60

BG = (15, 15, 20)
DIVIDER = (45, 45, 60)
PROMPT_BG = (25, 25, 35)
PROMPT_COLOR = (220, 200, 100)
TEXT_COLOR = (215, 215, 215)

MODEL_NAMES = ["SFT baseline", "DPO  (this model)"]
MODEL_KEYS = ["sft", "dpo"]
MODEL_COLORS = [(255, 150, 110), (100, 220, 140)]  # orange, green

FONT_PATHS = [
    "C:/Windows/Fonts/consola.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
FONT_PATHS_BOLD = [
    "C:/Windows/Fonts/consolab.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
]


def load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


font_body = load_font(FONT_PATHS, 16)
font_prompt = load_font(FONT_PATHS, 16)
font_hdr = load_font(FONT_PATHS_BOLD, 18)
font_lbl = load_font(FONT_PATHS_BOLD, 15)


def wrap(text, font, max_w, draw):
    lines, cur = [], []
    for w in text.split():
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


def col_x(c):
    return PAD + c * (COL_W + PAD)


def draw_block(draw, x, y, text, font, color, max_w, max_y, line_h):
    """Render text that may contain newlines; wrap each paragraph."""
    for para in text.split("\n"):
        for line in wrap(para, font, max_w, draw):
            if y + line_h > max_y:
                return y
            draw.text((x, y), line, font=font, fill=color)
            y += line_h
    return y


def draw_frame(prompt, gens, cursor):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Prompt header
    draw.rectangle([(0, 0), (W, HEADER_H)], fill=PROMPT_BG)
    draw.text((PAD, 10), "Prompt", font=font_lbl, fill=PROMPT_COLOR)
    y = 32
    for line in wrap(prompt, font_prompt, W - 2 * PAD, draw)[:2]:
        draw.text((PAD, y), line, font=font_prompt, fill=PROMPT_COLOR)
        y += 22
    draw.line([(0, HEADER_H), (W, HEADER_H)], fill=DIVIDER, width=1)

    # Column divider
    xd = col_x(1) - PAD // 2
    draw.line([(xd, HEADER_H), (xd, H)], fill=DIVIDER, width=1)

    for c, (name, key, color) in enumerate(zip(MODEL_NAMES, MODEL_KEYS, MODEL_COLORS)):
        x = col_x(c)
        draw.text((x, HEADER_H + 12), name, font=font_hdr, fill=color)
        draw.line([(x, HEADER_H + 38), (x + COL_W, HEADER_H + 38)], fill=color, width=1)
        text = gens[key] + ("█" if cursor else " ")
        draw_block(draw, x, HEADER_H + 52, text, font_body, TEXT_COLOR,
                   COL_W - 6, H - PAD, 24)
    return img


def main():
    scenes = json.load(open(INPUT, encoding="utf-8"))
    frames = []
    for sc in scenes:
        full = {k: sc[k] for k in MODEL_KEYS}
        max_len = max(len(v) for v in full.values())
        step = max(1, max_len // TYPE_STEPS)
        for i in range(0, max_len + step, step):
            frames.append(draw_frame(sc["prompt"], {k: v[:i] for k, v in full.items()}, True))
        for b in range(6):
            frames.append(draw_frame(sc["prompt"], full, b % 2 == 0))
        hold = draw_frame(sc["prompt"], full, False)
        frames += [hold] * HOLD_FRAMES

    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=True)
    print(f"Saved {OUTPUT}  ({len(frames)} frames, {len(frames)*FRAME_MS/1000:.1f}s)")


if __name__ == "__main__":
    main()
