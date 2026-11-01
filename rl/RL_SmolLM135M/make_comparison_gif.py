"""
Animated base-vs-SFT-vs-DPO-vs-RL typing GIF for the same prompts.

Reads base_sft_dpo_rl.json ([{prompt, base, sft, dpo, rl}, ...], from
generate_comparison.py) and renders a four-column typing animation. Answers longer
than MAX_CHARS are cut with "..." so the base model's rambling doesn't dominate.

Usage: uv run python make_comparison_gif.py
"""
import json
from PIL import Image, ImageDraw, ImageFont

INPUT = "base_sft_dpo_rl.json"
OUTPUT = "base_sft_dpo_rl.gif"

W, H = 1600, 600
COLS = 4
PAD = 16
COL_W = (W - PAD * (COLS + 1)) // COLS
HEADER_H = 92
FRAME_MS = 50
HOLD_FRAMES = 50
TYPE_STEPS = 40
MAX_CHARS = 620

BG = (15, 15, 20)
DIVIDER = (45, 45, 60)
PROMPT_BG = (25, 25, 35)
PROMPT_COLOR = (220, 200, 100)
TEXT_COLOR = (215, 215, 215)

MODEL_NAMES = ["Base", "SFT", "DPO", "RL  (this model)"]
MODEL_KEYS = ["base", "sft", "dpo", "rl"]
MODEL_COLORS = [(160, 160, 175), (255, 150, 110), (110, 180, 255), (100, 220, 140)]

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
FONT_INDEX_REGULAR = 0
FONT_INDEX_BOLD = 1

font_body = ImageFont.truetype(FONT_PATH, 14, index=FONT_INDEX_REGULAR)
font_prompt = ImageFont.truetype(FONT_PATH, 15, index=FONT_INDEX_REGULAR)
font_hdr = ImageFont.truetype(FONT_PATH, 17, index=FONT_INDEX_BOLD)
font_lbl = ImageFont.truetype(FONT_PATH, 14, index=FONT_INDEX_BOLD)


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

    for c, (name, key, color) in enumerate(zip(MODEL_NAMES, MODEL_KEYS, MODEL_COLORS)):
        x = col_x(c)
        if c:
            xd = x - PAD // 2
            draw.line([(xd, HEADER_H), (xd, H)], fill=DIVIDER, width=1)
        draw.text((x, HEADER_H + 12), name, font=font_hdr, fill=color)
        draw.line([(x, HEADER_H + 38), (x + COL_W, HEADER_H + 38)], fill=color, width=1)
        text = gens[key] + ("█" if cursor else " ")
        draw_block(draw, x, HEADER_H + 50, text, font_body, TEXT_COLOR, COL_W - 6, H - PAD, 21)
    return img


def clip(text):
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS].rstrip() + "..."


def main():
    scenes = json.load(open(INPUT, encoding="utf-8"))
    frames = []
    for sc in scenes:
        full = {k: clip(sc[k]) for k in MODEL_KEYS}
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
