"""
Animated DPO-vs-RL typing GIF on prompts where RL beats DPO, sized for a social post.

Reads dpo_vs_rl.json ({scenes: [{prompt, dpo, rl_think, rl, dpo_verdict,
rl_verdict, dpo_ok, rl_ok}, ...]}) and renders two columns, including the RL model's
reasoning. A verdict appears under each answer when typing ends. The scenes are picked
from the prompts where RL wins.

Usage: uv run python make_dpo_vs_rl_gif.py
"""
import json
from PIL import Image, ImageDraw, ImageFont

from make_comparison_gif import FONT_PATH, FONT_INDEX_BOLD, FONT_INDEX_REGULAR, wrap

INPUT = "dpo_vs_rl.json"
OUTPUT = "dpo_vs_rl.gif"

W, H = 1200, 854
PAD = 18
COL_W = (W - PAD * 3) // 2
HEADER_H = 116
VERDICT_H = 72
FRAME_MS = 50
HOLD_FRAMES = 110
TYPE_STEPS = 40
MAX_ANSWER_CHARS = 800
MAX_THINK_CHARS = 240

BG = (15, 15, 20)
DIVIDER = (45, 45, 60)
PANEL_BG = (25, 25, 35)
PROMPT_COLOR = (220, 200, 100)
TEXT_COLOR = (215, 215, 215)
THINK_COLOR = (125, 125, 145)
OK_COLOR = (100, 220, 140)
BAD_COLOR = (255, 120, 110)
DPO_COLOR = (110, 180, 255)
RL_COLOR = (100, 220, 140)

font_body = ImageFont.truetype(FONT_PATH, 20, index=FONT_INDEX_REGULAR)
font_prompt = ImageFont.truetype(FONT_PATH, 19, index=FONT_INDEX_REGULAR)
font_small = ImageFont.truetype(FONT_PATH, 16, index=FONT_INDEX_REGULAR)
font_hdr = ImageFont.truetype(FONT_PATH, 22, index=FONT_INDEX_BOLD)
font_lbl = ImageFont.truetype(FONT_PATH, 18, index=FONT_INDEX_BOLD)


def clip(text, limit):
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def draw_text(draw, x, y, text, font, color, max_y, line_h):
    for para in text.split("\n"):
        for line in wrap(para, font, COL_W - 6, draw):
            if y + line_h > max_y:
                return y
            draw.text((x, y), line, font=font, fill=color)
            y += line_h
    return y


def draw_frame(scene, dpo, think, rl, cursor, show_verdict):
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(0, 0), (W, HEADER_H)], fill=PANEL_BG)
    draw.text((PAD, 10), "Prompt", font=font_lbl, fill=PROMPT_COLOR)
    y = 36
    for line in wrap(scene["prompt"], font_prompt, W - 2 * PAD, draw)[:2]:
        draw.text((PAD, y), line, font=font_prompt, fill=PROMPT_COLOR)
        y += 28
    draw.line([(0, HEADER_H), (W, HEADER_H)], fill=DIVIDER, width=1)

    body_bottom = H - VERDICT_H
    columns = [
        (PAD, "DPO", DPO_COLOR, "dpo_verdict", "dpo_ok"),
        (PAD * 2 + COL_W, "RL  (this model)", RL_COLOR, "rl_verdict", "rl_ok"),
    ]
    draw.line([(PAD + COL_W + PAD // 2, HEADER_H), (PAD + COL_W + PAD // 2, H)], fill=DIVIDER, width=1)
    for x, name, color, verdict_key, ok_key in columns:
        draw.text((x, HEADER_H + 12), name, font=font_hdr, fill=color)
        draw.line([(x, HEADER_H + 38), (x + COL_W, HEADER_H + 38)], fill=color, width=1)
        if show_verdict:
            verdict_color = OK_COLOR if scene[ok_key] else BAD_COLOR
            draw_text(draw, x, body_bottom + 10, scene[verdict_key], font_lbl, verdict_color, H, 26)

    mark = "█" if cursor else " "
    draw_text(draw, PAD, HEADER_H + 50, dpo + mark, font_body, TEXT_COLOR, body_bottom, 27)
    y = HEADER_H + 50
    if think:
        y = draw_text(draw, columns[1][0], y, think, font_small, THINK_COLOR, body_bottom, 22) + 10
    draw_text(draw, columns[1][0], y, rl + mark, font_body, TEXT_COLOR, body_bottom, 27)

    return img


def main():
    data = json.load(open(INPUT, encoding="utf-8"))
    frames = []
    for scene in data["scenes"]:
        dpo = clip(scene["dpo"], scene.get("dpo_max_chars", MAX_ANSWER_CHARS))
        think = clip(scene["rl_think"], MAX_THINK_CHARS)
        rl = clip(scene["rl"], MAX_ANSWER_CHARS)
        # The RL column types its reasoning first, then its answer.
        rl_full = len(think) + len(rl)
        total = max(len(dpo), rl_full, 1)
        step = max(1, total // TYPE_STEPS)
        for i in range(0, total + step, step):
            frames.append(draw_frame(scene, dpo[:i], think[:i], rl[:max(0, i - len(think))], True, False))
        hold = draw_frame(scene, dpo, think, rl, False, True)
        frames += [hold] * HOLD_FRAMES

    frames[0].save(OUTPUT, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=True)
    print(f"Saved {OUTPUT}  ({len(frames)} frames, {len(frames)*FRAME_MS/1000:.1f}s)")


if __name__ == "__main__":
    main()
