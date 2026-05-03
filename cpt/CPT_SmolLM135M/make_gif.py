"""
Create an animated GIF showing side-by-side inference from 3 models.
Requires: pip install pillow
"""
import json
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
INPUT  = "comparison_results.json"
OUTPUT = "inference_comparison.gif"

W, H         = 1500, 560      # total canvas size
COLS         = 3
PAD          = 16
COL_W        = (W - PAD * (COLS + 1)) // COLS
HEADER_H     = 80             # prompt header height
FPS_DURATION = 40             # ms per frame
HOLD_FRAMES  = 40             # frames to hold at end of each scene

BG           = (15, 15, 20)
DIVIDER      = (45, 45, 60)
PROMPT_BG    = (25, 25, 35)

MODEL_COLORS = [
    (130, 180, 255),   # base  — blue
    (255, 140, 100),   # worst — orange
    (100, 220, 140),   # best  — green
]
PROMPT_COLOR = (220, 200, 100)
TEXT_COLOR   = (210, 210, 210)
DIM_COLOR    = (100, 100, 100)

MODEL_NAMES = [
    "SmolLM-135M (Base)",
    "SmolLM-135M\nFull Fine-Tuning (bf16)",
    "SmolLM-135M\nCPT LoRA r=32 ✦ best",
]
MODEL_KEYS = ["base_model", "worst_model", "best_model"]

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]
FONT_PATHS_BOLD = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",
]

def load_font(paths, size):
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except:
            pass
    return ImageFont.load_default()

font_sm   = load_font(FONT_PATHS, 12)
font_md   = load_font(FONT_PATHS, 13)
font_bold = load_font(FONT_PATHS_BOLD, 13)
font_hdr  = load_font(FONT_PATHS_BOLD, 14)

# ── Helpers ───────────────────────────────────────────────────────────────────
def wrap(text, font, max_w, draw):
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        if draw.textlength(test, font=font) > max_w and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def col_x(col):
    return PAD + col * (COL_W + PAD)


def draw_frame(prompt, generations, cursor_visible):
    img  = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # ── Prompt header ──────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (W, HEADER_H)], fill=PROMPT_BG)
    draw.text((PAD, 8), "Prompt", font=font_bold, fill=PROMPT_COLOR)
    prompt_short = prompt if len(prompt) <= 160 else prompt[:157] + "…"
    plines = wrap(prompt_short, font_md, W - PAD * 2, draw)
    y = 28
    for line in plines[:2]:
        draw.text((PAD, y), line, font=font_md, fill=PROMPT_COLOR)
        y += 18
    draw.line([(0, HEADER_H), (W, HEADER_H)], fill=DIVIDER, width=1)

    # ── Column dividers ────────────────────────────────────────────────────
    for c in range(1, COLS):
        x = col_x(c) - PAD // 2
        draw.line([(x, HEADER_H), (x, H)], fill=DIVIDER, width=1)

    # ── Model columns ──────────────────────────────────────────────────────
    for c, (name, gen, color) in enumerate(zip(MODEL_NAMES, generations, MODEL_COLORS)):
        x = col_x(c)

        # Model name (may have \n)
        name_lines = name.split("\n")
        ny = HEADER_H + 10
        for nl in name_lines:
            draw.text((x, ny), nl, font=font_hdr, fill=color)
            ny += 17
        draw.line([(x, ny + 2), (x + COL_W, ny + 2)], fill=color, width=1)

        # Generation text
        text = gen + ("█" if cursor_visible else " ")
        gen_lines = wrap(text, font_sm, COL_W - 4, draw)
        ty = ny + 10
        for line in gen_lines:
            if ty + 16 > H - PAD:
                break
            draw.text((x, ty), line, font=font_sm, fill=TEXT_COLOR)
            ty += 16

    return img


# ── Main ──────────────────────────────────────────────────────────────────────
with open(INPUT) as f:
    results = json.load(f)

frames = []

for item in results:
    prompt = item["prompt"]
    full_gens = [item["generations"].get(k, "") for k in MODEL_KEYS]
    max_len   = max(len(g) for g in full_gens)

    # Typing animation — step through characters
    STEPS = 40
    step  = max(1, max_len // STEPS)

    for i in range(0, max_len + step, step):
        partial = [g[:i] for g in full_gens]
        frames.append(draw_frame(prompt, partial, cursor_visible=True))

    # Blink cursor at end
    for b in range(6):
        frames.append(draw_frame(prompt, full_gens, cursor_visible=(b % 2 == 0)))

    # Hold full frame
    full_frame = draw_frame(prompt, full_gens, cursor_visible=False)
    for _ in range(HOLD_FRAMES):
        frames.append(full_frame)

frames[0].save(
    OUTPUT,
    save_all=True,
    append_images=frames[1:],
    duration=FPS_DURATION,
    loop=0,
    optimize=True,
)
print(f"Saved {OUTPUT}  ({len(frames)} frames, {len(frames)*FPS_DURATION/1000:.1f}s)")
