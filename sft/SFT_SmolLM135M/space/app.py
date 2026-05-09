"""
Paper Researcher — terminal-style Gradio demo with parallel streaming.
"""
import os
import re
import json
import ast

from paper_researcher import PaperResearcher
from paper_researcher.tasks import SYSTEM_PROMPT
import paper_researcher.tasks as t
import gradio as gr

MODEL_ID = os.environ.get("MODEL_ID", "JaydeepR/SmolLM-135M-SFT-exp01")
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.4

researcher = PaperResearcher(MODEL_ID)

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"],
).set(
    body_background_fill="#0d1117",
    block_background_fill="#0d1117",
    panel_background_fill="#0d1117",
    input_background_fill="#161b22",
    input_border_color="#30363d",
    input_border_color_focus="#58a6ff",
    block_border_color="#30363d",
    block_border_width="1px",
    block_label_text_color="#58a6ff",
    block_label_text_size="11px",
    body_text_color="#c9d1d9",
    body_text_size="12px",
    button_primary_background_fill="#238636",
    button_primary_background_fill_hover="#2ea043",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#21262d",
    button_secondary_border_color="#30363d",
    button_secondary_text_color="#c9d1d9",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
footer { display: none !important; }
.output-box textarea { color: #ffffff !important; font-size: 14px !important; line-height: 1.6 !important; }
ul[role="listbox"] { background: #161b22 !important; }
ul[role="listbox"] li { color: #e6edf3 !important; background: #161b22 !important; }
ul[role="listbox"] li:hover { background: #21262d !important; }
.messages-box textarea { color: #e3b341 !important; font-size: 11px !important; }
.temp-header p { color: #f78166 !important; font-size: 11px !important; font-weight: 600 !important; letter-spacing: 2px !important; margin: 4px 0 2px 0 !important; }
.gen-btn button { width: 100% !important; letter-spacing: 1px !important; }
.stats-bar p { color: #8b949e !important; font-size: 14px !important; letter-spacing: 1px !important; margin: 2px 0 4px 0 !important; }
.block { padding: 6px !important; }
.gap, .gap-2 { gap: 6px !important; }
.left-col { border-right: 1px solid #21262d !important; }
#right-panel .output-box textarea { height: calc(100vh - 200px) !important; overflow-y: auto !important; resize: none !important; }
"""

MODES = {
    "bullets":      {"desc": "Extract key points as bullets",          "hint": "Paste passage...",                                  "call": lambda r, x, **k: "\n".join(f"- {b}" for b in r.extract_bullets(x, **k))},
    "qa_pairs":     {"desc": "Generate Q&A pairs",                     "hint": "Paste passage...",                                  "call": lambda r, x, **k: "\n\n".join(f"Q: {p.question}\nA: {p.answer}" for p in r.generate_qa_pairs(x, **k))},
    "question":     {"desc": "Generate a question from passage",       "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.generate_question(x, **k)},
    "fact":         {"desc": "Extract a single fact",                  "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.extract_fact(x, **k)},
    "answer":       {"desc": "Answer question given passage",          "hint": "passage...\n\n[blank line]\n\nquestion",            "call": lambda r, x, **k: r.answer(*reversed(_split2(x)), **k)},
    "rephrase":     {"desc": "Rephrase and elaborate",                 "hint": "Paste passage...",                                  "call": lambda r, x, **k: r.rephrase(x, **k)},
    "continuation": {"desc": "Continue passage from beginning",        "hint": "Paste start of passage...",                        "call": lambda r, x, **k: r.continue_from(x, **k)},
    "triplets":     {"desc": "Extract knowledge graph triplets",       "hint": "Paste passage...",                                  "call": lambda r, x, **k: "\n".join(f"({tri.subject}, {tri.relation}, {tri.object})" for tri in r.extract_triplets(x, **k))},
    "comparison":   {"desc": "Compare two passages",                   "hint": "passage 1...\n\n[blank line]\n\npassage 2",        "call": lambda r, x, **k: r.compare(*_split2(x), **k)},
    "retrieval":    {"desc": "Find which passage answers a question",  "hint": "Passage 1: ...\n\nPassage 2: ...\n\nQuestion: ...", "call": lambda r, x, **k: _fmt_retrieval(r.find_relevant(*_parse_retrieval(x), **k))},
}

MODE_KEYS = list(MODES.keys())
MODE_CHOICES = [f"{k}  —  {MODES[k]['desc']}" for k in MODE_KEYS]

INSTRUCTION_MAP = {
    "bullets": t.BULLETS_INSTRUCTION, "qa_pairs": t.QA_PAIRS_INSTRUCTION,
    "question": t.QUESTION_FROM_PASSAGE_INSTRUCTION, "fact": t.FACT_FROM_PASSAGE_INSTRUCTION,
    "answer": t.QA_ANSWER_INSTRUCTION, "rephrase": t.REPHRASE_INSTRUCTION,
    "continuation": t.CONTINUATION_INSTRUCTION, "triplets": t.TRIPLETS_INSTRUCTION,
    "comparison": t.COMPARISON_INSTRUCTION, "retrieval": t.RETRIEVAL_INSTRUCTION,
}


def _split2(text):
    parts = text.split("\n\n", 1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

def _fmt_retrieval(r):
    idx = f"Passage {r.index + 1}" if r.index is not None else "None"
    return f"Passage: {idx}\n\n{r.reasoning}"

def _parse_retrieval(text):
    passages = re.findall(r"Passage \d+:\s*(.*?)(?=Passage \d+:|Question:|$)", text, re.DOTALL)
    passages = [p.strip() for p in passages if p.strip()]
    q = re.search(r"Question:\s*(.+)", text, re.DOTALL)
    return (q.group(1).strip() if q else ""), passages

def _maybe_prettify(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            val = ast.literal_eval(stripped)
            if isinstance(val, list):
                return "\n\n".join(str(item) for item in val)
        except (ValueError, SyntaxError):
            pass
    try:
        return json.dumps(json.loads(stripped), indent=2)
    except (json.JSONDecodeError, ValueError):
        pass
    return text

def key_from_choice(c): return c.split("  —  ")[0].strip()

def on_mode_change(choice):
    key = key_from_choice(choice)
    return gr.update(placeholder=MODES[key]["hint"])

def build_messages_preview(choice, text):
    if not text.strip(): return ""
    key = key_from_choice(choice)
    inst = INSTRUCTION_MAP.get(key, "")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{inst}\n\n{text}"}]
    return json.dumps(messages, indent=2)

def build_prompt(mode_key, user_text):
    inst = INSTRUCTION_MAP[mode_key]
    tok = researcher._backend.tokenizer
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{inst}\n\n{user_text}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(mode_choice, user_text):
    if not user_text.strip():
        return "// no input", ""
    key = key_from_choice(mode_choice)
    try:
        result = MODES[key]["call"](researcher, user_text, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS)
        return _maybe_prettify(str(result)), ""
    except Exception as e:
        return f"// error: {e}", ""


with gr.Blocks(title="paper_researcher", theme=THEME, css=CSS) as demo:
    gr.Markdown(
        f"<span style='color:#58a6ff;font-size:13px;font-weight:600'>// paper_researcher</span>"
        f"&nbsp;&nbsp;<span style='color:#8b949e;font-size:11px'>{MODEL_ID}</span>"
        f"&nbsp;&nbsp;<span style='color:#8b949e;font-size:10px'>⚠ CPU — responses may take 30-60s</span>"
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["left-col"], elem_id="left-panel"):
            mode_dd = gr.Dropdown(choices=MODE_CHOICES, value=MODE_CHOICES[0], label="mode")
            user_box = gr.Textbox(label="input", lines=10, placeholder=MODES[MODE_KEYS[0]]["hint"])
            messages_box = gr.Textbox(label="messages preview", lines=8, interactive=False, elem_classes=["messages-box"])
            gen_btn = gr.Button("▶  GENERATE", variant="primary", elem_classes=["gen-btn"])

        with gr.Column(scale=1, elem_id="right-panel"):
            stats_md = gr.Markdown("", elem_classes=["stats-bar"])
            output_box = gr.Textbox(show_label=False, lines=20, interactive=False, elem_classes=["output-box"])

    mode_dd.change(fn=on_mode_change, inputs=mode_dd, outputs=user_box)
    mode_dd.change(fn=build_messages_preview, inputs=[mode_dd, user_box], outputs=messages_box)
    user_box.change(fn=build_messages_preview, inputs=[mode_dd, user_box], outputs=messages_box)
    gen_btn.click(fn=generate, inputs=[mode_dd, user_box], outputs=[output_box, stats_md])

demo.launch()
