"""
Visualize the embedding landscape of generated responses.

Pipeline:
  1. SBERT embed all responses
  2. UMAP (2D, cosine metric) for nonlinear projection
  3. PCA (2D) for linear baseline
  4. Plotly interactive visualization with:
     - color by temperature, correctness, coherence, or task category
     - hover shows truncated response text + scores
     - two side-by-side panels (UMAP vs PCA)

Usage:
    uv run preference_optimization/embedding_viz.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_SBERT = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INPUT = (
    "preference_optimization/evals/"
    "smollm_135M_neuraltxt_v1_diversity_n100_r4_final_nogt.jsonl"
)
DEFAULT_OUT_DIR = "preference_optimization/evals/images"

TEMPS = [0.3, 0.5, 0.7, 1.0]
N_PER_TEMP = 4


# ── Task classification ──────────────────────────────────────────────────────

def classify(question: str) -> str:
    instr = question.split("\n\n")[0].strip().lower()
    if any(k in instr for k in ["bullet points", "as a python list of strings", "python list of the key points"]):
        return "bullets"
    if "json" in instr and ("question answer pairs" in instr or "return answer as a json" in instr or "json array of strings" in instr):
        return "qa_pairs"
    if any(k in instr for k in ["generate a set of questions", "list the important questions answered", "generate one question and its corresponding answer", "generate some important facts"]):
        return "qa_pairs"
    if "answer the user's question given the provided passage" in instr:
        return "qa_answer"
    if "generate a question from this passage" in instr:
        return "single_qa"
    if "generate an important fact" in instr:
        return "single_fact"
    if "extract knowledge graph triplets" in instr or "subject-relation-object triplets" in instr:
        return "triplets"
    if "rephrase" in instr:
        return "rephrase"
    if "continuation" in instr or "continue the passage" in instr or "missing middle" in instr:
        return "continuation"
    if "given the first 20%" in instr or "generate the rest of the passage" in instr:
        return "continuation"
    if "passage" in instr and "question" in instr and "answer" in instr:
        return "qa_answer"
    if "triplets" in instr:
        return "triplets"
    return "other"


# ── Temperature colour map ───────────────────────────────────────────────────

TEMP_COLORS = {0.3: "#2196F3", 0.5: "#4CAF50", 0.7: "#FF9800", 1.0: "#F44336"}

CATEGORY_COLORS = {
    "qa_answer":     "#1f77b4",
    "bullets":       "#ff7f0e",
    "qa_pairs":      "#2ca02c",
    "continuation":  "#d62728",
    "triplets":      "#9467bd",
    "single_qa":    "#8c564b",
    "single_fact":   "#e377c2",
    "rephrase":      "#7f7f7f",
    "other":         "#bcbd22",
}


def _temp_for_resp_idx(idx: int) -> float:
    return TEMPS[idx // N_PER_TEMP]


def _temp_label(t: float) -> str:
    return f"t={t}"


# ── Build data arrays ────────────────────────────────────────────────────────

def load_and_prepare(path: str):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    # Flatten: one row per response
    rows = []
    for r in records:
        rid = r["id"]
        q = r["question"]
        cat = classify(q)
        div = r["diversity"]
        for i, resp in enumerate(r["responses"]):
            s = r["response_scores"][i] if "response_scores" in r else {}
            rows.append({
                "record_id": rid,
                "response_idx": i,
                "temperature": _temp_for_resp_idx(i),
                "category": cat,
                "text": resp,
                "correctness": s.get("correctness", 0),
                "coherence": s.get("coherence", 0),
                "vendi": div["vendi"],
                "ead": div["ead"],
                "sbert_div": div["sbert"],
            })
    return rows


def embed_texts(texts: list[str], model_name: str = DEFAULT_SBERT) -> np.ndarray:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return np.asarray(model.encode(texts, convert_to_numpy=True, show_progress_bar=True))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Embedding landscape visualization.")
    parser.add_argument("--input", "-i", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sbert_model", type=str, default=DEFAULT_SBERT)
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows = load_and_prepare(args.input)
    texts = [r["text"] for r in rows]
    print(f"Loaded {len(rows)} responses from {len(set(r['record_id'] for r in rows))} prompts ...")

    # 1. Embed
    print("Embedding with SBERT ...")
    embeddings = embed_texts(texts, args.sbert_model)

    # 2. UMAP
    print("Running UMAP ...")
    umap = UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        random_state=42,
    )
    coords_umap = umap.fit_transform(embeddings)

    # 3. PCA
    print("Running PCA ...")
    pca = PCA(n_components=2)
    # L2-normalise before PCA for fair comparison
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit_embs = embeddings / np.maximum(norms, 1e-12)
    coords_pca = pca.fit_transform(unit_embs)

    # 4. Build plotly figure ─────────────────────────────────────────────────

    def make_hover(r, idx):
        t = r["text"].replace("\n", " ")
        return (
            f"Prompt #{r['record_id']} | {r['category']}<br>"
            f"Temp: {r['temperature']} | idx: {r['response_idx']}<br>"
            f"Correctness: {r['correctness']} | Coherence: {r['coherence']}<br>"
            f"<b>{t[:150]}{'...' if len(t) > 150 else ''}</b>"
        )

    # ── Panel A: UMAP colored by temperature ────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "UMAP — colored by Temperature",
            "PCA — colored by Temperature",
            "UMAP — colored by Correctness",
            "UMAP — colored by Task Category",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    # UMAP by Temperature
    for t_val in TEMPS:
        mask = [i for i, r in enumerate(rows) if r["temperature"] == t_val]
        fig.add_trace(
            go.Scattergl(
                x=coords_umap[mask, 0],
                y=coords_umap[mask, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=TEMP_COLORS[t_val],
                    opacity=0.55,
                    line=dict(width=0.3, color="white"),
                ),
                name=_temp_label(t_val),
                legendgroup=f"temp_{t_val}",
                hovertext=[make_hover(rows[i], i) for i in mask],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # PCA by Temperature
    for t_val in TEMPS:
        mask = [i for i, r in enumerate(rows) if r["temperature"] == t_val]
        fig.add_trace(
            go.Scattergl(
                x=coords_pca[mask, 0],
                y=coords_pca[mask, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=TEMP_COLORS[t_val],
                    opacity=0.55,
                    line=dict(width=0.3, color="white"),
                ),
                name=_temp_label(t_val),
                legendgroup=f"temp_{t_val}",
                showlegend=False,
                hovertext=[make_hover(rows[i], i) for i in mask],
                hoverinfo="text",
            ),
            row=1,
            col=2,
        )

    # UMAP by Correctness
    corr_vals = np.array([r["correctness"] for r in rows], dtype=float)
    fig.add_trace(
        go.Scattergl(
            x=coords_umap[:, 0],
            y=coords_umap[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=corr_vals,
                colorscale="RdYlGn",
                cmin=1,
                cmax=5,
                opacity=0.65,
                line=dict(width=0.3, color="white"),
                colorbar=dict(
                    title="Correctness",
                    x=0.46,
                    len=0.4,
                    y=0.45,
                ),
            ),
            name="Correctness",
            showlegend=False,
            hovertext=[make_hover(rows[i], i) for i in range(len(rows))],
            hoverinfo="text",
        ),
        row=2,
        col=1,
    )

    # UMAP by Task Category
    for cat in sorted(set(r["category"] for r in rows)):
        mask = [i for i, r in enumerate(rows) if r["category"] == cat]
        fig.add_trace(
            go.Scattergl(
                x=coords_umap[mask, 0],
                y=coords_umap[mask, 1],
                mode="markers",
                marker=dict(
                    size=5,
                    color=CATEGORY_COLORS.get(cat, "#888"),
                    opacity=0.55,
                    line=dict(width=0.3, color="white"),
                ),
                name=cat,
                legendgroup=cat,
                hovertext=[make_hover(rows[i], i) for i in mask],
                hoverinfo="text",
            ),
            row=2,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="Response Embedding Landscape — smollm_135M_neuraltxt_v1<br>"
                  "<sup>1600 responses across 100 prompts × 4 temperatures (0.3–1.0)</sup>",
            x=0.5,
        ),
        width=1400,
        height=1100,
        template="plotly_dark",
        font=dict(size=11),
    )

    fig.update_xaxes(title_text="UMAP-1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP-2", row=1, col=1)
    fig.update_xaxes(title_text="PC-1", row=1, col=2)
    fig.update_yaxes(title_text="PC-2", row=1, col=2)
    fig.update_xaxes(title_text="UMAP-1", row=2, col=1)
    fig.update_yaxes(title_text="UMAP-2", row=2, col=1)
    fig.update_xaxes(title_text="UMAP-1", row=2, col=2)
    fig.update_yaxes(title_text="UMAP-2", row=2, col=2)

    # Save interactive HTML + static PNG
    html_path = os.path.join(args.out_dir, "embedding_landscape.html")
    png_path = os.path.join(args.out_dir, "embedding_landscape.png")
    fig.write_html(html_path)
    print(f"Saved interactive HTML: {html_path}")

    try:
        fig.write_image(png_path, scale=2, engine="kaleido")
        print(f"Saved static PNG: {png_path}")
    except Exception as e:
        print(f"PNG export failed (kaleido): {e}")
        print("Install kaleido if you need static export: uv pip install kaleido")


if __name__ == "__main__":
    main()
