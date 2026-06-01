"""
Diversity metrics for evaluating lists of text responses.

Metrics implemented:
  - EAD (Embedding Average Distance)
  - SBert semantic diversity (mean pairwise cosine distance)
  - Vendi Score (effective number of distinct responses from similarity spectrum)

Example JSONL input format (one line per prompt):
  {"id": 0, "question": "...", "responses": ["...", "...", "..."]}

Usage:
    uv run preference_optimization/diversity.py -i responses.jsonl -o scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# ── Globals ────────────────────────────────────────────────────────────────────

_DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_sbert: SentenceTransformer | None = None


def _get_sbert(model_name: str = _DEFAULT_SBERT_MODEL) -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        # Trust remote code for modern SBERT models; auto-select device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _sbert = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return _sbert


# ── Core metrics ───────────────────────────────────────────────────────────────


def embed(texts: list[str], model_name: str = _DEFAULT_SBERT_MODEL) -> np.ndarray:
    """Return (len(texts), dim) NumPy array of SBERT embeddings."""
    model = _get_sbert(model_name)
    return np.asarray(model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


def calculate_ead(embeddings: np.ndarray) -> float:
    """
    Embedding Average Distance (EAD).

    For each response embedding, compute cosine distance to the mean embedding
    of the set, then return the mean of those distances.
    """
    if embeddings.ndim != 2:
        raise ValueError("EAD expects a 2-D array of embeddings.")
    n = embeddings.shape[0]
    if n == 0:
        return 0.0

    # L2-normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    mean_vec = unit.mean(axis=0)
    mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)

    cosine_sims = unit @ mean_vec
    cosine_sims = np.clip(cosine_sims, -1.0, 1.0)
    distances = 1.0 - cosine_sims
    return float(distances.mean())


def calculate_sbert_diversity(embeddings: np.ndarray) -> float:
    """
    SBert semantic diversity: mean pairwise cosine distance between sentence
    embeddings.
    """
    if embeddings.ndim != 2:
        raise ValueError("SBert diversity expects a 2-D array of embeddings.")
    n = embeddings.shape[0]
    if n <= 1:
        return 0.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    sim_matrix = unit @ unit.T
    sim_matrix = np.clip(sim_matrix, -1.0, 1.0)

    # Upper-triangle (i < j) cosine distances
    triu_idx = np.triu_indices(n, k=1)
    distances = 1.0 - sim_matrix[triu_idx]
    return float(distances.mean())


def calculate_vendi_score(embeddings: np.ndarray) -> float:
    """
    Vendi Score for a set of embeddings.

    Computes the exponential of the Shannon entropy of the eigenvalue
    distribution of the normalised similarity matrix K.

    Reference:
        Friedman & Dieng, "The Vendi Score: A Diversity Evaluation Metric
        for Machine Learning", 2022.
    """
    if embeddings.ndim != 2:
        raise ValueError("Vendi Score expects a 2-D array of embeddings.")
    n = embeddings.shape[0]
    if n == 0:
        return 0.0

    # Cosine similarity matrix K
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, 1e-12)
    K = unit @ unit.T

    # Numerical stabilisation: force symmetry and PSD
    K = (K + K.T) / 2.0
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.maximum(eigvals, 0.0)

    total = eigvals.sum()
    if total < 1e-12:
        return 0.0

    p = eigvals / total
    p = p[p > 1e-12]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def score_diversity(responses: list[str], model_name: str = _DEFAULT_SBERT_MODEL) -> dict[str, float]:
    """
    Compute all three diversity scores for a list of responses.

    Returns {"ead": float, "sbert": float, "vendi": float}
    """
    if len(responses) < 2:
        return {"ead": 0.0, "sbert": 0.0, "vendi": 0.0}

    embeddings = embed(responses, model_name=model_name)
    return {
        "ead": calculate_ead(embeddings),
        "sbert": calculate_sbert_diversity(embeddings),
        "vendi": calculate_vendi_score(embeddings),
    }


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute diversity metrics (EAD, SBert, Vendi) for response sets."
    )
    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        required=True,
        help="Path to JSONL where each record contains 'responses': [str, ...].",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        type=str,
        default=None,
        help="Path to write scored JSONL. Defaults to <input>_diversity.jsonl",
    )
    parser.add_argument(
        "--sbert_model",
        type=str,
        default=_DEFAULT_SBERT_MODEL,
        help="Sentence-Transformer model name for embeddings.",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="Process only the first N records.",
    )
    args = parser.parse_args()

    if args.output_file is None:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_diversity{ext}"

    # Pre-load model once so progress prints are clean
    print(f"Loading SBERT model: {args.sbert_model} ...")
    _get_sbert(args.sbert_model)

    records: list[dict[str, Any]] = []
    with open(args.input_file) as f:
        for line in f:
            records.append(json.loads(line))

    if args.limit:
        records = records[: args.limit]

    total = len(records)
    print(f"Scoring {total} records ...")

    results: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        responses = rec.get("responses", [])
        if not responses or len(responses) < 2:
            scores = {"ead": 0.0, "sbert": 0.0, "vendi": 0.0}
        else:
            scores = score_diversity(responses, model_name=args.sbert_model)

        out = {
            **rec,
            "diversity": scores,
        }
        results.append(out)
        print(
            f"[{idx + 1}/{total}]  ead={scores['ead']:.4f}  "
            f"sbert={scores['sbert']:.4f}  vendi={scores['vendi']:.4f}"
        )

    with open(args.output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary
    print("\n--- summary ---")
    for key in ("ead", "sbert", "vendi"):
        vals = [r["diversity"][key] for r in results]
        print(
            f"  {key:10s}  mean={sum(vals) / len(vals):.4f}  "
            f"min={min(vals):.4f}  max={max(vals):.4f}"
        )
    print(f"\nSaved to {args.output_file}")


if __name__ == "__main__":
    main()
