"""Held-out evaluation: validation perplexity.

Training loss is NOT an outcome — a model can lower it while getting worse on
unseen data. We report perplexity = exp(eval_loss) on the held-out split as the
metric the ablations are actually compared on.
"""
from __future__ import annotations

import math


def perplexity_from_loss(eval_loss: float) -> float:
    return math.exp(eval_loss)


def evaluate_perplexity(trainer) -> dict[str, float]:
    """Run the trainer's eval loop on its eval_dataset and return loss + perplexity."""
    metrics = trainer.evaluate()
    eval_loss = float(metrics["eval_loss"])
    return {"eval_loss": eval_loss, "eval_perplexity": perplexity_from_loss(eval_loss)}
