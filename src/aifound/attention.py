"""Scaled dot-product attention from scratch (NumPy), used as the reference for the PyTorch model.

Each token builds a query ("what am I looking for"), a key ("what do I offer") and a value ("what I
pass on"). Scores are query.key / sqrt(d); a causal mask hides future tokens; softmax turns each row
of scores into weights that sum to 1; the output is the weighted mix of values.
"""

from __future__ import annotations

import numpy as np

from .neural import softmax


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> tuple[np.ndarray, np.ndarray]:
    t, d = q.shape
    scores = q @ k.T / np.sqrt(d)
    if causal:
        scores = np.where(np.tril(np.ones((t, t), bool)), scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ v, w


def pronoun_demo() -> dict:
    """Hand-set vectors so the word 'it' asks for an animal. Shows where the weight goes; not trained."""
    words = ["the", "cat", "sat", "because", "it"]
    animal = np.array([1.0, 0.0, 0.0, 0.0])
    keys = np.array([[0, 0, 0, 1], [3, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 0]], float)
    q = np.zeros((5, 4))
    q[4] = animal * 2
    _, w = attention(q, keys, np.eye(5, 4), causal=True)
    return {"tokens": words, "weights_from_it": {wd: round(float(x), 3) for wd, x in zip(words, w[4], strict=True)}}
