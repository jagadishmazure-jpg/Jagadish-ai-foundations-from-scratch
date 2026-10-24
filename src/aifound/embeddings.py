"""Word embeddings trained from scratch: count co-occurrences, reweight (PPMI) and compress (SVD).

The corpus is generated here from templates in our own words, so the "meaning" the vectors pick up
comes only from which words appear near which. We then check two properties: similar words sit
close together, and a consistent relation (male -> female, here) becomes a consistent direction,
so king - man + woman lands nearest queen.
"""

from __future__ import annotations

import itertools

import numpy as np

from .pii import SafeError

FRUIT = ["mango", "apple", "banana", "papaya"]
ANIMALS = ["cat", "dog", "lion", "tiger"]
PAIRS = [("man", "woman"), ("king", "queen"), ("prince", "princess"), ("boy", "girl"), ("father", "mother"), ("uncle", "aunt")]
_CROWN = [["wore", "the", "crown"], ["ruled", "the", "realm"]]
_YOUNG = [["went", "to", "school"], ["played", "outside"]]
_ADULT = [["went", "to", "work"], ["cooked", "dinner"]]
ROLE = {
    "man": _ADULT,
    "king": _CROWN + _ADULT,
    "prince": _CROWN + _YOUNG,
    "boy": _YOUNG,
    "father": _ADULT + [["raised", "the", "kids"]],
    "uncle": _ADULT + [["visited", "the", "family"]],
}
# Analogy answers are searched among content nouns only (function words are filtered out first).
NOUNS = set(FRUIT + ANIMALS + [w for p in PAIRS for w in p])


def corpus(seed: int = 0, n: int = 6000) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    sents = []
    for _ in range(n):
        r = rng.random()
        if r < 0.2:
            f = rng.choice(FRUIT)
            sents.append([["she", "ate", "a", "ripe", f], ["the", f, "tasted", "sweet"], ["we", "sliced", "the", f, "for", "juice"]][rng.integers(3)])
        elif r < 0.4:
            a = rng.choice(ANIMALS)
            sents.append([["the", a, "slept", "in", "the", "sun"], ["a", a, "chased", "its", "tail"], ["the", a, "has", "sharp", "claws"]][rng.integers(3)])
        else:
            m, f = PAIRS[rng.integers(len(PAIRS))]
            w, g = (m, "he") if rng.random() < 0.5 else (f, "she")
            role = ROLE[m]
            sents.append(["the", w, "said", g, *role[rng.integers(len(role))]])
    return sents


def train(sents: list[list[str]], dim: int = 16, window: int = 3) -> tuple[dict[str, int], np.ndarray]:
    if dim < 1 or window < 1:
        raise SafeError("dim and window must be positive")
    vocab = sorted(set(itertools.chain.from_iterable(sents)))
    idx = {w: i for i, w in enumerate(vocab)}
    c = np.zeros((len(vocab), len(vocab)))
    for s in sents:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i != j:
                    c[idx[w], idx[s[j]]] += 1
    total = c.sum()
    pw = c.sum(1, keepdims=True) / total
    pc = c.sum(0, keepdims=True) / total
    with np.errstate(divide="ignore"):
        pmi = np.log((c / total) / (pw @ pc))
    ppmi = np.maximum(pmi, 0)
    u, s, _ = np.linalg.svd(ppmi)
    vec = u[:, :dim] * np.sqrt(s[:dim])
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    return idx, vec


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def nearest(idx, vec, q: np.ndarray, exclude: set[str], k: int = 1) -> list[str]:
    inv = {i: w for w, i in idx.items()}
    sims = vec @ q / (np.linalg.norm(q) + 1e-12)
    order = [inv[i] for i in np.argsort(-sims) if inv[i] not in exclude and inv[i] in NOUNS]
    return order[:k]


def analogy(idx, vec, a: str, b: str, c: str) -> str:
    """a is to b as c is to ?  (b - a + c)"""
    return nearest(idx, vec, vec[idx[b]] - vec[idx[a]] + vec[idx[c]], {a, b, c})[0]


def demo(seed: int = 0) -> dict:
    sents = corpus(seed)
    small_idx, small_vec = train(sents, dim=8)
    idx, vec = train(sents)
    within = [cos(vec[idx[x]], vec[idx[y]]) for grp in (FRUIT, ANIMALS) for x, y in itertools.combinations(grp, 2)]
    across = [cos(vec[idx[x]], vec[idx[y]]) for x in FRUIT for y in ANIMALS]
    tests = [("man", "king", "woman"), ("man", "boy", "woman"), ("king", "queen", "prince"), ("boy", "girl", "father"), ("man", "uncle", "woman")]
    expected = ["queen", "girl", "princess", "mother", "aunt"]
    got = [analogy(idx, vec, *t) for t in tests]
    return {
        "vocab": len(idx),
        "dims": vec.shape[1],
        "mean_cos_same_group": round(float(np.mean(within)), 3),
        "mean_cos_fruit_vs_animal": round(float(np.mean(across)), 3),
        "analogies": [{"query": f"{b} - {a} + {c}", "expected": e, "got": g} for (a, b, c), e, g in zip(tests, expected, got, strict=True)],
        "analogy_hits": f"{sum(e == g for e, g in zip(expected, got, strict=True))}/{len(tests)}",
        "analogy_hits_at_8_dims": f"{sum(e == analogy(small_idx, small_vec, *t) for t, e in zip(tests, expected, strict=True))}/{len(tests)}",
        "_idx": idx,
        "_vec": vec,
    }
