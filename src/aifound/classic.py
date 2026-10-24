"""Classic machine learning from scratch with NumPy only.

* rules_vs_learned: hand-written spam rules vs a logistic model trained on labelled messages.
* gradient_descent: fit house prices with plain gradient descent and sweep the step size.
* fit_sweep: polynomial degree sweep that shows underfitting, a sweet spot and overfitting.
* kmeans: unsupervised grouping of unlabelled laundry items.
* learning_curve: how held-out accuracy moves with the amount of training data.

All data is synthetic and generated here from a fixed seed.
"""

from __future__ import annotations

import numpy as np

from .pii import SafeError

# ---------------------------------------------------------------- spam: rules vs learned
SPAM_WORDS = ["prize", "winner", "free", "claim", "urgent", "cash", "offer", "bonus", "gift", "click"]
HAM_WORDS = ["meeting", "invoice", "lunch", "report", "agenda", "draft", "review", "schedule", "notes", "team"]
SHARED = ["today", "please", "now", "update", "account", "link", "reply", "thanks", "week", "call"]
VOCAB = SPAM_WORDS + HAM_WORDS + SHARED


def make_messages(n: int, seed: int = 0) -> tuple[list[list[str]], np.ndarray]:
    """Messages are bags of words. Spam leans on spam words but often avoids the obvious ones."""
    rng = np.random.default_rng(seed)
    msgs, labels = [], []
    for _ in range(n):
        spam = rng.random() < 0.5
        own = SPAM_WORDS if spam else HAM_WORDS
        other = HAM_WORDS if spam else SPAM_WORDS
        k_own = rng.integers(1, 4)
        k_other = rng.integers(0, 2)
        words = list(rng.choice(own, k_own)) + list(rng.choice(other, k_other)) + list(rng.choice(SHARED, rng.integers(2, 6)))
        msgs.append(words)
        labels.append(int(spam))
    return msgs, np.array(labels)


def featurize(msgs: list[list[str]]) -> np.ndarray:
    idx = {w: i for i, w in enumerate(VOCAB)}
    x = np.zeros((len(msgs), len(VOCAB)))
    for r, m in enumerate(msgs):
        for w in m:
            x[r, idx[w]] += 1
    return x


def rule_classifier(words: list[str]) -> int:
    """What a person might write by hand: three trigger words."""
    return int(any(w in {"prize", "winner", "free"} for w in words))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train_logistic(x: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 300) -> tuple[np.ndarray, float]:
    if lr <= 0 or epochs < 1:
        raise SafeError("lr must be > 0 and epochs >= 1")
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        g = p - y
        w -= lr * x.T @ g / len(y)
        b -= lr * g.mean()
    return w, b


def rules_vs_learned(seed: int = 0) -> dict:
    msgs, y = make_messages(2000, seed)
    tr, te = slice(0, 1500), slice(1500, 2000)
    x = featurize(msgs)
    rules_acc = float(np.mean([rule_classifier(m) for m in msgs[te]] == y[te]))
    w, b = train_logistic(x[tr], y[tr])
    learned_acc = float(np.mean((sigmoid(x[te] @ w + b) > 0.5) == y[te]))
    top = [VOCAB[i] for i in np.argsort(-w)[:5]]
    return {"test_messages": 500, "rules_accuracy": round(rules_acc, 3), "learned_accuracy": round(learned_acc, 3), "top_spam_weights": top}


# ---------------------------------------------------------------- house prices + gradient descent
def make_houses(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Size (100 m^2 units) and rooms -> price in $1,000s, plus noise. Synthetic."""
    rng = np.random.default_rng(seed)
    size = rng.uniform(0.5, 3.0, n)
    rooms = rng.integers(1, 6, n).astype(float)
    price = 120 * size + 15 * rooms + 40 + rng.normal(0, 10, n)
    return np.column_stack([size, rooms]), price


def gradient_descent(x: np.ndarray, y: np.ndarray, lr: float, steps: int = 500) -> tuple[np.ndarray, list[float]]:
    """Plain batch gradient descent on mean squared error. Returns weights [bias, w...] and loss history."""
    if lr <= 0 or steps < 1:
        raise SafeError("lr must be > 0 and steps >= 1")
    xb = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(xb.shape[1])
    hist = []
    for _ in range(steps):
        err = xb @ w - y
        loss = float(np.mean(err**2))
        hist.append(loss)
        if not np.isfinite(loss) or loss > 1e12:
            break
        w -= lr * 2 * xb.T @ err / len(y)
    return w, hist


def gd_sweep(seed: int = 0) -> dict:
    x, y = make_houses(400, seed)
    out = {}
    for lr in (0.001, 0.01, 0.05, 0.12):
        w, hist = gradient_descent(x, y, lr)
        diverged = (not np.isfinite(hist[-1])) or hist[-1] > hist[0]
        out[str(lr)] = {"start_mse": round(hist[0], 1), "final_mse": None if diverged else round(hist[-1], 1), "diverged": bool(diverged)}
    w, hist = gradient_descent(x, y, 0.05, 5000)
    out["best_fit"] = {"bias": round(float(w[0]), 1), "per_100m2": round(float(w[1]), 1), "per_room": round(float(w[2]), 1)}
    out["true_rule"] = {"bias": 40, "per_100m2": 120, "per_room": 15}
    return out


# ---------------------------------------------------------------- underfit / overfit sweep
def fit_sweep(seed: int = 0, n_train: int = 15) -> dict:
    """Price vs size with a gentle curve. Fit polynomials of growing degree on few points."""
    rng = np.random.default_rng(seed)

    def truth(s):
        return 50 + 150 * s - 25 * s**2

    s_tr = np.sort(rng.uniform(0.5, 3.0, n_train))
    s_va = rng.uniform(0.5, 3.0, 300)
    y_tr = truth(s_tr) + rng.normal(0, 12, n_train)
    y_va = truth(s_va) + rng.normal(0, 12, 300)
    rows = []
    for d in (0, 1, 2, 4, 8, 12):
        c = np.polynomial.polynomial.Polynomial.fit(s_tr, y_tr, d)
        rows.append(
            {
                "degree": d,
                "train_rmse": round(float(np.sqrt(np.mean((c(s_tr) - y_tr) ** 2))), 1),
                "val_rmse": round(float(np.sqrt(np.mean((c(s_va) - y_va) ** 2))), 1),
            }
        )
    best = min(rows, key=lambda r: r["val_rmse"])["degree"]
    return {"train_points": n_train, "val_points": 300, "rows": rows, "best_degree": best}


# ---------------------------------------------------------------- k-means
LAUNDRY = {"shirt": (60, 1.0), "trousers": (100, 2.5), "sock": (20, 0.3)}  # (length cm, weight-ish)


def make_laundry(n_each: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pts, kinds = [], []
    for k, (length, weight) in enumerate(LAUNDRY.values()):
        pts.append(np.column_stack([rng.normal(length, 8, n_each), rng.normal(weight, 0.25, n_each)]))
        kinds += [k] * n_each
    return np.vstack(pts), np.array(kinds)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
    if k < 1 or k > len(x):
        raise SafeError("k must be between 1 and the number of points")
    z = (x - x.mean(0)) / x.std(0)
    rng = np.random.default_rng(seed)
    # k-means++ style seeding
    centers = [z[rng.integers(len(z))]]
    for _ in range(1, k):
        d = np.min([((z - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(z[rng.choice(len(z), p=d / d.sum())])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((z[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.array([z[lab == j].mean(0) if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab, c


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    return float(sum(np.bincount(truth[labels == j]).max() for j in np.unique(labels)) / len(truth))


def kmeans_demo(seed: int = 0) -> dict:
    x, kinds = make_laundry(seed=seed)
    lab, _ = kmeans(x, 3, seed)
    return {"items": len(x), "labels_used_for_training": 0, "purity_vs_hidden_kind": round(purity(lab, kinds), 3)}


# ---------------------------------------------------------------- data volume
def learning_curve(seed: int = 0) -> list[dict]:
    """Same model, more labelled examples: held-out accuracy of the spam model."""
    msgs, y = make_messages(6000, seed + 1)
    x = featurize(msgs)
    xte, yte = x[5000:], y[5000:]
    rows = []
    for n in (10, 30, 100, 300, 1000, 5000):
        w, b = train_logistic(x[:n], y[:n])
        rows.append({"train_examples": n, "test_accuracy": round(float(np.mean((sigmoid(xte @ w + b) > 0.5) == yte)), 3)})
    return rows
