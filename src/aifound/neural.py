"""A neural network from scratch in NumPy: neurons, layers, ReLU, softmax and backpropagation.

Task: classify 8x8 pictures (synthetic) as a horizontal bar, a vertical bar, a box or an X.
We train networks of depth 0 (softmax regression), 1 and 2 hidden layers and compare them, and we
look inside: hidden units in the first layer respond to small stroke pieces; the output layer gives
logits that softmax turns into probabilities.
"""

from __future__ import annotations

import numpy as np

from .pii import SafeError

CLASSES = ["horizontal", "vertical", "box", "x"]
SIZE = 8


def draw(kind: int, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((SIZE, SIZE))
    a, b = sorted(rng.choice(np.arange(1, SIZE - 1), 2, replace=False))
    if b - a < 2:
        b = min(a + 3, SIZE - 1)
    if kind == 0:
        img[rng.integers(1, SIZE - 1), a : b + 1] = 1
    elif kind == 1:
        img[a : b + 1, rng.integers(1, SIZE - 1)] = 1
    elif kind == 2:
        img[a, a : b + 1] = img[b, a : b + 1] = img[a : b + 1, a] = img[a : b + 1, b] = 1
    else:
        for i in range(a, b + 1):
            img[i, i] = 1
            img[i, a + b - i] = 1
    img += rng.normal(0, 0.15, img.shape)
    return img


def make_shapes(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    return np.array([draw(k, rng).ravel() for k in y]), y


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)


def neuron(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """One neuron: weigh the inputs, add them up, fire (pass the value) only if it is positive."""
    return float(max(0.0, x @ w + b))


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        if len(sizes) < 2 or min(sizes) < 1:
            raise SafeError("need at least input and output sizes, all positive")
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b, strict=True)):
            z = acts[-1] @ w + b
            acts.append(z if i == len(self.W) - 1 else relu(z))
        return acts[-1], acts

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x)[0])

    def step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        logits, acts = self.forward(x)
        p = softmax(logits)
        loss = float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))
        g = p.copy()
        g[np.arange(len(y)), y] -= 1
        g /= len(y)
        for i in reversed(range(len(self.W))):
            gw, gb = acts[i].T @ g, g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
            self.W[i] -= lr * gw
            self.b[i] -= lr * gb
        return loss

    def fit(self, x, y, epochs=60, lr=0.1, batch=64, seed=0) -> list[float]:
        if lr <= 0 or epochs < 1 or batch < 1:
            raise SafeError("lr, epochs and batch must be positive")
        rng = np.random.default_rng(seed)
        hist = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            losses = [self.step(x[order[i : i + batch]], y[order[i : i + batch]], lr) for i in range(0, len(x), batch)]
            hist.append(float(np.mean(losses)))
        return hist


def depth_study(seed: int = 0, epochs: int = 60) -> dict:
    xtr, ytr = make_shapes(3000, seed)
    xte, yte = make_shapes(1000, seed + 99)
    rows = []
    nets = {}
    for name, hidden in (("0 hidden layers", []), ("1 hidden layer (32)", [32]), ("2 hidden layers (32, 16)", [32, 16])):
        net = MLP([64, *hidden, 4], seed)
        hist = net.fit(xtr, ytr, epochs=epochs, seed=seed)
        acc = float(np.mean(net.predict_proba(xte).argmax(1) == yte))
        rows.append({"model": name, "first_epoch_loss": round(hist[0], 3), "last_epoch_loss": round(hist[-1], 3), "test_accuracy": round(acc, 3)})
        nets[name] = net
    deep = nets["2 hidden layers (32, 16)"]
    sample_x, sample_y = make_shapes(1, seed + 7)
    logits, _ = deep.forward(sample_x)
    probs = softmax(logits)[0]
    return {
        "train_images": 3000,
        "test_images": 1000,
        "rows": rows,
        "example": {
            "true": CLASSES[int(sample_y[0])],
            "logits": [round(float(v), 2) for v in logits[0]],
            "probabilities": {c: round(float(p), 3) for c, p in zip(CLASSES, probs, strict=True)},
        },
        "_net": deep,
    }
