"""Multimodal in miniature (PyTorch, CPU): put pictures and captions in one shared vector space.

An image encoder (small MLP over 8x8 pixels) and a text encoder (mean of word embeddings) are trained
together with a contrastive loss: a picture and its own caption should score higher than that
picture with any other caption in the batch, and the other way round. Afterwards we retrieve the
right caption for unseen pictures and the right picture for a caption, in both directions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .neural import CLASSES, draw
from .pii import SafeError

SIDES = ["left", "right"]
WORDS = ["a", "picture", "of", "on", "the", *CLASSES, *SIDES]


def make_pairs(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """A shape drawn in the left or right half; caption names shape and side (8 combinations)."""
    rng = np.random.default_rng(seed)
    imgs, caps, labels = [], [], []
    for _ in range(n):
        k, s = int(rng.integers(4)), int(rng.integers(2))
        small = draw(k, rng).reshape(4, 2, 4, 2).max(axis=(1, 3))  # 2x2 max-pool to 4x4 keeps thin strokes
        img = np.zeros((8, 8)) + rng.normal(0, 0.1, (8, 8))
        img[2:6, (0 if s == 0 else 4) : (4 if s == 0 else 8)] += small
        imgs.append(img.ravel())
        caps.append([WORDS.index(w) for w in ["a", "picture", "of", CLASSES[k], "on", "the", SIDES[s]]])
        labels.append(k * 2 + s)
    return torch.tensor(np.array(imgs), dtype=torch.float32), torch.tensor(caps), np.array(labels)


class Dual(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))
        self.txt = nn.EmbeddingBag(len(WORDS), d, mode="mean")
        self.txt_proj = nn.Linear(d, d)
        self.scale = nn.Parameter(torch.tensor(2.0))

    def encode_image(self, x):
        return nn.functional.normalize(self.img(x), dim=-1)

    def encode_text(self, c):
        return nn.functional.normalize(self.txt_proj(self.txt(c)), dim=-1)


def train(steps: int = 400, seed: int = 0) -> Dual:
    if steps < 1:
        raise SafeError("steps must be positive")
    torch.manual_seed(seed)
    x, c, lab = make_pairs(4000, seed)
    m = Dual()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    lab_t = torch.tensor(lab)
    for _ in range(steps):
        ix = torch.randint(0, len(x), (128,), generator=g)
        logits = m.scale.exp() * m.encode_image(x[ix]) @ m.encode_text(c[ix]).T
        same = (lab_t[ix][:, None] == lab_t[ix][None]).float()  # identical captions count as matches
        tgt = same / same.sum(1, keepdim=True)
        loss = -(tgt * logits.log_softmax(1)).sum(1).mean() - (tgt * logits.log_softmax(0)).sum(0).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


def run(seed: int = 0, steps: int = 1000) -> dict:
    m = train(steps, seed)
    with torch.no_grad():
        return _evaluate(m, seed, steps)


def _evaluate(m: Dual, seed: int, steps: int) -> dict:
    x, _, lab = make_pairs(500, seed + 50)
    caps = torch.tensor([[WORDS.index(w) for w in ["a", "picture", "of", CLASSES[k], "on", "the", SIDES[s]]] for k in range(4) for s in range(2)])
    sims = m.encode_image(x) @ m.encode_text(caps).T
    pred = sims.argmax(1).numpy()
    img_to_text = float((pred == lab).mean())
    side_ok = float((pred % 2 == lab % 2).mean())
    shape_ok = float((pred // 2 == lab // 2).mean())
    # text -> image: for each caption, is the top image of that caption's kind?
    top = sims.argmax(0).numpy()
    text_to_img = float(np.mean([lab[top[j]] == j for j in range(8)]))
    return {
        "train_steps": steps,
        "test_images": 500,
        "captions": 8,
        "chance": 0.125,
        "image_to_caption_top1": round(img_to_text, 3),
        "caption_to_image_top1": round(text_to_img, 3),
        "side_right": round(side_ok, 3),
        "shape_right": round(shape_ok, 3),
    }
