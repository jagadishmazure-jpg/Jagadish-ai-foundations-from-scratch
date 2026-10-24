"""A tiny character-level transformer in PyTorch (CPU), written out by hand.

Pieces: token + position embeddings -> N blocks of [single-head causal attention, feed-forward
network] with residuals and layer norm -> logits over characters. Generation is the plain loop:
predict a distribution for the next character, sample one (temperature controls how flat the
distribution is), append it, repeat.

Also here: a two-stage training story. Stage 1 "pretraining" reads plain sentences about places.
Stage 2 "supervised fine-tuning" shows question/answer pairs for SOME places only. We then ask
about the held-out places and measure whether answers come back in the assistant format and
whether the fact (learned only in stage 1) is right.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .pii import SafeError

torch.set_num_threads(1)

PLACES = {
    "goa": ("warm", "beaches"),
    "shimla": ("cold", "hills"),
    "jaipur": ("dry", "forts"),
    "kochi": ("humid", "backwaters"),
    "leh": ("cold", "passes"),
    "pune": ("mild", "cafes"),
    "agra": ("hot", "tombs"),
    "ooty": ("cool", "gardens"),
}
SFT_PLACES = ["shimla", "jaipur", "kochi", "pune", "ooty"]
HELDOUT_PLACES = ["goa", "leh", "agra"]


def pretrain_text(seed: int = 0, n: int = 3000) -> str:
    rng = np.random.default_rng(seed)
    names = list(PLACES)
    out = []
    for _ in range(n):
        p = names[rng.integers(len(names))]
        w, f = PLACES[p]
        out.append([f"{p} is {w} and known for {f}.", f"people visit {p} for the {f}.", f"in {p} the weather is {w}.", f"{p} has {f} and {w} days."][rng.integers(4)])
    return " ".join(out) + " "


def sft_text(seed: int = 0, n: int = 1500) -> str:
    rng = np.random.default_rng(seed + 1)
    out = []
    for _ in range(n):
        p = SFT_PLACES[rng.integers(len(SFT_PLACES))]
        w, f = PLACES[p]
        out.append([f"Q: weather in {p}? A: {w}.\n", f"Q: {p} is known for? A: {f}.\n"][rng.integers(2)])
    return "".join(out)


class Head(nn.Module):
    """One causal self-attention head, the same maths as attention.attention()."""

    def __init__(self, d: int, block: int):
        super().__init__()
        self.q, self.k, self.v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.register_buffer("mask", torch.tril(torch.ones(block, block, dtype=torch.bool)))

    def forward(self, x, return_weights=False):
        t = x.shape[1]
        q, k, v = self.q(x), self.k(x), self.v(x)
        s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        s = s.masked_fill(~self.mask[:t, :t], float("-inf"))
        w = torch.softmax(s, dim=-1)
        return (w @ v, w) if return_weights else w @ v


class Block(nn.Module):
    def __init__(self, d: int, block: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Head(d, block)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int = 64, layers: int = 2, block: int = 64):
        super().__init__()
        if layers < 1 or d < 1 or block < 2:
            raise SafeError("layers, d and block must be positive")
        self.block = block
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(block, d)
        self.blocks = nn.Sequential(*[Block(d, block) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, vocab)

    def forward(self, idx):
        t = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(t))
        return self.out(self.norm(self.blocks(x)))


class CharCodec:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        unknown = sorted({c for c in s if c not in self.stoi})
        if unknown:
            raise SafeError(f"{len(unknown)} character(s) not in the vocabulary")
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


def train(model: TinyLM, data: torch.Tensor, steps: int, lr: float = 3e-3, batch: int = 32, seed: int = 0) -> list[float]:
    if steps < 1 or lr <= 0:
        raise SafeError("steps and lr must be positive")
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    b = model.block
    for _ in range(steps):
        ix = torch.randint(0, len(data) - b - 1, (batch,), generator=g)
        x = torch.stack([data[i : i + b] for i in ix])
        y = torch.stack([data[i + 1 : i + b + 1] for i in ix])
        loss = nn.functional.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    return hist


@torch.no_grad()
def eval_loss(model: TinyLM, data: torch.Tensor, n: int = 64) -> float:
    b = model.block
    starts = torch.linspace(0, len(data) - b - 2, n).long()
    x = torch.stack([data[i : i + b] for i in starts])
    y = torch.stack([data[i + 1 : i + b + 1] for i in starts])
    return float(nn.functional.cross_entropy(model(x).reshape(-1, model.out.out_features), y.reshape(-1)))


@torch.no_grad()
def next_token_probs(model: TinyLM, ids: list[int], temperature: float = 1.0) -> torch.Tensor:
    if temperature < 0:
        raise SafeError("temperature must be >= 0")
    logits = model(torch.tensor([ids[-model.block :]]))[0, -1]
    if temperature == 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    return torch.softmax(logits / temperature, -1)


@torch.no_grad()
def generate(model, codec: CharCodec, prompt: str, n: int, temperature: float = 0.0, seed: int = 0, stop: str | None = None) -> str:
    g = torch.Generator().manual_seed(seed)
    ids = codec.encode(prompt)
    for _ in range(n):
        p = next_token_probs(model, ids, temperature)
        ids.append(int(torch.multinomial(p, 1, generator=g)))
        if stop and codec.chars[ids[-1]] == stop:
            break
    return codec.decode(ids[len(codec.encode(prompt)) :])


def entropy_bits(p: torch.Tensor) -> float:
    p = p[p > 0]
    return abs(float(-(p * torch.log2(p)).sum()))


def run(seed: int = 0, pre_steps: int = 600, sft_steps: int = 300) -> dict:
    torch.manual_seed(seed)
    pre, sft = pretrain_text(seed), sft_text(seed)
    codec = CharCodec(pre + sft + "?")
    pre_ids = torch.tensor(codec.encode(pre))
    split = int(0.9 * len(pre_ids))
    tr, va = pre_ids[:split], pre_ids[split:]

    depth = []
    models = {}
    for layers in (1, 2):
        torch.manual_seed(seed)
        m = TinyLM(len(codec.chars), layers=layers)
        before = eval_loss(m, va)
        hist = train(m, tr, pre_steps, seed=seed)
        depth.append(
            {
                "layers": layers,
                "params": sum(p.numel() for p in m.parameters()),
                "val_loss_before": round(before, 3),
                "val_loss_after": round(eval_loss(m, va), 3),
                "train_loss_last": round(float(np.mean(hist[-20:])), 3),
            }
        )
        models[layers] = m
    base = models[2]

    # temperature sweep on the pretrained model
    prompt = "people visit "
    temps = []
    for t in (0.0, 0.5, 1.0, 1.5):
        outs = [generate(base, codec, prompt, 24, t, seed=s) for s in range(20)]
        temps.append(
            {
                "temperature": t,
                "distinct_of_20": len(set(outs)),
                "next_char_entropy_bits": round(entropy_bits(next_token_probs(base, codec.encode(prompt), t)), 3),
                "sample": outs[0],
            }
        )

    def qa_eval(model) -> dict:
        fmt = correct = 0
        cases = []
        for p in HELDOUT_PLACES:
            w, f = PLACES[p]
            for q, want in ((f"Q: weather in {p}? A:", w), (f"Q: {p} is known for? A:", f)):
                ans = generate(model, codec, q, 30, 0.0, stop="\n")
                ok_fmt = ans.endswith("\n") and len(ans.strip().split()) <= 2
                fmt += ok_fmt
                correct += ok_fmt and want in ans
                cases.append({"prompt": q, "answer": ans.strip()})
        n = 2 * len(HELDOUT_PLACES)
        return {"format_ok": f"{fmt}/{n}", "correct": f"{correct}/{n}", "cases": cases}

    after_pre = qa_eval(base)
    sft_ids = torch.tensor(codec.encode(sft + pre[: len(sft)]))  # mix in some plain text so stage-1 facts are kept
    train(base, sft_ids, sft_steps, lr=1e-3, seed=seed + 1)
    after_sft = qa_eval(base)
    return {
        "corpus_chars": len(pre),
        "vocab_chars": len(codec.chars),
        "pretrain_steps": pre_steps,
        "sft_steps": sft_steps,
        "depth": depth,
        "temperature": temps,
        "heldout_places": HELDOUT_PLACES,
        "qa_after_pretraining": after_pre,
        "qa_after_sft": after_sft,
        "_model": base,
        "_codec": codec,
    }
