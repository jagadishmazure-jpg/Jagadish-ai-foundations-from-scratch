# Real code and its output

Every output below is copied from `results/results.json` of the committed full run.

## Softmax and a forward pass (`neural.py`)

```python
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
```

Output for one test picture of an X: logits `[2.47, -3.78, -2.18, 10.0]` ->
probabilities `{horizontal: 0.001, vertical: 0.0, box: 0.0, x: 0.999}`.

## Attention (`attention.py`)

```python
scores = q @ k.T / np.sqrt(d)
if causal:
    scores = np.where(np.tril(np.ones((t, t), bool)), scores, -np.inf)
w = softmax(scores, axis=-1)
return w @ v, w
```

Hand-set toy (not trained) where "it" asks for an animal: weights from "it" =
`{the: 0.04, cat: 0.812, sat: 0.04, because: 0.04, it: 0.067}`.

## Analogy (`embeddings.py`)

```python
def analogy(idx, vec, a, b, c):
    return nearest(idx, vec, vec[idx[b]] - vec[idx[a]] + vec[idx[c]], {a, b, c})[0]
```

Output: `king - man + woman -> queen`, `queen - king + prince -> princess`; 5/5 at 16 dimensions,
4/5 at 8 (the miss is kept as a test).

## Sampling with temperature (`lm.py`)

```python
logits = model(torch.tensor([ids[-model.block:]]))[0, -1]
return torch.softmax(logits / temperature, -1)
```

Output, 20 samples of 24 characters after "people visit ": distinct continuations 1 / 10 / 12 / 17
at T = 0 / 0.5 / 1.0 / 1.5.

## Pretraining then SFT (`lm.py`), held-out place "goa"

- after pretraining: `Q: goa is known for? A:` -> `h tombs. people visit shimla f`
- after SFT: `Q: goa is known for? A:` -> `beaches.`

## DDPM step (`diffusion.py`)

```python
eps = model(x, torch.full((n,), t))
x = (x - beta[t] / (1 - abar[t]).sqrt() * eps) / alpha[t].sqrt()
```

Output: mean distance from the ring 0.567 (pure noise) -> 0.065 after 50 steps (real data 0.016).

## Grounding check (`trip.py`)

```python
ok = [c for c in claims if c.source in evidence]
```

Output: the planted claim `the beach is free of any entry fee all year` is dropped.
