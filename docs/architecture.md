# Architecture

```
src/aifound/
  classic.py      NumPy: rules vs learned spam, gradient descent, polynomial sweep, k-means, data curve
  neural.py       NumPy: neuron, ReLU, softmax, MLP with hand-written backprop, depth study
  embeddings.py   NumPy: co-occurrence -> PPMI -> SVD word vectors, nearest words, analogies
  attention.py    NumPy: scaled dot-product attention with causal mask (reference maths)
  lm.py           PyTorch: single-head transformer blocks, char codec, training, sampling, pretrain -> SFT
  diffusion.py    PyTorch: noise schedule, noise-predicting MLP, DDPM sampling on 2-D points
  multimodal.py   PyTorch: image encoder + text encoder trained with a contrastive loss
  trip.py         Pure Python: TF-IDF retrieval, JSON-RPC tool server, agent loop, planner, reviewer
  pii.py          PII scrubber, safe errors, log filter
  cli.py          `aifound demo`: runs everything, writes results.json and the evidence board
scripts/
  render_docs.py  copies numbers from results.json into README/docs; --check in CI
  count_tests.py  prints the real collected test count
  check_brands.py hashed blocklist of real company names
  overlap_check.py 8-word overlap check against reference text kept outside the repo
```

Design choices:

- **NumPy first, PyTorch only where autograd saves pages.** Backprop is written by hand once (in
  `neural.py`) and checked against a numeric gradient; after that PyTorch's autograd is used.
- **One reference, one fast version.** The PyTorch attention head is tested against the NumPy
  reference to 1e-5, so the readable version and the trained version are provably the same maths.
- **Everything deterministic on one machine.** Fixed seeds, `torch.set_num_threads(1)` for the LM.
- **Docs are generated from a run.** CI fails if README numbers or the test count drift.

See the ADRs in `docs/adr/`.
