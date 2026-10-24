# ADR 0001: NumPy first, PyTorch only where autograd pays off

Status: accepted (2026-10-24)

Context: the repo exists to show mechanisms. Library calls hide them.

Decision: classic ML, the MLP, embeddings and attention are NumPy with hand-written gradients. The
transformer, diffusion and multimodal models use PyTorch CPU autograd, and the attention head is
tested against the NumPy reference.

Consequences: slower than library code, but every step is readable; one extra dependency (CPU torch).
