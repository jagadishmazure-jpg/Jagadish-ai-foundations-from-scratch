# Implementation guide (for adopters)

1. **Start from the reference maths.** Copy `attention.attention` or `neural.MLP` when you need to
   explain or test a mechanism; keep a numeric gradient check (`test_backprop_matches_numeric_gradient`).
2. **Cross-check fast code against the reference.** Pattern: `test_torch_head_matches_numpy_reference`.
3. **Measure the trade-off, not just the best point.** Every module sweeps one knob (step size,
   degree, depth, dims, temperature, training data) so the failure side is visible.
4. **For agents, put the checks outside the model.** Reuse `trip._check` (schema), the JSON-RPC
   shape of `ToolServer`, `verify` (grounding) and `context_window(pinned=...)`.
5. **Swap the scripted policy for a model** by replacing `researcher_policy` / `booker_policy` with a
   function that asks a model for the next move in the same tuple format; keep `max_steps` and the
   reviewer unchanged.
6. **Keep numbers honest.** Run `aifound demo`, then `scripts/render_docs.py`; CI rejects drift.
