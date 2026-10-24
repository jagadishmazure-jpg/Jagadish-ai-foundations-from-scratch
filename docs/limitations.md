# Limitations and known misses

- **Toy scale.** Largest model is about 99k parameters. These runs show mechanisms, not quality.
- **Synthetic data, one seed.** Every dataset is generated in code. No confidence intervals.
- **Embeddings need room.** At 8 dimensions one of five analogies fails (`test_known_miss_at_8_dims`).
  Analogy answers are searched among content nouns only; with function words included, the
  nearest vector is often a template word such as "to".
- **SFT only partly transfers facts.** On held-out places, format goes 0/6 -> 5/6 but correct answers
  only 0/6 -> 2/6; "weather in goa" still continues as prose.
- **First-layer weights are noisy.** The evidence board shows them raw; they do not form clean edge
  detectors at this size and training length.
- **The 2012 story is half shown.** The learning curve covers data; the compute half (GPU speed-up)
  needs a GPU and is planned, not run.
- **RLHF is not rebuilt.** Preference tuning lives in Jagadish-agent-training-distill-rl (DPO, GRPO).
- **Scripted agent.** The trip planner's decisions are rules, so it never mis-plans; a model-driven
  loop would. Keyword retrieval ranks a scooter note high for a "beaches" query.
- **Cross-machine drift.** PyTorch results can differ slightly on other CPUs; CI checks that docs
  match the committed `results.json`, and tests check thresholds rather than exact values.
