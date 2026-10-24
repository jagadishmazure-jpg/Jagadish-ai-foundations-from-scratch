# Coverage map: what lives here and what lives elsewhere

Each idea from a typical "AI basics in one sitting" walkthrough, checked against every public repo in
this portfolio (83 repos searched with code search on 2026-10-24). Ideas already backed by real code
elsewhere are linked, not rebuilt. Ideas that were missing or only half there are built here.

| # | Idea | Before this repo | Where it lives now |
|---|---|---|---|
| 1 | AI > ML > deep learning > generative AI | missing (no code) | [how-it-works.md](how-it-works.md#the-ladder), each rung has a module here |
| 2 | Hand-written rules vs learned rules | missing | `classic.rules_vs_learned` |
| 3 | Supervised / unsupervised / reinforcement | partial: RL in [agent-training-distill-rl](https://github.com/jagadishmazure-jpg/Jagadish-agent-training-distill-rl) (GRPO, DPO); no from-scratch clustering | supervised + k-means here; RL linked |
| 4 | Training loop: guess, measure error, nudge parameters | partial (inside library calls) | `classic.gradient_descent`, `neural.MLP.step` |
| 5 | Gradient descent | missing from scratch | `classic.gradient_descent` + step-size sweep |
| 6 | Underfit / overfit / generalisation | partial (task labels only in coding-computer-use-agents) | `classic.fit_sweep` |
| 7 | Neurons, weights, activation, layers, logits -> softmax | partial (softmax helper in distill-rl) | `neural` (hand-written backprop, depth study) |
| 8 | Why 2012: data + compute | missing | `classic.learning_curve` (data side only, see limitations) |
| 9a | Tokenisation, token IDs | **covered**: [token-accounting](https://github.com/jagadishmazure-jpg/Jagadish-token-accounting) (real tokenizers, BPE from scratch) | linked |
| 9b | Embeddings, clusters, king - man + woman | partial (used via retrieval, never trained) | `embeddings` (PPMI + SVD trained here) |
| 10 | Attention, feed-forward, stacked layers | missing (only GQA arithmetic in llm-capacity-planning) | `attention` (NumPy reference) + `lm.TinyLM` |
| 11 | Next-token prediction loop | partial (behind model APIs) | `lm.generate` |
| 12 | Temperature | **covered**: [model-inference-lab](https://github.com/jagadishmazure-jpg/Jagadish-model-inference-lab) (`inferlab/generation.py`) | measured sweep added here on a real trained model |
| 13 | Pretraining, SFT, RLHF | partial: SFT in [lora-finetuning](https://github.com/jagadishmazure-jpg/Jagadish-lora-finetuning), preference/RL in distill-rl | pretrain -> SFT measured here; RLHF linked |
| 14 | Prompting and the context window | **covered**: [context-engineering](https://github.com/jagadishmazure-jpg/Jagadish-context-engineering) | small pinned-facts demo in `trip.context_window` |
| 15 | Hallucination | **covered**: [agent-reliability-lab](https://github.com/jagadishmazure-jpg/Jagadish-agent-reliability-lab), [agentic-ai](https://github.com/jagadishmazure-jpg/Jagadish-agentic-ai) eval suites | grounding check in `trip.verify` |
| 16 | RAG | **covered**: [rag-optimization](https://github.com/jagadishmazure-jpg/Jagadish-rag-optimization), [agentic-ai](https://github.com/jagadishmazure-jpg/Jagadish-agentic-ai) project 01 | hand-written TF-IDF retriever in `trip.Retriever` |
| 17 | Diffusion (denoising) | **missing** | `diffusion` |
| 18 | Multimodal | partial (sensor fusion stand-ins in azure-agent-labs) | `multimodal` (contrastive image/text) |
| 19 | Tools and function calling | **covered**: [agentic-ai](https://github.com/jagadishmazure-jpg/Jagadish-agentic-ai) `shared/tools`, [langchain-langgraph-langsmith](https://github.com/jagadishmazure-jpg/Jagadish-langchain-langgraph-langsmith) | schema-checked tools in `trip` |
| 20 | Agent loop | **covered**: agentic-ai, [agent-framework-bakeoff](https://github.com/jagadishmazure-jpg/Jagadish-agent-framework-bakeoff) | `trip.agent_loop` |
| 21 | Multi-agent with a planner | **covered**: [multi-agent-orchestration](https://github.com/jagadishmazure-jpg/Jagadish-multi-agent-orchestration) | planner/researcher/booker/reviewer in `trip` |
| 22 | MCP | **covered**: agentic-ai `shared/mcp_servers`, [azure-ai-integration-platform](https://github.com/jagadishmazure-jpg/Jagadish-azure-ai-integration-platform) MCP gateway; auth in [agent-auth-security](https://github.com/jagadishmazure-jpg/Jagadish-agent-auth-security) | JSON-RPC `tools/list` + `tools/call` in `trip.ToolServer` |
| 23 | One request through the whole stack | missing as a single walkthrough | [trip-walkthrough.md](trip-walkthrough.md) |

Tally before this repo: 9 covered, 9 partial, 5 missing.
