# ADR 0002: scripted policy for the trip agent

Status: accepted (2026-10-24)

Context: a model-driven agent needs a model. Local models are large; hosted ones cost money and are
not repeatable.

Decision: the trip planner's next-move decisions are a small scripted policy with the same
interface a model would fill. Retrieval, tools, the JSON-RPC server, the step cap and the grounding
check are all real.

Consequences: exact, free, testable runs; it cannot show model mistakes. The model-driven loop is
linked in Jagadish-agentic-ai.
