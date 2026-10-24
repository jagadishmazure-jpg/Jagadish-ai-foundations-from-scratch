# Threat model (trip planner)

Scope: `trip.py`, the only part that takes free text and calls tools. Everything else is offline
training on generated data.

| STRIDE | Threat | Mitigation here | Gap |
|---|---|---|---|
| Spoofing | caller pretends to be another traveller | none (single-user demo) | real auth: Jagadish-agent-auth-security |
| Tampering | crafted tool arguments (extra keys, negative prices) | strict schema, allow-list | no signing of tool results |
| Repudiation | no record of what the agent did | full step trace returned | not persisted |
| Information disclosure | PII echoed into logs or errors | scrubber on logs, errors, request echo | scrubber is pattern-based |
| Denial of service | endless loop | per-worker `max_steps` | no wall-clock timeout |
| Elevation of privilege | model asks for a tool it should not have | only four read-only tools exist | n/a |

OWASP LLM Top 10 notes: LLM01 prompt injection is out of scope because no model reads the text
(scripted policy); LLM06 excessive agency is limited by read-only tools; LLM09 overreliance /
misinformation is addressed by the grounding check.
