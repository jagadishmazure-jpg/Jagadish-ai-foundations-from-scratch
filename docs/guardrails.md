# Guardrails

| Risk | Control | Test |
|---|---|---|
| Bad numeric input (negative budget, NaN, inf, zero steps, lr <= 0) | `SafeError` at every entry point | `test_bad_budget_rejected`, `test_*_rejects_*` |
| Tool misuse (unknown tool, extra or missing args, wrong types, negative costs) | fixed allow-list + strict schema check in `trip._check` | `test_bad_tool_calls_rejected` |
| Malformed RPC | JSON-RPC error codes -32700 / -32602, no crash | `test_server_parse_error_and_unknown_method` |
| PII in requests, errors, logs | `pii.scrub` (Luhn cards, SSN rules, emails, phones); `ScrubFilter` on the logger; `SafeError` scrubs its own text | `test_request_email_is_masked`, `test_logs_are_scrubbed`, `test_error_messages_are_scrubbed` |
| Hallucinated facts in the answer | reviewer keeps only claims sourced to a retrieved note or a tool that ran | `test_planted_hallucination_is_dropped` |
| Inventing a plan that does not fit | infeasible budgets return an explicit "no plan fits" | `test_tight_budget_says_no_instead_of_inventing` |
| Runaway agent | `max_steps` cap per worker | `test_agent_loop_stops_at_max_steps` |
| Lost user facts when history is long | pinned facts survive context trimming | `test_context_window_drops_old_turns_and_pins_survive` |
| Real company names / secrets in the repo | hashed brand blocklist, gitleaks over full history, push protection | CI |
| Docs claiming numbers that were not measured | `render_docs.py --check` in CI | CI |
