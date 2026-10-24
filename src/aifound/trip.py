"""End-to-end: "plan my Goa trip under $500", built from the pieces in this repo.

Stack, in the order a request meets it:
1. Context window: the chat history is cut to a token budget; pinned facts survive the cut.
2. Planner agent splits the goal into jobs for a researcher, a booker and a reviewer.
3. Researcher uses retrieval (RAG) over a small travel knowledge base and a weather tool.
4. Booker calls stay, transport and calculator tools through an MCP-style JSON-RPC server
   (tools/list, tools/call) with strict argument checks.
5. Each worker is an agent loop: decide -> call a tool -> observe -> stop when done or out of steps.
6. Reviewer checks the budget and that every claim in the answer points at a retrieved passage or
   a tool result (an ungrounded claim is a hallucination and is removed).

The "brain" deciding the next action is a small scripted policy, not a language model, so runs are
exact and testable. The language-model side of the same loop lives in the agent repos linked in the
README. All places, hotels and prices are invented sample data.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .pii import SafeError, get_logger, scrub

log = get_logger("aifound.trip")

# ------------------------------------------------------------------ knowledge base + retrieval
KB = {
    "kb-goa-season": "goa dry season runs from november to february with calm sea and warm days",
    "kb-goa-monsoon": "goa monsoon months june to september bring heavy rain and many beach shacks close",
    "kb-goa-north": "north goa beaches are lively with night markets and water sports",
    "kb-goa-south": "south goa beaches are quieter with long sand and fewer crowds",
    "kb-goa-transport": "scooter rental is the cheapest way to move between goa beaches",
    "kb-shimla-snow": "shimla gets snow in january and roads to higher passes can close",
    "kb-visa": "domestic travellers need only a photo id for hotel check in",
}


STOP = {"the", "is", "in", "a", "an", "of", "to", "and", "with", "when", "are", "for", "my", "from", "can", "many", "way", "only"}


def _toks(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP]


class Retriever:
    """TF-IDF with cosine similarity, written out by hand."""

    def __init__(self, docs: dict[str, str]):
        self.ids = list(docs)
        self.tf = [Counter(_toks(docs[i])) for i in self.ids]
        df = Counter(w for tf in self.tf for w in tf)
        n = len(self.ids)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.docs = docs

    def _vec(self, tf: Counter) -> dict[str, float]:
        v = {w: c * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, query: str, k: int = 2) -> list[tuple[str, float]]:
        q = self._vec(Counter(_toks(query)))
        scored = [(i, sum(q.get(w, 0) * x for w, x in self._vec(tf).items())) for i, tf in zip(self.ids, self.tf, strict=True)]
        return [(i, round(s, 3)) for i, s in sorted(scored, key=lambda t: -t[1])[:k] if s > 0]


# ------------------------------------------------------------------ tools behind an MCP-style server
STAYS = {
    "goa": [
        {"name": "Palmrest Inn", "area": "south", "per_night": 38},
        {"name": "Tidewater Hostel", "area": "north", "per_night": 14},
        {"name": "Coralbay Suites", "area": "north", "per_night": 95},
    ],
    "shimla": [{"name": "Pinecrest Lodge", "area": "mall road", "per_night": 45}],
}
FARES = {("pune", "goa"): {"carrier": "Breeze Rail", "round_trip": 62}, ("pune", "shimla"): {"carrier": "Breeze Air", "round_trip": 210}}
WEATHER = {"goa": {"high_c": 31, "rain_mm": 2}, "shimla": {"high_c": 9, "rain_mm": 30}}


def _check(args: dict, schema: dict[str, type]) -> None:
    extra = set(args) - set(schema)
    missing = set(schema) - set(args)
    if extra or missing:
        raise SafeError(f"bad arguments: extra={sorted(extra)} missing={sorted(missing)}")
    for k, t in schema.items():
        if not isinstance(args[k], t) or isinstance(args[k], bool):
            raise SafeError(f"argument {k} must be {t.__name__}")
        if isinstance(args[k], (int, float)) and (args[k] < 0 or not math.isfinite(args[k])):
            raise SafeError(f"argument {k} must be a finite number >= 0")


def _weather(city: str) -> dict:
    if city not in WEATHER:
        raise SafeError("unknown city")
    return WEATHER[city]


def _stays(city: str, max_per_night: int) -> list[dict]:
    return sorted((s for s in STAYS.get(city, []) if s["per_night"] <= max_per_night), key=lambda s: s["per_night"])


def _fare(origin: str, dest: str) -> dict:
    if (origin, dest) not in FARES:
        raise SafeError("no fare for that route")
    return FARES[(origin, dest)]


def _calc(items: list, nights: int) -> dict:
    if any((not isinstance(x, (int, float))) or x < 0 for x in items):
        raise SafeError("costs must be numbers >= 0")
    return {"total": round(sum(items), 2), "nights": nights}


TOOLS = {
    "get_weather": ({"city": str}, _weather, "Forecast high and rainfall for a city"),
    "search_stays": ({"city": str, "max_per_night": int}, _stays, "Stays in a city at or under a nightly price"),
    "get_fare": ({"origin": str, "dest": str}, _fare, "Round-trip fare between two cities"),
    "calculator": ({"items": list, "nights": int}, _calc, "Add up a list of costs"),
}


class ToolServer:
    """In-process JSON-RPC 2.0 server with the two MCP-style methods every agent needs."""

    def __init__(self):
        self.calls = 0

    def handle(self, raw: str) -> str:
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
        rid = req.get("id")
        try:
            if req.get("method") == "tools/list":
                result = [{"name": n, "description": d, "params": {k: t.__name__ for k, t in s.items()}} for n, (s, _, d) in TOOLS.items()]
            elif req.get("method") == "tools/call":
                name, args = req["params"]["name"], req["params"]["arguments"]
                if name not in TOOLS:
                    raise SafeError("unknown tool")
                schema, fn, _ = TOOLS[name]
                _check(args, schema)
                self.calls += 1
                result = fn(**args)
            else:
                raise SafeError("unknown method")
            return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
        except (SafeError, KeyError, TypeError) as e:
            return json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": scrub(str(e))}})


class ToolClient:
    def __init__(self, server: ToolServer):
        self.server, self.next_id = server, 0

    def list(self) -> list[dict]:
        return self._rpc("tools/list", {})

    def call(self, name: str, **arguments):
        return self._rpc("tools/call", {"name": name, "arguments": arguments})

    def _rpc(self, method, params):
        self.next_id += 1
        resp = json.loads(self.server.handle(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params})))
        if "error" in resp:
            raise SafeError(resp["error"]["message"])
        return resp["result"]


# ------------------------------------------------------------------ context window
def context_window(turns: list[str], budget_tokens: int, pinned: list[str] | None = None) -> list[str]:
    """Keep pinned facts, then the newest turns that fit. Tokens here are words."""
    if budget_tokens < 0:
        raise SafeError("budget must be >= 0")
    kept, used = [], sum(len(p.split()) for p in pinned or [])
    for t in reversed(turns):
        n = len(t.split())
        if used + n > budget_tokens:
            break
        kept.append(t)
        used += n
    return list(pinned or []) + kept[::-1]


# ------------------------------------------------------------------ agents
@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)

    def add(self, agent: str, action: str, observation) -> None:
        self.steps.append({"agent": agent, "action": action, "observation": observation})
        log.info("%s %s", agent, scrub(action))


@dataclass
class Claim:
    text: str
    source: str  # kb id or tool:<name>


def agent_loop(name: str, policy, state: dict, client: ToolClient, retriever: Retriever, trace: Trace, max_steps: int = 6) -> dict:
    """Decide, act, observe, repeat. The policy returns ("tool", name, args), ("search", query) or ("done",)."""
    for _ in range(max_steps):
        move = policy(state)
        if move[0] == "done":
            return state
        if move[0] == "search":
            hits = retriever.search(move[1])
            state.setdefault("passages", []).extend(hits)
            trace.add(name, f"search: {move[1]}", hits)
        else:
            try:
                obs = client.call(move[1], **move[2])
            except SafeError as e:
                obs = {"error": str(e)}
            state.setdefault("results", {})[move[1]] = obs
            trace.add(name, f"tool: {move[1]}", obs)
    state["out_of_steps"] = True
    return state


def researcher_policy(state):
    if "passages" not in state:
        return ("search", f"{state['city']} season weather beaches quiet")
    if "get_weather" not in state.get("results", {}):
        return ("tool", "get_weather", {"city": state["city"]})
    return ("done",)


def booker_policy(state):
    r = state.get("results", {})
    if "get_fare" not in r:
        return ("tool", "get_fare", {"origin": state["origin"], "dest": state["city"]})
    if "error" in r["get_fare"]:
        return ("done",)
    if "search_stays" not in r:
        nightly_cap = max(0, int((state["budget"] - r["get_fare"]["round_trip"] - state["food_per_day"] * state["nights"]) // state["nights"]))
        return ("tool", "search_stays", {"city": state["city"], "max_per_night": nightly_cap})
    if "calculator" not in r and r["search_stays"]:
        stay = r["search_stays"][-1]  # best stay we can afford
        items = [r["get_fare"]["round_trip"], stay["per_night"] * state["nights"], state["food_per_day"] * state["nights"]]
        return ("tool", "calculator", {"items": items, "nights": state["nights"]})
    return ("done",)


def verify(claims: list[Claim], evidence: set[str]) -> tuple[list[Claim], list[Claim]]:
    """Reviewer's grounding check: keep only claims whose source was actually retrieved or called."""
    ok = [c for c in claims if c.source in evidence]
    return ok, [c for c in claims if c.source not in evidence]


def plan_trip(request: str, budget: float, nights: int = 4, origin: str = "pune", city: str = "goa", food_per_day: int = 20, planted_claim: str | None = None) -> dict:
    if not isinstance(budget, (int, float)) or budget < 0 or not math.isfinite(budget):
        raise SafeError("budget must be a finite number >= 0")
    if nights < 1:
        raise SafeError("nights must be >= 1")
    trace, server = Trace(), ToolServer()
    client, retriever = ToolClient(server), Retriever(KB)
    trace.add("planner", "plan", ["researcher: season + weather", "booker: fare + stay + total", "reviewer: budget + grounding"])
    trace.add("planner", "tools/list", [t["name"] for t in client.list()])
    research = agent_loop("researcher", researcher_policy, {"city": city}, client, retriever, trace)
    booking = agent_loop(
        "booker",
        booker_policy,
        {"city": city, "origin": origin, "budget": budget, "nights": nights, "food_per_day": food_per_day},
        client,
        retriever,
        trace,
    )
    r = booking.get("results", {})
    claims: list[Claim] = []
    for pid, _ in research.get("passages", [])[:2]:
        claims.append(Claim(KB[pid], pid))
    w = research["results"]["get_weather"]
    claims.append(Claim(f"forecast high {w['high_c']} C with {w['rain_mm']} mm rain", "tool:get_weather"))
    feasible = bool(r.get("search_stays")) and "calculator" in r and r["calculator"]["total"] <= budget
    if feasible:
        stay, fare = r["search_stays"][-1], r["get_fare"]
        claims.append(Claim(f"{fare['carrier']} round trip ${fare['round_trip']}", "tool:get_fare"))
        claims.append(Claim(f"{stay['name']} ({stay['area']} {city}) at ${stay['per_night']}/night", "tool:search_stays"))
        claims.append(Claim(f"total ${r['calculator']['total']} for {nights} nights incl. ${food_per_day}/day food", "tool:calculator"))
    if planted_claim:
        claims.append(Claim(planted_claim, "model-memory"))
    evidence = {pid for pid, _ in research.get("passages", [])} | {f"tool:{k}" for k in {**research.get("results", {}), **r}}
    kept, dropped = verify(claims, evidence)
    trace.add("reviewer", "verify", {"kept": len(kept), "dropped": len(dropped), "feasible": feasible})
    return {
        "request": scrub(request),
        "feasible": feasible,
        "total": r.get("calculator", {}).get("total"),
        "budget": budget,
        "answer": [f"{c.text} [{c.source}]" for c in kept] if feasible else ["no plan fits this budget with the stays on offer; raise the budget or cut nights"],
        "dropped_ungrounded": [c.text for c in dropped],
        "tool_calls": server.calls,
        "steps": len(trace.steps),
        "trace": trace.steps,
    }


def demo() -> dict:
    ok = plan_trip("plan my goa trip under $500, mail me at traveller@example.com", 500, planted_claim="the beach is free of any entry fee all year")
    tight = plan_trip("plan my goa trip under $150", 150)
    history = ["hi, my name is Asha and i travel with a scooter licence"] + [f"turn {i}: compare beach {i} with beach {i + 1} please" for i in range(30)]
    plain = context_window(history, 120)
    pinned = context_window(history, 120, pinned=["profile: name Asha, has scooter licence"])
    return {
        "under_500": {k: ok[k] for k in ("feasible", "total", "answer", "dropped_ungrounded", "tool_calls", "steps", "request")},
        "under_150": {k: tight[k] for k in ("feasible", "total", "answer", "tool_calls")},
        "context_window": {
            "turns": len(history),
            "budget_words": 120,
            "kept_plain": len(plain),
            "name_survives_plain": any("Asha" in t for t in plain),
            "kept_pinned": len(pinned),
            "name_survives_pinned": any("Asha" in t for t in pinned),
        },
        "_trace": ok["trace"],
    }
