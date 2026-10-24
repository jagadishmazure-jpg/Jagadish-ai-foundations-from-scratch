import json
import logging

import pytest

from aifound import trip
from aifound.pii import SafeError


def test_retriever_ranks_season_doc_for_season_query():
    r = trip.Retriever(trip.KB)
    assert r.search("when is the dry season in goa")[0][0] == "kb-goa-season"


def test_retriever_returns_nothing_for_unrelated_words():
    assert trip.Retriever(trip.KB).search("zebra quantum") == []


def test_tools_list_over_jsonrpc():
    names = {t["name"] for t in trip.ToolClient(trip.ToolServer()).list()}
    assert names == {"get_weather", "search_stays", "get_fare", "calculator"}


def test_tools_call_ok():
    c = trip.ToolClient(trip.ToolServer())
    assert c.call("get_weather", city="goa")["high_c"] == 31


@pytest.mark.parametrize(
    "name,args",
    [
        ("search_stays", {"city": "goa", "max_per_night": -5}),
        ("search_stays", {"city": "goa", "max_per_night": 50, "admin": True}),
        ("search_stays", {"city": "goa"}),
        ("get_weather", {"city": 3}),
        ("calculator", {"items": [10, -1], "nights": 2}),
        ("delete_everything", {}),
    ],
)
def test_bad_tool_calls_rejected(name, args):
    with pytest.raises(SafeError):
        trip.ToolClient(trip.ToolServer()).call(name, **args)


def test_server_parse_error_and_unknown_method():
    s = trip.ToolServer()
    assert json.loads(s.handle("{nope"))["error"]["code"] == -32700
    assert "error" in json.loads(s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "rm"})))


def test_error_messages_are_scrubbed():
    s = trip.ToolServer()
    resp = s.handle(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_weather", "arguments": {"city": "x", "who@example.com": 1}}}))
    assert "example.com" not in resp


def test_plan_under_500_is_feasible_and_within_budget():
    r = trip.plan_trip("plan my goa trip", 500)
    assert r["feasible"] and r["total"] <= 500


def test_every_kept_claim_has_a_source():
    r = trip.plan_trip("plan", 500)
    assert all(line.endswith("]") and "[" in line for line in r["answer"])


def test_planted_hallucination_is_dropped():
    r = trip.plan_trip("plan", 500, planted_claim="entry is free forever")
    assert r["dropped_ungrounded"] == ["entry is free forever"]
    assert not any("free forever" in a for a in r["answer"])


def test_tight_budget_says_no_instead_of_inventing():
    r = trip.plan_trip("plan", 150)
    assert not r["feasible"] and "no plan fits" in r["answer"][0]


def test_unknown_route_is_reported_not_crashed():
    r = trip.plan_trip("plan", 500, origin="mars")
    assert not r["feasible"]


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_bad_budget_rejected(budget):
    with pytest.raises(SafeError):
        trip.plan_trip("plan", budget)


def test_zero_nights_rejected():
    with pytest.raises(SafeError):
        trip.plan_trip("plan", 500, nights=0)


def test_request_email_is_masked():
    assert "[EMAIL]" in trip.plan_trip("mail me at a.b@example.com", 500)["request"]


def test_logs_are_scrubbed(caplog):
    with caplog.at_level(logging.INFO, logger="aifound.trip"):
        trip.plan_trip("call 555-123-4567", 500)
    assert "555-123-4567" not in caplog.text


def test_agent_loop_stops_at_max_steps():
    st = trip.agent_loop("x", lambda s: ("search", "goa"), {}, trip.ToolClient(trip.ToolServer()), trip.Retriever(trip.KB), trip.Trace(), max_steps=3)
    assert st["out_of_steps"]


def test_context_window_drops_old_turns_and_pins_survive():
    d = trip.demo()["context_window"]
    assert not d["name_survives_plain"] and d["name_survives_pinned"]


def test_context_window_rejects_negative_budget():
    with pytest.raises(SafeError):
        trip.context_window(["a"], -1)


def test_planner_uses_three_workers():
    agents = {s["agent"] for s in trip.plan_trip("plan", 500)["trace"]}
    assert agents == {"planner", "researcher", "booker", "reviewer"}
