"""Concurrency stress for shared mutable capability state.

The module-level memo, redaction state, and per-instance SessionStats sets are
touched from concurrent LLM calls. These stress the invariants under threads;
run repeatedly (flakiness is the signal).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from vetch.capabilities import extract_openai_tools_offered, reset_capability_state
from vetch.stats import SessionStats


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def _event(offered, invoked):
    return {
        "usage": {"text": {"input_tokens": 100, "output_tokens": 10}},
        "estimated_cost_usd": 0.01,
        "estimated_cost_input_usd": 0.005,
        "estimated_cost_cache_read_usd": 0.0,
        "cache_read_tokens": 0,
        "model": "gpt-4o",
        "tools_offered": [{"name": n, "kind": "function"} for n in offered],
        "tools_invoked": [{"name": n, "kind": "function"} for n in invoked],
        "tool_schema_tokens": {n: 10 for n in offered},
        "tool_call_count": len(invoked),
    }


@pytest.mark.parametrize("trial", range(3))  # repeat to surface flakiness
def test_concurrent_session_update_no_lost_writes(trial):
    stats = SessionStats()
    n = 400

    def worker(i):
        stats.update(_event([f"t{i % 10}"], [f"t{i % 10}"] if i % 2 == 0 else []))

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(n)))

    assert stats.total_requests == n  # no lost updates
    # union of offered tools is exactly the 10 distinct names
    assert stats.function_tools_offered == {f"t{i}" for i in range(10)}
    s = stats.summary()
    # every never-called name is a real offered name (no corruption / partial writes)
    assert set(s["function_tools_never_called"]) <= stats.function_tools_offered


@pytest.mark.parametrize("trial", range(3))
def test_concurrent_extraction_is_consistent(trial):
    # Each thread uses a fresh tool list object; results must be well-formed.
    def worker(i):
        tools = [{"type": "function", "function": {"name": f"f{i}_{j}"}} for j in range(5)]
        refs, sizes = extract_openai_tools_offered({"tools": tools})
        assert refs is not None and len(refs) == 5
        assert all(r["kind"] == "function" for r in refs)
        return len(refs)

    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(worker, range(200)))
    assert all(r == 5 for r in results)


def test_memo_does_not_deadlock_under_shared_object():
    shared = [{"type": "function", "function": {"name": "shared"}}]

    def worker(_):
        refs, _ = extract_openai_tools_offered({"tools": shared})
        return refs[0]["name"]

    with ThreadPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(worker, range(500)))
    assert set(results) == {"shared"}
