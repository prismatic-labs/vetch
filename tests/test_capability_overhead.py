"""Hot-path overhead gate: prove memoization avoids re-extracting identical tools.

Instrumentation on every LLM call must not re-walk/re-tokenize a stable tool
schema each request. The deterministic proof is a call counter on the tokenizer;
a lenient wall-clock budget guards against gross regressions.
"""

from __future__ import annotations

import time

import pytest

import vetch.capabilities as cap
from vetch.capabilities import extract_openai_tools_offered, reset_capability_state


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def _tools(n):
    return [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "x" * 200,
                "parameters": {"type": "object", "properties": {f"p{i}": {"type": "string"}}},
            },
        }
        for i in range(n)
    ]


def test_memoization_extracts_tokens_once_for_stable_object(monkeypatch):
    calls = {"n": 0}
    real = cap._estimate_tool_json_tokens

    def counting(obj):
        calls["n"] += 1
        return real(obj)

    monkeypatch.setattr("vetch.capabilities._estimate_tool_json_tokens", counting)

    tools = _tools(20)  # one stable list object, reused across "requests"
    first, _ = extract_openai_tools_offered({"tools": tools})
    after_first = calls["n"]
    assert after_first == 20  # tokenized each tool once

    # 50 more "calls" with the same object -> zero additional tokenization work
    for _ in range(50):
        again, _ = extract_openai_tools_offered({"tools": tools})
        assert again == first
    assert calls["n"] == after_first  # memo hit, no recompute


def test_memoized_call_is_much_cheaper_than_cold():
    tools = _tools(30)

    t0 = time.perf_counter()
    extract_openai_tools_offered({"tools": tools})  # cold: tokenizes
    cold = time.perf_counter() - t0

    # warm calls should be dominated by a dict lookup
    t1 = time.perf_counter()
    for _ in range(200):
        extract_openai_tools_offered({"tools": tools})
    warm_avg = (time.perf_counter() - t1) / 200

    # Generous budget: a memoized call is well under 200us and far cheaper than cold.
    assert warm_avg < 2e-3
    assert warm_avg < cold  # memo pays off


def test_no_tools_path_is_trivial():
    # The common no-tools call must short-circuit without work.
    t0 = time.perf_counter()
    for _ in range(1000):
        refs, sizes = extract_openai_tools_offered({})
        assert refs is None and sizes is None
    per_call = (time.perf_counter() - t0) / 1000
    assert per_call < 1e-4
