"""Long-running soak: memory boundedness + concurrency under sustained load.

Gated behind VETCH_SOAK=1 so it never runs in the normal suite. The
capability-perf-soak workflow sets it and scales the iteration count. Verifies
the per-session rollup stays memory-bounded (sets capped by cardinality) and
survives sustained concurrent updates without corruption.
"""

from __future__ import annotations

import gc
import os
import tracemalloc
from concurrent.futures import ThreadPoolExecutor

import pytest

from vetch.capabilities import reset_capability_state
from vetch.stats import SessionStats

pytestmark = pytest.mark.skipif(
    os.environ.get("VETCH_SOAK") != "1", reason="set VETCH_SOAK=1 to run soak"
)

ITERS = int(os.environ.get("VETCH_SOAK_ITERS", "50000"))
DISTINCT = int(os.environ.get("VETCH_SOAK_DISTINCT", "2000"))


def _event(i):
    name = f"tool_{i % DISTINCT}"
    return {
        "usage": {"text": {"input_tokens": 500, "output_tokens": 20}},
        "estimated_cost_usd": 0.01,
        "estimated_cost_input_usd": 0.005,
        "estimated_cost_cache_read_usd": 0.0,
        "cache_read_tokens": 0,
        "model": "gpt-4o",
        "tools_offered": [{"name": name, "kind": "function"}],
        "tools_invoked": [] if i % 3 else [{"name": name, "kind": "function"}],
        "tool_schema_tokens": {name: 12},
        "tool_call_count": 0 if i % 3 else 1,
    }


def test_session_rollup_memory_is_bounded():
    reset_capability_state()
    stats = SessionStats()
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]

    for i in range(ITERS):
        stats.update(_event(i))

    gc.collect()
    current = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()

    growth_mb = (current - base) / 1e6
    # Sets are capped by the cardinality limit, so growth must stay bounded even
    # over tens of thousands of updates with many distinct tool names.
    assert growth_mb < 50, f"rollup grew {growth_mb:.1f} MB over {ITERS} updates"
    assert stats.total_requests == ITERS


def test_concurrent_updates_preserve_capability_invariants():
    reset_capability_state()
    stats = SessionStats()

    def worker(i):
        stats.update(_event(i))

    with ThreadPoolExecutor(max_workers=32) as ex:
        list(ex.map(worker, range(ITERS)))

    # Concurrency check: no updates are lost under the lock.
    assert stats.total_requests == ITERS
    s = stats.summary()
    # Cardinality-bound invariant: every never-called tool must be one we
    # actually tracked as offered. This holds regardless of threading (it
    # reproduces single-threaded once DISTINCT exceeds the cardinality cap), so
    # a failure here points at the rollup's set bounding, not a data race.
    assert set(s["function_tools_never_called"]) <= stats.function_tools_offered
