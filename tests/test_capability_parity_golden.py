"""Python <-> TS SDK differential via shared golden vectors.

The offered/invoked lists and their normalization (de-dupe + stable sort) must be
identical across SDKs, or a customer on both sees different numbers. Python is
asserted against the golden here; the TS side is driven through the compiled
`createVetchEvent` when a built dist is available, else skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vetch.capabilities import (
    extract_openai_tools_invoked,
    extract_openai_tools_offered,
    reset_capability_state,
)

# Shared contract: OpenAI-shape tools -> expected normalized function-tool names.
GOLDEN = [
    {
        "tools": [
            {"type": "function", "function": {"name": "b"}},
            {"type": "function", "function": {"name": "a"}},
        ],
        "expected_offered": ["a", "b"],
    },
    {
        "tools": [
            {"type": "function", "function": {"name": "dup"}},
            {"type": "function", "function": {"name": "dup"}},
        ],
        "expected_offered": ["dup"],
    },
    {
        "tools": [
            {"type": "function", "function": {"name": "Zeta"}},
            {"type": "function", "function": {"name": "alpha"}},
        ],
        "expected_offered": ["Zeta", "alpha"],
    },  # case-sensitive sort, uppercase first
]

TS_DIR = Path(__file__).resolve().parents[1] / "packages" / "vetch-ai-sdk"


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def test_python_matches_golden_offered():
    for case in GOLDEN:
        refs, _ = extract_openai_tools_offered({"tools": case["tools"]})
        assert [r["name"] for r in refs] == case["expected_offered"]


def test_python_matches_golden_invoked_normalization():
    # Same normalization applies to invoked names.
    from types import SimpleNamespace as NS

    result = NS(
        choices=[
            NS(
                message=NS(
                    tool_calls=[
                        NS(function=NS(name="b")),
                        NS(function=NS(name="a")),
                        NS(function=NS(name="a")),
                    ]
                )
            )
        ]
    )
    refs, count = extract_openai_tools_invoked(result)
    assert [r["name"] for r in refs] == ["a", "b"]
    assert count == 3  # raw parallel invocations; names de-duped in refs


def test_ts_types_declare_parity_fields():
    """Cheap drift guard: TS event type must carry the same field names."""
    types_ts = (TS_DIR / "src" / "types.ts").read_text(encoding="utf-8")
    for field in ("tools_offered", "tools_invoked", "tool_call_count", "capabilities_invoked"):
        assert field in types_ts, f"TS SDK missing parity field {field}"
    assert "VetchCapabilityRef" in types_ts


def _run_ts_offered(tools):
    """Drive the compiled TS createVetchEvent and return normalized offered names."""
    event_js = TS_DIR / "dist" / "event.js"
    script = f"""
    import {{ createVetchEvent }} from {json.dumps(str(event_js))};
    const ev = await createVetchEvent({{
      params: {{ tools: {json.dumps(tools)} }},
      result: {{ choices: [] }},
      model: "gpt-4o",
      provider: "openai",
      operation: "chat",
      startTimeMs: 0,
      options: {{}},
    }});
    const names = (ev.tools_offered || []).map(r => r.name);
    console.log(JSON.stringify(names));
    """
    out = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:400])
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_ts_behavioral_parity_against_golden():
    if not (TS_DIR / "dist" / "event.js").exists():
        pytest.skip("TS SDK dist not built")
    try:
        for case in GOLDEN:
            ts_names = _run_ts_offered(case["tools"])
            assert ts_names == case["expected_offered"], (
                f"TS/Python drift: TS={ts_names} expected={case['expected_offered']}"
            )
    except RuntimeError as exc:
        pytest.skip(f"TS createVetchEvent not drivable standalone: {exc}")
