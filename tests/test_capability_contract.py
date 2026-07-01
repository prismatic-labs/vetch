"""Provider response-shape contract tests (drift canary).

These lock the exact object shapes each provider SDK returns for tool calls,
using attribute-access mocks that mimic the real pydantic objects (not dicts).
When an upstream SDK changes shape, these fail loudly and point at the provider.
A CI job should additionally run these against *unpinned* latest SDKs nightly.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from vetch.capabilities import (
    accumulate_anthropic_stream_tool_use,
    accumulate_openai_stream_tool_call,
    extract_anthropic_tools_invoked,
    extract_genai_tools_invoked,
    extract_openai_compat_tools_invoked,
    extract_openai_tools_invoked,
    finalize_anthropic_stream_tools,
    finalize_openai_stream_tools,
    reset_capability_state,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def test_openai_chat_completions_shape():
    result = NS(
        choices=[
            NS(
                message=NS(
                    tool_calls=[
                        NS(function=NS(name="get_weather")),
                        NS(function=NS(name="get_time")),
                    ]
                )
            )
        ]
    )
    refs, count = extract_openai_tools_invoked(result)
    assert [r["name"] for r in refs] == ["get_time", "get_weather"]  # sorted
    assert count == 2


def test_openai_no_tool_calls_is_empty_not_none():
    result = NS(choices=[NS(message=NS(tool_calls=None))])
    refs, count = extract_openai_tools_invoked(result)
    assert refs == [] and count == 0


def test_openai_parallel_calls_count_individually_but_dedupe_names():
    result = NS(
        choices=[
            NS(
                message=NS(
                    tool_calls=[
                        NS(function=NS(name="search")),
                        NS(function=NS(name="search")),
                    ]
                )
            )
        ]
    )
    refs, count = extract_openai_tools_invoked(result)
    assert [r["name"] for r in refs] == ["search"]  # names de-duped for set math
    assert count == 2  # but count is raw invocations (parallel calls counted)


def test_anthropic_tool_use_block_shape():
    result = NS(
        content=[
            NS(type="text", text="thinking"),
            NS(type="tool_use", name="lookup"),
        ]
    )
    refs, count = extract_anthropic_tools_invoked(result)
    assert [r["name"] for r in refs] == ["lookup"]
    assert count == 1


def test_anthropic_no_content_is_empty():
    refs, count = extract_anthropic_tools_invoked(NS(content=[]))
    assert refs == [] and count == 0


def test_genai_function_call_parts_shape():
    result = NS(
        candidates=[
            NS(
                content=NS(
                    parts=[
                        NS(function_call=NS(name="fetch")),
                        NS(function_call=None),
                    ]
                )
            )
        ]
    )
    refs, count = extract_genai_tools_invoked(result)
    assert [r["name"] for r in refs] == ["fetch"]
    assert count == 1


def test_ollama_openai_compat_shape():
    # Ollama returns message.tool_calls with dict-shaped function
    result = NS(message=NS(tool_calls=[NS(function={"name": "run"})]))
    refs, count = extract_openai_compat_tools_invoked(result)
    assert [r["name"] for r in refs] == ["run"]
    assert count == 1


def test_malformed_shapes_fail_open():
    # Garbage that resembles nothing -> None or empty, never raises.
    for junk in (NS(), NS(choices=[]), NS(choices=[NS(message=None)]), 42, "x"):
        refs, count = extract_openai_tools_invoked(junk)
        assert refs in (None, []) or isinstance(refs, list)


# --- streaming accumulation contract ---------------------------------------


def test_openai_stream_accumulates_by_index():
    acc: dict = {}
    accumulate_openai_stream_tool_call(
        acc, NS(choices=[NS(delta=NS(tool_calls=[NS(index=0, function=NS(name="search"))]))])
    )
    accumulate_openai_stream_tool_call(
        acc, NS(choices=[NS(delta=NS(tool_calls=[NS(index=1, function=NS(name="lookup"))]))])
    )
    refs, count = finalize_openai_stream_tools(acc, complete=True, error=False)
    assert [r["name"] for r in refs] == ["lookup", "search"]
    assert count == 2


def test_openai_stream_error_or_incomplete_yields_none():
    acc = {0: {"name": "x", "id": ""}}
    assert finalize_openai_stream_tools(acc, complete=False, error=True) == (None, None)
    assert finalize_openai_stream_tools(acc, complete=False, error=False) == (None, None)


def test_openai_stream_no_tool_calls_is_empty():
    assert finalize_openai_stream_tools({}, complete=True, error=False) == ([], 0)


def test_anthropic_stream_accumulates_tool_use_blocks():
    acc: list = []
    accumulate_anthropic_stream_tool_use(
        acc, NS(type="content_block_start", content_block=NS(type="tool_use", name="fetch"))
    )
    # non-tool_use / non-start events are ignored
    accumulate_anthropic_stream_tool_use(acc, NS(type="content_block_delta"))
    accumulate_anthropic_stream_tool_use(
        acc, NS(type="content_block_start", content_block=NS(type="text"))
    )
    refs, count = finalize_anthropic_stream_tools(acc, complete=True, error=False)
    assert [r["name"] for r in refs] == ["fetch"]
    assert count == 1


def test_anthropic_stream_interrupted_yields_none():
    assert finalize_anthropic_stream_tools(["fetch"], complete=False, error=False) == (None, None)
