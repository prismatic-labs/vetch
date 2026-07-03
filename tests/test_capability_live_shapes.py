"""Drift-canary sentinels: extraction against REAL provider SDK objects.

Unlike test_capability_contract.py (which uses SimpleNamespace mocks), these
build genuine pydantic response objects via ``model_validate`` so a breaking
upstream schema change fails loudly. They skip when the SDK isn't installed, so
locally they no-op; the nightly drift-canary workflow installs the latest
unpinned SDKs and runs them for real.
"""

from __future__ import annotations

import pytest

from vetch.capabilities import (
    extract_anthropic_tools_invoked,
    extract_genai_tools_invoked,
    extract_openai_compat_tools_invoked,
    extract_openai_tools_invoked,
    reset_capability_state,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_capability_state()
    yield
    reset_capability_state()


def test_openai_real_chatcompletion_shape():
    pytest.importorskip("openai")
    from openai.types.chat import ChatCompletion

    completion = ChatCompletion.model_validate(
        {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"},
                            },
                            {
                                "id": "2",
                                "type": "function",
                                "function": {"name": "get_time", "arguments": "{}"},
                            },
                        ],
                    },
                }
            ],
        }
    )
    refs, count = extract_openai_tools_invoked(completion)
    assert [r["name"] for r in refs] == ["get_time", "get_weather"]
    assert count == 2


def test_anthropic_real_message_shape():
    pytest.importorskip("anthropic")
    from anthropic.types import Message

    msg = Message.model_validate(
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    refs, count = extract_anthropic_tools_invoked(msg)
    assert [r["name"] for r in refs] == ["lookup"]
    assert count == 1


def test_genai_real_response_shape():
    pytest.importorskip("google.genai")
    from google.genai import types as gt

    resp = gt.GenerateContentResponse(
        candidates=[
            gt.Candidate(
                content=gt.Content(
                    parts=[gt.Part(function_call=gt.FunctionCall(name="fetch"))]
                )
            )
        ]
    )
    refs, count = extract_genai_tools_invoked(resp)
    assert [r["name"] for r in refs] == ["fetch"]
    assert count == 1


def test_ollama_real_response_shape():
    pytest.importorskip("ollama")
    from ollama._types import ChatResponse, Message

    resp = ChatResponse(
        model="llama3.1",
        message=Message(
            role="assistant",
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(name="run", arguments={})
                )
            ],
        ),
    )
    refs, count = extract_openai_compat_tools_invoked(resp)
    assert [r["name"] for r in refs] == ["run"]
    assert count == 1
