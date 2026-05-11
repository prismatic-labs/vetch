"""Helpers for Google GenAI and Vertex AI usage metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, cast

from vetch.schema import Usage

_MODALITIES = ("image", "audio", "video")
_INPUT_DETAIL_FIELDS = ("prompt_tokens_details", "tool_use_prompt_tokens_details")
_OUTPUT_DETAIL_FIELDS = ("candidates_tokens_details",)


def _positive_int(value: Any) -> int:
    """Return ``value`` as a positive int, ignoring mocks and non-numeric sentinels."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return 0


def _positive_float(value: Any) -> float:
    """Return ``value`` as a positive float, ignoring mocks and non-numeric sentinels."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return 0.0


def _first_positive_int(obj: Any, names: tuple[str, ...]) -> int:
    for name in names:
        value = _positive_int(getattr(obj, name, 0))
        if value:
            return value
    return 0


def _first_positive_float(obj: Any, names: tuple[str, ...]) -> float:
    for name in names:
        value = _positive_float(getattr(obj, name, 0.0))
        if value:
            return value
    return 0.0


def extract_thinking_tokens(usage_metadata: Any) -> int:
    """Extract Gemini thinking tokens from current and legacy SDK spellings."""
    return _first_positive_int(
        usage_metadata,
        ("thoughts_token_count", "thought_token_count", "thoughtsTokenCount"),
    )


def build_google_usage(usage_metadata: Any) -> Usage:
    """Build a Vetch Usage object from Google usage metadata."""
    usage_dict: Usage = {
        "text": {
            "input_tokens": _positive_int(
                getattr(usage_metadata, "prompt_token_count", 0)
            ),
            "output_tokens": _positive_int(
                getattr(usage_metadata, "candidates_token_count", 0)
            ),
            "total_tokens": _positive_int(
                getattr(usage_metadata, "total_token_count", 0)
            ),
        }
    }

    thinking_tokens = extract_thinking_tokens(usage_metadata)
    if thinking_tokens:
        usage_dict["reasoning"] = {
            "input_tokens": 0,
            "output_tokens": thinking_tokens,
            "total_tokens": thinking_tokens,
        }

    _add_modality_usage(usage_dict, usage_metadata)
    return usage_dict


def _iter_details(usage_metadata: Any, field_names: tuple[str, ...]) -> Iterable[Any]:
    for field_name in field_names:
        details = getattr(usage_metadata, field_name, None)
        if isinstance(details, (list, tuple)):
            yield from details


def _normalise_modality(raw: Any) -> Literal["image", "audio", "video", "text", "other"]:
    if raw is None:
        return "other"
    if hasattr(raw, "name"):
        value = str(raw.name)
    elif hasattr(raw, "value"):
        value = str(raw.value)
    else:
        value = str(raw)
    value = value.lower()
    if "image" in value or "vision" in value:
        return "image"
    if "audio" in value:
        return "audio"
    if "video" in value:
        return "video"
    if "text" in value:
        return "text"
    return "other"


def _add_detail_tokens(
    totals: dict[str, dict[str, int]],
    details: Iterable[Any],
    direction: Literal["input_tokens", "output_tokens"],
) -> None:
    for detail in details:
        modality = _normalise_modality(getattr(detail, "modality", None))
        if modality not in _MODALITIES:
            continue
        modality_key = cast(Literal["image", "audio", "video"], modality)
        token_count = _positive_int(getattr(detail, "token_count", 0))
        if token_count:
            totals[modality_key][direction] += token_count


def _add_modality_usage(usage_dict: Usage, usage_metadata: Any) -> None:
    totals = {
        modality: {"input_tokens": 0, "output_tokens": 0}
        for modality in _MODALITIES
    }
    _add_detail_tokens(
        totals,
        _iter_details(usage_metadata, _INPUT_DETAIL_FIELDS),
        "input_tokens",
    )
    _add_detail_tokens(
        totals,
        _iter_details(usage_metadata, _OUTPUT_DETAIL_FIELDS),
        "output_tokens",
    )

    direct_fields = {
        "image": (
            ("image_token_count", "prompt_image_token_count", "imageTokenCount"),
            ("image_output_token_count", "candidates_image_token_count"),
        ),
        "audio": (
            ("audio_token_count", "prompt_audio_token_count", "audioTokenCount"),
            ("audio_output_token_count", "candidates_audio_token_count"),
        ),
        "video": (
            ("video_token_count", "prompt_video_token_count", "videoTokenCount"),
            ("video_output_token_count", "candidates_video_token_count"),
        ),
    }
    for modality, (input_names, output_names) in direct_fields.items():
        if totals[modality]["input_tokens"] == 0:
            totals[modality]["input_tokens"] = _first_positive_int(
                usage_metadata, input_names
            )
        if totals[modality]["output_tokens"] == 0:
            totals[modality]["output_tokens"] = _first_positive_int(
                usage_metadata, output_names
            )

    for modality, token_totals in totals.items():
        input_tokens = token_totals["input_tokens"]
        output_tokens = token_totals["output_tokens"]
        if not input_tokens and not output_tokens:
            continue
        modality_usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if modality == "image":
            usage_dict["image"] = cast(Any, modality_usage)
        elif modality == "audio":
            usage_dict["audio"] = cast(Any, modality_usage)
        elif modality == "video":
            usage_dict["video"] = cast(Any, modality_usage)

    if usage_dict.get("audio"):
        audio = cast(dict[str, Any], usage_dict["audio"])
        input_seconds = _first_positive_float(
            usage_metadata,
            ("audio_duration_seconds", "prompt_audio_duration_seconds"),
        )
        output_seconds = _first_positive_float(
            usage_metadata,
            ("audio_output_duration_seconds", "candidates_audio_duration_seconds"),
        )
        if input_seconds:
            audio["input_seconds"] = input_seconds
        if output_seconds:
            audio["output_seconds"] = output_seconds

    if usage_dict.get("video"):
        video = cast(dict[str, Any], usage_dict["video"])
        input_seconds = _first_positive_float(
            usage_metadata,
            ("video_duration_seconds", "prompt_video_duration_seconds"),
        )
        output_seconds = _first_positive_float(
            usage_metadata,
            ("video_output_duration_seconds", "candidates_video_duration_seconds"),
        )
        if input_seconds:
            video["input_seconds"] = input_seconds
        if output_seconds:
            video["output_seconds"] = output_seconds
