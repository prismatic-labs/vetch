"""Context variable management for thread-safe tracking.

This module provides ContextVar-based storage for:
- Active tracking context (for nested wrapping)
- Captured metadata from LLM calls
- Configuration inheritance

ContextVars are safe for threading and asyncio.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vetch.schema import EnergyOverride, Usage

# Active tracking context stack
_active_context: ContextVar[TrackingContext | None] = ContextVar(
    "vetch_active_context", default=None
)


@dataclass
class CapturedCall:
    """Metadata captured from an LLM API call.

    This stores everything we need from the response without
    storing any prompt/completion content.
    """

    model: str
    provider: str
    usage: Usage | None = None
    is_stream: bool = False
    is_embedding: bool = False  # True if embedding generation (not text completion)
    accumulated_chars: int = 0
    complete: bool = True
    error: bool = False
    error_type: str | None = None
    warnings: list[str] = field(default_factory=list)
    # Prompt caching metadata
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    # Streaming token estimation (tiktoken or script-aware char ratio)
    accumulated_tik_tokens: int = 0
    content_type_hint: str = "en"
    # Privacy-safe output diagnostics: counts/status only, never content.
    visible_output_chars: int | None = None
    finish_reason: str | None = None
    requested_max_tokens: int | None = None


@dataclass
class TrackingContext:
    """Thread-safe context for a single wrap() invocation.

    Stores configuration and captures LLM call metadata.
    Supports nesting through parent reference.
    """

    region: str | None = None
    tags: dict[str, str] | None = None
    energy_override: EnergyOverride | None = None
    parent: TrackingContext | None = None

    # Captured during the wrapped call
    captured_call: CapturedCall | None = None
    warnings: list[str] = field(default_factory=list)

    # Token for context restoration
    _token: Token[TrackingContext | None] | None = field(default=None, repr=False)

    def __enter__(self) -> TrackingContext:
        """Enter context and set as active."""
        # Get current context as parent
        self.parent = _active_context.get()

        # Inherit region from parent if not specified
        if self.region is None and self.parent is not None:
            self.region = self.parent.region

        # Merge tags (child overrides parent on conflict)
        if self.parent is not None and self.parent.tags:
            merged_tags = dict(self.parent.tags)
            if self.tags:
                merged_tags.update(self.tags)
            self.tags = merged_tags

        # Set this context as active
        self._token = _active_context.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context and restore previous."""
        if self._token is not None:
            _active_context.reset(self._token)
            self._token = None

    def capture(
        self,
        model: str,
        provider: str,
        usage: Usage | None = None,
        is_stream: bool = False,
        is_embedding: bool = False,
        accumulated_chars: int = 0,
        complete: bool = True,
        error: bool = False,
        error_type: str | None = None,
        warnings: list[str] | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        accumulated_tik_tokens: int = 0,
        content_type_hint: str = "en",
        visible_output_chars: int | None = None,
        finish_reason: str | None = None,
        requested_max_tokens: int | None = None,
    ) -> None:
        """Capture metadata from an LLM call.

        Args:
            model: Model identifier from the response.
            provider: Provider name (openai, vertexai).
            usage: Token usage from response.
            is_stream: Whether this was a streaming call.
            is_embedding: Whether this is an embedding generation call.
            accumulated_chars: Character count for streams.
            complete: Whether the call completed successfully.
            error: Whether an error occurred.
            error_type: Exception class name if error.
            warnings: Diagnostic warnings.
            cache_read_tokens: Tokens read from prompt cache (cost savings).
            cache_creation_tokens: Tokens written to prompt cache (extra cost).
            visible_output_chars: Number of visible output characters, without content.
            finish_reason: Provider finish status when exposed.
            requested_max_tokens: Output-token cap requested by the caller.
        """
        self.captured_call = CapturedCall(
            model=model,
            provider=provider,
            usage=usage,
            is_stream=is_stream,
            is_embedding=is_embedding,
            accumulated_chars=accumulated_chars,
            complete=complete,
            error=error,
            error_type=error_type,
            warnings=warnings or [],
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            accumulated_tik_tokens=accumulated_tik_tokens,
            content_type_hint=content_type_hint,
            visible_output_chars=visible_output_chars,
            finish_reason=finish_reason,
            requested_max_tokens=requested_max_tokens,
        )


def get_active_context() -> TrackingContext | None:
    """Get the currently active tracking context.

    Returns:
        The active TrackingContext, or None if not in a wrap() block.
    """
    return _active_context.get()


def has_active_context() -> bool:
    """Check if there's an active tracking context.

    Returns:
        True if inside a wrap() block, False otherwise.
    """
    return _active_context.get() is not None
