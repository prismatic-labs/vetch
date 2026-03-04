"""Internal security utilities for sanitizing sensitive data.

This module provides functions to redact secrets, API keys, and other
sensitive information from error messages, logs, and URLs.

IMPORTANT: This is an internal module. Do not import directly in user code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Patterns for common secrets
_SECRET_PATTERNS = [
    # API keys (OpenAI-style sk- prefix, various lengths)
    (re.compile(r'\b(sk-[a-zA-Z0-9]{10,})', re.IGNORECASE), '[REDACTED_API_KEY]'),
    (
        re.compile(r'\b(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})', re.IGNORECASE),
        r'\1[REDACTED_API_KEY]',
    ),

    # Bearer tokens
    (re.compile(r'\b(bearer\s+)([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE), r'\1[REDACTED_TOKEN]'),

    # AWS keys
    (re.compile(r'\b(AKIA[0-9A-Z]{16})', re.IGNORECASE), '[REDACTED_AWS_KEY]'),
    (re.compile(r'\b([a-zA-Z0-9+/]{40})', re.IGNORECASE), '[REDACTED_SECRET]'),

    # Passwords
    (
        re.compile(r'\b(password["\']?\s*[:=]\s*["\']?)([^\s"\']{6,})', re.IGNORECASE),
        r'\1[REDACTED_PASSWORD]',
    ),
    (
        re.compile(r'\b(passwd["\']?\s*[:=]\s*["\']?)([^\s"\']{6,})', re.IGNORECASE),
        r'\1[REDACTED_PASSWORD]',
    ),

    # Connection strings
    (re.compile(r'(://[^:@]*:)([^@]+)(@)', re.IGNORECASE), r'\1[REDACTED]\3'),

    # JWT tokens (rough pattern)
    (
        re.compile(r'\beyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+', re.IGNORECASE),
        '[REDACTED_JWT]',
    ),
]

# Patterns for file paths (more conservative - only redact home directories)
_PATH_PATTERNS = [
    (re.compile(r'/Users/[^/\s]+'), '/Users/[USERNAME]'),
    (re.compile(r'/home/[^/\s]+'), '/home/[USERNAME]'),
    (re.compile(r'C:\\Users\\[^\\s]+', re.IGNORECASE), r'C:\\Users\\[USERNAME]'),
]


def sanitize_exception_message(message: str, max_length: int = 500) -> str:
    """Sanitize an exception message to remove secrets and sensitive paths.

    This function is designed to make error messages safe for logging,
    telemetry, and URL encoding (e.g., GitHub issue links).

    Redacts:
    - API keys (sk-*, AKIA*, etc.)
    - Bearer tokens
    - Passwords in connection strings
    - User home directories in file paths
    - JWT tokens

    Args:
        message: The exception message to sanitize.
        max_length: Maximum length of the sanitized message (default: 500).

    Returns:
        Sanitized message safe for public logging.

    Example:
        >>> sanitize_exception_message("Invalid API key: sk-abc123def456")
        'Invalid API key: [REDACTED_API_KEY]'

        >>> sanitize_exception_message("Connection failed: postgresql://user:pass@host/db")
        'Connection failed: postgresql://user:[REDACTED]@host/db'
    """
    if not message:
        return message

    sanitized = str(message)

    # Apply secret patterns
    for pattern, replacement in _SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    # Apply path patterns (less aggressive - only home dirs)
    for pattern, replacement in _PATH_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"

    return sanitized


def sanitize_traceback(tb_string: str, remove_local_vars: bool = True) -> str:
    """Sanitize a traceback string to remove local variables and secrets.

    Args:
        tb_string: The full traceback string from traceback.format_exc().
        remove_local_vars: Whether to remove lines showing local variables.

    Returns:
        Sanitized traceback with secrets redacted.
    """
    lines = tb_string.splitlines()
    sanitized_lines = []

    for line in lines:
        # Remove local variable values (lines with " = " that aren't file paths)
        if remove_local_vars:
            is_file_line = line.strip().startswith("File ")
            is_traceback = line.startswith("Traceback")
            if " = " in line and not is_file_line and not is_traceback:
                continue

        # Sanitize each line for secrets
        sanitized_line = sanitize_exception_message(line, max_length=1000)
        sanitized_lines.append(sanitized_line)

    return "\n".join(sanitized_lines)


def is_safe_output_path(output_path: Path, allowed_roots: list[Path]) -> bool:
    """Check if an output path is safe (no traversal outside allowed roots).

    This is a defense-in-depth check against path traversal attacks.
    Use this when accepting user-provided file paths for output.

    Args:
        output_path: The path to validate (should be resolved/absolute).
        allowed_roots: List of allowed root directories.

    Returns:
        True if the path is within one of the allowed roots, False otherwise.

    Example:
        >>> from pathlib import Path
        >>> cwd = Path.cwd()
        >>> is_safe_output_path(cwd / "output.txt", [cwd])
        True
        >>> is_safe_output_path(Path("/etc/passwd"), [cwd])
        False
    """
    # Resolve to absolute path (follows symlinks, removes ..)
    try:
        resolved = output_path.resolve()
    except (OSError, RuntimeError):
        return False

    # Check if resolved path is within any allowed root
    for root in allowed_roots:
        try:
            root_resolved = root.resolve()
            # Check if output_path is relative to root
            resolved.relative_to(root_resolved)
            return True
        except (ValueError, OSError, RuntimeError):
            continue

    return False


def sanitize_for_url(text: str, max_length: int = 1000) -> str:
    """Sanitize text before URL encoding (for query parameters).

    Applies aggressive sanitization suitable for URLs that may be logged
    in browser history, server logs, and proxies.

    Args:
        text: Text to sanitize.
        max_length: Maximum length before truncation.

    Returns:
        Sanitized text safe for URL encoding.
    """
    # First apply standard sanitization
    sanitized = sanitize_exception_message(text, max_length=max_length)

    # Additional URL-specific redactions
    # Redact anything that looks like a hash or ID (could be session tokens)
    sanitized = re.sub(r'\b[a-f0-9]{32,}\b', '[REDACTED_HASH]', sanitized, flags=re.IGNORECASE)

    return sanitized


def get_safe_exception_context(e: Exception, max_message_length: int = 200) -> dict[str, Any]:
    """Get a safe context dictionary from an exception for telemetry.

    Args:
        e: The exception to extract context from.
        max_message_length: Max length for the exception message.

    Returns:
        Dictionary with safe exception context.
    """
    return {
        "type": type(e).__name__,
        "message": sanitize_exception_message(str(e), max_length=max_message_length),
        "module": type(e).__module__,
    }
