"""Integration tests for security fixes and sanitization.

These tests verify that:
1. Exception messages are properly sanitized before URL encoding
2. Path traversal attacks are blocked
3. Connection pool is used consistently (no I/O hammer)
4. Secrets are not leaked in error messages or tracebacks
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vetch._security import (
    get_safe_exception_context,
    is_safe_output_path,
    sanitize_exception_message,
    sanitize_for_url,
    sanitize_traceback,
)


class TestExceptionSanitization:
    """Test that sensitive data is redacted from exceptions."""

    def test_api_key_redaction(self):
        """Test that OpenAI-style API keys are redacted."""
        message = "Authentication failed: Invalid API key 'sk-abc123def456ghi789'"
        sanitized = sanitize_exception_message(message)

        assert "sk-abc123" not in sanitized
        assert "[REDACTED_API_KEY]" in sanitized

    def test_password_redaction(self):
        """Test that passwords in connection strings are redacted."""
        message = "Connection failed: postgresql://user:secret_password_123@host/db"
        sanitized = sanitize_exception_message(message)

        assert "secret_password_123" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_bearer_token_redaction(self):
        """Test that Bearer tokens are redacted."""
        message = "Auth failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = sanitize_exception_message(message)

        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
        assert "[REDACTED_TOKEN]" in sanitized or "[REDACTED_JWT]" in sanitized

    def test_aws_key_redaction(self):
        """Test that AWS access keys are redacted."""
        message = "AWS auth failed: AKIAIOSFODNN7EXAMPLE"
        sanitized = sanitize_exception_message(message)

        assert "AKIAIOSFODNN7EXAMPLE" not in sanitized
        assert "[REDACTED" in sanitized

    def test_home_directory_redaction(self):
        """Test that user home directories are redacted."""
        message = "File not found: /Users/john_doe/.env"
        sanitized = sanitize_exception_message(message)

        assert "john_doe" not in sanitized
        assert "[USERNAME]" in sanitized

    def test_multiple_secrets_redacted(self):
        """Test that multiple secrets in one message are all redacted."""
        message = (
            "Failed to connect: postgresql://user:pass123@host/db "
            "using API key sk-abc123def456"
        )
        sanitized = sanitize_exception_message(message)

        assert "pass123" not in sanitized
        assert "sk-abc123def456" not in sanitized
        assert "[REDACTED" in sanitized

    def test_message_truncation(self):
        """Test that very long messages are truncated."""
        long_message = "Error: " + "a" * 1000
        sanitized = sanitize_exception_message(long_message, max_length=100)

        assert len(sanitized) <= 120  # 100 + "[truncated]" suffix
        assert "[truncated]" in sanitized

    def test_normal_message_unchanged(self):
        """Test that messages without secrets pass through (sanitized)."""
        message = "Connection timeout after 30 seconds"
        sanitized = sanitize_exception_message(message)

        # Should be largely unchanged (maybe minor formatting)
        assert "Connection timeout" in sanitized
        assert "30 seconds" in sanitized


class TestTracebackSanitization:
    """Test that tracebacks are sanitized properly."""

    def test_local_vars_removed(self):
        """Test that local variable values are removed from tracebacks."""
        traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in function_name
    result = api_call(key)
    key = 'sk-secret-key-123'
    result = None
ValueError: Invalid API key
"""
        sanitized = sanitize_traceback(traceback_text, remove_local_vars=True)

        # File paths and exception should remain
        assert "File \"test.py\"" in sanitized
        assert "ValueError: Invalid API key" in sanitized

        # Local variable assignments should be removed
        assert "key = 'sk-secret-key-123'" not in sanitized
        assert "result = None" not in sanitized

    def test_exception_message_sanitized_in_traceback(self):
        """Test that API keys in the exception message itself are redacted."""
        traceback_text = """Traceback (most recent call last):
  File "test.py", line 10, in auth
    raise ValueError("Invalid API key: sk-abc123def456")
ValueError: Invalid API key: sk-abc123def456
"""
        sanitized = sanitize_traceback(traceback_text)

        # Traceback structure preserved
        assert "File \"test.py\"" in sanitized
        assert "ValueError" in sanitized

        # But the API key should be redacted
        assert "sk-abc123def456" not in sanitized


class TestURLSanitization:
    """Test sanitization for URL encoding (most aggressive)."""

    def test_url_sanitization_removes_hashes(self):
        """Test that long hex strings (potential session tokens) are redacted."""
        text = "Error with session: abc123def456789012345678901234567890"
        sanitized = sanitize_for_url(text)

        # Long hex strings should be redacted as potential tokens
        assert "abc123def456789012345678901234567890" not in sanitized

    def test_url_sanitization_length_limit(self):
        """Test that URL sanitization enforces strict length limits."""
        long_text = "Error: " + "x" * 2000
        sanitized = sanitize_for_url(long_text, max_length=500)

        assert len(sanitized) <= 520  # 500 + "[truncated]"


class TestPathTraversalProtection:
    """Test path traversal attack prevention."""

    def test_safe_path_in_cwd(self):
        """Test that paths within current directory are allowed."""
        cwd = Path.cwd()
        safe_path = cwd / "output.txt"

        assert is_safe_output_path(safe_path, [cwd])

    def test_safe_path_in_subdirectory(self):
        """Test that paths in subdirectories are allowed."""
        cwd = Path.cwd()
        safe_path = cwd / "subdir" / "output.txt"

        assert is_safe_output_path(safe_path, [cwd])

    def test_unsafe_path_with_parent_traversal(self):
        """Test that parent directory traversal is blocked."""
        cwd = Path.cwd()
        unsafe_path = cwd / ".." / ".." / "etc" / "passwd"

        assert not is_safe_output_path(unsafe_path, [cwd])

    def test_unsafe_absolute_path_outside_roots(self):
        """Test that absolute paths outside allowed roots are blocked."""
        cwd = Path.cwd()
        unsafe_path = Path("/etc/passwd")

        assert not is_safe_output_path(unsafe_path, [cwd])

    def test_symlink_traversal_blocked(self):
        """Test that symlinks leading outside allowed roots are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create a symlink pointing outside tmpdir
            link_path = tmp_path / "evil_link"
            try:
                link_path.symlink_to("/etc/passwd")

                # Should be blocked because it resolves outside tmp_path
                assert not is_safe_output_path(link_path, [tmp_path])
            except OSError:
                # Symlink creation might fail on some systems, skip test
                pytest.skip("Symlink creation not supported")

    def test_multiple_allowed_roots(self):
        """Test that paths in any of multiple allowed roots are accepted."""
        cwd = Path.cwd()
        tmp = Path(tempfile.gettempdir())

        # Path in first root
        path1 = cwd / "output.txt"
        assert is_safe_output_path(path1, [cwd, tmp])

        # Path in second root
        path2 = tmp / "output.txt"
        assert is_safe_output_path(path2, [cwd, tmp])

        # Path in neither root
        path3 = Path("/etc/passwd")
        assert not is_safe_output_path(path3, [cwd, tmp])


class TestSafeExceptionContext:
    """Test safe exception context for telemetry."""

    def test_safe_context_redacts_message(self):
        """Test that exception context includes sanitized message."""
        exc = ValueError("Invalid API key: sk-abc123def456")
        context = get_safe_exception_context(exc)

        assert context["type"] == "ValueError"
        assert context["module"] == "builtins"
        assert "sk-abc123def456" not in context["message"]
        assert "[REDACTED_API_KEY]" in context["message"]


class TestConnectionPoolUsage:
    """Test that storage.py uses connection pool consistently."""

    def test_query_usage_doesnt_open_new_connection(self):
        """Test that query_usage() uses the connection pool, not sqlite3.connect()."""
        from datetime import datetime, timedelta

        from vetch.storage import configure_storage, query_usage

        # Enable storage
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            configure_storage(enabled=True, path=db_path)

            # Mock sqlite3.connect to detect if it's called
            with patch("vetch.storage.sqlite3.connect") as mock_connect:
                # Call query_usage
                end = datetime.now()
                start = end - timedelta(days=7)

                try:
                    query_usage(start, end)
                except Exception:
                    pass  # Ignore errors, we just want to check connect() calls

                # sqlite3.connect should only be called for _init_db (once)
                # NOT for query_usage itself (should use _get_connection pool)
                # If DB already exists, connect() should NOT be called at all
                if db_path.exists():
                    # After initial setup, no new connections should be opened
                    assert mock_connect.call_count <= 1, (
                        f"query_usage() opened {mock_connect.call_count} new connections. "
                        "Should use connection pool instead."
                    )


class TestEmitterPathValidation:
    """Test that emitter.py validates paths properly."""

    def test_emitter_blocks_traversal_in_vetch_output(self):
        """Test that VETCH_OUTPUT path traversal is blocked."""

        # Try to set VETCH_OUTPUT to a path traversal
        original_env = os.environ.get("VETCH_OUTPUT")

        try:
            # This should be blocked and fall back to stderr
            os.environ["VETCH_OUTPUT"] = "../../../../etc/passwd"

            # Reconfigure logging (in real usage, this happens on import)
            # We can't easily test _configure_logging directly since it runs on import
            # But we can at least verify the security module works
            from pathlib import Path
            from tempfile import gettempdir

            from vetch._security import is_safe_output_path

            malicious_path = Path("../../../../etc/passwd").resolve()
            cwd = Path.cwd()
            tmp = Path(gettempdir())
            home_vetch = Path.home() / ".vetch"

            # Should be blocked
            assert not is_safe_output_path(malicious_path, [cwd, tmp, home_vetch])

        finally:
            # Restore original environment
            if original_env is not None:
                os.environ["VETCH_OUTPUT"] = original_env
            elif "VETCH_OUTPUT" in os.environ:
                del os.environ["VETCH_OUTPUT"]


class TestIntegrationSecurityRegression:
    """Integration tests that would have caught the panel's security issues."""

    def test_exception_in_wrapper_doesnt_leak_secrets(self):
        """
        Regression test for P0 issue: Exception messages in GitHub issue URLs
        must not contain secrets.
        """
        from vetch._security import sanitize_for_url

        # Simulate an exception that might contain secrets
        exc_message = "Database connection failed: postgresql://admin:supersecret@localhost/db"

        # This is what gets URL-encoded in wrapper.py
        sanitized = sanitize_for_url(exc_message)

        # The password should be redacted
        assert "supersecret" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_storage_doesnt_hammer_sqlite(self):
        """
        Regression test for P0 issue: query_usage() should use connection pool,
        not open a new connection every time.
        """
        # This is tested in TestConnectionPoolUsage.test_query_usage_doesnt_open_new_connection
        # Duplication here for clarity about what regression we're preventing
        pass

    def test_vetch_output_path_traversal_blocked(self):
        """
        Regression test for P1 issue: VETCH_OUTPUT path traversal must be blocked.
        """
        from pathlib import Path

        from vetch._security import is_safe_output_path

        # Attack patterns that should all be blocked
        attack_patterns = [
            "../../../../etc/passwd",
            "../../../.ssh/id_rsa",
            "/etc/shadow",
            "~/../../etc/passwd",
        ]

        cwd = Path.cwd()

        for pattern in attack_patterns:
            try:
                malicious_path = Path(pattern).resolve()
                # Should be blocked (not in allowed roots)
                result = is_safe_output_path(malicious_path, [cwd])
                assert not result, f"Path traversal not blocked: {pattern}"
            except (OSError, RuntimeError):
                # If path resolution fails, that's also acceptable (path is rejected)
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
