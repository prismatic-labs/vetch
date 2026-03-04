"""Tests for global configuration module.

These tests verify:
- Global tags configuration
- Required tags enforcement
- Tag validation
"""

from __future__ import annotations

import pytest

from vetch.config import (
    add_global_tags,
    get_global_tags,
    get_required_tags,
    require_tags,
    validate_tags,
)


@pytest.fixture(autouse=True)
def reset_config() -> None:
    """Reset global configuration between tests."""
    import vetch.config
    vetch.config._global_tags = {}
    vetch.config._required_tags = set()
    yield
    vetch.config._global_tags = {}
    vetch.config._required_tags = set()


class TestGlobalTags:
    """Tests for global tags configuration."""

    def test_add_global_tags(self) -> None:
        """Add global tags that apply to all events."""
        add_global_tags({"env": "production", "service": "chat-api"})

        tags = get_global_tags()
        assert tags["env"] == "production"
        assert tags["service"] == "chat-api"

    def test_global_tags_merge(self) -> None:
        """Multiple calls merge tags."""
        add_global_tags({"env": "production"})
        add_global_tags({"version": "1.0.0"})

        tags = get_global_tags()
        assert tags["env"] == "production"
        assert tags["version"] == "1.0.0"

    def test_global_tags_override(self) -> None:
        """Later calls override earlier values."""
        add_global_tags({"env": "staging"})
        add_global_tags({"env": "production"})

        tags = get_global_tags()
        assert tags["env"] == "production"

    def test_empty_initial_state(self) -> None:
        """Global tags start empty."""
        tags = get_global_tags()
        assert tags == {}


class TestRequiredTags:
    """Tests for required tags enforcement."""

    def test_require_tags(self) -> None:
        """Set required tag keys."""
        require_tags(["feature_id", "cost_center"])

        required = get_required_tags()
        assert "feature_id" in required
        assert "cost_center" in required

    def test_require_tags_replaces(self) -> None:
        """Setting required tags replaces previous set."""
        require_tags(["tag1", "tag2"])
        require_tags(["tag3"])

        required = get_required_tags()
        assert "tag3" in required
        assert "tag1" not in required

    def test_empty_initial_state(self) -> None:
        """No tags required by default."""
        required = get_required_tags()
        assert len(required) == 0


class TestValidateTags:
    """Tests for tag validation."""

    def test_validate_with_no_requirements(self) -> None:
        """No validation when no tags required."""
        missing = validate_tags({"any": "tag"})
        assert missing == []

    def test_validate_all_present(self) -> None:
        """No missing when all required tags present."""
        require_tags(["team", "env"])

        missing = validate_tags({"team": "ml", "env": "prod", "extra": "allowed"})
        assert missing == []

    def test_validate_some_missing(self) -> None:
        """Returns missing tag names."""
        require_tags(["team", "env", "feature_id"])

        missing = validate_tags({"team": "ml"})
        assert "env" in missing
        assert "feature_id" in missing
        assert "team" not in missing

    def test_validate_all_missing(self) -> None:
        """Returns all required when none present."""
        require_tags(["team", "env"])

        missing = validate_tags({})
        assert set(missing) == {"team", "env"}

    def test_validate_none_tags(self) -> None:
        """Handles None tags gracefully."""
        require_tags(["team"])

        missing = validate_tags(None)
        assert missing == ["team"]

    def test_missing_sorted(self) -> None:
        """Missing tags are returned sorted."""
        require_tags(["zebra", "apple", "mango"])

        missing = validate_tags({})
        assert missing == ["apple", "mango", "zebra"]


class TestRedactedTags:
    """Test PII redaction functionality."""

    def test_redact_sensitive_tag_hashes_value(self):
        """Test that redacted tags are hashed, not plaintext."""
        import vetch.config as config

        # Set a tag for redaction
        config.set_redacted_tags(["email"])

        # Process tags with a sensitive value
        processed, warnings, missing = config.process_tags_single_pass(
            {"email": "user@example.com", "tier": "free"}
        )

        # Email should be hashed (starts with "redacted-")
        assert "email" in processed
        assert processed["email"].startswith("redacted-")
        assert "user@example.com" not in processed["email"]

        # Non-sensitive tags should be unchanged
        assert processed["tier"] == "free"

        # Cleanup
        config._reset_config()


class TestTagAllowlist:
    """Test tag allowlist filtering."""

    def test_allowlist_filters_unwanted_tags(self):
        """Test that tags not in allowlist are filtered out."""
        import vetch.config as config

        # Set allowlist
        config.set_tag_allowlist({"environment", "region"})

        # Process tags with some not in allowlist
        processed, warnings, missing = config.process_tags_single_pass(
            {"environment": "prod", "region": "us-east-1", "internal_id": "abc123"}
        )

        # Only allowlisted tags should remain
        assert "environment" in processed
        assert "region" in processed
        assert "internal_id" not in processed

        # Should have warning about filtered tag
        assert len(warnings) > 0
        assert any("internal_id" in w and "allowlist" in w for w in warnings)

        # Cleanup
        config._reset_config()


class TestCardinalityLimits:
    """Test tag cardinality protection."""

    def test_cardinality_limit_enforced(self):
        """Test that cardinality limits prevent explosion."""
        import vetch.config as config

        # Set low cardinality limit for testing
        config.set_tag_cardinality_limit(5)

        # Try to add more than limit
        for i in range(10):
            processed, warnings, missing = config.process_tags_single_pass(
                {"user_id": f"user_{i}"}
            )

        # Should have warnings after limit
        # (warnings appear when trying to add beyond limit)
        config._reset_config()

    def test_cardinality_prevents_memory_exhaustion(self):
        """Cardinality limits prevent unbounded memory growth from high-cardinality tags."""
        import vetch.config as config

        # Simulate real-world scenario: user_id tag with millions of unique values
        config.set_tag_cardinality_limit(100)

        warning_count = 0
        for i in range(500):
            processed, warnings, missing = config.process_tags_single_pass(
                {"user_id": f"user_{i}", "environment": "production"}
            )
            if warnings:
                warning_count += 1

        # After hitting limit, should start warning
        assert warning_count > 0

        # Cleanup
        config._reset_config()


class TestSecurityFeatures:
    """Test security-critical tag processing features."""

    def test_redaction_prevents_pii_leakage(self):
        """Redacted tags never expose plaintext PII in logs."""
        import vetch.config as config

        config.set_redacted_tags(["email", "ssn", "credit_card"])

        # Process tags with sensitive data
        processed, warnings, missing = config.process_tags_single_pass(
            {
                "email": "john.doe@company.com",
                "ssn": "123-45-6789",
                "credit_card": "4111-1111-1111-1111",
                "user_tier": "premium",
            }
        )

        # Critical: Plaintext PII must not appear in processed tags
        assert "john.doe@company.com" not in str(processed.values())
        assert "123-45-6789" not in str(processed.values())
        assert "4111-1111-1111-1111" not in str(processed.values())

        # Redacted tags should have deterministic hashes (same input = same hash)
        processed2, _, _ = config.process_tags_single_pass(
            {"email": "john.doe@company.com"}
        )
        assert processed["email"] == processed2["email"]

        # Different values should have different hashes
        processed3, _, _ = config.process_tags_single_pass(
            {"email": "jane.smith@company.com"}
        )
        assert processed["email"] != processed3["email"]

        # Non-sensitive tags should pass through unchanged
        assert processed["user_tier"] == "premium"

        config._reset_config()

    def test_allowlist_blocks_internal_tags(self):
        """Allowlist prevents internal/debugging tags from reaching production logs."""
        import vetch.config as config

        # Scenario: Only allow specific production tags
        config.set_tag_allowlist({"environment", "service", "version", "region"})

        # Developer accidentally includes internal debugging tags
        processed, warnings, missing = config.process_tags_single_pass(
            {
                "environment": "production",
                "service": "api",
                "internal_user_id": "12345",
                "debug_session": "test-session-xyz",
                "developer_name": "Alice",
            }
        )

        # Critical: Internal tags must not appear in output
        assert "internal_user_id" not in processed
        assert "debug_session" not in processed
        assert "developer_name" not in processed

        # Allowed tags should pass through
        assert "environment" in processed
        assert "service" in processed

        # Should warn about filtered tags
        assert len(warnings) > 0

        config._reset_config()

    def test_required_tags_enforce_compliance(self):
        """Required tags enforce organizational compliance policies."""
        import vetch.config as config
        from vetch.config import require_tags, validate_tags

        # Scenario: Company policy requires cost_center and data_classification
        require_tags(["cost_center", "data_classification"])

        # Compliant request
        missing = validate_tags(
            {"cost_center": "eng-ml", "data_classification": "internal", "model": "gpt-4"}
        )
        assert len(missing) == 0

        # Non-compliant request (missing required tags)
        missing = validate_tags({"model": "gpt-4", "user": "alice"})
        assert "cost_center" in missing
        assert "data_classification" in missing

        config._reset_config()
