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
