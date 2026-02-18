"""Tests for configuration module."""

from __future__ import annotations

from vetch.config import (
    add_global_tags,
    get_global_tags,
    get_required_tags,
    require_tags,
    validate_tags,
)


class TestConfig:
    """Tests for global configuration."""

    def test_global_tags(self) -> None:
        """Verify adding and getting global tags."""
        # Initial state should be empty
        assert get_global_tags() == {}

        tags = {"env": "test", "version": "1.0"}
        add_global_tags(tags)
        assert get_global_tags() == tags

        # Incremental update
        add_global_tags({"service": "api"})
        assert get_global_tags()["service"] == "api"
        assert get_global_tags()["env"] == "test"

    def test_required_tags(self) -> None:
        """Verify specifying required tags."""
        require_tags(["feature_id", "user"])
        assert get_required_tags() == {"feature_id", "user"}

    def test_validate_tags(self) -> None:
        """Verify tag validation logic."""
        require_tags(["org", "app"])

        # Valid
        assert validate_tags({"org": "p", "app": "v"}) == []

        # Missing
        missing = validate_tags({"org": "p"})
        assert missing == ["app"]

        # Empty
        assert validate_tags(None) == ["app", "org"]
