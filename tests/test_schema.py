"""Tests for schema module."""


from vetch.schema import (
    SCHEMA_VERSION,
    InferenceEvent,
    validate_energy_override,
)


class TestSchemaVersion:
    """Tests for schema versioning."""

    def test_schema_version_is_string(self) -> None:
        """Schema version should be a string."""
        assert isinstance(SCHEMA_VERSION, str)

    def test_schema_version_is_one(self) -> None:
        """Initial schema version should be 1."""
        assert SCHEMA_VERSION == "1"


class TestValidateEnergyOverride:
    """Tests for energy_override validation.

    Note: validate_energy_override returns (result, warnings) tuple.
    """

    def test_valid_override_minimal(self) -> None:
        """Valid override with only required fields."""
        override = {
            "wh_per_1k_input": 0.5,
            "wh_per_1k_output": 1.5,
        }
        result, warnings = validate_energy_override(override)
        assert result is not None
        assert result["wh_per_1k_input"] == 0.5
        assert result["wh_per_1k_output"] == 1.5
        assert "tier" not in result
        assert "source" not in result
        assert warnings == []

    def test_valid_override_full(self) -> None:
        """Valid override with all fields."""
        override = {
            "wh_per_1k_input": 0.8,
            "wh_per_1k_output": 2.4,
            "tier": 2,
            "source": "Internal benchmark, 2026-01",
        }
        result, warnings = validate_energy_override(override)
        assert result is not None
        assert result["wh_per_1k_input"] == 0.8
        assert result["wh_per_1k_output"] == 2.4
        assert result["tier"] == 2
        assert result["source"] == "Internal benchmark, 2026-01"
        assert warnings == []

    def test_invalid_not_dict(self) -> None:
        """Non-dict input returns None with warning."""
        result, warnings = validate_energy_override("invalid")  # type: ignore[arg-type]
        assert result is None
        assert len(warnings) == 1
        assert "must be a dict" in warnings[0]

    def test_invalid_missing_input(self) -> None:
        """Missing wh_per_1k_input returns None with warning."""
        override = {"wh_per_1k_output": 1.5}
        result, warnings = validate_energy_override(override)
        assert result is None
        assert len(warnings) == 1
        assert "wh_per_1k_input" in warnings[0]

    def test_invalid_missing_output(self) -> None:
        """Missing wh_per_1k_output returns None with warning."""
        override = {"wh_per_1k_input": 0.5}
        result, warnings = validate_energy_override(override)
        assert result is None
        assert len(warnings) == 1
        assert "wh_per_1k_output" in warnings[0]

    def test_invalid_zero_input(self) -> None:
        """Zero wh_per_1k_input returns None with warning."""
        override = {"wh_per_1k_input": 0, "wh_per_1k_output": 1.5}
        result, warnings = validate_energy_override(override)
        assert result is None
        assert len(warnings) == 1
        assert "positive number" in warnings[0]

    def test_invalid_negative_output(self) -> None:
        """Negative wh_per_1k_output returns None with warning."""
        override = {"wh_per_1k_input": 0.5, "wh_per_1k_output": -1.0}
        result, warnings = validate_energy_override(override)
        assert result is None
        assert len(warnings) == 1
        assert "positive number" in warnings[0]

    def test_invalid_tier_out_of_range(self) -> None:
        """Invalid tier is warned and defaults to 1."""
        override = {
            "wh_per_1k_input": 0.5,
            "wh_per_1k_output": 1.5,
            "tier": 5,  # Invalid, should be 0-3
        }
        result, warnings = validate_energy_override(override)
        assert result is not None
        # Invalid tier gets warning and defaults to 1
        assert result["tier"] == 1
        assert len(warnings) == 1
        assert "tier must be 0-3" in warnings[0]

    def test_integer_values_converted_to_float(self) -> None:
        """Integer values should be converted to float."""
        override = {
            "wh_per_1k_input": 1,
            "wh_per_1k_output": 3,
        }
        result, warnings = validate_energy_override(override)
        assert result is not None
        assert result["wh_per_1k_input"] == 1.0
        assert result["wh_per_1k_output"] == 3.0
        assert isinstance(result["wh_per_1k_input"], float)
        assert isinstance(result["wh_per_1k_output"], float)
        assert warnings == []


class TestInferenceEventTypedDict:
    """Tests for InferenceEvent TypedDict structure."""

    def test_can_create_minimal_event(self) -> None:
        """Can create event with minimal fields."""
        event: InferenceEvent = {
            "schema_version": "1",
            "vetch_version": "0.1.0",
            "event_id": "test-id",
            "timestamp": "2026-02-12T00:00:00Z",
            "signal_quality": "unknown",
        }
        assert event["schema_version"] == "1"

    def test_can_create_full_event(self) -> None:
        """Can create event with all fields."""
        event: InferenceEvent = {
            "schema_version": "1",
            "vetch_version": "0.1.0",
            "event_id": "test-id",
            "timestamp": "2026-02-12T00:00:00Z",
            "model": "gpt-4o",
            "provider": "openai",
            "model_known": True,
            "usage": {
                "text": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "total_tokens": 150,
                }
            },
            "estimated_energy_wh": 0.5,
            "estimated_carbon_g": 0.2,
            "estimated_cost_usd": 0.01,
            "estimated_cost_input_usd": 0.005,
            "estimated_cost_output_usd": 0.005,
            "billing_tier": "list",
            "signal_quality": "live",
            "energy_tier": 3,
            "energy_source": "registry",
            "energy_override_source": None,
            "energy_basis": "Test basis",
            "grid_intensity_gco2e_kwh": 420.0,
            "grid_intensity_timestamp": "2026-02-12T00:00:00Z",
            "region": "us-east-1",
            "is_stream": False,
            "complete": True,
            "latency_ms": 1234.5,
            "tags": {"team": "ml"},
            "error": False,
            "error_type": None,
            "tracking_disabled": False,
            "vetch_warnings": None,
            "budget_energy_wh": None,
            "budget_carbon_g": None,
            "budget_cost_usd": None,
            "budget_exceeded": None,
            "usage_estimated": False,
            "usage_estimation_method": None,
        }
        assert event["model"] == "gpt-4o"
        assert event["signal_quality"] == "live"
