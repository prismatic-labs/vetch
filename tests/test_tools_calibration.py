"""Tests for vetch.tools.calibration module."""

import pytest

from vetch.tools.calibration import (
    CalibrationResult,
    Scenario,
    ValidationResult,
    calibrate_from_measurement,
    explore_scenarios,
    format_calibration_table,
    propagate_to_family,
    validate_calibration,
)


class TestCalibrateFromMeasurement:
    """Test calibrate_from_measurement function."""

    def test_basic_calibration(self):
        """Test basic calibration with Google's Gemini measurement."""
        result = calibrate_from_measurement(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
        )

        assert isinstance(result, CalibrationResult)
        assert result.measurement_wh == 0.24
        assert result.pue == 1.10
        assert result.median_tokens == 800
        assert result.median_input_tokens == 400
        assert result.median_output_tokens == 400
        assert result.output_input_ratio == 3.0

        # IT energy should be measurement / PUE
        assert abs(result.it_energy_wh - 0.218) < 0.001

        # Per-1k values should match moderate scenario
        assert abs(result.energy_per_1k_input - 0.136) < 0.001
        assert abs(result.energy_per_1k_output - 0.409) < 0.001

    def test_custom_output_input_ratio(self):
        """Test calibration with custom output/input ratio."""
        result = calibrate_from_measurement(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
            output_input_ratio=2.0,  # Lower ratio
        )

        # Output energy should be 2x input (not 3x)
        assert abs(result.energy_per_1k_output / result.energy_per_1k_input - 2.0) < 0.001

    def test_custom_input_output_split(self):
        """Test calibration with custom input/output split."""
        result = calibrate_from_measurement(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=1000,
            input_output_split=0.7,  # 70% input, 30% output
        )

        assert result.median_input_tokens == 700
        assert result.median_output_tokens == 300

    def test_invalid_measurement_wh(self):
        """Test error on non-positive measurement."""
        with pytest.raises(ValueError, match="measurement_wh must be positive"):
            calibrate_from_measurement(
                measurement_wh=0.0,
                pue=1.10,
                median_tokens=800,
            )

        with pytest.raises(ValueError, match="measurement_wh must be positive"):
            calibrate_from_measurement(
                measurement_wh=-0.1,
                pue=1.10,
                median_tokens=800,
            )

    def test_invalid_pue(self):
        """Test error on PUE < 1.0."""
        with pytest.raises(ValueError, match="PUE must be >= 1.0"):
            calibrate_from_measurement(
                measurement_wh=0.24,
                pue=0.9,
                median_tokens=800,
            )

    def test_invalid_median_tokens(self):
        """Test error on non-positive median tokens."""
        with pytest.raises(ValueError, match="median_tokens must be positive"):
            calibrate_from_measurement(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=0,
            )

    def test_invalid_output_input_ratio(self):
        """Test error on non-positive output/input ratio."""
        with pytest.raises(ValueError, match="output_input_ratio must be positive"):
            calibrate_from_measurement(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=800,
                output_input_ratio=0.0,
            )

    def test_invalid_input_output_split(self):
        """Test error on invalid input/output split."""
        with pytest.raises(ValueError, match="input_output_split must be in"):
            calibrate_from_measurement(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=800,
                input_output_split=0.0,
            )

        with pytest.raises(ValueError, match="input_output_split must be in"):
            calibrate_from_measurement(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=800,
                input_output_split=1.0,
            )


class TestValidateCalibration:
    """Test validate_calibration function."""

    def test_perfect_validation(self):
        """Test validation with perfect reproduction."""
        result = validate_calibration(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
            expected_input_per_1k=0.136,
            expected_output_per_1k=0.409,
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert abs(result.reproduced_wh - 0.24) < 0.01
        assert result.error_pct < 5.0

    def test_out_of_tolerance(self):
        """Test validation fails when error exceeds tolerance."""
        result = validate_calibration(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
            expected_input_per_1k=0.200,  # Wrong value
            expected_output_per_1k=0.600,  # Wrong value
            tolerance_pct=5.0,
        )

        assert not result.is_valid
        assert result.error_pct > 5.0

    def test_custom_tolerance(self):
        """Test validation with custom tolerance."""
        result = validate_calibration(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
            expected_input_per_1k=0.150,  # Slightly off
            expected_output_per_1k=0.450,
            tolerance_pct=10.0,  # Higher tolerance
        )

        # Should pass with higher tolerance
        assert result.tolerance_pct == 10.0


class TestExploreScenarios:
    """Test explore_scenarios function."""

    def test_basic_scenarios(self):
        """Test generating multiple scenarios."""
        scenarios = explore_scenarios(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=[40, 400, 800, 1600, 4000],
        )

        assert len(scenarios) == 5
        assert all(isinstance(s, Scenario) for s in scenarios)

        # Check values are in descending order (more tokens = less energy per token)
        energies = [s.calibration.energy_per_1k_input for s in scenarios]
        assert energies == sorted(energies, reverse=True)

        # Check 100x range
        min_e = min(energies)
        max_e = max(energies)
        assert abs(max_e / min_e - 100) < 5  # ~100x range

    def test_custom_scenario_names(self):
        """Test scenarios with custom names."""
        scenarios = explore_scenarios(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=[100, 1000],
            scenario_names=["Low", "High"],
            use_cases=["Short prompts", "Long prompts"],
        )

        assert len(scenarios) == 2
        assert scenarios[0].name == "Low"
        assert scenarios[1].name == "High"
        assert scenarios[0].use_case == "Short prompts"
        assert scenarios[1].use_case == "Long prompts"

    def test_mismatched_names_length(self):
        """Test error when names length doesn't match tokens."""
        with pytest.raises(ValueError, match="scenario_names length"):
            explore_scenarios(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=[100, 1000],
                scenario_names=["Only one name"],
            )

    def test_mismatched_use_cases_length(self):
        """Test error when use_cases length doesn't match tokens."""
        with pytest.raises(ValueError, match="use_cases length"):
            explore_scenarios(
                measurement_wh=0.24,
                pue=1.10,
                median_tokens=[100, 1000],
                use_cases=["Only one use case"],
            )


class TestPropagateToFamily:
    """Test propagate_to_family function."""

    def test_basic_propagation(self):
        """Test propagating anchor to model family."""
        anchor = calibrate_from_measurement(0.24, 1.10, 800)

        family = propagate_to_family(
            anchor=anchor,
            efficiency_ratios={
                "model-1.5": 1.16,
                "model-2.0": 1.00,
                "model-2.5": 0.80,
            },
        )

        assert len(family) == 3
        assert "model-1.5" in family
        assert "model-2.0" in family
        assert "model-2.5" in family

        # model-2.0 should match anchor
        assert abs(family["model-2.0"].energy_per_1k_input - anchor.energy_per_1k_input) < 0.001

        # model-1.5 should be 16% higher (less efficient)
        assert (
            abs(family["model-1.5"].energy_per_1k_input - anchor.energy_per_1k_input * 1.16)
            < 0.001
        )

        # model-2.5 should be 20% lower (more efficient)
        assert (
            abs(family["model-2.5"].energy_per_1k_input - anchor.energy_per_1k_input * 0.80)
            < 0.001
        )

    def test_invalid_efficiency_ratio(self):
        """Test error on non-positive efficiency ratio."""
        anchor = calibrate_from_measurement(0.24, 1.10, 800)

        with pytest.raises(ValueError, match="Efficiency ratio.*must be positive"):
            propagate_to_family(
                anchor=anchor,
                efficiency_ratios={"bad-model": 0.0},
            )


class TestFormatCalibrationTable:
    """Test format_calibration_table function."""

    def test_format_table(self):
        """Test formatting scenarios as markdown table."""
        scenarios = explore_scenarios(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=[40, 400, 800],
            scenario_names=["Very Conservative", "Conservative", "Moderate"],
            use_cases=["Very short", "Short", "Mixed"],
        )

        table = format_calibration_table(scenarios)

        # Should be markdown table
        assert "| Scenario" in table
        assert "| Median" in table
        assert "|------" in table

        # Should have all scenarios
        assert "Very Conservative" in table
        assert "Conservative" in table
        assert "Moderate" in table

        # Should have values
        assert "40" in table
        assert "400" in table
        assert "800" in table


class TestGeminiCalibrationExample:
    """Test real-world Gemini calibration scenario."""

    def test_google_gemini_measurement(self):
        """Test calibrating from Google's actual Gemini measurement."""
        # Generate 5 scenarios matching GEMINI_CALIBRATION.md
        scenarios = explore_scenarios(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=[40, 400, 800, 1600, 4000],
            scenario_names=[
                "Very Conservative",
                "Conservative",
                "Moderate",
                "Optimistic",
                "Maximum",
            ],
            use_cases=[
                "Very short queries",
                "Short Q&A",
                "Mixed usage",
                "Long context",
                "Very long context",
            ],
        )

        # Validate moderate scenario (800 tokens) reproduces 0.24 Wh
        moderate = scenarios[2]
        validation = validate_calibration(
            measurement_wh=0.24,
            pue=1.10,
            median_tokens=800,
            expected_input_per_1k=moderate.calibration.energy_per_1k_input,
            expected_output_per_1k=moderate.calibration.energy_per_1k_output,
        )

        assert validation.is_valid
        assert validation.error_pct < 1.0  # Should be very accurate

        # Propagate to Gemini family
        family = propagate_to_family(
            anchor=moderate.calibration,
            efficiency_ratios={
                "gemini-1.5-flash": 1.16,
                "gemini-2.0-flash": 1.00,
                "gemini-2.5-flash": 0.80,
            },
        )

        # gemini-2.0-flash should be the anchor
        assert (
            abs(
                family["gemini-2.0-flash"].energy_per_1k_input
                - moderate.calibration.energy_per_1k_input
            )
            < 0.001
        )

        # gemini-1.5-flash should be 16% less efficient
        assert (
            abs(
                family["gemini-1.5-flash"].energy_per_1k_input
                - moderate.calibration.energy_per_1k_input * 1.16
            )
            < 0.001
        )

        # gemini-2.5-flash should be 25% more efficient
        assert (
            abs(
                family["gemini-2.5-flash"].energy_per_1k_input
                - moderate.calibration.energy_per_1k_input * 0.80
            )
            < 0.001
        )
