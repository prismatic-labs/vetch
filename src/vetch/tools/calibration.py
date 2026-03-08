"""Energy calibration from aggregate measurements.

This module provides tools to reverse-engineer per-token energy consumption from
aggregate measurements published by AI providers. Unlike hardware-level calibration
(which measures local GPU power), this approach works backwards from datacenter-scale
measurements to derive per-token values.

Overview
--------

When an AI provider publishes aggregate energy metrics (e.g., "0.24 Wh per median
prompt"), we can reverse-engineer per-token values if we make assumptions about:

1. **Median prompt length** - Total tokens in the "median prompt"
2. **PUE (Power Usage Effectiveness)** - Datacenter efficiency (1.0 = perfect)
3. **Output/input ratio** - Relative compute cost of generation vs. encoding

The calibration process:

1. Remove PUE overhead to get IT equipment energy
2. Assume prompt structure (e.g., 50% input, 50% output tokens)
3. Apply output/input ratio (typically 3:1 for autoregressive models)
4. Solve for per-token energy values
5. Validate by reproducing the original measurement

Mathematical Foundation
-----------------------

Given an aggregate measurement M (in Wh) for a median prompt of N_total tokens:

.. code-block:: text

    Step 1: Extract IT energy
    --------------------------
    IT_energy = M / PUE

    Step 2: Define prompt structure
    --------------------------------
    N_in = N_total / 2    (assume equal split)
    N_out = N_total / 2

    Step 3: Apply output/input ratio
    ---------------------------------
    Let R = output/input ratio (typically 3.0)
    E_out = R × E_in

    Step 4: Solve for E_in
    -----------------------
    IT_energy = (N_in × E_in) + (N_out × E_out)
    IT_energy = (N_in × E_in) + (N_out × R × E_in)
    IT_energy = E_in × (N_in + R × N_out)

    E_in = IT_energy / (N_in + R × N_out)

    Per-1k values:
    E_in_per_1k = E_in × 1000
    E_out_per_1k = E_in_per_1k × R

Uncertainty
-----------

**CRITICAL:** This methodology cannot produce a single "correct" value. Results are
highly sensitive to median prompt length assumptions:

- If actual median is 40 tokens: ~2.7 Wh/1k input
- If actual median is 800 tokens: ~0.14 Wh/1k input
- If actual median is 4000 tokens: ~0.03 Wh/1k input

**100x range!** Always present results as scenario-based ranges, not single values.

Use Cases
---------

**1. Validate existing calibrations**

.. code-block:: python

    from vetch.tools.calibration import validate_calibration

    # Validate Google's Gemini measurement
    result = validate_calibration(
        measurement_wh=0.24,
        pue=1.10,
        median_tokens=800,
        expected_input_per_1k=0.136,
        expected_output_per_1k=0.409,
    )
    print(f"Valid: {result.is_valid}, Error: {result.error_pct:.2f}%")

**2. Explore custom median assumptions**

.. code-block:: python

    from vetch.tools.calibration import explore_scenarios

    # See how energy values change across median assumptions
    scenarios = explore_scenarios(
        measurement_wh=0.24,
        pue=1.10,
        median_tokens=[40, 400, 800, 1600, 4000],
    )

    for s in scenarios:
        print(f"{s.median_tokens:4d} tokens: {s.energy_per_1k_input:.3f} Wh/1k")

**3. Calibrate custom measurements**

.. code-block:: python

    from vetch.tools.calibration import calibrate_from_measurement

    # Reverse-engineer from your own measurement
    result = calibrate_from_measurement(
        measurement_wh=0.15,  # Your measurement
        pue=1.08,
        median_tokens=1200,
    )

    print(f"Input:  {result.energy_per_1k_input:.3f} Wh/1k")
    print(f"Output: {result.energy_per_1k_output:.3f} Wh/1k")

**4. Compare model families**

.. code-block:: python

    from vetch.tools.calibration import propagate_to_family

    # Apply efficiency ratios to derive model family
    anchor = calibrate_from_measurement(0.24, 1.10, 800)

    family = propagate_to_family(
        anchor=anchor,
        efficiency_ratios={
            "model-1.5": 1.16,  # 16% less efficient
            "model-2.0": 1.00,  # Anchor
            "model-2.5": 0.80,  # 25% more efficient
        },
    )

    for name, values in family.items():
        print(f"{name}: {values.energy_per_1k_input:.3f} Wh/1k")

References
----------

.. [1] Google Cloud Blog (Aug 2025)
   "Measuring the environmental impact of AI inference"
   https://cloud.google.com/blog/products/infrastructure/measuring-environmental-impact-of-ai-inference

.. [2] Vetch Gemini Calibration Methodology
   src/vetch/registry/GEMINI_CALIBRATION.md

See Also
--------

- vetch.calibrate : Hardware-level energy measurement (GPU power draw)
- vetch.registry : Energy and pricing data registry

Notes
-----

This module is part of vetch's transparent energy accounting approach. All
calibrations are documented with assumptions, methodology, and uncertainty bounds.

For production use:
- Always present multiple scenarios (show the range)
- Document assumptions clearly
- Apply appropriate uncertainty margins (±50% typical)
- Prefer conservative estimates for carbon accounting
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a calibration from an aggregate measurement.

    Attributes:
        measurement_wh: Original aggregate measurement (Wh per prompt)
        pue: Power Usage Effectiveness used
        median_tokens: Assumed median prompt length (total tokens)
        median_input_tokens: Assumed input tokens in median prompt
        median_output_tokens: Assumed output tokens in median prompt
        output_input_ratio: Output/input energy ratio used
        it_energy_wh: IT equipment energy (after removing PUE)
        energy_per_1k_input: Energy per 1k input tokens (Wh)
        energy_per_1k_output: Energy per 1k output tokens (Wh)
        energy_per_input_token: Energy per single input token (Wh)
        energy_per_output_token: Energy per single output token (Wh)

    Example:
        >>> result = calibrate_from_measurement(0.24, 1.10, 800)
        >>> print(f"Input: {result.energy_per_1k_input:.3f} Wh/1k")
        Input: 0.136 Wh/1k
        >>> print(f"Output: {result.energy_per_1k_output:.3f} Wh/1k")
        Output: 0.409 Wh/1k
    """

    measurement_wh: float
    pue: float
    median_tokens: int
    median_input_tokens: int
    median_output_tokens: int
    output_input_ratio: float
    it_energy_wh: float
    energy_per_1k_input: float
    energy_per_1k_output: float
    energy_per_input_token: float
    energy_per_output_token: float


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating a calibration against expected values.

    Attributes:
        reproduced_wh: Measurement reproduced from per-token values
        original_wh: Original aggregate measurement
        error_wh: Absolute error (reproduced - original)
        error_pct: Percentage error
        is_valid: Whether error is within tolerance (±5%)
        tolerance_pct: Tolerance threshold used (default 5%)

    Example:
        >>> result = validate_calibration(0.24, 1.10, 800, 0.136, 0.409)
        >>> if result.is_valid:
        ...     print(f"✅ Valid (error: {result.error_pct:.2f}%)")
        ... else:
        ...     print(f"❌ Invalid (error: {result.error_pct:.2f}%)")
    """

    reproduced_wh: float
    original_wh: float
    error_wh: float
    error_pct: float
    is_valid: bool
    tolerance_pct: float


@dataclass(frozen=True)
class Scenario:
    """A calibration scenario with specific median assumption.

    Used for scenario-based uncertainty analysis. Each scenario represents
    a different assumption about the median prompt length.

    Attributes:
        name: Descriptive name (e.g., "Conservative", "Moderate")
        median_tokens: Total tokens in median prompt
        use_case: Description of when this scenario applies
        calibration: Calibration result for this scenario

    Example:
        >>> scenarios = explore_scenarios(0.24, 1.10, [40, 400, 800])
        >>> for s in scenarios:
        ...     print(f"{s.name}: {s.calibration.energy_per_1k_input:.3f} Wh/1k")
        40 tokens: 2.725 Wh/1k
        400 tokens: 0.273 Wh/1k
        800 tokens: 0.136 Wh/1k
    """

    name: str
    median_tokens: int
    use_case: str
    calibration: CalibrationResult


def calibrate_from_measurement(
    measurement_wh: float,
    pue: float,
    median_tokens: int,
    output_input_ratio: float = 3.0,
    input_output_split: float = 0.5,
) -> CalibrationResult:
    """Reverse-engineer per-token energy from an aggregate measurement.

    This function implements the core calibration algorithm. Given an aggregate
    measurement (e.g., "0.24 Wh per median prompt"), it calculates per-token
    energy values by:

    1. Removing PUE overhead
    2. Splitting tokens between input/output
    3. Applying output/input ratio
    4. Solving for per-token energy

    Args:
        measurement_wh: Aggregate energy measurement in Wh (e.g., 0.24 for Google's
            Gemini measurement). This is the total datacenter energy per prompt,
            including PUE overhead.
        pue: Power Usage Effectiveness. Ratio of total datacenter power to IT
            equipment power. Values typically range from 1.05 (excellent) to 1.5
            (poor). Google's fleet average is 1.10.
        median_tokens: Assumed total tokens in the "median prompt". This is the
            CRITICAL uncertainty - results scale linearly with this assumption.
            Example: 800 means 400 input + 400 output.
        output_input_ratio: Relative energy cost of output vs input tokens.
            Default 3.0 reflects autoregressive generation overhead (each output
            token attends to all previous tokens). Typically 2.0-4.0.
        input_output_split: Fraction of tokens that are input (0-1). Default 0.5
            assumes equal split. Adjust if you know the typical input/output ratio.

    Returns:
        CalibrationResult with per-token energy values and metadata.

    Raises:
        ValueError: If parameters are invalid (non-positive, out of range)

    Example:
        >>> # Google's Gemini measurement: 0.24 Wh, PUE 1.10, assume 800 tokens
        >>> result = calibrate_from_measurement(0.24, 1.10, 800)
        >>> print(f"Input:  {result.energy_per_1k_input:.3f} Wh/1k")
        Input:  0.136 Wh/1k
        >>> print(f"Output: {result.energy_per_1k_output:.3f} Wh/1k")
        Output: 0.409 Wh/1k

        >>> # Show full calculation details
        >>> print(f"IT energy: {result.it_energy_wh:.3f} Wh (removed PUE)")
        IT energy: 0.218 Wh (removed PUE)
        >>> print(f"Median: {result.median_input_tokens} in + {result.median_output_tokens} out")
        Median: 400 in + 400 out

    Notes:
        - Results scale linearly with median_tokens assumption
        - If actual median is 2x higher, energy per token is 2x lower
        - Always present multiple scenarios to show uncertainty
        - See GEMINI_CALIBRATION.md for full methodology

    See Also:
        - explore_scenarios: Generate multiple scenarios
        - validate_calibration: Verify by reproducing measurement
    """
    # Validate inputs
    if measurement_wh <= 0:
        raise ValueError(f"measurement_wh must be positive, got {measurement_wh}")
    if pue < 1.0:
        raise ValueError(f"PUE must be >= 1.0 (perfect efficiency), got {pue}")
    if median_tokens <= 0:
        raise ValueError(f"median_tokens must be positive, got {median_tokens}")
    if output_input_ratio <= 0:
        raise ValueError(
            f"output_input_ratio must be positive, got {output_input_ratio}"
        )
    if not 0 < input_output_split < 1:
        raise ValueError(
            f"input_output_split must be in (0, 1), got {input_output_split}"
        )

    # Step 1: Extract IT-level energy (remove PUE overhead)
    it_energy_wh = measurement_wh / pue

    # Step 2: Split tokens between input and output
    median_input_tokens = int(median_tokens * input_output_split)
    median_output_tokens = median_tokens - median_input_tokens

    # Step 3: Solve for E_in using the formula:
    #   IT_energy = (N_in × E_in) + (N_out × R × E_in)
    #   IT_energy = E_in × (N_in + R × N_out)
    #   E_in = IT_energy / (N_in + R × N_out)
    denominator = median_input_tokens + (output_input_ratio * median_output_tokens)
    energy_per_input_token = it_energy_wh / denominator

    # Step 4: Calculate output energy
    energy_per_output_token = energy_per_input_token * output_input_ratio

    # Step 5: Convert to per-1k values (standard industry unit)
    energy_per_1k_input = energy_per_input_token * 1000
    energy_per_1k_output = energy_per_output_token * 1000

    return CalibrationResult(
        measurement_wh=measurement_wh,
        pue=pue,
        median_tokens=median_tokens,
        median_input_tokens=median_input_tokens,
        median_output_tokens=median_output_tokens,
        output_input_ratio=output_input_ratio,
        it_energy_wh=it_energy_wh,
        energy_per_1k_input=energy_per_1k_input,
        energy_per_1k_output=energy_per_1k_output,
        energy_per_input_token=energy_per_input_token,
        energy_per_output_token=energy_per_output_token,
    )


def validate_calibration(
    measurement_wh: float,
    pue: float,
    median_tokens: int,
    expected_input_per_1k: float,
    expected_output_per_1k: float,
    tolerance_pct: float = 5.0,
    output_input_ratio: float = 3.0,
    input_output_split: float = 0.5,
) -> ValidationResult:
    """Validate a calibration by reproducing the original measurement.

    Given per-token energy values, this function calculates what the aggregate
    measurement SHOULD be and compares it to the actual measurement. If the
    reproduced value is within tolerance (default ±5%), the calibration is valid.

    This is the inverse of calibrate_from_measurement:
    - calibrate: measurement → per-token values
    - validate: per-token values → measurement (check if it matches)

    Args:
        measurement_wh: Original aggregate measurement to validate against
        pue: Power Usage Effectiveness
        median_tokens: Median prompt length used in calibration
        expected_input_per_1k: Expected input energy (Wh per 1k tokens)
        expected_output_per_1k: Expected output energy (Wh per 1k tokens)
        tolerance_pct: Acceptable error percentage (default 5%)
        output_input_ratio: Output/input ratio (must match calibration)
        input_output_split: Input/output split (must match calibration)

    Returns:
        ValidationResult indicating whether calibration is valid

    Example:
        >>> # Validate Google Gemini calibration (800 token scenario)
        >>> result = validate_calibration(
        ...     measurement_wh=0.24,
        ...     pue=1.10,
        ...     median_tokens=800,
        ...     expected_input_per_1k=0.136,
        ...     expected_output_per_1k=0.409,
        ... )
        >>> print(f"Reproduced: {result.reproduced_wh:.3f} Wh")
        Reproduced: 0.240 Wh
        >>> print(f"Original: {result.original_wh} Wh")
        Original: 0.24 Wh
        >>> print(f"Error: {result.error_pct:.2f}%")
        Error: 0.00%
        >>> print(f"Valid: {result.is_valid}")
        Valid: True

    See Also:
        - calibrate_from_measurement: Generate values to validate
    """
    # Calculate token counts
    median_input = int(median_tokens * input_output_split)
    median_output = median_tokens - median_input

    # Reproduce IT energy from per-token values
    it_energy_reproduced = (
        median_input * expected_input_per_1k + median_output * expected_output_per_1k
    ) / 1000

    # Add PUE overhead
    reproduced_wh = it_energy_reproduced * pue

    # Calculate error
    error_wh = reproduced_wh - measurement_wh
    error_pct = abs(error_wh) / measurement_wh * 100

    # Check if within tolerance
    is_valid = error_pct <= tolerance_pct

    return ValidationResult(
        reproduced_wh=reproduced_wh,
        original_wh=measurement_wh,
        error_wh=error_wh,
        error_pct=error_pct,
        is_valid=is_valid,
        tolerance_pct=tolerance_pct,
    )


def explore_scenarios(
    measurement_wh: float,
    pue: float,
    median_tokens: Sequence[int],
    output_input_ratio: float = 3.0,
    input_output_split: float = 0.5,
    scenario_names: Sequence[str] | None = None,
    use_cases: Sequence[str] | None = None,
) -> list[Scenario]:
    """Generate multiple calibration scenarios for uncertainty analysis.

    This function is the key to transparent energy accounting. Instead of
    presenting a single "correct" value (which doesn't exist without knowing
    the actual median), it generates multiple scenarios showing how results
    vary with median assumptions.

    Use this to:
    - Show the full range of uncertainty
    - Let users select the scenario matching their workload
    - Demonstrate sensitivity to assumptions

    Args:
        measurement_wh: Aggregate energy measurement
        pue: Power Usage Effectiveness
        median_tokens: List of median token counts to explore
        output_input_ratio: Output/input energy ratio
        input_output_split: Input/output token split
        scenario_names: Optional custom names (defaults to token counts)
        use_cases: Optional use case descriptions (defaults to generic)

    Returns:
        List of Scenario objects, one per median assumption

    Example:
        >>> # Gemini calibration with 5 scenarios
        >>> scenarios = explore_scenarios(
        ...     measurement_wh=0.24,
        ...     pue=1.10,
        ...     median_tokens=[40, 400, 800, 1600, 4000],
        ...     scenario_names=[
        ...         "Very Conservative",
        ...         "Conservative",
        ...         "Moderate",
        ...         "Optimistic",
        ...         "Maximum",
        ...     ],
        ...     use_cases=[
        ...         "Very short queries",
        ...         "Short Q&A",
        ...         "Mixed usage",
        ...         "Long context",
        ...         "Very long context",
        ...     ],
        ... )
        >>>
        >>> print("Scenario Analysis:")
        >>> print("-" * 60)
        >>> for s in scenarios:
        ...     e_in = s.calibration.energy_per_1k_input
        ...     print(f"{s.name:20s} {s.median_tokens:4d} tokens  {e_in:6.3f} Wh/1k")
        Scenario Analysis:
        ------------------------------------------------------------
        Very Conservative     40 tokens   2.725 Wh/1k
        Conservative         400 tokens   0.273 Wh/1k
        Moderate             800 tokens   0.136 Wh/1k
        Optimistic          1600 tokens   0.068 Wh/1k
        Maximum             4000 tokens   0.027 Wh/1k

        >>> # Show range
        >>> min_e = min(s.calibration.energy_per_1k_input for s in scenarios)
        >>> max_e = max(s.calibration.energy_per_1k_input for s in scenarios)
        >>> print(f"\\nRange: {min_e:.3f} - {max_e:.3f} Wh/1k ({max_e/min_e:.0f}x)")
        Range: 0.027 - 2.725 Wh/1k (100x)

    Notes:
        - Results demonstrate extreme sensitivity to median assumption
        - 100x range is typical without knowing actual median
        - Always present multiple scenarios, never a single value
        - Document which scenario you selected and why

    See Also:
        - calibrate_from_measurement: Generate single scenario
        - validate_calibration: Verify scenarios reproduce measurement
    """
    if scenario_names is not None and len(scenario_names) != len(median_tokens):
        raise ValueError(
            f"scenario_names length ({len(scenario_names)}) must match "
            f"median_tokens length ({len(median_tokens)})"
        )

    if use_cases is not None and len(use_cases) != len(median_tokens):
        raise ValueError(
            f"use_cases length ({len(use_cases)}) must match "
            f"median_tokens length ({len(median_tokens)})"
        )

    # Generate default names and use cases if not provided
    if scenario_names is None:
        scenario_names = [f"{n} tokens" for n in median_tokens]

    if use_cases is None:
        use_cases = ["Generic workload"] * len(median_tokens)

    scenarios = []
    for name, tokens, use_case in zip(scenario_names, median_tokens, use_cases):
        calibration = calibrate_from_measurement(
            measurement_wh=measurement_wh,
            pue=pue,
            median_tokens=tokens,
            output_input_ratio=output_input_ratio,
            input_output_split=input_output_split,
        )

        scenarios.append(
            Scenario(
                name=name,
                median_tokens=tokens,
                use_case=use_case,
                calibration=calibration,
            )
        )

    return scenarios


def propagate_to_family(
    anchor: CalibrationResult,
    efficiency_ratios: dict[str, float],
) -> dict[str, CalibrationResult]:
    """Propagate calibration to model family using efficiency ratios.

    Once you have a calibrated anchor model, you can derive other models in
    the family using published efficiency improvements. For example, if
    gemini-2.5-flash is "25% more efficient" than gemini-2.0-flash, multiply
    by 0.80.

    Args:
        anchor: Calibration result for the anchor model
        efficiency_ratios: Dictionary mapping model names to efficiency multipliers.
            - 1.0 = same efficiency as anchor
            - 0.80 = 25% more efficient (uses 80% of energy)
            - 1.16 = 16% less efficient (uses 116% of energy)

    Returns:
        Dictionary mapping model names to calibration results

    Example:
        >>> # Calibrate gemini-2.0-flash (anchor)
        >>> anchor = calibrate_from_measurement(0.24, 1.10, 800)
        >>>
        >>> # Derive family using Google's efficiency claims
        >>> family = propagate_to_family(
        ...     anchor=anchor,
        ...     efficiency_ratios={
        ...         "gemini-1.5-flash": 1.16,  # 16% less efficient
        ...         "gemini-2.0-flash": 1.00,  # Anchor
        ...         "gemini-2.5-flash": 0.80,  # 25% more efficient
        ...     },
        ... )
        >>>
        >>> for name, cal in family.items():
        ...     e_in = cal.energy_per_1k_input
        ...     print(f"{name:20s} {e_in:.3f} Wh/1k")
        gemini-1.5-flash     0.158 Wh/1k
        gemini-2.0-flash     0.136 Wh/1k
        gemini-2.5-flash     0.109 Wh/1k

    Notes:
        - Efficiency ratios should come from provider documentation
        - Be cautious: "20% fewer tokens" may mean better quality, not lower energy
        - Document source of efficiency ratios in basis field
        - Derived models inherit uncertainty from anchor

    See Also:
        - calibrate_from_measurement: Create anchor calibration
    """
    family = {}

    for model_name, ratio in efficiency_ratios.items():
        if ratio <= 0:
            raise ValueError(
                f"Efficiency ratio for {model_name} must be positive, got {ratio}"
            )

        # Scale energy values by ratio
        family[model_name] = CalibrationResult(
            measurement_wh=anchor.measurement_wh,
            pue=anchor.pue,
            median_tokens=anchor.median_tokens,
            median_input_tokens=anchor.median_input_tokens,
            median_output_tokens=anchor.median_output_tokens,
            output_input_ratio=anchor.output_input_ratio,
            it_energy_wh=anchor.it_energy_wh * ratio,
            energy_per_1k_input=anchor.energy_per_1k_input * ratio,
            energy_per_1k_output=anchor.energy_per_1k_output * ratio,
            energy_per_input_token=anchor.energy_per_input_token * ratio,
            energy_per_output_token=anchor.energy_per_output_token * ratio,
        )

    return family


def format_calibration_table(scenarios: Sequence[Scenario]) -> str:
    """Format scenarios as a markdown table for documentation.

    Args:
        scenarios: List of scenarios to format

    Returns:
        Markdown table string

    Example:
        >>> scenarios = explore_scenarios(0.24, 1.10, [40, 400, 800])
        >>> print(format_calibration_table(scenarios))
        | Scenario   | Median | Input (Wh/1k) | Output (Wh/1k) | Use Case |
        |------------|--------|---------------|----------------|----------|
        | 40 tokens  |     40 |         2.725 |          8.175 | Generic  |
        | 400 tokens |    400 |         0.273 |          0.818 | Generic  |
        | 800 tokens |    800 |         0.136 |          0.409 | Generic  |
    """
    lines = [
        "| Scenario | Median | Input (Wh/1k) | Output (Wh/1k) | Use Case |",
        "|----------|--------|---------------|----------------|----------|",
    ]

    for s in scenarios:
        cal = s.calibration
        lines.append(
            f"| {s.name:20s} | {s.median_tokens:6d} | "
            f"{cal.energy_per_1k_input:13.3f} | "
            f"{cal.energy_per_1k_output:14.3f} | "
            f"{s.use_case:20s} |"
        )

    return "\n".join(lines)


# Example usage and validation
if __name__ == "__main__":
    print("=" * 80)
    print("Vetch Energy Calibration Tool")
    print("=" * 80)
    print()
    print("Example: Google Gemini Calibration")
    print("-" * 80)
    print()

    # Generate scenarios
    scenarios = explore_scenarios(
        measurement_wh=0.24,
        pue=1.10,
        median_tokens=[40, 400, 800, 1600, 4000],
        scenario_names=[
            "Very Conservative",
            "Conservative",
            "Moderate (SELECTED)",
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

    # Display scenarios
    print("Scenario Analysis:")
    print()
    print(format_calibration_table(scenarios))
    print()

    # Show range
    min_e = min(s.calibration.energy_per_1k_input for s in scenarios)
    max_e = max(s.calibration.energy_per_1k_input for s in scenarios)
    print(f"Range: {min_e:.3f} - {max_e:.3f} Wh/1k ({max_e/min_e:.0f}x)")
    print()

    # Validate moderate scenario
    print("-" * 80)
    print("Validation (Moderate Scenario):")
    print()

    moderate = scenarios[2]  # 800 tokens
    validation = validate_calibration(
        measurement_wh=0.24,
        pue=1.10,
        median_tokens=800,
        expected_input_per_1k=moderate.calibration.energy_per_1k_input,
        expected_output_per_1k=moderate.calibration.energy_per_1k_output,
    )

    print(f"Original measurement: {validation.original_wh:.4f} Wh")
    print(f"Reproduced:           {validation.reproduced_wh:.4f} Wh")
    print(f"Error:                {validation.error_pct:.2f}%")
    print(f"Status:               {'✅ PASS' if validation.is_valid else '❌ FAIL'}")
    print()

    # Propagate to family
    print("-" * 80)
    print("Model Family (from Moderate scenario):")
    print()

    family = propagate_to_family(
        anchor=moderate.calibration,
        efficiency_ratios={
            "gemini-1.5-flash": 1.16,
            "gemini-2.0-flash": 1.00,
            "gemini-2.5-flash": 0.80,
        },
    )

    for name, cal in family.items():
        print(
            f"{name:20s} {cal.energy_per_1k_input:.3f} / "
            f"{cal.energy_per_1k_output:.3f} Wh/1k"
        )

    print()
    print("=" * 80)
