"""Command-line interface for Vetch.

Provides tools for:
- Estimating energy/carbon/cost without running code
- Comparing models
- Viewing methodology
- Validating environment
- Configuring defaults (PUE, region, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

from vetch import __version__


def positive_int(value: str) -> int:
    """Validate that a string represents a positive integer.

    Args:
        value: String to parse.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: If value is not a positive integer.
    """
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid integer") from None

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a positive integer (got {ivalue})"
        )
    return ivalue


def positive_float(value: str) -> float:
    """Validate that a string represents a positive float.

    Args:
        value: String to parse.

    Returns:
        Parsed positive float.

    Raises:
        argparse.ArgumentTypeError: If value is not a positive float.
    """
    try:
        fvalue = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a valid number") from None

    if fvalue <= 0:
        raise argparse.ArgumentTypeError(
            f"'{value}' must be a positive number (got {fvalue})"
        )
    return fvalue


# Config file location
CONFIG_PATH = Path.home() / ".vetch" / "config.json"


def load_config() -> dict[str, Any]:
    """Load Vetch config from ~/.vetch/config.json."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return cast("dict[str, Any]", json.loads(CONFIG_PATH.read_text()))
    except Exception:
        return {}


def save_config(config: dict[str, object]) -> None:
    """Save Vetch config to ~/.vetch/config.json."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def estimate(args: argparse.Namespace) -> None:
    """Estimate energy/carbon/cost for a specific model and token counts."""
    from vetch.calculation import calculate_carbon, calculate_cost, calculate_energy
    from vetch.sensing.grid import get_carbon_intensity

    # 1. Get grid data
    grid = get_carbon_intensity(args.region)

    # 2. Perform calculations
    energy_wh, tier, uncertainty_pct, source, basis, known = calculate_energy(
        args.input_tokens, args.output_tokens, args.model
    )

    carbon_g = calculate_carbon(energy_wh, grid.intensity_gco2e_kwh)

    cost_usd, _, _, _ = calculate_cost(
        args.input_tokens, args.output_tokens, args.model
    )

    # 3. Output
    if args.format == "json":
        result = {
            "model": args.model,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "energy_wh": energy_wh,
            "carbon_g": carbon_g,
            "cost_usd": cost_usd,
            "energy_tier": tier,
            "energy_uncertainty_pct": uncertainty_pct,
            "grid_region": args.region or "global",
            "grid_intensity": grid.intensity_gco2e_kwh,
            "signal_quality": grid.signal_quality,
        }
        print(json.dumps(result, indent=2))
        return

    # Text output with uncertainty indicators
    uncertainty_label = f"±{uncertainty_pct}%" if uncertainty_pct < 1000 else "order of magnitude"
    print(f"Energy:  ~{energy_wh:.2f} Wh ({uncertainty_label})  [Tier {tier}]")
    intensity = grid.intensity_gco2e_kwh
    print(f"Carbon:  ~{carbon_g:.2f}g       [{intensity:.0f} gCO2e/kWh, {grid.signal_quality}]")
    print(f"Cost:    ${cost_usd:.2f}       [list pricing]")
    print()
    print(f"Basis: {basis}")
    if not known:
        print("\nNote: Model not found in registry. Using conservative fallback.")
        print("Have better data? Run: vetch methodology --contribute")


def compare(args: argparse.Namespace) -> None:
    """Compare multiple models for the same token counts."""
    from vetch.calculation import calculate_carbon, calculate_cost, calculate_energy
    from vetch.sensing.grid import get_carbon_intensity

    models = [m.strip() for m in args.models.split(",")]
    grid = get_carbon_intensity(args.region)

    results = []
    for model in models:
        energy_wh, tier, uncertainty_pct, _, _, known = calculate_energy(
            args.input_tokens, args.output_tokens, model
        )
        carbon_g = calculate_carbon(energy_wh, grid.intensity_gco2e_kwh)
        cost_usd, _, _, _ = calculate_cost(
            args.input_tokens, args.output_tokens, model
        )
        results.append({
            "model": model,
            "energy": energy_wh,
            "cost": cost_usd,
            "carbon": carbon_g,
            "known": known
        })

    # Sort by energy ascending
    results.sort(key=lambda x: x["energy"])

    if args.format == "json":
        print(json.dumps(results, indent=2))
        return

    # Table output
    print(f"{'Model':<20} {'Energy (Wh)':<15} {'Cost ($)':<12} {'Carbon (g)':<10}")
    print("─" * 60)
    for r in results:
        indicator = "" if r["known"] else "*"
        name = r['model'] + indicator
        print(f"{name:<20} ~{r['energy']:<14.2f} ${r['cost']:<11.2f} ~{r['carbon']:<9.1f}")

    region_name = args.region or 'global'
    intensity = grid.intensity_gco2e_kwh
    print(f"\nGrid: {region_name} ({intensity:.0f} gCO2e/kWh, {grid.signal_quality})")
    print("All estimates are tier 3 (±10x uncertainty).")
    if any(not r["known"] for r in results):
        print("* Model not in registry, using conservative fallback.")
    print("Run 'vetch estimate --model <name>' for derivation details.")


def methodology(args: argparse.Namespace) -> None:
    """Show methodology documentation."""
    # Try to find METHODOLOGY.md in the package
    from pathlib import Path
    methodology_path = Path(__file__).parent / "METHODOLOGY.md"

    if not methodology_path.exists():
        print("METHODOLOGY.md not found in package.")
        sys.exit(1)

    content = methodology_path.read_text()

    if args.contribute:
        # Just show the contribution section
        if "## Contributing Energy Estimates" in content:
            print(content.split("## Contributing Energy Estimates")[1])
        else:
            print(content)
    elif args.full:
        print(content)
    else:
        # Show preamble and key sections
        sections = content.split("##")
        print(sections[0])
        if len(sections) > 1:
            print("##" + sections[1])
        print("\nRun 'vetch methodology --full' to see the complete methodology.")


def check(args: argparse.Namespace) -> None:
    """Validate environment and connectivity."""
    from vetch.sensing.cache import get_file_cache
    from vetch.sensing.grid import get_carbon_intensity

    print(f"Vetch v{__version__} environment check (ALPHA)\n")

    # 1. Grid API
    print("Checking Grid API connectivity...")
    try:
        grid = get_carbon_intensity("us-east-1", force_refresh=True)
        print(f"  OK: {grid.intensity_gco2e_kwh:.0f} gCO2e/kWh (signal: {grid.signal_quality})")
    except Exception as e:
        print(f"  FAILED: {e}")

    # 2. Cache
    cache = get_file_cache()
    print("Checking cache status...")
    print(f"  Path: {cache.path}")
    if cache.path.exists():
        print(f"  Size: {cache.path.stat().st_size} bytes")
    else:
        print("  Status: Not yet created")

    # 3. Environment
    print("Checking environment variables...")
    vars = ["VETCH_REGION", "ELECTRICITY_MAPS_API_KEY", "VETCH_OUTPUT", "VETCH_DEFAULT_PUE"]
    for v in vars:
        val = os.environ.get(v)
        status = val if val else "not set"
        if v == "ELECTRICITY_MAPS_API_KEY" and val:
            status = "********"
        print(f"  {v}: {status}")

    # 4. Observability
    print("Checking observability bridges...")
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
        span = trace.get_current_span()
        status = "Active" if span.is_recording() else "Installed, but no active span"
        print(f"  OpenTelemetry: {status}")
    except ImportError:
        print("  OpenTelemetry: Not installed")

    # 5. Registry Provenance
    print("Checking model registry...")
    from vetch.calculation import _ENERGY_PATH
    prov_path = _ENERGY_PATH.parent / "PROVENANCE.md"
    if prov_path.exists():
        print(f"  Provenance: Verified ({prov_path})")
    else:
        print("  Provenance: Missing audit trail")


def quickstart(args: argparse.Namespace) -> None:
    """Print quickstart examples."""
    print(f"""
Vetch v{__version__} - Planet-aware observability for LLM inference
GitHub: https://github.com/prismatic-labs/vetch

QUICKSTART
==========

1. Basic Usage (OpenAI)
-----------------------
from vetch import wrap
from openai import OpenAI

client = OpenAI()

with wrap() as ctx:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{{"role": "user", "content": "Hello!"}}]
    )

# Access metrics after the call
print(f"Energy: {{ctx.event['estimated_energy_wh']:.3f}} Wh")
print(f"Carbon: {{ctx.event['estimated_carbon_g']:.2f}} g CO2e")
print(f"Cost:   ${{ctx.event['estimated_cost_usd']:.4f}}")

2. Quiet Mode (no JSON output)
------------------------------
with wrap(emit=False) as ctx:
    response = client.chat.completions.create(...)

# Metrics still available in ctx.event

3. With Region (for accurate carbon)
------------------------------------
with wrap(region="us-west-2") as ctx:
    ...

# Or set environment variable:
# export VETCH_REGION=us-west-2

4. CLI Tools
------------
# Estimate without running code
vetch estimate --model gpt-4o --input-tokens 1000 --output-tokens 500

# Compare models
vetch compare --models gpt-4o,claude-3.5-sonnet --input-tokens 1000

# Check environment
vetch check

ENVIRONMENT VARIABLES
=====================
VETCH_REGION          - Grid region (e.g., us-east-1, eu-west-2)
VETCH_OUTPUT          - Output target: none, stderr, or file path
VETCH_DEFAULT_PUE     - Power Usage Effectiveness (default: 1.1)
ELECTRICITY_MAPS_API_KEY - For real-time grid carbon data

DOCUMENTATION
=============
https://github.com/prismatic-labs/vetch#readme
""")


def clean(args: argparse.Namespace) -> None:
    """Clean up Vetch cache and lock files."""
    import shutil

    from vetch.sensing.cache import DEFAULT_CACHE_DIR

    if not DEFAULT_CACHE_DIR.exists():
        print("No cache directory found.")
        return

    try:
        shutil.rmtree(DEFAULT_CACHE_DIR)
        print(f"Successfully cleaned {DEFAULT_CACHE_DIR}")
    except Exception as e:
        print(f"Failed to clean cache: {e}")


def audit(args: argparse.Namespace) -> None:
    """Analyze token usage patterns and generate advisories."""
    from vetch.advisory import format_advisories, generate_advisories
    from vetch.stats import get_session_stats

    stats = get_session_stats()

    if stats.total_requests == 0:
        print("No requests recorded in this session.")
        print("\nTo use the audit feature:")
        print("  1. Use vetch.wrap() around your LLM calls")
        print("  2. Run multiple requests")
        print("  3. Call 'vetch audit' to analyze patterns")
        return

    # Generate and display advisories
    advisories = generate_advisories(stats)
    print(format_advisories(advisories, args.format))

    # Also show summary
    if args.format == "text":
        from typing import cast

        print("\nSession Summary")
        print("-" * 30)
        summary = stats.summary()
        print(f"Total requests: {summary['total_requests']}")
        in_tokens = cast(int, summary['total_input_tokens'])
        out_tokens = cast(int, summary['total_output_tokens'])
        print(f"Total tokens: {in_tokens + out_tokens:,}")
        ratio = summary.get('average_input_output_ratio')
        if ratio is not None:
            print(f"Avg input:output ratio: {cast(float, ratio):.2f}:1")


def calibrate(args: argparse.Namespace) -> None:
    """Calibrate energy measurement using GPU power sensors.

    Also supports setting PUE and provider defaults without GPU.
    """
    from vetch.calibrate import (
        get_gpu_error,
        is_gpu_available,
    )

    # Handle PUE setting (works without GPU)
    if args.pue is not None:
        config = load_config()
        config["pue"] = args.pue
        if args.provider:
            config["provider"] = args.provider
        save_config(config)
        print(f"Saved PUE = {args.pue} to {CONFIG_PATH}")
        print(f"To apply globally: export VETCH_DEFAULT_PUE={args.pue}")
        if not is_gpu_available():
            print("\nNote: GPU not available for hardware calibration.")
            print("Using configured PUE for estimate adjustments.")
        return

    # Check GPU availability for actual calibration
    if not is_gpu_available():
        print("GPU power measurement not available.")
        print(f"Reason: {get_gpu_error()}")
        print("\nTo enable GPU calibration:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install pynvml: pip install nvidia-ml-py3")
        print("  3. Run calibration again")
        print()
        print("Without a GPU, you can still adjust estimates:")
        print("  vetch calibrate --pue 1.2  # Set data center PUE")
        print("  vetch config --set pue=1.2 # Same via config command")
        sys.exit(1)

    # Show help if no workload specified
    if not args.workload and not args.interactive:
        print("GPU Calibration")
        print("=" * 40)
        print()
        print("Calibration measures actual GPU power draw during inference")
        print("to provide more accurate energy estimates for your hardware.")
        print()
        print("Quick start (in Python):")
        print("  from vetch.calibrate import calibrate_model, format_calibration_result")
        print()
        print("  def my_workload():")
        print("      response = ollama.generate(model='llama3.1:8b', prompt='Hello')")
        print("      return 100, 50  # (input_tokens, output_tokens)")
        print()
        print("  result = calibrate_model('ollama', 'llama3.1:8b', workload=my_workload)")
        print("  print(format_calibration_result(result))")
        print()
        print("Without calibration, you can still adjust the PUE:")
        print("  vetch calibrate --pue 1.2")
        print()
        print("Check calibration status:")
        print("  vetch calibrate --status")
        return

    if args.interactive:
        print("Interactive calibration is not yet available.")
        print()
        print("Use the Python API for now:")
        print("  from vetch.calibrate import calibrate_model")
        print(f"  result = calibrate_model('{args.provider or 'ollama'}', "
              f"'{args.model or 'your-model'}', workload=your_function)")
        sys.exit(1)

    # Workload mode placeholder
    print("CLI workload execution is not yet available.")
    print()
    print("Use the Python API:")
    print("  from vetch.calibrate import calibrate_model, format_calibration_result")
    print()
    provider = args.provider or "ollama"
    model = args.model or "your-model"
    print(f"  result = calibrate_model('{provider}', '{model}', workload=your_function)")
    print("  print(format_calibration_result(result))")


def calibrate_status(args: argparse.Namespace) -> None:
    """Show calibration status and saved calibrations."""
    from vetch.calibrate import get_gpu_error, is_gpu_available

    print("Vetch Calibration Status\n")

    # GPU availability
    if is_gpu_available():
        from vetch.calibrate import GPUMonitor
        try:
            with GPUMonitor() as monitor:
                info = monitor.get_gpu_info()
                print(f"GPU: {info['name']}")
                print(f"Memory: {info['memory_total_mb']} MB")
                print(f"Driver: {info['driver_version']}")
        except Exception as e:
            print(f"GPU error: {e}")
    else:
        print(f"GPU: Not available ({get_gpu_error()})")

    # Saved calibrations
    cache_dir = Path.home() / ".vetch" / "calibrations"
    print(f"\nCalibrations directory: {cache_dir}")

    if cache_dir.exists():
        calibrations = list(cache_dir.glob("*.json"))
        if calibrations:
            print(f"Saved calibrations: {len(calibrations)}")
            for cal_file in calibrations:
                print(f"  - {cal_file.stem}")
        else:
            print("No saved calibrations.")
    else:
        print("No calibrations directory yet.")


def config_cmd(args: argparse.Namespace) -> None:
    """Manage Vetch configuration.

    Supports:
    - vetch config --show: Show current config
    - vetch config --set pue=1.2: Set a config value
    - vetch config --set region=us-east-1: Set default region
    """
    import os

    config = load_config()

    # Show current config
    if args.show or (not args.set and not args.unset):
        print(f"Config file: {CONFIG_PATH}")
        print()

        if not config:
            print("No configuration set.")
            print("\nAvailable settings:")
            print("  pue      - Power Usage Effectiveness (default: 1.1)")
            print("  region   - Default grid region (e.g., us-east-1)")
            print("  provider - Default local inference provider (e.g., ollama)")
            print()
            print("Set with: vetch config --set pue=1.2")
            return

        print("Current configuration:")
        for key, value in sorted(config.items()):
            env_override = None
            if key == "pue":
                env_override = os.environ.get("VETCH_DEFAULT_PUE")
            elif key == "region":
                env_override = os.environ.get("VETCH_REGION")

            if env_override:
                print(f"  {key}: {value} (overridden by env: {env_override})")
            else:
                print(f"  {key}: {value}")

        print(f"\nTo apply PUE to calculations, set VETCH_DEFAULT_PUE={config.get('pue', 1.1)}")
        return

    # Set values
    if args.set:
        for setting in args.set:
            if "=" not in setting:
                print(f"Error: Invalid format '{setting}'. Use key=value")
                sys.exit(1)

            key_to_del, value = setting.split("=", 1)
            key_to_del = key_to_del.strip().lower()

            # Validate known keys
            if key_to_del == "pue":
                try:
                    pue = float(value)
                    if pue < 1.0:
                        print(f"Error: PUE must be >= 1.0 (got {pue})")
                        sys.exit(1)
                    config["pue"] = pue
                    print(f"Set pue = {pue}")
                    print(f"  To apply: export VETCH_DEFAULT_PUE={pue}")
                except ValueError:
                    print(f"Error: PUE must be a number (got '{value}')")
                    sys.exit(1)
            elif key_to_del == "region":
                config["region"] = value.strip()
                print(f"Set region = {value}")
                print(f"  To apply: export VETCH_REGION={value}")
            elif key_to_del == "provider":
                config["provider"] = value.strip()
                print(f"Set provider = {value}")
            else:
                # Allow arbitrary keys for extensibility
                config[key_to_del] = value
                print(f"Set {key_to_del} = {value}")

        save_config(config)
        print(f"\nConfig saved to {CONFIG_PATH}")

    # Unset values
    if args.unset:
        for key_to_unset in args.unset:
            key_to_unset = key_to_unset.strip().lower()
            if key_to_unset in config:
                del config[key_to_unset]
                print(f"Removed {key_to_unset}")
            else:
                print(f"Key '{key_to_unset}' not found in config")

        save_config(config)
        print(f"\nConfig saved to {CONFIG_PATH}")


def report(args: argparse.Namespace) -> None:
    """Generate usage report from stored events."""
    from datetime import datetime, timedelta

    from vetch.storage import (
        get_db_path,
        get_top_consumers,
        is_storage_enabled,
        query_usage,
    )

    if not is_storage_enabled():
        # Try to enable with default path
        from vetch.storage import configure_storage
        configure_storage()

    if not is_storage_enabled():
        print("Storage is disabled by default to respect privacy.")
        print("\nTo enable persistent storage, add to your code:")
        print("")
        print("  from vetch.storage import configure_storage")
        print("  configure_storage(enabled=True)  # Stores to ~/.vetch/usage.db")
        print("")
        print("Then re-run your LLM calls to start recording history.")
        sys.exit(1)

    db_path = get_db_path()
    if db_path and not db_path.exists():
        print(f"No data found at {db_path}")
        print("\nStorage is configured but no events have been recorded yet.")
        print("Use vetch.wrap() around your LLM calls to start tracking.")
        sys.exit(1)

    # Parse time range
    days = args.days
    end = datetime.now()
    start = end - timedelta(days=days)

    # Parse tags filter
    tags = None
    if args.tags:
        tags = {}
        for tag_spec in args.tags.split(","):
            if "=" in tag_spec:
                key, value = tag_spec.split("=", 1)
                tags[key.strip()] = value.strip()

    # Query usage
    summary = query_usage(start=start, end=end, model=args.model, tags=tags)

    if args.format == "json":
        print(json.dumps(summary.to_dict(), indent=2))
        return

    # Text output
    print("Vetch Usage Report")
    print(f"Period: {summary.start_time.strftime('%Y-%m-%d')} to "
          f"{summary.end_time.strftime('%Y-%m-%d')}")
    print("=" * 50)
    print()

    if summary.total_requests == 0:
        print("No events found for the specified period.")
        return

    # Totals
    print("Totals")
    print("-" * 30)
    print(f"  Requests:      {summary.total_requests:,}")
    total_tokens = summary.total_input_tokens + summary.total_output_tokens
    print(f"  Tokens:        {total_tokens:,}")
    print(f"  Energy:        {summary.total_energy_wh:.2f} Wh")
    print(f"  Carbon:        {summary.total_carbon_g:.2f} g CO2e")
    print(f"  Cost:          ${summary.total_cost_usd:.2f}")
    print()

    # By model
    if summary.by_model:
        print("By Model")
        print("-" * 30)
        for model, data in summary.by_model.items():
            print(f"  {model}")
            print(f"    Requests: {int(data['requests']):,}")
            print(f"    Cost:     ${data['cost_usd']:.2f}")
            print(f"    Energy:   {data['energy_wh']:.2f} Wh")
        print()

    # By tag (show top tag breakdowns)
    if summary.by_tag:
        for tag_key, tag_values in list(summary.by_tag.items())[:3]:
            print(f"By {tag_key.title()}")
            print("-" * 30)
            for tag_value, data in list(tag_values.items())[:5]:
                print(f"  {tag_value}: ${data['cost_usd']:.2f} "
                      f"({int(data['requests'])} requests)")
            print()

    # Top consumers
    if args.top:
        top_key = args.top_by or "team"
        top_metric = args.top_metric or "cost"
        top_list = get_top_consumers(
            metric=top_metric,
            tag_key=top_key,
            days=days,
            limit=10,
        )
        if top_list:
            print(f"Top {top_key.title()}s by {top_metric.title()}")
            print("-" * 30)
            for item in top_list:
                value = item.get(top_metric, 0)
                if top_metric == "cost":
                    print(f"  {item['tag_value']}: ${value:.2f}")
                elif top_metric == "energy":
                    print(f"  {item['tag_value']}: {value:.2f} Wh")
                else:
                    print(f"  {item['tag_value']}: {value:,}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="vetch", description="Vetch CLI")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Estimate
    est_parser = subparsers.add_parser("estimate", help="Estimate for a model")
    est_parser.add_argument(
        "--model", required=True, help="Model name (e.g., gpt-4o, claude-3-opus)"
    )
    est_parser.add_argument(
        "--input-tokens", type=positive_int, default=1000,
        help="Number of input tokens (must be positive)"
    )
    est_parser.add_argument(
        "--output-tokens", type=positive_int, default=1000,
        help="Number of output tokens (must be positive)"
    )
    est_parser.add_argument("--region", help="Grid region (e.g., us-east-1, eu-west-1)")
    est_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Compare
    comp_parser = subparsers.add_parser("compare", help="Compare multiple models")
    comp_parser.add_argument(
        "--models", required=True,
        help="Comma-separated model names (e.g., gpt-4o,claude-3-opus,gemini-1.5-pro)"
    )
    comp_parser.add_argument(
        "--input-tokens", type=positive_int, default=1000,
        help="Number of input tokens (must be positive)"
    )
    comp_parser.add_argument(
        "--output-tokens", type=positive_int, default=1000,
        help="Number of output tokens (must be positive)"
    )
    comp_parser.add_argument("--region", help="Grid region (e.g., us-east-1, eu-west-1)")
    comp_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # Methodology
    meth_parser = subparsers.add_parser("methodology", help="Show methodology")
    meth_parser.add_argument("--full", action="store_true", help="Show full methodology")
    meth_parser.add_argument("--contribute", action="store_true", help="Show contribution guide")

    # Check
    subparsers.add_parser("check", help="Check environment")

    # Quickstart
    subparsers.add_parser("quickstart", help="Show quickstart examples")

    # Clean
    subparsers.add_parser("clean", help="Clean up cache and lock files")

    # Audit
    audit_parser = subparsers.add_parser("audit", help="Analyze token usage patterns")
    audit_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Calibrate
    cal_parser = subparsers.add_parser(
        "calibrate", help="Calibrate energy using GPU power sensors or adjust PUE"
    )
    cal_parser.add_argument("--provider", help="Provider name (e.g., ollama, vllm)")
    cal_parser.add_argument("--model", help="Model name (e.g., llama3.1:8b)")
    cal_parser.add_argument("--workload", help="Workload command to run")
    cal_parser.add_argument(
        "--interactive", action="store_true", help="Interactive calibration mode"
    )
    cal_parser.add_argument(
        "--status", action="store_true", help="Show calibration status"
    )
    cal_parser.add_argument(
        "--pue", type=positive_float, metavar="VALUE",
        help="Set Power Usage Effectiveness (e.g., 1.2). Saves to config."
    )
    cal_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Report
    report_parser = subparsers.add_parser("report", help="Generate usage report")
    report_parser.add_argument(
        "--days", type=positive_int, default=7, help="Number of days to report (default: 7)"
    )
    report_parser.add_argument("--model", help="Filter by model name")
    report_parser.add_argument(
        "--tags", help="Filter by tags (e.g., 'team=ml,env=prod')"
    )
    report_parser.add_argument(
        "--top", action="store_true", help="Show top consumers"
    )
    report_parser.add_argument(
        "--top-by", dest="top_by", default="team", help="Tag to group top consumers by"
    )
    report_parser.add_argument(
        "--top-metric",
        dest="top_metric",
        choices=["cost", "energy", "tokens"],
        default="cost",
        help="Metric for top consumers",
    )
    report_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Config
    config_parser = subparsers.add_parser(
        "config", help="Manage Vetch configuration (PUE, region, etc.)"
    )
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )
    config_parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE",
        help="Set a config value (e.g., --set pue=1.2 --set region=us-east-1)"
    )
    config_parser.add_argument(
        "--unset", action="append", metavar="KEY",
        help="Remove a config value"
    )

    args = parser.parse_args()

    if args.command == "estimate":
        estimate(args)
    elif args.command == "compare":
        compare(args)
    elif args.command == "methodology":
        methodology(args)
    elif args.command == "check":
        check(args)
    elif args.command == "quickstart":
        quickstart(args)
    elif args.command == "clean":
        clean(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "calibrate":
        if args.status:
            calibrate_status(args)
        else:
            calibrate(args)
    elif args.command == "report":
        report(args)
    elif args.command == "config":
        config_cmd(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
