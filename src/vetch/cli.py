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
import re
import sys
import tempfile
from datetime import timedelta
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


_DURATION_PART_RE = re.compile(
    r"\s*(?P<number>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>w|week|weeks|d|day|days|h|hr|hrs|hour|hours|m|min|mins|minute|minutes)\s*",
    re.IGNORECASE,
)


def parse_duration(value: str) -> timedelta:
    """Parse a strict duration string such as ``6h``, ``7d``, or ``1h30m``."""
    raw = value.strip()
    if not raw:
        raise argparse.ArgumentTypeError("duration cannot be empty")

    position = 0
    total_seconds = 0.0
    matched = False
    for match in _DURATION_PART_RE.finditer(raw):
        if match.start() != position:
            raise argparse.ArgumentTypeError(
                f"invalid duration '{value}'. Use forms like 6h, 7d, or 1h30m"
            )
        matched = True
        amount = float(match.group("number"))
        unit = match.group("unit").lower()
        if unit in {"w", "week", "weeks"}:
            total_seconds += amount * 7 * 86400
        elif unit in {"d", "day", "days"}:
            total_seconds += amount * 86400
        elif unit in {"h", "hr", "hrs", "hour", "hours"}:
            total_seconds += amount * 3600
        elif unit in {"m", "min", "mins", "minute", "minutes"}:
            total_seconds += amount * 60
        position = match.end()

    if not matched or position != len(raw) or total_seconds <= 0:
        raise argparse.ArgumentTypeError(
            f"invalid duration '{value}'. Use forms like 6h, 7d, or 1h30m"
        )

    return timedelta(seconds=total_seconds)


def _parse_tag_filter(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    tags: dict[str, str] = {}
    for tag_spec in raw.split(","):
        if "=" in tag_spec:
            key, value = tag_spec.split("=", 1)
            tags[key.strip()] = value.strip()
    return tags or None


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

    carbon_g, pue, pue_tier, pue_source = calculate_carbon(
        energy_wh, grid.intensity_gco2e_kwh, model=args.model
    )

    cost_usd, _, _, _, _, _ = calculate_cost(
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
            "pue": pue,
            "pue_tier": pue_tier,
            "pue_source": pue_source,
            "grid_region": args.region or "global",
            "grid_intensity": grid.intensity_gco2e_kwh,
            "signal_quality": grid.signal_quality,
        }
        print(json.dumps(result, indent=2))
        return

    # Text output with uncertainty indicators
    uncertainty_label = (
        f"±{uncertainty_pct}%" if uncertainty_pct < 1000 else "order of magnitude"
    )
    print(f"Energy:  ~{energy_wh:.2f} Wh ({uncertainty_label})  [Tier {tier}]")
    intensity = grid.intensity_gco2e_kwh
    pue_label = f"PUE {pue:.2f}" if pue_tier == 1 else f"PUE ~{pue:.2f}"
    carbon_line = (
        f"Carbon:  ~{carbon_g:.2f}g       "
        f"[{intensity:.0f} gCO2e/kWh, {pue_label}, {grid.signal_quality}]"
    )
    print(carbon_line)
    print(f"Cost:    ${cost_usd:.2f}       [list pricing]")
    print()
    print(f"Energy basis: {basis}")
    print(f"PUE source:   {pue_source}")
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
        carbon_g, _, _, _ = calculate_carbon(energy_wh, grid.intensity_gco2e_kwh, model=model)
        cost_usd, _, _, _, _, _ = calculate_cost(
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
    print("All estimates are tier 3 (order of magnitude uncertainty).")
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
        from opentelemetry import trace
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
VETCH_DEFAULT_PUE     - Power Usage Effectiveness (default: 1.2)
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
    """Generate a stored metadata audit, falling back to current-session advisories."""
    from datetime import datetime, timezone

    from vetch.audit_report import build_audit_report, format_audit_report

    if not getattr(args, "session", False):
        window = getattr(args, "window", timedelta(days=7))
        end = datetime.now(timezone.utc)
        start = end - window
        expected_caps = getattr(args, "expected_capabilities", None)
        expected_list = None
        if expected_caps:
            expected_list = [s.strip() for s in expected_caps.split(",") if s.strip()]
        report = build_audit_report(
            start=start,
            end=end,
            model=getattr(args, "model", None),
            tags=_parse_tag_filter(getattr(args, "tags", None)),
            expected_capabilities=expected_list,
        )
        if report.total_requests > 0 or getattr(args, "stored", False):
            print(format_audit_report(report, getattr(args, "format", "text")))
            return

    _audit_session(args)


def _audit_session(args: argparse.Namespace) -> None:
    """Analyze current in-memory token usage patterns and generate advisories."""
    from vetch.advisory import format_advisories, generate_advisories
    from vetch.stats import get_session_stats

    stats = get_session_stats()

    if stats.total_requests == 0:
        if getattr(args, "format", "text") == "json":
            print(json.dumps({
                "total_requests": 0,
                "advisories": [],
                "message": "No requests recorded in this session.",
            }, indent=2))
            return
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


def calibrate_apple_silicon_cmd(args: argparse.Namespace) -> None:
    """Calibrate energy using Apple Silicon powermetrics (requires sudo)."""
    from vetch.calibrate_metal import (
        calibrate_apple_silicon,
        download_calibration_images,
        format_calibration_result_apple,
        is_apple_silicon,
    )

    if args.fetch_images:
        download_calibration_images(strict=getattr(args, "strict_images", False))
        return

    if not is_apple_silicon():
        print(
            "ERROR: calibrate-apple-silicon requires macOS on Apple Silicon.\n"
            "Use 'vetch calibrate-cuda' for NVIDIA GPU calibration.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = args.model or "moondream:latest"
    provider = args.provider or "self-hosted"
    base_url = args.base_url or "http://localhost:11434"
    iterations = args.iterations or 1
    verbose = args.verbose
    if not args.precision:
        print(
            "ERROR: --precision is required (e.g. apple-native, gguf:q4_k_m).\n"
            "It is part of the calibration identity; omitting it lets distinct\n"
            "quantizations overwrite each other and resolve as exact Tier 0.",
            file=sys.stderr,
        )
        sys.exit(2)
    precision = args.precision
    serving_engine = args.serving_engine or "ollama"
    from vetch.calibration_store import is_cloud_provider
    if is_cloud_provider(provider):
        print(
            f"ERROR: --provider {provider} is a cloud/API vendor and is refused for "
            "calibration (ambiguous with the real hosted API). Use --provider "
            "self-hosted or ollama.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        result, record_path = calibrate_apple_silicon(
            model=model,
            provider=provider,
            base_url=base_url,
            iterations=iterations,
            verbose=verbose,
            precision=precision,
            serving_engine=serving_engine,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        import dataclasses
        import json as _json
        print(_json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(format_calibration_result_apple(result, record_path))

    if not result.active:
        if result.rejection_reasons:
            print("Active calibration NOT installed:", file=sys.stderr)
            for reason in result.rejection_reasons:
                print(f"  - {reason}", file=sys.stderr)
        sys.exit(1)


def calibrate_cuda_cmd(args: argparse.Namespace) -> None:
    """Calibrate energy on an NVIDIA GPU using NVML (no sudo)."""
    from vetch.calibrate_cuda import (
        calibrate_cuda,
        calibrate_cuda_batched,
        is_cuda_available,
    )

    if not is_cuda_available():
        print(
            "ERROR: calibrate-cuda requires an NVIDIA GPU visible to NVML.\n"
            "Confirm 'nvidia-smi' works and 'pip install nvidia-ml-py'.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = args.model or "moondream:latest"
    provider = args.provider or "self-hosted"
    iterations = args.iterations or 1

    if not args.precision:
        print(
            "ERROR: --precision is required (e.g. bf16, fp8-e4m3, gguf:q4_k_m).\n"
            "It is part of the calibration identity; omitting it lets distinct\n"
            "quantizations overwrite each other and resolve as exact Tier 0.",
            file=sys.stderr,
        )
        sys.exit(2)
    if args.backend == "openai" and not args.serving_engine:
        print(
            "ERROR: --serving-engine is required when --backend openai\n"
            "(e.g. vllm, sglang). Otherwise distinct stacks collide under\n"
            "a generic 'openai' backend key.",
            file=sys.stderr,
        )
        sys.exit(2)
    from vetch.calibration_store import is_cloud_provider
    if is_cloud_provider(provider):
        print(
            f"ERROR: --provider {provider} is a cloud/API vendor and is refused for "
            "calibration (ambiguous with the real hosted API). Use --provider "
            "self-hosted or vllm.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        if getattr(args, "batched", False):
            conc_raw = getattr(args, "concurrency", None) or "1,4,8,16,32"
            try:
                concs = tuple(
                    int(x.strip()) for x in str(conc_raw).split(",") if x.strip()
                )
            except ValueError:
                print(
                    "ERROR: --concurrency must be a comma-separated list of ints "
                    "(e.g. 1,4,8,16,32).",
                    file=sys.stderr,
                )
                sys.exit(2)
            print(
                "NOTE: --batched is an EXPERIMENTAL preview. It sweeps concurrency "
                "and fits Wh/1k ≈ a/C + b, but its output is NOT Tier-0 and no "
                "batched records ship with vetch. Use the default batch=1 path for "
                "reproducible calibration.",
                file=sys.stderr,
            )
            results = calibrate_cuda_batched(
                model=model,
                provider=provider,
                base_url=args.base_url,
                device_id=args.device,
                verbose=args.verbose,
                backend=args.backend,
                precision=args.precision,
                serving_engine=args.serving_engine,
                concurrencies=concs,
                requests_per_level=getattr(args, "requests_per_level", 64) or 64,
                out_tokens=getattr(args, "out_tokens", 64) or 64,
            )
            if args.format == "json":
                import dataclasses
                import json as _json
                print(_json.dumps([dataclasses.asdict(r) for r in results], indent=2))
            else:
                print(f"\nBatched calibration complete: {len(results)} concurrency records.")
                for c, r in zip(concs, results):
                    img = (
                        f"  wh_per_image={r.wh_per_image:.6f}"
                        if r.wh_per_image is not None else ""
                    )
                    print(
                        f"  C={c}: wh_per_1k_output={r.wh_per_1k_output:.6f}{img} "
                        f"active={r.active}"
                    )
            if any(not r.active for r in results):
                sys.exit(1)
            return

        result = calibrate_cuda(
            model=model,
            provider=provider,
            base_url=args.base_url,  # None -> per-backend default in calibrate_cuda
            device_id=args.device,
            iterations=iterations,
            verbose=args.verbose,
            backend=args.backend,
            precision=args.precision,
            serving_engine=args.serving_engine,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.format == "json":
        import dataclasses
        import json as _json
        print(_json.dumps(dataclasses.asdict(result), indent=2))
    else:
        print(f"\nModel:        {result.model}")
        print(f"GPU:          {result.gpu_name}")
        print(f"Tier:         {result.tier} (hardware-measured)")
        print(f"wh_per_1k_in: {result.wh_per_1k_input:.5f}")
        print(f"wh_per_1k_out:{result.wh_per_1k_output:.5f}")
        if result.wh_per_image is not None:
            print(f"wh_per_image: {result.wh_per_image:.5f}")
            print(f"visual_tok/img:{result.visual_tokens_per_image}")
        print(f"intercept_wh: {result.intercept_wh}")
        print(f"samples:      {result.samples}")
        status = "ACTIVE (saved to ~/.vetch/calibrations/)" if result.active else "NOT installed"
        print(f"Status:       {status}")

    if not result.active:
        if result.rejection_reasons:
            print("Active calibration NOT installed:", file=sys.stderr)
            for reason in result.rejection_reasons:
                print(f"  - {reason}", file=sys.stderr)
        sys.exit(1)


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
            print("  pue      - Power Usage Effectiveness (default: 1.2)")
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

        print(f"\nTo apply PUE to calculations, set VETCH_DEFAULT_PUE={config.get('pue', 1.2)}")
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
    tags = _parse_tag_filter(args.tags)

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


def savings(args: argparse.Namespace) -> None:
    """Show realized savings and circuit breaker intervention summary."""
    from datetime import datetime, timedelta

    from vetch.storage import get_db_path, is_storage_enabled, query_usage

    if not is_storage_enabled():
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

    days = getattr(args, "days", 30)
    end = datetime.now()
    start = end - timedelta(days=days)

    summary = query_usage(start=start, end=end)

    if getattr(args, "format", "text") == "json":
        import json as _json
        out = {
            "period_days": days,
            "requests": summary.total_requests,
            "realized_cache_savings_usd": summary.total_cache_cost_saving_usd,
            "realized_cache_energy_savings_wh": summary.total_cache_energy_saving_wh,
            "realized_cache_carbon_savings_g": summary.total_cache_carbon_saving_g,
            "circuit_breaker_interventions": summary.total_circuit_breaker_interventions,
            "intervention_cost_at_risk_usd": summary.total_intervention_cost_at_risk_usd,
        }
        print(_json.dumps(out, indent=2))
        return

    sep = "━" * 48
    print(f"\nSAVINGS SUMMARY  (last {days} days)")
    print(sep)
    print(f"Requests tracked:              {summary.total_requests:,}")
    print("")
    print("Realized cache savings")
    print(f"  Cost saved via caching:      ${summary.total_cache_cost_saving_usd:,.2f}")
    print(f"  Energy saved via caching:    {summary.total_cache_energy_saving_wh:,.2f} Wh")
    print(f"  Carbon saved via caching:    {summary.total_cache_carbon_saving_g:,.2f} gCO2e")
    print("")
    print("Circuit breaker interventions")
    print(f"  Interventions:               {summary.total_circuit_breaker_interventions}")
    print(f"  Cost at risk interrupted:    ${summary.total_intervention_cost_at_risk_usd:,.2f}")
    print("")
    print(f"Total realized cache savings:  ${summary.total_cache_cost_saving_usd:,.2f}")
    if days > 0 and summary.total_cache_cost_saving_usd > 0:
        monthly = (summary.total_cache_cost_saving_usd / days) * 30
        print(f"Monthly run-rate:              ${monthly:,.2f} / month")
    print(sep)
    print(f"Generated by Vetch v{__version__}")
    print("")


def status(args: argparse.Namespace) -> None:
    """Show Vetch status including registry, connectivity, and providers."""
    print(f"Vetch v{__version__}\n")

    # 1. Registry status
    print("Registry:")
    from vetch.calculation import _ENERGY_PATH

    bundled_mtime = _ENERGY_PATH.stat().st_mtime if _ENERGY_PATH.exists() else 0
    from datetime import datetime

    if bundled_mtime:
        bundled_date = datetime.fromtimestamp(bundled_mtime).strftime("%Y-%m-%d")
    else:
        bundled_date = "unknown"
    print(f"  Bundled version: {bundled_date}")

    # Check remote registry
    remote_status = "disabled"
    remote_version = "n/a"
    try:
        from vetch.registry.remote import get_remote_fetcher

        fetcher = get_remote_fetcher()
        if fetcher is not None:
            if fetcher.has_remote_data:
                import time

                age_s = time.monotonic() - fetcher.last_fetch_time
                if age_s < 3600:
                    age_str = f"{int(age_s / 60)}m ago"
                else:
                    age_str = f"{int(age_s / 3600)}h ago"
                remote_version = f"fetched {age_str}"
                remote_status = "connected"
            else:
                remote_status = "no data yet"
        else:
            remote_status = "disabled"
    except Exception:
        remote_status = "error"

    print(f"  Remote status:   {remote_status}")
    if remote_version != "n/a":
        print(f"  Remote data:     {remote_version}")

    offline_path = os.environ.get("VETCH_REGISTRY_PATH")
    print(f"  Offline mode:    {'true (' + offline_path + ')' if offline_path else 'false'}")
    print()

    # 2. Grid API
    print("Grid API:")
    api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY")
    if api_key:
        print("  API key:         configured")
        try:
            from vetch.sensing.grid import get_carbon_intensity

            grid = get_carbon_intensity("us-east-1")
            print(f"  Status:          connected ({grid.signal_quality})")
        except Exception as e:
            print(f"  Status:          error ({e})")
    else:
        print("  API key:         not set")
        print("  Status:          using fallback data")
    print()

    # 3. Providers
    print("Providers:")
    providers = {
        "OpenAI": ("vetch.providers.openai", "_module_instrumented"),
        "Azure OpenAI": ("vetch.providers.azure_openai", "_module_instrumented"),
        "Anthropic": ("vetch.providers.anthropic", "_module_instrumented"),
        "Vertex AI": ("vetch.providers.vertexai", "_module_instrumented"),
    }

    for name, (module_path, flag) in providers.items():
        try:
            import importlib

            mod = importlib.import_module(module_path)
            instrumented = getattr(mod, flag, False)
            sdk_name = module_path.split(".")[-1]

            # Check if SDK is installed
            if sdk_name in ("openai", "azure_openai"):
                sdk_installed = "openai" in sys.modules
            elif sdk_name == "anthropic":
                sdk_installed = "anthropic" in sys.modules
            elif sdk_name == "vertexai":
                sdk_installed = (
                    "google.cloud.aiplatform" in sys.modules
                    or "vertexai" in sys.modules
                )
            else:
                sdk_installed = False

            if instrumented:
                print(f"  {name + ':':<17} instrumented")
            elif sdk_installed:
                print(f"  {name + ':':<17} detected (not instrumented)")
            else:
                print(f"  {name + ':':<17} not detected")
        except ImportError:
            print(f"  {name + ':':<17} not detected")
    print()

    # 4. Budgets
    print("Budgets:")
    try:
        from vetch.budget import get_budget_status

        budgets = get_budget_status()
        if budgets:
            for budget_name, budget_info in budgets.items():
                used = budget_info.get("accumulated", 0)
                limit = budget_info.get("limit", 0)
                pct = (used / limit * 100) if limit > 0 else 0
                print(f"  {budget_name}: {used:.2f} / {limit:.2f} ({pct:.1f}%)")
        else:
            print("  No budgets configured")
    except Exception:
        print("  No budgets configured")
    print()

    # 5. Config
    print("Config:")
    config_vars = {
        "VETCH_REGION": os.environ.get("VETCH_REGION", "not set"),
        "VETCH_DISABLED": os.environ.get("VETCH_DISABLED", "false"),
        "VETCH_OUTPUT": os.environ.get("VETCH_OUTPUT", "stderr"),
        "VETCH_DEFAULT_PUE": os.environ.get("VETCH_DEFAULT_PUE", "1.2"),
        "VETCH_HOME": os.environ.get("VETCH_HOME", str(Path.home() / ".vetch")),
    }
    for var, val in config_vars.items():
        print(f"  {var}: {val}")


def dashboard(args: argparse.Namespace) -> None:
    """Export dashboard templates."""
    dashboard_dir = Path(__file__).parent / "dashboards"

    if args.list:
        print("Available dashboard templates:")
        if dashboard_dir.exists():
            for f in sorted(dashboard_dir.glob("*.json")):
                print(f"  - {f.stem}")
        else:
            print("  No dashboard templates found.")
        return

    export_type = args.export
    if export_type == "grafana":
        template_path = dashboard_dir / "grafana_vetch.json"
    else:
        print(f"Unknown dashboard type: {export_type}")
        print("Available types: grafana")
        sys.exit(1)

    if not template_path.exists():
        print(f"Dashboard template not found: {template_path}")
        sys.exit(1)

    content = template_path.read_text()

    if args.output:
        output_path = Path(args.output).resolve()

        # Security: Prevent path traversal attacks
        # Only allow output in current directory, subdirectories, or system temp directory
        cwd = Path.cwd().resolve()
        tmp = Path(tempfile.gettempdir()).resolve()

        in_cwd = False
        in_tmp = False
        try:
            output_path.relative_to(cwd)
            in_cwd = True
        except ValueError:
            pass

        try:
            output_path.relative_to(tmp)
            in_tmp = True
        except ValueError:
            pass

        if not (in_cwd or in_tmp):
            print("Error: Output path must be within current directory or temp directory")
            print(f"  Current directory: {cwd}")
            print(f"  Temp directory: {tmp}")
            print(f"  Attempted path: {output_path}")
            sys.exit(1)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Dashboard exported to {output_path}")
    else:
        print(content)


def registry_freeze(args: argparse.Namespace) -> None:
    """Freeze remote registry to a local file."""
    from vetch.registry.remote import freeze_registry

    output = args.output or "vetch_registry.json"
    print(f"Freezing registry to {output}...")

    success = freeze_registry(output)
    if success:
        print(f"Registry frozen successfully to {output}")
        print("Use VETCH_REGISTRY_PATH to load this in CI/CD:")
        print(f"  export VETCH_REGISTRY_PATH={Path(output).parent}")
    else:
        print("Failed to freeze registry. Check network connectivity.")
        sys.exit(1)


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
    audit_parser = subparsers.add_parser("audit", help="Generate inference waste audit")
    audit_parser.add_argument(
        "--window",
        type=parse_duration,
        default=timedelta(days=7),
        help="Stored audit window, e.g. 6h, 7d, 1w, or 1h30m (default: 7d)",
    )
    audit_parser.add_argument("--model", help="Filter stored audit by model name")
    audit_parser.add_argument(
        "--tags", help="Filter stored audit by tags (e.g., 'team=ml,env=prod')"
    )
    audit_parser.add_argument(
        "--stored",
        action="store_true",
        help="Force stored audit output even when no stored events are found",
    )
    audit_parser.add_argument(
        "--session",
        action="store_true",
        help="Use current in-process session stats instead of stored metadata",
    )
    audit_parser.add_argument(
        "--expected-capabilities",
        metavar="CAPS",
        help=(
            "Comma-separated capability manifest for CAP-001 "
            "(e.g. model:image,model:embedding)"
        ),
    )
    audit_parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
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

    # Calibrate Apple Silicon
    cal_as_parser = subparsers.add_parser(
        "calibrate-apple-silicon",
        help="Calibrate energy on Apple Silicon using powermetrics (requires sudo)",
    )
    cal_as_parser.add_argument(
        "--model", default="moondream:latest", help="Ollama model (default: moondream:latest)"
    )
    cal_as_parser.add_argument(
        "--provider", default="self-hosted",
        help="Provider label for the identity (default: self-hosted)",
    )
    cal_as_parser.add_argument(
        "--base-url", dest="base_url", default="http://localhost:11434", help="Ollama API base URL"
    )
    cal_as_parser.add_argument(
        "--precision", required=True,
        help="Precision identity dimension (e.g. apple-native, gguf:q4_k_m). "
             "Required so distinct quantizations do not collide.",
    )
    cal_as_parser.add_argument(
        "--serving-engine", dest="serving_engine", default="ollama",
        help="Serving engine label (default: ollama)",
    )
    cal_as_parser.add_argument(
        "--iterations", type=int, default=1,
        help="Grid iteration multiplier (default: 1, ~22 runs)",
    )
    cal_as_parser.add_argument("--verbose", action="store_true", help="Print per-run details")
    cal_as_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )
    cal_as_parser.add_argument(
        "--fetch-images", dest="fetch_images", action="store_true",
        help="Download the Wikimedia standard image set (no sudo needed) and exit",
    )
    cal_as_parser.add_argument(
        "--strict-images",
        dest="strict_images",
        action="store_true",
        help="With --fetch-images, fail if any Wikimedia download is missing",
    )

    # Calibrate NVIDIA GPU (NVML energy counter; no sudo)
    cal_cuda_parser = subparsers.add_parser(
        "calibrate-cuda",
        help="Calibrate energy on an NVIDIA GPU using NVML (VLM-aware, no sudo)",
    )
    cal_cuda_parser.add_argument(
        "--model", default="moondream:latest",
        help="Model name: Ollama tag, or HF repo id for vLLM "
             "(e.g. Qwen/Qwen2.5-32B-Instruct). Default: moondream:latest",
    )
    cal_cuda_parser.add_argument(
        "--provider", default="self-hosted",
        help="Provider label written into the calibration identity "
             "(default: self-hosted). Must match production provider_hint "
             "for an exact Tier-0 load; bare 'openai' events do not cross-match.",
    )
    cal_cuda_parser.add_argument(
        "--backend", choices=["ollama", "openai"], default="ollama",
        help="Serving backend: 'ollama' (/api) or 'openai' (OpenAI-compatible "
             "/chat/completions, e.g. vLLM in BF16). Default: ollama",
    )
    cal_cuda_parser.add_argument(
        "--base-url", dest="base_url", default=None,
        help="Server base URL. Default: http://localhost:11434 (ollama) or "
             "http://localhost:8000/v1 (openai/vLLM)",
    )
    cal_cuda_parser.add_argument(
        "--device", type=int, default=0, help="NVML device index to measure (default: 0)"
    )
    cal_cuda_parser.add_argument(
        "--precision", required=True,
        help="REQUIRED. Precision/quantization identity dimension "
             "(e.g. bf16, fp8-e4m3, gguf:q4_k_m). Distinct values must not share a file.",
    )
    cal_cuda_parser.add_argument(
        "--serving-engine", dest="serving_engine", default=None,
        help="Serving engine label for the identity (e.g. vllm, sglang). "
             "Required when --backend openai. Defaults to --backend for ollama.",
    )
    cal_cuda_parser.add_argument(
        "--iterations", type=int, default=1,
        help="Grid iteration multiplier (default: 1, ~22 runs)",
    )
    cal_cuda_parser.add_argument(
        "--batched", action="store_true",
        help="Production-representative concurrency sweep: measure Wh/1k_output "
             "at each concurrency and write concurrency-keyed records + amortization "
             "curve (does not replace the default batch=1 grid).",
    )
    cal_cuda_parser.add_argument(
        "--concurrency", default="1,4,8,16,32",
        help="With --batched: comma-separated concurrency levels "
             "(default: 1,4,8,16,32).",
    )
    cal_cuda_parser.add_argument(
        "--requests-per-level", dest="requests_per_level", type=int, default=64,
        help="With --batched: requests fired at each concurrency (default: 64).",
    )
    cal_cuda_parser.add_argument(
        "--out-tokens", dest="out_tokens", type=int, default=64,
        help="With --batched: target completion tokens per request (default: 64).",
    )
    cal_cuda_parser.add_argument("--verbose", action="store_true", help="Print per-run details")
    cal_cuda_parser.add_argument(
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

    # Savings summary
    savings_parser = subparsers.add_parser("savings", help="Show savings and intervention summary")
    savings_parser.add_argument(
        "--days", type=positive_int, default=30, help="Number of days to report (default: 30)"
    )
    savings_parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format"
    )

    # Status
    subparsers.add_parser("status", help="Show Vetch status and configuration")

    # Dashboard
    dash_parser = subparsers.add_parser(
        "dashboard", help="Export dashboard templates (Grafana, etc.)"
    )
    dash_parser.add_argument(
        "--export", default="grafana",
        help="Dashboard type to export (default: grafana)"
    )
    dash_parser.add_argument(
        "--output", "-o", help="Write to file instead of stdout"
    )
    dash_parser.add_argument(
        "--list", action="store_true", help="List available dashboard templates"
    )

    # Registry
    reg_parser = subparsers.add_parser(
        "registry", help="Manage model registry"
    )
    reg_subparsers = reg_parser.add_subparsers(dest="registry_command")
    freeze_parser = reg_subparsers.add_parser(
        "freeze", help="Freeze remote registry to local file"
    )
    freeze_parser.add_argument(
        "--output", "-o", default="vetch_registry.json",
        help="Output file path (default: vetch_registry.json)"
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
    elif args.command == "calibrate-apple-silicon":
        calibrate_apple_silicon_cmd(args)
    elif args.command == "calibrate-cuda":
        calibrate_cuda_cmd(args)
    elif args.command == "report":
        report(args)
    elif args.command == "savings":
        savings(args)
    elif args.command == "config":
        config_cmd(args)
    elif args.command == "status":
        status(args)
    elif args.command == "dashboard":
        dashboard(args)
    elif args.command == "registry":
        if args.registry_command == "freeze":
            registry_freeze(args)
        else:
            # Show registry subcommand help
            parser.parse_args(["registry", "--help"])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
