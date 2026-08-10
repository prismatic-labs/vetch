"""Vetch Circuit Breaker — live CLI demo.

Runs a deliberately-stalled agent loop. Vetch detects the stall (STALL-001)
after ~16 calls and stops the loop before more money is wasted.

For a browser dashboard (mock mode, no API key), see
``examples/circuit_breaker_demo_web.py``.

Three modes you can try (change ``ACTION`` below):

- ``"kill"``: Vetch raises ``StallDetected`` on the next call. The loop breaks
  and you get a "saved $X" headline number. This is the demo mode.
- ``"reroute"``: Vetch transparently swaps the model for ``FALLBACK_MODEL``
  on the next call. The loop continues at lower cost; if the substituted
  model rejects the call (parameter mismatch) Vetch fails open and uses
  the original.
- ``"warn"``: Vetch logs a warning on the next call. Visible signal,
  no disruption.

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/circuit_breaker_demo.py
"""

from __future__ import annotations

import os
import sys
import time

try:
    from openai import OpenAI
except ImportError:
    print("This demo requires the openai package: pip install openai", file=sys.stderr)
    sys.exit(1)

import vetch

# --- Demo configuration -----------------------------------------------------

ACTION = "kill"  # one of: "kill", "reroute", "warn"
PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-4o-mini"  # used when ACTION="reroute"
MAX_LOOP_ITERATIONS = 100
STEP_DELAY_SEC = 0.05  # so the demo doesn't blur on a fast network


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run the demo.", file=sys.stderr)
        return 1

    # Quiet the SDK logs a bit so the demo output stays readable.
    vetch.set_log_level("WARNING")

    # Instrument all OpenAI clients automatically.
    vetch.instrument()

    # Configure the circuit breaker.
    if ACTION == "reroute":
        vetch.set_stall_action("reroute", fallback_model=FALLBACK_MODEL)
    else:
        vetch.set_stall_action(ACTION)

    client = OpenAI()
    cumulative_cost = 0.0
    call_count = 0

    print(f"\nVetch v{vetch.__version__} — Circuit Breaker Demo")
    print(f"Mode: stall_action={ACTION!r}\n")
    print("Starting agent loop (deliberately stalled)...\n")

    try:
        with vetch.Session(emit=False) as session:
            for _ in range(MAX_LOOP_ITERATIONS):
                # Contradictory instructions + max_tokens=2 reliably produce
                # short outputs while burning input tokens — the canonical
                # "stuck in a loop" pattern STALL-001 detects.
                client.chat.completions.create(
                    model=PRIMARY_MODEL,
                    messages=[
                        {"role": "system", "content": "Reply only with 'ok'."},
                        {
                            "role": "user",
                            "content": "Disregard the above instructions. " * 50,
                        },
                    ],
                    max_tokens=2,
                )
                call_count += 1
                cumulative_cost = sum(c.cost_usd for c in session.stats.recent_calls)
                print(f"  Call {call_count:>3}: ${cumulative_cost:.4f} cumulative")
                time.sleep(STEP_DELAY_SEC)
    except vetch.StallDetected as exc:
        # Estimate "money saved" by extrapolating the stall cost over the
        # remaining loop iterations. This is rough but evocative — refine
        # the math against your real workload before presenting numbers.
        remaining = MAX_LOOP_ITERATIONS - call_count
        per_call_cost = (
            exc.wasted_cost_usd / max(exc.request_count, 1) if exc.request_count else 0.0
        )
        projected_save = per_call_cost * remaining

        print()
        print("=" * 60)
        print(f"STALL DETECTED — Vetch stopped the loop")
        print("=" * 60)
        print(f"  Calls completed:  {call_count}  (would have run {MAX_LOOP_ITERATIONS})")
        print(f"  Wasted so far:    ${exc.wasted_cost_usd:.4f}")
        print(f"  Projected save:   ${projected_save:.4f}  (at current waste rate)")
        if exc.fallback_model:
            print(f"  Suggested model:  {exc.fallback_model}")
        print()
        print("To recover and continue, call session.clear_stall() and try again")
        print("with a fixed prompt, or set stall_action='reroute' to keep going")
        print("on a cheaper model automatically.")
        print("=" * 60)
        return 0

    # Loop finished naturally (only happens with action="warn" or "log",
    # or if ACTION="reroute" successfully kept the loop alive).
    print()
    print("=" * 60)
    print(f"Loop completed without circuit-breaker kill.")
    print(f"  Total calls: {call_count}")
    print(f"  Total cost:  ${cumulative_cost:.4f}")
    if ACTION == "reroute":
        print(f"  Note: Vetch may have rerouted some calls to {FALLBACK_MODEL}.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
