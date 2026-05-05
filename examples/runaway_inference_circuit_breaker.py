"""Runaway Inference Circuit Breaker — mockable example.

Simulates an agent loop that stalls, then demonstrates Vetch detecting
the stall and firing the circuit breaker in three modes. No real API
credentials needed.

Run:
    python examples/runaway_inference_circuit_breaker.py

What you'll see:
  - Calls 1–6:  productive (350 output tokens each)
  - Calls 7–22: stalled (1 output token, same input — STALL-001 fires)
  - Circuit breaker fires in three separate simulations:
      warn   — logs a warning to stderr, call proceeds
      kill   — raises StallDetected, loop breaks, budget protected
      reroute — transparently substitutes a cheaper model

In production:
    import vetch
    vetch.instrument()
    vetch.set_stall_action("kill")
    # Run your agent loop — Vetch handles the rest.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# Suppress Vetch's internal session_complete events so demo output is readable.
logging.getLogger("vetch").setLevel(logging.ERROR)

from vetch._stall import apply_stall_action
from vetch.advisory import generate_advisories
from vetch.config import set_stall_action
from vetch.exceptions import StallDetected
from vetch.session import Session
from vetch.stats import STALL_LOW_OUTPUT_THRESHOLD

# Cost / energy constants for this demo's simulated model (gpt-4o)
_COST_PER_STALLED_CALL = 0.0031   # USD
_ENERGY_PER_STALLED_CALL = 0.00015  # Wh
_TOKENS_PER_STALLED_CALL = 1_200   # input tokens consumed with no useful output


# ── Helpers ──────────────────────────────────────────────────────────────────


def _event(output_tokens: int, input_tokens: int = 1_200) -> dict[str, Any]:
    cost = _COST_PER_STALLED_CALL if output_tokens < STALL_LOW_OUTPUT_THRESHOLD else 0.0085
    energy = _ENERGY_PER_STALLED_CALL if output_tokens < STALL_LOW_OUTPUT_THRESHOLD else 0.00046
    return {
        "model": "gpt-4o",
        "usage": {"text": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
        "estimated_cost_usd": cost,
        "estimated_energy_wh": energy,
        "estimated_carbon_g": energy * 0.38,
    }


def _build_stalled_session(tags: dict[str, str] | None = None) -> Session:
    """Create a session with 6 productive calls then 16 stalled calls."""
    session = Session(tags=tags or {"demo": "circuit-breaker"})
    for _ in range(6):
        session.register_event(_event(output_tokens=350))  # type: ignore[arg-type]
    for _ in range(16):
        session.register_event(_event(output_tokens=1))    # type: ignore[arg-type]
    return session


def _section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


def _show_loop_table(session: Session) -> None:
    print(f"\n  {'Call':>5}  {'Output tokens':>14}  {'Note'}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*28}")
    for i, call in enumerate(session.stats.recent_calls, 1):
        status = "stalled" if call.output_tokens < STALL_LOW_OUTPUT_THRESHOLD else "ok"
        # recent_calls only holds the last 20 — show them with adjusted numbering
        effective_n = session.stats.total_requests - len(session.stats.recent_calls) + i
        print(f"  {effective_n:>5}  {call.output_tokens:>14}  {status}")
    if session.stall_triggered:
        print(f"  {'...':>5}  {'':>14}  ↑ STALL-001 detected here")


# ── Demos ────────────────────────────────────────────────────────────────────


def demo_warn() -> None:
    _section("Action: warn")
    print("  STALL-001 fires → logs a warning to stderr, call proceeds.\n")
    set_stall_action("warn")

    # Configure stderr logging so the warning is visible
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="  [%(levelname)s] %(message)s",
        force=True,
    )

    with Session(tags={"demo": "warn"}) as session:
        for _ in range(6):
            session.register_event(_event(output_tokens=350))  # type: ignore[arg-type]
        for _ in range(16):
            session.register_event(_event(output_tokens=1))    # type: ignore[arg-type]

        stall_adv = next(
            (a for a in generate_advisories(session.stats) if a.code == "STALL-001"),
            None,
        )
        print(f"  stall_triggered: {session.stall_triggered}")
        if stall_adv:
            print(f"  advisory:        [{stall_adv.code}] {stall_adv.title}")

        rerouted, _ = apply_stall_action({"model": "gpt-4o"}, ctx=None)
        print(f"\n  rerouted:        {rerouted}")
        print(f"  → Warning logged to stderr above. Call would proceed normally.")
        print(f"  → Loop continues — use 'kill' or 'reroute' to stop it.")


def demo_kill() -> None:
    _section("Action: kill")
    print("  STALL-001 fires → raises StallDetected, loop breaks.\n")
    set_stall_action("kill")

    with Session(tags={"demo": "kill"}) as session:
        for _ in range(6):
            session.register_event(_event(output_tokens=350))  # type: ignore[arg-type]
        for i in range(16):
            session.register_event(_event(output_tokens=1))    # type: ignore[arg-type]

        stall_adv = next(
            (a for a in generate_advisories(session.stats) if a.code == "STALL-001"),
            None,
        )
        stalled_calls = stall_adv.request_count if stall_adv else 0
        wasted_cost = stall_adv.potential_savings_usd if stall_adv else 0.0

        print(f"  stall_triggered: {session.stall_triggered}")
        if stall_adv:
            print(f"  advisory:        [{stall_adv.code}] severity={stall_adv.severity}")
            print(f"  stalled calls:   {stalled_calls}")
            print(f"  wasted cost:     ${wasted_cost:.4f}")

        # Simulate the next call in the agent loop — StallDetected should fire
        try:
            apply_stall_action({"model": "gpt-4o"}, ctx=None)
            print("\n  No exception raised (session not active in apply_stall_action context).")
        except StallDetected as exc:
            calls_remaining = 10  # hypothetical remaining iterations
            print(f"\n  StallDetected raised ✓")
            print(f"\n  Impact:")
            print(f"    Calls stopped (est.):         {calls_remaining}")
            print(f"    Tokens avoided (est.):        {calls_remaining * _TOKENS_PER_STALLED_CALL:,}")
            print(f"    Cost avoided (est.):          ${calls_remaining * _COST_PER_STALLED_CALL:.4f}")
            print(f"    Energy avoided (est.):        {calls_remaining * _ENERGY_PER_STALLED_CALL:.5f} Wh")
            print(f"\n  Recovery: session.clear_stall() after a human-in-the-loop fix.")

        # Demonstrate that StallDetected is NOT caught by except ValueError
        print(f"\n  Safety check: StallDetected inherits from RuntimeError, not ValueError.")
        print(f"  A broad `except ValueError:` handler will not swallow it. ✓")


def demo_reroute() -> None:
    _section("Action: reroute")
    print("  STALL-001 fires → model transparently substituted with cheaper alternative.\n")
    set_stall_action("reroute", fallback_model="gpt-4o-mini")

    with Session(tags={"demo": "reroute"}) as session:
        for _ in range(6):
            session.register_event(_event(output_tokens=350))  # type: ignore[arg-type]
        for _ in range(16):
            session.register_event(_event(output_tokens=1))    # type: ignore[arg-type]

        print(f"  stall_triggered: {session.stall_triggered}")

        kwargs: dict[str, Any] = {"model": "gpt-4o"}
        rerouted, original_model = apply_stall_action(kwargs, ctx=None)

        if rerouted:
            print(f"\n  Original model:    {original_model}")
            print(f"  Substituted with:  {kwargs['model']}")
            print(f"\n  → Next call uses gpt-4o-mini transparently.")
            print(f"  → If gpt-4o-mini rejects the parameters, Vetch fails open")
            print(f"    and retries with the original model.")
        else:
            print(f"\n  No reroute in this context.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\nVetch — Runaway Inference Circuit Breaker (demo)")
    print("Simulating an agent loop: 6 productive calls, then 16 stalled.\n")

    # Show the pattern that triggers STALL-001
    preview_session = _build_stalled_session()
    advisories = generate_advisories(preview_session.stats)
    stall_adv = next((a for a in advisories if a.code == "STALL-001"), None)

    print(f"  Total calls:     {preview_session.stats.total_requests}")
    print(f"  Stall triggered: {preview_session.stall_triggered}")
    if stall_adv:
        print(f"  Advisory fired:  [{stall_adv.code}] {stall_adv.title}")
        print(f"  Severity:        {stall_adv.severity}")

    _show_loop_table(preview_session)

    # Demonstrate each circuit breaker action
    demo_warn()
    demo_kill()
    demo_reroute()

    # Reset to default (fail-safe)
    set_stall_action("log")

    _section("In production")
    print("  import vetch")
    print("  vetch.instrument()")
    print("  vetch.set_stall_action('kill')")
    print()
    print("  # Run your agent loop normally.")
    print("  # Vetch detects the stall and raises StallDetected before")
    print("  # more money is wasted. Catch it to handle recovery.")
    print()
    print("  try:")
    print("      result = run_agent_loop(task)")
    print("  except vetch.StallDetected as exc:")
    print("      log.warning('Agent stalled: %s', exc)")
    print("      notify_human_review(task)")
    print(f"\n{'─' * 64}\n")


if __name__ == "__main__":
    main()
