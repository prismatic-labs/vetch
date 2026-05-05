"""Inference Waste Audit — mockable example.

Simulates three common waste patterns across tagged sessions and
produces an advisory-style audit report. No real API credentials needed.

Run:
    python examples/inference_waste_audit.py

What this demonstrates:
  - RAG-001:   rag-search session with high input:output ratio
  - CACHE-001: document-qa session with repeated identical inputs
  - STALL-001: agent-research session that stalls mid-loop

In production, these advisories fire automatically from real LLM calls.
See QUICKSTART.md to get started with live instrumentation.
"""

from __future__ import annotations

from typing import Any

from vetch.advisory import generate_advisories
from vetch.session import Session
from vetch.stats import SessionStats


# ── Event helpers ────────────────────────────────────────────────────────────


def _event(
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    energy_wh: float,
    model: str = "gpt-4o",
) -> dict[str, Any]:
    return {
        "model": model,
        "usage": {"text": {"input_tokens": input_tokens, "output_tokens": output_tokens}},
        "estimated_cost_usd": cost_usd,
        "estimated_energy_wh": energy_wh,
        "estimated_carbon_g": energy_wh * 0.38,  # us-east-1 ~380 gCO2/kWh
    }


# ── Simulated sessions ───────────────────────────────────────────────────────


def simulate_rag_bloat() -> tuple[SessionStats, dict[str, Any]]:
    """Simulate rag-search: massive context, tiny output (RAG-001)."""
    stats = SessionStats()
    calls = 30
    for _ in range(calls):
        stats.update(_event(input_tokens=8_000, output_tokens=80, cost_usd=0.022, energy_wh=0.0027))
    return stats, {
        "feature": "rag-search",
        "calls": calls,
        "total_cost_usd": stats.total_cost_usd,
        "total_energy_wh": stats.total_energy_wh,
        "total_carbon_g": stats.total_carbon_g,
    }


def simulate_cache_opportunity() -> tuple[SessionStats, dict[str, Any]]:
    """Simulate document-qa: identical input tokens on every call (CACHE-001)."""
    stats = SessionStats()
    calls = 50
    for _ in range(calls):
        # Fixed 2,000-token system prompt + small user message = identical input count
        stats.update(_event(
            input_tokens=2_050, output_tokens=320,
            cost_usd=0.0105, energy_wh=0.00140,
            model="claude-3.7-sonnet",
        ))
    return stats, {
        "feature": "document-qa",
        "calls": calls,
        "total_cost_usd": stats.total_cost_usd,
        "total_energy_wh": stats.total_energy_wh,
        "total_carbon_g": stats.total_carbon_g,
    }


def simulate_stalled_agent() -> tuple[SessionStats, dict[str, Any]]:
    """Simulate agent-research: 6 productive calls then 16 stalled (STALL-001)."""
    session = Session(tags={"feature": "agent-research", "customer": "acme"})

    # Productive phase
    for _ in range(6):
        session.register_event(_event(  # type: ignore[arg-type]
            input_tokens=1_200, output_tokens=350, cost_usd=0.0085, energy_wh=0.00046,
        ))

    # Stalled phase — short outputs, same input token count
    for _ in range(16):
        session.register_event(_event(  # type: ignore[arg-type]
            input_tokens=1_200, output_tokens=1, cost_usd=0.0031, energy_wh=0.00015,
        ))

    stalled_cost = sum(
        c.cost_usd for c in session.stats.recent_calls if c.output_tokens < 5
    )
    stalled_energy = sum(
        c.cost_usd * (0.00015 / 0.0031)  # proportional energy estimate
        for c in session.stats.recent_calls if c.output_tokens < 5
    )

    return session.stats, {
        "feature": "agent-research",
        "customer": "acme",
        "calls": session.stats.total_requests,
        "total_cost_usd": session.stats.total_cost_usd,
        "total_energy_wh": session.stats.total_energy_wh,
        "total_carbon_g": session.stats.total_carbon_g,
        "stall_triggered": session.stall_triggered,
        "stalled_cost_usd": stalled_cost,
        "stalled_calls": sum(1 for c in session.stats.recent_calls if c.output_tokens < 5),
    }


# ── Formatting helpers ───────────────────────────────────────────────────────


def _section(title: str) -> None:
    width = 64
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\nVetch — Inference Waste Audit (demo)")
    print("Simulating 7 days of tagged sessions across three features...\n")

    rag_stats, rag_meta = simulate_rag_bloat()
    cache_stats, cache_meta = simulate_cache_opportunity()
    agent_stats, agent_meta = simulate_stalled_agent()

    sessions = [
        ("rag-search", rag_stats, rag_meta),
        ("document-qa", cache_stats, cache_meta),
        ("agent-research (acme)", agent_stats, agent_meta),
    ]

    # ── Summary table ────────────────────────────────────────────────────────
    _section("Session summary")
    print(f"  {'Feature':<26} {'Calls':>6}  {'Cost ($)':>10}  {'Energy (Wh)':>12}  {'Carbon (g)':>11}")
    print(f"  {'-'*26} {'-'*6}  {'-'*10}  {'-'*12}  {'-'*11}")

    total_cost = total_energy = total_carbon = 0.0
    for feature, _, meta in sessions:
        c = meta["total_cost_usd"]
        e = meta["total_energy_wh"]
        g = meta["total_carbon_g"]
        total_cost += c
        total_energy += e
        total_carbon += g
        print(f"  {feature:<26} {meta['calls']:>6}  {c:>10.4f}  {e:>12.5f}  {g:>11.5f}")
    print(f"  {'─'*26} {'─'*6}  {'─'*10}  {'─'*12}  {'─'*11}")
    print(f"  {'TOTAL':<26} {'':>6}  {total_cost:>10.4f}  {total_energy:>12.5f}  {total_carbon:>11.5f}")

    # ── Advisories ───────────────────────────────────────────────────────────
    _section("Waste advisories")
    all_codes: set[str] = set()

    for feature, stats, meta in sessions:
        advisories = generate_advisories(stats)
        if not advisories:
            print(f"\n  {feature}: no advisories")
            continue
        print(f"\n  Feature: {feature}")
        for adv in advisories:
            all_codes.add(adv.code)
            icon = "🔴" if adv.severity == "CRITICAL" else "🟡" if adv.severity == "WARNING" else "🔵"
            print(f"    {icon} [{adv.code}] {adv.title}")
            # Wrap description at 72 chars
            desc = adv.description
            while len(desc) > 68:
                split = desc[:68].rfind(" ")
                print(f"       {desc[:split]}")
                desc = desc[split + 1:]
            print(f"       {desc}")
            if adv.potential_savings_usd and "wasted cost" not in adv.description.lower():
                print(f"       Estimated wasted cost: ${adv.potential_savings_usd:.4f}")

    # ── Stall detail ─────────────────────────────────────────────────────────
    if agent_meta.get("stall_triggered"):
        _section("Stall detail — agent-research / acme")
        print(f"  Stall triggered:           YES")
        print(f"  Stalled calls (in window): {agent_meta['stalled_calls']}")
        print(f"  Estimated wasted cost:     ${agent_meta['stalled_cost_usd']:.4f}")
        print()
        print(f"  To stop future stalls automatically:")
        print(f"    vetch.set_stall_action('kill')")
        print(f"    # or: vetch.set_stall_action('reroute', fallback_model='gpt-4o-mini')")

    # ── Recommended policies ─────────────────────────────────────────────────
    _section("Recommended policies")
    if "STALL-001" in all_codes:
        print("  STALL-001  → set_stall_action('kill') to stop stalled agent loops")
    if "CACHE-001" in all_codes:
        print("  CACHE-001  → enable prompt caching on the static system prompt")
        print("               (up to 90% cost reduction on cached input tokens)")
    if "RAG-001" in all_codes:
        print("  RAG-001    → tighten retriever relevance threshold")
        print("               (current avg ratio > 50:1 suggests over-retrieval)")

    print(f"\n{'─' * 64}")
    print("  In production: run 'vetch audit' after live sessions for")
    print("  per-session advisories from real LLM calls.")
    print(f"{'─' * 64}\n")


if __name__ == "__main__":
    main()
