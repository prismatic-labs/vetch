"""Budget alerts for LLM inference cost/energy tracking.

This module provides warn-only budget monitoring. By design, Vetch
NEVER blocks inference calls. Budget alerts are purely observational.

Philosophy:
    "Measure, don't gate. Warn, don't block."

When budgets are exceeded, Vetch:
1. Logs a warning to stderr
2. Sets budget_exceeded=True in the event
3. Calls optional webhook/callback
4. ALWAYS allows the LLM call to proceed
"""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Literal

# Window durations in seconds
_WINDOW_SECONDS = {
    "request": 0,  # Reset every request
    "session": 0,  # Never reset (process lifetime)
    "hour": 3600,
    "day": 86400,
}

logger = logging.getLogger(__name__)

# Alert callback type
AlertCallback = Callable[["BudgetAlert"], None]

# Thread-safe global state
_lock = threading.Lock()
_budgets: dict[str, Budget] = {}
_alert_callbacks: list[AlertCallback] = []

# Bounded LRU set for warning deduplication (prevents memory leak)
_MAX_WARNED_KEYS = 1000
_warned_once: OrderedDict[str, None] = OrderedDict()


@dataclass
class Budget:
    """Budget threshold configuration.

    Budgets are WARN-ONLY by design. They never block calls.

    Attributes:
        name: Unique identifier for this budget (e.g., "daily", "per-request").
        cost_usd: Maximum cost in USD before warning.
        energy_wh: Maximum energy in Wh before warning.
        carbon_g: Maximum carbon in gCO2e before warning.
        window: Time window for accumulation ("request", "session", "hour", "day").
        warn_at_pct: Percentage threshold for early warning (default 80%).
        tags_filter: Only apply to events matching these tags.
    """

    name: str
    cost_usd: float | None = None
    energy_wh: float | None = None
    carbon_g: float | None = None
    window: Literal["request", "session", "hour", "day"] = "request"
    warn_at_pct: float = 80.0
    tags_filter: dict[str, str] | None = None

    # Internal tracking
    _accumulated_cost: float = field(default=0.0, repr=False)
    _accumulated_energy: float = field(default=0.0, repr=False)
    _accumulated_carbon: float = field(default=0.0, repr=False)
    _alert_count: int = field(default=0, repr=False)
    _last_reset_at: float = field(default_factory=time.time, repr=False)

    def reset(self) -> None:
        """Reset accumulated values and update timestamp."""
        self._accumulated_cost = 0.0
        self._accumulated_energy = 0.0
        self._accumulated_carbon = 0.0
        self._last_reset_at = time.time()


@dataclass
class BudgetAlert:
    """Alert generated when a budget threshold is approached or exceeded."""

    budget_name: str
    metric: Literal["cost_usd", "energy_wh", "carbon_g"]
    threshold: float
    current_value: float
    percentage: float
    exceeded: bool
    window: str
    tags: dict[str, str] | None = None

    def __str__(self) -> str:
        status = "EXCEEDED" if self.exceeded else "WARNING"
        return (
            f"[Vetch Budget {status}] {self.budget_name}: "
            f"{self.metric}={self.current_value:.4f} "
            f"({self.percentage:.1f}% of {self.threshold})"
        )


def set_budget(
    name: str,
    cost_usd: float | None = None,
    energy_wh: float | None = None,
    carbon_g: float | None = None,
    window: Literal["request", "session", "hour", "day"] = "request",
    warn_at_pct: float = 80.0,
    tags_filter: dict[str, str] | None = None,
) -> Budget:
    """Configure a budget threshold.

    Budgets are WARN-ONLY. They never block inference calls.

    Args:
        name: Unique name for this budget.
        cost_usd: Cost threshold in USD.
        energy_wh: Energy threshold in Wh.
        carbon_g: Carbon threshold in gCO2e.
        window: Accumulation window:
            - "request": Reset after each call (per-call budget)
            - "session": Accumulate for process lifetime
            - "hour": Reset every hour (based on wall-clock time)
            - "day": Reset every 24 hours (based on wall-clock time)
        warn_at_pct: Percentage at which to start warning (default 80%).
        tags_filter: Only apply to events with matching tags.

    Returns:
        The configured Budget object.

    Example::

        # Warn when any single request costs > $0.10
        set_budget("per-request", cost_usd=0.10, window="request")

        # Warn when session energy exceeds 1 Wh
        set_budget("session-energy", energy_wh=1.0, window="session")

        # Warn when hourly cost exceeds $5 (resets automatically)
        set_budget("hourly-cost", cost_usd=5.0, window="hour")

        # Warn for production environment only
        set_budget("prod-cost", cost_usd=1.0, tags_filter={"env": "production"})
    """
    budget = Budget(
        name=name,
        cost_usd=cost_usd,
        energy_wh=energy_wh,
        carbon_g=carbon_g,
        window=window,
        warn_at_pct=warn_at_pct,
        tags_filter=tags_filter,
    )
    with _lock:
        _budgets[name] = budget
    logger.debug(f"Budget configured: {name}")
    return budget


def remove_budget(name: str) -> bool:
    """Remove a budget by name.

    Args:
        name: Budget name to remove.

    Returns:
        True if removed, False if not found.
    """
    with _lock:
        if name in _budgets:
            del _budgets[name]
            return True
        return False


def clear_budgets() -> None:
    """Remove all configured budgets."""
    with _lock:
        _budgets.clear()
        _warned_once.clear()


def on_budget_alert(callback: AlertCallback) -> AlertCallback:
    """Register a callback for budget alerts.

    Can be used as a decorator or called directly.

    Args:
        callback: Function to call with BudgetAlert when threshold approached/exceeded.

    Example::

        @on_budget_alert
        def notify_slack(alert: BudgetAlert):
            slack.post(f"Budget alert: {alert}")

        # Or:
        on_budget_alert(lambda a: print(a))
    """
    _alert_callbacks.append(callback)
    return callback


def _add_warned_key(key: str) -> bool:
    """Add key to bounded warning set. Returns True if key was new."""
    # Must be called with _lock held
    if key in _warned_once:
        # Move to end (most recent)
        _warned_once.move_to_end(key)
        return False
    # Add new key
    _warned_once[key] = None
    # Evict oldest if over limit
    while len(_warned_once) > _MAX_WARNED_KEYS:
        _warned_once.popitem(last=False)
    return True


def check_budgets(
    cost_usd: float | None,
    energy_wh: float | None,
    carbon_g: float | None,
    tags: dict[str, str] | None = None,
) -> tuple[bool, list[BudgetAlert]]:
    """Check all budgets against current values.

    This is called internally by VetchContext after each inference.
    NEVER blocks - only returns alerts. Thread-safe.

    Args:
        cost_usd: Cost of this request.
        energy_wh: Energy of this request.
        carbon_g: Carbon of this request.
        tags: Event tags for filtering.

    Returns:
        Tuple of (any_exceeded, list of alerts).
    """
    alerts: list[BudgetAlert] = []
    any_exceeded = False
    callbacks_to_fire: list[tuple[AlertCallback, BudgetAlert]] = []

    with _lock:
        now = time.time()
        for budget in _budgets.values():
            # Check tag filter
            if budget.tags_filter:
                if not tags:
                    continue
                if not all(tags.get(k) == v for k, v in budget.tags_filter.items()):
                    continue

            # Handle window-based resets
            if budget.window == "request":
                # Reset for each request
                budget.reset()
            elif budget.window in ("hour", "day"):
                # Time-based reset: check if window has elapsed
                window_seconds = _WINDOW_SECONDS[budget.window]
                if now - budget._last_reset_at >= window_seconds:
                    budget.reset()
            # "session" window never resets automatically

            # Accumulate values (atomic under lock)
            if cost_usd is not None:
                budget._accumulated_cost += cost_usd
            if energy_wh is not None:
                budget._accumulated_energy += energy_wh
            if carbon_g is not None:
                budget._accumulated_carbon += carbon_g

            # Check each metric
            checks = [
                ("cost_usd", budget.cost_usd, budget._accumulated_cost),
                ("energy_wh", budget.energy_wh, budget._accumulated_energy),
                ("carbon_g", budget.carbon_g, budget._accumulated_carbon),
            ]

            for metric, threshold, current in checks:
                if threshold is None or threshold <= 0:
                    continue

                pct = (current / threshold) * 100
                exceeded = pct >= 100
                should_warn = pct >= budget.warn_at_pct

                if exceeded:
                    any_exceeded = True

                if should_warn:
                    alert = BudgetAlert(
                        budget_name=budget.name,
                        metric=metric,  # type: ignore[arg-type]
                        threshold=threshold,
                        current_value=current,
                        percentage=pct,
                        exceeded=exceeded,
                        window=budget.window,
                        tags=tags,
                    )
                    alerts.append(alert)
                    budget._alert_count += 1

                    # Log warning (dedupe per budget+metric combo, bounded LRU)
                    warn_key = f"{budget.name}:{metric}"
                    is_new = _add_warned_key(warn_key)
                    if is_new or exceeded:
                        logger.warning(str(alert))

                    # Queue callbacks to fire outside lock
                    for callback in _alert_callbacks:
                        callbacks_to_fire.append((callback, alert))

    # Fire callbacks outside lock to avoid deadlocks
    for callback, alert in callbacks_to_fire:
        try:
            callback(alert)
        except Exception as e:
            logger.debug(f"Budget alert callback failed: {e}")

    return any_exceeded, alerts


def get_budget_status() -> dict[str, dict[str, float | int]]:
    """Get current status of all budgets. Thread-safe.

    Returns:
        Dict mapping budget names to their accumulated values and alert counts.

    Example::

        status = get_budget_status()
        # {
        #     "session": {
        #         "accumulated_cost": 0.45,
        #         "accumulated_energy": 0.12,
        #         "accumulated_carbon": 5.2,
        #         "alert_count": 2
        #     }
        # }
    """
    with _lock:
        return {
            name: {
                "accumulated_cost": b._accumulated_cost,
                "accumulated_energy": b._accumulated_energy,
                "accumulated_carbon": b._accumulated_carbon,
                "alert_count": b._alert_count,
            }
            for name, b in _budgets.items()
        }


# Environment-based auto-configuration
def _auto_configure_from_env() -> None:
    """Auto-configure budgets from environment variables.

    Supported variables:
        VETCH_BUDGET_COST_USD: Per-request cost threshold
        VETCH_BUDGET_ENERGY_WH: Per-request energy threshold
        VETCH_BUDGET_CARBON_G: Per-request carbon threshold
        VETCH_BUDGET_SESSION_COST_USD: Session-wide cost threshold
    """
    cost = os.environ.get("VETCH_BUDGET_COST_USD")
    if cost:
        try:
            set_budget("env-cost", cost_usd=float(cost), window="request")
        except ValueError:
            warnings.warn(f"Invalid VETCH_BUDGET_COST_USD: {cost}", stacklevel=2)

    energy = os.environ.get("VETCH_BUDGET_ENERGY_WH")
    if energy:
        try:
            set_budget("env-energy", energy_wh=float(energy), window="request")
        except ValueError:
            warnings.warn(f"Invalid VETCH_BUDGET_ENERGY_WH: {energy}", stacklevel=2)

    carbon = os.environ.get("VETCH_BUDGET_CARBON_G")
    if carbon:
        try:
            set_budget("env-carbon", carbon_g=float(carbon), window="request")
        except ValueError:
            warnings.warn(f"Invalid VETCH_BUDGET_CARBON_G: {carbon}", stacklevel=2)

    session_cost = os.environ.get("VETCH_BUDGET_SESSION_COST_USD")
    if session_cost:
        try:
            set_budget("env-session-cost", cost_usd=float(session_cost), window="session")
        except ValueError:
            warnings.warn(f"Invalid VETCH_BUDGET_SESSION_COST_USD: {session_cost}", stacklevel=2)


# Auto-configure on import
_auto_configure_from_env()
