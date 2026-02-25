"""Tests for budget alert functionality.

Budget alerts are WARN-ONLY by design. They never block inference.
"""

from __future__ import annotations

import pytest

from vetch.budget import (
    _WINDOW_SECONDS,
    Budget,
    BudgetAlert,
    check_budgets,
    clear_budgets,
    get_budget_status,
    on_budget_alert,
    remove_budget,
    set_budget,
)


class TestSetBudget:
    """Tests for budget configuration."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_set_budget_returns_budget_object(self) -> None:
        """set_budget returns a Budget object."""
        budget = set_budget("test", cost_usd=1.0)

        assert isinstance(budget, Budget)
        assert budget.name == "test"
        assert budget.cost_usd == 1.0

    def test_set_budget_with_all_metrics(self) -> None:
        """Can set all metric thresholds."""
        budget = set_budget(
            "full",
            cost_usd=10.0,
            energy_wh=1.0,
            carbon_g=100.0,
        )

        assert budget.cost_usd == 10.0
        assert budget.energy_wh == 1.0
        assert budget.carbon_g == 100.0

    def test_set_budget_with_tags_filter(self) -> None:
        """Can filter budget by tags."""
        budget = set_budget(
            "prod-only",
            cost_usd=5.0,
            tags_filter={"env": "production"},
        )

        assert budget.tags_filter == {"env": "production"}

    def test_remove_budget_returns_true(self) -> None:
        """remove_budget returns True when budget exists."""
        set_budget("removable", cost_usd=1.0)

        result = remove_budget("removable")

        assert result is True

    def test_remove_budget_returns_false_when_missing(self) -> None:
        """remove_budget returns False when budget doesn't exist."""
        result = remove_budget("nonexistent")

        assert result is False


class TestCheckBudgets:
    """Tests for budget checking."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_no_budgets_returns_empty(self) -> None:
        """Returns no alerts when no budgets configured."""
        exceeded, alerts = check_budgets(
            cost_usd=100.0,
            energy_wh=10.0,
            carbon_g=1000.0,
        )

        assert exceeded is False
        assert alerts == []

    def test_under_threshold_no_alert(self) -> None:
        """No alert when under threshold."""
        set_budget("test", cost_usd=1.0)

        exceeded, alerts = check_budgets(
            cost_usd=0.5,
            energy_wh=None,
            carbon_g=None,
        )

        assert exceeded is False
        assert alerts == []

    def test_over_warn_threshold_alerts(self) -> None:
        """Alert when over warn_at_pct threshold."""
        set_budget("test", cost_usd=1.0, warn_at_pct=80.0)

        exceeded, alerts = check_budgets(
            cost_usd=0.85,  # 85% of threshold
            energy_wh=None,
            carbon_g=None,
        )

        assert exceeded is False  # Not exceeded yet
        assert len(alerts) == 1
        assert alerts[0].metric == "cost_usd"
        assert alerts[0].percentage == 85.0

    def test_exceeded_threshold_alerts(self) -> None:
        """Alert and exceeded=True when over 100%."""
        set_budget("test", cost_usd=1.0)

        exceeded, alerts = check_budgets(
            cost_usd=1.5,  # 150% of threshold
            energy_wh=None,
            carbon_g=None,
        )

        assert exceeded is True
        assert len(alerts) == 1
        assert alerts[0].exceeded is True

    def test_tags_filter_matches(self) -> None:
        """Budget applies when tags match filter."""
        set_budget("prod", cost_usd=1.0, tags_filter={"env": "production"})

        exceeded, alerts = check_budgets(
            cost_usd=1.5,
            energy_wh=None,
            carbon_g=None,
            tags={"env": "production"},
        )

        assert exceeded is True
        assert len(alerts) == 1

    def test_tags_filter_no_match(self) -> None:
        """Budget doesn't apply when tags don't match."""
        set_budget("prod", cost_usd=1.0, tags_filter={"env": "production"})

        exceeded, alerts = check_budgets(
            cost_usd=1.5,
            energy_wh=None,
            carbon_g=None,
            tags={"env": "development"},
        )

        assert exceeded is False
        assert alerts == []


class TestBudgetCallback:
    """Tests for budget alert callbacks."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_callback_fires_on_alert(self) -> None:
        """Callback is called when alert is triggered."""
        alerts_received: list[BudgetAlert] = []

        @on_budget_alert
        def capture(alert: BudgetAlert) -> None:
            alerts_received.append(alert)

        set_budget("test", cost_usd=1.0, warn_at_pct=50.0)
        check_budgets(cost_usd=0.6, energy_wh=None, carbon_g=None)

        assert len(alerts_received) == 1
        assert alerts_received[0].budget_name == "test"


class TestGetBudgetStatus:
    """Tests for budget status retrieval."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_returns_accumulated_values(self) -> None:
        """Status includes accumulated values."""
        set_budget("session", cost_usd=100.0, window="session")

        # Simulate two requests
        check_budgets(cost_usd=10.0, energy_wh=1.0, carbon_g=50.0)
        check_budgets(cost_usd=20.0, energy_wh=2.0, carbon_g=100.0)

        status = get_budget_status()

        assert "session" in status
        assert status["session"]["accumulated_cost"] == 30.0
        assert status["session"]["accumulated_energy"] == 3.0
        assert status["session"]["accumulated_carbon"] == 150.0


class TestBudgetAlert:
    """Tests for BudgetAlert class."""

    def test_str_format_warning(self) -> None:
        """String format for warning (not exceeded)."""
        alert = BudgetAlert(
            budget_name="test",
            metric="cost_usd",
            threshold=1.0,
            current_value=0.85,
            percentage=85.0,
            exceeded=False,
            window="request",
        )

        result = str(alert)

        assert "WARNING" in result
        assert "test" in result
        assert "85.0%" in result

    def test_str_format_exceeded(self) -> None:
        """String format for exceeded budget."""
        alert = BudgetAlert(
            budget_name="test",
            metric="cost_usd",
            threshold=1.0,
            current_value=1.5,
            percentage=150.0,
            exceeded=True,
            window="request",
        )

        result = str(alert)

        assert "EXCEEDED" in result


class TestTimeBasedBudgets:
    """Tests for time-based budget windows (hour, day)."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_hour_window_configured(self) -> None:
        """Can configure hourly budget."""
        budget = set_budget("hourly", cost_usd=10.0, window="hour")

        assert budget.window == "hour"
        assert budget._last_reset_at > 0

    def test_day_window_configured(self) -> None:
        """Can configure daily budget."""
        budget = set_budget("daily", cost_usd=100.0, window="day")

        assert budget.window == "day"
        assert budget._last_reset_at > 0

    def test_hour_budget_accumulates_within_window(self) -> None:
        """Hourly budget accumulates within the hour."""
        set_budget("hourly", cost_usd=10.0, window="hour")

        # First request
        check_budgets(cost_usd=3.0, energy_wh=None, carbon_g=None)
        status = get_budget_status()
        assert status["hourly"]["accumulated_cost"] == 3.0

        # Second request (should accumulate, not reset)
        check_budgets(cost_usd=2.0, energy_wh=None, carbon_g=None)
        status = get_budget_status()
        assert status["hourly"]["accumulated_cost"] == 5.0

    def test_window_seconds_defined(self) -> None:
        """Window durations are properly defined."""
        assert _WINDOW_SECONDS["request"] == 0
        assert _WINDOW_SECONDS["session"] == 0
        assert _WINDOW_SECONDS["hour"] == 3600
        assert _WINDOW_SECONDS["day"] == 86400


class TestGetCleanestRegion:
    """Tests for get_cleanest_region API."""

    def test_returns_cleanest_region(self) -> None:
        """Returns region with lowest carbon intensity."""
        from vetch.sensing.grid import get_cleanest_region

        # Using known fallback values
        region, intensity = get_cleanest_region(["us-east-1", "eu-north-1"])

        assert region in ["us-east-1", "eu-north-1"]
        assert intensity > 0

    def test_raises_on_empty_candidates(self) -> None:
        """Raises ValueError if candidates list is empty."""
        from vetch.sensing.grid import get_cleanest_region

        with pytest.raises(ValueError, match="cannot be empty"):
            get_cleanest_region([])

    def test_single_candidate_returns_it(self) -> None:
        """Single candidate is returned."""
        from vetch.sensing.grid import get_cleanest_region

        region, intensity = get_cleanest_region(["us-west-2"])

        assert region == "us-west-2"
        assert intensity > 0


class TestAlertCooldown:
    """Tests for alert cooldown throttling."""

    def setup_method(self) -> None:
        """Clear budgets before each test."""
        clear_budgets()

    def teardown_method(self) -> None:
        """Clear budgets after each test."""
        clear_budgets()

    def test_cooldown_suppresses_repeated_alerts(self) -> None:
        """Repeated alerts within cooldown window are suppressed."""
        set_budget(
            "test",
            cost_usd=1.0,
            warn_at_pct=50.0,
            alert_cooldown_seconds=60.0,
        )

        # First check triggers alert
        _, alerts1 = check_budgets(cost_usd=0.6, energy_wh=None, carbon_g=None)
        assert len(alerts1) == 1

        # Second check within cooldown is suppressed
        _, alerts2 = check_budgets(cost_usd=0.1, energy_wh=None, carbon_g=None)
        assert len(alerts2) == 0

    def test_zero_cooldown_allows_all_alerts(self) -> None:
        """Zero cooldown allows every alert to fire."""
        set_budget(
            "test",
            cost_usd=1.0,
            warn_at_pct=50.0,
            alert_cooldown_seconds=0.0,
            window="session",  # Accumulates across calls
        )

        _, alerts1 = check_budgets(cost_usd=0.6, energy_wh=None, carbon_g=None)
        assert len(alerts1) == 1

        # Session window: 0.6 + 0.1 = 0.7 = 70% > 50% warn_at_pct
        _, alerts2 = check_budgets(cost_usd=0.1, energy_wh=None, carbon_g=None)
        assert len(alerts2) == 1

    def test_cooldown_per_metric(self) -> None:
        """Cooldown is tracked per budget+metric pair."""
        set_budget(
            "test",
            cost_usd=1.0,
            energy_wh=1.0,
            warn_at_pct=50.0,
            alert_cooldown_seconds=60.0,
        )

        # First check: both metrics fire
        _, alerts1 = check_budgets(cost_usd=0.6, energy_wh=0.6, carbon_g=None)
        assert len(alerts1) == 2

        # Second check: both suppressed
        _, alerts2 = check_budgets(cost_usd=0.1, energy_wh=0.1, carbon_g=None)
        assert len(alerts2) == 0

    def test_default_cooldown_is_60_seconds(self) -> None:
        """Default alert_cooldown_seconds is 60."""
        budget = set_budget("test", cost_usd=1.0)
        assert budget.alert_cooldown_seconds == 60.0
