"""Global test configuration and fixtures."""

from __future__ import annotations

import pytest

from vetch.calculation import _reset_registries
from vetch.config import _reset_config
from vetch.sensing.cache import reset_file_cache
from vetch.stats import _reset_session_stats


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset all global state between tests."""
    _reset_config()
    _reset_registries()
    reset_file_cache()
    _reset_session_stats()

    # Also reset CI stats if possible, but let's start with config
    from vetch.ci import _CI_STATS
    _CI_STATS["count"] = 0
    _CI_STATS["energy_wh"] = 0.0
    _CI_STATS["carbon_g"] = 0.0
    _CI_STATS["cost_usd"] = 0.0

    yield

    _reset_config()
    _reset_registries()
    reset_file_cache()
    _reset_session_stats()
