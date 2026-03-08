"""Health check utilities for production monitoring.

Provides status checks for Vetch components:
- Registry circuit breaker state
- OTLP export queue health
- Tracking error counters
- Memory cache status
"""

from __future__ import annotations

from typing import Any, Callable


def get_health_status() -> dict[str, Any]:
    """Get health status of Vetch components for monitoring.

    Returns comprehensive health metrics suitable for:
    - Kubernetes liveness/readiness probes
    - Prometheus /metrics endpoint
    - Application health dashboards
    - Alerting systems

    Returns:
        Dictionary with health status:
        - status: "healthy" | "degraded" | "unhealthy"
        - components: Individual component statuses
        - timestamp: ISO8601 UTC timestamp

    Example::

        health = vetch.health.get_health_status()
        if health["status"] != "healthy":
            logger.warning(f"Vetch degraded: {health['components']}")

    Example Kubernetes probe::

        livenessProbe:
          exec:
            command:
            - python
            - -c
            - |
              import vetch.health
              h = vetch.health.get_health_status()
              exit(0 if h['status'] in ('healthy', 'degraded') else 1)
    """
    from datetime import datetime, timezone

    components: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    # Check registry circuit breaker
    try:
        from vetch.registry.remote import get_remote_fetcher

        fetcher = get_remote_fetcher()
        if fetcher and fetcher._circuit_open_until > 0:
            import time

            if time.monotonic() < fetcher._circuit_open_until:
                components["circuit_breaker"] = {
                    "status": "open",
                    "healthy": False,
                    "message": "Circuit breaker open (registry fetch failing)",
                }
                issues.append("circuit_breaker_open")
            else:
                components["circuit_breaker"] = {
                    "status": "closed",
                    "healthy": True,
                }
        else:
            components["circuit_breaker"] = {
                "status": "closed",
                "healthy": True,
            }
    except Exception as e:
        components["circuit_breaker"] = {
            "status": "unknown",
            "healthy": True,  # Assume healthy if can't check
            "message": f"Could not check: {e}",
        }

    # Check OTLP queue
    try:
        from vetch.otel import get_otlp_stats

        otlp_stats = get_otlp_stats()
        queue_utilization = (
            otlp_stats["queue_current"] / otlp_stats["queue_size"]
            if otlp_stats["queue_size"] > 0
            else 0.0
        )

        otlp_healthy = True
        otlp_status = "healthy"
        otlp_message = None

        if otlp_stats["dropped_events"] > 1000:
            otlp_healthy = False
            otlp_status = "degraded"
            otlp_message = f"{otlp_stats['dropped_events']} events dropped"
            issues.append("otlp_high_drop_rate")
        elif queue_utilization > 0.9:
            otlp_status = "degraded"
            otlp_message = f"Queue {int(queue_utilization*100)}% full"
            issues.append("otlp_queue_near_full")

        components["otlp_queue"] = {
            "status": otlp_status,
            "healthy": otlp_healthy,
            "queue_utilization": f"{int(queue_utilization*100)}%",
            "dropped_events": otlp_stats["dropped_events"],
            "message": otlp_message,
        }
    except Exception as e:
        components["otlp_queue"] = {
            "status": "unknown",
            "healthy": True,
            "message": f"Could not check: {e}",
        }

    # Check tracking errors
    try:
        from vetch.wrapper import get_tracking_stats

        tracking_stats = get_tracking_stats()
        tracking_healthy = True
        tracking_status = "healthy"
        tracking_message = None

        # High error rates indicate problems
        if tracking_stats["model_unknown"] > 100:
            tracking_status = "degraded"
            tracking_message = f"{tracking_stats['model_unknown']} unknown models"
            issues.append("high_unknown_model_rate")

        if tracking_stats["missing_required_tags"] > 10:
            tracking_healthy = False
            tracking_status = "unhealthy"
            tracking_message = f"{tracking_stats['missing_required_tags']} compliance violations"
            issues.append("compliance_violations")

        components["tracking"] = {
            "status": tracking_status,
            "healthy": tracking_healthy,
            "error_counts": tracking_stats,
            "message": tracking_message,
        }
    except Exception as e:
        components["tracking"] = {
            "status": "unknown",
            "healthy": True,
            "message": f"Could not check: {e}",
        }

    # Overall health status
    all_healthy = all(c.get("healthy", True) for c in components.values())
    any_degraded = any(c.get("status") == "degraded" for c in components.values())

    if all_healthy and not any_degraded:
        overall_status = "healthy"
    elif all_healthy:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return {
        "status": overall_status,
        "components": components,
        "issues": issues,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def create_health_endpoint() -> tuple[Callable[[], Any], Callable[[], Any]]:
    """Create HTTP health check endpoint handlers.

    Returns functions for both Flask and FastAPI that can be used as
    health check endpoints in production deployments.

    Returns:
        Tuple of (flask_handler, fastapi_handler).

    Example (Flask)::

        from flask import Flask
        from vetch.health import create_health_endpoint

        app = Flask(__name__)
        flask_health, _ = create_health_endpoint()
        app.route("/health")(flask_health)

    Example (FastAPI)::

        from fastapi import FastAPI
        from vetch.health import create_health_endpoint

        app = FastAPI()
        _, fastapi_health = create_health_endpoint()
        app.get("/health")(fastapi_health)
    """

    def flask_handler() -> Any:
        """Flask health check endpoint handler."""
        try:
            from flask import jsonify  # type: ignore[import-not-found]
        except ImportError:
            return {"error": "Flask not installed"}, 500

        health = get_health_status()
        status_code = 200 if health["status"] in ("healthy", "degraded") else 503
        return jsonify(health), status_code

    async def fastapi_handler() -> Any:
        """FastAPI health check endpoint handler."""
        try:
            from fastapi.responses import (  # type: ignore[import-not-found]
                JSONResponse,
            )
        except ImportError:
            # Return dict since JSONResponse is not available
            return {
                "error": "FastAPI not installed",
                "status_code": 500,
            }

        health = get_health_status()
        status_code = 200 if health["status"] in ("healthy", "degraded") else 503
        return JSONResponse(health, status_code=status_code)

    return flask_handler, fastapi_handler
