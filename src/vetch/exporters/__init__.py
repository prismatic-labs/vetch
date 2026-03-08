"""Vetch Exporters - Export inference events to observability platforms.

This package provides exporters for vetch inference events to various
observability and monitoring platforms:

- opentelemetry: Export to OpenTelemetry (Jaeger, Datadog, New Relic, etc.)

Example:
    >>> from vetch.exporters.opentelemetry import configure_auto_export
    >>> configure_auto_export(enabled=True)
"""

from __future__ import annotations

__all__ = ["opentelemetry"]
