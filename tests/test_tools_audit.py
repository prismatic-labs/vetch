from __future__ import annotations

from vetch.tools.audit import audit_registry


def test_registry_audit_passes_current_registry() -> None:
    """Registry audit should fail only on structural issues, not heuristic warnings."""
    assert audit_registry() is True
