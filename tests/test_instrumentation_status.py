"""Task 4: instrumentation_status() shape + installed-but-not-imported warning."""

from __future__ import annotations

import subprocess
import sys

import pytest

import vetch

_PROVIDERS = {"openai", "anthropic", "azure_openai", "vertexai", "google_genai", "ollama"}
_FIELDS = {"installed", "imported", "instrumented", "version", "tested"}


def test_instrumentation_status_shape():
    status = vetch.instrumentation_status()
    assert set(status) == _PROVIDERS
    for st in status.values():
        assert set(st) == _FIELDS
        assert isinstance(st["installed"], bool)
        assert isinstance(st["imported"], bool)
        assert isinstance(st["instrumented"], bool)
        assert isinstance(st["tested"], bool)
        assert st["version"] is None or isinstance(st["version"], str)


def test_instrumentation_status_docstring_clarifies_module_patch():
  """instrumented reflects module-level patch, not per-instance wrapping."""
  assert "module" in vetch.instrumentation_status.__doc__.lower()
  assert "is_client_instrumented" in vetch.instrumentation_status.__doc__


def test_status_reports_versions():
    import importlib.util

    if importlib.util.find_spec("anthropic") is None:
        pytest.skip("anthropic not installed")
    if importlib.util.find_spec("google.genai") is None:
        pytest.skip("google.genai not installed")
    import anthropic  # noqa: F401
    import google.genai  # noqa: F401

    vetch.instrument()
    try:
        st = vetch.instrumentation_status()
        assert st["anthropic"]["version"]
        assert st["google_genai"]["version"]
    finally:
        vetch.uninstrument()


def test_is_client_instrumented_openai():
    pytest.importorskip("openai")
    import openai

    import vetch
    from vetch.proxy import is_vetch_patched

    vetch.instrument()
    try:
        client = openai.OpenAI(api_key="sk-test")
        assert vetch.is_client_instrumented(client)
        assert is_vetch_patched(client.chat.completions.create)
    finally:
        vetch.uninstrument()
    # openai is installed in the venv. In a fresh process `import vetch` does not
    # import it, so instrument() must warn that it is not instrumented.
    # Skip if openai is not installed (CI minimal test extras).
    import importlib.util

    if importlib.util.find_spec("openai") is None:
        pytest.skip("openai not installed")
    code = (
        "import logging, io, sys\n"
        "buf = io.StringIO()\n"
        "lg = logging.getLogger('vetch')\n"
        "lg.addHandler(logging.StreamHandler(buf)); lg.setLevel(logging.WARNING)\n"
        "import vetch\n"
        "assert 'openai' not in sys.modules, 'openai imported too early'\n"
        "vetch.instrument()\n"
        "out = buf.getvalue()\n"
        "assert 'openai is installed but was not imported' in out, repr(out)\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_anthropic_genai_have_tested_ranges():
    """anthropic/genai must resolve a tested range so instrument() does not
    spuriously warn 'outside tested version range' for these supported providers."""
    from vetch.compat import (
        TESTED_ANTHROPIC_VERSIONS,
        TESTED_GENAI_VERSIONS,
        get_genai_version,
        version_in_range,
    )

    assert version_in_range("0.40.0", *TESTED_ANTHROPIC_VERSIONS) is True
    assert version_in_range("2.10.0", *TESTED_GENAI_VERSIONS) is True
    info = get_genai_version()
    if info.installed:  # present in the test env via langchain-google-genai
        assert info.version is not None
        assert info.tested is True
