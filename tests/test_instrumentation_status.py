"""Task 4: instrumentation_status() shape + installed-but-not-imported warning."""

from __future__ import annotations

import subprocess
import sys

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


def test_installed_but_not_imported_warns_in_fresh_process():
    # openai is installed in the venv. In a fresh process `import vetch` does not
    # import it, so instrument() must warn that it is not instrumented.
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
