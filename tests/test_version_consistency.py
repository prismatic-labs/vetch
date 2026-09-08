"""Guard against version-string drift and undocumented releases.

`vetch/__init__.py` reads the real version from installed package metadata and
only falls back to a hard-coded string when running from an uninstalled source
checkout. That fallback has silently gone stale across releases (the release
commit bumps ``pyproject.toml`` but not the fallback), so source checkouts
report the wrong version. These tests fail the build when the two disagree.

Kept dependency-free (regex, not ``tomllib``) so it runs on every Python in the
CI matrix, including 3.9/3.10 which lack ``tomllib``.
"""

from __future__ import annotations

import re
from pathlib import Path

import vetch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYPROJECT = _REPO_ROOT / "pyproject.toml"
_INIT = _REPO_ROOT / "src" / "vetch" / "__init__.py"


def _pyproject_version() -> str:
    """Read the version from the [project] table of pyproject.toml."""
    section = None
    for line in _PYPROJECT.read_text(encoding="utf-8").splitlines():
        header = re.match(r"\s*\[([^\]]+)\]", line)
        if header:
            section = header.group(1)
            continue
        if section == "project":
            m = re.match(r"""\s*version\s*=\s*["']([^"']+)["']""", line)
            if m:
                return m.group(1)
    raise AssertionError("No [project] version found in pyproject.toml")


def _init_fallback_version() -> str:
    """Read the hard-coded __version__ fallback literal from __init__.py."""
    text = _INIT.read_text(encoding="utf-8")
    # Match only the quoted literal (the metadata call has no quoted version).
    m = re.search(r"""__version__\s*=\s*["']([0-9][^"']*)["']""", text)
    if not m:
        raise AssertionError("No quoted __version__ fallback found in __init__.py")
    return m.group(1)


def test_init_fallback_matches_pyproject() -> None:
    pyproject = _pyproject_version()
    fallback = _init_fallback_version()
    assert fallback == pyproject, (
        f"__version__ source fallback ({fallback!r}) is out of sync with "
        f"pyproject.toml ({pyproject!r}). Bump the fallback in "
        f"src/vetch/__init__.py when you cut a release."
    )


def test_runtime_version_matches_pyproject() -> None:
    # Holds whether the version came from installed metadata or the fallback.
    assert vetch.__version__ == _pyproject_version()


def _changelog() -> str:
    return (_REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_version_is_documented() -> None:
    """A bump with no entry ships changes nobody can find.

    During protocol-v2 development the public event gained two fields and
    changed when p95 is populated, while the version stayed put and the
    CHANGELOG stopped at the first commit of the branch. Version agreement
    alone does not catch that.
    """
    version = _pyproject_version()
    changelog = _changelog()
    if f"## [{version}]" in changelog:
        return  # already cut as a release section
    # Anchor on exactly one newline. Allowing \s* here lets an EMPTY Unreleased
    # section swallow the separator and capture the next release's body, which
    # makes this guard silently inert -- the failure mode it exists to prevent.
    unreleased = re.search(r"## \[Unreleased\]\n(.*?)(?=\n## \[)", changelog, re.S)
    assert unreleased and unreleased.group(1).strip(), (
        f"pyproject.toml is at {version} but CHANGELOG.md has neither a section "
        "for it nor any content under [Unreleased]."
    )
