"""Entrypoint for ``python -m vetch.mcp``."""

from __future__ import annotations

import asyncio

from vetch.mcp.server import main


def _run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    _run()
