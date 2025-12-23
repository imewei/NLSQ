"""
Runtime patches for executing notebooks in automated runs.
"""

from __future__ import annotations

import asyncio


def _ensure_event_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


_ensure_event_loop()
