"""
Entry point for running NLSQ Qt GUI as a module.

Usage:
    python -m nlsq.gui_qt
    NLSQ_DEBUG=1 python -m nlsq.gui_qt  # With debug logging
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    """Main entry point for the NLSQ Qt GUI."""
    # Enable debug logging if requested
    if os.environ.get("NLSQ_DEBUG"):
        import logging

        logging.basicConfig(level=logging.DEBUG)

    from nlsq.gui_qt import run_desktop

    return run_desktop()


if __name__ == "__main__":
    sys.exit(main())
