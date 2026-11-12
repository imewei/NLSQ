"""
NLSQ Qt GUI - Native PySide6 Desktop Application

This module provides a native Qt-based desktop application for NLSQ curve fitting,
with GPU-accelerated plotting via pyqtgraph and native desktop integration.

Usage:
    python -m nlsq.gui_qt

Or programmatically:
    from nlsq.gui_qt import run_desktop
    run_desktop()
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import os
import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from PySide6.QtWidgets import QApplication

__all__ = ["GUI_INSTALL_HINT", "run_desktop"]

GUI_INSTALL_HINT = 'The NLSQ desktop app needs the GUI extra: pip install "nlsq[gui]"'

# ---------------------------------------------------------------------------
# macOS SIGBUS Prevention — Defence in Depth
#
# SIGBUS on macOS is caused by Metal/OpenGL/XLA conflicts in the GPU driver.
# Layer 1 (nlsq/__init__.py): env vars — JAX_PLATFORM_NAME, MPLBACKEND, etc.
# Layer 2 (here): explicit matplotlib.use(), faulthandler, Qt RHI guard
# Layer 3 (plots/__init__.py): pyqtgraph useOpenGL off on macOS, offscreen, no-GL
# Layer 4 (main_window.py): deferred QMessageBox via QTimer.singleShot
# ---------------------------------------------------------------------------

# Enable faulthandler early so any SIGBUS/SIGSEGV prints a traceback
# to stderr instead of dying silently.
with contextlib.suppress(AttributeError, ValueError, RuntimeError, OSError):
    faulthandler.enable()  # may lack a usable stderr fileno; see nlsq/__init__.py

if sys.platform == "darwin":
    # Force matplotlib to non-interactive backend before anything imports it.
    # The MPLBACKEND env var (set in nlsq/__init__.py) covers first import,
    # but an explicit use() call is the belt-and-suspenders guarantee.
    try:
        import matplotlib

        matplotlib.use("Agg")
    except Exception:
        pass

    # Force Qt to use software OpenGL rendering instead of hardware
    # Metal/OpenGL.  On macOS 26+ with Apple Silicon, the Metal-backed QRhi
    # can SIGBUS when creating surfaces before the native window handle is
    # ready, or when mixing OpenGL and Metal contexts.
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QSG_RHI_BACKEND", "software")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")


def _exception_hook(exc_type: type, exc_value: BaseException, exc_tb: object) -> None:
    """Global exception hook to handle uncaught exceptions.

    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_tb: Exception traceback
    """
    # Format the traceback
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # Log to stderr
    sys.stderr.write(f"Uncaught exception:\n{tb_str}\n")

    # Try to show error dialog if Qt is available
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance()
        if app is not None:
            # Try to save autosave before showing dialog
            try:
                from nlsq.gui_qt.autosave import AUTOSAVE_FILE

                # Write crash marker
                crash_info = {
                    "crash": True,
                    "error": str(exc_value),
                    "traceback": tb_str,
                }
                import json

                if AUTOSAVE_FILE.exists():
                    data = json.loads(AUTOSAVE_FILE.read_text(encoding="utf-8"))
                    data["crash_info"] = crash_info
                    AUTOSAVE_FILE.write_text(
                        json.dumps(data, indent=2),
                        encoding="utf-8",
                    )
            except Exception:
                pass

            # Show error dialog (skip on macOS if window may not be
            # fully realised — showing a dialog can itself SIGBUS)
            if sys.platform != "darwin" or app.activeWindow() is not None:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Icon.Critical)
                msg.setWindowTitle("NLSQ Error")
                msg.setText("An unexpected error occurred.")
                msg.setInformativeText(
                    "The application encountered an error and needs to close.\n\n"
                    "Your session will be recovered on next launch.",
                )
                msg.setDetailedText(tb_str)
                msg.exec()
    except Exception:
        pass


_ENV_DUMP_PREFIXES = (
    "QT_", "XDG_", "LC_", "DYLD_", "JAX_", "XLA_", "OMP_", "NLSQ_", "PYTHON",
)  # fmt: skip
_ENV_DUMP_EXACT = frozenset(
    {
        "DISPLAY", "WAYLAND_DISPLAY", "LANG", "LANGUAGE", "PATH", "LD_LIBRARY_PATH",
        "VIRTUAL_ENV", "MALLOCNANOZONE", "XDG_SESSION_TYPE",
    }
)  # fmt: skip
_ENV_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASS", "CRED")


def _env_dump_items(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    """Diagnostic subset of ``environ``: allowlisted names, secrets redacted."""
    items = []
    for k in sorted(environ):
        ku = k.upper()
        if ku in _ENV_DUMP_EXACT or ku.startswith(_ENV_DUMP_PREFIXES):
            redact = any(m in ku for m in _ENV_SECRET_MARKERS)
            items.append((k, "<redacted>" if redact else environ[k]))
    return items


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse launcher options; unknown (Qt) options are left for QApplication."""
    parser = argparse.ArgumentParser(
        prog="nlsq-gui",
        description="Launch the NLSQ Qt desktop application.",
        epilog="Standard Qt options (e.g. -style, -platform) are passed through.",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="dark",
        help="initial color scheme (default: dark)",
    )
    return parser.parse_known_args(list(argv))[0]


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("nlsq")
    except PackageNotFoundError:
        return "unknown"


def run_desktop(argv: Sequence[str] | None = None) -> int:
    """Launch the NLSQ Qt desktop application.

    Args:
        argv: Command-line arguments without the program name. Defaults to
            ``sys.argv[1:]``. ``--help`` and ``--version`` exit before any Qt
            object is created.

    Returns:
        int: Application exit code (0 for success)
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(argv)
    if args.version:
        print(f"nlsq-gui {_package_version()}")
        return 0

    _debug = os.environ.get("NLSQ_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    def _stage(msg: str) -> None:
        if _debug:
            sys.stderr.write(f"[nlsq-gui] {msg}\n")
            sys.stderr.flush()

    # Check both GUI extras before installing the Qt exception hook or creating
    # any Qt object so a missing extra never surfaces as a traceback.
    try:
        import pyqtgraph
        import PySide6.QtWidgets
    except ImportError as e:
        # Keep the cause: an installed PySide6 with a missing system library
        # (libGL, libxcb) is not fixed by installing the extra.
        sys.stderr.write(f"{GUI_INSTALL_HINT}\nDetails: {e}\n")
        return 1

    print("Launching NLSQ Qt GUI...")

    # Install global exception hook
    sys.excepthook = _exception_hook
    _stage("exception hook installed")

    from PySide6.QtGui import Qt
    from PySide6.QtWidgets import QApplication

    _stage("PySide6 imported")

    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.main_window import MainWindow

    _stage("app modules imported")

    # Create application
    app = QApplication([sys.argv[0], *argv])
    _stage("QApplication created")

    app.setApplicationName("NLSQ Curve Fitting")
    app.setOrganizationName("NLSQ")

    # Apply Fusion style for cross-platform consistency
    app.setStyle("Fusion")
    _stage("Fusion style applied")

    # Apply the requested (default dark) theme using Qt 6.5+ built-in color scheme.
    # Protected: some macOS + PySide6 combinations crash here.
    import contextlib

    with contextlib.suppress(Exception):
        app.styleHints().setColorScheme(
            Qt.ColorScheme.Light if args.theme == "light" else Qt.ColorScheme.Dark
        )
    _stage("color scheme set")

    # Let the platform plugin fully initialise its native surfaces before
    # constructing heavyweight widgets.  On macOS this ensures Metal has a
    # valid context, preventing SIGBUS when the first widget paints.
    app.processEvents()
    _stage("processEvents done")

    # Create centralized state
    app_state = AppState()
    _stage("AppState created")

    # Create and show main window
    window = MainWindow(app_state)
    _stage("MainWindow created")

    window.show()
    _stage("window shown — entering event loop")

    # Event-loop debug probes: these fire INSIDE app.exec() so we can see
    # exactly how far the event loop gets before the SIGBUS.
    if _debug:
        from PySide6.QtCore import QTimer as _QT

        _QT.singleShot(0, lambda: _stage("EVENT LOOP: first tick (0 ms)"))
        _QT.singleShot(100, lambda: _stage("EVENT LOOP: tick at 100 ms"))
        _QT.singleShot(500, lambda: _stage("EVENT LOOP: stable (500 ms)"))

    # Env dump: compare interactive terminal vs subprocess env when
    # diagnosing crashes that reproduce only interactively.
    if os.environ.get("NLSQ_DUMP_ENV"):
        _stage("=== ENVIRONMENT DUMP ===")
        for k, v in _env_dump_items(os.environ):
            _stage(f"  {k}={v}")
        _stage("=== END DUMP ===")

    return app.exec()
