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

import faulthandler
import os
import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

__all__ = ["run_desktop"]

# ---------------------------------------------------------------------------
# macOS SIGBUS Prevention — Defence in Depth
#
# SIGBUS on macOS is caused by Metal/OpenGL/XLA conflicts in the GPU driver.
# Layer 1 (nlsq/__init__.py): env vars — JAX_PLATFORM_NAME, MPLBACKEND, etc.
# Layer 2 (here): explicit matplotlib.use(), faulthandler, Qt RHI guard
# Layer 3 (plots/__init__.py): pyqtgraph useOpenGL=False
# Layer 4 (main_window.py): deferred QMessageBox via QTimer.singleShot
# ---------------------------------------------------------------------------

# Enable faulthandler early so any SIGBUS/SIGSEGV prints a traceback
# to stderr instead of dying silently.
faulthandler.enable()

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
                        json.dumps(data, indent=2), encoding="utf-8"
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
                    "Your session will be recovered on next launch."
                )
                msg.setDetailedText(tb_str)
                msg.exec()
    except Exception:
        pass


def run_desktop() -> int:
    """Launch the NLSQ Qt desktop application.

    Returns:
        int: Application exit code (0 for success)
    """
    _debug = bool(os.environ.get("NLSQ_DEBUG"))

    def _stage(msg: str) -> None:
        if _debug:
            sys.stderr.write(f"[nlsq-gui] {msg}\n")
            sys.stderr.flush()

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
    app = QApplication(sys.argv)
    _stage("QApplication created")

    app.setApplicationName("NLSQ Curve Fitting")
    app.setOrganizationName("NLSQ")

    # Apply Fusion style for cross-platform consistency
    app.setStyle("Fusion")
    _stage("Fusion style applied")

    # Apply dark theme by default using Qt 6.5+ built-in color scheme.
    # Protected: some macOS + PySide6 combinations crash here.
    import contextlib

    with contextlib.suppress(Exception):
        app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
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
        for k in sorted(os.environ):
            if any(
                p in k.upper()
                for p in (
                    "QT",
                    "DISPLAY",
                    "XDG",
                    "LANG",
                    "LC_",
                    "DYLD",
                    "MallocNanoZone",
                    "METAL",
                    "GPU",
                    "GL",
                    "JAX",
                    "XLA",
                    "OMP",
                    "NLSQ",
                    "PYTHON",
                    "VIRTUAL",
                    "PATH",
                )
            ):
                _stage(f"  {k}={os.environ[k]}")
        _stage("=== END DUMP ===")

    return app.exec()
