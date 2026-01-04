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

import sys
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

__all__ = ["run_desktop"]


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

            # Show error dialog
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
    # Install global exception hook
    sys.excepthook = _exception_hook

    from PySide6.QtGui import Qt
    from PySide6.QtWidgets import QApplication

    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.main_window import MainWindow

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("NLSQ Curve Fitting")
    app.setOrganizationName("NLSQ")

    # Apply Fusion style for cross-platform consistency
    app.setStyle("Fusion")

    # Apply dark theme by default using Qt 6.5+ built-in color scheme
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)

    # Create centralized state
    app_state = AppState()

    # Create and show main window
    window = MainWindow(app_state)
    window.show()

    return app.exec()
