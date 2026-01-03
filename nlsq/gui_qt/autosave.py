"""
NLSQ Qt GUI Autosave Manager

This module provides automatic session state saving and crash recovery
functionality for the Qt GUI application.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState

__all__ = ["AutosaveManager"]

# Autosave directory
AUTOSAVE_DIR = Path(tempfile.gettempdir()) / "nlsq_autosave"
AUTOSAVE_FILE = AUTOSAVE_DIR / "session_autosave.json"
AUTOSAVE_INTERVAL_MS = 60000  # 1 minute


class AutosaveManager(QObject):
    """Manages automatic session saving and recovery.

    Provides:
    - Periodic autosave of session state
    - Crash recovery on startup
    - Clean shutdown handling
    """

    # Signals
    autosave_completed = Signal()
    recovery_available = Signal(str)  # timestamp of recovery file

    def __init__(self, app_state: AppState, parent: QObject | None = None) -> None:
        """Initialize the autosave manager.

        Args:
            app_state: Application state manager
            parent: Parent QObject
        """
        super().__init__(parent)
        self._app_state = app_state
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._do_autosave)

        # Ensure autosave directory exists
        AUTOSAVE_DIR.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        """Start the autosave timer."""
        self._timer.start(AUTOSAVE_INTERVAL_MS)

    def stop(self) -> None:
        """Stop the autosave timer."""
        self._timer.stop()

    def check_recovery(self) -> bool:
        """Check if a recovery file exists.

        Returns:
            True if recovery file exists and is recent
        """
        if not AUTOSAVE_FILE.exists():
            return False

        try:
            data = json.loads(AUTOSAVE_FILE.read_text(encoding="utf-8"))
            timestamp = data.get("timestamp")
            if timestamp:
                self.recovery_available.emit(timestamp)
                return True
        except Exception:
            pass

        return False

    def recover(self) -> bool:
        """Attempt to recover session from autosave.

        Returns:
            True if recovery was successful
        """
        if not AUTOSAVE_FILE.exists():
            return False

        try:
            data = json.loads(AUTOSAVE_FILE.read_text(encoding="utf-8"))
            self._restore_state(data)
            return True
        except Exception:
            return False

    def clear_recovery(self) -> None:
        """Clear the recovery file."""
        if AUTOSAVE_FILE.exists():
            AUTOSAVE_FILE.unlink()

    def save_now(self) -> None:
        """Perform an immediate save."""
        self._do_autosave()

    def _do_autosave(self) -> None:
        """Perform the autosave operation."""
        try:
            data = self._serialize_state()
            AUTOSAVE_FILE.write_text(
                json.dumps(data, indent=2),
                encoding="utf-8",
            )
            self.autosave_completed.emit()
        except Exception:
            # Silently fail - autosave is best-effort
            pass

    def _serialize_state(self) -> dict[str, Any]:
        """Serialize current session state to a dictionary.

        Returns:
            Serializable dictionary
        """
        state = self._app_state.state
        data: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
        }

        # Serialize data arrays
        if state.xdata is not None:
            data["xdata"] = state.xdata.tolist()
        if state.ydata is not None:
            data["ydata"] = state.ydata.tolist()
        if state.sigma is not None:
            data["sigma"] = state.sigma.tolist()

        # Serialize model config
        if state.model_config is not None:
            data["model_config"] = state.model_config

        # Serialize fitting options
        if state.p0 is not None:
            data["p0"] = list(state.p0)
        if state.bounds is not None:
            data["bounds"] = state.bounds

        # Serialize other state attributes
        data["auto_p0"] = getattr(state, "auto_p0", True)

        return data

    def _restore_state(self, data: dict[str, Any]) -> None:
        """Restore session state from a dictionary.

        Args:
            data: Serialized state dictionary
        """
        state = self._app_state.state

        # Restore data arrays
        if "xdata" in data:
            state.xdata = np.array(data["xdata"])
        if "ydata" in data:
            state.ydata = np.array(data["ydata"])
        if "sigma" in data:
            state.sigma = np.array(data["sigma"])

        # Restore model config
        if "model_config" in data:
            state.model_config = data["model_config"]

        # Restore fitting options
        if "p0" in data:
            state.p0 = data["p0"]
        if "bounds" in data:
            state.bounds = data["bounds"]

        # Restore other attributes
        if "auto_p0" in data:
            state.auto_p0 = data["auto_p0"]

        # Emit signals to update UI
        if state.xdata is not None:
            self._app_state.data_changed.emit()
        if state.model_config is not None:
            self._app_state.model_changed.emit()
