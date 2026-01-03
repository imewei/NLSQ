"""
NLSQ Qt GUI Application State

This module provides the AppState class, a Qt-observable wrapper around SessionState
that emits signals for reactive UI updates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QObject, Signal

from nlsq.gui_qt.session_state import SessionState

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["AppState"]


class AppState(QObject):
    """Qt-observable wrapper around SessionState.

    This class wraps the existing SessionState dataclass and provides Qt signals
    for reactive UI updates. All state changes should go through this class's
    methods to ensure signals are properly emitted.
    """

    # Signals for reactive updates
    data_changed = Signal()  # When xdata/ydata/sigma changes
    model_changed = Signal()  # When model configuration changes
    fit_started = Signal()  # When fitting begins
    fit_progress = Signal(int, float)  # (iteration, cost)
    fit_completed = Signal(object)  # When fitting completes (result)
    fit_aborted = Signal()  # When fit is cancelled
    theme_changed = Signal(str)  # "light" or "dark"
    page_access_changed = Signal()  # When navigation guards update

    def __init__(self) -> None:
        """Initialize AppState with a fresh SessionState."""
        super().__init__()
        self._state = SessionState()
        self._theme = "dark"

    @property
    def state(self) -> SessionState:
        """Get the underlying SessionState."""
        return self._state

    @property
    def theme(self) -> str:
        """Get the current theme name."""
        return self._theme

    def set_data(
        self,
        xdata: NDArray[np.floating] | None,
        ydata: NDArray[np.floating] | None,
        sigma: NDArray[np.floating] | None = None,
        file_name: str | None = None,
    ) -> None:
        """Update data arrays and emit signal.

        Args:
            xdata: X data array
            ydata: Y data array
            sigma: Optional uncertainty array
            file_name: Optional source file name
        """
        self._state.xdata = xdata
        self._state.ydata = ydata
        self._state.sigma = sigma
        self._state.data_file_name = file_name
        self.data_changed.emit()
        self.page_access_changed.emit()

    def set_model(
        self,
        model_type: str,
        config: dict | None = None,
        model_func: object | None = None,
        **kwargs: object,
    ) -> None:
        """Update model configuration and emit signal.

        Args:
            model_type: Model type ("builtin", "polynomial", "custom")
            config: Model configuration dictionary
            model_func: The model function callable
            **kwargs: Additional model configuration (legacy)
        """
        self._state.model_type = model_type

        # Store config and function
        if config is not None:
            self._state.model_config = config
        if model_func is not None:
            self._state.model_func = model_func

        # Handle legacy kwargs
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

        self.model_changed.emit()
        self.page_access_changed.emit()

    def set_fit_running(self, running: bool) -> None:
        """Set the fit running state.

        Args:
            running: True if fit is in progress
        """
        self._state.fit_running = running
        if running:
            self.fit_started.emit()
        self.page_access_changed.emit()

    def set_fit_aborted(self) -> None:
        """Mark the fit as aborted."""
        self._state.fit_running = False
        self._state.fit_aborted = True
        self.fit_aborted.emit()
        self.page_access_changed.emit()

    def set_fit_result(self, result: object) -> None:
        """Store fit result and emit signal.

        Args:
            result: The fit result object
        """
        self._state.fit_result = result
        self._state.fit_running = False
        self._state.fit_aborted = False
        self.fit_completed.emit(result)
        self.page_access_changed.emit()

    def emit_fit_progress(self, iteration: int, cost: float) -> None:
        """Emit fit progress signal.

        Args:
            iteration: Current iteration number
            cost: Current cost function value
        """
        self.fit_progress.emit(iteration, cost)

    def set_theme_name(self, theme: str) -> None:
        """Set the theme name and emit signal.

        Args:
            theme: Theme name ("light" or "dark")
        """
        self._theme = theme
        self.theme_changed.emit(theme)

    def reset(self) -> None:
        """Reset state to initial values."""
        self._state = SessionState()
        self.data_changed.emit()
        self.model_changed.emit()
        self.page_access_changed.emit()
