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
    session_restored = Signal()  # Rehydrate controls after crash recovery
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
        # Bumped whenever data or model change; a fit started under an older
        # epoch must not attach its result to the new inputs.
        self.input_epoch = 0
        # Settings the displayed fit_result was computed with (see fit_settings)
        self._fit_settings: SessionState | None = None

    @property
    def fit_settings(self) -> SessionState:
        """Settings that produced the current fit result.

        Exports must describe the fit whose popt/pcov are shown, not whatever
        the options widgets say now. Falls back to live state when no snapshot
        was recorded (result set directly, or no result yet).

        Returns
        -------
        SessionState
            The snapshot taken when the current fit result was stored, or the
            live session state if there is no result or no snapshot.
        """
        if self._state.fit_result is not None and self._fit_settings is not None:
            return self._fit_settings
        return self._state

    @property
    def state(self) -> SessionState:
        """Get the underlying SessionState.

        Returns
        -------
        SessionState
            The wrapped session state.
        """
        return self._state

    @property
    def theme(self) -> str:
        """Get the current theme name.

        Returns
        -------
        str
            Current theme name (``"light"`` or ``"dark"``).
        """
        return self._theme

    def set_data(
        self,
        xdata: NDArray[np.floating] | None,
        ydata: NDArray[np.floating] | None,
        sigma: NDArray[np.floating] | None = None,
        file_name: str | None = None,
    ) -> None:
        """Update data arrays and emit signal.

        Parameters
        ----------
        xdata : ndarray or None
            X data array.
        ydata : ndarray or None
            Y data array.
        sigma : ndarray, optional
            Uncertainty array.
        file_name : str, optional
            Source file name.
        """
        self._state.xdata = xdata
        self._state.ydata = ydata
        self._state.sigma = sigma
        self._state.data_file_name = file_name
        self.input_epoch += 1
        self._clear_fit_result()
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

        Parameters
        ----------
        model_type : str
            Model type (``"builtin"``, ``"polynomial"``, ``"custom"``).
        config : dict, optional
            Model configuration dictionary.
        model_func : callable, optional
            The model function.
        **kwargs
            Additional model configuration (legacy); each key that is an
            attribute of the session state is set on it.
        """
        previous = self._state.model_config  # None right after a YAML import
        model_changed = model_type != self._state.model_type or (
            config is not None and previous is not None and config != previous
        )
        self._state.model_type = model_type
        self.input_epoch += 1

        # p0/bounds belong to the previous model's parameters; replaying them
        # positionally onto a different model silently mis-constrains the fit.
        if model_changed:
            self._state.p0 = None
            self._state.bounds = None
            if model_func is None:  # the old callable would run under the new name
                self._state.model_func = None

        # Store config and function
        if config is not None:
            self._state.model_config = config
        if model_func is not None:
            self._state.model_func = model_func

        # Handle legacy kwargs
        for key, value in kwargs.items():
            if hasattr(self._state, key):
                setattr(self._state, key, value)

        self._clear_fit_result()
        self.model_changed.emit()
        self.page_access_changed.emit()

    def _clear_fit_result(self) -> None:
        """Invalidate a stale fit result when data or model changes.

        Without this, Results/Export pages stay accessible (they gate purely
        on ``fit_result is not None``) and read live xdata/ydata/model_func
        alongside a fit computed against the previous data or model.
        """
        if self._state.fit_result is not None:
            self._state.fit_result = None
            self._fit_settings = None
            self.fit_completed.emit(None)

    def set_fit_running(self, running: bool) -> None:
        """Set the fit running state.

        Parameters
        ----------
        running : bool
            ``True`` if a fit is in progress.
        """
        if running:
            # A previous fit_result is about to be superseded. Clear it now
            # (not just on success) so a re-fit that errors or is aborted
            # doesn't leave Results/Export showing a now-unrelated result --
            # PageState.can_access() gates purely on fit_result is not None,
            # with no regard for fit_running/fit_aborted.
            self._clear_fit_result()
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

    def set_fit_result(
        self, result: object, settings: SessionState | None = None
    ) -> None:
        """Store fit result and emit signal.

        Parameters
        ----------
        result : object
            The fit result object.
        settings : SessionState, optional
            Snapshot of the settings the fit ran with (see ``fit_settings``).
        """
        self._state.fit_result = result
        self._fit_settings = settings
        self._state.fit_running = False
        self._state.fit_aborted = False
        self.fit_completed.emit(result)
        self.page_access_changed.emit()

    def emit_fit_progress(self, iteration: int, cost: float) -> None:
        """Emit fit progress signal.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        cost : float
            Current cost function value.
        """
        self.fit_progress.emit(iteration, cost)

    def set_theme_name(self, theme: str) -> None:
        """Set the theme name and emit signal.

        Parameters
        ----------
        theme : str
            Theme name (``"light"`` or ``"dark"``).
        """
        self._theme = theme
        self.theme_changed.emit(theme)

    def reset(self) -> None:
        """Reset state to initial values."""
        had_fit_result = self._state.fit_result is not None
        self._state = SessionState()
        self._fit_settings = None
        self.input_epoch += 1
        if had_fit_result:
            self.fit_completed.emit(None)
        self.data_changed.emit()
        self.model_changed.emit()
        self.page_access_changed.emit()
