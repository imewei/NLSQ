"""
NLSQ Qt GUI Pages

This module contains QWidget-based page implementations for the NLSQ curve fitting workflow.
Each page corresponds to a step in the workflow: Data Loading, Model Selection,
Fitting Options, Results, and Export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlsq.gui_qt.session_state import SessionState

__all__ = [
    "DataLoadingPage",
    "ExportPage",
    "FittingOptionsPage",
    "ModelSelectionPage",
    "PageState",
    "ResultsPage",
]


def __getattr__(name: str):
    """Lazy import pages to avoid importing Qt dependencies at module load time."""
    if name == "DataLoadingPage":
        from nlsq.gui_qt.pages.data_loading import DataLoadingPage

        return DataLoadingPage
    elif name == "ModelSelectionPage":
        from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

        return ModelSelectionPage
    elif name == "FittingOptionsPage":
        from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

        return FittingOptionsPage
    elif name == "ResultsPage":
        from nlsq.gui_qt.pages.results import ResultsPage

        return ResultsPage
    elif name == "ExportPage":
        from nlsq.gui_qt.pages.export import ExportPage

        return ExportPage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class PageState:
    """Navigation state derived from SessionState.

    Used to determine which pages are accessible based on the current workflow state.
    """

    data_loaded: bool = False
    model_selected: bool = False
    fit_complete: bool = False
    fit_running: bool = False

    @classmethod
    def from_session_state(cls, state: SessionState) -> PageState:
        """Create PageState from the current SessionState.

        Args:
            state: The current session state

        Returns:
            PageState with computed access flags
        """
        data_loaded = state.xdata is not None and state.ydata is not None
        model_selected = bool(state.model_type) and (
            state.model_type in {"builtin", "polynomial"}
            or (state.model_type == "custom" and state.custom_code)
        )
        fit_complete = state.fit_result is not None
        fit_running = getattr(state, "fit_running", False)

        return cls(
            data_loaded=data_loaded,
            model_selected=model_selected,
            fit_complete=fit_complete,
            fit_running=fit_running,
        )

    def can_access(self, page: str) -> bool:
        """Check if a page is accessible based on workflow state.

        Args:
            page: Page name (data_loading, model_selection, fitting_options, results, export)

        Returns:
            True if the page can be accessed
        """
        access_rules = {
            "data_loading": True,
            "model_selection": self.data_loaded,
            "fitting_options": self.data_loaded and self.model_selected,
            "results": self.fit_complete,
            "export": self.fit_complete,
        }
        return access_rules.get(page, True)
