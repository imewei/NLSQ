"""
NLSQ Qt GUI Widgets

This module contains reusable Qt widget components for the NLSQ curve fitting application.
These widgets provide specialized functionality like column selection, code editing,
parameter configuration, and results display.
"""

from __future__ import annotations

__all__: list[str] = [
    "AdvancedOptionsWidget",
    "CodeEditor",
    "CodeEditorWidget",
    "ColumnSelector",
    "ColumnSelectorWidget",
    "FitStatisticsWidget",
    "IterationTableWidget",
    "ParamConfigWidget",
    "ParamResultsWidget",
    "PythonHighlighter",
]


def __getattr__(name: str):
    """Lazy import widgets to avoid importing Qt dependencies at module load time."""
    if name in ("ColumnSelector", "ColumnSelectorWidget"):
        from nlsq.gui_qt.widgets.column_selector import ColumnSelectorWidget

        return ColumnSelectorWidget
    elif name in ("CodeEditor", "CodeEditorWidget"):
        from nlsq.gui_qt.widgets.code_editor import CodeEditorWidget

        return CodeEditorWidget
    elif name == "PythonHighlighter":
        from nlsq.gui_qt.widgets.code_editor import PythonHighlighter

        return PythonHighlighter
    elif name == "ParamConfigWidget":
        from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

        return ParamConfigWidget
    elif name == "AdvancedOptionsWidget":
        from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

        return AdvancedOptionsWidget
    elif name == "IterationTableWidget":
        from nlsq.gui_qt.widgets.iteration_table import IterationTableWidget

        return IterationTableWidget
    elif name == "ParamResultsWidget":
        from nlsq.gui_qt.widgets.param_results import ParamResultsWidget

        return ParamResultsWidget
    elif name == "FitStatisticsWidget":
        from nlsq.gui_qt.widgets.fit_statistics import FitStatisticsWidget

        return FitStatisticsWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
