"""
NLSQ Qt GUI Plot Widgets

This module contains pyqtgraph-based plot widgets for scientific visualization.
All plots support theme switching and are optimized for large datasets (500K+ points).
"""

from __future__ import annotations

__all__: list[str] = [
    "BasePlot",
    "LiveCostPlot",
    "LiveCostPlotWidget",
    "FitPlotWidget",
    "ResidualsPlotWidget",
    "HistogramPlotWidget",
]

# Flag to track if pyqtgraph has been configured
_pg_configured = False


def _configure_pyqtgraph() -> None:
    """Configure pyqtgraph global settings for optimal performance."""
    global _pg_configured
    if not _pg_configured:
        import pyqtgraph as pg

        pg.setConfigOptions(
            useOpenGL=True,  # GPU acceleration for large datasets
            antialias=True,  # Smooth lines
            enableExperimental=False,  # Stable features only
        )
        _pg_configured = True


def __getattr__(name: str):
    """Lazy import plot widgets to avoid importing Qt dependencies at module load time."""
    _configure_pyqtgraph()

    if name == "BasePlot":
        from nlsq.gui_qt.plots.base_plot import BasePlotWidget

        return BasePlotWidget
    elif name == "LiveCostPlot":
        from nlsq.gui_qt.plots.live_cost_plot import LiveCostPlotWidget

        return LiveCostPlotWidget
    elif name == "LiveCostPlotWidget":
        from nlsq.gui_qt.plots.live_cost_plot import LiveCostPlotWidget

        return LiveCostPlotWidget
    elif name == "FitPlotWidget":
        from nlsq.gui_qt.plots.fit_plot import FitPlotWidget

        return FitPlotWidget
    elif name == "ResidualsPlotWidget":
        from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

        return ResidualsPlotWidget
    elif name == "HistogramPlotWidget":
        from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

        return HistogramPlotWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
