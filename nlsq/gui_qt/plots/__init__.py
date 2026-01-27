"""
NLSQ Qt GUI Plot Widgets

This module contains pyqtgraph-based plot widgets for scientific visualization.
All plots support theme switching and are optimized for large datasets (500K+ points).
"""

from __future__ import annotations

import logging
import os
import sys

__all__: list[str] = [
    "BasePlot",
    "FitPlotWidget",
    "HistogramPlotWidget",
    "LiveCostPlot",
    "LiveCostPlotWidget",
    "ResidualsPlotWidget",
]

# Mutable container to track if pyqtgraph has been configured
_pg_state = {"configured": False}


_log = logging.getLogger(__name__)


def _configure_pyqtgraph() -> None:
    """Configure pyqtgraph global settings for optimal performance.

    On macOS, OpenGL acceleration is disabled by default because Apple
    deprecated OpenGL in macOS 10.14 and has progressively reduced driver
    support.  PySide6 (6.5+) uses Metal for its rendering backend, and
    creating an OpenGL context alongside Metal can produce SIGBUS crashes
    at the GPU-driver level.

    Set the environment variable ``NLSQ_GUI_USE_OPENGL=1`` to force-enable
    OpenGL on macOS at your own risk (e.g. for benchmarking).
    """
    if not _pg_state["configured"]:
        import pyqtgraph as pg

        use_opengl_env = os.getenv("NLSQ_GUI_USE_OPENGL", "").strip().lower()
        if use_opengl_env:
            use_opengl = use_opengl_env in {"1", "true", "yes", "on"}
        else:
            # macOS deprecated OpenGL in favour of Metal (10.14+).  Mixing an
            # OpenGL rendering context with PySide6's Metal backend causes
            # SIGBUS on macOS 26+.  Default to software rendering there.
            use_opengl = sys.platform != "darwin"

        if not use_opengl and sys.platform == "darwin":
            _log.info(
                "pyqtgraph OpenGL disabled on macOS (Apple deprecated OpenGL). "
                "Set NLSQ_GUI_USE_OPENGL=1 to override."
            )

        pg.setConfigOptions(
            useOpenGL=use_opengl,
            antialias=True,
            enableExperimental=False,
        )
        _pg_state["configured"] = True


# Eagerly configure pyqtgraph when this package is imported (not just when
# widgets are accessed via __getattr__).  Direct submodule imports like
# ``from nlsq.gui_qt.plots.fit_plot import FitPlotWidget`` bypass __getattr__
# but still trigger __init__.py, so this ensures the OpenGL/Metal guard
# always runs before any pyqtgraph widget is created.
_configure_pyqtgraph()


def __getattr__(name: str):
    """Lazy import plot widgets to avoid importing Qt dependencies at module load time."""
    _configure_pyqtgraph()

    if name == "BasePlot":
        from nlsq.gui_qt.plots.base_plot import BasePlotWidget

        return BasePlotWidget
    elif name in {"LiveCostPlot", "LiveCostPlotWidget"}:
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
