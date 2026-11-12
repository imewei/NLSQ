"""
NLSQ Qt GUI Plot Widgets

This module contains pyqtgraph-based plot widgets for scientific visualization.
All plots support theme switching and are optimized for large datasets (500K+ points).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = [
    "FitPlotWidget",
    "HistogramPlotWidget",
    "LiveCostPlot",
    "LiveCostPlotWidget",
    "ResidualsPlotWidget",
]

# Mutable container to track if pyqtgraph has been configured
_pg_state = {"configured": False}


_log = logging.getLogger(__name__)


def _downsample(
    x: NDArray, y: NDArray, threshold: int, *, log: bool = True
) -> tuple[NDArray, NDArray]:
    """Keep paired plot data on the existing floor-division stride.

    Decimation is display-only (fits always use the full data) but is logged
    so a thinned plot is never mistaken for the whole dataset.
    """
    if len(x) <= threshold:
        return x, y
    stride = len(x) // threshold
    if log:
        _log.info(
            "Plot shows every %d-th point (%d of %d)",
            stride,
            len(x[::stride]),
            len(x),
        )
    return x[::stride], y[::stride]


def decimation_note(n: int, threshold: int) -> str:
    """Status-line text describing plot decimation, or "" if none applies."""
    if n <= threshold:
        return ""
    stride = n // threshold
    return f"plots show {len(range(0, n, stride))} of {n} points"


_NO_GL_PLATFORMS = frozenset({"offscreen", "minimal", "minimalegl", "vnc"})


def _opengl_default() -> tuple[bool, bool]:
    """Decide the OpenGL default as ``(use_opengl, final)``.

    ``final`` is False while no QGuiApplication exists: only the platform
    environment variable can be inspected then, so widgets re-run the decision
    once the application (and a probe GL context) is available.
    """
    if sys.platform == "darwin":
        # Apple deprecated OpenGL in favour of Metal (10.14+); mixing a GL
        # context with PySide6's Metal backend causes SIGBUS on macOS 26+.
        return False, True
    env_plat = os.getenv("QT_QPA_PLATFORM", "").split(":")[0].strip().lower()
    if env_plat in _NO_GL_PLATFORMS:
        return False, True
    from PySide6.QtGui import QGuiApplication, QOpenGLContext

    if QGuiApplication.instance() is None:
        return True, False
    if QGuiApplication.platformName().lower() in _NO_GL_PLATFORMS:
        return False, True
    # No usable GL (VM, remote X, CI): pyqtgraph's GL viewport would emit
    # paintGL tracebacks and render nothing, so fall back to software.
    try:
        ctx = QOpenGLContext()
        usable = bool(ctx.create() and ctx.isValid())
    except Exception:
        usable = False
    return usable, True


def _configure_pyqtgraph() -> None:
    """Configure pyqtgraph global settings for optimal performance.

    OpenGL acceleration is disabled by default on macOS (Metal/SIGBUS), on the
    ``offscreen``/``minimal`` Qt platform plugins, and whenever a probe
    ``QOpenGLContext`` cannot be created.  ``NLSQ_GUI_USE_OPENGL=1`` /
    ``=0`` overrides the default in either direction (at your own risk).
    """
    if not _pg_state["configured"]:
        import pyqtgraph as pg

        use_opengl_env = os.getenv("NLSQ_GUI_USE_OPENGL", "").strip().lower()
        if use_opengl_env:
            use_opengl, final = use_opengl_env in {"1", "true", "yes", "on"}, True
        else:
            use_opengl, final = _opengl_default()

        if not use_opengl and final:
            _log.info(
                "pyqtgraph OpenGL disabled (macOS, offscreen/minimal platform, or "
                "no usable GL context). Set NLSQ_GUI_USE_OPENGL=1 to override.",
            )

        pg.setConfigOptions(
            useOpenGL=use_opengl,
            antialias=True,
            enableExperimental=False,
        )
        _pg_state["configured"] = final


# Eagerly configure pyqtgraph when this package is imported (not just when
# widgets are accessed via __getattr__).  Direct submodule imports like
# ``from nlsq.gui_qt.plots.fit_plot import FitPlotWidget`` bypass __getattr__
# but still trigger __init__.py, so this ensures the OpenGL/Metal guard
# always runs before any pyqtgraph widget is created.
_configure_pyqtgraph()


def __getattr__(name: str):
    """Lazy import plot widgets to avoid importing Qt dependencies at module load time."""
    _configure_pyqtgraph()

    if name in {"LiveCostPlot", "LiveCostPlotWidget"}:
        from nlsq.gui_qt.plots.live_cost_plot import LiveCostPlotWidget

        return LiveCostPlotWidget
    if name == "FitPlotWidget":
        from nlsq.gui_qt.plots.fit_plot import FitPlotWidget

        return FitPlotWidget
    if name == "ResidualsPlotWidget":
        from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

        return ResidualsPlotWidget
    if name == "HistogramPlotWidget":
        from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

        return HistogramPlotWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
