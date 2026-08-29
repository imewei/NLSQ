"""Tests for ResidualsPlotWidget.get_statistics() -- finite-value filtering.

Uses __new__ to bypass Qt widget construction (and the pyqtgraph OpenGL
rendering set_residuals() would trigger via _update_plot()) entirely, since
get_statistics() only reads self._residuals -- pure numpy, no Qt/GL
involved. This avoids the SIGABRT-on-OpenGL-init crash documented in
test_gui_startup.py for real widget construction on headless Linux CI,
so these run without a QApplication and without @pytest.mark.serial.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget


def _make_widget_stub(residuals: np.ndarray) -> ResidualsPlotWidget:
    widget = ResidualsPlotWidget.__new__(ResidualsPlotWidget)
    widget._residuals = residuals
    return widget


class TestGetStatisticsFiniteFiltering:
    """get_statistics() must filter non-finite values before computing
    mean/std/min/max/median, matching the widget's own finite-filtering
    elsewhere -- otherwise a diverged fit (NaN/Inf residuals) returns
    all-NaN stats instead of stats over the finite subset."""

    def test_nan_values_excluded_from_statistics(self):
        widget = _make_widget_stub(np.array([1.0, -1.0, np.nan, 2.0, -2.0]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["max"])

    def test_inf_values_excluded_from_statistics(self):
        widget = _make_widget_stub(np.array([1.0, -1.0, np.inf, -np.inf]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["max"] == pytest.approx(1.0)

    def test_all_non_finite_returns_empty_dict(self):
        widget = _make_widget_stub(np.array([np.nan, np.inf, -np.inf]))

        assert widget.get_statistics() == {}
