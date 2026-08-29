"""Tests for ResidualsPlotWidget.get_statistics() -- finite-value filtering."""

from __future__ import annotations

import numpy as np
import pytest


class TestGetStatisticsFiniteFiltering:
    """get_statistics() must filter non-finite values before computing
    mean/std/min/max/median, matching the widget's own finite-filtering
    elsewhere -- otherwise a diverged fit (NaN/Inf residuals) returns
    all-NaN stats instead of stats over the finite subset."""

    def test_nan_values_excluded_from_statistics(self, qtbot):
        from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

        widget = ResidualsPlotWidget()
        qtbot.addWidget(widget)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.array([1.0, -1.0, np.nan, 2.0, -2.0])
        widget.set_residuals(x, residuals)

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert np.isfinite(stats["max"])

    def test_inf_values_excluded_from_statistics(self, qtbot):
        from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

        widget = ResidualsPlotWidget()
        qtbot.addWidget(widget)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        residuals = np.array([1.0, -1.0, np.inf, -np.inf])
        widget.set_residuals(x, residuals)

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["max"] == pytest.approx(1.0)

    def test_all_non_finite_returns_empty_dict(self, qtbot):
        from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget

        widget = ResidualsPlotWidget()
        qtbot.addWidget(widget)
        x = np.array([1.0, 2.0, 3.0])
        residuals = np.array([np.nan, np.inf, -np.inf])
        widget.set_residuals(x, residuals)

        assert widget.get_statistics() == {}
