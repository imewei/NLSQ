"""Tests for HistogramPlotWidget.get_statistics() -- finite-value filtering."""

from __future__ import annotations

import numpy as np
import pytest


class TestGetStatisticsFiniteFiltering:
    """get_statistics() must filter non-finite values before computing
    mean/std/skewness/kurtosis, matching _update_plot's own finite-filtering
    -- otherwise a diverged fit (NaN/Inf in the residuals) returns all-NaN
    stats instead of stats over the finite subset."""

    def test_nan_values_excluded_from_statistics(self, qtbot):
        from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

        widget = HistogramPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(np.array([1.0, 2.0, 3.0, np.nan, np.nan]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["mean"] == pytest.approx(2.0)

    def test_inf_values_excluded_from_statistics(self, qtbot):
        from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

        widget = HistogramPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(np.array([1.0, 2.0, 3.0, np.inf, -np.inf]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["mean"] == pytest.approx(2.0)

    def test_all_non_finite_returns_empty_dict(self, qtbot):
        from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget

        widget = HistogramPlotWidget()
        qtbot.addWidget(widget)
        widget.set_data(np.array([np.nan, np.inf, -np.inf]))

        assert widget.get_statistics() == {}
