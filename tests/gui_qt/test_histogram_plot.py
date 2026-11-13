"""Tests for HistogramPlotWidget.get_statistics() -- finite-value filtering.

Uses __new__ to bypass Qt widget construction (and the pyqtgraph OpenGL
rendering set_data() would trigger via _update_plot()) entirely, since
get_statistics() only reads self._data -- pure numpy, no Qt/GL involved.
This avoids the SIGABRT-on-OpenGL-init crash documented in
test_gui_startup.py for real widget construction on headless Linux CI,
so these run without a QApplication and without @pytest.mark.serial.
"""

from __future__ import annotations

import numpy as np
import pytest

from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget, _histogram


def _make_widget_stub(data: np.ndarray) -> HistogramPlotWidget:
    widget = HistogramPlotWidget.__new__(HistogramPlotWidget)
    widget._data = data
    return widget


class TestGetStatisticsFiniteFiltering:
    """get_statistics() must filter non-finite values before computing
    mean/std/skewness/kurtosis, matching _update_plot's own finite-filtering
    -- otherwise a diverged fit (NaN/Inf in the residuals) returns all-NaN
    stats instead of stats over the finite subset."""

    def test_nan_values_excluded_from_statistics(self):
        widget = _make_widget_stub(np.array([1.0, 2.0, 3.0, np.nan, np.nan]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["mean"] == pytest.approx(2.0)

    def test_inf_values_excluded_from_statistics(self):
        widget = _make_widget_stub(np.array([1.0, 2.0, 3.0, np.inf, -np.inf]))

        stats = widget.get_statistics()

        assert np.isfinite(stats["mean"])
        assert stats["mean"] == pytest.approx(2.0)

    def test_all_non_finite_returns_empty_dict(self):
        widget = _make_widget_stub(np.array([np.nan, np.inf, -np.inf]))

        assert widget.get_statistics() == {}


class TestDegenerateRangeHistogram:
    """np.histogram raises "Too many bins for data range" when the finite data
    has no representable bin width; the Results page must show one bar instead."""

    @pytest.mark.parametrize(
        "data",
        [
            np.array([0.0, 5e-324, 1e-323] * 10),  # subnormal spread
        ],
    )
    def test_numpy_would_raise_but_helper_returns_one_bar(self, data):
        with np.errstate(all="ignore"), pytest.raises(ValueError, match="bins"):
            np.histogram(data, bins=30)

        with np.errstate(all="ignore"):
            counts, edges = _histogram(data, 30)

        assert counts.tolist() == [data.size]
        assert len(edges) == 2
        assert np.all(np.isfinite(edges)) and edges[1] > edges[0]

    def test_overflowing_range_keeps_real_counts(self):
        data = np.array([-1e308, -1e308, 1e308])
        with np.errstate(all="ignore"):
            counts, edges = _histogram(data, 30)
        assert counts.sum() == 3 and counts[0] == 2 and counts[-1] == 1
        assert np.all(np.isfinite(edges)) and edges[0] <= -1e308 and edges[-1] >= 1e308

    def test_regular_data_unchanged(self):
        data = np.linspace(0.0, 1.0, 101)
        counts, edges = _histogram(data, 10)
        expected_counts, expected_edges = np.histogram(data, bins=10)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(edges, expected_edges)

    def test_constant_data_still_uses_numpy(self):
        counts, edges = _histogram(np.full(8, 3.0), 5)
        assert counts.sum() == 8 and len(edges) == 6
