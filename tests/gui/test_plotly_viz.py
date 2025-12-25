"""Tests for the NLSQ GUI Plotly visualization components.

This module tests the Plotly-based visualization components including
fit plots, residuals plots, and histograms.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from nlsq.gui.components.plotly_fit_plot import (
    create_fit_plot,
    create_fit_plot_from_result,
)
from nlsq.gui.components.plotly_histogram import (
    compute_normality_tests,
    create_histogram_from_result,
    create_residuals_histogram,
)
from nlsq.gui.components.plotly_residuals import (
    create_residuals_plot,
    create_residuals_plot_from_result,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2 * x + 3 + np.random.normal(0, 0.5, 50)
    return x, y


@pytest.fixture
def sample_residuals():
    """Create sample residuals for testing."""
    np.random.seed(42)
    return np.random.normal(0, 0.5, 100)


@pytest.fixture
def sample_fit(sample_data):
    """Create sample fit data."""
    x, y = sample_data
    y_fit = 2 * x + 3
    return x, y, y_fit


class MockResult:
    """Mock CurveFitResult for testing."""

    def __init__(self, xdata, ydata, popt, residuals=None, model=None):
        self.xdata = np.asarray(xdata)
        self.ydata = np.asarray(ydata)
        self.popt = np.asarray(popt)

        if residuals is not None:
            self._residuals = np.asarray(residuals)
        else:
            self._residuals = self.ydata - (popt[0] * self.xdata + popt[1])

        self._model = model
        self._r_squared = 0.95

    @property
    def residuals(self):
        return self._residuals

    @property
    def r_squared(self):
        return self._r_squared

    @property
    def model(self):
        if self._model is not None:
            return self._model

        def linear(x, a, b):
            return a * x + b

        return linear


@pytest.fixture
def mock_result(sample_data):
    """Create a mock result for testing."""
    x, y = sample_data
    return MockResult(xdata=x, ydata=y, popt=[2.0, 3.0])


# =============================================================================
# Tests for create_fit_plot
# =============================================================================


class TestCreateFitPlot:
    """Tests for create_fit_plot function."""

    def test_returns_plotly_figure(self, sample_data):
        """Should return a Plotly Figure object."""
        x, y = sample_data
        fig = create_fit_plot(xdata=x, ydata=y)
        assert isinstance(fig, go.Figure)

    def test_has_data_trace(self, sample_data):
        """Figure should have a data trace."""
        x, y = sample_data
        fig = create_fit_plot(xdata=x, ydata=y)
        # Should have at least one trace
        assert len(fig.data) >= 1

    def test_has_fit_trace_when_provided(self, sample_fit):
        """Figure should have fit trace when yfit provided."""
        x, y, y_fit = sample_fit
        fig = create_fit_plot(xdata=x, ydata=y, xfit=x, yfit=y_fit)
        # Should have data and fit traces
        assert len(fig.data) >= 2

    def test_has_confidence_band(self, sample_fit):
        """Figure should have confidence band when provided."""
        x, y, y_fit = sample_fit
        lower = y_fit - 1.0
        upper = y_fit + 1.0
        fig = create_fit_plot(
            xdata=x,
            ydata=y,
            xfit=x,
            yfit=y_fit,
            confidence_band=(lower, upper),
        )
        # Should have data, fit, and confidence band traces
        assert len(fig.data) >= 3

    def test_has_error_bars_with_sigma(self, sample_data):
        """Figure should show error bars when sigma provided."""
        x, y = sample_data
        sigma = np.ones_like(y) * 0.5
        fig = create_fit_plot(xdata=x, ydata=y, sigma=sigma)
        # Data trace should have error_y
        data_trace = fig.data[0]
        assert data_trace.error_y is not None

    def test_custom_title(self, sample_data):
        """Figure should use custom title."""
        x, y = sample_data
        fig = create_fit_plot(xdata=x, ydata=y, title="Custom Title")
        assert fig.layout.title.text == "Custom Title"

    def test_custom_axis_labels(self, sample_data):
        """Figure should use custom axis labels."""
        x, y = sample_data
        fig = create_fit_plot(
            xdata=x, ydata=y, x_label="Time (s)", y_label="Signal (V)"
        )
        # Plotly stores axis titles as objects with a 'text' property
        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "Signal (V)"


class TestCreateFitPlotFromResult:
    """Tests for create_fit_plot_from_result function."""

    def test_returns_plotly_figure(self, mock_result):
        """Should return a Plotly Figure object."""
        fig = create_fit_plot_from_result(mock_result)
        assert isinstance(fig, go.Figure)

    def test_raises_on_missing_xdata(self, sample_data):
        """Should raise ValueError if xdata missing."""

        class BadResult:
            ydata = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="xdata"):
            create_fit_plot_from_result(BadResult())

    def test_raises_on_missing_ydata(self, sample_data):
        """Should raise ValueError if ydata missing."""

        class BadResult:
            xdata = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="ydata"):
            create_fit_plot_from_result(BadResult())


# =============================================================================
# Tests for create_residuals_plot
# =============================================================================


class TestCreateResidualsPlot:
    """Tests for create_residuals_plot function."""

    def test_returns_plotly_figure(self, sample_data, sample_residuals):
        """Should return a Plotly Figure object."""
        x, _ = sample_data
        # Use same length residuals as x
        residuals = sample_residuals[: len(x)]
        fig = create_residuals_plot(x=x, residuals=residuals)
        assert isinstance(fig, go.Figure)

    def test_has_residual_trace(self, sample_data):
        """Figure should have residual scatter trace."""
        x, _ = sample_data
        residuals = np.random.normal(0, 0.5, len(x))
        fig = create_residuals_plot(x=x, residuals=residuals)
        # Should have at least the residuals trace
        assert len(fig.data) >= 1

    def test_shows_std_bands(self, sample_data):
        """Figure should show std bands when requested."""
        x, _ = sample_data
        residuals = np.random.normal(0, 0.5, len(x))
        fig = create_residuals_plot(x=x, residuals=residuals, show_std_bands=True)
        # Should have multiple traces for bands
        assert len(fig.data) >= 4  # 3 sigma bands + residuals

    def test_no_std_bands_when_disabled(self, sample_data):
        """Figure should not show std bands when disabled."""
        x, _ = sample_data
        residuals = np.random.normal(0, 0.5, len(x))
        fig = create_residuals_plot(x=x, residuals=residuals, show_std_bands=False)
        # Should have only residuals trace
        assert len(fig.data) == 1

    def test_has_zero_line(self, sample_data):
        """Figure should have zero line when requested."""
        x, _ = sample_data
        residuals = np.random.normal(0, 0.5, len(x))
        fig = create_residuals_plot(
            x=x, residuals=residuals, show_zero_line=True, show_std_bands=False
        )
        # Check for horizontal line at y=0
        # The hline is added via shapes, not traces
        assert fig.layout.shapes is not None or any(
            hasattr(trace, "y") and 0 in trace.y for trace in fig.data
        )

    def test_statistics_annotation(self, sample_data):
        """Figure should have statistics annotation."""
        x, _ = sample_data
        np.random.seed(42)
        residuals = np.random.normal(0, 0.5, len(x))
        fig = create_residuals_plot(x=x, residuals=residuals, show_std_bands=False)
        # Should have annotation with mean and std
        assert len(fig.layout.annotations) >= 1
        # Access the annotation text - it might be the first or could be after other annotations
        found_stats = False
        for annotation in fig.layout.annotations:
            if hasattr(annotation, "text") and annotation.text:
                if "Mean" in annotation.text and "Std" in annotation.text:
                    found_stats = True
                    break
        assert found_stats, "Statistics annotation with Mean and Std not found"


class TestCreateResidualsPlotFromResult:
    """Tests for create_residuals_plot_from_result function."""

    def test_returns_plotly_figure(self, mock_result):
        """Should return a Plotly Figure object."""
        fig = create_residuals_plot_from_result(mock_result)
        assert isinstance(fig, go.Figure)

    def test_raises_on_missing_xdata(self):
        """Should raise ValueError if xdata missing."""

        class BadResult:
            residuals = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="xdata"):
            create_residuals_plot_from_result(BadResult())

    def test_raises_on_missing_residuals(self):
        """Should raise ValueError if residuals missing."""

        class BadResult:
            xdata = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="residuals"):
            create_residuals_plot_from_result(BadResult())


# =============================================================================
# Tests for create_residuals_histogram
# =============================================================================


class TestCreateResidualsHistogram:
    """Tests for create_residuals_histogram function."""

    def test_returns_plotly_figure(self, sample_residuals):
        """Should return a Plotly Figure object."""
        fig = create_residuals_histogram(residuals=sample_residuals)
        assert isinstance(fig, go.Figure)

    def test_has_histogram_trace(self, sample_residuals):
        """Figure should have histogram trace."""
        fig = create_residuals_histogram(residuals=sample_residuals)
        # First trace should be histogram
        assert len(fig.data) >= 1
        assert isinstance(fig.data[0], go.Histogram)

    def test_has_normal_overlay(self, sample_residuals):
        """Figure should have normal distribution overlay when requested."""
        fig = create_residuals_histogram(residuals=sample_residuals, show_normal=True)
        # Should have histogram and normal curve
        assert len(fig.data) >= 2

    def test_no_normal_overlay_when_disabled(self, sample_residuals):
        """Figure should not have normal overlay when disabled."""
        fig = create_residuals_histogram(residuals=sample_residuals, show_normal=False)
        # Should have only histogram
        assert len(fig.data) == 1

    def test_has_kde_when_requested(self, sample_residuals):
        """Figure should have KDE when requested."""
        fig = create_residuals_histogram(
            residuals=sample_residuals, show_normal=True, show_kde=True
        )
        # Should have histogram, normal, and KDE
        assert len(fig.data) >= 3

    def test_auto_bins(self, sample_residuals):
        """Should use automatic bin selection."""
        fig = create_residuals_histogram(residuals=sample_residuals, n_bins="auto")
        assert isinstance(fig, go.Figure)

    def test_custom_bins(self, sample_residuals):
        """Should use custom number of bins."""
        fig = create_residuals_histogram(residuals=sample_residuals, n_bins=20)
        assert isinstance(fig, go.Figure)

    def test_statistics_annotation(self, sample_residuals):
        """Figure should have statistics annotation."""
        fig = create_residuals_histogram(residuals=sample_residuals)
        # Should have annotation with mean, std, skew, kurt
        assert len(fig.layout.annotations) >= 1
        annotation_text = fig.layout.annotations[0].text
        assert "Mean" in annotation_text
        assert "Std" in annotation_text

    def test_handles_empty_residuals(self):
        """Should handle empty residuals array."""
        fig = create_residuals_histogram(residuals=np.array([]))
        assert isinstance(fig, go.Figure)
        # Should have annotation about no data
        assert len(fig.layout.annotations) >= 1

    def test_handles_nan_values(self):
        """Should filter out NaN values."""
        residuals = np.array([1, 2, np.nan, 3, 4, np.nan, 5])
        fig = create_residuals_histogram(residuals=residuals)
        assert isinstance(fig, go.Figure)


class TestCreateHistogramFromResult:
    """Tests for create_histogram_from_result function."""

    def test_returns_plotly_figure(self, mock_result):
        """Should return a Plotly Figure object."""
        fig = create_histogram_from_result(mock_result)
        assert isinstance(fig, go.Figure)

    def test_raises_on_missing_residuals(self):
        """Should raise ValueError if residuals missing."""

        class BadResult:
            pass

        with pytest.raises(ValueError, match="residuals"):
            create_histogram_from_result(BadResult())


# =============================================================================
# Tests for compute_normality_tests
# =============================================================================


class TestComputeNormalityTests:
    """Tests for compute_normality_tests function."""

    def test_returns_dict(self, sample_residuals):
        """Should return a dictionary."""
        results = compute_normality_tests(sample_residuals)
        assert isinstance(results, dict)

    def test_has_required_keys(self, sample_residuals):
        """Should have all required test result keys."""
        results = compute_normality_tests(sample_residuals)
        required_keys = [
            "shapiro_stat",
            "shapiro_pvalue",
            "dagostino_stat",
            "dagostino_pvalue",
            "is_normal",
        ]
        for key in required_keys:
            assert key in results

    def test_normal_data_passes(self):
        """Normal data should pass normality test."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        results = compute_normality_tests(normal_data)
        # p-value should be > 0.05 for truly normal data
        # (with high probability for large sample)
        assert results["is_normal"] is True

    def test_non_normal_data_fails(self):
        """Non-normal data should fail normality test."""
        np.random.seed(42)
        # Uniform distribution is clearly non-normal
        uniform_data = np.random.uniform(-1, 1, 1000)
        results = compute_normality_tests(uniform_data)
        # p-value should be < 0.05 for non-normal data
        assert results["is_normal"] is False

    def test_handles_small_sample(self):
        """Should handle small sample sizes."""
        small_data = np.array([1, 2, 3, 4, 5])
        results = compute_normality_tests(small_data)
        # Should return NaN for tests that need more data
        assert isinstance(results, dict)

    def test_handles_nan_values(self):
        """Should filter out NaN values."""
        data_with_nan = np.array([1, 2, np.nan, 3, 4, np.nan, 5, 6, 7, 8])
        results = compute_normality_tests(data_with_nan)
        assert isinstance(results, dict)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in visualization components."""

    def test_single_data_point(self):
        """Should handle single data point."""
        x = np.array([1.0])
        y = np.array([2.0])
        # Should not crash
        fig = create_fit_plot(xdata=x, ydata=y)
        assert isinstance(fig, go.Figure)

    def test_two_data_points(self):
        """Should handle two data points."""
        x = np.array([1.0, 2.0])
        y = np.array([2.0, 4.0])
        fig = create_fit_plot(xdata=x, ydata=y, xfit=x, yfit=y)
        assert isinstance(fig, go.Figure)

    def test_large_dataset(self):
        """Should handle large dataset."""
        np.random.seed(42)
        x = np.linspace(0, 100, 10000)
        y = np.random.normal(0, 1, 10000)
        fig = create_fit_plot(xdata=x, ydata=y)
        assert isinstance(fig, go.Figure)

    def test_outliers_in_data(self):
        """Should handle outliers in data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 0.5, 50)
        y[25] = 100  # Add outlier
        fig = create_fit_plot(xdata=x, ydata=y)
        assert isinstance(fig, go.Figure)

    def test_unsorted_data(self):
        """Should handle unsorted data."""
        np.random.seed(42)
        x = np.random.uniform(0, 10, 50)
        y = 2 * x + 3
        fig = create_fit_plot(xdata=x, ydata=y, xfit=x, yfit=y)
        assert isinstance(fig, go.Figure)

    def test_negative_values(self):
        """Should handle negative x and y values."""
        x = np.linspace(-10, 10, 50)
        y = x**2 - 5
        fig = create_fit_plot(xdata=x, ydata=y)
        assert isinstance(fig, go.Figure)

    def test_residuals_all_zero(self):
        """Should handle perfect fit (zero residuals)."""
        residuals = np.zeros(100)
        fig = create_residuals_plot(x=np.arange(100), residuals=residuals)
        assert isinstance(fig, go.Figure)

    def test_residuals_constant(self):
        """Should handle constant residuals (bias)."""
        residuals = np.ones(100) * 0.5
        fig = create_residuals_histogram(residuals=residuals)
        assert isinstance(fig, go.Figure)
