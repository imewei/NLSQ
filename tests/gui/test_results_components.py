"""Tests for the NLSQ GUI results components.

This module tests the parameter results and fit statistics components
used on the Results page of the Streamlit GUI.
"""

import numpy as np
import pandas as pd
import pytest

from nlsq.gui.components.fit_statistics import (
    format_convergence_info,
    format_statistics,
    get_fit_quality_label,
)
from nlsq.gui.components.param_results import (
    compute_confidence_intervals,
    format_confidence_intervals,
    format_parameter_table,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class MockResult:
    """Mock CurveFitResult for testing."""

    def __init__(
        self,
        popt: np.ndarray,
        pcov: np.ndarray | None = None,
        r_squared: float = 0.95,
        adj_r_squared: float = 0.94,
        rmse: float = 0.5,
        mae: float = 0.4,
        aic: float = 100.0,
        bic: float = 110.0,
        success: bool = True,
        message: str = "Optimization terminated successfully",
        nfev: int = 50,
        cost: float = 1.5,
        optimality: float = 1e-8,
    ):
        self.popt = np.asarray(popt)
        self.pcov = np.asarray(pcov) if pcov is not None else None
        self._r_squared = r_squared
        self._adj_r_squared = adj_r_squared
        self._rmse = rmse
        self._mae = mae
        self._aic = aic
        self._bic = bic
        self.success = success
        self.message = message
        self.nfev = nfev
        self.cost = cost
        self.optimality = optimality

    @property
    def r_squared(self):
        return self._r_squared

    @property
    def adj_r_squared(self):
        return self._adj_r_squared

    @property
    def rmse(self):
        return self._rmse

    @property
    def mae(self):
        return self._mae

    @property
    def aic(self):
        return self._aic

    @property
    def bic(self):
        return self._bic


@pytest.fixture
def simple_result():
    """Create a simple mock result for testing."""
    popt = np.array([2.5, 0.1, 3.0])
    pcov = np.diag([0.01, 0.001, 0.05])
    return MockResult(popt=popt, pcov=pcov)


@pytest.fixture
def result_no_pcov():
    """Create a mock result without covariance."""
    popt = np.array([1.0, 2.0])
    return MockResult(popt=popt, pcov=None)


@pytest.fixture
def failed_result():
    """Create a mock result for failed optimization."""
    popt = np.array([1.0])
    return MockResult(
        popt=popt,
        success=False,
        message="Maximum iterations reached",
        r_squared=0.3,
    )


# =============================================================================
# Tests for format_parameter_table
# =============================================================================


class TestFormatParameterTable:
    """Tests for format_parameter_table function."""

    def test_returns_dataframe(self, simple_result):
        """format_parameter_table should return a DataFrame."""
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, simple_result):
        """DataFrame should have Parameter, Value, Std Error, Rel Error columns."""
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
        )
        expected_cols = ["Parameter", "Value", "Std Error", "Rel Error (%)"]
        assert list(df.columns) == expected_cols

    def test_correct_number_of_rows(self, simple_result):
        """DataFrame should have one row per parameter."""
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
        )
        assert len(df) == len(simple_result.popt)

    def test_uses_custom_param_names(self, simple_result):
        """Should use provided parameter names."""
        names = ["a", "b", "c"]
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            param_names=names,
        )
        assert list(df["Parameter"]) == names

    def test_generates_default_names(self, simple_result):
        """Should generate p0, p1, p2... if no names provided."""
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
        )
        assert list(df["Parameter"]) == ["p0", "p1", "p2"]

    def test_handles_none_pcov(self, result_no_pcov):
        """Should handle None covariance matrix."""
        df = format_parameter_table(
            popt=result_no_pcov.popt,
            pcov=None,
        )
        # Std Error should be N/A
        assert all(v == "N/A" for v in df["Std Error"])

    def test_precision_parameter(self, simple_result):
        """Should respect precision parameter."""
        df = format_parameter_table(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            precision=2,
        )
        # Values should be formatted with lower precision
        assert "2.5" in df["Value"].iloc[0]


# =============================================================================
# Tests for compute_confidence_intervals
# =============================================================================


class TestComputeConfidenceIntervals:
    """Tests for compute_confidence_intervals function."""

    def test_returns_list_of_tuples(self, simple_result):
        """Should return a list of (lower, upper) tuples."""
        ci = compute_confidence_intervals(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            n_data=100,
        )
        assert isinstance(ci, list)
        assert len(ci) == len(simple_result.popt)
        for interval in ci:
            assert isinstance(interval, tuple)
            assert len(interval) == 2

    def test_lower_less_than_upper(self, simple_result):
        """Lower bound should be less than upper bound."""
        ci = compute_confidence_intervals(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            n_data=100,
        )
        for lower, upper in ci:
            assert lower < upper

    def test_interval_contains_popt(self, simple_result):
        """Confidence interval should contain the point estimate."""
        ci = compute_confidence_intervals(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            n_data=100,
        )
        for i, (lower, upper) in enumerate(ci):
            assert lower <= simple_result.popt[i] <= upper

    def test_handles_none_pcov(self, result_no_pcov):
        """Should return infinite intervals when pcov is None."""
        ci = compute_confidence_intervals(
            popt=result_no_pcov.popt,
            pcov=None,
        )
        for lower, upper in ci:
            assert lower == -np.inf
            assert upper == np.inf

    def test_different_alpha_values(self, simple_result):
        """Smaller alpha should give wider intervals."""
        ci_95 = compute_confidence_intervals(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            n_data=100,
            alpha=0.05,
        )
        ci_99 = compute_confidence_intervals(
            popt=simple_result.popt,
            pcov=simple_result.pcov,
            n_data=100,
            alpha=0.01,
        )

        # 99% CI should be wider than 95% CI
        for i in range(len(simple_result.popt)):
            width_95 = ci_95[i][1] - ci_95[i][0]
            width_99 = ci_99[i][1] - ci_99[i][0]
            assert width_99 > width_95


class TestFormatConfidenceIntervals:
    """Tests for format_confidence_intervals function."""

    def test_returns_dataframe(self):
        """Should return a DataFrame."""
        intervals = [(1.0, 3.0), (0.05, 0.15)]
        df = format_confidence_intervals(intervals)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self):
        """Should have Parameter, Lower, Upper columns."""
        intervals = [(1.0, 3.0)]
        df = format_confidence_intervals(intervals, alpha=0.05)
        assert "Parameter" in df.columns
        assert "95% CI Lower" in df.columns
        assert "95% CI Upper" in df.columns


# =============================================================================
# Tests for format_statistics
# =============================================================================


class TestFormatStatistics:
    """Tests for format_statistics function."""

    def test_returns_dict(self, simple_result):
        """Should return a dictionary."""
        stats = format_statistics(simple_result)
        assert isinstance(stats, dict)

    def test_has_required_keys(self, simple_result):
        """Should have all required statistics keys."""
        stats = format_statistics(simple_result)
        required_keys = ["r_squared", "adj_r_squared", "rmse", "mae", "aic", "bic"]
        for key in required_keys:
            assert key in stats

    def test_values_are_strings(self, simple_result):
        """All values should be formatted strings."""
        stats = format_statistics(simple_result)
        for value in stats.values():
            assert isinstance(value, str)

    def test_formats_valid_values(self, simple_result):
        """Valid values should be formatted, not N/A."""
        stats = format_statistics(simple_result)
        assert stats["r_squared"] != "N/A"
        assert stats["rmse"] != "N/A"

    def test_handles_missing_attributes(self):
        """Should return N/A for missing attributes."""

        class IncompleteResult:
            popt = np.array([1.0])

        stats = format_statistics(IncompleteResult())
        assert stats["r_squared"] == "N/A"


# =============================================================================
# Tests for format_convergence_info
# =============================================================================


class TestFormatConvergenceInfo:
    """Tests for format_convergence_info function."""

    def test_returns_dict(self, simple_result):
        """Should return a dictionary."""
        info = format_convergence_info(simple_result)
        assert isinstance(info, dict)

    def test_has_required_keys(self, simple_result):
        """Should have all required info keys."""
        info = format_convergence_info(simple_result)
        required_keys = ["success", "success_str", "message", "nfev", "cost"]
        for key in required_keys:
            assert key in info

    def test_success_is_bool(self, simple_result):
        """success should be a boolean."""
        info = format_convergence_info(simple_result)
        assert isinstance(info["success"], bool)

    def test_success_str_is_string(self, simple_result):
        """success_str should be a string."""
        info = format_convergence_info(simple_result)
        assert isinstance(info["success_str"], str)
        assert info["success_str"] in ["Yes", "No", "Unknown"]

    def test_nfev_formatted(self, simple_result):
        """nfev_str should be formatted with commas."""
        info = format_convergence_info(simple_result)
        assert info["nfev_str"] == "50"

    def test_handles_failed_result(self, failed_result):
        """Should correctly report failed optimization."""
        info = format_convergence_info(failed_result)
        assert info["success"] is False
        assert info["success_str"] == "No"


# =============================================================================
# Tests for get_fit_quality_label
# =============================================================================


class TestGetFitQualityLabel:
    """Tests for get_fit_quality_label function."""

    def test_excellent_fit(self):
        """R^2 >= 0.99 should be Excellent."""
        label, color = get_fit_quality_label(0.995)
        assert label == "Excellent"
        assert color == "green"

    def test_very_good_fit(self):
        """R^2 >= 0.95 should be Very Good."""
        label, color = get_fit_quality_label(0.96)
        assert label == "Very Good"
        assert color == "green"

    def test_good_fit(self):
        """R^2 >= 0.90 should be Good."""
        label, color = get_fit_quality_label(0.92)
        assert label == "Good"
        assert color == "blue"

    def test_moderate_fit(self):
        """R^2 >= 0.80 should be Moderate."""
        label, color = get_fit_quality_label(0.85)
        assert label == "Moderate"
        assert color == "orange"

    def test_weak_fit(self):
        """R^2 >= 0.50 should be Weak."""
        label, color = get_fit_quality_label(0.6)
        assert label == "Weak"
        assert color == "orange"

    def test_poor_fit(self):
        """R^2 < 0.50 should be Poor."""
        label, color = get_fit_quality_label(0.3)
        assert label == "Poor"
        assert color == "red"

    def test_nan_value(self):
        """NaN should be Unknown."""
        label, color = get_fit_quality_label(np.nan)
        assert label == "Unknown"
        assert color == "gray"

    def test_negative_r_squared(self):
        """Negative R^2 should be Poor."""
        label, color = get_fit_quality_label(-0.5)
        assert label == "Poor"
        assert color == "red"


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_popt(self):
        """Should handle empty parameter array."""
        df = format_parameter_table(
            popt=np.array([]),
            pcov=None,
        )
        assert len(df) == 0

    def test_single_parameter(self):
        """Should handle single parameter."""
        popt = np.array([5.0])
        pcov = np.array([[0.1]])
        df = format_parameter_table(popt=popt, pcov=pcov)
        assert len(df) == 1

    def test_large_values(self):
        """Should handle large parameter values."""
        popt = np.array([1e10, 1e-10])
        pcov = np.diag([1e18, 1e-22])
        df = format_parameter_table(popt=popt, pcov=pcov)
        assert len(df) == 2
        # Values should be in scientific notation
        assert "e" in df["Value"].iloc[0].lower() or "1" in df["Value"].iloc[0]

    def test_zero_std_error(self):
        """Should handle zero variance (perfect fit)."""
        popt = np.array([1.0, 2.0])
        pcov = np.diag([0.0, 0.0])
        df = format_parameter_table(popt=popt, pcov=pcov)
        # Should not raise an error
        assert len(df) == 2

    def test_infinite_covariance(self):
        """Should handle infinite covariance entries."""
        popt = np.array([1.0])
        pcov = np.array([[np.inf]])
        ci = compute_confidence_intervals(popt=popt, pcov=pcov)
        # Should return infinite intervals
        assert ci[0] == (-np.inf, np.inf)

    def test_negative_diagonal_pcov(self):
        """Should handle negative diagonal in pcov (invalid case)."""
        popt = np.array([1.0])
        pcov = np.array([[-1.0]])
        # Should not crash, but return N/A or infinite interval
        df = format_parameter_table(popt=popt, pcov=pcov)
        assert len(df) == 1
