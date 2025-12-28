"""
Tests for Numerical Stability Pre-flight Checks
================================================

Tests the pre-flight stability check system for detecting and fixing
numerical issues before optimization.
"""

import numpy as np
import pytest

from nlsq.stability.guard import (
    apply_automatic_fixes,
    check_problem_stability,
    detect_collinearity,
    detect_parameter_scale_mismatch,
    estimate_condition_number,
)


class TestEstimateConditionNumber:
    """Test condition number estimation."""

    def test_well_conditioned_1d(self):
        """Test with well-conditioned 1D data."""
        x = np.linspace(0, 10, 100)
        cond = estimate_condition_number(x)

        # Should be reasonable
        assert cond < 1e6

    def test_ill_conditioned_1d(self):
        """Test with ill-conditioned 1D data (large range)."""
        x = np.linspace(0, 1e6, 100)
        cond = estimate_condition_number(x)

        # Should be high
        assert cond > 1e10

    def test_well_conditioned_2d(self):
        """Test with well-conditioned 2D data."""
        x = np.random.randn(100, 3)
        cond = estimate_condition_number(x)

        # Should be reasonable for random data
        assert cond < 1e8

    def test_constant_data(self):
        """Test with constant data."""
        x = np.ones(100) * 5.0
        cond = estimate_condition_number(x)

        # Should be infinite (rank deficient)
        assert np.isinf(cond) or cond > 1e15


class TestDetectParameterScaleMismatch:
    """Test parameter scale mismatch detection."""

    def test_no_mismatch(self):
        """Test with similar scale parameters."""
        p0 = np.array([1.0, 2.0, 0.5])
        has_mismatch, ratio = detect_parameter_scale_mismatch(p0)

        assert has_mismatch is False
        assert ratio < 10

    def test_mismatch_detected(self):
        """Test with large scale mismatch."""
        p0 = np.array([1e-6, 1.0, 1e6])
        has_mismatch, ratio = detect_parameter_scale_mismatch(p0)

        assert has_mismatch is True
        assert ratio > 1e6

    def test_zero_parameters(self):
        """Test with zero parameters."""
        p0 = np.array([0.0, 0.0, 0.0])
        has_mismatch, ratio = detect_parameter_scale_mismatch(p0)

        assert has_mismatch is False
        assert ratio == 1.0

    def test_mixed_zero_nonzero(self):
        """Test with mixed zero and non-zero parameters."""
        p0 = np.array([0.0, 1.0, 1000.0])
        _has_mismatch, ratio = detect_parameter_scale_mismatch(p0)

        # Zeros are ignored, ratio is between nonzero values
        assert ratio == 1000.0

    def test_custom_threshold(self):
        """Test with custom threshold."""
        p0 = np.array([1.0, 100.0])
        has_mismatch, _ratio = detect_parameter_scale_mismatch(p0, threshold=50.0)

        assert has_mismatch is True


class TestDetectCollinearity:
    """Test collinearity detection."""

    def test_no_collinearity(self):
        """Test with uncorrelated variables."""
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = np.random.randn(100)
        xdata = np.column_stack([x1, x2])

        has_coll, pairs = detect_collinearity(xdata)

        assert has_coll is False
        assert len(pairs) == 0

    def test_perfect_collinearity(self):
        """Test with perfectly collinear variables."""
        x1 = np.linspace(0, 10, 100)
        x2 = 2.0 * x1  # Perfect linear relationship
        xdata = np.column_stack([x1, x2])

        has_coll, pairs = detect_collinearity(xdata)

        assert has_coll is True
        assert len(pairs) > 0
        assert pairs[0][2] > 0.95  # High correlation

    def test_near_collinearity(self):
        """Test with near-collinear variables."""
        np.random.seed(42)
        x1 = np.linspace(0, 10, 100)
        x2 = 2.0 * x1 + 0.1 * np.random.randn(100)  # Nearly collinear
        xdata = np.column_stack([x1, x2])

        has_coll, _pairs = detect_collinearity(xdata)

        assert has_coll is True

    def test_1d_data(self):
        """Test with 1D data (should return False)."""
        xdata = np.linspace(0, 10, 100)

        has_coll, pairs = detect_collinearity(xdata)

        assert has_coll is False
        assert len(pairs) == 0


class TestCheckProblemStability:
    """Test comprehensive problem stability checking."""

    def test_well_conditioned_problem(self):
        """Test with well-conditioned problem."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(100)
        p0 = np.array([2.0, 1.0])

        report = check_problem_stability(x, y, p0)

        assert report["severity"] == "ok"
        assert len(report["issues"]) == 0

    def test_ill_conditioned_data(self):
        """Test with ill-conditioned data."""
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        p0 = np.array([2.0, 1.0])

        report = check_problem_stability(x, y, p0)

        assert report["severity"] in ["warning", "critical"]
        assert len(report["issues"]) > 0
        assert report["condition_number"] > 1e10

    def test_nan_data(self):
        """Test with NaN data."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        y[50] = np.nan

        report = check_problem_stability(x, y)

        assert report["severity"] == "critical"
        assert any("NaN" in msg for _, msg, _ in report["issues"])

    def test_inf_data(self):
        """Test with Inf data."""
        x = np.linspace(0, 10, 100)
        x[10] = np.inf
        y = 2.0 * x + 1.0

        report = check_problem_stability(x, y)

        assert report["severity"] == "critical"
        assert any("Inf" in msg for _, msg, _ in report["issues"])

    def test_constant_data(self):
        """Test with constant data."""
        x = np.linspace(0, 10, 100)
        y = np.ones(100) * 5.0

        report = check_problem_stability(x, y)

        assert report["severity"] == "warning"
        assert any("zero range" in msg for _, msg, _ in report["issues"])

    def test_parameter_scale_mismatch(self):
        """Test detection of parameter scale mismatch."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        p0 = np.array([1e-6, 1e6])  # Huge mismatch

        report = check_problem_stability(x, y, p0)

        assert report["parameter_scale_ratio"] > 1e6
        assert any("scale" in msg.lower() for _, msg, _ in report["issues"])

    def test_large_data_range(self):
        """Test with large data range."""
        x = np.linspace(0, 1e7, 100)
        y = 2.0 * x + 1.0

        report = check_problem_stability(x, y)

        assert any("large range" in msg for _, msg, _ in report["issues"])

    def test_recommendations_provided(self):
        """Test that recommendations are provided for issues."""
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0

        report = check_problem_stability(x, y)

        assert len(report["recommendations"]) > 0


class TestApplyAutomaticFixes:
    """Test automatic fixes for stability issues."""

    def test_rescale_ill_conditioned(self):
        """Test automatic rescaling of ill-conditioned data."""
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0

        x_fixed, _y_fixed, _, info = apply_automatic_fixes(x, y)

        # xdata should be rescaled
        assert np.ptp(x_fixed) == pytest.approx(1.0, abs=1e-10)
        assert np.min(x_fixed) == pytest.approx(0.0, abs=1e-10)
        assert len(info["applied_fixes"]) > 0

    def test_rescale_large_range(self):
        """Test rescaling of large range data."""
        x = np.linspace(0, 1e5, 100)
        y = np.linspace(0, 1e5, 100)

        x_fixed, y_fixed, _, _info = apply_automatic_fixes(x, y)

        # Both should be rescaled
        assert np.ptp(x_fixed) <= 1.0
        assert np.ptp(y_fixed) <= 1.0

    def test_fix_nan_data(self):
        """Test fixing NaN data."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        y[50] = np.nan

        _x_fixed, y_fixed, _, info = apply_automatic_fixes(x, y)

        # NaN should be fixed
        assert not np.any(np.isnan(y_fixed))
        assert any("NaN" in fix for fix in info["applied_fixes"])

    def test_fix_inf_data(self):
        """Test fixing Inf data."""
        x = np.linspace(0, 10, 100)
        x[10] = np.inf
        y = 2.0 * x + 1.0

        x_fixed, _y_fixed, _, info = apply_automatic_fixes(x, y)

        # Inf should be fixed
        assert not np.any(np.isinf(x_fixed))
        assert any("NaN/Inf" in fix or "Inf" in fix for fix in info["applied_fixes"])

    def test_fix_parameter_scales(self):
        """Test fixing parameter scale mismatch."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        p0 = np.array([1e-6, 1e6])

        _x_fixed, _y_fixed, p0_fixed, _info = apply_automatic_fixes(x, y, p0)

        # Parameters should be rescaled to similar magnitude
        if p0_fixed is not None:
            scale_ratio = np.max(np.abs(p0_fixed)) / np.min(
                np.abs(p0_fixed[p0_fixed != 0])
            )
            # After fix, ratio should be much smaller
            assert scale_ratio < 1e6

    def test_preserve_good_data(self):
        """Test that good data is not unnecessarily modified."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        p0 = np.array([2.0, 1.0])

        _x_fixed, _y_fixed, _p0_fixed, info = apply_automatic_fixes(x, y, p0)

        # Should not apply fixes to already good data
        # (though minor rescaling might still occur)
        assert len(info["applied_fixes"]) <= 1  # At most minor adjustments

    def test_with_stability_report(self):
        """Test using pre-computed stability report."""
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0

        report = check_problem_stability(x, y)
        _x_fixed, _y_fixed, _, info = apply_automatic_fixes(
            x, y, stability_report=report
        )

        # Should use the provided report
        assert len(info["applied_fixes"]) > 0

    def test_scaling_info_returned(self):
        """Test that scaling information is returned."""
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0

        _x_fixed, _y_fixed, _, info = apply_automatic_fixes(x, y)

        # Should return scaling factors
        assert "x_scale" in info
        assert "y_scale" in info
        assert "x_offset" in info
        assert "y_offset" in info
        assert info["x_scale"] > 1.0  # Data was rescaled


class TestIntegration:
    """Integration tests for stability system."""

    def test_check_then_fix_workflow(self):
        """Test typical workflow: check then fix."""
        # Problem with multiple issues
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        y[50] = np.nan
        p0 = np.array([1e-6, 1e6])

        # Step 1: Check
        report = check_problem_stability(x, y, p0)
        assert report["severity"] == "critical"

        # Step 2: Fix
        x_fixed, y_fixed, p0_fixed, _info = apply_automatic_fixes(
            x, y, p0, stability_report=report
        )

        # Step 3: Verify fixes
        report2 = check_problem_stability(x_fixed, y_fixed, p0_fixed)
        assert report2["severity"] in ["ok", "warning"]  # Should be better
        assert report2["condition_number"] < report["condition_number"]

    def test_multidimensional_collinearity(self):
        """Test collinearity detection with 2D data."""
        np.random.seed(42)
        x1 = np.linspace(0, 10, 100)
        x2 = 2.0 * x1 + 0.01 * np.random.randn(100)
        xdata = np.column_stack([x1, x2])
        y = 3.0 * x1 + 4.0 * x2 + 1.0

        report = check_problem_stability(xdata, y)

        assert report["has_collinearity"] is True
        assert len(report["collinear_pairs"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
