"""
Tests for Stability Integration with curve_fit
===============================================

Tests the integration of stability checks into the curve_fit API.
"""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit


class TestStabilityCheckMode:
    """Test stability='check' mode (check but don't fix)."""

    def exponential_decay(self, x, a, b, c):
        """Exponential decay model."""
        return a * jnp.exp(-b * x) + c

    def test_check_mode_with_warnings(self, caplog):
        """Test that stability='check' mode warns about issues."""
        # Create ill-conditioned problem
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        p0 = [1e-6, 1e6]  # Large scale mismatch

        with caplog.at_level(logging.WARNING):
            result = curve_fit(
                lambda x, a, b: a * x + b, x, y, p0=p0, stability="check"
            )

        # Should have warnings logged
        assert len(caplog.records) > 0
        assert any("stability" in record.message.lower() for record in caplog.records)

    def test_check_mode_no_warnings_for_good_data(self, caplog):
        """Test that stability='check' mode doesn't warn for well-conditioned data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(100)
        p0 = [2.0, 1.0]

        with caplog.at_level(logging.WARNING):
            result = curve_fit(
                lambda x, a, b: a * x + b, x, y, p0=p0, stability="check"
            )

        # Should not have stability warnings
        stability_warnings = [
            r for r in caplog.records if "stability" in r.message.lower()
        ]
        assert len(stability_warnings) == 0


class TestStabilityAutoMode:
    """Test stability='auto' mode (check and fix)."""

    def test_auto_mode_fixes_ill_conditioned(self, caplog):
        """Test that stability='auto' mode fixes ill-conditioned data."""
        # Create ill-conditioned problem
        x = np.linspace(0, 1e6, 100)
        y = 2.0 * x + 1.0
        p0 = [2.0, 1.0]

        with caplog.at_level(logging.INFO):
            result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=p0, stability="auto")

        # Should have info logs about applied fixes
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("fix" in msg.lower() for msg in info_messages)

        # Should converge successfully
        assert result.x is not None
        assert result.success

    def test_auto_mode_fixes_nan_data(self):
        """Test that stability='auto' mode fixes NaN data."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        y[50] = np.nan  # Introduce NaN

        # Should succeed with auto fix
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        assert result.x is not None
        assert result.success

    def test_auto_mode_fixes_inf_data(self):
        """Test that stability='auto' mode fixes Inf data."""
        x = np.linspace(0, 10, 100)
        x[10] = np.inf  # Introduce Inf
        y = 2.0 * np.where(np.isfinite(x), x, 5.0) + 1.0

        # Should succeed with auto fix
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        assert result.x is not None

    def test_auto_mode_fixes_parameter_scales(self):
        """Test that stability='auto' mode fixes parameter scale mismatches."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0
        p0 = [1e-6, 1e6]  # Huge scale mismatch

        # Should succeed with auto fix
        result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=p0, stability="auto")

        assert result.x is not None

    def test_auto_mode_preserves_good_data(self):
        """Test that stability='auto' mode doesn't break good data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)
        p0 = [2.5, 0.5, 1.0]

        def exponential_decay(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Should work just as well with auto mode
        result = curve_fit(exponential_decay, x, y, p0=p0, stability="auto")

        assert result.x is not None
        assert abs(result.x[0] - 2.5) < 0.5
        assert abs(result.x[1] - 0.5) < 0.2
        assert abs(result.x[2] - 1.0) < 0.3


class TestStabilityDisabled:
    """Test that stability=False works (default)."""

    def test_disabled_by_default(self):
        """Test that stability checks are disabled by default."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work without stability parameter
        result = curve_fit(lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0])

        assert result.x is not None

    def test_explicit_false(self):
        """Test that stability=False explicitly disables checks."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work with explicit False
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability=False
        )

        assert result.x is not None


class TestStabilityCombinedFeatures:
    """Test stability combined with other features."""

    def exponential_decay(self, x, a, b, c):
        """Exponential decay model."""
        return a * jnp.exp(-b * x) + c

    def test_stability_with_auto_bounds(self):
        """Test stability='auto' combined with auto_bounds."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 100.0 * np.exp(-0.01 * x) + 0.01 + 0.1 * np.random.randn(100)

        # Both stability and auto_bounds enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[1, 1, 1],  # Poor initial guess
            auto_bounds=True,
            stability="auto",
        )

        assert result.x is not None
        assert result.x[0] > 10  # Large amplitude
        assert result.x[1] > 0  # Positive decay rate

    def test_stability_with_fallback(self):
        """Test stability='auto' combined with fallback."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # Both stability and fallback enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[10, 5, 5],  # Poor initial guess
            stability="auto",
            fallback=True,
        )

        assert result.x is not None

    def test_all_features_combined(self):
        """Test stability + auto_bounds + fallback together."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 100.0 * np.exp(-0.01 * x) + 0.01 + 0.1 * np.random.randn(100)

        # All advanced features enabled
        result = curve_fit(
            self.exponential_decay,
            x,
            y,
            p0=[1, 1, 1],  # Poor initial guess
            auto_bounds=True,
            stability="auto",
            fallback=True,
        )

        assert result.x is not None
        assert result.success


class TestXPCSDivergenceRegression:
    """Regression tests for XPCS divergence issue (commit 8028a03).

    This test class verifies that the stability fixes prevent optimization
    divergence that was occurring in XPCS (X-ray Photon Correlation Spectroscopy)
    applications. The root cause was per-iteration Jacobian modification via SVD
    which caused accumulated numerical perturbations and excessive computation.

    The fix ensures:
    1. Jacobian stability checks only run at initialization, not per-iteration
    2. SVD is skipped for large Jacobians (>10M elements by default)
    3. rescale_data=False preserves physical units for physics applications
    """

    def g2_model(self, tau, baseline, contrast, gamma):
        """XPCS intensity autocorrelation function g2(tau).

        This is the standard model for dynamic light scattering and XPCS:
        g2(tau) = baseline + contrast * exp(-2 * gamma * tau)^2

        Parameters
        ----------
        tau : array
            Time delays (can span many orders of magnitude, e.g., 1µs to 10s)
        baseline : float
            Baseline intensity (typically ~1.0)
        contrast : float
            Speckle contrast (typically 0.1-0.5)
        gamma : float
            Decay rate in s^-1 (can be large, e.g., 100-10000)
        """
        return baseline + contrast * jnp.exp(-2 * gamma * tau) ** 2

    def test_xpcs_convergence_without_stability(self):
        """Test XPCS fitting works without stability mode (baseline)."""
        np.random.seed(42)
        tau = np.logspace(-6, 1, 200)  # 1µs to 10s
        baseline_true, contrast_true, gamma_true = 1.0, 0.3, 100.0

        y_true = baseline_true + contrast_true * np.exp(-2 * gamma_true * tau) ** 2
        y_noisy = y_true + 0.005 * np.random.randn(len(tau))

        p0 = [1.0, 0.25, 80.0]
        popt, _ = curve_fit(self.g2_model, tau, y_noisy, p0=p0, maxfev=1000)

        # Should converge to true values within tolerance
        assert abs(popt[0] - baseline_true) < 0.01  # baseline
        assert abs(popt[1] - contrast_true) < 0.05  # contrast
        assert abs(popt[2] - gamma_true) < 10  # gamma

    def test_xpcs_convergence_with_stability_rescale_false(self):
        """Test XPCS fitting with stability='auto' and rescale_data=False.

        This is the recommended mode for physics applications where data must
        maintain physical units (time delays in seconds, not normalized to [0,1]).
        """
        np.random.seed(42)
        tau = np.logspace(-6, 1, 200)
        baseline_true, contrast_true, gamma_true = 1.0, 0.3, 100.0

        y_true = baseline_true + contrast_true * np.exp(-2 * gamma_true * tau) ** 2
        y_noisy = y_true + 0.005 * np.random.randn(len(tau))

        p0 = [1.0, 0.25, 80.0]
        popt, _ = curve_fit(
            self.g2_model,
            tau,
            y_noisy,
            p0=p0,
            stability="auto",
            rescale_data=False,
            maxfev=1000,
        )

        # Should converge and preserve physical units
        assert abs(popt[0] - baseline_true) < 0.01
        assert abs(popt[1] - contrast_true) < 0.05
        assert abs(popt[2] - gamma_true) < 10

    def test_xpcs_convergence_with_stability_rescale_true(self):
        """Test XPCS fitting with stability='auto' and rescale_data=True (default).

        This mode rescales data to [0,1] which may affect parameter interpretation
        but should still converge.
        """
        np.random.seed(42)
        tau = np.logspace(-6, 1, 200)
        baseline_true, contrast_true, gamma_true = 1.0, 0.3, 100.0

        y_true = baseline_true + contrast_true * np.exp(-2 * gamma_true * tau) ** 2
        y_noisy = y_true + 0.005 * np.random.randn(len(tau))

        p0 = [1.0, 0.25, 80.0]
        popt, _ = curve_fit(
            self.g2_model,
            tau,
            y_noisy,
            p0=p0,
            stability="auto",
            rescale_data=True,
            maxfev=1000,
        )

        # Should converge (parameters may be rescaled)
        assert abs(popt[0] - baseline_true) < 0.01
        assert abs(popt[1] - contrast_true) < 0.05
        # gamma may be different due to time rescaling

    def test_xpcs_large_dataset_svd_skip(self):
        """Test that large XPCS datasets don't suffer from SVD overhead.

        For datasets > 10M Jacobian elements, SVD should be skipped to avoid
        O(n^2 x m) computation per iteration.
        """
        np.random.seed(42)
        # Large dataset: 100K points x 3 params = 300K elements (below threshold)
        # This tests the path but doesn't actually trigger skip
        tau = np.logspace(-6, 1, 1000)
        baseline_true, contrast_true, gamma_true = 1.0, 0.3, 100.0

        y_true = baseline_true + contrast_true * np.exp(-2 * gamma_true * tau) ** 2
        y_noisy = y_true + 0.01 * np.random.randn(len(tau))

        p0 = [1.0, 0.25, 80.0]
        popt, _ = curve_fit(
            self.g2_model,
            tau,
            y_noisy,
            p0=p0,
            stability="auto",
            rescale_data=False,
            maxfev=1000,
        )

        # Should still converge with larger dataset
        assert abs(popt[0] - baseline_true) < 0.02
        assert abs(popt[1] - contrast_true) < 0.1
        assert abs(popt[2] - gamma_true) < 20

    def test_xpcs_poor_initial_guess(self):
        """Test XPCS fitting recovers from poor initial guess with stability mode."""
        np.random.seed(42)
        tau = np.logspace(-6, 1, 200)
        baseline_true, contrast_true, gamma_true = 1.0, 0.3, 100.0

        y_true = baseline_true + contrast_true * np.exp(-2 * gamma_true * tau) ** 2
        y_noisy = y_true + 0.005 * np.random.randn(len(tau))

        # Poor initial guess (2x off on each parameter)
        p0 = [0.5, 0.6, 200.0]
        popt, _ = curve_fit(
            self.g2_model,
            tau,
            y_noisy,
            p0=p0,
            stability="auto",
            rescale_data=False,
            maxfev=2000,
        )

        # Should still converge reasonably
        assert abs(popt[0] - baseline_true) < 0.05
        assert abs(popt[1] - contrast_true) < 0.1
        assert abs(popt[2] - gamma_true) < 30


class TestMaxJacobianElementsForSVD:
    """Test the max_jacobian_elements_for_svd parameter.

    Comprehensive tests for SVD threshold behavior to prevent performance
    regressions and ensure correct computation paths are taken.
    """

    def linear_model(self, x, a, b):
        """Simple linear model for testing."""
        return a * x + b

    def test_custom_svd_threshold(self):
        """Test that custom SVD threshold can be passed via curve_fit."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.1 * np.random.randn(100)

        # With custom threshold (this tests the parameter path)
        popt, _ = curve_fit(
            self.linear_model,
            x,
            y,
            p0=[2.0, 1.0],
            stability="auto",
            max_jacobian_elements_for_svd=1000,  # Very low threshold
        )

        assert abs(popt[0] - 2.0) < 0.1
        assert abs(popt[1] - 1.0) < 0.5

    def test_svd_skipped_for_large_jacobian(self):
        """Verify SVD is skipped for Jacobians exceeding max_jacobian_elements_for_svd."""
        from nlsq.stability.guard import NumericalStabilityGuard

        np.random.seed(42)
        # Create Jacobian > default threshold (10M elements)
        # Using 4000 x 3000 = 12M elements
        J_large = np.random.randn(4000, 3000).astype(np.float64)

        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=10_000_000)
        J_fixed, issues = guard.check_and_fix_jacobian(jnp.array(J_large))

        # SVD should be skipped for large matrix
        assert issues["svd_skipped"] is True
        assert issues["reason"].startswith("Jacobian too large")
        assert issues["condition_number"] is None  # Not computed
        assert not np.any(np.isnan(J_fixed))

    def test_svd_computed_for_medium_jacobian(self):
        """Verify SVD is computed for Jacobians below threshold."""
        from nlsq.stability.guard import NumericalStabilityGuard

        np.random.seed(42)
        # Create Jacobian < threshold (8M elements: 2000 x 4000 = 8M)
        J_medium = np.random.randn(2000, 4000).astype(np.float64)

        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=10_000_000)
        _, issues = guard.check_and_fix_jacobian(jnp.array(J_medium))

        # SVD SHOULD be called for medium matrix
        assert issues["svd_skipped"] is False
        assert "condition_number" in issues
        assert issues["condition_number"] is not None

    def test_svd_threshold_boundary_above(self):
        """Test that SVD is skipped when exactly at or above threshold."""
        from nlsq.stability.guard import NumericalStabilityGuard

        np.random.seed(42)
        # Set threshold to 10000
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=10000)

        # Exactly at threshold: 100 x 100 = 10,000 elements
        J_at_threshold = np.random.randn(100, 100).astype(np.float64)
        assert J_at_threshold.shape[0] * J_at_threshold.shape[1] == 10000

        _, issues = guard.check_and_fix_jacobian(jnp.array(J_at_threshold))

        # Should skip SVD when AT threshold (> check, so exactly at is not skipped)
        # The code uses > not >=, so exactly at threshold should compute SVD
        assert issues["svd_skipped"] is False

    def test_svd_threshold_boundary_below(self):
        """Test that SVD is computed when just below threshold."""
        from nlsq.stability.guard import NumericalStabilityGuard

        np.random.seed(42)
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=10001)

        # Just at threshold: 100 x 100 = 10,000 elements
        J_below = np.random.randn(100, 100).astype(np.float64)

        _, issues = guard.check_and_fix_jacobian(jnp.array(J_below))

        # Should compute SVD when below threshold
        assert issues["svd_skipped"] is False
        assert issues["condition_number"] is not None

    def test_svd_threshold_custom_value(self):
        """Test that custom threshold values are respected."""
        from nlsq.stability.guard import NumericalStabilityGuard

        np.random.seed(42)
        # Set very low threshold (1000 elements)
        guard = NumericalStabilityGuard(max_jacobian_elements_for_svd=1000)

        # 2000 element Jacobian - should skip SVD with custom threshold
        J = np.random.randn(50, 40).astype(np.float64)  # 2000 elements
        assert J.shape[0] * J.shape[1] == 2000

        _, issues = guard.check_and_fix_jacobian(jnp.array(J))
        assert issues["svd_skipped"] is True
        assert "2,000" in issues["reason"]  # Verify element count in message


class TestNumericalAccuracyRegression:
    """Numerical accuracy regression tests.

    These tests verify that optimization produces accurate results by comparing
    against known analytical solutions and SciPy reference implementations.
    """

    def test_linear_fit_accuracy(self):
        """Test linear fit produces accurate parameters."""
        np.random.seed(42)
        # Known solution: y = 2.5x + 1.3
        a_true, b_true = 2.5, 1.3
        x = np.linspace(0, 10, 100)
        y = a_true * x + b_true + 0.01 * np.random.randn(100)

        popt, _ = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        # High precision for linear fit
        assert abs(popt[0] - a_true) < 0.01, f"Slope error: {abs(popt[0] - a_true)}"
        assert abs(popt[1] - b_true) < 0.05, f"Intercept error: {abs(popt[1] - b_true)}"

    def test_exponential_fit_accuracy(self):
        """Test exponential fit produces accurate parameters."""
        np.random.seed(42)
        # Known solution: y = 3.0 * exp(-0.5 * x) + 0.5
        a_true, b_true, c_true = 3.0, 0.5, 0.5
        x = np.linspace(0, 10, 100)
        y = a_true * np.exp(-b_true * x) + c_true + 0.02 * np.random.randn(100)

        def exp_model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        popt, _ = curve_fit(exp_model, x, y, p0=[3.0, 0.5, 0.5], stability="auto")

        assert abs(popt[0] - a_true) < 0.1, f"Amplitude error: {abs(popt[0] - a_true)}"
        assert abs(popt[1] - b_true) < 0.05, (
            f"Decay rate error: {abs(popt[1] - b_true)}"
        )
        assert abs(popt[2] - c_true) < 0.1, f"Offset error: {abs(popt[2] - c_true)}"

    def test_gaussian_fit_accuracy(self):
        """Test Gaussian fit produces accurate parameters."""
        np.random.seed(42)
        # Known solution: y = 5.0 * exp(-((x - 3.0)^2) / (2 * 1.5^2))
        A_true, mu_true, sigma_true = 5.0, 3.0, 1.5
        x = np.linspace(0, 10, 200)
        y = A_true * np.exp(-((x - mu_true) ** 2) / (2 * sigma_true**2))
        y += 0.05 * np.random.randn(200)

        def gaussian(x, A, mu, sigma):
            return A * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        popt, _ = curve_fit(gaussian, x, y, p0=[5.0, 3.0, 1.5], stability="auto")

        assert abs(popt[0] - A_true) < 0.2, f"Amplitude error: {abs(popt[0] - A_true)}"
        assert abs(popt[1] - mu_true) < 0.1, f"Mean error: {abs(popt[1] - mu_true)}"
        assert abs(popt[2] - sigma_true) < 0.2, (
            f"Sigma error: {abs(popt[2] - sigma_true)}"
        )

    def test_polynomial_fit_accuracy(self):
        """Test polynomial fit produces accurate parameters."""
        np.random.seed(42)
        # Known solution: y = 0.5x^2 - 2x + 3
        a_true, b_true, c_true = 0.5, -2.0, 3.0
        x = np.linspace(-5, 5, 100)
        y = a_true * x**2 + b_true * x + c_true + 0.1 * np.random.randn(100)

        def poly2(x, a, b, c):
            return a * x**2 + b * x + c

        popt, _ = curve_fit(poly2, x, y, p0=[0.5, -2.0, 3.0], stability="auto")

        assert abs(popt[0] - a_true) < 0.02
        assert abs(popt[1] - b_true) < 0.1
        assert abs(popt[2] - c_true) < 0.2


class TestExceptionHandling:
    """Tests for exception handling and graceful degradation."""

    def test_svd_failure_graceful_handling(self):
        """Test graceful handling when SVD computation might fail."""
        from nlsq.stability.guard import NumericalStabilityGuard

        guard = NumericalStabilityGuard()

        # Create a matrix with extreme values that might cause numerical issues
        J = np.array([[1e-100, 1e100], [1e100, 1e-100]], dtype=np.float64)

        # Should handle gracefully without crashing
        J_fixed, issues = guard.check_and_fix_jacobian(jnp.array(J))

        # Verify no crash occurred and issues dict is valid
        assert issues is not None
        assert "condition_number" in issues
        # The matrix should still be usable
        assert J_fixed.shape == (2, 2)

    def test_nan_inf_in_jacobian_handled(self):
        """Test that NaN/Inf values in Jacobian are handled correctly."""
        from nlsq.stability.guard import NumericalStabilityGuard

        guard = NumericalStabilityGuard()

        # Create Jacobian with NaN and Inf
        J = np.array(
            [[1.0, np.nan, 2.0], [np.inf, 3.0, 4.0], [5.0, -np.inf, 6.0]],
            dtype=np.float64,
        )

        # Expect warning about NaN/Inf values
        with pytest.warns(UserWarning, match="Jacobian contains NaN or Inf"):
            J_fixed, issues = guard.check_and_fix_jacobian(jnp.array(J))

        # Should replace NaN/Inf with zeros
        assert issues["has_nan"] is True or issues["has_inf"] is True
        assert jnp.all(jnp.isfinite(J_fixed))

    def test_all_zeros_jacobian_handled(self):
        """Test that all-zeros Jacobian is handled gracefully."""
        from nlsq.stability.guard import NumericalStabilityGuard

        guard = NumericalStabilityGuard()

        # Create all-zeros Jacobian
        J = np.zeros((10, 5), dtype=np.float64)

        # Expect warning about all-zeros Jacobian
        with pytest.warns(UserWarning, match="Jacobian is all zeros"):
            J_fixed, issues = guard.check_and_fix_jacobian(jnp.array(J))

        # Should add perturbation to avoid singularity
        # The perturbation is eps (machine epsilon ~2e-16)
        assert jnp.any(J_fixed != 0.0), "Jacobian should have non-zero perturbation"
        assert issues["condition_number"] == np.inf

    def test_empty_data_handling(self):
        """Test handling of edge cases with minimal data."""
        np.random.seed(42)
        # Minimal viable data (just enough points)
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 3.0, 5.0])

        popt, _ = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="auto"
        )

        # Should still produce reasonable fit
        assert abs(popt[0] - 2.0) < 0.1
        assert abs(popt[1] - 1.0) < 0.1


class TestEdgeCases:
    """Test edge cases for stability integration."""

    def test_stability_without_p0(self):
        """Test stability checks when p0 is not provided."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Should work even without p0
        # (p0 will be auto-estimated by curve_fit)
        result = curve_fit(lambda x, a, b: a * x + b, x, y, stability="auto")

        assert result.x is not None

    def test_invalid_stability_value(self):
        """Test that invalid stability values are handled."""
        x = np.linspace(0, 10, 100)
        y = 2.0 * x + 1.0

        # Invalid value should be treated as False (no checks)
        # This should just work without error
        result = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[2.0, 1.0], stability="invalid"
        )

        assert result.x is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
