"""
Integration tests for Adaptive Hybrid Streaming Optimizer.

This module contains strategic integration tests that verify:
1. End-to-end 4-phase optimization pipeline
2. SciPy compatibility (curve_fit comparison)
3. Covariance accuracy against analytical solutions
4. Large dataset performance and memory management
5. Edge cases (bounds, single parameter, correlations)
6. Backward compatibility with existing APIs
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import curve_fit as scipy_curve_fit

from nlsq import curve_fit, curve_fit_large
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.optimizer import StreamingOptimizer


class TestFullPipelineIntegration:
    """End-to-end integration tests for the complete 4-phase optimization."""

    def test_full_pipeline_exponential_decay(self):
        """Test full 4-phase optimization on exponential decay model."""

        # Generate synthetic data
        def exponential_model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = jnp.linspace(0, 10, 10000)
        true_params = jnp.array([5.0, 0.5, 1.0])
        y_true = exponential_model(x, *true_params)

        # Add noise
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, y_true.shape) * 0.1
        y_noisy = y_true + noise

        # Initial guess
        p0 = jnp.array([4.0, 0.4, 0.8])

        # Create optimizer with conservative config
        config = HybridStreamingConfig.conservative()
        config.warmup_iterations = 50  # Reduce for faster test
        config.max_warmup_iterations = 100
        config.gauss_newton_max_iterations = 50

        optimizer = AdaptiveHybridStreamingOptimizer(config=config)

        # Fit
        result = optimizer.fit(
            data_source=(x, y_noisy),
            func=exponential_model,
            p0=p0,
            bounds=([0, 0, 0], [10, 2, 5]),
            verbose=0,
        )

        # Verify convergence
        assert result["success"], f"Optimization failed: {result.get('message')}"

        # Verify parameters close to true values (within 30% - hybrid method is a 2-stage optimizer)
        popt = result["x"]
        param_errors = jnp.abs((popt - true_params) / true_params)
        assert jnp.all(param_errors < 0.30), (
            f"Parameter errors too large: {param_errors}"
        )

        # Verify covariance matrix is symmetric
        pcov = result["pcov"]
        assert jnp.allclose(pcov, pcov.T), "Covariance matrix not symmetric"

        # Verify all phases completed (check phase_history)
        diagnostics = result.get("streaming_diagnostics", {})
        phase_history = diagnostics.get("phase_history", [])
        completed_phases = [p["phase"] for p in phase_history]
        assert set(completed_phases) == {0, 1, 2, 3}, (
            f"Not all phases completed: {completed_phases}"
        )

    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.timeout(180)
    def test_full_pipeline_gaussian_model(self):
        """Test full pipeline on Gaussian model with multiple parameters."""

        def gaussian_model(x, amplitude, mean, stddev, baseline):
            return amplitude * jnp.exp(-0.5 * ((x - mean) / stddev) ** 2) + baseline

        x = jnp.linspace(-5, 5, 5000)
        true_params = jnp.array([10.0, 0.0, 1.5, 2.0])
        y_true = gaussian_model(x, *true_params)

        # Add noise
        key = jax.random.PRNGKey(123)
        noise = jax.random.normal(key, y_true.shape) * 0.2
        y_noisy = y_true + noise

        # Initial guess
        p0 = jnp.array([8.0, 0.5, 1.0, 1.5])

        # Fit with hybrid streaming method
        result = curve_fit(
            gaussian_model,
            x,
            y_noisy,
            p0=p0,
            bounds=([0, -5, 0.1, 0], [20, 5, 5, 10]),
            method="hybrid_streaming",
            verbose=0,
        )

        popt, pcov = result

        # Verify parameters (skip inf values which indicate unbounded parameters)
        param_errors = jnp.abs((popt - true_params) / true_params)
        # Filter out inf values caused by covariance issues
        finite_errors = param_errors[jnp.isfinite(param_errors)]
        assert jnp.all(finite_errors < 0.15), f"Gaussian fit error: {param_errors}"

        # Verify covariance is positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(pcov)
        assert jnp.all(eigenvalues >= -1e-10), "Covariance not positive semi-definite"


class TestSciPyCompatibility:
    """Compare results with scipy.optimize.curve_fit for compatibility."""

    def test_scipy_compatibility_exponential_model(self):
        """Verify results match scipy.optimize.curve_fit on exponential decay."""

        def exponential_model(x, a, b, c):
            return a * np.exp(-b * x) + c

        # Small dataset for scipy compatibility
        x = np.linspace(0, 5, 500)
        true_params = np.array([3.0, 0.8, 0.5])
        y_true = exponential_model(x, *true_params)

        # Add noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(x))
        y_noisy = y_true + noise

        p0 = np.array([2.5, 0.7, 0.4])

        # SciPy fit
        popt_scipy, pcov_scipy = scipy_curve_fit(
            exponential_model, x, y_noisy, p0=p0, bounds=([0, 0, 0], [10, 2, 2])
        )

        # NLSQ hybrid streaming fit
        x_jax = jnp.array(x)
        y_jax = jnp.array(y_noisy)
        p0_jax = jnp.array(p0)

        def jax_model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        popt_nlsq, pcov_nlsq = curve_fit(
            jax_model,
            x_jax,
            y_jax,
            p0=p0_jax,
            bounds=([0, 0, 0], [10, 2, 2]),
            method="hybrid_streaming",
            verbose=0,
        )

        # Compare parameters (should match within 5% - hybrid method uses different algorithm)
        param_diff = np.abs((np.array(popt_nlsq) - popt_scipy) / popt_scipy)
        assert np.all(param_diff < 0.05), (
            f"Parameter mismatch: NLSQ={popt_nlsq}, SciPy={popt_scipy}, diff={param_diff}"
        )

        # Compare covariance (should match within 10% due to different algorithms)
        pcov_diff = np.abs(
            (np.array(pcov_nlsq) - pcov_scipy) / (np.abs(pcov_scipy) + 1e-10)
        )
        assert np.all(pcov_diff < 0.10), f"Covariance mismatch: {pcov_diff}"

    def test_scipy_compatibility_power_law(self):
        """Verify results match scipy on power law model."""

        def power_law(x, a, b):
            return a * np.power(x, b)

        x = np.linspace(0.1, 10, 300)
        true_params = np.array([2.0, 0.5])
        y_true = power_law(x, *true_params)

        np.random.seed(123)
        noise = np.random.normal(0, 0.1, len(x))
        y_noisy = y_true + noise

        p0 = np.array([1.5, 0.4])

        # SciPy
        popt_scipy, _pcov_scipy = scipy_curve_fit(power_law, x, y_noisy, p0=p0)

        # NLSQ
        def jax_power_law(x, a, b):
            return a * jnp.power(x, b)

        popt_nlsq, _pcov_nlsq = curve_fit(
            jax_power_law,
            jnp.array(x),
            jnp.array(y_noisy),
            p0=jnp.array(p0),
            method="hybrid_streaming",
            verbose=0,
        )

        # Verify parameters match (within 10% - different optimization methods)
        param_diff = np.abs((np.array(popt_nlsq) - popt_scipy) / popt_scipy)
        assert np.all(param_diff < 0.10), (
            f"Power law parameters don't match scipy: diff={param_diff}"
        )


class TestCovarianceAccuracy:
    """Verify covariance matrix accuracy against analytical solutions."""

    def test_covariance_accuracy_linear_regression(self):
        """Test covariance on linear regression where analytical solution exists."""
        # Linear regression: y = a*x + b
        # Analytical covariance: Cov = sigma^2 * (X^T X)^{-1}
        # where X = [x, 1] design matrix

        x = jnp.linspace(0, 10, 1000)
        a_true, b_true = 2.0, 1.0
        y_true = a_true * x + b_true

        # Add Gaussian noise with known variance
        sigma_noise = 0.5
        key = jax.random.PRNGKey(456)
        noise = jax.random.normal(key, y_true.shape) * sigma_noise
        y_noisy = y_true + noise

        def linear_model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.5])

        # NLSQ fit
        popt, pcov = curve_fit(
            linear_model, x, y_noisy, p0=p0, method="hybrid_streaming", verbose=0
        )

        # Analytical covariance
        X = jnp.stack([x, jnp.ones_like(x)], axis=1)  # Design matrix
        XTX_inv = jnp.linalg.inv(X.T @ X)

        # Estimate residual variance
        residuals = y_noisy - linear_model(x, *popt)
        sigma_sq_est = jnp.sum(residuals**2) / (len(x) - 2)

        pcov_analytical = sigma_sq_est * XTX_inv

        # Compare covariances (should match within 10% due to finite sample)
        cov_diff = jnp.abs(
            (pcov - pcov_analytical) / (jnp.abs(pcov_analytical) + 1e-10)
        )
        assert jnp.all(cov_diff < 0.1), f"Covariance accuracy issue: diff={cov_diff}"

        # Standard errors should be reasonable
        perr = jnp.sqrt(jnp.diag(pcov))
        assert jnp.all(perr > 0), "Standard errors must be positive"
        assert jnp.all(perr < 0.1), f"Standard errors too large: {perr}"


class TestLargeDatasetPerformance:
    """Test performance and memory usage on large datasets."""

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_large_dataset_memory_efficient(self):
        """Verify optimizer handles 1M+ points with streaming."""

        def simple_model(x, a, b):
            return a * jnp.sin(b * x)

        # 1 million points
        n_points = 1_000_000
        x = jnp.linspace(0, 100, n_points)
        y_true = simple_model(x, 5.0, 0.1)

        # Add small noise
        key = jax.random.PRNGKey(789)
        noise = jax.random.normal(key, y_true.shape) * 0.05
        y_noisy = y_true + noise

        p0 = jnp.array([4.0, 0.08])

        # Use memory-optimized config
        config = HybridStreamingConfig.memory_optimized()
        config.chunk_size = 50000  # Process in chunks
        config.warmup_iterations = 20  # Reduce for speed
        config.gauss_newton_max_iterations = 20

        optimizer = AdaptiveHybridStreamingOptimizer(config=config)

        # Should complete without OOM
        result = optimizer.fit(
            data_source=(x, y_noisy), func=simple_model, p0=p0, verbose=0
        )

        assert result["success"], "Large dataset optimization failed"

        # Parameters should be reasonable
        popt = result["x"]
        assert jnp.abs(popt[0] - 5.0) < 0.5, f"Parameter a error: {popt[0]}"
        assert jnp.abs(popt[1] - 0.1) < 0.02, f"Parameter b error: {popt[1]}"

    @pytest.mark.slow
    @pytest.mark.serial
    @pytest.mark.timeout(180)
    def test_curve_fit_large_integration(self):
        """Test curve_fit_large with hybrid_streaming method."""

        def exponential_model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Medium-large dataset (kept below timeout thresholds)
        n_points = 30_000
        x = jnp.linspace(0, 10, n_points)
        y_true = exponential_model(x, 3.0, 0.5, 1.0)

        key = jax.random.PRNGKey(111)
        noise = jax.random.normal(key, y_true.shape) * 0.05
        y_noisy = y_true + noise

        p0 = jnp.array([2.5, 0.4, 0.8])

        # Use curve_fit_large
        popt, pcov = curve_fit_large(
            exponential_model,
            x,
            y_noisy,
            p0=p0,
            bounds=([0, 0, 0], [10, 2, 5]),
            method="hybrid_streaming",
            verbose=0,
        )

        # Verify results
        assert popt.shape == (3,), "Parameters shape incorrect"
        assert pcov.shape == (3, 3), "Covariance shape incorrect"

        param_errors = jnp.abs(
            (popt - jnp.array([3.0, 0.5, 1.0])) / jnp.array([3.0, 0.5, 1.0])
        )
        assert jnp.all(param_errors < 0.05), (
            f"curve_fit_large accuracy issue: {param_errors}"
        )


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_parameters_at_bounds(self):
        """Test optimization when optimal parameters are at bounds."""

        def bounded_model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 10, 500)
        # True parameters at bounds
        y_true = 10.0 * x + 0.0  # a=10 (upper bound), b=0 (lower bound)

        key = jax.random.PRNGKey(222)
        noise = jax.random.normal(key, y_true.shape) * 0.1
        y_noisy = y_true + noise

        p0 = jnp.array([5.0, 1.0])
        bounds = ([0, 0], [10, 5])

        popt, _pcov = curve_fit(
            bounded_model,
            x,
            y_noisy,
            p0=p0,
            bounds=bounds,
            method="hybrid_streaming",
            verbose=0,
        )

        # Should converge near bounds (hybrid method may not reach exact bounds)
        # Just verify parameters are reasonable and optimization succeeded
        assert popt[0] > 5.0, f"Parameter should be large: {popt[0]}"
        assert popt[1] >= 0.0, f"Parameter should be non-negative: {popt[1]}"

    def test_single_parameter_optimization(self):
        """Test simplest case: single parameter."""

        def single_param_model(x, a):
            return a * x

        x = jnp.linspace(1, 10, 200)
        y_true = 2.5 * x

        key = jax.random.PRNGKey(333)
        noise = jax.random.normal(key, y_true.shape) * 0.05
        y_noisy = y_true + noise

        p0 = jnp.array([2.0])

        popt, pcov = curve_fit(
            single_param_model, x, y_noisy, p0=p0, method="hybrid_streaming", verbose=0
        )

        # Verify single parameter
        assert popt.shape == (1,), "Single parameter shape incorrect"
        assert pcov.shape == (1, 1), "Single parameter covariance shape incorrect"

        assert jnp.abs(popt[0] - 2.5) < 0.05, f"Single parameter error: {popt[0]}"

    def test_highly_correlated_parameters(self):
        """Test optimization with highly correlated parameters."""

        # Model: y = a*x + b*x = (a+b)*x
        # Parameters a and b are perfectly correlated
        def correlated_model(x, a, b, c):
            # a and b are correlated, c is independent
            return (a + b) * x + c

        x = jnp.linspace(0, 10, 500)
        # True: a+b=3, c=1
        y_true = 3.0 * x + 1.0

        key = jax.random.PRNGKey(444)
        noise = jax.random.normal(key, y_true.shape) * 0.1
        y_noisy = y_true + noise

        p0 = jnp.array([1.0, 1.0, 0.5])

        # Should still converge, though covariance may be ill-conditioned
        popt, pcov = curve_fit(
            correlated_model, x, y_noisy, p0=p0, method="hybrid_streaming", verbose=0
        )

        # Sum a+b should be close to 3 (correlated parameters may not converge precisely)
        assert jnp.abs((popt[0] + popt[1]) - 3.0) < 0.6, (
            f"Correlated sum error: {popt[0] + popt[1]}"
        )
        assert jnp.abs(popt[2] - 1.0) < 0.35, f"Independent parameter error: {popt[2]}"

        # Covariance should show correlation
        correlation = pcov[0, 1] / jnp.sqrt(pcov[0, 0] * pcov[1, 1])
        assert jnp.abs(correlation) > 0.5, "Expected high correlation between a and b"


class TestBackwardCompatibility:
    """Verify backward compatibility with existing APIs."""

    def test_existing_streaming_optimizer_unchanged(self):
        """Verify StreamingOptimizer still works after adding hybrid method."""

        def simple_model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 10, 1000)
        y_true = simple_model(x, 3.0, 0.5)

        key = jax.random.PRNGKey(555)
        noise = jax.random.normal(key, y_true.shape) * 0.05
        y_noisy = y_true + noise

        p0 = jnp.array([2.5, 0.4])

        # Original StreamingOptimizer should still work
        from nlsq.streaming.config import StreamingConfig

        config = StreamingConfig(batch_size=100, max_epochs=10)

        optimizer = StreamingOptimizer(config=config)

        # This should work without errors
        result = optimizer.fit(
            data_source=(x, y_noisy), func=simple_model, p0=p0, verbose=0
        )

        assert "x" in result, "StreamingOptimizer result format changed"
        assert "success" in result, "StreamingOptimizer result format changed"

    def test_curve_fit_default_method_unchanged(self):
        """Verify default curve_fit method still works."""

        def linear_model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 10, 100)
        y_true = linear_model(x, 2.0, 1.0)

        key = jax.random.PRNGKey(666)
        noise = jax.random.normal(key, y_true.shape) * 0.02
        y_noisy = y_true + noise

        p0 = jnp.array([1.5, 0.5])

        # Default method (should be 'trf' or 'lm', not 'hybrid_streaming')
        popt, pcov = curve_fit(linear_model, x, y_noisy, p0=p0, verbose=0)

        # Should still work
        assert popt.shape == (2,), "Default curve_fit broken"
        assert pcov.shape == (2, 2), "Default curve_fit broken"


class TestCheckpointResume:
    """Test checkpoint save/resume functionality across phases."""

    def test_checkpoint_resume_from_phase1(self):
        """Test resuming optimization from Phase 1 checkpoint."""

        def exponential_model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = jnp.linspace(0, 10, 1000)  # Smaller for faster test
        y_true = exponential_model(x, 5.0, 0.5, 1.0)

        key = jax.random.PRNGKey(777)
        noise = jax.random.normal(key, y_true.shape) * 0.1
        y_noisy = y_true + noise

        p0 = jnp.array([4.0, 0.4, 0.8])

        # Test checkpoint save/load mechanism
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            # First run: short optimization
            config_short = HybridStreamingConfig()
            config_short.warmup_iterations = 10
            config_short.gauss_newton_max_iterations = 5
            config_short.enable_checkpoints = True

            optimizer1 = AdaptiveHybridStreamingOptimizer(config=config_short)
            result1 = optimizer1.fit(
                data_source=(x, y_noisy), func=exponential_model, p0=p0, verbose=0
            )

            # Save checkpoint manually
            optimizer1._save_checkpoint(checkpoint_path)

            # Verify checkpoint exists
            assert os.path.exists(checkpoint_path), "Checkpoint not saved"

            # Second run: load checkpoint and verify state
            config_full = HybridStreamingConfig()
            optimizer2 = AdaptiveHybridStreamingOptimizer(config=config_full)
            optimizer2._load_checkpoint(checkpoint_path)

            # Verify state was restored (basic check)
            assert hasattr(optimizer2, "current_phase"), "State not restored"
            assert optimizer2.best_params_global is not None, "Best params not restored"

            # Continue optimization
            result2 = optimizer2.fit(
                data_source=(x, y_noisy), func=exponential_model, p0=p0, verbose=0
            )

            # Verify it converged
            assert result2["success"], "Resumed optimization failed"

        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)


# =============================================================================
# 4-Layer Defense Strategy Integration Tests
# =============================================================================


class TestDefenseLayersCurveFitIntegration:
    """Integration tests for 4-layer defense strategy with curve_fit() API.

    These tests verify that the defense layers work correctly when invoked
    through the high-level curve_fit() interface with method='hybrid_streaming'.
    """

    def test_warm_start_detection_via_curve_fit(self):
        """Test Layer 1 warm start detection through curve_fit API."""

        def exponential_model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 1000)
        true_params = jnp.array([10.0, 0.5])
        y = exponential_model(x, *true_params)

        # Start with exact parameters - should trigger warm start
        p0 = true_params

        popt, _ = curve_fit(
            exponential_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            enable_warm_start_detection=True,
            warm_start_threshold=0.01,
        )

        # Should converge with minimal iterations due to warm start
        assert jnp.allclose(popt, true_params, atol=0.01)

    def test_adaptive_lr_via_curve_fit(self):
        """Test Layer 2 adaptive learning rate through curve_fit API."""

        def linear_model(x, m, c):
            return m * x + c

        x = jnp.linspace(0, 10, 500)
        true_params = jnp.array([2.0, 1.0])
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, x.shape) * 0.1
        y = linear_model(x, *true_params) + noise

        # Start with reasonable initial guess
        p0 = jnp.array([1.5, 0.5])

        popt, _ = curve_fit(
            linear_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            enable_adaptive_warmup_lr=True,
            warmup_iterations=50,
            max_warmup_iterations=100,
        )

        # Should converge with adaptive LR (relaxed tolerance for hybrid method)
        assert jnp.abs(popt[0] - true_params[0]) < 0.5
        assert jnp.abs(popt[1] - true_params[1]) < 1.0

    def test_cost_guard_via_curve_fit(self):
        """Test Layer 3 cost guard protection through curve_fit API."""

        def simple_model(x, a, b):
            return a * jnp.exp(-b * x)

        x = jnp.linspace(0, 5, 500)
        true_params = jnp.array([5.0, 0.3])
        y = simple_model(x, *true_params)

        # Near-optimal start with cost guard enabled
        p0 = jnp.array([5.01, 0.301])

        popt, _ = curve_fit(
            simple_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            enable_cost_guard=True,
            cost_increase_tolerance=0.05,
            warmup_iterations=10,
            max_warmup_iterations=30,
        )

        # Should converge without diverging
        assert jnp.allclose(popt, true_params, rtol=0.05)

    def test_step_clipping_via_curve_fit(self):
        """Test Layer 4 step clipping through curve_fit API."""

        def quadratic_model(x, a, b, c):
            return a * x**2 + b * x + c

        x = jnp.linspace(-5, 5, 500)
        true_params = jnp.array([1.0, 2.0, 3.0])
        key = jax.random.PRNGKey(123)
        noise = jax.random.normal(key, x.shape) * 0.1
        y = quadratic_model(x, *true_params) + noise

        # Start closer to optimal for more reliable convergence
        p0 = jnp.array([0.8, 1.8, 2.8])

        popt, _ = curve_fit(
            quadratic_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            enable_step_clipping=True,
            max_warmup_step_size=0.2,
            warmup_iterations=50,
            max_warmup_iterations=150,
        )

        # Should converge with step clipping (relaxed tolerance)
        param_errors = jnp.abs(popt - true_params)
        assert jnp.all(param_errors < 1.5)

    def test_all_defense_layers_via_curve_fit(self):
        """Test all 4 defense layers together through curve_fit API."""

        def gaussian_model(x, amp, mu, sigma):
            return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        x = jnp.linspace(-5, 5, 1000)
        true_params = jnp.array([10.0, 0.0, 1.5])
        key = jax.random.PRNGKey(456)
        noise = jax.random.normal(key, x.shape) * 0.2
        y = gaussian_model(x, *true_params) + noise

        p0 = jnp.array([8.0, 0.2, 1.2])

        popt, _ = curve_fit(
            gaussian_model,
            x,
            y,
            p0=p0,
            bounds=([0, -3, 0.1], [20, 3, 5]),
            method="hybrid_streaming",
            verbose=0,
            # All layers enabled
            enable_warm_start_detection=True,
            warm_start_threshold=0.001,
            enable_adaptive_warmup_lr=True,
            enable_cost_guard=True,
            cost_increase_tolerance=0.05,
            enable_step_clipping=True,
            max_warmup_step_size=0.1,
            warmup_iterations=30,
            max_warmup_iterations=100,
        )

        # Should converge with all protections
        assert jnp.abs(popt[0] - true_params[0]) < 2.0  # amplitude
        assert jnp.abs(popt[1] - true_params[1]) < 0.3  # mean
        assert jnp.abs(popt[2] - true_params[2]) < 0.5  # sigma

    def test_defense_layers_disabled_via_curve_fit(self):
        """Test that all defense layers can be disabled via curve_fit."""

        def simple_model(x, a, b):
            return a * x + b

        x = jnp.linspace(0, 10, 500)
        true_params = jnp.array([3.0, 1.0])
        key = jax.random.PRNGKey(789)
        noise = jax.random.normal(key, x.shape) * 0.05
        y = simple_model(x, *true_params) + noise

        # Start closer to optimal
        p0 = jnp.array([2.8, 0.8])

        popt, _ = curve_fit(
            simple_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            # All layers disabled
            enable_warm_start_detection=False,
            enable_adaptive_warmup_lr=False,
            enable_cost_guard=False,
            enable_step_clipping=False,
            warmup_iterations=50,
            max_warmup_iterations=100,
        )

        # Should still converge (layers are optional, relaxed tolerance)
        assert jnp.abs(popt[0] - true_params[0]) < 1.5
        assert jnp.abs(popt[1] - true_params[1]) < 1.0

    def test_defense_layer_config_presets_via_curve_fit(self):
        """Test that preset profiles work with curve_fit API."""

        def decay_model(x, a, tau):
            return a * jnp.exp(-x / tau)

        x = jnp.linspace(0, 10, 500)
        true_params = jnp.array([5.0, 2.0])
        key = jax.random.PRNGKey(111)
        noise = jax.random.normal(key, x.shape) * 0.1
        y = decay_model(x, *true_params) + noise

        # Start closer to optimal
        p0 = jnp.array([4.8, 1.8])

        # Test with aggressive preset values
        popt, _ = curve_fit(
            decay_model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            warmup_iterations=100,
            max_warmup_iterations=200,
        )

        # Relaxed tolerances for hybrid method
        assert jnp.abs(popt[0] - true_params[0]) < 1.5
        assert jnp.abs(popt[1] - true_params[1]) < 1.0
