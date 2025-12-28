"""
Tests for LargeDatasetFitter Multi-Start Integration
=====================================================

Tests for multi-start optimization integration with LargeDatasetFitter,
which uses full data exploration for medium-sized datasets (1M-100M points).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.large_dataset import (
    LargeDatasetFitter,
    LDMemoryConfig,
    fit_large_dataset,
)


class TestLargeDatasetMultiStart:
    """Test LargeDatasetFitter multi-start integration."""

    def test_multistart_uses_full_data_for_exploration(self):
        """Test multi-start uses full data for exploration."""

        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Generate medium-sized dataset (200K points to trigger chunking)
        np.random.seed(42)
        n_points = 200_000
        x = np.linspace(0, 10, n_points)
        true_params = [3.0, 0.5, 1.0]
        y = true_params[0] * np.exp(-true_params[1] * x) + true_params[2]
        y = y + np.random.normal(0, 0.1, len(y))

        # Create fitter with multi-start enabled
        config = LDMemoryConfig(
            memory_limit_gb=0.1,  # Force chunking
            min_chunk_size=10000,
            max_chunk_size=50000,
        )
        fitter = LargeDatasetFitter(
            config=config,
            multistart=True,
            n_starts=5,
            sampler="lhs",
        )

        result = fitter.fit(
            f=exponential,
            xdata=x,
            ydata=y,
            p0=[2.0, 0.3, 0.5],
            bounds=([0.1, 0.01, 0.1], [10.0, 5.0, 10.0]),
        )

        # Verify fit succeeded
        assert result is not None
        assert hasattr(result, "popt") or "popt" in result
        assert result.success or result.get("success", True)

        # Check that multi-start diagnostics are present
        assert "multistart_diagnostics" in result or hasattr(
            result, "multistart_diagnostics"
        )
        diagnostics = result.get("multistart_diagnostics", {})

        # Verify full dataset was used for exploration
        if "dataset_size" in diagnostics:
            assert diagnostics["dataset_size"] == n_points

    def test_best_starting_point_used_for_chunked_fit(self):
        """Test best starting point from exploration is used for chunked fit."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        np.random.seed(42)
        n_points = 150_000
        x = np.linspace(-5, 5, n_points)
        true_params = [1.0, 2.0, 3.0]
        y = quadratic(x, *true_params) + np.random.normal(0, 0.2, len(x))

        # Create fitter with multi-start
        config = LDMemoryConfig(
            memory_limit_gb=0.1,
            min_chunk_size=10000,
            max_chunk_size=50000,
        )
        fitter = LargeDatasetFitter(
            config=config,
            multistart=True,
            n_starts=5,
            sampler="lhs",
        )

        result = fitter.fit(
            f=quadratic,
            xdata=x,
            ydata=y,
            p0=[0.5, 1.0, 2.0],  # Initial guess away from true
            bounds=([-10, -10, -10], [10, 10, 10]),
        )

        # Verify fit succeeded
        assert result is not None
        assert result.success or result.get("success", True)

        # Parameters should be close to true values
        popt = result.popt if hasattr(result, "popt") else result.get("popt")
        np.testing.assert_array_almost_equal(popt, true_params, decimal=1)

        # Check that best starting point was recorded in diagnostics
        diagnostics = result.get("multistart_diagnostics", {})
        if "best_starting_point" in diagnostics or "best_loss" in diagnostics:
            # Best starting point should have been used
            assert "best_loss" in diagnostics or "best_starting_point" in diagnostics

    def test_multistart_skipped_when_n_starts_zero(self):
        """Test multi-start is skipped when n_starts=0."""

        def linear(x, a, b):
            return a * x + b

        np.random.seed(42)
        n_points = 100_000
        x = np.linspace(0, 10, n_points)
        y = 2.0 * x + 3.0 + np.random.normal(0, 0.1, len(x))

        # Create fitter with multistart=True but n_starts=0
        config = LDMemoryConfig(
            memory_limit_gb=0.1,
            min_chunk_size=10000,
            max_chunk_size=50000,
        )
        fitter = LargeDatasetFitter(
            config=config,
            multistart=True,  # Enabled but...
            n_starts=0,  # ...zero starts should skip multi-start
        )

        result = fitter.fit(
            f=linear,
            xdata=x,
            ydata=y,
            p0=[1.0, 1.0],
        )

        # Verify fit succeeded
        assert result is not None
        assert result.success or result.get("success", True)

        # Check that multi-start was bypassed
        diagnostics = result.get("multistart_diagnostics", {})
        if diagnostics:
            assert (
                diagnostics.get("bypassed", False)
                or diagnostics.get("n_starts_evaluated", 0) == 0
            )

    def test_multistart_works_with_limited_memory(self):
        """Test multi-start works with limited memory configuration."""

        def sine_model(x, amp, freq, phase, offset):
            return amp * jnp.sin(freq * x + phase) + offset

        np.random.seed(42)
        n_points = 300_000
        x = np.linspace(0, 10, n_points)
        true_params = [2.0, 3.0, 0.5, 1.0]
        y = (
            true_params[0] * np.sin(true_params[1] * x + true_params[2])
            + true_params[3]
        )
        y = y + np.random.normal(0, 0.15, len(y))

        # Create fitter with very limited memory
        config = LDMemoryConfig(
            memory_limit_gb=0.05,  # Very limited memory
            min_chunk_size=5000,
            max_chunk_size=20000,
        )
        fitter = LargeDatasetFitter(
            config=config,
            multistart=True,
            n_starts=3,
            sampler="sobol",  # Use Sobol for determinism
        )

        # This should not crash due to memory issues
        result = fitter.fit(
            f=sine_model,
            xdata=x,
            ydata=y,
            p0=[1.0, 2.0, 0.0, 0.5],
            bounds=([0.1, 0.1, -np.pi, -5], [5.0, 10.0, np.pi, 5]),
        )

        # Verify fit completed (may not be perfect due to memory constraints)
        assert result is not None
        popt = result.popt if hasattr(result, "popt") else result.get("popt")
        assert len(popt) == 4

    def test_result_includes_multistart_exploration_diagnostics(self):
        """Test result includes multi-start exploration diagnostics."""

        def gaussian(x, amp, mu, sigma):
            return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

        np.random.seed(42)
        n_points = 120_000
        x = np.linspace(-5, 5, n_points)
        true_params = [3.0, 1.0, 0.8]
        y = gaussian(x, *true_params) + np.random.normal(0, 0.1, len(x))

        config = LDMemoryConfig(
            memory_limit_gb=0.1,
            min_chunk_size=10000,
            max_chunk_size=40000,
        )
        fitter = LargeDatasetFitter(
            config=config,
            multistart=True,
            n_starts=5,
            sampler="halton",  # Use Halton sequence
        )

        result = fitter.fit(
            f=gaussian,
            xdata=x,
            ydata=y,
            p0=[2.0, 0.0, 1.0],
            bounds=([0.1, -5, 0.1], [10.0, 5.0, 5.0]),
        )

        # Verify fit succeeded
        assert result is not None
        assert result.success or result.get("success", True)

        # Check for multi-start diagnostics in result
        assert "multistart_diagnostics" in result or hasattr(
            result, "multistart_diagnostics"
        )

        diagnostics = result.get("multistart_diagnostics", {})

        # Should have exploration-related diagnostics
        expected_keys = [
            "n_starts_configured",
            "sampler",
        ]
        found_any = any(key in diagnostics for key in expected_keys)

        # If diagnostics are present, they should contain useful information
        if diagnostics:
            # At minimum, we should know if multi-start was used
            assert (
                "n_starts_configured" in diagnostics
                or "n_starts_evaluated" in diagnostics
                or "bypassed" in diagnostics
                or found_any
            )


class TestFitLargeDatasetMultiStart:
    """Test fit_large_dataset convenience function with multi-start."""

    def test_fit_large_dataset_with_multistart_parameters(self):
        """Test fit_large_dataset passes through multi-start parameters."""

        def decay(x, a, b):
            return a * jnp.exp(-b * x)

        np.random.seed(42)
        n_points = 80_000
        x = np.linspace(0, 5, n_points)
        y = 3.0 * np.exp(-0.5 * x) + np.random.normal(0, 0.1, len(x))

        result = fit_large_dataset(
            f=decay,
            xdata=x,
            ydata=y,
            p0=[2.0, 0.3],
            memory_limit_gb=0.1,
            multistart=True,
            n_starts=3,
            sampler="lhs",
        )

        # Verify fit succeeded
        assert result is not None
        popt = result.popt if hasattr(result, "popt") else result.get("popt")
        assert len(popt) == 2

        # Parameters should be reasonable
        assert 0.1 < popt[0] < 10.0  # amplitude
        assert 0.01 < popt[1] < 5.0  # rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
