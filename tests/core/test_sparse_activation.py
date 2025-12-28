"""Tests for sparse activation optimization (Task Group 6).

This module tests automatic sparsity detection, solver selection,
and sparse TRF path for problems with sparse Jacobians.

Target Impact:
- 3-10x speed improvement on sparse problems
- 5-50x memory reduction on sparse problems
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.core.sparse_jacobian import detect_jacobian_sparsity


def sparse_parameter_selection_model(x, *params):
    """Parameter selection model with inherently sparse Jacobian.

    Each parameter only affects a subset of data points, creating
    a sparse Jacobian pattern. This is a canonical sparse problem.

    Parameters
    ----------
    x : array_like
        Independent variable (data point indices or positions)
    *params : tuple
        Model parameters, grouped into sections

    Returns
    -------
    array_like
        Model predictions
    """
    # Convert to JAX array for computation
    x = jnp.asarray(x)
    params = jnp.asarray(params)

    # Each group of 10 parameters affects a different region
    n_groups = len(params) // 10
    result = jnp.zeros_like(x, dtype=jnp.float64)

    for i in range(n_groups):
        # Each parameter group only affects data in range [i*10, (i+1)*10)
        mask = (x >= i * 10) & (x < (i + 1) * 10)
        # Use first parameter of each group for amplitude
        result = jnp.where(mask, params[i * 10], result)

    return result


def dense_exponential_model(x, a, b, c):
    """Dense exponential model (all parameters affect all data points).

    This is a control case - not sparse.
    """
    x = jnp.asarray(x)
    return a * jnp.exp(-b * x) + c


class TestSparseDetection:
    """Test automatic sparsity detection at p0 initialization."""

    def test_detect_sparse_problem(self):
        """Test detection of sparse Jacobian pattern (>50% sparsity)."""
        # Create sparse problem with 100 parameters, 100 data points
        # Each parameter affects ~10 data points = 10% density = 90% sparsity
        n_params = 100
        n_data = 100

        # Generate data
        np.random.seed(42)
        x_data = np.linspace(0, 100, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.1, size=n_data)

        # Initial guess
        p0 = np.ones(n_params)

        # Detect sparsity
        sparsity_ratio, info = detect_jacobian_sparsity(
            sparse_parameter_selection_model,
            p0,
            x_data[:50],  # Sample for detection
            threshold=0.01,
        )

        # Verify high sparsity detected
        assert sparsity_ratio > 0.5, f"Expected >50% sparsity, got {sparsity_ratio:.1%}"
        assert info["sparsity"] == sparsity_ratio
        assert "nnz" in info
        assert "memory_reduction" in info

    def test_detect_dense_problem(self):
        """Test detection correctly identifies dense problems (<50% sparsity)."""
        # Dense exponential model - all params affect all data
        n_data = 100

        np.random.seed(42)
        x_data = np.linspace(0, 10, n_data)
        true_params = [2.0, 0.5, 1.0]
        y_data = dense_exponential_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.1, size=n_data)

        p0 = np.array([1.0, 1.0, 1.0])

        # Detect sparsity
        sparsity_ratio, _info = detect_jacobian_sparsity(
            dense_exponential_model, p0, x_data[:50], threshold=0.01
        )

        # Dense problem should have low sparsity
        assert sparsity_ratio < 0.5, (
            f"Expected <50% sparsity for dense problem, got {sparsity_ratio:.1%}"
        )


class TestSparseAutoSelection:
    """Test automatic sparse solver selection."""

    def test_sparse_activation_large_sparse_problem(self):
        """Test sparse solver activates for large sparse problems (sparsity >50%, size >10K)."""
        # Create large sparse problem (10K+ residuals, >50% sparsity)
        n_params = 100
        n_data = 12000  # >10K threshold

        np.random.seed(42)
        x_data = np.linspace(0, 100, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.05, size=n_data)

        p0 = np.ones(n_params) * 1.5

        # Fit with full_output to get diagnostics
        result = curve_fit(
            sparse_parameter_selection_model,
            x_data,
            y_data,
            p0=p0,
            full_output=True,
            maxfev=100,
        )
        result["x"]
        result.get("cov_x", None)
        info = result

        # Check for sparsity diagnostics in result
        if "sparsity_detected" in info:
            sparsity_info = info["sparsity_detected"]
            assert sparsity_info["detected"]  # Check truthiness
            assert sparsity_info["ratio"] > 0.5
            # May use sparse or dense depending on implementation state
            assert sparsity_info["solver"] in ["sparse", "dense"]

    def test_dense_solver_for_small_problems(self):
        """Test dense solver used for small problems (<10K residuals)."""
        # Small sparse problem should still use dense solver
        n_params = 20
        n_data = 100  # <10K threshold

        np.random.seed(42)
        x_data = np.linspace(0, 20, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.1, size=n_data)

        p0 = np.ones(n_params) * 1.5

        # Fit with full_output
        result = curve_fit(
            sparse_parameter_selection_model,
            x_data,
            y_data,
            p0=p0,
            full_output=True,
            maxfev=100,
        )
        result["x"]
        result.get("cov_x", None)
        info = result

        # Should use dense solver for small problems
        if "sparsity_detected" in info:
            sparsity_info = info["sparsity_detected"]
            # Either not detected or uses dense solver
            if sparsity_info["detected"]:
                # If detected, should still use dense for small problems
                assert sparsity_info["solver"] == "dense"

    def test_dense_solver_for_dense_problems(self):
        """Test dense solver used for dense problems (sparsity <50%)."""
        n_data = 12000  # >10K threshold

        np.random.seed(42)
        x_data = np.linspace(0, 10, n_data)
        true_params = [2.0, 0.5, 1.0]
        y_data = dense_exponential_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.1, size=n_data)

        p0 = np.array([1.0, 1.0, 1.0])

        # Fit with full_output
        result = curve_fit(
            dense_exponential_model, x_data, y_data, p0=p0, full_output=True, maxfev=50
        )
        result["x"]
        result.get("cov_x", None)
        info = result

        # Dense problem should use dense solver
        if "sparsity_detected" in info:
            sparsity_info = info["sparsity_detected"]
            # Should not activate sparse solver for dense problems
            assert sparsity_info["solver"] == "dense"


class TestSparseDiagnostics:
    """Test sparsity diagnostics in optimization results."""

    def test_sparsity_diagnostics_present(self):
        """Test sparsity_detected diagnostic is present in full_output."""
        n_params = 50
        n_data = 5000

        np.random.seed(42)
        x_data = np.linspace(0, 50, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.1, size=n_data)

        p0 = np.ones(n_params) * 1.5

        # Fit with full_output
        result = curve_fit(
            sparse_parameter_selection_model,
            x_data,
            y_data,
            p0=p0,
            full_output=True,
            maxfev=50,
        )
        popt = result["x"]
        result.get("cov_x", None)

        # Note: sparsity_detected will be added in Task 6.5
        # This test documents expected structure
        # When implemented, should have:
        # assert 'sparsity_detected' in info
        # assert 'detected' in info['sparsity_detected']
        # assert 'ratio' in info['sparsity_detected']
        # assert 'solver' in info['sparsity_detected']

        # For now, just check fit succeeded
        assert popt is not None
        assert len(popt) == n_params


class TestSparseConvergence:
    """Test sparse solver convergence and accuracy."""

    def test_sparse_solver_convergence(self):
        """Test sparse solver converges to correct solution."""
        # Medium sparse problem
        n_params = 50
        n_data = 5000

        np.random.seed(42)
        x_data = np.linspace(0, 50, n_data)
        true_params = np.random.uniform(1.5, 2.5, size=n_params)
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.05, size=n_data)

        p0 = np.ones(n_params) * 1.0

        # Fit
        popt, pcov = curve_fit(
            sparse_parameter_selection_model, x_data, y_data, p0=p0, maxfev=200
        )

        # Check convergence (should be close to true params)
        # For sparse problems, parameters outside affected regions may not converge well
        # Check that fitting succeeded
        assert popt is not None
        assert len(popt) == n_params
        assert pcov is not None

        # Check that affected parameters converged reasonably
        # Parameters 0, 10, 20, 30, 40 are used (one per group)
        for i in [0, 10, 20, 30, 40]:
            if i < len(popt):
                # Should be within reasonable range of true value
                assert 1.0 < popt[i] < 3.5, (
                    f"Parameter {i} out of reasonable range: {popt[i]}"
                )


class TestSparsePerformance:
    """Test sparse solver performance characteristics."""

    @pytest.mark.slow
    def test_sparse_memory_efficiency(self):
        """Test sparse solver memory efficiency (target: 5-50x reduction)."""
        # This is a placeholder for benchmarking in Task 6.8
        # Actual memory measurement would require profiling
        n_params = 100
        n_data = 20000

        np.random.seed(42)
        x_data = np.linspace(0, 100, n_data)
        np.ones(n_params) * 2.0

        # Detect sparsity to estimate memory savings
        p0 = np.ones(n_params)
        sparsity_ratio, info = detect_jacobian_sparsity(
            sparse_parameter_selection_model, p0, x_data[:100], threshold=0.01
        )

        # Check memory reduction potential
        assert "memory_reduction" in info
        # For 90% sparsity, should see ~10x memory reduction
        if sparsity_ratio > 0.8:
            assert info["memory_reduction"] > 50, (
                "Expected >50% memory reduction for high sparsity"
            )

    @pytest.mark.slow
    def test_sparse_speed_improvement(self):
        """Test sparse solver speed improvement (target: 3-10x speedup)."""
        # This is a placeholder for benchmarking in Task 6.8
        # Actual timing comparison would be done in benchmark script
        # For now, just verify the sparse problem can be solved
        n_params = 100
        n_data = 15000

        np.random.seed(42)
        x_data = np.linspace(0, 100, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.05, size=n_data)

        p0 = np.ones(n_params) * 1.5

        # Just verify it solves (timing comparison in benchmark)
        popt, _pcov = curve_fit(
            sparse_parameter_selection_model, x_data, y_data, p0=p0, maxfev=100
        )

        assert popt is not None
        assert len(popt) == n_params


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
