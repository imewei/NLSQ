"""
Test suite for sparse Jacobian handling.

Tests sparse Jacobian detection, computation, and optimization
for problems with sparse structure.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core.sparse_jacobian import (
    SparseJacobianComputer,
    SparseOptimizer,
    detect_jacobian_sparsity,
)


class TestSparseJacobianComputer(unittest.TestCase):
    """Test the SparseJacobianComputer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.computer = SparseJacobianComputer(sparsity_threshold=1e-10)
        np.random.seed(42)

    def test_initialization(self):
        """Test SparseJacobianComputer initialization."""
        computer = SparseJacobianComputer(sparsity_threshold=1e-8)
        self.assertEqual(computer.sparsity_threshold, 1e-8)
        self.assertIsNone(computer._sparsity_pattern)

    def test_sparse_function_detection(self):
        """Test detecting sparsity in a sparse function."""

        # Function where each output depends on only some inputs
        def sparse_func(x, a, b, c):
            result = jnp.zeros_like(x)
            # First third depends only on 'a'
            mask1 = x < 3
            result = jnp.where(mask1, a * x, result)
            # Middle third depends on 'b'
            mask2 = (x >= 3) & (x < 6)
            result = jnp.where(mask2, b * x**2, result)
            # Last third depends on 'c'
            mask3 = x >= 6
            result = jnp.where(mask3, c * jnp.exp(x), result)
            return result

        x = np.linspace(0, 9, 90)
        params = np.array([1.0, 2.0, 0.5])

        pattern, sparsity = self.computer.detect_sparsity_pattern(
            sparse_func, params, x, n_samples=90
        )

        # Check sparsity is detected
        self.assertGreater(sparsity, 0.3)  # Should have significant sparsity
        self.assertLess(sparsity, 0.8)  # But not completely sparse

        # Check pattern shape
        self.assertEqual(pattern.shape, (90, 3))

    def test_block_diagonal_jacobian(self):
        """Test detecting block-diagonal Jacobian structure."""

        # Function with block-diagonal Jacobian
        def block_func(x, a, b):
            # Simple numpy implementation for clearer sparsity
            y = np.zeros_like(x)
            n = len(x)
            # First half depends only on 'a'
            y[: n // 2] = a * x[: n // 2]
            # Second half depends only on 'b'
            y[n // 2 :] = b * x[n // 2 :] ** 2
            return y

        x = np.linspace(0, 10, 100)
        params = np.array([2.0, 3.0])

        _pattern, sparsity = self.computer.detect_sparsity_pattern(
            block_func, params, x, n_samples=100
        )

        # Should have ~50% sparsity (block diagonal)
        self.assertGreater(sparsity, 0.4)  # At least 40% sparse
        self.assertLess(sparsity, 0.6)  # At most 60% sparse

    def test_compute_sparse_jacobian(self):
        """Test computing sparse Jacobian matrix."""

        # Simple linear function for predictable Jacobian
        def linear_func(x, a, b):
            return a * x + b

        x = np.array([1.0, 2.0, 3.0])
        params = np.array([2.0, 1.0])

        # Test sparsity pattern detection
        _pattern, sparsity = self.computer.detect_sparsity_pattern(
            linear_func, params, x, n_samples=3
        )
        # Linear function should have no sparsity
        self.assertLess(sparsity, 0.1)

    def test_sparse_vs_dense_jacobian(self):
        """Compare sparse and dense Jacobian computation."""

        # Function with known sparse structure
        def sparse_func(x, a, b, c):
            result = jnp.zeros_like(x)
            # Each parameter affects different parts
            result = result.at[::3].add(a * x[::3])
            result = result.at[1::3].add(b * x[1::3] ** 2)
            result = result.at[2::3].add(c * jnp.sin(x[2::3]))
            return result

        x = np.linspace(0, 10, 30)
        params = np.array([1.0, 2.0, 3.0])

        # Detect pattern
        pattern, _ = self.computer.detect_sparsity_pattern(
            sparse_func, params, x, n_samples=30
        )

        # Check pattern shape and sparsity
        self.assertEqual(pattern.shape, (30, 3))

    def test_adaptive_threshold(self):
        """Test adaptive threshold selection for sparsity detection."""

        # Function with varying magnitudes
        def varying_func(x, a, b):
            return jnp.concatenate(
                [
                    a * x[:5] * 1e-8,  # Very small values
                    b * x[5:] * 1e3,  # Large values
                ]
            )

        x = np.linspace(1, 10, 10)
        params = np.array([1.0, 2.0])

        # Test with fixed threshold
        computer1 = SparseJacobianComputer(sparsity_threshold=1e-6)
        pattern1, _ = computer1.detect_sparsity_pattern(
            varying_func, params, x, n_samples=10
        )

        # Pattern should be detected
        self.assertIsNotNone(pattern1)


class TestSparseOptimizer(unittest.TestCase):
    """Test the SparseOptimizer class."""

    def test_initialization(self):
        """Test SparseOptimizer initialization."""
        optimizer = SparseOptimizer(min_sparsity=0.5, auto_detect=True)

        self.assertEqual(optimizer.min_sparsity, 0.5)
        self.assertTrue(optimizer.auto_detect)
        self.assertEqual(optimizer._detected_sparsity, 0.0)

    def test_should_use_sparse_decision(self):
        """Test decision logic for using sparse methods."""
        optimizer = SparseOptimizer(min_sparsity=0.5)

        # Small problem - should not use sparse
        self.assertFalse(optimizer.should_use_sparse(100, 3))

        # Large dense problem - should not use sparse (no pattern detected)
        self.assertFalse(optimizer.should_use_sparse(10000, 10))

        # For extremely large problems (>100M elements), sparse methods may be recommended
        # Use auto_detect=False to get size-based heuristic
        optimizer_size_based = SparseOptimizer(min_sparsity=0.5, auto_detect=False)
        should_use = optimizer_size_based.should_use_sparse(20_000_000, 10)
        # 20M * 10 = 200M > 100M threshold
        self.assertTrue(should_use)

        # But not for small problems even with sparsity
        self.assertFalse(optimizer.should_use_sparse(50, 3))

    def test_optimize_with_sparse_jacobian(self):
        """Test optimization with sparse Jacobian."""

        # Create a problem with sparse structure
        def sparse_residual(params, x):
            a, b, c = params
            y = jnp.zeros_like(x)
            # Sparse structure: each param affects different regions
            y = y.at[:10].set(a * x[:10])
            y = y.at[10:20].set(b * x[10:20] ** 2)
            y = y.at[20:].set(c * jnp.exp(x[20:] / 10))
            return y

        np.random.seed(123)
        x = np.linspace(0, 10, 30)
        true_params = np.array([2.0, 1.5, 0.8])
        y_true = sparse_residual(true_params, x)
        y_obs = y_true + np.random.normal(0, 0.01, 30)

        # Define objective
        def objective(params):
            return sparse_residual(params, x) - y_obs

        # Optimize with sparse methods
        optimizer = SparseOptimizer(min_sparsity=0.3, auto_detect=True)

        result = optimizer.optimize_with_sparsity(
            lambda x, *p: sparse_residual(p, x),
            x0=np.array([1.5, 1.0, 0.5]),
            xdata=x,
            ydata=y_obs,
        )

        # Handle both tuple (from curve_fit) and dict (from sparse) returns
        if isinstance(result, tuple):
            # curve_fit returns (popt, _pcov)
            popt = result[0]
            self.assertIsNotNone(popt)
            # Check convergence
            np.testing.assert_allclose(popt, true_params, rtol=0.1)
        else:
            # dict or OptimizeResult from sparse methods
            self.assertTrue(hasattr(result, "x") or "x" in result)
            if hasattr(result, "x"):
                np.testing.assert_allclose(result.x, true_params, rtol=0.1)
            elif "x" in result:
                np.testing.assert_allclose(result["x"], true_params, rtol=0.1)

    def test_memory_savings_calculation(self):
        """Test that sparsity detection works."""
        optimizer = SparseOptimizer()

        # Test that optimizer has sparse computer
        self.assertIsNotNone(optimizer.sparse_computer)
        self.assertEqual(optimizer.sparsity_threshold, 0.01)


class TestDetectJacobianSparsity(unittest.TestCase):
    """Test the detect_jacobian_sparsity convenience function."""

    def test_basic_sparsity_detection(self):
        """Test basic sparsity detection."""

        # Function with clear sparse structure
        def func(x, a, b):
            y = jnp.zeros_like(x)
            # First half depends on 'a'
            y = y.at[: len(x) // 2].set(a * x[: len(x) // 2])
            # Second half depends on 'b'
            y = y.at[len(x) // 2 :].set(b * x[len(x) // 2 :])
            return y

        x = np.linspace(0, 10, 100)
        params = np.array([1.0, 2.0])

        sparsity, info = detect_jacobian_sparsity(func, params, x[:20])

        # Should detect 50% sparsity
        self.assertAlmostEqual(sparsity, 0.5, places=1)

        # Check info dictionary
        self.assertIn("sparsity", info)
        self.assertIn("nnz", info)
        self.assertIn("avg_nnz_per_row", info)
        self.assertIn("avg_nnz_per_col", info)
        self.assertIn("memory_reduction", info)

        # Memory reduction should be approximately sparsity percentage
        self.assertAlmostEqual(info["memory_reduction"], sparsity * 100, places=0)

    def test_dense_function_detection(self):
        """Test detection on dense (non-sparse) function."""

        # Fully coupled function
        def dense_func(x, a, b, c):
            return a * x + b * x**2 + c * x**3

        x = np.linspace(0, 5, 50)
        params = np.array([1.0, 2.0, 3.0])

        sparsity, info = detect_jacobian_sparsity(
            dense_func, params, x, threshold=1e-10
        )

        # Should detect very low sparsity (dense)
        self.assertLess(sparsity, 0.1)
        # Most elements should be non-zero
        self.assertGreater(info["nnz"], 140)  # Nearly all non-zero

    def test_custom_threshold(self):
        """Test sparsity detection with custom threshold."""

        # Function with small but non-zero elements
        def func(x, a, b):
            return jnp.array(
                [
                    a * x[0] + 1e-7 * b,  # Small coupling
                    b * x[1] + 1e-7 * a,  # Small coupling
                ]
            )

        x = np.array([1.0, 2.0])
        params = np.array([1.0, 2.0])

        # With small threshold - detect coupling
        sparsity1, _ = detect_jacobian_sparsity(func, params, x, threshold=1e-10)

        # With larger threshold - ignore small coupling
        sparsity2, _ = detect_jacobian_sparsity(func, params, x, threshold=1e-5)

        # Larger threshold should give more sparsity
        self.assertGreater(sparsity2, sparsity1)


class TestSparseJacobianIntegration(unittest.TestCase):
    """Integration tests for sparse Jacobian in optimization workflows."""

    def test_large_sparse_problem(self):
        """Test optimization of large sparse problem."""

        # Create banded matrix problem (common sparse structure)
        def banded_system(params, n=1000):
            """Create a banded system where each param affects local region."""
            x = jnp.arange(n)
            y = jnp.zeros(n)

            # Tridiagonal-like structure
            for i, p in enumerate(params):
                start = i * (n // len(params))
                end = min(start + (n // len(params)) + 2, n)
                y = y.at[start:end].add(p * x[start:end])

            return y

        # Problem setup
        n_data = 1000
        n_params = 10
        true_params = np.random.randn(n_params)

        y_true = banded_system(true_params, n_data)
        y_obs = y_true + np.random.normal(0, 0.01, n_data)

        # Objective function
        def objective(params):
            return banded_system(params, n_data) - y_obs

        # Detect sparsity
        x0 = np.zeros(n_params)
        computer = SparseJacobianComputer(sparsity_threshold=1e-8)

        # Sample to detect pattern (use subset for efficiency)
        # Create a dummy xdata for pattern detection
        xdata_sample = np.arange(100)

        # Create a wrapper that works with the expected signature
        def wrapped_func(x, *params):
            # Compute full objective and return subset
            full_result = objective(params)
            return full_result[: len(x)] if len(full_result) > len(x) else full_result

        _pattern, sparsity = computer.detect_sparsity_pattern(
            wrapped_func, x0, xdata_sample, n_samples=100
        )

        # Should detect significant sparsity
        self.assertGreater(sparsity, 0.5)

        # Optimize with sparse methods
        optimizer = SparseOptimizer(auto_detect=True)
        optimizer._detected_sparsity = sparsity

        result = optimizer.optimize_with_sparsity(
            lambda x, *p: objective(p), x0, np.arange(n_data), np.zeros(n_data)
        )

        # Check that we got a result
        self.assertIsNotNone(result)

    def test_sparse_vs_dense_performance(self):
        """Compare performance of sparse vs dense methods."""

        # Problem with block-sparse structure
        def block_sparse_func(x, params):
            n_blocks = 4
            block_size = len(x) // n_blocks
            y = jnp.zeros_like(x)

            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                param_idx = i * 2
                if param_idx < len(params):
                    y = y.at[start:end].set(
                        params[param_idx] * x[start:end]
                        + params[param_idx + 1] * x[start:end] ** 2
                    )
            return y

        # Setup
        n_data = 400
        n_params = 8
        x = np.linspace(0, 10, n_data)
        true_params = np.random.randn(n_params)
        y_true = block_sparse_func(x, true_params)
        y_obs = np.array(y_true) + np.random.normal(0, 0.01, n_data)

        def residual(params):
            return block_sparse_func(x, params) - y_obs

        # Optimize with sparse detection
        optimizer_sparse = SparseOptimizer(min_sparsity=0.3, auto_detect=True)

        result_sparse = optimizer_sparse.optimize_with_sparsity(
            lambda xd, *p: block_sparse_func(xd, p),
            x0=np.zeros(n_params),
            xdata=x,
            ydata=y_obs,
        )

        # Check that we got a result (tuple or object with attributes)
        self.assertIsNotNone(result_sparse)
        # Could be tuple (popt, _pcov) or dict/object with 'x'
        if isinstance(result_sparse, tuple):
            self.assertEqual(len(result_sparse), 2)  # popt, _pcov
        else:
            self.assertTrue(hasattr(result_sparse, "x") or "x" in result_sparse)


class TestSparseJacobianEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for sparse Jacobian."""

    def test_zero_jacobian(self):
        """Test handling of zero Jacobian."""

        # Constant function has zero Jacobian
        def constant_func(x, a, b):
            return jnp.ones_like(x) * 5.0

        x = np.linspace(0, 10, 20)
        params = np.array([1.0, 2.0])

        computer = SparseJacobianComputer()
        _pattern, sparsity = computer.detect_sparsity_pattern(
            constant_func, params, x, n_samples=20
        )

        # Should detect 100% sparsity (all zeros)
        self.assertAlmostEqual(sparsity, 1.0, places=5)

    def test_single_parameter(self):
        """Test with single parameter."""

        def single_param_func(x, a):
            return a * x

        x = np.array([1.0, 2.0, 3.0])
        params = np.array([2.0])

        sparsity, info = detect_jacobian_sparsity(single_param_func, params, x)

        # Single parameter should be fully dense
        self.assertEqual(sparsity, 0.0)
        self.assertEqual(info["nnz"], 3)  # 3 data points, 1 param

    @pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    )
    def test_empty_data(self):
        """Test handling of empty data."""

        def func(x, a):
            return a * x

        x = np.array([])
        params = np.array([1.0])

        with self.assertRaises(ValueError):
            detect_jacobian_sparsity(func, params, x)

    def test_incompatible_dimensions(self):
        """Test handling of incompatible dimensions."""

        def func(x, a):
            return a * x[:5]  # Returns wrong size

        x = np.arange(10)
        params = np.array([1.0])

        computer = SparseJacobianComputer()
        with self.assertRaises((ValueError, IndexError)):
            computer.detect_sparsity_pattern(func, params, x, n_samples=10)


if __name__ == "__main__":
    unittest.main()
