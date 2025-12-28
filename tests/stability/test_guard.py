"""Comprehensive tests for NLSQ stability optimizations.

This module tests all stability enhancements including:
- Numerical stability guards
- Input validation
- Recovery mechanisms
- Algorithm selection
- Memory management
- Robust decomposition
- Smart caching
- Convergence diagnostics
"""

import unittest

import numpy as np
import pytest

from nlsq.config import JAXConfig

_jax_config = JAXConfig()

import jax.numpy as jnp

from nlsq.caching.memory_manager import MemoryManager
from nlsq.caching.smart_cache import SmartCache, cached_function
from nlsq.precision.algorithm_selector import AlgorithmSelector
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.stability.recovery import OptimizationRecovery
from nlsq.stability.robust_decomposition import RobustDecomposition
from nlsq.utils.diagnostics import ConvergenceMonitor
from nlsq.utils.validators import InputValidator


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability guard functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.guard = NumericalStabilityGuard()

    def test_safe_exp(self):
        """Test safe exponential function."""
        # Test normal values
        x = jnp.array([1.0, 2.0, 3.0])
        result = self.guard.safe_exp(x)
        self.assertTrue(jnp.all(jnp.isfinite(result)))

        # Test extreme values that would overflow
        x_extreme = jnp.array([1000.0, -1000.0, 500.0])
        result = self.guard.safe_exp(x_extreme)
        self.assertTrue(jnp.all(jnp.isfinite(result)))

    def test_safe_log(self):
        """Test safe logarithm function."""
        # Test normal values
        x = jnp.array([1.0, 2.0, 3.0])
        result = self.guard.safe_log(x)
        self.assertTrue(jnp.all(jnp.isfinite(result)))

        # Test near-zero and negative values
        x_problematic = jnp.array([0.0, -1.0, 1e-20])
        result = self.guard.safe_log(x_problematic)
        self.assertTrue(jnp.all(jnp.isfinite(result)))

    @pytest.mark.filterwarnings(
        "ignore:Jacobian contains NaN or Inf values:UserWarning"
    )
    @pytest.mark.filterwarnings("ignore:Ill-conditioned Jacobian:UserWarning")
    def test_check_and_fix_jacobian(self):
        """Test Jacobian checking and fixing."""
        # Create problematic Jacobian
        J = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1e-15, 0.0],  # Near-zero singular value
                [0.0, 0.0, jnp.nan],  # NaN value
            ]
        )

        J_fixed, issues = self.guard.check_and_fix_jacobian(J)

        # Check that issues were detected
        self.assertIn("has_nan", issues)
        self.assertTrue(issues["has_nan"])

        # Check that Jacobian is now valid
        self.assertTrue(jnp.all(jnp.isfinite(J_fixed)))

    def test_detect_numerical_issues(self):
        """Test detection of numerical issues."""
        # Test array with NaN
        arr_nan = jnp.array([1.0, jnp.nan, 3.0])
        issues = self.guard.detect_numerical_issues(arr_nan)
        self.assertTrue(issues["has_nan"])
        self.assertFalse(issues["has_inf"])

        # Test array with Inf
        arr_inf = jnp.array([1.0, jnp.inf, 3.0])
        issues = self.guard.detect_numerical_issues(arr_inf)
        self.assertTrue(issues["has_inf"])
        self.assertFalse(issues["has_nan"])


class TestInputValidator(unittest.TestCase):
    """Test input validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = InputValidator()

    def test_validate_curve_fit_inputs(self):
        """Test curve fit input validation."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        xdata = np.linspace(0, 4, 50)
        ydata = np.array(model(xdata, 2.5, 1.3))  # Convert to numpy for mutability

        # Test valid inputs
        errors, _validation_warnings, xdata_valid, ydata_valid = (
            self.validator.validate_curve_fit_inputs(model, xdata, ydata, p0=[2.0, 1.0])
        )
        self.assertEqual(errors, [])  # No errors for valid inputs
        self.assertTrue(np.all(np.isfinite(xdata_valid)))
        self.assertTrue(np.all(np.isfinite(ydata_valid)))

        # Test invalid inputs with NaN
        ydata_nan = ydata.copy()
        ydata_nan[10] = np.nan

        # Validator with check_finite=True should detect NaN
        errors, _warnings, xdata_valid, ydata_valid = (
            self.validator.validate_curve_fit_inputs(
                model, xdata, ydata_nan, check_finite=True
            )
        )
        self.assertTrue(len(errors) > 0)  # Should have errors for NaN values

    def test_validate_least_squares_inputs(self):
        """Test least squares input validation."""
        x0 = np.array([1.0, 2.0, 3.0])
        # Use array bounds to match x0 length
        bounds = (np.array([-10, -10, -10]), np.array([10, 10, 10]))

        # Create a dummy function for validation
        def dummy_fun(x):
            return x

        # Test valid inputs
        errors, _validation_warnings, x0_valid = (
            self.validator.validate_least_squares_inputs(dummy_fun, x0, bounds)
        )
        self.assertEqual(errors, [])  # No errors for valid inputs
        self.assertTrue(np.all(np.isfinite(x0_valid)))

        # Test invalid bounds (lower > upper)
        invalid_bounds = (np.array([10, 10, 10]), np.array([-10, -10, -10]))
        errors, _warnings, x0_valid = self.validator.validate_least_squares_inputs(
            dummy_fun, x0, invalid_bounds
        )
        self.assertTrue(len(errors) > 0)  # Should have errors for invalid bounds


class TestAlgorithmSelector(unittest.TestCase):
    """Test automatic algorithm selection."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_analyze_problem(self):
        """Test problem analysis."""

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        xdata = np.linspace(0, 4, 100)
        ydata = model(xdata, 2.5, 1.3)
        p0 = np.array([2.0, 1.0])

        analysis = self.selector.analyze_problem(model, xdata, ydata, p0)

        # Check analysis results
        self.assertEqual(analysis["n_points"], 100)
        self.assertEqual(analysis["n_params"], 2)
        self.assertEqual(analysis["size_class"], "small")
        self.assertIn("has_outliers", analysis)
        self.assertIn("is_noisy", analysis)

    def test_select_algorithm(self):
        """Test algorithm selection based on problem characteristics."""
        # Small problem without bounds
        analysis_small = {
            "n_points": 50,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": False,
            "condition_estimate": 10,
        }
        rec = self.selector.select_algorithm(analysis_small)
        self.assertEqual(
            rec["algorithm"], "trf"
        )  # TRF is the only implemented algorithm

        # Large problem with bounds
        analysis_large = {
            "n_points": 1000000,
            "n_params": 5,
            "has_bounds": True,
            "has_outliers": False,
            "condition_estimate": 100,
        }
        rec = self.selector.select_algorithm(analysis_large)
        self.assertEqual(rec["algorithm"], "trf")  # TRF for all cases

        # Problem with outliers
        analysis_outliers = {
            "n_points": 1000,
            "n_params": 4,
            "has_bounds": False,
            "has_outliers": True,
            "outlier_fraction": 0.15,
            "condition_estimate": 50,
        }
        rec = self.selector.select_algorithm(analysis_outliers)
        self.assertEqual(rec["loss"], "cauchy")  # Robust loss for outliers


class TestOptimizationRecovery(unittest.TestCase):
    """Test optimization recovery mechanisms."""

    def setUp(self):
        """Set up test fixtures."""
        self.recovery = OptimizationRecovery(max_retries=2)

    def test_perturb_parameters(self):
        """Test parameter perturbation strategy."""
        state = {"params": np.array([1.0, 2.0, 3.0]), "method": "trf"}

        modified = self.recovery._perturb_parameters("convergence", state, retry=0)

        # Check that parameters were perturbed
        self.assertFalse(np.allclose(state["params"], modified["params"]))

    def test_switch_algorithm(self):
        """Test algorithm switching strategy."""
        state = {"method": "trf", "params": np.array([1.0, 2.0])}

        modified = self.recovery._switch_algorithm("convergence", state, retry=0)

        # Check that algorithm was switched
        self.assertNotEqual(state["method"], modified["method"])
        self.assertIn(modified["method"], ["lm", "dogbox"])

    def test_adjust_regularization(self):
        """Test regularization adjustment."""
        state = {"method": "trf", "regularization": 1e-10, "has_outliers": True}

        modified = self.recovery._adjust_regularization("numerical", state, retry=0)

        # Check that regularization was increased
        self.assertGreater(modified["regularization"], state["regularization"])
        # Check that loss function was changed for outliers
        self.assertIn("loss", modified)


class TestMemoryManager(unittest.TestCase):
    """Test memory management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory_manager = MemoryManager()

    def test_predict_memory_requirement(self):
        """Test memory requirement prediction."""
        # Small problem
        bytes_needed = self.memory_manager.predict_memory_requirement(
            n_points=1000, n_params=5, algorithm="trf"
        )
        self.assertGreater(bytes_needed, 0)
        self.assertLess(bytes_needed, 1e9)  # Less than 1 GB

        # Large problem
        bytes_needed_large = self.memory_manager.predict_memory_requirement(
            n_points=1000000, n_params=10, algorithm="trf"
        )
        self.assertGreater(bytes_needed_large, bytes_needed)

    def test_allocate_array(self):
        """Test array allocation with pooling."""
        shape = (100, 50)
        arr1 = self.memory_manager.allocate_array(shape)

        # Check allocation
        self.assertEqual(arr1.shape, shape)
        self.assertEqual(arr1.dtype, np.float64)

        # Free and reallocate - should reuse from pool
        self.memory_manager.free_array(arr1)
        arr2 = self.memory_manager.allocate_array(shape)

        # Arrays should be the same object from pool
        self.assertTrue(np.shares_memory(arr1, arr2))

    def test_memory_guard(self):
        """Test memory guard context manager."""
        bytes_needed = 1000000  # 1 MB

        with self.memory_manager.memory_guard(bytes_needed):
            # Should succeed for reasonable memory
            pass

        # Test with unreasonable memory (100 TB)
        with (
            self.assertRaises(MemoryError),
            self.memory_manager.memory_guard(100 * 1024**4),
        ):
            pass


class TestSmartCache(unittest.TestCase):
    """Test smart caching functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = SmartCache(max_memory_items=10, disk_cache_enabled=False)

    def test_cache_key_generation(self):
        """Test cache key generation."""
        # Test with arrays
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])
        key1 = self.cache.cache_key(arr1, param=5)
        key2 = self.cache.cache_key(arr2, param=5)
        self.assertEqual(key1, key2)  # Same content = same key

        # Different content = different key
        arr3 = np.array([1, 2, 4])
        key3 = self.cache.cache_key(arr3, param=5)
        self.assertNotEqual(key1, key3)

    def test_cache_set_get(self):
        """Test cache set and get operations."""
        key = "test_key"
        value = np.array([1, 2, 3])

        # Set value
        self.cache.set(key, value)

        # Get value
        retrieved = self.cache.get(key)
        self.assertTrue(np.array_equal(value, retrieved))

        # Miss for non-existent key
        miss = self.cache.get("nonexistent")
        self.assertIsNone(miss)

    def test_cached_function_decorator(self):
        """Test cached function decorator."""
        call_count = 0

        @cached_function(cache=self.cache)
        def expensive_func(x):
            nonlocal call_count
            call_count += 1
            return x**2

        # First call - compute
        result1 = expensive_func(5)
        self.assertEqual(result1, 25)
        self.assertEqual(call_count, 1)

        # Second call - from cache
        result2 = expensive_func(5)
        self.assertEqual(result2, 25)
        self.assertEqual(call_count, 1)  # Not incremented

        # Different argument - compute again
        result3 = expensive_func(6)
        self.assertEqual(result3, 36)
        self.assertEqual(call_count, 2)


class TestConvergenceMonitor(unittest.TestCase):
    """Test convergence monitoring."""

    def setUp(self):
        """Set up test fixtures."""
        self.monitor = ConvergenceMonitor()

    def test_detect_oscillation(self):
        """Test oscillation detection."""
        # Add oscillating cost values
        for i in range(10):
            cost = 100 + 10 * (-1) ** i
            self.monitor.add_iteration(cost, np.array([1.0, 2.0]))

        is_oscillating, _score = self.monitor.detect_oscillation()
        self.assertTrue(is_oscillating)

    def test_detect_stagnation(self):
        """Test stagnation detection."""
        # Add stagnant cost values
        for i in range(10):
            cost = 100.0 + 1e-10 * i  # Very small changes
            self.monitor.add_iteration(cost, np.array([1.0, 2.0]))

        is_stagnant, _score = self.monitor.detect_stagnation()
        self.assertTrue(is_stagnant)

    def test_detect_divergence(self):
        """Test divergence detection."""
        # Add diverging cost values
        for i in range(10):
            cost = 100 * (1.5**i)  # Exponentially increasing
            self.monitor.add_iteration(cost, np.array([1.0, 2.0]))

        is_diverging, _score = self.monitor.detect_divergence()
        self.assertTrue(is_diverging)


class TestRobustDecomposition(unittest.TestCase):
    """Test robust matrix decomposition."""

    def setUp(self):
        """Set up test fixtures."""
        self.decomp = RobustDecomposition()

    def test_svd_fallback(self):
        """Test SVD with fallback."""
        # Create test matrix
        A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        U, s, Vt = self.decomp.svd(A)

        # Verify SVD properties
        self.assertTrue(jnp.all(jnp.isfinite(U)))
        self.assertTrue(jnp.all(jnp.isfinite(s)))
        self.assertTrue(jnp.all(jnp.isfinite(Vt)))
        self.assertTrue(jnp.all(s >= 0))  # Singular values non-negative

        # Verify reconstruction
        A_reconstructed = U @ jnp.diag(s) @ Vt
        self.assertTrue(jnp.allclose(A, A_reconstructed, atol=1e-10))

    def test_qr_fallback(self):
        """Test QR decomposition with fallback."""
        # Create test matrix
        A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        Q, R = self.decomp.qr(A)

        # Verify QR properties
        self.assertTrue(jnp.all(jnp.isfinite(Q)))
        self.assertTrue(jnp.all(jnp.isfinite(R)))

        # Q should be orthogonal
        QtQ = Q.T @ Q
        self.assertTrue(jnp.allclose(QtQ, jnp.eye(2), atol=1e-10))

        # Verify reconstruction
        A_reconstructed = Q @ R
        self.assertTrue(jnp.allclose(A, A_reconstructed, atol=1e-10))

    def test_cholesky_fallback(self):
        """Test Cholesky decomposition with fallback."""
        # Create positive definite matrix
        A = jnp.array([[4.0, 2.0], [2.0, 3.0]])

        L = self.decomp.cholesky(A)

        # Verify Cholesky properties
        self.assertTrue(jnp.all(jnp.isfinite(L)))

        # Verify reconstruction
        A_reconstructed = L @ L.T
        self.assertTrue(jnp.allclose(A, A_reconstructed, atol=1e-10))


class TestIntegration(unittest.TestCase):
    """Test integration of all stability features."""

    def test_curve_fit_with_stability(self):
        """Test curve_fit with all stability features enabled."""
        from nlsq import curve_fit

        # Create problematic data with outliers and noise
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = 2.5 * np.exp(-0.5 * x) + 1.0
        noise = np.random.normal(0, 0.1, 100)
        y = y_true + noise

        # Add outliers
        outlier_indices = [10, 30, 50, 70]
        for idx in outlier_indices:
            y[idx] += 5 * np.random.randn()

        # Fit with stability features
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # This should succeed despite problematic data
        popt, _pcov = curve_fit(
            model, x, y, p0=[3.0, 0.3, 0.5], bounds=([0, 0, 0], [10, 5, 5])
        )

        # Check that parameters are reasonable
        self.assertTrue(np.all(np.isfinite(popt)))
        self.assertTrue(np.all(np.isfinite(_pcov)))
        self.assertAlmostEqual(popt[0], 2.5, delta=0.5)  # Close to true value
        self.assertAlmostEqual(
            popt[1], 0.5, delta=0.25
        )  # Increased tolerance due to outliers
        self.assertAlmostEqual(popt[2], 1.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
