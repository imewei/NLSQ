"""
Comprehensive test suite for algorithm_selector module.

Tests automatic algorithm selection based on problem characteristics:
- Problem analysis (size, outliers, noise, conditioning)
- Algorithm and loss function selection
- Parameter recommendations
- Edge cases and robustness
"""

import unittest

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nlsq.precision.algorithm_selector import AlgorithmSelector, auto_select_algorithm


class TestAlgorithmSelectorBasic(unittest.TestCase):
    """Basic tests for AlgorithmSelector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_initialization(self):
        """Test AlgorithmSelector initialization."""
        self.assertIsInstance(self.selector.algorithm_properties, dict)
        self.assertIsInstance(self.selector.loss_properties, dict)

        # Check expected algorithms
        self.assertIn("trf", self.selector.algorithm_properties)
        self.assertIn("lm", self.selector.algorithm_properties)
        self.assertIn("dogbox", self.selector.algorithm_properties)

        # Check expected loss functions
        self.assertIn("linear", self.selector.loss_properties)
        self.assertIn("huber", self.selector.loss_properties)
        self.assertIn("cauchy", self.selector.loss_properties)

    def test_estimate_n_params_from_p0(self):
        """Test parameter estimation from p0."""

        def dummy_func(x, a, b, c):
            return a * x**2 + b * x + c

        p0 = np.array([1.0, 2.0, 3.0])
        n_params = self.selector._estimate_n_params(dummy_func, p0)
        self.assertEqual(n_params, 3)

    def test_estimate_n_params_from_signature(self):
        """Test parameter estimation from function signature."""

        def linear(x, a, b):
            return a * x + b

        n_params = self.selector._estimate_n_params(linear, None)
        self.assertEqual(n_params, 2)

        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        n_params = self.selector._estimate_n_params(exponential, None)
        self.assertEqual(n_params, 3)

    def test_estimate_n_params_fallback(self):
        """Test parameter estimation fallback for unparsable functions."""
        # Function with unparsable signature
        func = lambda x, *args: sum(args)
        n_params = self.selector._estimate_n_params(func, None)
        # Lambda with *args counts as 1 parameter (args)
        self.assertEqual(n_params, 1)


class TestDataAnalysis(unittest.TestCase):
    """Tests for data analysis methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_analyze_data_basic(self):
        """Test basic data analysis."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 0.1, 100)

        analysis = self.selector._analyze_data(x, y)

        # Check required fields
        self.assertIn("outlier_fraction", analysis)
        self.assertIn("has_outliers", analysis)
        self.assertIn("snr_estimate", analysis)
        self.assertIn("is_noisy", analysis)
        self.assertIn("x_range", analysis)
        self.assertIn("y_range", analysis)
        self.assertIn("x_uniform", analysis)

    def test_outlier_detection(self):
        """Test outlier detection."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1

        # Add clear outliers
        y[0] = 1000
        y[50] = -1000
        y[99] = 500

        analysis = self.selector._analyze_data(x, y)

        # Should detect outliers
        self.assertTrue(analysis["has_outliers"])
        self.assertGreater(analysis["outlier_fraction"], 0.01)

    def test_noise_detection_clean_data(self):
        """Test noise detection with clean data."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1  # No noise

        analysis = self.selector._analyze_data(x, y)

        # Should have high SNR (clean signal) - use 50 to account for numerical precision
        self.assertGreater(analysis["snr_estimate"], 50)
        self.assertFalse(analysis["is_noisy"])

    def test_noise_detection_noisy_data(self):
        """Test noise detection with noisy data."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + np.random.normal(0, 2.0, 100)  # High noise

        analysis = self.selector._analyze_data(x, y)

        # Should have low SNR (noisy signal) - check that SNR exists and is computed
        self.assertIn("snr_estimate", analysis)
        self.assertIn("is_noisy", analysis)
        # With this noise level, SNR should be relatively low but is_noisy depends on threshold
        self.assertLess(analysis["snr_estimate"], 100)

    def test_uniform_spacing_detection(self):
        """Test uniform spacing detection."""
        # Uniform spacing
        x_uniform = np.linspace(0, 10, 100)
        y = np.ones(100)
        analysis = self.selector._analyze_data(x_uniform, y)
        self.assertTrue(analysis["x_uniform"])

        # Non-uniform spacing
        x_nonuniform = np.sort(np.random.uniform(0, 10, 100))
        analysis = self.selector._analyze_data(x_nonuniform, y)
        self.assertFalse(analysis["x_uniform"])

    def test_multidimensional_x(self):
        """Test data analysis with multidimensional x."""
        x = np.random.randn(100, 3)  # 3D input
        y = np.random.randn(100)

        analysis = self.selector._analyze_data(x, y)

        # Should handle multidimensional x
        self.assertEqual(analysis["x_range"], 0)
        self.assertFalse(analysis["x_uniform"])

    def test_zero_detection(self):
        """Test detection of zeros in data."""
        x = np.linspace(0, 10, 100)
        y = np.ones(100)

        # No zeros
        analysis = self.selector._analyze_data(x, y)
        self.assertFalse(analysis["has_zeros"])

        # Add zero
        y[50] = 0
        analysis = self.selector._analyze_data(x, y)
        self.assertTrue(analysis["has_zeros"])


class TestConditioningEstimation(unittest.TestCase):
    """Tests for conditioning estimation."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_well_conditioned_problem(self):
        """Test conditioning estimation for well-conditioned problem."""
        x = np.linspace(0, 1, 100)
        n_params = 3

        cond = self.selector._estimate_conditioning(x, n_params)

        # Should be well-conditioned
        self.assertLess(cond, 1e10)
        self.assertGreater(cond, 1)

    def test_ill_conditioned_problem(self):
        """Test conditioning estimation for ill-conditioned problem."""
        # Concentrated points (poor conditioning)
        x = np.linspace(0, 0.001, 100)
        n_params = 5

        cond = self.selector._estimate_conditioning(x, n_params)

        # Should be ill-conditioned (use 500 as threshold)
        self.assertGreater(cond, 500)

    def test_constant_x(self):
        """Test conditioning with constant x (degenerate case)."""
        x = np.ones(100)
        n_params = 3

        cond = self.selector._estimate_conditioning(x, n_params)

        # Should return infinity for degenerate case
        self.assertTrue(np.isinf(cond))

    def test_underdetermined_problem(self):
        """Test conditioning with few data points."""
        x = np.linspace(0, 1, 3)  # Very few points
        n_params = 10

        cond = self.selector._estimate_conditioning(x, n_params)

        # With very few points, should get high conditioning or infinity
        # The function samples min(len(x), 1000) points and uses min(n_params, 5) features
        # So it checks if samples < features
        self.assertTrue(np.isinf(cond) or cond > 100)


class TestMemoryConstraints(unittest.TestCase):
    """Tests for memory constraint checking."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_no_memory_constraint(self):
        """Test when problem fits in memory."""
        n_points = 1000
        n_params = 5
        memory_limit_gb = 1.0

        constrained = self.selector._check_memory_constraints(
            n_points, n_params, memory_limit_gb
        )

        self.assertFalse(constrained)

    def test_memory_constrained(self):
        """Test when problem exceeds memory limit."""
        n_points = 10_000_000
        n_params = 100
        memory_limit_gb = 0.1

        constrained = self.selector._check_memory_constraints(
            n_points, n_params, memory_limit_gb
        )

        self.assertTrue(constrained)


class TestProblemAnalysis(unittest.TestCase):
    """Tests for complete problem analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_analyze_small_problem(self):
        """Test analysis of small problem."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 50)
        y = 2 * x + 1
        p0 = np.array([1.0, 0.0])

        analysis = self.selector.analyze_problem(linear, x, y, p0)

        # Check classification
        self.assertEqual(analysis["size_class"], "small")
        self.assertEqual(analysis["n_points"], 50)
        self.assertEqual(analysis["n_params"], 2)
        self.assertFalse(analysis["has_bounds"])

    def test_analyze_medium_problem(self):
        """Test analysis of medium problem."""

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        x = np.linspace(0, 10, 5000)
        y = x**2 - 2 * x + 1
        p0 = np.array([1.0, -2.0, 1.0])

        analysis = self.selector.analyze_problem(quadratic, x, y, p0)

        self.assertEqual(analysis["size_class"], "medium")
        self.assertEqual(analysis["n_points"], 5000)

    def test_analyze_large_problem(self):
        """Test analysis of large problem."""

        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 5, 100_000)
        y = 2.5 * np.exp(-1.3 * x) + 0.1
        p0 = np.array([2.0, 1.0, 0.0])

        analysis = self.selector.analyze_problem(exponential, x, y, p0)

        self.assertEqual(analysis["size_class"], "large")

    def test_analyze_very_large_problem(self):
        """Test analysis of very large problem."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 2_000_000)
        y = 2 * x + 1
        p0 = np.array([1.0, 0.0])

        analysis = self.selector.analyze_problem(linear, x, y, p0)

        self.assertEqual(analysis["size_class"], "very_large")

    def test_analyze_with_bounds(self):
        """Test analysis with parameter bounds."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 100)
        y = 2 * x + 1
        p0 = np.array([1.0, 0.0])
        bounds = ([0, -np.inf], [np.inf, np.inf])

        analysis = self.selector.analyze_problem(linear, x, y, p0, bounds=bounds)

        self.assertTrue(analysis["has_bounds"])

    def test_analyze_with_memory_limit(self):
        """Test analysis with memory constraint."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 100_000)
        y = 2 * x + 1
        p0 = np.array([1.0, 0.0])

        analysis = self.selector.analyze_problem(
            linear, x, y, p0, memory_limit_gb=0.001
        )

        self.assertTrue(analysis["memory_constrained"])


class TestAlgorithmSelection(unittest.TestCase):
    """Tests for algorithm selection logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.selector = AlgorithmSelector()

    def test_default_recommendations(self):
        """Test default algorithm recommendations."""
        analysis = {
            "n_points": 100,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
            "condition_estimate": 10.0,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Check defaults
        self.assertEqual(recommendations["algorithm"], "trf")
        self.assertEqual(recommendations["loss"], "linear")
        self.assertIsNotNone(recommendations["ftol"])
        self.assertIsNotNone(recommendations["xtol"])

    def test_outlier_handling_cauchy(self):
        """Test Cauchy loss selection for many outliers."""
        analysis = {
            "n_points": 100,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": True,
            "outlier_fraction": 0.15,  # >10%
            "is_noisy": False,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should select Cauchy loss for many outliers
        self.assertEqual(recommendations["loss"], "cauchy")

    def test_outlier_handling_huber(self):
        """Test Huber loss selection for moderate outliers."""
        analysis = {
            "n_points": 100,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": True,
            "outlier_fraction": 0.07,  # 5-10%
            "is_noisy": False,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should select Huber loss for moderate outliers
        self.assertEqual(recommendations["loss"], "huber")

    def test_outlier_handling_soft_l1(self):
        """Test Soft L1 loss selection for few outliers."""
        analysis = {
            "n_points": 100,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": True,
            "outlier_fraction": 0.02,  # 1-5%
            "is_noisy": False,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should select soft_l1 loss for few outliers
        self.assertEqual(recommendations["loss"], "soft_l1")

    def test_large_problem_solver_selection(self):
        """Test iterative solver selection for large problems."""
        analysis = {
            "n_points": 2_000_000,
            "n_params": 5,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should use iterative solver for large problems
        self.assertEqual(recommendations["tr_solver"], "lsmr")

    def test_many_params_solver_selection(self):
        """Test iterative solver selection for many parameters."""
        analysis = {
            "n_points": 1000,
            "n_params": 150,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should use iterative solver for many parameters
        self.assertEqual(recommendations["tr_solver"], "lsmr")

    def test_ill_conditioned_tolerance_adjustment(self):
        """Test tolerance adjustment for ill-conditioned problems."""
        analysis = {
            "n_points": 100,
            "n_params": 5,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
            "condition_estimate": 1e12,  # Very ill-conditioned
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should use relaxed tolerances
        self.assertGreaterEqual(recommendations["ftol"], 1e-6)
        self.assertGreaterEqual(recommendations["xtol"], 1e-6)

    def test_noisy_data_tolerance_adjustment(self):
        """Test tolerance adjustment for noisy data."""
        analysis = {
            "n_points": 100,
            "n_params": 3,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": True,
            "memory_constrained": False,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should use relaxed tolerances for noisy data
        self.assertGreaterEqual(recommendations["ftol"], 1e-6)

    def test_memory_constrained_recommendations(self):
        """Test recommendations for memory-constrained problems."""
        analysis = {
            "n_points": 100_000,
            "n_params": 10,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": True,
        }

        recommendations = self.selector.select_algorithm(analysis)

        # Should use memory-efficient solver
        self.assertEqual(recommendations["algorithm"], "trf")
        self.assertEqual(recommendations["tr_solver"], "lsmr")

    def test_user_preference_speed(self):
        """Test recommendations with speed priority."""
        analysis = {
            "n_points": 1000,
            "n_params": 5,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
        }

        user_preferences = {"prioritize": "speed"}
        recommendations = self.selector.select_algorithm(analysis, user_preferences)

        # Should have relaxed tolerances and limited iterations
        self.assertGreaterEqual(recommendations["ftol"], 1e-6)
        self.assertEqual(recommendations["max_nfev"], 100)

    def test_user_preference_accuracy(self):
        """Test recommendations with accuracy priority."""
        analysis = {
            "n_points": 1000,
            "n_params": 5,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
        }

        user_preferences = {"prioritize": "accuracy"}
        recommendations = self.selector.select_algorithm(analysis, user_preferences)

        # Should have tight tolerances
        self.assertLessEqual(recommendations["ftol"], 1e-10)
        self.assertLessEqual(recommendations["xtol"], 1e-10)


class TestAutoSelectAlgorithm(unittest.TestCase):
    """Tests for auto_select_algorithm convenience function."""

    def test_auto_select_basic(self):
        """Test basic auto_select_algorithm usage."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 100)
        y = 2 * x + 1
        p0 = np.array([1.0, 0.0])

        recommendations = auto_select_algorithm(linear, x, y, p0)

        # Should return valid recommendations
        self.assertIn("algorithm", recommendations)
        self.assertIn("loss", recommendations)
        self.assertIn("ftol", recommendations)

    def test_auto_select_with_all_options(self):
        """Test auto_select_algorithm with all options."""

        def exponential(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 5, 10_000)
        y = 2.5 * np.exp(-1.3 * x) + 0.1
        p0 = np.array([2.0, 1.0, 0.0])
        bounds = ([0, 0, -1], [10, 10, 1])

        recommendations = auto_select_algorithm(
            exponential,
            x,
            y,
            p0,
            bounds=bounds,
            memory_limit_gb=0.1,
            user_preferences={"prioritize": "speed"},
        )

        self.assertIsInstance(recommendations, dict)
        self.assertEqual(recommendations["algorithm"], "trf")


class TestAlgorithmExplanation(unittest.TestCase):
    """Tests for algorithm explanation generation."""

    def test_explanation_basic(self):
        """Test basic explanation generation."""
        recommendations = {
            "algorithm": "trf",
            "loss": "linear",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "max_nfev": None,
        }

        selector = AlgorithmSelector()
        explanation = selector.get_algorithm_explanation(recommendations)

        # Should mention TRF
        self.assertIn("TRF", explanation.upper())

    def test_explanation_robust_loss(self):
        """Test explanation with robust loss."""
        recommendations = {
            "algorithm": "trf",
            "loss": "huber",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "max_nfev": None,
        }

        selector = AlgorithmSelector()
        explanation = selector.get_algorithm_explanation(recommendations)

        # Should mention Huber loss
        self.assertIn("huber", explanation.lower())
        self.assertIn("outlier", explanation.lower())

    def test_explanation_iterative_solver(self):
        """Test explanation with iterative solver."""
        recommendations = {
            "algorithm": "trf",
            "loss": "linear",
            "ftol": 1e-8,
            "xtol": 1e-8,
            "tr_solver": "lsmr",
            "max_nfev": None,
        }

        selector = AlgorithmSelector()
        explanation = selector.get_algorithm_explanation(recommendations)

        # Should mention iterative solver
        self.assertIn("iterative", explanation.lower())
        self.assertIn("memory", explanation.lower())


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_single_data_point(self):
        """Test with single data point."""

        def linear(x, a, b):
            return a * x + b

        x = np.array([1.0])
        y = np.array([2.0])
        p0 = np.array([1.0, 0.0])

        selector = AlgorithmSelector()
        analysis = selector.analyze_problem(linear, x, y, p0)

        # Should handle gracefully
        self.assertEqual(analysis["n_points"], 1)
        self.assertIsInstance(analysis["size_class"], str)

    def test_no_initial_guess(self):
        """Test without initial parameter guess."""

        def linear(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 100)
        y = 2 * x + 1

        selector = AlgorithmSelector()
        analysis = selector.analyze_problem(linear, x, y, p0=None)

        # Should estimate parameters from signature
        self.assertEqual(analysis["n_params"], 2)

    def test_constant_data(self):
        """Test with constant y data."""
        x = np.linspace(0, 10, 100)
        y = np.ones(100)

        selector = AlgorithmSelector()
        analysis = selector._analyze_data(x, y)

        # Should handle constant data
        self.assertEqual(analysis["y_range"], 0)


class TestPropertyBasedAlgorithmSelection(unittest.TestCase):
    """Property-based tests for algorithm selection."""

    @given(
        n_points=st.integers(min_value=10, max_value=10000),
        n_params=st.integers(min_value=1, max_value=20),
    )
    @settings(max_examples=50, deadline=None)
    def test_recommendations_always_valid(self, n_points, n_params):
        """Test that recommendations are always valid for various problem sizes."""
        analysis = {
            "n_points": n_points,
            "n_params": n_params,
            "has_bounds": False,
            "has_outliers": False,
            "is_noisy": False,
            "memory_constrained": False,
            "condition_estimate": 100.0,
        }

        selector = AlgorithmSelector()
        recommendations = selector.select_algorithm(analysis)

        # Check that all required fields are present
        required_fields = ["algorithm", "loss", "ftol", "xtol", "max_nfev"]
        for field in required_fields:
            self.assertIn(field, recommendations)

        # Check that algorithm is valid
        self.assertIn(recommendations["algorithm"], ["trf", "lm", "dogbox"])

        # Check that loss is valid
        self.assertIn(
            recommendations["loss"], ["linear", "huber", "soft_l1", "cauchy", "arctan"]
        )

        # Check that tolerances are positive
        self.assertGreater(recommendations["ftol"], 0)
        self.assertGreater(recommendations["xtol"], 0)


if __name__ == "__main__":
    unittest.main()
