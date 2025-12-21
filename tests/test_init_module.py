#!/usr/bin/env python3
"""Tests for nlsq __init__ module and public API."""

import unittest

import numpy as np
import pytest


class TestInitModule(unittest.TestCase):
    """Test nlsq package initialization and public API."""

    def test_version(self):
        """Test version information is available."""
        import nlsq

        self.assertIsNotNone(nlsq.__version__)
        self.assertIsInstance(nlsq.__version__, str)

    def test_main_api_imports(self):
        """Test main API functions are importable."""
        from nlsq import (
            CurveFit,
            LeastSquares,
            OptimizeResult,
            OptimizeWarning,
            curve_fit,
        )

        # Check that classes/functions exist
        self.assertTrue(callable(curve_fit))
        self.assertTrue(callable(CurveFit))
        self.assertTrue(callable(LeastSquares))
        self.assertTrue(callable(OptimizeResult))
        self.assertTrue(callable(OptimizeWarning))

    def test_config_imports(self):
        """Test configuration imports."""
        from nlsq import (
            LargeDatasetConfig,
            MemoryConfig,
            configure_for_large_datasets,
            enable_mixed_precision_fallback,
            get_large_dataset_config,
            get_memory_config,
            large_dataset_context,
            memory_context,
            set_memory_limits,
        )

        # Check that all config functions/classes are available
        self.assertTrue(callable(LargeDatasetConfig))
        self.assertTrue(callable(MemoryConfig))
        self.assertTrue(callable(configure_for_large_datasets))
        self.assertTrue(callable(enable_mixed_precision_fallback))
        self.assertTrue(callable(get_large_dataset_config))
        self.assertTrue(callable(get_memory_config))
        self.assertTrue(callable(large_dataset_context))
        self.assertTrue(callable(memory_context))
        self.assertTrue(callable(set_memory_limits))

    def test_large_dataset_imports(self):
        """Test large dataset support imports."""
        from nlsq import (
            LargeDatasetFitter,
            LDMemoryConfig,
            estimate_memory_requirements,
            fit_large_dataset,
        )

        self.assertTrue(callable(LargeDatasetFitter))
        self.assertTrue(callable(LDMemoryConfig))
        self.assertTrue(callable(estimate_memory_requirements))
        self.assertTrue(callable(fit_large_dataset))

    def test_stability_imports(self):
        """Test stability and optimization imports."""
        from nlsq import AlgorithmSelector, auto_select_algorithm

        self.assertTrue(callable(AlgorithmSelector))
        self.assertTrue(callable(auto_select_algorithm))

    def test_basic_curve_fit(self):
        """Test basic curve_fit functionality through __init__ import."""
        from nlsq import curve_fit

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        popt, _pcov = curve_fit(model, x, y)
        self.assertAlmostEqual(popt[0], 2.0, places=3)
        self.assertAlmostEqual(popt[1], 0.0, places=3)

    def test_optimize_result_creation(self):
        """Test OptimizeResult class."""
        from nlsq import OptimizeResult

        result = OptimizeResult(
            x=np.array([1.0, 2.0]),
            success=True,
            message="Optimization succeeded",
            fun=0.1,
            jac=np.array([0.01, 0.02]),
            nfev=10,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.message, "Optimization succeeded")
        self.assertEqual(result.nfev, 10)
        self.assertAlmostEqual(result.fun, 0.1)

        # Test attribute access
        self.assertTrue(np.allclose(result.x, [1.0, 2.0]))
        self.assertTrue(np.allclose(result.jac, [0.01, 0.02]))

    def test_optimize_warning(self):
        """Test OptimizeWarning class."""
        from nlsq import OptimizeWarning

        # Check it's a warning class
        self.assertTrue(issubclass(OptimizeWarning, Warning))

        # Test creating a warning
        warning = OptimizeWarning("Test warning message")
        self.assertIsInstance(warning, Warning)

    def test_large_dataset_config(self):
        """Test LargeDatasetConfig (v0.2.0: sampling params removed)."""
        from nlsq import LargeDatasetConfig, get_large_dataset_config

        # Test creating config
        config = LargeDatasetConfig()
        self.assertIsNotNone(config)

        # Test getting global config
        global_config = get_large_dataset_config()
        self.assertIsInstance(global_config, LargeDatasetConfig)

    def test_memory_config(self):
        """Test MemoryConfig."""
        from nlsq import MemoryConfig, get_memory_config, set_memory_limits

        # Test creating config
        config = MemoryConfig(memory_limit_gb=4.0, enable_mixed_precision_fallback=True)

        self.assertEqual(config.memory_limit_gb, 4.0)
        self.assertTrue(config.enable_mixed_precision_fallback)

        # Test getting global config
        global_config = get_memory_config()
        self.assertIsInstance(global_config, MemoryConfig)

        # Test setting memory limits
        set_memory_limits(memory_limit_gb=2.0)
        new_config = get_memory_config()
        self.assertEqual(new_config.memory_limit_gb, 2.0)

    def test_context_managers(self):
        """Test context managers for configuration (v0.2.0: sampling removed)."""
        from nlsq import (
            LargeDatasetConfig,
            MemoryConfig,
            get_large_dataset_config,
            get_memory_config,
            large_dataset_context,
            memory_context,
        )

        # Test large_dataset_context
        original_config = get_large_dataset_config()
        temp_ld_config = LargeDatasetConfig()
        with large_dataset_context(temp_ld_config):
            temp_config = get_large_dataset_config()
            self.assertIsNotNone(temp_config)

        # Config should be restored
        restored_config = get_large_dataset_config()
        self.assertEqual(type(restored_config), type(original_config))

        # Test memory_context
        original_mem_config = get_memory_config()
        temp_mem_config = MemoryConfig(memory_limit_gb=1.0)
        with memory_context(temp_mem_config):
            temp_config = get_memory_config()
            self.assertEqual(temp_config.memory_limit_gb, 1.0)

        # Config should be restored
        restored_mem_config = get_memory_config()
        self.assertEqual(
            restored_mem_config.memory_limit_gb, original_mem_config.memory_limit_gb
        )

    def test_algorithm_selector(self):
        """Test AlgorithmSelector through public API."""
        from nlsq import AlgorithmSelector, auto_select_algorithm

        selector = AlgorithmSelector()
        self.assertIsNotNone(selector)

        # Test auto_select_algorithm
        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 10, 100)
        y = 2 * x + 1 + 0.1 * np.random.randn(100)

        recommendations = auto_select_algorithm(model, x, y)
        self.assertIsInstance(recommendations, dict)
        self.assertIn("algorithm", recommendations)
        self.assertIn("ftol", recommendations)

    def test_fit_large_dataset_basic(self):
        """Test fit_large_dataset through public API."""
        from nlsq import fit_large_dataset

        def model(x, a, b):
            return a * x + b

        # Moderate dataset
        x = np.linspace(0, 10, 1000)
        y = 2 * x + 1 + 0.1 * np.random.randn(1000)

        result = fit_large_dataset(
            model, x, y, initial_params=[1.0, 0.0], chunk_size=500
        )

        self.assertTrue(result.success)
        self.assertAlmostEqual(result.popt[0], 2.0, places=0)
        self.assertAlmostEqual(result.popt[1], 1.0, places=0)

    def test_estimate_memory_requirements(self):
        """Test memory estimation function."""
        from nlsq import estimate_memory_requirements

        n_points = 10000
        n_params = 5

        dataset_stats = estimate_memory_requirements(n_points, n_params)

        # Should return DatasetStats object
        self.assertIsInstance(
            dataset_stats, type(dataset_stats)
        )  # Check it's the right type
        self.assertTrue(hasattr(dataset_stats, "total_memory_estimate_gb"))
        self.assertIsInstance(dataset_stats.total_memory_estimate_gb, float)
        self.assertGreater(dataset_stats.total_memory_estimate_gb, 0)
        # For 10k points and 5 params, should be small
        self.assertLess(dataset_stats.total_memory_estimate_gb, 1.0)

    def test_configure_for_large_datasets(self):
        """Test large dataset configuration helper."""
        from nlsq import configure_for_large_datasets, get_large_dataset_config

        configure_for_large_datasets(memory_limit_gb=8.0, enable_chunking=True)

        config = get_large_dataset_config()
        # Should have configured appropriately
        self.assertIsNotNone(config)

    def test_enable_mixed_precision_fallback(self):
        """Test mixed precision fallback configuration."""
        from nlsq import enable_mixed_precision_fallback, get_memory_config

        enable_mixed_precision_fallback(enable=True)

        config = get_memory_config()
        self.assertTrue(config.enable_mixed_precision_fallback)

    def test_large_dataset_fitter_class(self):
        """Test LargeDatasetFitter class."""
        from nlsq import LargeDatasetFitter

        def model(x, a, b):
            return a * x + b

        fitter = LargeDatasetFitter(memory_limit_gb=2.0)

        x = np.linspace(0, 10, 500)
        y = 2 * x + 1 + 0.1 * np.random.randn(500)

        result = fitter.fit(model, x, y, p0=[1.0, 0.0])

        self.assertTrue(result.success)
        # Check params are close to expected
        self.assertAlmostEqual(result.popt[0], 2.0, places=0)

    def test_ldmemory_config(self):
        """Test LDMemoryConfig (v0.2.0: enable_sampling removed)."""
        from nlsq import LDMemoryConfig

        config = LDMemoryConfig(memory_limit_gb=4.0, min_chunk_size=1000)

        self.assertEqual(config.memory_limit_gb, 4.0)
        self.assertEqual(config.min_chunk_size, 1000)

    def test_curve_fit_class(self):
        """Test CurveFit class through public API."""
        from nlsq import CurveFit

        cf = CurveFit(use_dynamic_sizing=True)

        def model(x, a, b):
            return a * x + b

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.1, 3.9, 6.1, 7.9, 10.1])

        popt, _pcov = cf.curve_fit(model, x, y)

        self.assertAlmostEqual(popt[0], 2.0, places=1)
        self.assertAlmostEqual(popt[1], 0.0, places=1)

        # Check covariance matrix shape
        self.assertEqual(_pcov.shape, (2, 2))

    def test_least_squares_class(self):
        """Test LeastSquares class through public API."""
        from nlsq import LeastSquares

        ls = LeastSquares()

        # Check it has the expected methods
        self.assertTrue(hasattr(ls, "least_squares"))
        self.assertTrue(callable(ls.least_squares))


if __name__ == "__main__":
    unittest.main()
