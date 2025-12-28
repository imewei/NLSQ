"""Comprehensive tests for nlsq.__init__ module.

This test suite covers:
- Module imports and __all__ exports
- curve_fit_large function with various parameters
- Edge cases and error handling
- Memory limit detection and fallback
"""

import unittest
from unittest.mock import patch

import numpy as np
import pytest

import nlsq
from nlsq import (
    curve_fit_large,
)


class TestModuleImports(unittest.TestCase):
    """Tests for module imports and exports."""

    def test_version_import(self):
        """Test that __version__ is accessible."""
        self.assertTrue(hasattr(nlsq, "__version__"))
        self.assertIsInstance(nlsq.__version__, str)

    def test_all_exports_accessible(self):
        """Test that all exported items are accessible."""
        for item in nlsq.__all__:
            self.assertTrue(hasattr(nlsq, item), f"{item} not found in nlsq")

    def test_main_api_functions(self):
        """Test that main API functions are accessible."""
        self.assertTrue(callable(nlsq.curve_fit))
        self.assertTrue(callable(nlsq.curve_fit_large))

    def test_configuration_classes(self):
        """Test that configuration classes are accessible."""
        self.assertTrue(hasattr(nlsq, "MemoryConfig"))
        self.assertTrue(hasattr(nlsq, "LargeDatasetConfig"))

    def test_utility_functions(self):
        """Test that utility functions are accessible."""
        self.assertTrue(callable(nlsq.configure_for_large_datasets))
        self.assertTrue(callable(nlsq.set_memory_limits))


class TestCurveFitLargeBasic(unittest.TestCase):
    """Tests for basic curve_fit_large functionality."""

    def setUp(self):
        """Set up test fixtures."""

        # Simple linear function
        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_small_dataset_uses_standard_curve_fit(self):
        """Test that small datasets use standard curve_fit."""
        xdata = np.linspace(0, 10, 100)
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            auto_size_detection=True,
            size_threshold=1000,
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)
        np.testing.assert_allclose(popt, [2.0, 1.0], atol=0.01)

    def test_medium_dataset_with_chunking(self):
        """Test medium dataset with chunked processing."""
        np.random.seed(42)
        xdata = np.linspace(0, 10, 15000)  # Larger dataset to avoid chunk size issues
        ydata = 2.0 * xdata + 1.0 + 0.01 * np.random.randn(len(xdata))

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            auto_size_detection=True,
            size_threshold=1000,  # Force large dataset processing
            memory_limit_gb=4.0,
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)
        # Should still fit reasonably well
        np.testing.assert_allclose(popt, [2.0, 1.0], atol=0.1)

    def test_with_initial_guess(self):
        """Test curve_fit_large with initial parameter guess."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func, xdata, ydata, p0=[1.0, 0.0], size_threshold=10
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)

    def test_with_bounds(self):
        """Test curve_fit_large with parameter bounds."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        bounds = ([0.0, -10.0], [10.0, 10.0])

        popt, _pcov = curve_fit_large(
            self.linear_func, xdata, ydata, bounds=bounds, size_threshold=10
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)
        # Popt should be within bounds
        self.assertTrue(np.all(popt >= bounds[0]))
        self.assertTrue(np.all(popt <= bounds[1]))

    def test_with_sigma(self):
        """Test curve_fit_large with sigma (uncertainties)."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0
        sigma = np.ones_like(ydata) * 0.1

        popt, _pcov = curve_fit_large(
            self.linear_func, xdata, ydata, sigma=sigma, size_threshold=10
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)

    def test_with_method_specified(self):
        """Test curve_fit_large with explicit method."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func, xdata, ydata, method="trf", size_threshold=10
        )

        self.assertIsInstance(popt, np.ndarray)
        self.assertEqual(len(popt), 2)


class TestCurveFitLargeEdgeCases(unittest.TestCase):
    """Tests for edge cases in curve_fit_large."""

    def setUp(self):
        """Set up test fixtures."""

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_empty_xdata_error(self):
        """Test error when xdata is empty."""
        xdata = np.array([])
        ydata = np.array([])

        with self.assertRaises(ValueError) as ctx:
            curve_fit_large(self.linear_func, xdata, ydata)

        self.assertIn("xdata", str(ctx.exception).lower())
        self.assertIn("empty", str(ctx.exception).lower())

    def test_empty_ydata_error(self):
        """Test error when ydata is empty."""
        xdata = np.array([1.0, 2.0])
        ydata = np.array([])

        with self.assertRaises(ValueError) as ctx:
            curve_fit_large(self.linear_func, xdata, ydata)

        self.assertIn("ydata", str(ctx.exception).lower())
        self.assertIn("empty", str(ctx.exception).lower())

    def test_length_mismatch_error(self):
        """Test error when xdata and ydata have different lengths."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0])

        with self.assertRaises(ValueError) as ctx:
            curve_fit_large(self.linear_func, xdata, ydata)

        self.assertIn("same length", str(ctx.exception).lower())

    def test_insufficient_points_error(self):
        """Test error when less than 2 data points."""
        xdata = np.array([1.0])
        ydata = np.array([2.0])

        with self.assertRaises(ValueError) as ctx:
            curve_fit_large(self.linear_func, xdata, ydata)

        self.assertIn("at least 2 data points", str(ctx.exception).lower())

    def test_no_progress_bar(self):
        """Test with show_progress=False (default)."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            show_progress=False,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_with_progress_bar(self):
        """Test with show_progress=True."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        # This may or may not show progress depending on implementation
        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            show_progress=True,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)


class TestCurveFitLargeMemory(unittest.TestCase):
    """Tests for memory management in curve_fit_large."""

    def setUp(self):
        """Set up test fixtures."""

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_explicit_memory_limit(self):
        """Test with explicitly set memory limit."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            memory_limit_gb=2.0,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_memory_limit_from_psutil(self):
        """Test auto-detection of memory limit from psutil."""
        # This test just verifies that auto-detection works when psutil is available
        # (psutil is imported inside curve_fit_large, so we can't easily mock it)
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        # If psutil is available, this will use it; otherwise, fallback to default
        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            memory_limit_gb=None,  # Should auto-detect
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_memory_limit_fallback_without_psutil(self):
        """Test fallback when psutil is not available."""
        # This test relies on the ImportError handling in curve_fit_large
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        # Patch psutil to raise ImportError
        with patch.dict("sys.modules", {"psutil": None}):
            popt, _pcov = curve_fit_large(
                self.linear_func,
                xdata,
                ydata,
                memory_limit_gb=None,
                size_threshold=10,
            )

            self.assertIsInstance(popt, np.ndarray)

    def test_chunk_size_override(self):
        """Test with manually specified chunk size."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            chunk_size=5000,  # Reasonable chunk size
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)


class TestCurveFitLargeSamplingRemoved(unittest.TestCase):
    """Tests that removed sampling parameters raise TypeError."""

    def setUp(self):
        """Set up test fixtures."""

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_sampling_params_rejected(self):
        """Test that removed sampling parameters raise TypeError."""
        xdata = np.linspace(0, 10, 15000)
        ydata = 2.0 * xdata + 1.0

        with self.assertRaises(TypeError):
            curve_fit_large(
                self.linear_func,
                xdata,
                ydata,
                enable_sampling=True,  # Removed parameter
                size_threshold=10,
            )

        with self.assertRaises(TypeError):
            curve_fit_large(
                self.linear_func,
                xdata,
                ydata,
                sampling_threshold=50,  # Removed parameter
                size_threshold=10,
            )

        with self.assertRaises(TypeError):
            curve_fit_large(
                self.linear_func,
                xdata,
                ydata,
                max_sampled_size=30,  # Removed parameter
                size_threshold=10,
            )


class TestCurveFitLargeOptionalParameters(unittest.TestCase):
    """Tests for optional parameters in curve_fit_large."""

    def setUp(self):
        """Set up test fixtures."""

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_absolute_sigma_false(self):
        """Test with absolute_sigma=False."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0
        sigma = np.ones_like(ydata)

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            sigma=sigma,
            absolute_sigma=False,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_check_finite_false(self):
        """Test with check_finite=False."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            check_finite=False,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_additional_kwargs(self):
        """Test passing additional kwargs (ftol, xtol, etc.)."""
        xdata = np.linspace(0, 10, 15000)  # Large dataset
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
            max_nfev=1000,
            size_threshold=10,
        )

        self.assertIsInstance(popt, np.ndarray)


class TestCurveFitLargeAutoSizeDetection(unittest.TestCase):
    """Tests for auto size detection logic."""

    def setUp(self):
        """Set up test fixtures."""

        def linear(x, a, b):
            return a * x + b

        self.linear_func = linear

    def test_auto_size_detection_enabled(self):
        """Test with auto_size_detection=True (default)."""
        # Small dataset should use regular curve_fit
        xdata_small = np.linspace(0, 10, 100)
        ydata_small = 2.0 * xdata_small + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata_small,
            ydata_small,
            auto_size_detection=True,
            size_threshold=1000,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_auto_size_detection_disabled(self):
        """Test with auto_size_detection=False."""
        # Even small dataset should use large dataset processing
        xdata = np.linspace(0, 10, 15000)  # Large dataset to avoid chunk size issues
        ydata = 2.0 * xdata + 1.0

        popt, _pcov = curve_fit_large(
            self.linear_func,
            xdata,
            ydata,
            auto_size_detection=False,
            memory_limit_gb=4.0,
        )

        self.assertIsInstance(popt, np.ndarray)

    def test_custom_size_threshold(self):
        """Test with custom size threshold."""
        xdata = np.linspace(0, 10, 500)
        ydata = 2.0 * xdata + 1.0

        # With threshold=1000, should use regular curve_fit
        popt, _pcov = curve_fit_large(
            self.linear_func, xdata, ydata, size_threshold=1000
        )

        self.assertIsInstance(popt, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
