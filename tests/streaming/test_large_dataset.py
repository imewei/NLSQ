"""
Comprehensive test suite for large dataset handling in NLSQ.

Tests cover memory estimation, chunking strategies, streaming optimization,
and optimized fitting for datasets >20M points.
"""

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np

from nlsq import (
    LargeDatasetFitter,
    curve_fit_large,
    estimate_memory_requirements,
    fit_large_dataset,
)
from nlsq.streaming.large_dataset import (
    DataChunker,
    DatasetStats,
    LDMemoryConfig,
    MemoryEstimator,
)


class TestMemoryEstimation(unittest.TestCase):
    """Test memory estimation for various dataset sizes."""

    def test_small_dataset_estimation(self):
        """Test memory estimation for small datasets (<1M points)."""
        stats = estimate_memory_requirements(100_000, 3)

        self.assertIsInstance(stats, DatasetStats)
        self.assertEqual(stats.n_points, 100_000)
        self.assertEqual(stats.n_params, 3)
        self.assertGreater(stats.total_memory_estimate_gb, 0)
        self.assertLess(stats.total_memory_estimate_gb, 1.0)  # Should be <1GB
        self.assertEqual(stats.n_chunks, 1)  # Should fit in single chunk

    def test_medium_dataset_estimation(self):
        """Test memory estimation for medium datasets (1M-10M points)."""
        stats = estimate_memory_requirements(5_000_000, 5)

        self.assertEqual(stats.n_points, 5_000_000)
        self.assertEqual(stats.n_params, 5)
        self.assertGreater(stats.total_memory_estimate_gb, 0.5)
        self.assertLess(stats.total_memory_estimate_gb, 10.0)
        self.assertGreaterEqual(stats.n_chunks, 1)

    def test_large_dataset_estimation_20M(self):
        """Test memory estimation for 20M point dataset."""
        stats = estimate_memory_requirements(20_000_000, 4)

        self.assertEqual(stats.n_points, 20_000_000)
        self.assertEqual(stats.n_params, 4)
        self.assertGreater(stats.total_memory_estimate_gb, 2.0)
        self.assertGreater(stats.n_chunks, 1)  # Should require chunking
        self.assertEqual(stats.recommended_chunk_size, 1_000_000)  # Default chunk size

    def test_very_large_dataset_estimation_50M(self):
        """Test memory estimation for 50M point dataset."""
        stats = estimate_memory_requirements(50_000_000, 6)

        self.assertEqual(stats.n_points, 50_000_000)
        self.assertEqual(stats.n_params, 6)
        self.assertGreater(stats.total_memory_estimate_gb, 5.0)
        self.assertEqual(stats.n_chunks, 50)  # 50M / 1M chunk size

    def test_extremely_large_dataset_estimation_100M(self):
        """Test memory estimation for 100M point dataset."""
        stats = estimate_memory_requirements(100_000_000, 10)

        self.assertEqual(stats.n_points, 100_000_000)
        self.assertEqual(stats.n_params, 10)
        self.assertGreater(stats.total_memory_estimate_gb, 10.0)
        self.assertEqual(stats.n_chunks, 100)  # 100M / 1M chunk size
        # Streaming optimization can handle unlimited data

    def test_billion_point_dataset_estimation(self):
        """Test memory estimation for 1B point dataset (extreme case)."""
        stats = estimate_memory_requirements(1_000_000_000, 5)

        self.assertEqual(stats.n_points, 1_000_000_000)
        self.assertEqual(stats.n_params, 5)
        self.assertGreater(stats.total_memory_estimate_gb, 100.0)
        self.assertEqual(stats.n_chunks, 1000)  # 1B / 1M chunk size
        # Streaming optimization handles unlimited data without subsampling


class TestLargeDatasetFitter(unittest.TestCase):
    """Test the LargeDatasetFitter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple exponential model for testing
        self.model = lambda x, a, b: a * jnp.exp(-b * x)
        self.true_params = [2.5, 1.3]
        np.random.seed(42)

    def test_fitter_initialization(self):
        """Test LargeDatasetFitter initialization with different configs."""
        # Default initialization
        fitter = LargeDatasetFitter()
        self.assertIsNotNone(fitter.config)
        self.assertEqual(fitter.config.memory_limit_gb, 8.0)  # Default

        # Custom memory limit
        fitter = LargeDatasetFitter(memory_limit_gb=4.0)
        self.assertEqual(fitter.config.memory_limit_gb, 4.0)

        # With config object
        config = LDMemoryConfig(memory_limit_gb=16.0, max_chunk_size=500000)
        fitter = LargeDatasetFitter(config=config)
        self.assertEqual(fitter.config.memory_limit_gb, 16.0)
        self.assertEqual(fitter.config.max_chunk_size, 500000)

    def test_memory_recommendations(self):
        """Test getting memory recommendations for different scenarios."""
        fitter = LargeDatasetFitter(memory_limit_gb=2.0)

        # Small dataset - should fit in memory
        recs = fitter.get_memory_recommendations(100_000, 3)
        self.assertEqual(recs["processing_strategy"], "single_chunk")
        self.assertEqual(recs["recommendations"]["n_chunks"], 1)

        # Large dataset - should require chunking
        recs = fitter.get_memory_recommendations(10_000_000, 3)
        self.assertEqual(recs["processing_strategy"], "chunked")
        self.assertGreater(recs["recommendations"]["n_chunks"], 1)

        # Very large dataset - will use chunked or streaming
        recs = fitter.get_memory_recommendations(100_000_000, 5)
        self.assertEqual(recs["processing_strategy"], "chunked")

    def test_fit_small_dataset(self):
        """Test fitting a small dataset that fits in memory."""
        n_points = 1000
        x = np.linspace(0, 4, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        fitter = LargeDatasetFitter(memory_limit_gb=1.0)
        result = fitter.fit(self.model, x, y, p0=[2.0, 1.0])

        self.assertTrue(result.success)
        self.assertEqual(len(result.popt), 2)
        np.testing.assert_allclose(result.popt, self.true_params, rtol=0.1)

    def test_fit_with_chunking(self):
        """Test fitting with forced chunking."""
        n_points = 10_000
        x = np.linspace(0, 4, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        # Force chunking with very small memory limit
        fitter = LargeDatasetFitter(
            memory_limit_gb=0.001
        )  # Very small to force chunking
        result = fitter.fit(self.model, x, y, p0=[2.0, 1.0])

        self.assertTrue(result.success)
        # Should have used multiple chunks (check if n_chunks is available)
        if hasattr(result, "n_chunks"):
            self.assertGreater(result.n_chunks, 1)
        np.testing.assert_allclose(result.popt, self.true_params, rtol=0.2)

    def test_fit_with_progress(self):
        """Test fitting with progress reporting."""
        n_points = 5000
        x = np.linspace(0, 4, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        fitter = LargeDatasetFitter(memory_limit_gb=0.01)  # Force chunking

        # Test that fit_with_progress works (progress goes via logger)
        result = fitter.fit_with_progress(self.model, x, y, p0=[2.0, 1.0])

        # Check that fitting succeeded
        self.assertTrue(result.success)
        # Progress is reported via logger, not print, so we just check the result


class TestCurveFitLarge(unittest.TestCase):
    """Test the curve_fit_large convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = lambda x, a, b, c: a * jnp.exp(-b * x) + c
        self.true_params = [3.0, 1.5, 0.5]
        np.random.seed(123)

    def test_curve_fit_large_basic(self):
        """Test basic functionality of curve_fit_large."""
        n_points = 5000
        x = np.linspace(0, 5, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        popt, _pcov = curve_fit_large(
            self.model, x, y, p0=[2.5, 1.0, 0.3], memory_limit_gb=2.0
        )

        self.assertEqual(len(popt), 3)
        self.assertEqual(_pcov.shape, (3, 3))
        np.testing.assert_allclose(popt, self.true_params, rtol=0.1)

    def test_curve_fit_large_with_bounds(self):
        """Test curve_fit_large with parameter bounds."""
        n_points = 3000
        x = np.linspace(0, 5, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        # Set bounds
        bounds = ([0, 0, 0], [10, 10, 2])

        popt, _pcov = curve_fit_large(
            self.model, x, y, p0=[2.5, 1.0, 0.3], bounds=bounds, memory_limit_gb=1.0
        )

        # Check bounds are respected
        self.assertTrue(all(popt >= bounds[0]))
        self.assertTrue(all(popt <= bounds[1]))
        np.testing.assert_allclose(popt, self.true_params, rtol=0.1)

    def test_curve_fit_large_20M_simulation(self):
        """Simulate fitting a 20M point dataset using chunking."""
        # For testing, we'll use a smaller dataset to represent the large dataset
        n_sample = 100_000  # Use 100K to simulate 20M dataset behavior
        x = np.linspace(0, 10, n_sample)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_sample)

        # Simulate large dataset behavior with small memory limit
        popt, _pcov = curve_fit_large(
            self.model,
            x,
            y,
            p0=[2.5, 1.0, 0.3],
            memory_limit_gb=0.01,  # Very small to force extreme chunking
            show_progress=False,
        )

        self.assertEqual(len(popt), 3)
        np.testing.assert_allclose(popt, self.true_params, rtol=0.15)


class TestDataChunker(unittest.TestCase):
    """Test the DataChunker class for handling data chunks."""

    def test_chunk_creation(self):
        """Test creating data chunks."""
        # Create test data
        x = np.arange(5000)
        y = np.arange(5000) * 2

        # DataChunker is a static class
        chunks = list(DataChunker.create_chunks(x, y, chunk_size=1000))

        self.assertEqual(len(chunks), 5)  # 5000 / 1000 = 5 chunks

        # Check first chunk - now returns 4 values including valid_length
        x_chunk, y_chunk, _idx, valid_length = chunks[0]
        self.assertEqual(len(x_chunk), 1024)  # Padded to power-of-2 bucket
        self.assertEqual(len(y_chunk), 1024)
        self.assertEqual(valid_length, 1000)  # Original length
        np.testing.assert_array_equal(x_chunk[:1000], np.arange(1000))

        # Check last chunk
        x_chunk, y_chunk, _idx, valid_length = chunks[-1]
        self.assertEqual(len(x_chunk), 1024)  # Padded to power-of-2 bucket
        self.assertEqual(valid_length, 1000)
        np.testing.assert_array_equal(x_chunk[:1000], np.arange(4000, 5000))

    def test_chunk_processing_uneven(self):
        """Test processing data that doesn't divide evenly.

        After the padding fix, all chunks (including the last one) should have
        uniform size to prevent JAX JIT recompilation. The last chunk is padded
        by repeating points cyclically.
        """
        # Create test data with uneven size
        x = np.arange(5500)  # Not evenly divisible by 1000
        y = np.arange(5500) * 2

        chunks = list(DataChunker.create_chunks(x, y, chunk_size=1000))

        self.assertEqual(len(chunks), 6)  # 5 full + 1 partial

        # Check all chunks have uniform size (padding fix for JIT recompilation)
        # Now returns 4 values: x, y, idx, valid_length
        x_chunk, y_chunk, _idx, valid_length = chunks[-1]
        self.assertEqual(
            len(x_chunk), 1024
        )  # Padded to power-of-2 bucket (1024 >= 500)
        self.assertEqual(valid_length, 500)  # Original 500 points

        # Verify padding: first 500 are original data, rest are cyclic repeat
        # Original data: x[5000:5500], padded with cyclic repeat to 1024
        expected_x_original = np.arange(5000, 5500)
        expected_x_padded = np.resize(expected_x_original, 1024)
        np.testing.assert_array_equal(x_chunk, expected_x_padded)

        # Verify y-data padding matches x-data pattern
        expected_y_original = np.arange(5000, 5500) * 2
        expected_y_padded = np.resize(expected_y_original, 1024)
        np.testing.assert_array_equal(y_chunk, expected_y_padded)


class TestMemoryEstimator(unittest.TestCase):
    """Test the MemoryEstimator utility class."""

    def test_memory_per_point_estimation(self):
        """Test estimating memory per data point."""
        # Without Jacobian
        mem_no_jac = MemoryEstimator.estimate_memory_per_point(
            n_params=3, use_jacobian=False
        )
        self.assertGreater(mem_no_jac, 0)

        # With Jacobian
        mem_with_jac = MemoryEstimator.estimate_memory_per_point(
            n_params=3, use_jacobian=True
        )
        self.assertGreater(mem_with_jac, mem_no_jac)

        # More parameters = more memory
        mem_more_params = MemoryEstimator.estimate_memory_per_point(
            n_params=10, use_jacobian=True
        )
        self.assertGreater(mem_more_params, mem_with_jac)

    def test_optimal_chunk_size_calculation(self):
        """Test calculating optimal chunk sizes."""
        config = LDMemoryConfig(memory_limit_gb=4.0)

        # Small dataset - single chunk
        chunk_size, stats = MemoryEstimator.calculate_optimal_chunk_size(
            n_points=10_000, n_params=3, memory_config=config
        )
        self.assertEqual(chunk_size, 10_000)  # All in one chunk
        self.assertEqual(stats.n_chunks, 1)

        # Large dataset - multiple chunks
        chunk_size, stats = MemoryEstimator.calculate_optimal_chunk_size(
            n_points=50_000_000, n_params=5, memory_config=config
        )
        self.assertLess(chunk_size, 50_000_000)  # Should be chunked
        self.assertGreater(stats.n_chunks, 1)
        self.assertEqual(chunk_size, stats.recommended_chunk_size)

    @patch("nlsq.streaming.large_dataset.psutil")
    def test_available_memory_detection(self, mock_psutil):
        """Test detecting available system memory."""
        # Mock psutil to return specific memory info
        mock_memory = MagicMock()
        mock_memory.available = 8 * 1024**3  # 8 GB in bytes
        mock_psutil.virtual_memory.return_value = mock_memory

        available_gb = MemoryEstimator.get_available_memory_gb()
        self.assertAlmostEqual(available_gb, 8.0, places=1)


class TestIntegrationLargeDatasets(unittest.TestCase):
    """Integration tests for complete large dataset workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = lambda x, a, b: a * jnp.exp(-b * x)
        self.true_params = [2.5, 1.3]
        np.random.seed(456)

    def test_end_to_end_20M_workflow(self):
        """Test complete workflow for 20M point dataset (simulated)."""
        # Simulate with smaller dataset for testing
        n_points = 200_000  # Simulate 20M behavior
        x = np.linspace(0, 10, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        # Step 1: Estimate memory requirements
        stats = estimate_memory_requirements(20_000_000, 2)
        self.assertGreater(stats.n_chunks, 1)

        # Step 2: Fit with automatic strategy selection
        result = fit_large_dataset(
            self.model,
            x,
            y,
            p0=[2.0, 1.0],
            memory_limit_gb=0.1,  # Small limit to force chunking
            show_progress=False,
        )

        self.assertTrue(result.success)
        np.testing.assert_allclose(result.popt, self.true_params, rtol=0.15)

    def test_memory_config_context_manager(self):
        """Test using memory configuration context manager."""
        from nlsq import MemoryConfig, memory_context

        n_points = 10_000
        x = np.linspace(0, 5, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        config = MemoryConfig(memory_limit_gb=2.0, enable_mixed_precision_fallback=True)

        with memory_context(config):
            # Fit within the context
            popt, _pcov = curve_fit_large(
                self.model, x, y, p0=[2.0, 1.0], memory_limit_gb=1.0
            )

            np.testing.assert_allclose(popt, self.true_params, rtol=0.1)

    def test_progressive_fitting_strategy(self):
        """Test progressive fitting with chunked processing."""
        n_points = 100_000
        x = np.linspace(0, 10, n_points)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, n_points)

        fitter = LargeDatasetFitter(memory_limit_gb=0.05)

        # Should automatically handle via chunking
        result = fitter.fit(self.model, x, y, p0=[2.0, 1.0])

        self.assertTrue(result.success)
        # Check n_chunks if it exists in the result
        if hasattr(result, "n_chunks"):
            self.assertGreater(result.n_chunks, 1)
        np.testing.assert_allclose(result.popt, self.true_params, rtol=0.2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling for edge cases."""

    def test_invalid_memory_limit(self):
        """Test handling of invalid memory limits."""
        # Test with very small memory limit (should still work, just with small chunks)
        fitter = LargeDatasetFitter(memory_limit_gb=0.001)
        self.assertIsNotNone(fitter)

    def test_mismatched_data_sizes(self):
        """Test handling of mismatched x and y data."""
        model = lambda x, a: a * x
        x = np.arange(100)
        y = np.arange(50)  # Wrong size

        fitter = LargeDatasetFitter()
        result = fitter.fit(model, x, y, p0=[1.0])

        # Should fail gracefully
        self.assertFalse(result.success)
        self.assertIn("lengths", result.message.lower())

    def test_convergence_failure_handling(self):
        """Test handling of convergence failures."""
        # Create a problem that genuinely cannot converge with limited iterations
        # Use a highly nonlinear model with poor initial guess on structured data
        model = lambda x, a, b, c, d: a * jnp.sin(b * x + c) + d
        x = np.linspace(0, 10, 1000)

        # Use deterministic data that requires many iterations to fit
        np.random.seed(42)
        true_params = [5.0, 2.0, 1.0, 3.0]
        y = (
            true_params[0] * np.sin(true_params[1] * x + true_params[2])
            + true_params[3]
        )
        y += np.random.normal(0, 0.5, len(x))  # Add noise

        # Use very poor initial guess to ensure non-convergence
        poor_guess = [0.1, 0.1, 0.0, 0.0]  # Far from true params

        fitter = LargeDatasetFitter()
        result = fitter.fit(
            model,
            x,
            y,
            p0=poor_guess,
            max_nfev=5,  # Very limited iterations (reduced from 10)
            ftol=1e-12,  # Very tight tolerance to prevent premature convergence
            xtol=1e-12,
            gtol=1e-12,
        )

        # With only 5 iterations and tight tolerances, should not converge
        # Check that either success is False OR final cost is still high
        if result.success:
            # If it claims success, verify cost is actually low (< 250)
            # If cost is high, the test is checking that we handle the case
            self.assertLess(
                result.cost, 250.0, "Optimizer claims success but cost is still high"
            )
        else:
            # Expected case: optimizer correctly reports non-convergence
            self.assertIsNotNone(result.message)

    def test_shape_validation_catches_mismatch(self):
        """Test that shape validation detects model-chunking incompatibility."""

        # Model that ignores xdata size (always returns fixed size)
        def bad_model(xdata, a, b):
            return jnp.ones(10000)  # Always returns 10000, regardless of xdata size

        # Use larger dataset to ensure chunking occurs
        xdata = jnp.arange(10000)
        ydata = jnp.ones(10000)

        # Force chunking with small chunk size
        config = LDMemoryConfig(
            memory_limit_gb=0.001, min_chunk_size=100, max_chunk_size=5000
        )
        fitter = LargeDatasetFitter(config=config)

        # Should raise ValueError with shape mismatch message
        with self.assertRaises(ValueError) as context:
            fitter.fit(bad_model, xdata, ydata, p0=[1, 1])

        # Verify error message mentions shape mismatch
        error_message = str(context.exception)
        self.assertIn("SHAPE MISMATCH", error_message)
        self.assertIn("expected shape", error_message.lower())

    def test_shape_validation_passes_correct_model(self):
        """Test that correct chunking-compatible models pass validation."""

        # Correct model that respects xdata size
        def good_model(xdata, a, b):
            return a * jnp.exp(-b * xdata)

        np.random.seed(42)
        # Use smaller dataset for faster test
        xdata = jnp.linspace(0, 1, 1000)
        ydata = 2.0 * jnp.exp(-1.0 * xdata) + 0.01 * np.random.randn(1000)

        # Force chunking with small chunk size
        config = LDMemoryConfig(
            memory_limit_gb=0.001, min_chunk_size=100, max_chunk_size=500
        )
        fitter = LargeDatasetFitter(config=config)

        # Should work without errors
        result = fitter.fit(good_model, xdata, ydata, p0=[2.0, 1.0])

        # Should succeed
        self.assertTrue(result.success)
        self.assertIsNotNone(result.popt)

        # Verify parameters are close to truth
        self.assertAlmostEqual(result.popt[0], 2.0, delta=0.5)
        self.assertAlmostEqual(result.popt[1], 1.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
