"""
Integration tests for v0.2.0 deprecation warnings.

Tests that deprecated sampling parameters emit proper DeprecationWarnings
with migration guidance, and that the code continues to work gracefully.
"""

import unittest
import warnings

import jax.numpy as jnp
import numpy as np

from nlsq import LargeDatasetConfig, LDMemoryConfig, curve_fit_large


class TestDeprecationWarnings(unittest.TestCase):
    """Test deprecation warnings for removed sampling parameters."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        # Use 10000 points and override size_threshold to trigger large dataset path
        self.x = np.linspace(0, 5, 10000)
        self.y = (
            2.0 * np.exp(-0.5 * self.x) + 0.3 + np.random.normal(0, 0.05, len(self.x))
        )
        # Set size_threshold below our data size to force large dataset processing
        self.size_threshold = 5000

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        self.model = model
        self.p0 = [1.5, 0.4, 0.2]

    def test_curve_fit_large_enable_sampling_rejected(self):
        """Test that curve_fit_large rejects removed enable_sampling parameter."""
        with self.assertRaises(TypeError) as ctx:
            curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,
                enable_sampling=True,  # Removed parameter
            )

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("enable_sampling", str(ctx.exception))

    def test_curve_fit_large_sampling_threshold_rejected(self):
        """Test that curve_fit_large rejects removed sampling_threshold parameter."""
        with self.assertRaises(TypeError) as ctx:
            curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,
                sampling_threshold=100_000,  # Removed parameter
            )

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("sampling_threshold", str(ctx.exception))

    def test_curve_fit_large_max_sampled_size_rejected(self):
        """Test that curve_fit_large rejects removed max_sampled_size parameter."""
        with self.assertRaises(TypeError) as ctx:
            curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,
                max_sampled_size=50_000,  # Removed parameter
            )

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("max_sampled_size", str(ctx.exception))

    def test_curve_fit_large_no_warning_without_deprecated_params(self):
        """Test that no warning is emitted when no deprecated params are used."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call without any deprecated parameters
            popt, _pcov = curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
            )

            # Check that no DeprecationWarning was raised
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            self.assertEqual(len(deprecation_warnings), 0)

            # Verify fit works
            self.assertEqual(len(popt), 3)

    def test_large_dataset_config_no_sampling_params(self):
        """Test that LargeDatasetConfig rejects removed sampling parameters."""
        # These should raise TypeError since parameters were removed
        with self.assertRaises(TypeError) as ctx:
            LargeDatasetConfig(enable_sampling=True)

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("enable_sampling", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            LargeDatasetConfig(sampling_threshold=100_000)

        self.assertIn("unexpected keyword argument", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            LargeDatasetConfig(max_sampled_size=50_000)

        self.assertIn("unexpected keyword argument", str(ctx.exception))

    def test_ldmemory_config_no_sampling_params(self):
        """Test that LDMemoryConfig rejects removed sampling parameters."""
        # These should raise TypeError since parameters were removed
        with self.assertRaises(TypeError) as ctx:
            LDMemoryConfig(enable_sampling=True)

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("enable_sampling", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            LDMemoryConfig(sampling_threshold=100_000)

        self.assertIn("unexpected keyword argument", str(ctx.exception))

        with self.assertRaises(TypeError) as ctx:
            LDMemoryConfig(max_sampled_size=50_000)

        self.assertIn("unexpected keyword argument", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
