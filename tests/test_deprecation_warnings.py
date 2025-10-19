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

    def test_curve_fit_large_enable_sampling_warning(self):
        """Test that curve_fit_large emits warning for enable_sampling parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with deprecated enable_sampling parameter
            popt, _pcov = curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                enable_sampling=True,  # Deprecated
            )

            # Check that DeprecationWarning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

            # Check message content
            message = str(w[0].message)
            self.assertIn("enable_sampling", message)
            self.assertIn("deprecated", message.lower())
            self.assertIn("streaming", message.lower())

            # Verify fit still works (chunked fit may not converge perfectly)
            self.assertEqual(len(popt), 3)
            self.assertIsInstance(popt, np.ndarray)
            # Just verify parameters are reasonable, not exact
            self.assertIsInstance(popt[0], (np.floating, float))
            self.assertIsInstance(popt[1], (np.floating, float))
            self.assertIsInstance(popt[2], (np.floating, float))

    def test_curve_fit_large_sampling_threshold_warning(self):
        """Test that curve_fit_large emits warning for sampling_threshold parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with deprecated sampling_threshold parameter
            popt, _pcov = curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                sampling_threshold=100_000,  # Deprecated
            )

            # Check that DeprecationWarning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

            # Check message content
            message = str(w[0].message)
            self.assertIn("sampling_threshold", message)
            self.assertIn("deprecated", message.lower())
            self.assertIn("streaming", message.lower())

            # Verify fit still works
            self.assertEqual(len(popt), 3)

    def test_curve_fit_large_max_sampled_size_warning(self):
        """Test that curve_fit_large emits warning for max_sampled_size parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with deprecated max_sampled_size parameter
            popt, _pcov = curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                max_sampled_size=50_000,  # Deprecated
            )

            # Check that DeprecationWarning was raised
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))

            # Check message content
            message = str(w[0].message)
            self.assertIn("max_sampled_size", message)
            self.assertIn("deprecated", message.lower())

            # Verify fit still works
            self.assertEqual(len(popt), 3)

    def test_curve_fit_large_multiple_deprecated_params_warning(self):
        """Test that multiple deprecated parameters emit multiple warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with multiple deprecated parameters
            popt, _pcov = curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                enable_sampling=True,  # Deprecated
                sampling_threshold=100_000,  # Deprecated
                max_sampled_size=50_000,  # Deprecated
            )

            # Check that multiple DeprecationWarnings were raised
            self.assertEqual(len(w), 3)
            for warning in w:
                self.assertTrue(issubclass(warning.category, DeprecationWarning))

            # Check that each parameter is mentioned
            messages = [str(warning.message) for warning in w]
            combined_message = " ".join(messages)
            self.assertIn("enable_sampling", combined_message)
            self.assertIn("sampling_threshold", combined_message)
            self.assertIn("max_sampled_size", combined_message)

            # Verify fit still works despite multiple deprecated params
            self.assertEqual(len(popt), 3)

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

    def test_warning_stacklevel_correct(self):
        """Test that deprecation warnings have correct stacklevel."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Call with deprecated parameter
            curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                enable_sampling=True,
            )

            # Check that warning points to this test file, not internal code
            self.assertEqual(len(w), 1)
            warning_filename = w[0].filename
            self.assertTrue(
                "test_deprecation_warnings.py" in warning_filename,
                f"Warning should point to test file, got: {warning_filename}",
            )

    def test_deprecation_warning_content_quality(self):
        """Test that deprecation warning messages are helpful and complete."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            curve_fit_large(
                self.model,
                self.x,
                self.y,
                p0=self.p0,
                size_threshold=self.size_threshold,  # Force large dataset path
                enable_sampling=True,
            )

            message = str(w[0].message)

            # Check for key information in the message
            required_content = [
                "enable_sampling",  # Parameter name
                "deprecated",  # Clear statement it's deprecated
                "streaming",  # Replacement feature
                "zero accuracy loss",  # Benefit explanation
            ]

            for content in required_content:
                self.assertIn(
                    content.lower(),
                    message.lower(),
                    f"Missing '{content}' in deprecation warning message",
                )

            # Check for benefit explanation (either phrase)
            self.assertTrue(
                "zero accuracy loss" in message.lower()
                or "100% of data" in message.lower(),
                "Warning should explain streaming processes all data",
            )


if __name__ == "__main__":
    unittest.main()
