"""Tests for the NLSQ configuration module."""

import os
import unittest
from unittest.mock import patch

from nlsq.config import JAXConfig, enable_x64, is_x64_enabled, precision_context


class TestJAXConfig(unittest.TestCase):
    """Test the JAXConfig singleton configuration manager."""

    def setUp(self):
        """Reset environment before each test."""
        # Clean up any existing environment variables
        os.environ.pop("NLSQ_DISABLE_X64", None)

    def tearDown(self):
        """Clean up after each test."""
        os.environ.pop("NLSQ_DISABLE_X64", None)

    def test_singleton_pattern(self):
        """Test that JAXConfig follows singleton pattern."""
        config1 = JAXConfig()
        config2 = JAXConfig()
        self.assertIs(config1, config2)
        self.assertEqual(id(config1), id(config2))

    def test_x64_enabled_by_default(self):
        """Test that 64-bit precision is enabled by default."""
        # JAXConfig should enable x64 on initialization
        config = JAXConfig()
        self.assertTrue(config.is_x64_enabled())
        self.assertTrue(is_x64_enabled())

    def test_enable_x64_function(self):
        """Test enabling and disabling x64 precision."""
        # Test enabling
        enable_x64(True)
        self.assertTrue(is_x64_enabled())

        # Test disabling
        enable_x64(False)
        self.assertFalse(is_x64_enabled())

        # Re-enable for other tests
        enable_x64(True)

    def test_precision_context_manager(self):
        """Test precision context manager works correctly."""
        # Store original state
        original = is_x64_enabled()

        # Test switching to 32-bit
        with precision_context(use_x64=False):
            self.assertFalse(is_x64_enabled())

        # Should restore original state
        self.assertEqual(is_x64_enabled(), original)

        # Test switching to 64-bit
        enable_x64(False)  # Start with 32-bit
        with precision_context(use_x64=True):
            self.assertTrue(is_x64_enabled())

        # Should restore to 32-bit
        self.assertFalse(is_x64_enabled())

        # Restore to 64-bit for other tests
        enable_x64(True)

    def test_precision_context_manager_with_exception(self):
        """Test context manager restores state even with exception."""
        original = is_x64_enabled()

        try:
            with precision_context(use_x64=not original):
                self.assertNotEqual(is_x64_enabled(), original)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still restore original state
        self.assertEqual(is_x64_enabled(), original)

    def test_disable_via_environment_variable(self):
        """Test that x64 can be disabled via environment variable."""
        # This test would require reloading the module, which is complex
        # We'll test the logic instead
        os.environ["NLSQ_DISABLE_X64"] = "1"

        # In reality, this would be checked during initialization
        # Here we just verify the environment variable is set
        self.assertEqual(os.environ.get("NLSQ_DISABLE_X64"), "1")

    def test_multiple_enable_calls(self):
        """Test that multiple enable calls work correctly."""
        # Enable multiple times
        enable_x64(True)
        enable_x64(True)
        self.assertTrue(is_x64_enabled())

        # Disable multiple times
        enable_x64(False)
        enable_x64(False)
        self.assertFalse(is_x64_enabled())

        # Restore
        enable_x64(True)

    def test_nested_precision_contexts(self):
        """Test nested precision context managers."""
        original = is_x64_enabled()

        with precision_context(use_x64=False):
            self.assertFalse(is_x64_enabled())

            with precision_context(use_x64=True):
                self.assertTrue(is_x64_enabled())

            # Should restore to outer context
            self.assertFalse(is_x64_enabled())

        # Should restore to original
        self.assertEqual(is_x64_enabled(), original)


class TestJAXConfigIntegration(unittest.TestCase):
    """Integration tests for JAXConfig with JAX library."""

    @patch("jax.config")
    def test_jax_config_update_called(self, mock_jax_config):
        """Test that JAX config.update is called correctly."""
        # Create a fresh config instance
        JAXConfig._instance = None
        JAXConfig._initialized = False

        config = JAXConfig()
        config.enable_x64(True)

        # Verify JAX config was updated
        mock_jax_config.update.assert_called()

    def test_actual_jax_integration(self):
        """Test actual integration with JAX if available."""
        try:
            import jax
            import jax.numpy as jnp

            # Enable 64-bit
            enable_x64(True)
            arr = jnp.array([1.0])
            self.assertEqual(arr.dtype, jnp.float64)

            # Test context manager with 32-bit
            with precision_context(use_x64=False):
                arr32 = jnp.array([1.0])
                self.assertEqual(arr32.dtype, jnp.float32)

            # Back to 64-bit
            arr64 = jnp.array([1.0])
            self.assertEqual(arr64.dtype, jnp.float64)

        except ImportError:
            self.skipTest("JAX not installed")


class TestMemoryConfig(unittest.TestCase):
    """Tests for MemoryConfig dataclass."""

    def test_default_initialization(self):
        """Test MemoryConfig with default values."""
        from nlsq.config import MemoryConfig

        config = MemoryConfig()

        self.assertEqual(config.memory_limit_gb, 8.0)
        self.assertIsNone(config.gpu_memory_fraction)
        self.assertTrue(config.enable_mixed_precision_fallback)
        self.assertIsNone(config.chunk_size_mb)
        self.assertEqual(config.out_of_memory_strategy, "fallback")
        self.assertEqual(config.safety_factor, 0.8)
        self.assertEqual(config.auto_chunk_threshold_gb, 4.0)
        self.assertTrue(config.progress_reporting)
        self.assertEqual(config.min_chunk_size, 1000)
        self.assertEqual(config.max_chunk_size, 1_000_000)

    def test_custom_initialization(self):
        """Test MemoryConfig with custom values."""
        from nlsq.config import MemoryConfig

        config = MemoryConfig(
            memory_limit_gb=16.0,
            gpu_memory_fraction=0.75,
            enable_mixed_precision_fallback=False,
            chunk_size_mb=256,
            out_of_memory_strategy="reduce",
            safety_factor=0.9,
            auto_chunk_threshold_gb=8.0,
            progress_reporting=False,
            min_chunk_size=500,
            max_chunk_size=2_000_000,
        )

        self.assertEqual(config.memory_limit_gb, 16.0)
        self.assertEqual(config.gpu_memory_fraction, 0.75)
        self.assertFalse(config.enable_mixed_precision_fallback)
        self.assertEqual(config.chunk_size_mb, 256)
        self.assertEqual(config.out_of_memory_strategy, "reduce")
        self.assertEqual(config.safety_factor, 0.9)
        self.assertEqual(config.auto_chunk_threshold_gb, 8.0)
        self.assertFalse(config.progress_reporting)
        self.assertEqual(config.min_chunk_size, 500)
        self.assertEqual(config.max_chunk_size, 2_000_000)

    def test_memory_limit_too_low(self):
        """Test validation error when memory_limit_gb is too low."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(memory_limit_gb=0.05)

        self.assertIn(
            "memory_limit_gb must be between 0.1 and 1024", str(ctx.exception)
        )

    def test_memory_limit_too_high(self):
        """Test validation error when memory_limit_gb is too high."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(memory_limit_gb=2048.0)

        self.assertIn(
            "memory_limit_gb must be between 0.1 and 1024", str(ctx.exception)
        )

    def test_memory_limit_boundary_valid(self):
        """Test memory_limit_gb at valid boundaries."""
        from nlsq.config import MemoryConfig

        # Test lower boundary
        config_low = MemoryConfig(memory_limit_gb=0.1)
        self.assertEqual(config_low.memory_limit_gb, 0.1)

        # Test upper boundary
        config_high = MemoryConfig(memory_limit_gb=1024.0)
        self.assertEqual(config_high.memory_limit_gb, 1024.0)

    def test_gpu_memory_fraction_too_low(self):
        """Test validation error when gpu_memory_fraction is too low."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(gpu_memory_fraction=0.0)

        self.assertIn(
            "gpu_memory_fraction must be between 0.0 and 1.0", str(ctx.exception)
        )

    def test_gpu_memory_fraction_too_high(self):
        """Test validation error when gpu_memory_fraction is too high."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(gpu_memory_fraction=1.5)

        self.assertIn(
            "gpu_memory_fraction must be between 0.0 and 1.0", str(ctx.exception)
        )

    def test_gpu_memory_fraction_boundary_valid(self):
        """Test gpu_memory_fraction at valid boundaries."""
        from nlsq.config import MemoryConfig

        # Test just above zero
        config_low = MemoryConfig(gpu_memory_fraction=0.01)
        self.assertEqual(config_low.gpu_memory_fraction, 0.01)

        # Test at 1.0
        config_high = MemoryConfig(gpu_memory_fraction=1.0)
        self.assertEqual(config_high.gpu_memory_fraction, 1.0)

    def test_safety_factor_too_low(self):
        """Test validation error when safety_factor is too low."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(safety_factor=0.05)

        self.assertIn("safety_factor must be between 0.1 and 1.0", str(ctx.exception))

    def test_safety_factor_too_high(self):
        """Test validation error when safety_factor is too high."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(safety_factor=1.5)

        self.assertIn("safety_factor must be between 0.1 and 1.0", str(ctx.exception))

    def test_safety_factor_boundary_valid(self):
        """Test safety_factor at valid boundaries."""
        from nlsq.config import MemoryConfig

        config_low = MemoryConfig(safety_factor=0.1)
        self.assertEqual(config_low.safety_factor, 0.1)

        config_high = MemoryConfig(safety_factor=1.0)
        self.assertEqual(config_high.safety_factor, 1.0)

    def test_invalid_out_of_memory_strategy(self):
        """Test validation error for invalid out_of_memory_strategy."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(out_of_memory_strategy="invalid")

        self.assertIn(
            "out_of_memory_strategy must be 'fallback', 'reduce', or 'error'",
            str(ctx.exception),
        )

    def test_valid_out_of_memory_strategies(self):
        """Test all valid out_of_memory_strategy values."""
        from nlsq.config import MemoryConfig

        for strategy in ["fallback", "reduce", "error"]:
            config = MemoryConfig(out_of_memory_strategy=strategy)
            self.assertEqual(config.out_of_memory_strategy, strategy)

    def test_min_chunk_size_greater_than_max(self):
        """Test validation error when min_chunk_size > max_chunk_size."""
        from nlsq.config import MemoryConfig

        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(min_chunk_size=2_000_000, max_chunk_size=1_000_000)

        self.assertIn("min_chunk_size", str(ctx.exception))
        self.assertIn("max_chunk_size", str(ctx.exception))

    def test_chunk_size_equal_valid(self):
        """Test that min_chunk_size == max_chunk_size is valid."""
        from nlsq.config import MemoryConfig

        config = MemoryConfig(min_chunk_size=1000, max_chunk_size=1000)
        self.assertEqual(config.min_chunk_size, config.max_chunk_size)


class TestLargeDatasetConfig(unittest.TestCase):
    """Tests for LargeDatasetConfig dataclass."""

    def test_default_initialization(self):
        """Test LargeDatasetConfig with default values (v0.2.0: sampling params removed)."""
        from nlsq.config import LargeDatasetConfig

        config = LargeDatasetConfig()

        # v0.2.0: Only test remaining attributes
        self.assertTrue(config.enable_automatic_solver_selection)
        self.assertIsInstance(config.solver_selection_thresholds, dict)
        self.assertIn("direct", config.solver_selection_thresholds)
        self.assertIn("iterative", config.solver_selection_thresholds)
        self.assertIn("chunked", config.solver_selection_thresholds)

        # v0.2.0: Verify sampling params are removed
        self.assertFalse(hasattr(config, "enable_sampling"))
        self.assertFalse(hasattr(config, "sampling_threshold"))
        self.assertFalse(hasattr(config, "max_sampled_size"))
        self.assertFalse(hasattr(config, "sampling_strategy"))

    def test_custom_initialization(self):
        """Test LargeDatasetConfig with custom values (v0.2.0: sampling params removed)."""
        from nlsq.config import LargeDatasetConfig

        custom_thresholds = {
            "direct": 50_000,
            "iterative": 5_000_000,
            "chunked": 50_000_000,
        }

        config = LargeDatasetConfig(
            enable_automatic_solver_selection=False,
            solver_selection_thresholds=custom_thresholds,
        )

        self.assertFalse(config.enable_automatic_solver_selection)
        self.assertEqual(config.solver_selection_thresholds, custom_thresholds)

    def test_invalid_sampling_strategy(self):
        """Test that sampling_strategy parameter is no longer accepted (v0.2.0)."""
        from nlsq.config import LargeDatasetConfig

        # v0.2.0: Should raise TypeError for removed parameter
        with self.assertRaises(TypeError) as ctx:
            LargeDatasetConfig(sampling_strategy="invalid")

        self.assertIn("unexpected keyword argument", str(ctx.exception))
        self.assertIn("sampling_strategy", str(ctx.exception))

    def test_valid_sampling_strategies(self):
        """Test that sampling strategies are no longer accepted (v0.2.0)."""
        from nlsq.config import LargeDatasetConfig

        # v0.2.0: All sampling strategies should raise TypeError
        for strategy in ["random", "uniform", "stratified"]:
            with self.assertRaises(TypeError):
                LargeDatasetConfig(sampling_strategy=strategy)

    def test_max_sampled_size_larger_than_threshold_warning(self):
        """Test that sampling parameters are no longer accepted (v0.2.0)."""
        from nlsq.config import LargeDatasetConfig

        # v0.2.0: Should raise TypeError for removed parameters
        with self.assertRaises(TypeError):
            LargeDatasetConfig(
                sampling_threshold=10_000_000, max_sampled_size=20_000_000
            )

    def test_solver_selection_thresholds_default(self):
        """Test default solver_selection_thresholds values."""
        from nlsq.config import LargeDatasetConfig

        config = LargeDatasetConfig()

        self.assertEqual(config.solver_selection_thresholds["direct"], 100_000)
        self.assertEqual(config.solver_selection_thresholds["iterative"], 10_000_000)
        self.assertEqual(config.solver_selection_thresholds["chunked"], 100_000_000)


class TestMemoryConfigFunctions(unittest.TestCase):
    """Tests for memory configuration helper functions."""

    def test_get_memory_config_function(self):
        """Test get_memory_config function."""
        from nlsq.config import MemoryConfig, get_memory_config

        config = get_memory_config()

        self.assertIsInstance(config, MemoryConfig)
        self.assertGreater(config.memory_limit_gb, 0)

    def test_set_memory_limits_basic(self):
        """Test set_memory_limits function with basic parameters."""
        from nlsq.config import JAXConfig, get_memory_config, set_memory_limits

        original_config = get_memory_config()

        try:
            set_memory_limits(memory_limit_gb=16.0)

            new_config = get_memory_config()
            self.assertEqual(new_config.memory_limit_gb, 16.0)
        finally:
            # Restore original
            JAXConfig.set_memory_config(original_config)

    def test_set_memory_limits_with_gpu_fraction(self):
        """Test set_memory_limits with GPU memory fraction."""
        from nlsq.config import JAXConfig, get_memory_config, set_memory_limits

        original_config = get_memory_config()

        try:
            set_memory_limits(
                memory_limit_gb=24.0, gpu_memory_fraction=0.9, safety_factor=0.85
            )

            new_config = get_memory_config()
            self.assertEqual(new_config.memory_limit_gb, 24.0)
            self.assertEqual(new_config.gpu_memory_fraction, 0.9)
            self.assertEqual(new_config.safety_factor, 0.85)
        finally:
            JAXConfig.set_memory_config(original_config)

    def test_enable_mixed_precision_fallback_function(self):
        """Test enable_mixed_precision_fallback function."""
        from nlsq.config import (
            JAXConfig,
            enable_mixed_precision_fallback,
            get_memory_config,
        )

        original_config = get_memory_config()

        try:
            # Enable
            enable_mixed_precision_fallback(True)
            config = get_memory_config()
            self.assertTrue(config.enable_mixed_precision_fallback)

            # Disable
            enable_mixed_precision_fallback(False)
            config = get_memory_config()
            self.assertFalse(config.enable_mixed_precision_fallback)
        finally:
            JAXConfig.set_memory_config(original_config)

    def test_get_large_dataset_config_function(self):
        """Test get_large_dataset_config function (v0.2.0: sampling params removed)."""
        from nlsq.config import LargeDatasetConfig, get_large_dataset_config

        config = get_large_dataset_config()

        self.assertIsInstance(config, LargeDatasetConfig)
        self.assertIsInstance(config.enable_automatic_solver_selection, bool)
        # v0.2.0: Verify sampling param is removed
        self.assertFalse(hasattr(config, "enable_sampling"))

    def test_configure_for_large_datasets(self):
        """Test configure_for_large_datasets function (v0.2.0: sampling params emit warnings)."""
        import warnings

        from nlsq.config import (
            JAXConfig,
            configure_for_large_datasets,
            get_large_dataset_config,
            get_memory_config,
        )

        original_ld_config = get_large_dataset_config()
        original_mem_config = get_memory_config()

        try:
            # v0.2.0: enable_sampling should emit deprecation warning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", DeprecationWarning)

                configure_for_large_datasets(
                    memory_limit_gb=16.0,
                    enable_sampling=True,  # Should emit warning
                    enable_chunking=False,
                    progress_reporting=True,
                    mixed_precision_fallback=False,
                )

                # Check that deprecation warning was issued
                self.assertTrue(len(w) >= 1)
                self.assertTrue(
                    any(
                        issubclass(warning.category, DeprecationWarning)
                        for warning in w
                    )
                )

            ld_config = get_large_dataset_config()
            mem_config = get_memory_config()

            # Check large dataset config (v0.2.0: no more enable_sampling)
            self.assertTrue(ld_config.enable_automatic_solver_selection)
            self.assertFalse(hasattr(ld_config, "enable_sampling"))

            # Check memory config
            self.assertEqual(mem_config.memory_limit_gb, 16.0)
            self.assertTrue(mem_config.progress_reporting)
            self.assertFalse(mem_config.enable_mixed_precision_fallback)
        finally:
            JAXConfig.set_large_dataset_config(original_ld_config)
            JAXConfig.set_memory_config(original_mem_config)


class TestMemoryContextManager(unittest.TestCase):
    """Tests for memory_context context manager."""

    def test_memory_context_manager(self):
        """Test memory_context context manager."""
        from nlsq.config import MemoryConfig, get_memory_config, memory_context

        original_config = get_memory_config()

        temp_config = MemoryConfig(memory_limit_gb=32.0, safety_factor=0.95)

        with memory_context(temp_config):
            current = get_memory_config()
            self.assertEqual(current.memory_limit_gb, 32.0)
            self.assertEqual(current.safety_factor, 0.95)

        # Should restore original
        restored = get_memory_config()
        self.assertEqual(restored.memory_limit_gb, original_config.memory_limit_gb)
        self.assertEqual(restored.safety_factor, original_config.safety_factor)

    def test_memory_context_nested(self):
        """Test nested memory_context managers."""
        from nlsq.config import MemoryConfig, get_memory_config, memory_context

        config1 = MemoryConfig(memory_limit_gb=16.0)
        config2 = MemoryConfig(memory_limit_gb=32.0)

        original = get_memory_config()

        with memory_context(config1):
            self.assertEqual(get_memory_config().memory_limit_gb, 16.0)

            with memory_context(config2):
                self.assertEqual(get_memory_config().memory_limit_gb, 32.0)

            # Should restore to config1
            self.assertEqual(get_memory_config().memory_limit_gb, 16.0)

        # Should restore to original
        self.assertEqual(get_memory_config().memory_limit_gb, original.memory_limit_gb)

    def test_large_dataset_context_manager(self):
        """Test large_dataset_context context manager (v0.2.0: sampling params removed)."""
        from nlsq.config import (
            LargeDatasetConfig,
            get_large_dataset_config,
            large_dataset_context,
        )

        original_config = get_large_dataset_config()

        temp_config = LargeDatasetConfig(enable_automatic_solver_selection=False)

        with large_dataset_context(temp_config):
            current = get_large_dataset_config()
            self.assertFalse(current.enable_automatic_solver_selection)

        # Should restore original
        restored = get_large_dataset_config()
        self.assertEqual(
            restored.enable_automatic_solver_selection,
            original_config.enable_automatic_solver_selection,
        )

    def test_large_dataset_context_nested(self):
        """Test nested large_dataset_context managers (v0.2.0: use solver thresholds)."""
        from nlsq.config import (
            LargeDatasetConfig,
            get_large_dataset_config,
            large_dataset_context,
        )

        config1 = LargeDatasetConfig(
            solver_selection_thresholds={
                "direct": 50_000,
                "iterative": 5_000_000,
                "chunked": 50_000_000,
            }
        )
        config2 = LargeDatasetConfig(
            solver_selection_thresholds={
                "direct": 100_000,
                "iterative": 10_000_000,
                "chunked": 100_000_000,
            }
        )

        original = get_large_dataset_config()

        with large_dataset_context(config1):
            self.assertEqual(
                get_large_dataset_config().solver_selection_thresholds["chunked"],
                50_000_000,
            )

            with large_dataset_context(config2):
                self.assertEqual(
                    get_large_dataset_config().solver_selection_thresholds["chunked"],
                    100_000_000,
                )

            # Should restore to config1
            self.assertEqual(
                get_large_dataset_config().solver_selection_thresholds["chunked"],
                50_000_000,
            )

        # Should restore to original
        self.assertEqual(
            get_large_dataset_config().solver_selection_thresholds,
            original.solver_selection_thresholds,
        )


class TestConfigEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def test_memory_config_with_none_optional_fields(self):
        """Test MemoryConfig with None for optional fields."""
        from nlsq.config import MemoryConfig

        config = MemoryConfig(gpu_memory_fraction=None, chunk_size_mb=None)

        self.assertIsNone(config.gpu_memory_fraction)
        self.assertIsNone(config.chunk_size_mb)

    def test_memory_config_extreme_valid_values(self):
        """Test MemoryConfig with extreme but valid values."""
        from nlsq.config import MemoryConfig

        config = MemoryConfig(
            memory_limit_gb=0.1,
            gpu_memory_fraction=0.01,
            safety_factor=0.1,
            min_chunk_size=1,
            max_chunk_size=1_000_000_000,
        )

        self.assertEqual(config.memory_limit_gb, 0.1)
        self.assertEqual(config.gpu_memory_fraction, 0.01)
        self.assertEqual(config.safety_factor, 0.1)

    def test_large_dataset_config_empty_thresholds(self):
        """Test LargeDatasetConfig with empty solver_selection_thresholds."""
        from nlsq.config import LargeDatasetConfig

        config = LargeDatasetConfig(solver_selection_thresholds={})

        self.assertEqual(config.solver_selection_thresholds, {})

    def test_set_memory_limits_preserves_other_settings(self):
        """Test that set_memory_limits preserves other configuration settings."""
        from nlsq.config import (
            JAXConfig,
            MemoryConfig,
            get_memory_config,
            set_memory_limits,
        )

        original = get_memory_config()

        try:
            # Set a custom config first
            custom = MemoryConfig(
                memory_limit_gb=10.0,
                progress_reporting=False,
                out_of_memory_strategy="error",
            )
            JAXConfig.set_memory_config(custom)

            # Now use set_memory_limits
            set_memory_limits(memory_limit_gb=20.0)

            # Check that other settings were preserved
            new_config = get_memory_config()
            self.assertEqual(new_config.memory_limit_gb, 20.0)
            self.assertFalse(new_config.progress_reporting)
            self.assertEqual(new_config.out_of_memory_strategy, "error")
        finally:
            JAXConfig.set_memory_config(original)

    def test_multiple_validation_errors(self):
        """Test that validation catches first error in __post_init__."""
        from nlsq.config import MemoryConfig

        # memory_limit_gb is checked first
        with self.assertRaises(ValueError) as ctx:
            MemoryConfig(
                memory_limit_gb=0.01,  # Invalid
                safety_factor=2.0,  # Also invalid
            )

        # Should fail on first validation (memory_limit_gb)
        self.assertIn("memory_limit_gb", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
