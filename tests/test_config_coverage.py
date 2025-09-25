#!/usr/bin/env python3
"""Additional tests for config module to reach 60% coverage."""

import unittest
from nlsq.config import (
    LargeDatasetConfig,
    MemoryConfig,
    get_large_dataset_config,
    get_memory_config,
    set_memory_limits,
    configure_for_large_datasets,
    enable_mixed_precision_fallback,
    large_dataset_context,
    memory_context
)


class TestConfigCoverage(unittest.TestCase):
    """Tests to improve config module coverage."""

    def test_memory_config_basic(self):
        """Test basic MemoryConfig functionality."""
        config = MemoryConfig()
        self.assertIsNotNone(config)

        # Test with parameters
        config = MemoryConfig(max_memory_gb=4.0)
        self.assertEqual(config.max_memory_gb, 4.0)

    def test_large_dataset_config_basic(self):
        """Test basic LargeDatasetConfig functionality."""
        config = LargeDatasetConfig()
        self.assertIsNotNone(config)

        # Test with parameters
        config = LargeDatasetConfig(chunk_size=1000)
        self.assertEqual(config.chunk_size, 1000)

    def test_get_configs(self):
        """Test getting global configs."""
        mem_config = get_memory_config()
        self.assertIsInstance(mem_config, MemoryConfig)

        ld_config = get_large_dataset_config()
        self.assertIsInstance(ld_config, LargeDatasetConfig)

    def test_set_memory_limits_simple(self):
        """Test setting memory limits."""
        set_memory_limits(max_memory_gb=2.0)
        config = get_memory_config()
        self.assertEqual(config.max_memory_gb, 2.0)

    def test_configure_for_large_datasets_simple(self):
        """Test configuring for large datasets."""
        configure_for_large_datasets(memory_limit_gb=4.0)
        # Just check it doesn't crash
        self.assertTrue(True)

    def test_enable_mixed_precision_simple(self):
        """Test enabling mixed precision."""
        enable_mixed_precision_fallback()
        # Just check it doesn't crash
        self.assertTrue(True)

    def test_context_managers_simple(self):
        """Test context managers."""
        with large_dataset_context(chunk_size=500):
            config = get_large_dataset_config()
            self.assertEqual(config.chunk_size, 500)

        with memory_context(max_memory_gb=1.0):
            config = get_memory_config()
            self.assertEqual(config.max_memory_gb, 1.0)


if __name__ == '__main__':
    unittest.main()