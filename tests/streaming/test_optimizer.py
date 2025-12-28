"""
Test suite for streaming optimizer functionality.

Tests the StreamingOptimizer for handling unlimited-size datasets
through streaming and batch processing.

NOTE: Legacy streaming API tests (13 tests) were removed in v0.1.3+
The deprecated subsampling parameters are no longer supported.
See CLAUDE.md 'Deprecations and Changes (v0.1.3+)' for details.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

# Skip all tests in this module if h5py is not available
h5py = pytest.importorskip("h5py")

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestStreamingConfig(unittest.TestCase):
    """Test StreamingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()

        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.momentum, 0.9)
        self.assertEqual(config.max_epochs, 10)
        self.assertEqual(config.convergence_tol, 1e-6)
        self.assertTrue(config.use_adam)

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            batch_size=5000, learning_rate=0.001, max_epochs=50, use_adam=True
        )

        self.assertEqual(config.batch_size, 5000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.max_epochs, 50)
        self.assertTrue(config.use_adam)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test that we can create configs with different values
        config1 = StreamingConfig(batch_size=5000)
        self.assertEqual(config1.batch_size, 5000)

        config2 = StreamingConfig(learning_rate=0.001)
        self.assertEqual(config2.learning_rate, 0.001)

        config3 = StreamingConfig(convergence_tol=1e-8)
        self.assertEqual(config3.convergence_tol, 1e-8)


class TestStreamingOptimizer(unittest.TestCase):
    """Test the StreamingOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = lambda x, a, b: a * jnp.exp(-b * x)
        self.true_params = [2.5, 1.3]
        np.random.seed(42)

    def test_optimizer_initialization(self):
        """Test StreamingOptimizer initialization."""
        config = StreamingConfig(batch_size=500)
        optimizer = StreamingOptimizer(config)

        self.assertEqual(optimizer.config.batch_size, 500)
        self.assertEqual(optimizer.iteration, 0)
        self.assertEqual(optimizer.epoch, 0)
        self.assertEqual(optimizer.best_loss, float("inf"))
        self.assertIsNone(optimizer.best_params)

    def test_reset_state(self):
        """Test resetting optimizer state."""
        optimizer = StreamingOptimizer()

        # Modify state
        optimizer.iteration = 100
        optimizer.epoch = 10
        optimizer.best_loss = 0.5
        optimizer.best_params = np.array([1.0, 2.0])

        # Reset
        optimizer.reset_state()

        self.assertEqual(optimizer.iteration, 0)
        self.assertEqual(optimizer.epoch, 0)
        self.assertEqual(optimizer.best_loss, float("inf"))
        self.assertIsNone(optimizer.best_params)

    def test_adaptive_learning_rate(self):
        """Test learning rate behavior."""
        config = StreamingConfig(learning_rate=0.1, warmup_steps=10)
        optimizer = StreamingOptimizer(config)

        # Generate simple data
        x = np.linspace(0, 5, 100)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 100)

        # Simple linear model for testing
        model = lambda x, a, b: a * x + b

        # Create a generator
        def data_gen():
            yield x, y

        result = optimizer.fit(data_gen(), model, p0=np.array([1.5, 0.5]), verbose=0)

        # Check that optimization completed
        self.assertIn("x", result)

    def test_adam_optimizer(self):
        """Test using Adam optimizer instead of SGD."""
        config = StreamingConfig(use_adam=True, learning_rate=0.01)
        optimizer = StreamingOptimizer(config)

        # Generate test data
        x = np.linspace(0, 5, 500)
        y = self.model(x, *self.true_params)
        y = np.array(y) + np.random.normal(0, 0.05, 500)

        # Create a simple generator
        def data_gen():
            yield x[:250], y[:250]
            yield x[250:], y[250:]

        result = optimizer.fit(
            data_gen(), self.model, p0=np.array([2.0, 1.0]), verbose=0
        )

        self.assertIn("x", result)


class TestStreamingIntegration(unittest.TestCase):
    """Integration tests for streaming optimizer workflows."""

    def test_online_learning_scenario(self):
        """Test online learning with continuously arriving data."""
        model = lambda x, a: a * x
        true_param = 2.5

        # Simulate online data arrival
        def online_data():
            """Simulate data arriving continuously."""
            for t in range(50):  # 50 time steps
                # Data changes slightly over time
                x = np.random.uniform(t, t + 1, 20)
                param_drift = true_param + 0.01 * t  # Parameter drift
                y = param_drift * x + np.random.normal(0, 0.05, 20)
                yield x, y

        config = StreamingConfig(batch_size=20, learning_rate=0.05)

        optimizer = StreamingOptimizer(config)

        result = optimizer.fit(online_data(), model, p0=np.array([2.0]), verbose=0)

        self.assertIn("x", result)
        # Should have a result
        self.assertIsNotNone(result["x"][0])


if __name__ == "__main__":
    unittest.main()
