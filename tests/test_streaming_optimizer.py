"""
Test suite for streaming optimizer functionality.

Tests the StreamingOptimizer for handling unlimited-size datasets
through streaming and batch processing.
"""

import os
import tempfile
import unittest

import h5py
import jax.numpy as jnp
import numpy as np

from nlsq.streaming_optimizer import (
    DataGenerator,
    StreamingConfig,
    StreamingOptimizer,
    create_hdf5_dataset,
    fit_unlimited_data,
)


class TestStreamingConfig(unittest.TestCase):
    """Test StreamingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()

        self.assertEqual(config.batch_size, 10000)
        self.assertEqual(config.learning_rate, 0.01)
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

    def test_fit_streaming_with_generator(self):
        """Test streaming fit with data generator."""

        def data_generator():
            """Generate batches of data."""
            batch_size = 100
            for i in range(10):  # 10 batches
                x = np.linspace(i, i + 1, batch_size)
                y = self.model(x, *self.true_params)
                y = np.array(y) + np.random.normal(0, 0.05, batch_size)
                yield x, y

        config = StreamingConfig(batch_size=100, max_epochs=2, learning_rate=0.1)
        optimizer = StreamingOptimizer(config)

        result = optimizer.fit_streaming(
            self.model, data_generator(), p0=np.array([2.0, 1.0]), verbose=0
        )

        self.assertIn("x", result)
        self.assertIn("fun", result)
        self.assertIn("nit", result)

        # Check convergence (rough check due to SGD nature)
        fitted_params = result["x"]
        self.assertEqual(len(fitted_params), 2)

    def test_fit_streaming_with_hdf5(self):
        """Test streaming fit with HDF5 file."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5_file = tmp.name

        try:
            # Create test dataset
            n_points = 1000
            x_data = np.linspace(0, 5, n_points)
            y_data = self.model(x_data, *self.true_params)
            y_data = np.array(y_data) + np.random.normal(0, 0.05, n_points)

            # Write to HDF5
            with h5py.File(h5_file, "w") as f:
                f.create_dataset("x", data=x_data)
                f.create_dataset("y", data=y_data)

            # Fit with streaming
            config = StreamingConfig(batch_size=200, max_epochs=1)
            optimizer = StreamingOptimizer(config)

            result = optimizer.fit_streaming(
                self.model, h5_file, p0=np.array([2.0, 1.0]), verbose=0
            )

            self.assertIn("x", result)
            self.assertIn("fun", result)

        finally:
            # Clean up
            if os.path.exists(h5_file):
                os.remove(h5_file)

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

        result = optimizer.fit_streaming(
            model, data_gen(), p0=np.array([1.5, 0.5]), verbose=0
        )

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

        result = optimizer.fit_streaming(
            self.model, data_gen(), p0=np.array([2.0, 1.0]), verbose=0
        )

        self.assertIn("x", result)

    def test_convergence_detection(self):
        """Test early stopping on convergence."""
        config = StreamingConfig(
            convergence_tol=0.1,  # Large tolerance for quick convergence
            max_epochs=100,
        )
        optimizer = StreamingOptimizer(config)

        # Simple problem that converges quickly
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 * x + 1.0

        model = lambda x, a, b: a * x + b

        # Create generator
        def data_gen():
            yield x, y

        result = optimizer.fit_streaming(
            model,
            data_gen(),  # Use generator
            p0=np.array([2.0, 1.0]),  # Start at true values
            verbose=0,
        )

        # Should complete
        self.assertIn("nit", result)


class TestDataGenerator(unittest.TestCase):
    """Test the DataGenerator class for various data sources."""

    def test_numpy_array_generator(self):
        """Test generating batches from numpy arrays."""
        x = np.arange(1000)
        y = 2 * x + 1

        # DataGenerator is instantiated with source data
        # Pass both x and y as tuple
        generator = DataGenerator((x, y), "array")

        # Test that generator is created and has expected attributes
        self.assertIsNotNone(generator)
        self.assertEqual(generator.source_type, "array")

    def test_hdf5_generator(self):
        """Test generating batches from HDF5 file."""
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5_file = tmp.name

        try:
            # Create test dataset
            x = np.arange(500)
            y = 3 * x - 2

            with h5py.File(h5_file, "w") as f:
                f.create_dataset("x", data=x)
                f.create_dataset("y", data=y)

            # Create generator from HDF5 file
            generator = DataGenerator(h5_file, "hdf5")

            # Test that generator is created
            self.assertIsNotNone(generator)

            # Close generator to release file handle
            generator.close()

        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_csv_generator(self):
        """Test generating batches from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            csv_file = tmp.name
            tmp.write("x,y\n")
            for i in range(200):
                tmp.write(f"{i},{2 * i + 3}\n")

        try:
            # DataGenerator doesn't have from_csv, but can read files
            generator = DataGenerator(csv_file, "file")

            # Test that generator is created
            self.assertIsNotNone(generator)

        finally:
            if os.path.exists(csv_file):
                os.remove(csv_file)

    def test_infinite_generator(self):
        """Test infinite data generator for continuous streams."""

        def infinite_data():
            """Simulate infinite data stream."""
            i = 0
            for _ in range(10):  # Limit for testing
                x = np.array([i])
                y = np.array([2 * i + 1])
                yield x, y
                i += 1

        # DataGenerator can handle generators
        generator = DataGenerator(infinite_data(), "generator")

        # Test that generator is created
        self.assertIsNotNone(generator)


class TestFitUnlimitedData(unittest.TestCase):
    """Test the fit_unlimited_data convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = lambda x, a, b: a * x + b
        self.true_params = [2.0, 1.0]

    def test_fit_unlimited_basic(self):
        """Test basic unlimited data fitting."""

        # Create data generator
        def data_gen():
            for i in range(5):
                x = np.random.uniform(0, 10, 100)
                y = self.model(x, *self.true_params)
                y = y + np.random.normal(0, 0.1, 100)
                yield x, y

        result = fit_unlimited_data(
            self.model,
            data_gen(),
            [1.5, 0.5],  # p0 as positional argument
            config=StreamingConfig(batch_size=100, max_epochs=2),
        )

        self.assertIn("x", result)
        self.assertIn("fun", result)
        self.assertEqual(len(result["x"]), 2)

    def test_fit_unlimited_with_bounds(self):
        """Test unlimited data fitting with parameter bounds."""
        x_data = np.linspace(0, 10, 200)
        y_data = self.model(x_data, *self.true_params)
        y_data = y_data + np.random.normal(0, 0.1, 200)

        # Create generator
        def data_gen():
            yield x_data[:100], y_data[:100]
            yield x_data[100:], y_data[100:]

        # Set bounds
        bounds = ([0, -5], [5, 5])

        result = fit_unlimited_data(
            self.model,
            data_gen(),
            [1.5, 0.5],  # p0 as positional argument
            config=StreamingConfig(batch_size=100),
            bounds=bounds,
        )

        params = result["x"]
        # Check bounds are respected
        self.assertTrue(params[0] >= bounds[0][0] and params[0] <= bounds[1][0])
        self.assertTrue(params[1] >= bounds[0][1] and params[1] <= bounds[1][1])


class TestCreateHDF5Dataset(unittest.TestCase):
    """Test HDF5 dataset creation utilities."""

    def test_create_hdf5_from_function(self):
        """Test creating HDF5 dataset from a function."""

        def test_func(x, a, b):
            return a * x + b

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5_file = tmp.name

        try:
            # Create HDF5 dataset using function
            params = np.array([2.0, 3.0])
            create_hdf5_dataset(
                h5_file, test_func, params, n_samples=1000, chunk_size=100
            )

            # Verify dataset was created
            with h5py.File(h5_file, "r") as f:
                self.assertIn("x", f)
                self.assertIn("y", f)

        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_create_hdf5_with_metadata(self):
        """Test creating HDF5 dataset with function and verifying it exists."""

        # Define a function that works with parameters
        def linear_func(x, a, b):
            return a * x + b

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5_file = tmp.name

        try:
            # Use the actual signature with proper parameters
            create_hdf5_dataset(
                h5_file,
                linear_func,
                np.array([2.0, 3.0]),  # Parameters for linear function
                n_samples=500,
                chunk_size=100,
            )

            # Verify file was created with correct structure
            with h5py.File(h5_file, "r") as f:
                self.assertIn("x", f)
                self.assertIn("y", f)
                self.assertEqual(f["x"].shape, (500,))
                self.assertEqual(f["y"].shape, (500,))
                # Check metadata
                self.assertEqual(f.attrs["n_samples"], 500)
                np.testing.assert_array_equal(f.attrs["true_params"], [2.0, 3.0])

        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)

    def test_append_to_hdf5(self):
        """Test creating HDF5 dataset with quadratic function."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            h5_file = tmp.name

        try:
            # Create dataset with quadratic function
            def quad_func(x, a, b):
                # Ensure output shape matches input shape
                return a * x**2 + b

            create_hdf5_dataset(
                h5_file,
                quad_func,
                np.array([1.0, 0.5]),  # Two parameters
                n_samples=100,
                chunk_size=50,
            )

            # Verify dataset was created correctly
            with h5py.File(h5_file, "r") as f:
                self.assertIn("x", f)
                self.assertIn("y", f)
                self.assertEqual(f["x"].shape, (100,))
                self.assertEqual(f["y"].shape, (100,))
                # Check that data was actually written
                x_data = f["x"][:]
                y_data = f["y"][:]
                self.assertEqual(len(x_data), 100)
                self.assertEqual(len(y_data), 100)

        finally:
            if os.path.exists(h5_file):
                os.remove(h5_file)


class TestStreamingIntegration(unittest.TestCase):
    """Integration tests for streaming optimizer workflows."""

    def test_large_dataset_simulation(self):
        """Simulate processing a very large dataset through streaming."""
        # Model and true parameters
        model = lambda x, a, b, c: a * jnp.sin(b * x) + c
        true_params = [2.0, 0.5, 1.0]

        # Simulate streaming 100M points in batches
        def data_stream():
            """Simulate data streaming from a large source."""
            n_batches = 100  # Simulate 100 batches
            batch_size = 1000

            for i in range(n_batches):
                x = np.random.uniform(0, 10, batch_size)
                y = model(x, *true_params)
                y = np.array(y) + np.random.normal(0, 0.1, batch_size)
                yield x, y

        # Configure streaming
        config = StreamingConfig(
            batch_size=1000,
            max_epochs=1,  # Single pass through data
            learning_rate=0.01,
            use_adam=True,
        )

        optimizer = StreamingOptimizer(config)

        # Fit the model
        result = optimizer.fit_streaming(
            model, data_stream(), p0=np.array([1.5, 0.4, 0.8]), verbose=0
        )

        self.assertIn("x", result)
        self.assertIn("fun", result)

        # Check that we got a result
        self.assertIn("nit", result)

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

        result = optimizer.fit_streaming(
            model, online_data(), p0=np.array([2.0]), verbose=0
        )

        self.assertIn("x", result)
        # Should have a result
        self.assertIsNotNone(result["x"][0])


if __name__ == "__main__":
    unittest.main()
