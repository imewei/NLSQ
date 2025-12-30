"""Tests for the NLSQ logging system."""

import logging
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from nlsq.utils.logging import (
    LogLevel,
    NLSQLogger,
    enable_debug_mode,
    enable_performance_tracking,
    get_logger,
    set_global_level,
)


class TestNLSQLogger(unittest.TestCase):
    """Test the NLSQLogger class."""

    def setUp(self):
        """Set up test environment."""
        # Clean up environment variables
        self.original_env = {}
        for key in [
            "NLSQ_DEBUG",
            "NLSQ_VERBOSE",
            "NLSQ_TRACE_JAX",
            "NLSQ_SAVE_ITERATIONS",
            "NLSQ_LOG_DIR",
        ]:
            self.original_env[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]

    def test_logger_creation(self):
        """Test logger creation and naming."""
        logger = NLSQLogger("test_module")
        self.assertIsInstance(logger, NLSQLogger)
        self.assertEqual(logger.name, "nlsq.test_module")

    def test_get_logger_function(self):
        """Test get_logger convenience function."""
        logger = get_logger("test_module")
        self.assertIsInstance(logger, NLSQLogger)
        self.assertEqual(logger.name, "nlsq.test_module")

        # Should return same instance
        logger2 = get_logger("test_module")
        self.assertIs(logger, logger2)

    def test_log_levels(self):
        """Test custom log levels."""
        NLSQLogger("level_test", level=LogLevel.DEBUG)

        # Should have the custom PERFORMANCE level
        self.assertEqual(LogLevel.PERFORMANCE, 25)
        self.assertTrue(LogLevel.DEBUG < LogLevel.PERFORMANCE < LogLevel.WARNING)

    def test_timer_context_manager(self):
        """Test timer context manager records time correctly."""
        logger = NLSQLogger("timer_test")

        with logger.timer("test_operation", log_result=False):
            time.sleep(0.01)  # Sleep for 10ms

        self.assertIn("test_operation", logger.timers)
        self.assertGreaterEqual(logger.timers["test_operation"], 0.01)
        # Relaxed from 0.1 to 0.15 to account for CI timing variance
        self.assertLess(
            logger.timers["test_operation"], 0.15
        )  # Should be less than 150ms (was 100ms)

    def test_timer_with_exception(self):
        """Test timer context manager handles exceptions."""
        logger = NLSQLogger("timer_exception_test")

        try:
            with logger.timer("failing_operation", log_result=False):
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Timer should still be recorded
        self.assertIn("failing_operation", logger.timers)
        self.assertGreaterEqual(logger.timers["failing_operation"], 0.01)

    def test_optimization_step_tracking(self):
        """Test optimization step logging."""
        logger = NLSQLogger("opt_test")

        logger.optimization_step(
            iteration=1, cost=0.5, gradient_norm=0.1, step_size=0.01, nfev=10
        )

        self.assertEqual(len(logger.optimization_history), 1)

        history = logger.optimization_history[0]
        self.assertEqual(history["iter"], 1)
        self.assertEqual(history["cost"], 0.5)
        self.assertEqual(history["grad_norm"], 0.1)
        self.assertEqual(history["step"], 0.01)
        self.assertEqual(history["nfev"], 10)
        self.assertIn("timestamp", history)

    def test_multiple_optimization_steps(self):
        """Test tracking multiple optimization steps."""
        logger = NLSQLogger("multi_opt_test")

        for i in range(5):
            logger.optimization_step(
                iteration=i, cost=1.0 / (i + 1), gradient_norm=0.1 / (i + 1)
            )

        self.assertEqual(len(logger.optimization_history), 5)
        self.assertEqual(logger.optimization_history[-1]["iter"], 4)

    def test_convergence_logging(self):
        """Test convergence information logging."""
        logger = NLSQLogger("convergence_test")

        # This should not raise an exception
        logger.convergence(
            reason="ftol satisfied",
            iterations=10,
            final_cost=1e-6,
            time_elapsed=2.5,
            final_gradient_norm=1e-8,
        )

    def test_matrix_info(self):
        """Test matrix information logging."""
        logger = NLSQLogger("matrix_test")

        matrix = np.random.randn(10, 10)

        # This should not raise an exception
        logger.matrix_info("test_matrix", matrix, compute_condition=True)

        # Test with 1D array
        vector = np.random.randn(10)
        logger.matrix_info("test_vector", vector, compute_condition=False)

    def test_jax_compilation_logging(self):
        """Test JAX compilation event logging."""
        logger = NLSQLogger("jax_test")

        # Without environment variable, should not log
        logger.jax_compilation("test_function", input_shape=(100, 10))

        # With environment variable, should log
        os.environ["NLSQ_TRACE_JAX"] = "1"
        logger.jax_compilation(
            "test_function", input_shape=(100, 10), compilation_time=0.5
        )

    def test_debug_mode_file_creation(self):
        """Test that debug mode creates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["NLSQ_DEBUG"] = "1"
            os.environ["NLSQ_LOG_DIR"] = tmpdir

            logger = NLSQLogger("debug_test")
            logger.debug("Test debug message")
            logger.info("Test info message")

            # Close logger handlers to release file handles BEFORE reading (needed on Windows)
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)

            # Check that log file was created
            log_files = [f for f in os.listdir(tmpdir) if f.startswith("nlsq_debug_")]
            self.assertEqual(len(log_files), 1)

            # Verify content was written
            log_path = Path(tmpdir) / log_files[0]
            with open(log_path) as f:
                content = f.read()
                self.assertIn("Test debug message", content)
                self.assertIn("Test info message", content)

    def test_verbose_mode(self):
        """Test verbose mode logging."""
        os.environ["NLSQ_VERBOSE"] = "1"
        logger = NLSQLogger("verbose_test")

        # In verbose mode, info messages should be shown
        # We can't easily test console output, but we can verify the setup
        handler = logger.logger.handlers[0]
        self.assertEqual(handler.level, logging.INFO)

    def test_save_iteration_data(self):
        """Test saving optimization history to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = NLSQLogger("save_test")

            # Add some optimization history
            for i in range(3):
                logger.optimization_step(
                    iteration=i, cost=1.0 / (i + 1), gradient_norm=0.1 / (i + 1)
                )

            # Save to file
            logger.save_iteration_data(output_dir=tmpdir)

            # Check that file was created
            npz_files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
            self.assertEqual(len(npz_files), 1)

            # Load and verify content
            with np.load(Path(tmpdir) / npz_files[0]) as data:
                self.assertIn("iter", data)
                self.assertIn("timestamp", data)
                self.assertEqual(len(data["iter"]), 3)
                np.testing.assert_array_equal(data["iter"], [0, 1, 2])

    def test_save_iteration_data_with_env_var(self):
        """Test saving with environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["NLSQ_SAVE_ITERATIONS"] = tmpdir
            logger = NLSQLogger("env_save_test")

            logger.optimization_step(iteration=0, cost=1.0)
            logger.save_iteration_data()

            npz_files = [f for f in os.listdir(tmpdir) if f.endswith(".npz")]
            self.assertEqual(len(npz_files), 1)

    def test_structured_logging(self):
        """Test structured logging with kwargs."""
        logger = NLSQLogger("structured_test")

        # Test various log methods with structured data
        logger.debug("Debug message", key1="value1", key2=42)
        logger.info("Info message", data={"nested": "dict"})
        logger.warning("Warning message", array=[1, 2, 3])
        logger.error("Error message", exc_info=False, error_code=123)

        # Should not raise any exceptions

    def test_performance_logging(self):
        """Test performance-specific logging."""
        logger = NLSQLogger("perf_test")

        logger.performance("Performance metric", latency_ms=10.5, throughput=1000)

        # Should use the PERFORMANCE log level
        # We can't easily verify the output, but it shouldn't raise errors

    def test_set_global_level(self):
        """Test setting global log level."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")

        # Set global level to ERROR
        set_global_level(LogLevel.ERROR)

        # Both loggers should have ERROR level
        self.assertEqual(logger1.logger.level, LogLevel.ERROR)
        self.assertEqual(logger2.logger.level, LogLevel.ERROR)

    def test_enable_debug_mode(self):
        """Test enabling debug mode."""
        # Create logger before enabling debug mode
        logger = get_logger("debug_mode_test")

        # Enable debug mode
        enable_debug_mode()

        # Should set environment variable and change log level
        self.assertEqual(os.environ.get("NLSQ_DEBUG"), "1")
        self.assertEqual(logger.logger.level, LogLevel.DEBUG)

    def test_enable_performance_tracking(self):
        """Test enabling performance tracking."""
        enable_performance_tracking()

        # Should set environment variables
        self.assertEqual(os.environ.get("NLSQ_TRACE_JAX"), "1")
        self.assertEqual(os.environ.get("NLSQ_SAVE_ITERATIONS"), "1")


class TestLoggerIntegration(unittest.TestCase):
    """Integration tests for logger with NLSQ components."""

    def test_logger_with_optimization(self):
        """Test logger integration with optimization workflow."""
        logger = get_logger("integration_test")

        # Simulate an optimization run
        with logger.timer("optimization"):
            for i in range(5):
                logger.optimization_step(
                    iteration=i,
                    cost=100 / (i + 1),
                    gradient_norm=10 / (i + 1),
                    step_size=0.1,
                    nfev=i * 2,
                )
                time.sleep(0.001)

            logger.convergence(
                reason="ftol satisfied",
                iterations=5,
                final_cost=20.0,
                time_elapsed=logger.timers["optimization"],
            )

        # Verify history
        self.assertEqual(len(logger.optimization_history), 5)
        self.assertIn("optimization", logger.timers)

    def test_multiple_logger_instances(self):
        """Test multiple logger instances work independently."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        logger1.optimization_step(iteration=1, cost=1.0)
        logger2.optimization_step(iteration=1, cost=2.0)

        # Should have independent histories
        self.assertEqual(len(logger1.optimization_history), 1)
        self.assertEqual(len(logger2.optimization_history), 1)
        self.assertEqual(logger1.optimization_history[0]["cost"], 1.0)
        self.assertEqual(logger2.optimization_history[0]["cost"], 2.0)


if __name__ == "__main__":
    unittest.main()
