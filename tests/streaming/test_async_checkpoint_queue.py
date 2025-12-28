"""Tests for async checkpoint queue functionality.

This module tests the async checkpoint queue implementation (Task Group 2)
which eliminates 50-500ms blocking per checkpoint save by using a background
thread with a queue for HDF5 disk I/O.

Test coverage:
- Task 2.1: Checkpoint queue is created with maxsize=2
- Task 2.1: Checkpoint saves are non-blocking (main thread proceeds immediately)
- Task 2.1: Atomic file replacement (write to .tmp, then os.replace())
- Task 2.1: Graceful thread shutdown with sentinel value (None)
- Task 2.1: Queue full behavior (skip save with warning log)
"""

import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest import TestCase

import h5py
import numpy as np
import pytest

from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.optimizer import StreamingOptimizer


class TestAsyncCheckpointQueue(TestCase):
    """Tests for async checkpoint queue functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simple model for testing
        self.model_func = lambda x, a, b: a * np.exp(-b * x)

        # Create test data
        np.random.seed(42)
        self.n_samples = 500
        self.x_data = np.linspace(0, 5, self.n_samples)
        self.true_params = [2.5, 0.3]
        self.y_data = self.model_func(
            self.x_data, *self.true_params
        ) + 0.1 * np.random.randn(self.n_samples)

        self.p0 = np.array([1.0, 0.1])

        # Track optimizers for cleanup
        self._optimizers = []

    def _create_optimizer(self, config):
        """Create and track an optimizer for cleanup."""
        optimizer = StreamingOptimizer(config)
        self._optimizers.append(optimizer)
        return optimizer

    def tearDown(self):
        """Clean up test files and optimizer threads."""
        # Shut down all optimizer threads first
        for optimizer in self._optimizers:
            if hasattr(optimizer, "_shutdown_checkpoint_worker"):
                optimizer._shutdown_checkpoint_worker()
        self._optimizers.clear()

        # Then clean up temp files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_queue_maxsize(self):
        """Test that checkpoint queue is created with maxsize=2.

        Task 2.1: Verify the queue has the correct maxsize to limit memory
        overhead while allowing double buffering.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=10,
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Check that checkpoint queue exists and has correct maxsize
        self.assertTrue(hasattr(optimizer, "_checkpoint_queue"))
        self.assertEqual(optimizer._checkpoint_queue.maxsize, 2)

    def test_checkpoint_saves_are_non_blocking(self):
        """Test that checkpoint saves are non-blocking (main thread proceeds immediately).

        Task 2.1: The main optimization loop should not block during checkpoint
        saves. This test verifies that _save_checkpoint returns quickly without
        waiting for the HDF5 write to complete.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,  # Checkpoint every batch
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Initialize optimizer state needed for checkpoint
        optimizer.params = self.p0.copy()
        optimizer.best_params = self.p0.copy()
        optimizer.best_loss = 1.0
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0

        # Initialize optimizer state for Adam
        if config.use_adam:
            optimizer.m = np.zeros_like(self.p0)
            optimizer.v = np.zeros_like(self.p0)
        else:
            optimizer.velocity = np.zeros_like(self.p0)

        # Time how long _save_checkpoint takes
        start_time = time.time()
        optimizer._save_checkpoint(self.p0, [1.0, 0.9, 0.8])
        elapsed = time.time() - start_time

        # Non-blocking save should return very quickly (under 50ms)
        # Actual disk I/O happens in background thread
        self.assertLess(
            elapsed,
            0.05,
            f"_save_checkpoint took {elapsed:.3f}s, expected < 0.05s for non-blocking",
        )

        # Give the background thread time to complete
        time.sleep(0.5)

        # Verify checkpoint was actually saved
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_iter_*.h5"))
        self.assertGreater(
            len(checkpoint_files), 0, "Checkpoint file should be created"
        )

    def test_atomic_file_replacement(self):
        """Test atomic file replacement (write to .tmp, then os.replace()).

        Task 2.1: Checkpoint saves should use atomic file replacement to prevent
        partial/corrupt checkpoint files. The sync method writes to a .tmp file
        first, then atomically moves it to the final location.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Initialize state
        optimizer.params = self.p0.copy()
        optimizer.best_params = self.p0.copy()
        optimizer.best_loss = 1.0
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0

        if config.use_adam:
            optimizer.m = np.zeros_like(self.p0)
            optimizer.v = np.zeros_like(self.p0)
        else:
            optimizer.velocity = np.zeros_like(self.p0)

        # Save checkpoint
        optimizer._save_checkpoint(self.p0, [1.0])

        # Wait for background save to complete
        time.sleep(0.5)

        # Check that no .tmp files remain (they should be renamed)
        tmp_files = list(self.checkpoint_dir.glob("*.tmp"))
        self.assertEqual(len(tmp_files), 0, "No .tmp files should remain after save")

        # Check that checkpoint file exists and is valid HDF5
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_iter_*.h5"))
        self.assertGreater(len(checkpoint_files), 0)

        # Verify it's a valid HDF5 file
        with h5py.File(checkpoint_files[0], "r") as f:
            self.assertIn("version", f.attrs)
            self.assertIn("parameters", f)
            self.assertIn("progress", f)

    def test_graceful_thread_shutdown(self):
        """Test graceful thread shutdown with sentinel value (None).

        Task 2.1: The checkpoint worker thread should shut down gracefully when
        a sentinel value (None) is sent to the queue. This happens in __del__
        or cleanup methods.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=10,
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Verify thread is running
        self.assertTrue(hasattr(optimizer, "_checkpoint_thread"))
        self.assertTrue(optimizer._checkpoint_thread.is_alive())

        # Get thread reference before cleanup
        thread = optimizer._checkpoint_thread

        # Trigger cleanup
        optimizer._shutdown_checkpoint_worker()

        # Thread should stop after shutdown
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive(), "Thread should have stopped after shutdown")

    def test_queue_full_behavior(self):
        """Test queue full behavior (skip save with warning log).

        Task 2.1: When the checkpoint queue is full (maxsize=2), additional
        checkpoint saves should be skipped with a warning logged rather than
        blocking the main thread.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Initialize state
        optimizer.params = self.p0.copy()
        optimizer.best_params = self.p0.copy()
        optimizer.best_loss = 1.0
        optimizer.epoch = 0
        optimizer.batch_idx = 0

        if config.use_adam:
            optimizer.m = np.zeros_like(self.p0)
            optimizer.v = np.zeros_like(self.p0)
        else:
            optimizer.velocity = np.zeros_like(self.p0)

        # Pause the worker thread so queue fills up
        # We do this by setting a flag that causes the worker to sleep
        optimizer._checkpoint_worker_pause = True

        # Wait for worker to enter pause state (worker checks pause at loop start,
        # but may be blocked on queue.get() for up to 0.5s before it can loop)
        time.sleep(0.6)

        # Capture log output - use the specific logger name since warnings
        # are logged to nlsq.streaming.optimizer, not the root logger
        with self.assertLogs(
            logger="nlsq.streaming.optimizer", level="WARNING"
        ) as log_capture:
            # Try to queue more checkpoints than maxsize allows
            # Queue has maxsize=2, so 5 saves should overflow
            for i in range(5):
                optimizer.iteration = i + 1
                optimizer._save_checkpoint(self.p0, [float(i)])

            # Check that a warning about queue full was logged
            queue_full_logged = any(
                "queue" in msg.lower()
                and ("full" in msg.lower() or "skip" in msg.lower())
                for msg in log_capture.output
            )

        # Unpause worker
        optimizer._checkpoint_worker_pause = False

        # Should have logged warning about skipped saves
        self.assertTrue(
            queue_full_logged,
            f"Expected warning about queue full, got: {log_capture.output}",
        )


class TestAsyncCheckpointIntegration(TestCase):
    """Integration tests for async checkpoint with full optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = Path(self.temp_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Simple model for testing
        self.model_func = lambda x, a, b: a * np.exp(-b * x)

        # Create test data
        np.random.seed(42)
        self.n_samples = 500
        self.x_data = np.linspace(0, 5, self.n_samples)
        self.true_params = [2.5, 0.3]
        self.y_data = self.model_func(
            self.x_data, *self.true_params
        ) + 0.1 * np.random.randn(self.n_samples)

        self.p0 = np.array([1.0, 0.1])

        # Track optimizers for cleanup
        self._optimizers = []

    def _create_optimizer(self, config):
        """Create and track an optimizer for cleanup."""
        optimizer = StreamingOptimizer(config)
        self._optimizers.append(optimizer)
        return optimizer

    def tearDown(self):
        """Clean up test files and optimizer threads."""
        # Shut down all optimizer threads first
        for optimizer in self._optimizers:
            if hasattr(optimizer, "_shutdown_checkpoint_worker"):
                optimizer._shutdown_checkpoint_worker()
        self._optimizers.clear()

        # Then clean up temp files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_files_are_valid_hdf5(self):
        """Test that checkpoint files created by async queue are valid HDF5.

        Task 2.7: Verify checkpoint files are valid HDF5 with correct structure
        after a full optimization run.
        """
        config = StreamingConfig(
            batch_size=50,
            max_epochs=2,
            checkpoint_frequency=5,
            checkpoint_dir=str(self.checkpoint_dir),
            enable_checkpoints=True,
        )

        optimizer = self._create_optimizer(config)

        # Run optimization
        result = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for any pending checkpoints
        time.sleep(0.5)

        # Check checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_iter_*.h5"))
        self.assertGreater(len(checkpoint_files), 0, "Should have checkpoint files")

        # Verify each checkpoint is valid HDF5 with correct structure
        for checkpoint_path in checkpoint_files:
            with h5py.File(checkpoint_path, "r") as f:
                # Check version
                self.assertIn("version", f.attrs)
                self.assertEqual(f.attrs["version"], "2.0")

                # Check main groups
                self.assertIn("parameters", f)
                self.assertIn("optimizer_state", f)
                self.assertIn("progress", f)
                self.assertIn("diagnostics", f)

                # Check parameters
                self.assertIn("current", f["parameters"])
                self.assertIn("best", f["parameters"])

                # Check progress
                self.assertIn("iteration", f["progress"])
                self.assertIn("epoch", f["progress"])
                self.assertIn("batch_idx", f["progress"])
                self.assertIn("best_loss", f["progress"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
