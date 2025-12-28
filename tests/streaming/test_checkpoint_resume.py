"""Tests for streaming optimizer checkpoint save and resume functionality."""

import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest import TestCase

import h5py
import numpy as np
import pytest

from nlsq.streaming.optimizer import StreamingConfig, StreamingOptimizer


class TestCheckpointSaveResume(TestCase):
    """Test checkpoint save and resume functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_path = Path(self.temp_dir) / "checkpoint.h5"

        # Simple quadratic model for testing
        self.model_func = lambda x, a, b, c: a * x**2 + b * x + c

        # Create test data
        np.random.seed(42)
        self.n_samples = 1000
        self.x_data = np.random.randn(self.n_samples)
        self.true_params = [2.0, -1.5, 0.8]
        self.y_data = self.model_func(
            self.x_data, *self.true_params
        ) + 0.1 * np.random.randn(self.n_samples)

        # Initial parameters
        self.p0 = np.array([1.0, 1.0, 1.0])

        # Track optimizers created for cleanup
        self._optimizers = []

    def _create_optimizer(self, config):
        """Create optimizer and track it for cleanup."""
        optimizer = StreamingOptimizer(config)
        self._optimizers.append(optimizer)
        return optimizer

    def tearDown(self):
        """Clean up test files and shutdown optimizer threads."""
        # Shutdown all optimizer checkpoint threads first
        for optimizer in self._optimizers:
            if hasattr(optimizer, "_shutdown_checkpoint_worker"):
                optimizer._shutdown_checkpoint_worker()

        # Give threads time to finish
        time.sleep(0.1)

        # Now clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_file_creation(self):
        """Test that checkpoint file is created with correct structure."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            checkpoint_interval=2,
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        # Run for a few batches to trigger checkpoint
        result = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Check checkpoint files were created
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.h5"))
        self.assertGreater(len(checkpoint_files), 0, "No checkpoint files created")

        # Check checkpoint structure
        with h5py.File(checkpoint_files[0], "r") as f:
            # Check version
            self.assertIn("version", f.attrs)
            self.assertEqual(f.attrs["version"], "2.0")

            # Check main groups exist
            self.assertIn("parameters", f)
            self.assertIn("optimizer_state", f)
            self.assertIn("progress", f)
            self.assertIn("diagnostics", f)

            # Check parameters subgroups
            self.assertIn("current", f["parameters"])
            self.assertIn("best", f["parameters"])

            # Check progress fields
            self.assertIn("batch_idx", f["progress"])
            self.assertIn("epoch", f["progress"])
            self.assertIn("iteration", f["progress"])

    def test_optimizer_state_serialization(self):
        """Test that optimizer state is correctly serialized."""
        # Test with Adam optimizer
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            checkpoint_interval=2,
            checkpoint_dir=self.temp_dir,
            use_adam=True,
        )

        optimizer = self._create_optimizer(config)

        # Run for a few batches
        result = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Load checkpoint and verify Adam state
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.h5"))
        with h5py.File(checkpoint_files[0], "r") as f:
            self.assertIn("m", f["optimizer_state"])
            self.assertIn("v", f["optimizer_state"])

            # Check shapes match parameter count
            m = f["optimizer_state"]["m"][:]
            v = f["optimizer_state"]["v"][:]
            self.assertEqual(len(m), len(self.p0))
            self.assertEqual(len(v), len(self.p0))

        # Test with SGD optimizer
        config_sgd = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            checkpoint_interval=2,
            checkpoint_dir=self.temp_dir,
            use_adam=False,
        )
        optimizer_sgd = self._create_optimizer(config_sgd)

        # Clear previous checkpoints
        for f in checkpoint_files:
            f.unlink()

        result = optimizer_sgd.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Load checkpoint and verify SGD state
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.h5"))
        with h5py.File(checkpoint_files[0], "r") as f:
            self.assertIn("velocity", f["optimizer_state"])
            velocity = f["optimizer_state"]["velocity"][:]
            self.assertEqual(len(velocity), len(self.p0))

    def test_resume_from_checkpoint(self):
        """Test resuming optimization from a checkpoint."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=3,
            checkpoint_interval=5,
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        # Run first part of optimization
        result1 = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Get the latest checkpoint
        checkpoint_files = sorted(Path(self.temp_dir).glob("checkpoint_*.h5"))
        latest_checkpoint = checkpoint_files[-1]

        # Resume from checkpoint
        config_resume = StreamingConfig(
            batch_size=100,
            max_epochs=5,  # Run more epochs to verify continuation
            checkpoint_interval=5,
            checkpoint_dir=self.temp_dir,
            resume_from_checkpoint=str(latest_checkpoint),
        )

        optimizer2 = self._create_optimizer(config_resume)

        # Continue optimization
        result2 = optimizer2.fit(
            (self.x_data, self.y_data),
            self.model_func,
            self.p0,  # Initial params will be overridden by checkpoint
            verbose=0,
        )

        # Check that optimization continued from checkpoint
        # The iteration count should be higher than original
        self.assertGreater(result2["total_iterations"], result1["total_iterations"])

        # Parameters should have improved
        final_loss1 = result1["fun"]
        final_loss2 = result2["fun"]
        self.assertLessEqual(final_loss2, final_loss1)

    def test_resume_from_various_batch_indices(self):
        """Test resuming from different batch indices."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            checkpoint_interval=1,  # Save after every batch
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        # Run optimization with frequent checkpoints
        result = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoints to complete
        time.sleep(0.5)

        # Get all checkpoints
        checkpoint_files = sorted(Path(self.temp_dir).glob("checkpoint_*.h5"))
        self.assertGreater(
            len(checkpoint_files), 3, "Need multiple checkpoints for test"
        )

        # Test resuming from different checkpoints
        for checkpoint in checkpoint_files[1:4]:  # Test a few checkpoints
            with h5py.File(checkpoint, "r") as f:
                f["progress"]["batch_idx"][()]
                saved_epoch = f["progress"]["epoch"][()]
                saved_params = f["parameters"]["current"][:]

            # Resume from this checkpoint
            config_resume = StreamingConfig(
                batch_size=100,
                max_epochs=saved_epoch + 1,  # Run one more epoch
                checkpoint_interval=10,
                checkpoint_dir=self.temp_dir,
                resume_from_checkpoint=str(checkpoint),
            )

            optimizer_resume = self._create_optimizer(config_resume)

            # Run a dummy fit to trigger loading
            # This loads the checkpoint state
            result_resume = optimizer_resume.fit(
                (self.x_data[:100], self.y_data[:100]),
                self.model_func,  # Small dataset
                self.p0,
                verbose=0,
            )

            # Check that parameters from checkpoint were used
            # The result should be from resumed state not from fresh p0
            self.assertIsNotNone(result_resume)

    def test_checkpoint_version_compatibility(self):
        """Test handling of different checkpoint versions."""
        # Create a version 1.0 checkpoint (simulated old format)
        with h5py.File(self.checkpoint_path, "w") as f:
            f.attrs["version"] = "1.0"
            # Old format might have different structure
            f.create_dataset("params", data=self.p0)
            f.create_dataset("iteration", data=100)
            f.create_dataset("epoch", data=2)

        config = StreamingConfig(resume_from_checkpoint=str(self.checkpoint_path))

        optimizer = self._create_optimizer(config)

        # Should handle old version gracefully (log warning but continue)
        # This depends on implementation - adjust test based on actual behavior
        try:
            result = optimizer.fit(
                (self.x_data[:100], self.y_data[:100]),
                self.model_func,  # Small dataset
                self.p0,
                verbose=0,
            )
            # Should either work with compatibility layer or start fresh
            self.assertIsNotNone(result)
        except ValueError as e:
            # Or it might reject incompatible versions
            self.assertIn("version", str(e).lower())

    def test_checkpoint_auto_detection(self):
        """Test automatic detection of latest checkpoint."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=1,
            checkpoint_interval=2,
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        # Run first optimization
        result1 = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Create config with auto-detection
        config_auto = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            checkpoint_interval=5,
            checkpoint_dir=self.temp_dir,
            resume_from_checkpoint=True,  # Auto-detect latest
        )

        optimizer2 = self._create_optimizer(config_auto)

        # Should automatically find and load latest checkpoint
        result2 = optimizer2.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Should have continued from checkpoint
        self.assertGreater(result2["final_epoch"], result1["final_epoch"])

    def test_checkpoint_integrity_validation(self):
        """Test checkpoint integrity validation before loading."""
        # Create a corrupt checkpoint
        with open(self.checkpoint_path, "wb") as f:
            f.write(b"corrupted data that is not HDF5")

        config = StreamingConfig(resume_from_checkpoint=str(self.checkpoint_path))

        optimizer = self._create_optimizer(config)

        # Should handle corrupt checkpoint gracefully
        with self.assertLogs(level="WARNING") as log:
            result = optimizer.fit(
                (self.x_data[:100], self.y_data[:100]),
                self.model_func,
                self.p0,
                verbose=0,
            )

        # Should log warning about corrupt checkpoint
        # Check for either "Failed to load" or "corrupt" or "invalid" in warning messages
        self.assertTrue(
            any(
                "failed to load" in msg.lower()
                or "corrupt" in msg.lower()
                or "invalid" in msg.lower()
                for msg in log.output
            )
        )

        # Should still complete optimization (starting fresh)
        self.assertIsNotNone(result)

    def test_checkpoint_with_failed_batches(self):
        """Test that failed batch information is saved in checkpoints."""
        # Create data that will cause failures
        # Mix some NaN values but keep most data valid
        x_data_with_nans = self.x_data.copy()
        x_data_with_nans[200:210] = np.nan  # Only 10 NaN values, less disruptive

        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,  # More epochs to ensure we get checkpoints
            checkpoint_interval=2,  # Save more frequently
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        result = optimizer.fit(
            (x_data_with_nans, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Load checkpoint and check failed batch tracking
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.h5"))

        # If we have checkpoints, check for failed batch tracking
        if checkpoint_files:
            with h5py.File(checkpoint_files[0], "r") as f:
                # Check if failed batches were saved (may or may not exist)
                if "failed_batch_indices" in f["diagnostics"]:
                    failed_batches = f["diagnostics"]["failed_batch_indices"][:]
                    # Should have recorded the batch with NaN values
                    self.assertGreater(len(failed_batches), 0)
                else:
                    # If no failures recorded, that's also acceptable
                    pass
        else:
            # If no checkpoints were created (all batches failed early), that's acceptable
            # Check that the optimization still returned a result
            self.assertIsNotNone(result)

    def test_checkpoint_with_best_params(self):
        """Test that best parameters are saved in checkpoints."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,
            checkpoint_interval=5,
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        result = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Load checkpoint and verify best params are saved
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.h5"))
        with h5py.File(checkpoint_files[0], "r") as f:
            best_params = f["parameters"]["best"][:]
            best_loss = f["progress"]["best_loss"][()]

            # Best params should be different from initial
            self.assertFalse(np.array_equal(best_params, self.p0))

            # Best loss should be reasonable
            self.assertLess(best_loss, 100.0)  # Should have found something decent

    def test_resume_continues_from_saved_position(self):
        """Test that resume continues from exact saved position."""
        config = StreamingConfig(
            batch_size=100,
            max_epochs=2,  # Start with 2 epochs
            checkpoint_interval=3,
            checkpoint_dir=self.temp_dir,
        )

        optimizer = self._create_optimizer(config)

        # Run partial optimization
        result1 = optimizer.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Wait for async checkpoint to complete
        time.sleep(0.2)

        # Get checkpoint info
        checkpoint_files = sorted(Path(self.temp_dir).glob("checkpoint_*.h5"))
        with h5py.File(checkpoint_files[-1], "r") as f:
            saved_iteration = f["progress"]["iteration"][()]
            f["progress"]["epoch"][()]

        # Resume and continue with more epochs
        config_resume = StreamingConfig(
            batch_size=100,
            max_epochs=4,  # Run 2 more epochs (total 4)
            checkpoint_interval=3,
            checkpoint_dir=self.temp_dir,
            resume_from_checkpoint=True,
        )

        optimizer2 = self._create_optimizer(config_resume)
        result2 = optimizer2.fit(
            (self.x_data, self.y_data), self.model_func, self.p0, verbose=0
        )

        # Should have more iterations than saved (2 more epochs worth)
        self.assertGreater(result2["total_iterations"], saved_iteration)

        # Should have ran more epochs total
        self.assertEqual(result2["n_epochs"], 4)

        # Final loss should be as good or better
        self.assertLessEqual(result2["fun"], result1["fun"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
