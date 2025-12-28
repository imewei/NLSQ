"""Benchmark tests for checkpoint I/O latency.

Task 6.2: Compare sync vs async checkpoint I/O latency.

This module benchmarks:
- Sync checkpoint save latency (blocking)
- Async checkpoint save latency (non-blocking via background thread)
- Blocking time during optimization loop
- Verification that async saves do not block main thread

Expected results:
- Async saves should return in <50ms (non-blocking)
- Sync saves may take 50-500ms depending on data size
- Main optimization loop should not be blocked during checkpoint saves

Uses pytest-benchmark for consistent measurement methodology.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.config import StreamingConfig
from nlsq.streaming.optimizer import StreamingOptimizer


@pytest.fixture
def checkpoint_test_setup():
    """Set up test environment for checkpoint benchmarks."""
    temp_dir = tempfile.mkdtemp()
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Test data
    np.random.seed(42)
    n_samples = 1000
    x_data = np.linspace(0, 5, n_samples)
    true_params = [2.5, 0.3]
    y_data = true_params[0] * np.exp(-true_params[1] * x_data) + 0.1 * np.random.randn(
        n_samples
    )
    p0 = np.array([1.0, 0.1])

    yield {
        "temp_dir": temp_dir,
        "checkpoint_dir": checkpoint_dir,
        "x_data": x_data,
        "y_data": y_data,
        "p0": p0,
    }

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def model_func(x, a, b):
    """Simple exponential model for benchmarking.

    Uses jax.numpy for JIT compatibility with the streaming optimizer.
    """
    return a * jnp.exp(-b * x)


class TestAsyncCheckpointLatency:
    """Benchmark async checkpoint save latency."""

    @pytest.mark.benchmark(group="checkpoint_latency")
    def test_async_checkpoint_call_latency(self, benchmark, checkpoint_test_setup):
        """Benchmark the time for _save_checkpoint to return (async).

        The async implementation should return quickly without waiting
        for the actual disk I/O to complete.
        """
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
        )

        optimizer = StreamingOptimizer(config)

        # Initialize optimizer state needed for checkpoint
        optimizer.params = setup["p0"].copy()
        optimizer.best_params = setup["p0"].copy()
        optimizer.best_loss = 1.0
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0
        optimizer.losses = [1.0, 0.9, 0.8]

        if config.use_adam:
            optimizer.m = np.zeros_like(setup["p0"])
            optimizer.v = np.zeros_like(setup["p0"])
        else:
            optimizer.velocity = np.zeros_like(setup["p0"])

        def save_checkpoint():
            optimizer._save_checkpoint(setup["p0"], [1.0, 0.9, 0.8])
            optimizer.iteration += 1  # Increment to avoid filename collision

        benchmark(save_checkpoint)

        # Cleanup: wait for background thread to complete
        time.sleep(0.5)

    def test_async_checkpoint_non_blocking(self, checkpoint_test_setup):
        """Verify async checkpoint saves do not block main thread.

        The _save_checkpoint call should return in under 50ms even if
        the actual disk I/O takes longer.
        """
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
        )

        optimizer = StreamingOptimizer(config)

        # Initialize state
        optimizer.params = setup["p0"].copy()
        optimizer.best_params = setup["p0"].copy()
        optimizer.best_loss = 1.0
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0
        optimizer.losses = [1.0, 0.9, 0.8]

        if config.use_adam:
            optimizer.m = np.zeros_like(setup["p0"])
            optimizer.v = np.zeros_like(setup["p0"])
        else:
            optimizer.velocity = np.zeros_like(setup["p0"])

        # Measure call latency
        latencies = []
        for i in range(5):
            optimizer.iteration = i + 1
            start = time.perf_counter()
            optimizer._save_checkpoint(setup["p0"], [1.0, 0.9, 0.8])
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        # Wait for background thread to complete all saves
        time.sleep(1.0)

        mean_latency = np.mean(latencies)
        max_latency = np.max(latencies)

        print(f"\n[Async Checkpoint] Mean call latency: {mean_latency * 1000:.2f}ms")
        print(f"[Async Checkpoint] Max call latency: {max_latency * 1000:.2f}ms")

        # Verify non-blocking behavior
        assert max_latency < 0.1, (
            f"Async save should return in <100ms, got {max_latency * 1000:.2f}ms"
        )

        # Verify checkpoints were actually created
        checkpoint_files = list(setup["checkpoint_dir"].glob("checkpoint_iter_*.h5"))
        assert len(checkpoint_files) > 0, "Checkpoint files should be created"


class TestCheckpointDuringOptimization:
    """Benchmark checkpoint impact during optimization loop."""

    @pytest.mark.benchmark(group="checkpoint_optimization")
    def test_optimization_with_checkpoints(self, benchmark, checkpoint_test_setup):
        """Benchmark optimization throughput with checkpoints enabled."""
        setup = checkpoint_test_setup

        def run_with_checkpoints():
            config = StreamingConfig(
                batch_size=50,
                max_epochs=2,
                checkpoint_frequency=5,  # Checkpoint every 5 iterations
                checkpoint_dir=str(setup["checkpoint_dir"]),
                enable_checkpoints=True,
                enable_fault_tolerance=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit(
                (setup["x_data"], setup["y_data"]),
                model_func,
                p0=setup["p0"],
                verbose=0,
            )
            return result

        result = benchmark(run_with_checkpoints)
        assert result["success"], "Optimization should succeed"

        # Cleanup checkpoint files between runs
        for f in setup["checkpoint_dir"].glob("checkpoint_iter_*.h5"):
            f.unlink()

    @pytest.mark.benchmark(group="checkpoint_optimization")
    def test_optimization_without_checkpoints(self, benchmark, checkpoint_test_setup):
        """Benchmark optimization throughput without checkpoints (baseline)."""
        setup = checkpoint_test_setup

        def run_without_checkpoints():
            config = StreamingConfig(
                batch_size=50,
                max_epochs=2,
                enable_checkpoints=False,
                enable_fault_tolerance=False,
            )
            optimizer = StreamingOptimizer(config)
            result = optimizer.fit(
                (setup["x_data"], setup["y_data"]),
                model_func,
                p0=setup["p0"],
                verbose=0,
            )
            return result

        result = benchmark(run_without_checkpoints)
        assert result["success"], "Optimization should succeed"


class TestCheckpointBlockingTime:
    """Measure blocking time during checkpoint saves."""

    def test_measure_blocking_time_async(self, checkpoint_test_setup):
        """Measure total blocking time from async checkpoint saves.

        With async implementation, blocking time should be minimal
        (just the time to copy state and enqueue).
        """
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=3,
            checkpoint_frequency=5,
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
            enable_fault_tolerance=False,
        )
        optimizer = StreamingOptimizer(config)

        # Track time spent in checkpoint saves during optimization
        start_time = time.perf_counter()
        result = optimizer.fit(
            (setup["x_data"], setup["y_data"]), model_func, p0=setup["p0"], verbose=0
        )
        total_time = time.perf_counter() - start_time

        # Get checkpoint save times from optimizer (if tracked)
        if (
            hasattr(optimizer, "checkpoint_save_times")
            and optimizer.checkpoint_save_times
        ):
            total_checkpoint_time = sum(optimizer.checkpoint_save_times)
            checkpoint_percentage = (total_checkpoint_time / total_time) * 100
            print(
                f"\n[Async] Total checkpoint blocking time: {total_checkpoint_time * 1000:.2f}ms"
            )
            print(
                f"[Async] Checkpoint overhead: {checkpoint_percentage:.1f}% of total time"
            )
        else:
            print("\n[Async] Checkpoint timing not tracked in this run")

        print(f"[Async] Total optimization time: {total_time * 1000:.2f}ms")
        assert result["success"], "Optimization should succeed"


class TestCheckpointQueueBehavior:
    """Verify checkpoint queue behavior under load."""

    def test_queue_full_behavior(self, checkpoint_test_setup):
        """Test behavior when checkpoint queue is full.

        When the queue is full (maxsize=2), new checkpoint requests
        should be skipped with a warning rather than blocking.
        """
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,  # Checkpoint every iteration
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
        )

        optimizer = StreamingOptimizer(config)

        # Initialize state
        optimizer.params = setup["p0"].copy()
        optimizer.best_params = setup["p0"].copy()
        optimizer.best_loss = 1.0
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0
        optimizer.losses = [1.0]

        if config.use_adam:
            optimizer.m = np.zeros_like(setup["p0"])
            optimizer.v = np.zeros_like(setup["p0"])
        else:
            optimizer.velocity = np.zeros_like(setup["p0"])

        # Pause the worker thread to fill up the queue
        optimizer._checkpoint_worker_pause = True

        # Try to fill the queue beyond capacity
        latencies = []
        for i in range(5):
            optimizer.iteration = i + 1
            start = time.perf_counter()
            optimizer._save_checkpoint(setup["p0"], [1.0])
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        # Resume worker and wait for it to drain
        optimizer._checkpoint_worker_pause = False
        time.sleep(1.0)

        # All calls should return quickly (non-blocking)
        max_latency = max(latencies)
        print(f"\n[Queue Full Test] Max call latency: {max_latency * 1000:.2f}ms")
        print(
            f"[Queue Full Test] All latencies: {[f'{l * 1000:.2f}ms' for l in latencies]}"
        )

        # Even when queue is full, calls should not block
        assert max_latency < 0.5, (
            f"All save calls should return quickly, got {max_latency * 1000:.2f}ms"
        )

    def test_queue_thread_shutdown(self, checkpoint_test_setup):
        """Verify checkpoint worker thread shuts down gracefully."""
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
        )

        optimizer = StreamingOptimizer(config)

        # Verify thread is running
        assert optimizer._checkpoint_thread.is_alive(), (
            "Worker thread should be running"
        )

        # Trigger shutdown
        optimizer._shutdown_checkpoint_worker()

        # Give thread time to shut down
        time.sleep(0.5)

        # Thread should have stopped
        assert not optimizer._checkpoint_thread.is_alive(), (
            "Worker thread should have stopped after shutdown"
        )


class TestCheckpointFileIntegrity:
    """Verify checkpoint files are valid after async save."""

    def test_checkpoint_file_integrity(self, checkpoint_test_setup):
        """Verify async checkpoint produces valid HDF5 files."""
        setup = checkpoint_test_setup

        config = StreamingConfig(
            batch_size=50,
            max_epochs=1,
            checkpoint_frequency=1,
            checkpoint_dir=str(setup["checkpoint_dir"]),
            enable_checkpoints=True,
        )

        optimizer = StreamingOptimizer(config)

        # Initialize state
        test_params = np.array([2.0, 0.5])
        optimizer.params = test_params.copy()
        optimizer.best_params = test_params.copy()
        optimizer.best_loss = 0.5
        optimizer.iteration = 1
        optimizer.epoch = 0
        optimizer.batch_idx = 0
        optimizer.losses = [1.0, 0.8, 0.5]

        if config.use_adam:
            optimizer.m = np.zeros_like(test_params)
            optimizer.v = np.zeros_like(test_params)
        else:
            optimizer.velocity = np.zeros_like(test_params)

        # Save checkpoint
        optimizer._save_checkpoint(test_params, [1.0, 0.8, 0.5])

        # Wait for background thread to complete
        time.sleep(1.0)

        # Find and verify checkpoint file
        checkpoint_files = list(setup["checkpoint_dir"].glob("checkpoint_iter_*.h5"))
        assert len(checkpoint_files) == 1, "Should have exactly one checkpoint file"

        checkpoint_path = checkpoint_files[0]
        with h5py.File(checkpoint_path, "r") as f:
            # Verify required fields exist (version 2.0 format)
            assert "parameters" in f, "Checkpoint should contain parameters group"
            assert "parameters/current" in f, (
                "Checkpoint should contain parameters/current"
            )
            assert "parameters/best" in f, "Checkpoint should contain parameters/best"
            assert "progress" in f, "Checkpoint should contain progress group"
            assert "progress/iteration" in f, (
                "Checkpoint should contain progress/iteration"
            )
            assert "progress/epoch" in f, "Checkpoint should contain progress/epoch"

            # Verify values
            saved_params = f["parameters/current"][:]
            np.testing.assert_array_almost_equal(
                saved_params, test_params, err_msg="Saved params should match"
            )
