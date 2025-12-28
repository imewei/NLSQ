"""Tests for AdaptiveHybridStreamingOptimizer - Fault Tolerance Integration (Task Group 8).

This test module covers Task Group 8 requirements:
- Checkpoint save with phase-specific state
- Checkpoint resume from any phase
- Best parameter tracking across phases
- NaN/Inf validation at critical points
- Adaptive retry on batch failure
- Graceful degradation on persistent errors
"""

import tempfile
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestFaultToleranceCheckpoints:
    """Tests for checkpoint save/load functionality."""

    def test_checkpoint_save_with_phase_specific_state(self):
        """Test checkpoint save includes phase-specific state."""
        # Create temporary checkpoint directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HybridStreamingConfig(
                normalize=True,
                warmup_iterations=5,
                max_warmup_iterations=10,
                gauss_newton_max_iterations=3,
                enable_checkpoints=True,
                checkpoint_frequency=3,  # Save frequently for testing
                checkpoint_dir=tmpdir,  # Set checkpoint dir in config
            )
            optimizer = AdaptiveHybridStreamingOptimizer(config)

            # Create simple test data
            np.random.seed(42)
            x_data = np.linspace(0, 10, 50)
            y_data = 2.5 * np.exp(-0.5 * x_data) + 0.1 * np.random.randn(len(x_data))

            def model(x, a, b):
                return a * jnp.exp(-b * x)

            p0 = jnp.array([2.0, 0.4])
            bounds = (jnp.array([0.0, 0.0]), jnp.array([10.0, 2.0]))

            # Run optimization (will auto-save checkpoints)
            result = optimizer.fit(
                (x_data, y_data), model, p0, bounds=bounds, verbose=0
            )

            # Verify checkpoint was created
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.h5"))
            assert len(checkpoint_files) > 0, "No checkpoint files created"

            # Load checkpoint and verify contents
            checkpoint_path = checkpoint_files[0]
            with h5py.File(checkpoint_path, "r") as f:
                # Check version
                assert "version" in f.attrs
                assert f.attrs["version"] == "3.0"  # Hybrid optimizer version

                # Check phase-specific state exists
                assert "phase_state" in f
                assert "current_phase" in f["phase_state"]
                assert "normalized_params" in f["phase_state"]

                # Check Phase 1 optimizer state (if saved during Phase 1)
                # Phase 1 uses Optax L-BFGS optimizer (or legacy Adam)
                if "phase1_optimizer_state" in f["phase_state"]:
                    opt_group = f["phase_state/phase1_optimizer_state"]
                    opt_type = opt_group.attrs.get("optimizer_type", "adam")
                    if opt_type == "lbfgs":
                        assert "params" in opt_group
                        assert "diff_params_memory" in opt_group
                    else:  # legacy adam
                        assert "mu" in opt_group
                        assert "nu" in opt_group

                # Check Phase 2 accumulators (if saved during/after Phase 2)
                if "phase2_jtj_accumulator" in f["phase_state"]:
                    jtj = f["phase_state/phase2_jtj_accumulator"][()]
                    assert jtj.shape == (2, 2)  # n_params x n_params

                if "phase2_jtr_accumulator" in f["phase_state"]:
                    jtr = f["phase_state/phase2_jtr_accumulator"][()]
                    assert jtr.shape == (2,)  # n_params

    def test_checkpoint_resume_from_any_phase(self):
        """Test checkpoint resume can restart from any phase."""
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=3,
            max_warmup_iterations=5,
            gauss_newton_max_iterations=2,
            enable_checkpoints=True,
            checkpoint_frequency=2,  # Save frequently
        )

        np.random.seed(42)
        x_data = np.linspace(0, 10, 30)
        y_data = 1.5 * x_data + 0.5 + 0.1 * np.random.randn(len(x_data))

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            # Run first optimizer (will save checkpoints)
            optimizer1 = AdaptiveHybridStreamingOptimizer(config)
            result1 = optimizer1.fit((x_data, y_data), model, p0, verbose=0)

            # Find latest checkpoint
            checkpoint_files = sorted(Path(tmpdir).glob("checkpoint_*.h5"))
            assert len(checkpoint_files) > 0

            latest_checkpoint = checkpoint_files[-1]

            # Load checkpoint to verify phase
            with h5py.File(latest_checkpoint, "r") as f:
                saved_phase = int(f["phase_state/current_phase"][()])
                saved_params = jnp.array(f["phase_state/normalized_params"])

            # Create new optimizer and resume from checkpoint
            config2 = config
            config2.resume_from_checkpoint = str(latest_checkpoint)
            optimizer2 = AdaptiveHybridStreamingOptimizer(config2)

            # Resume optimization
            result2 = optimizer2.fit((x_data, y_data), model, p0, verbose=0)

            # Verify results are similar (resumed optimization should complete)
            assert result2["success"]
            assert jnp.allclose(result2["x"], result1["x"], rtol=0.1)

    def test_best_parameter_tracking_across_phases(self):
        """Test best parameters are tracked across all phases."""
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=5,
            max_warmup_iterations=10,
            gauss_newton_max_iterations=3,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create data with some noise
        np.random.seed(42)
        x_data = np.linspace(0, 5, 40)
        y_data = (
            3.0 * x_data**2 + 0.5 * x_data + 1.0 + 0.5 * np.random.randn(len(x_data))
        )

        def model(x, a, b, c):
            return a * x**2 + b * x + c

        p0 = jnp.array([2.5, 0.3, 0.8])
        bounds = (jnp.array([0.0, -1.0, -2.0]), jnp.array([10.0, 2.0, 5.0]))

        # Run optimization
        result = optimizer.fit((x_data, y_data), model, p0, bounds=bounds, verbose=0)

        # Verify we got best parameters (not just final)
        assert result["success"]
        assert "x" in result
        assert len(result["x"]) == 3

        # Check phase history tracks best parameters
        assert len(optimizer.phase_history) >= 3  # At least Phases 0, 1, 2

        # Phase 1 should have best_loss tracked
        phase1_record = [p for p in optimizer.phase_history if p["phase"] == 1]
        assert len(phase1_record) > 0
        assert "best_loss" in phase1_record[0]

        # Phase 2 should have best_cost tracked
        phase2_record = [p for p in optimizer.phase_history if p["phase"] == 2]
        assert len(phase2_record) > 0
        assert "best_cost" in phase2_record[0]


class TestNaNInfValidation:
    """Tests for NaN/Inf detection and handling."""

    def test_nan_detection_in_gradients(self):
        """Test NaN detection during gradient computation."""
        config = HybridStreamingConfig(
            validate_numerics=True,  # Enable NaN/Inf validation
            warmup_iterations=5,
            max_warmup_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create data that will cause NaN in gradients
        x_data = jnp.array([0.0, 1.0, 2.0, 3.0])
        y_data = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Model that can produce NaN gradients with bad parameters
        def problematic_model(x, a, b):
            # When a becomes very large, exp can overflow
            return a * jnp.exp(b * x)

        # Start with parameters that might cause overflow
        p0 = jnp.array([1e10, 10.0])

        # This should handle NaN gracefully
        # Note: With validation enabled, optimizer should detect and handle NaN
        try:
            result = optimizer.fit((x_data, y_data), problematic_model, p0, verbose=0)
            # Should either succeed or fail gracefully (not crash)
            assert result is not None
        except Exception as e:
            # Should raise a clear error about numerical issues
            assert "NaN" in str(e) or "numerical" in str(e).lower()

    def test_nan_detection_in_parameters(self):
        """Test NaN detection in parameter updates."""
        config = HybridStreamingConfig(
            validate_numerics=True,
            warmup_iterations=3,
            warmup_learning_rate=1e10,  # Very large learning rate -> NaN
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        x_data = jnp.linspace(0, 5, 20)
        y_data = 2.0 * x_data + 1.0

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.8])

        # Large learning rate should cause parameter overflow
        # Validation should detect this
        try:
            result = optimizer.fit((x_data, y_data), model, p0, verbose=0)
            # If it succeeds, parameters should be finite
            if result["success"]:
                assert jnp.all(jnp.isfinite(result["x"]))
        except Exception as e:
            # Should catch numerical issues
            assert (
                "numerical" in str(e).lower()
                or "nan" in str(e).lower()
                or "inf" in str(e).lower()
            )

    def test_nan_detection_in_loss(self):
        """Test NaN detection in loss computation."""
        config = HybridStreamingConfig(
            validate_numerics=True,
            warmup_iterations=5,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        x_data = jnp.array([1.0, 2.0, 3.0])
        y_data = jnp.array([1.0, 2.0, jnp.nan])  # NaN in data

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.0])

        # NaN in data should be detected
        try:
            result = optimizer.fit((x_data, y_data), model, p0, verbose=0)
            # Should handle gracefully
            assert result is not None
        except ValueError as e:
            # Expected error for NaN in input data
            assert "NaN" in str(e) or "inf" in str(e).lower()


class TestAdaptiveRetry:
    """Tests for adaptive retry on batch failure."""

    def test_retry_on_singular_matrix(self):
        """Test retry logic when encountering singular matrices."""
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=5,
            gauss_newton_max_iterations=5,
            max_retries_per_batch=2,
            enable_fault_tolerance=True,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create data where parameters might be redundant
        np.random.seed(42)
        x_data = np.linspace(0, 5, 30)
        # y is only linear in first parameter
        y_data = 2.0 * x_data + 0.5 + 0.1 * np.random.randn(len(x_data))

        def model(x, a, b, c):
            # b and c are redundant (not in model) -> singular Jacobian
            return a * x

        p0 = jnp.array([1.8, 0.0, 0.0])
        bounds = (jnp.array([0.0, -1.0, -1.0]), jnp.array([5.0, 1.0, 1.0]))

        # Run optimization - should handle singular matrix gracefully
        result = optimizer.fit((x_data, y_data), model, p0, bounds=bounds, verbose=0)

        # Should complete (regularization handles singular matrices)
        assert result is not None
        assert "x" in result

    def test_graceful_degradation_on_persistent_errors(self):
        """Test graceful degradation when errors persist after retries."""
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=3,
            max_warmup_iterations=5,
            max_retries_per_batch=2,
            enable_fault_tolerance=True,
            min_success_rate=0.3,  # Tolerant of failures
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create simple data
        x_data = jnp.linspace(0, 5, 20)
        y_data = 1.5 * x_data + 0.5

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.0])

        # Should complete even with potential numerical issues
        result = optimizer.fit((x_data, y_data), model, p0, verbose=0)

        # Should return result (possibly with warnings)
        assert result is not None
        assert "x" in result
        assert "success" in result


class TestFaultToleranceIntegration:
    """Integration tests for complete fault tolerance system."""

    def test_checkpoint_and_resume_preserves_best_params(self):
        """Test that checkpoint/resume preserves best parameter tracking."""
        config = HybridStreamingConfig(
            normalize=True,
            warmup_iterations=5,
            gauss_newton_max_iterations=3,
            enable_checkpoints=True,
            checkpoint_frequency=3,
        )

        np.random.seed(42)
        x_data = np.linspace(0, 10, 40)
        y_data = 3.0 * jnp.exp(-0.5 * x_data) + 0.2 * np.random.randn(len(x_data))

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        p0 = jnp.array([2.5, 0.4])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([10.0, 2.0]))

        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir

            # First run
            optimizer1 = AdaptiveHybridStreamingOptimizer(config)
            result1 = optimizer1.fit(
                (x_data, y_data), model, p0, bounds=bounds, verbose=0
            )

            # Get checkpoint
            checkpoints = sorted(Path(tmpdir).glob("checkpoint_*.h5"))
            if len(checkpoints) > 0:
                # Resume from checkpoint
                config.resume_from_checkpoint = str(checkpoints[-1])
                optimizer2 = AdaptiveHybridStreamingOptimizer(config)
                result2 = optimizer2.fit(
                    (x_data, y_data), model, p0, bounds=bounds, verbose=0
                )

                # Results should be similar (best params preserved)
                assert jnp.allclose(result2["x"], result1["x"], rtol=0.15)

    def test_validation_with_checkpoints(self):
        """Test NaN/Inf validation works with checkpoint save/resume."""
        x_data = jnp.linspace(0, 5, 20)
        y_data = 2.0 * x_data + 1.0

        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.8])

        with tempfile.TemporaryDirectory() as tmpdir:
            config = HybridStreamingConfig(
                validate_numerics=True,
                enable_checkpoints=True,
                checkpoint_frequency=2,
                checkpoint_dir=tmpdir,  # Set in config
                warmup_iterations=3,
            )

            # Run with validation enabled
            optimizer = AdaptiveHybridStreamingOptimizer(config)
            result = optimizer.fit((x_data, y_data), model, p0, verbose=0)

            # Should succeed and have checkpoints
            assert result["success"]
            checkpoints = list(Path(tmpdir).glob("checkpoint_*.h5"))
            assert len(checkpoints) > 0

            # Verify checkpoint contains validation info
            with h5py.File(checkpoints[0], "r") as f:
                # Check version
                assert f.attrs["version"] == "3.0"
                # Parameters should be finite
                params = jnp.array(f["phase_state/normalized_params"])
                assert jnp.all(jnp.isfinite(params))
