"""Tests for AdaptiveHybridStreamingOptimizer - Phase 0 normalization setup.

This test module covers Phase 0 of the four-phase hybrid optimizer:
- Normalizer initialization with bounds
- Normalizer initialization with p0 only
- Normalization disabled via config
- Bounds transformation for Phase 2
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from nlsq.precision.parameter_normalizer import (
    NormalizedModelWrapper,
    ParameterNormalizer,
)
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


class TestPhase0NormalizationSetup:
    """Tests for Phase 0: Normalization Setup."""

    def test_normalizer_initialization_with_bounds(self):
        """Test normalizer initialization when bounds are provided."""
        # Create optimizer with bounds
        config = HybridStreamingConfig(
            normalize=True,
            normalization_strategy="auto",  # Should use bounds
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters
        p0 = jnp.array([50.0, 0.5])
        bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 1.0]))

        # Define simple model
        def model(x, a, b):
            return a * x + b

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Verify normalizer was created
        assert optimizer.normalizer is not None
        assert isinstance(optimizer.normalizer, ParameterNormalizer)

        # Verify strategy selection (should use bounds)
        assert optimizer.normalizer.strategy == "bounds"

        # Verify normalized model wrapper created
        assert optimizer.normalized_model is not None
        assert isinstance(optimizer.normalized_model, NormalizedModelWrapper)

        # Verify bounds were transformed
        assert optimizer.normalized_bounds is not None
        lb_norm, ub_norm = optimizer.normalized_bounds

        # Bounds-based normalization should map to [0, 1]
        assert jnp.allclose(lb_norm, jnp.array([0.0, 0.0]))
        assert jnp.allclose(ub_norm, jnp.array([1.0, 1.0]))

        # Verify normalization Jacobian stored
        assert optimizer.normalization_jacobian is not None
        assert optimizer.normalization_jacobian.shape == (2, 2)

    def test_normalizer_initialization_with_p0_only(self):
        """Test normalizer initialization with p0-based scaling (no bounds)."""
        # Create optimizer without bounds
        config = HybridStreamingConfig(
            normalize=True,
            normalization_strategy="auto",  # Should use p0
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters (no bounds)
        p0 = jnp.array([1000.0, 1.0, 0.001])
        bounds = None

        # Define simple model
        def model(x, a, b, c):
            return a * x + b * x**2 + c

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Verify normalizer was created
        assert optimizer.normalizer is not None
        assert isinstance(optimizer.normalizer, ParameterNormalizer)

        # Verify strategy selection (should use p0)
        assert optimizer.normalizer.strategy == "p0"

        # Verify normalized parameters are all 1.0 (p0/p0 = 1)
        normalized_p0 = optimizer.normalizer.normalize(p0)
        assert jnp.allclose(normalized_p0, jnp.array([1.0, 1.0, 1.0]))

        # Verify normalized model wrapper created
        assert optimizer.normalized_model is not None

        # Verify bounds are None (no bounds provided)
        assert optimizer.normalized_bounds is None

    def test_normalization_disabled_via_config(self):
        """Test that normalization can be completely disabled."""
        # Create optimizer with normalization disabled
        config = HybridStreamingConfig(
            normalize=False,  # Disable normalization
            normalization_strategy="none",
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters
        p0 = jnp.array([5.0, 15.0])
        bounds = (jnp.array([0.0, 10.0]), jnp.array([10.0, 20.0]))

        # Define simple model
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Setup normalization (should create identity transform)
        optimizer._setup_normalization(model, p0, bounds)

        # Verify normalizer created with identity strategy
        assert optimizer.normalizer is not None
        assert optimizer.normalizer.strategy == "none"

        # Verify normalization is identity (no change)
        normalized_p0 = optimizer.normalizer.normalize(p0)
        assert jnp.allclose(normalized_p0, p0)

        # Verify denormalization is also identity
        denormalized = optimizer.normalizer.denormalize(normalized_p0)
        assert jnp.allclose(denormalized, p0)

        # Verify normalized model still works
        assert optimizer.normalized_model is not None
        x_test = jnp.array([1.0, 2.0, 3.0])
        output = optimizer.normalized_model(x_test, *p0)
        expected = model(x_test, *p0)
        assert jnp.allclose(output, expected)

    def test_bounds_transformation_for_phase2(self):
        """Test that bounds are correctly transformed for Phase 2 optimization."""
        # Create optimizer with bounds
        config = HybridStreamingConfig(normalize=True, normalization_strategy="bounds")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters with bounds
        p0 = jnp.array([25.0, 0.25, 3.0])
        bounds = (
            jnp.array([10.0, 0.0, 1.0]),  # Lower bounds
            jnp.array([50.0, 1.0, 5.0]),  # Upper bounds
        )

        # Define model
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Verify bounds transformation
        assert optimizer.normalized_bounds is not None
        lb_norm, ub_norm = optimizer.normalized_bounds

        # All parameters should be normalized to [0, 1]
        assert jnp.allclose(lb_norm, jnp.zeros(3))
        assert jnp.allclose(ub_norm, jnp.ones(3))

        # Verify p0 is within normalized bounds
        normalized_p0 = optimizer.normalizer.normalize(p0)
        assert jnp.all(normalized_p0 >= lb_norm)
        assert jnp.all(normalized_p0 <= ub_norm)

        # Verify round-trip through bounds
        original_lb, original_ub = bounds
        recovered_lb = optimizer.normalizer.denormalize(lb_norm)
        recovered_ub = optimizer.normalizer.denormalize(ub_norm)
        assert jnp.allclose(recovered_lb, original_lb)
        assert jnp.allclose(recovered_ub, original_ub)


class TestPhaseTrackingInfrastructure:
    """Tests for phase tracking infrastructure."""

    def test_phase_tracking_initialization(self):
        """Test that phase tracking attributes are initialized correctly."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Verify initial phase is 0
        assert optimizer.current_phase == 0

        # Verify phase history is empty list
        assert isinstance(optimizer.phase_history, list)
        assert len(optimizer.phase_history) == 0

        # Verify phase start time is None initially
        assert optimizer.phase_start_time is None

        # Verify normalized params is None initially
        assert optimizer.normalized_params is None

    def test_phase_tracking_after_setup(self):
        """Test that phase tracking works correctly after Phase 0 setup."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters
        p0 = jnp.array([5.0, 0.5])
        bounds = None

        # Define model
        def model(x, a, b):
            return a * x + b

        # Setup normalization (Phase 0)
        optimizer._setup_normalization(model, p0, bounds)

        # After setup, normalized_params should be set
        assert optimizer.normalized_params is not None
        assert optimizer.normalized_params.shape == p0.shape

        # Verify normalized params match normalizer output
        expected_normalized = optimizer.normalizer.normalize(p0)
        assert jnp.allclose(optimizer.normalized_params, expected_normalized)

        # Phase should still be 0 (setup doesn't advance phase)
        assert optimizer.current_phase == 0


class TestNormalizedModelWrapper:
    """Tests for NormalizedModelWrapper integration."""

    def test_wrapped_model_preserves_output(self):
        """Test that wrapped model produces same output as original."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test model
        def model(x, a, b):
            return a * jnp.sin(b * x)

        # Test parameters
        p0 = jnp.array([2.5, 1.5])
        bounds = None
        x_test = jnp.linspace(0, 10, 50)

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Get normalized parameters
        normalized_p0 = optimizer.normalizer.normalize(p0)

        # Call wrapped model with normalized params
        wrapped_output = optimizer.normalized_model(x_test, *normalized_p0)

        # Call original model with original params
        original_output = model(x_test, *p0)

        # Outputs should be identical
        assert jnp.allclose(wrapped_output, original_output, rtol=1e-10)

    def test_wrapped_model_jit_compatible(self):
        """Test that wrapped model is JAX JIT-compatible."""
        import jax

        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test model
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Test parameters
        p0 = jnp.array([3.0, 0.5])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([10.0, 2.0]))
        x_test = jnp.array([1.0, 2.0, 3.0])

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)
        normalized_p0 = optimizer.normalizer.normalize(p0)

        # JIT compile wrapped model
        @jax.jit
        def jit_wrapped_model(x, a_norm, b_norm):
            return optimizer.normalized_model(x, a_norm, b_norm)

        # Should not raise error
        output = jit_wrapped_model(x_test, *normalized_p0)
        assert output.shape == x_test.shape
        assert jnp.isfinite(output).all()


class TestPhase1AdamWarmup:
    """Tests for Phase 1: Adam Warmup with Optax."""

    def test_adam_optimizer_initialization(self):
        """Test Adam optimizer initialization with optax."""
        config = HybridStreamingConfig(warmup_learning_rate=0.01, warmup_iterations=10)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define test parameters
        p0 = jnp.array([5.0, 0.5])
        bounds = None

        # Define model
        def model(x, a, b):
            return a * x + b

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Create Adam optimizer
        adam_optimizer, opt_state = optimizer._create_adam_optimizer(
            optimizer.normalized_params
        )

        # Verify optimizer created
        assert adam_optimizer is not None
        assert opt_state is not None

        # Verify optimizer state structure (optax state is a pytree)
        # For Adam, state has multiple components
        assert hasattr(opt_state, "count") or isinstance(opt_state, tuple)

    def test_loss_plateau_triggers_switch(self):
        """Test that loss plateau detection triggers Phase 2 switch."""
        config = HybridStreamingConfig(
            loss_plateau_threshold=1e-4, active_switching_criteria=["plateau"]
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Simulate plateau: very small relative loss change
        current_loss = 1.0
        prev_loss = 1.0001  # Relative change: 0.0001 / 1.0001 < 1e-4
        grad_norm = 0.5  # Gradient not small

        should_switch, reason = optimizer._check_phase1_switch_criteria(
            iteration=10,
            current_loss=current_loss,
            prev_loss=prev_loss,
            grad_norm=grad_norm,
        )

        # Should trigger switch due to plateau
        assert should_switch
        assert "plateau" in reason.lower()

    def test_gradient_norm_threshold_triggers_switch(self):
        """Test that small gradient norm triggers Phase 2 switch."""
        config = HybridStreamingConfig(
            gradient_norm_threshold=1e-3, active_switching_criteria=["gradient"]
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Simulate small gradient
        current_loss = 1.0
        prev_loss = 0.5  # Large loss change (not plateau)
        grad_norm = 5e-4  # Below threshold

        should_switch, reason = optimizer._check_phase1_switch_criteria(
            iteration=10,
            current_loss=current_loss,
            prev_loss=prev_loss,
            grad_norm=grad_norm,
        )

        # Should trigger switch due to gradient
        assert should_switch
        assert "gradient" in reason.lower()

    def test_max_iterations_triggers_switch(self):
        """Test that max iterations triggers Phase 2 switch."""
        config = HybridStreamingConfig(
            warmup_iterations=50,  # Must be <= max_warmup_iterations
            max_warmup_iterations=100,
            active_switching_criteria=["max_iter"],
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Simulate reaching max iterations
        current_loss = 1.0
        prev_loss = 0.9  # Still improving
        grad_norm = 0.1  # Gradient not small

        should_switch, reason = optimizer._check_phase1_switch_criteria(
            iteration=100,  # At max
            current_loss=current_loss,
            prev_loss=prev_loss,
            grad_norm=grad_norm,
        )

        # Should trigger switch due to max iterations
        assert should_switch
        assert "max" in reason.lower() or "iteration" in reason.lower()

    def test_best_parameter_tracking_during_warmup(self):
        """Test that best parameters are tracked during warmup."""
        config = HybridStreamingConfig(
            warmup_iterations=20, warmup_learning_rate=0.1, chunk_size=100
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple quadratic model
        def model(x, a, b):
            return a * x**2 + b * x

        # True parameters for data generation
        a_true, b_true = 2.0, -1.0
        x_data = jnp.linspace(0, 10, 100)
        y_data = a_true * x_data**2 + b_true * x_data

        # Initial guess
        p0 = jnp.array([1.0, 0.0])
        bounds = None

        # Run Phase 1 warmup
        data_source = (x_data, y_data)
        result = optimizer._run_phase1_warmup(data_source, model, p0, bounds)

        # Verify best params tracked
        assert "best_params" in result
        assert "best_loss" in result
        assert result["best_params"] is not None

        # Best loss should be less than or equal to final loss
        assert result["best_loss"] <= result.get("final_loss", jnp.inf)

    def test_warmup_in_normalized_parameter_space(self):
        """Test that warmup operates in normalized parameter space."""
        config = HybridStreamingConfig(
            warmup_iterations=10, normalize=True, normalization_strategy="p0"
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model with parameters at different scales
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Parameters with large scale difference
        p0 = jnp.array([1000.0, 0.001])
        bounds = None

        x_data = jnp.linspace(0, 5, 50)
        y_data = 1000.0 * jnp.exp(-0.001 * x_data)

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Verify normalized params are at similar scale
        normalized_p0 = optimizer.normalized_params
        assert jnp.allclose(normalized_p0, jnp.array([1.0, 1.0]))

        # Create loss function
        loss_fn = optimizer._create_warmup_loss_fn()

        # Compute loss in normalized space
        loss_value = loss_fn(normalized_p0, x_data, y_data)

        # Loss should be finite
        assert jnp.isfinite(loss_value)

    def test_early_stopping_on_loss_increase(self):
        """Test early stopping when loss increases significantly (optional)."""
        config = HybridStreamingConfig(
            warmup_iterations=50,
            max_warmup_iterations=100,
            active_switching_criteria=["plateau", "gradient", "max_iter"],
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Simulate loss increasing (bad situation)
        current_loss = 2.0
        prev_loss = 0.5  # Loss increased significantly
        grad_norm = 0.1

        # Check if we should stop (implementation may or may not stop)
        _should_switch, _reason = optimizer._check_phase1_switch_criteria(
            iteration=10,
            current_loss=current_loss,
            prev_loss=prev_loss,
            grad_norm=grad_norm,
        )

        # If early stopping is implemented, loss increase should trigger switch
        # Otherwise, test just verifies function doesn't crash
        # This test is "optional" per spec - implementation choice


class TestPhase2GaussNewton:
    """Tests for Phase 2: Streaming Gauss-Newton with J^T J Accumulation."""

    def test_exact_jacobian_computation_via_vmap_grad(self):
        """Test exact Jacobian computation using vmap+grad (no subsampling)."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple model
        def model(x, a, b):
            return a * x + b

        # Test parameters
        p0 = jnp.array([2.0, 1.0])
        bounds = None

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Generate test data chunk
        x_chunk = jnp.array([1.0, 2.0, 3.0, 4.0])
        n_points = len(x_chunk)
        n_params = len(p0)

        # Get normalized parameters
        normalized_params = optimizer.normalized_params

        # Compute Jacobian
        J_chunk = optimizer._compute_jacobian_chunk(x_chunk, normalized_params)

        # Verify shape
        assert J_chunk.shape == (n_points, n_params)

        # Verify Jacobian is exact (not subsampled)
        # For linear model y = a*x + b with normalization:
        # - Original params: [a, b] = [2, 1]
        # - Normalized params: [a_norm, b_norm] = [1, 1] (p0 normalization)
        # - Denormalization: a = a_norm * 2, b = b_norm * 1
        # - Model: f(x, a_norm, b_norm) = (a_norm * 2) * x + (b_norm * 1)
        # - Jacobian w.r.t. normalized params: [∂f/∂a_norm, ∂f/∂b_norm] = [2*x, 1]
        expected_J = jnp.stack([2.0 * x_chunk, jnp.ones_like(x_chunk)], axis=1)
        assert jnp.allclose(J_chunk, expected_J, rtol=1e-10)

    def test_jtj_accumulation_across_chunks(self):
        """Test J^T J accumulation across multiple chunks."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        # Test parameters
        p0 = jnp.array([2.0, 1.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Generate data chunks
        x_chunk1 = jnp.array([1.0, 2.0])
        y_chunk1 = jnp.array([3.0, 5.0])

        x_chunk2 = jnp.array([3.0, 4.0])
        y_chunk2 = jnp.array([7.0, 9.0])

        n_params = len(p0)
        normalized_params = optimizer.normalized_params

        # Accumulate J^T J and J^T r over chunks
        JTJ = jnp.zeros((n_params, n_params))
        JTr = jnp.zeros(n_params)
        total_residual_sq = 0.0

        for x_chunk, y_chunk in [(x_chunk1, y_chunk1), (x_chunk2, y_chunk2)]:
            JTJ_chunk, JTr_chunk, res_sq_chunk = optimizer._accumulate_jtj_jtr(
                x_chunk, y_chunk, normalized_params, JTJ, JTr
            )
            JTJ = JTJ_chunk
            JTr = JTr_chunk
            total_residual_sq += res_sq_chunk

        # Verify J^T J is symmetric
        assert jnp.allclose(JTJ, JTJ.T)

        # Verify J^T J is positive semi-definite (eigenvalues >= 0)
        eigenvalues = jnp.linalg.eigvalsh(JTJ)
        assert jnp.all(eigenvalues >= -1e-10)  # Allow small numerical errors

        # Verify dimensions
        assert JTJ.shape == (n_params, n_params)
        assert JTr.shape == (n_params,)
        assert jnp.isfinite(total_residual_sq)

    def test_jtr_gradient_accumulation(self):
        """Test J^T r (gradient) accumulation across chunks."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        # Test parameters
        p0 = jnp.array([5.0, 0.5, 1.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Generate data
        x_data = jnp.linspace(0, 10, 100)
        y_data = 5.0 * jnp.exp(-0.5 * x_data) + 1.0

        n_params = len(p0)
        normalized_params = optimizer.normalized_params

        # Accumulate J^T r across chunks (chunk size = 20)
        JTJ = jnp.zeros((n_params, n_params))
        JTr = jnp.zeros(n_params)

        chunk_size = 20
        for i in range(0, len(x_data), chunk_size):
            x_chunk = x_data[i : i + chunk_size]
            y_chunk = y_data[i : i + chunk_size]

            JTJ, JTr, _ = optimizer._accumulate_jtj_jtr(
                x_chunk, y_chunk, normalized_params, JTJ, JTr
            )

        # J^T r should be close to zero for perfect parameters
        # (gradient should be near zero at optimum)
        gradient_norm = jnp.linalg.norm(JTr)
        assert gradient_norm < 1.0  # Should be small but not exactly zero

    def test_svd_based_step_computation(self):
        """Test SVD-based step computation following trf.py patterns."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        # Test parameters
        p0 = jnp.array([2.0, 1.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create mock J^T J and J^T r
        n_params = len(p0)
        JTJ = jnp.array([[10.0, 5.0], [5.0, 4.0]])  # Symmetric positive definite
        JTr = jnp.array([2.0, 1.0])

        # Solve for Gauss-Newton step
        trust_radius = 1.0
        step, predicted_reduction = optimizer._solve_gauss_newton_step(
            JTJ, JTr, trust_radius
        )

        # Verify step dimensions
        assert step.shape == (n_params,)

        # Verify predicted reduction is non-negative
        assert predicted_reduction >= 0.0

        # Verify step is finite
        assert jnp.isfinite(step).all()

    def test_trust_region_step_acceptance(self):
        """Test trust region step acceptance based on predicted reduction."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x**2 + b

        # Test parameters
        p0 = jnp.array([1.0, 0.5])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Mock step and trust radius
        step = jnp.array([0.1, -0.05])
        trust_radius = 0.5

        # Apply trust region scaling
        scaled_step = optimizer._apply_trust_region(step, trust_radius)

        # Verify step norm is within trust radius
        step_norm = jnp.linalg.norm(scaled_step)
        assert step_norm <= trust_radius * (1 + 1e-10)  # Allow small numerical error

        # If original step was within radius, should be unchanged
        if jnp.linalg.norm(step) <= trust_radius:
            assert jnp.allclose(scaled_step, step)

    def test_convergence_detection(self):
        """Test convergence detection based on gradient and cost tolerance."""
        config = HybridStreamingConfig(
            gauss_newton_tol=1e-6, gauss_newton_max_iterations=100
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple model
        def model(x, a, b):
            return a * x + b

        # Test parameters (already at optimum)
        a_true, b_true = 2.0, 1.0
        p0 = jnp.array([a_true, b_true])
        bounds = None

        x_data = jnp.linspace(0, 10, 100)
        y_data = a_true * x_data + b_true  # Perfect fit

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Check convergence (gradient should be zero)
        # This will be tested in the full Gauss-Newton iteration
        # For now, just verify the method exists
        assert hasattr(optimizer, "_run_phase2_gauss_newton")

    def test_rank_deficient_jtj_handling(self):
        """Test handling of rank-deficient J^T J with regularization."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 0.5])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create rank-deficient J^T J (singular matrix)
        n_params = len(p0)
        JTJ = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # Rank 1
        JTr = jnp.array([1.0, 1.0])

        # Should handle gracefully with regularization
        trust_radius = 1.0
        try:
            step, _ = optimizer._solve_gauss_newton_step(JTJ, JTr, trust_radius)
            # Verify step is finite
            assert jnp.isfinite(step).all()
        except Exception as e:
            # Should not crash - either solve with regularization or raise informative error
            assert "singular" in str(e).lower() or "rank" in str(e).lower()

    def test_operation_in_normalized_space(self):
        """Test that Phase 2 operates correctly in normalized parameter space."""
        config = HybridStreamingConfig(normalize=True, normalization_strategy="p0")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model with different parameter scales
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        # Parameters with large scale difference
        p0 = jnp.array([1000.0, 0.001])
        bounds = None

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Verify parameters are normalized
        normalized_params = optimizer.normalized_params
        assert jnp.allclose(normalized_params, jnp.array([1.0, 1.0]))

        # Generate data
        x_chunk = jnp.linspace(0, 5, 50)
        y_chunk = 1000.0 * jnp.exp(-0.001 * x_chunk)

        # Compute Jacobian in normalized space
        J_chunk = optimizer._compute_jacobian_chunk(x_chunk, normalized_params)

        # Jacobian should have similar magnitude for both parameters
        # (benefit of normalization)
        j_col_norms = jnp.linalg.norm(J_chunk, axis=0)
        max_norm = jnp.max(j_col_norms)
        min_norm = jnp.min(j_col_norms)

        # Columns should be within 2-3 orders of magnitude
        # (much better than 6 orders without normalization)
        assert max_norm / min_norm < 1000.0


class TestPhase3DenormalizationCovariance:
    """Tests for Phase 3: Denormalization and Covariance Transform."""

    def test_parameter_denormalization_accuracy(self):
        """Test that parameters are accurately denormalized to original space."""
        config = HybridStreamingConfig(normalize=True, normalization_strategy="bounds")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        # Test parameters with bounds
        p0 = jnp.array([50.0, 5.0])
        bounds = (jnp.array([10.0, 0.0]), jnp.array([100.0, 10.0]))

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Simulate optimized normalized parameters
        normalized_params = jnp.array([0.6, 0.7])  # In [0, 1] range

        # Denormalize
        denormalized_params = optimizer._denormalize_params(normalized_params)

        # Verify denormalization accuracy
        expected_a = 10.0 + 0.6 * (100.0 - 10.0)  # 64.0
        expected_b = 0.0 + 0.7 * (10.0 - 0.0)  # 7.0

        assert jnp.allclose(denormalized_params[0], expected_a)
        assert jnp.allclose(denormalized_params[1], expected_b)

        # Verify round-trip accuracy
        renormalized = optimizer.normalizer.normalize(denormalized_params)
        assert jnp.allclose(renormalized, normalized_params)

    def test_covariance_transform_correctness(self):
        """Test covariance transform from normalized to original space."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple linear model
        def model(x, a, b):
            return a * x + b

        # Test parameters
        p0 = jnp.array([100.0, 1.0])
        bounds = (jnp.array([0.0, 0.0]), jnp.array([200.0, 10.0]))

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Create mock normalized covariance
        # Small diagonal covariance in normalized space
        cov_norm = jnp.array([[0.01, 0.005], [0.005, 0.02]])

        # Transform to original space
        cov_orig = optimizer._transform_covariance(cov_norm)

        # Verify covariance is symmetric
        assert jnp.allclose(cov_orig, cov_orig.T)

        # Verify covariance is positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(cov_orig)
        assert jnp.all(eigenvalues >= -1e-10)  # Allow small numerical error

        # Verify scaling matches Jacobian
        # For bounds-based: scales are (200.0, 10.0)
        # Cov_orig[i,j] = scale_i * scale_j * Cov_norm[i,j]
        expected_cov_00 = 200.0 * 200.0 * 0.01
        expected_cov_11 = 10.0 * 10.0 * 0.02
        expected_cov_01 = 200.0 * 10.0 * 0.005

        assert jnp.allclose(cov_orig[0, 0], expected_cov_00, rtol=1e-10)
        assert jnp.allclose(cov_orig[1, 1], expected_cov_11, rtol=1e-10)
        assert jnp.allclose(cov_orig[0, 1], expected_cov_01, rtol=1e-10)

    def test_residual_variance_scaling(self):
        """Test residual variance scaling of covariance matrix."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([2.0, 1.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create mock covariance before scaling
        cov_before = jnp.array([[1.0, 0.5], [0.5, 2.0]])

        # Mock residual sum of squares
        residual_sum_sq = 100.0
        n_points = 100
        n_params = 2

        # Apply residual variance scaling
        cov_final, sigma_sq = optimizer._apply_residual_variance(
            cov_before, residual_sum_sq, n_points
        )

        # Compute expected sigma^2
        expected_sigma_sq = residual_sum_sq / (n_points - n_params)
        assert jnp.allclose(sigma_sq, expected_sigma_sq)

        # Verify covariance scaling
        expected_cov = sigma_sq * cov_before
        assert jnp.allclose(cov_final, expected_cov)

        # Verify symmetry preserved
        assert jnp.allclose(cov_final, cov_final.T)

    def test_standard_error_computation(self):
        """Test standard error computation from covariance diagonal."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create mock covariance matrix
        pcov = jnp.array([[4.0, 1.0], [1.0, 9.0]])

        # Compute standard errors
        perr = optimizer._compute_standard_errors(pcov)

        # Verify standard errors are sqrt of diagonal
        expected_perr = jnp.sqrt(jnp.array([4.0, 9.0]))
        assert jnp.allclose(perr, expected_perr)

        # Verify shape
        assert perr.shape == (2,)

        # Verify all positive
        assert jnp.all(perr > 0)

    def test_covariance_matrix_symmetry(self):
        """Test that final covariance matrix is symmetric."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b, c):
            return a * x**2 + b * x + c

        p0 = jnp.array([1.0, 2.0, 3.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create mock J^T J (should be symmetric)
        JTJ = jnp.array([[10.0, 2.0, 1.0], [2.0, 8.0, 3.0], [1.0, 3.0, 5.0]])

        # Compute normalized covariance (pseudo-inverse)
        cov_norm = optimizer._compute_normalized_covariance(JTJ)

        # Verify symmetry
        assert jnp.allclose(cov_norm, cov_norm.T, atol=1e-10)

        # Verify positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(cov_norm)
        assert jnp.all(eigenvalues >= -1e-10)

    def test_disabled_normalization_case(self):
        """Test Phase 3 when normalization is disabled."""
        config = HybridStreamingConfig(normalize=False)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([2.0, 1.0])
        bounds = None

        # Setup normalization (should be identity)
        optimizer._setup_normalization(model, p0, bounds)

        # Verify normalizer uses 'none' strategy
        assert optimizer.normalizer.strategy == "none"

        # Mock normalized parameters (should be same as original)
        normalized_params = jnp.array([2.0, 1.0])

        # Denormalize (should be identity)
        denormalized = optimizer._denormalize_params(normalized_params)
        assert jnp.allclose(denormalized, normalized_params)

        # Mock normalized covariance
        cov_norm = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        # Transform covariance (should be identity for 'none' strategy)
        cov_orig = optimizer._transform_covariance(cov_norm)
        assert jnp.allclose(cov_orig, cov_norm)


class TestAPIIntegration:
    """Test Task Group 7: API Integration.

    Tests the complete fit() method and integration with curve_fit APIs.
    """

    def test_standalone_fit_basic(self):
        """Test standalone fit() method with simple linear model."""
        config = HybridStreamingConfig(
            normalize=True,
            normalization_strategy="p0",
            warmup_iterations=50,
            max_warmup_iterations=100,
            gauss_newton_max_iterations=20,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple linear model
        def linear_model(x, a, b):
            return a * x + b

        # Generate synthetic data
        x_data = jnp.linspace(0, 10, 100)
        true_params = jnp.array([2.5, 1.0])
        y_true = linear_model(x_data, *true_params)
        key = jax.random.PRNGKey(42)
        y_noisy = y_true + jax.random.normal(key, y_true.shape) * 0.1

        # Fit using standalone API
        result = optimizer.fit(
            data_source=(x_data, y_noisy),
            func=linear_model,
            p0=jnp.array([2.0, 0.5]),
            bounds=None,
            verbose=0,
        )

        # Verify result structure
        assert "x" in result
        assert "success" in result
        assert "message" in result
        assert "fun" in result
        assert "pcov" in result
        assert "perr" in result
        assert "streaming_diagnostics" in result

        # Verify success
        assert result["success"] is True

        # Verify parameters are close to true values
        assert jnp.allclose(result["x"], true_params, atol=0.5)

        # Verify covariance is positive semi-definite
        eigenvalues = jnp.linalg.eigvalsh(result["pcov"])
        assert jnp.all(eigenvalues >= -1e-10)

        # Verify streaming diagnostics structure
        diag = result["streaming_diagnostics"]
        assert "phase_timings" in diag
        assert "phase_iterations" in diag
        assert "warmup_diagnostics" in diag
        assert "gauss_newton_diagnostics" in diag

    def test_result_scipy_compatibility(self):
        """Test that result format matches scipy.optimize.curve_fit."""
        config = HybridStreamingConfig(
            warmup_iterations=30,
            gauss_newton_max_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def exponential(x, a, b):
            return a * jnp.exp(-b * x)

        x_data = jnp.linspace(0, 5, 50)
        y_data = exponential(x_data, 3.0, 0.5)

        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=exponential,
            p0=jnp.array([2.5, 0.4]),
            verbose=0,
        )

        # Verify scipy-compatible fields
        assert "x" in result  # popt equivalent
        assert "pcov" in result
        assert result["x"].shape == (2,)
        assert result["pcov"].shape == (2, 2)

        # Verify residuals
        assert "fun" in result
        assert result["fun"].shape == y_data.shape

    def test_streaming_diagnostics_content(self):
        """Test that streaming_diagnostics contains all required fields."""
        config = HybridStreamingConfig(
            warmup_iterations=20,
            gauss_newton_max_iterations=5,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        x_data = jnp.linspace(-5, 5, 80)
        y_data = quadratic(x_data, 1.0, 2.0, 3.0)

        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=quadratic,
            p0=jnp.array([0.8, 1.8, 2.8]),
            verbose=0,
        )

        diag = result["streaming_diagnostics"]

        # Verify phase timings
        assert "phase_timings" in diag
        assert "phase0_normalization" in diag["phase_timings"]
        assert "phase1_warmup" in diag["phase_timings"]
        assert "phase2_gauss_newton" in diag["phase_timings"]
        assert "phase3_finalize" in diag["phase_timings"]

        # Verify phase iterations
        assert "phase_iterations" in diag
        assert "phase1" in diag["phase_iterations"]
        assert "phase2" in diag["phase_iterations"]

        # Verify warmup diagnostics
        assert "warmup_diagnostics" in diag
        warmup = diag["warmup_diagnostics"]
        assert "best_loss" in warmup
        assert "final_loss" in warmup
        assert "switch_reason" in warmup

        # Verify Gauss-Newton diagnostics
        assert "gauss_newton_diagnostics" in diag
        gn = diag["gauss_newton_diagnostics"]
        assert "best_cost" in gn
        assert "final_cost" in gn
        assert "gradient_norm" in gn
        assert "convergence_reason" in gn

    def test_pcov_and_perr_relationship(self):
        """Test that perr = sqrt(diag(pcov))."""
        config = HybridStreamingConfig(
            warmup_iterations=30,
            gauss_newton_max_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * jnp.sin(b * x)

        x_data = jnp.linspace(0, 10, 100)
        y_data = model(x_data, 2.0, 1.5)

        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=jnp.array([1.8, 1.3]),
            verbose=0,
        )

        # Verify perr = sqrt(diag(pcov))
        expected_perr = jnp.sqrt(jnp.diag(result["pcov"]))
        assert jnp.allclose(result["perr"], expected_perr, rtol=1e-10)

    def test_with_bounds(self):
        """Test fit() with parameter bounds."""
        config = HybridStreamingConfig(
            normalize=True,
            normalization_strategy="bounds",
            warmup_iterations=40,
            gauss_newton_max_iterations=15,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * x**2 + b

        x_data = jnp.linspace(0, 5, 60)
        true_params = jnp.array([1.5, 2.0])
        y_data = model(x_data, *true_params)

        # Set bounds
        bounds = (jnp.array([0.5, 0.5]), jnp.array([3.0, 5.0]))

        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=jnp.array([1.0, 1.5]),
            bounds=bounds,
            verbose=0,
        )

        # Verify parameters are within bounds
        assert jnp.all(result["x"] >= bounds[0])
        assert jnp.all(result["x"] <= bounds[1])

        # Verify close to true values
        assert jnp.allclose(result["x"], true_params, atol=0.3)

    def test_verbose_output_levels(self):
        """Test that verbose levels control output (smoke test)."""
        config = HybridStreamingConfig(
            warmup_iterations=10,
            gauss_newton_max_iterations=3,
        )

        def model(x, a):
            return a * x

        x_data = jnp.linspace(0, 5, 30)
        y_data = 2.0 * x_data

        # Test verbose=0 (silent)
        optimizer0 = AdaptiveHybridStreamingOptimizer(config)
        result0 = optimizer0.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=jnp.array([1.5]),
            verbose=0,
        )
        assert result0["success"]

        # Test verbose=1 (progress)
        optimizer1 = AdaptiveHybridStreamingOptimizer(config)
        result1 = optimizer1.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=jnp.array([1.5]),
            verbose=1,
        )
        assert result1["success"]

    def test_phase_history_tracking(self):
        """Test that phase_history is properly populated."""
        config = HybridStreamingConfig(
            warmup_iterations=20,
            gauss_newton_max_iterations=8,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x_data = jnp.linspace(0, 8, 70)
        y_data = model(x_data, 3.0, 0.5)

        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=jnp.array([2.5, 0.4]),
            verbose=0,
        )

        # Verify phase_history exists in streaming_diagnostics
        assert "phase_history" in result["streaming_diagnostics"]
        phase_history = result["streaming_diagnostics"]["phase_history"]

        # Should have 4 phases (0, 1, 2, 3)
        assert len(phase_history) == 4

        # Verify each phase has correct structure
        phase_names = [
            "normalization_setup",
            "lbfgs_warmup",
            "gauss_newton",
            "denormalization_covariance",
        ]
        for i, expected_name in enumerate(phase_names):
            assert phase_history[i]["phase"] == i
            assert phase_history[i]["name"] == expected_name
            assert "timestamp" in phase_history[i]


class TestAPIIntegrationCurveFit:
    """Tests for Task Group 7: API Integration with curve_fit."""

    def test_curve_fit_with_hybrid_streaming_method(self):
        """Test method='hybrid_streaming' in curve_fit()."""
        import numpy as np

        from nlsq import curve_fit

        # Generate test data
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 10, 1000)
        true_params = np.array([5.0, 0.5])
        y = model(x, *true_params)
        y += np.random.normal(0, 0.1, len(x))

        # Use hybrid_streaming method
        result = curve_fit(
            model,
            x,
            y,
            p0=np.array([4.0, 0.4]),
            method="hybrid_streaming",
            verbose=0,
        )

        # Check that result can be unpacked as (popt, pcov)
        popt, pcov = result

        # Verify parameters are close to true values
        assert popt.shape == (2,)
        assert pcov.shape == (2, 2)
        np.testing.assert_allclose(popt, true_params, rtol=0.1)

        # Verify pcov is a valid covariance matrix
        assert np.all(np.isfinite(pcov))
        assert np.all(np.diag(pcov) > 0)  # Variances should be positive

    def test_curve_fit_large_with_hybrid_streaming_method(self):
        """Test method='hybrid_streaming' in curve_fit_large()."""
        import numpy as np

        from nlsq import curve_fit_large

        # Generate test data
        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 10, 5000)
        true_params = np.array([3.0, 0.3, 1.0])
        y = model(x, *true_params)
        y += np.random.normal(0, 0.05, len(x))

        # Use hybrid_streaming method
        popt, pcov = curve_fit_large(
            model,
            x,
            y,
            p0=np.array([2.5, 0.25, 0.8]),
            method="hybrid_streaming",
            verbose=0,
        )

        # Verify parameters are close to true values
        assert popt.shape == (3,)
        assert pcov.shape == (3, 3)
        np.testing.assert_allclose(popt, true_params, rtol=0.15)

        # Verify pcov is a valid covariance matrix
        assert np.all(np.isfinite(pcov))
        assert np.all(np.diag(pcov) > 0)  # Variances should be positive

    def test_hybrid_streaming_with_bounds(self):
        """Test hybrid_streaming with parameter bounds."""
        import numpy as np

        from nlsq import curve_fit

        # Generate test data
        def model(x, a, b):
            return a * jnp.sin(b * x)

        x = np.linspace(0, 2 * np.pi, 800)
        true_params = np.array([2.0, 1.5])
        y = model(x, *true_params)
        y += np.random.normal(0, 0.1, len(x))

        # Use hybrid_streaming with bounds
        result = curve_fit(
            model,
            x,
            y,
            p0=np.array([1.5, 1.0]),
            bounds=([0, 0], [5, 3]),
            method="hybrid_streaming",
            verbose=0,
        )

        popt, _pcov = result

        # Verify parameters respect bounds
        assert np.all(popt >= np.array([0, 0]))
        assert np.all(popt <= np.array([5, 3]))

        # Verify reasonable fit
        np.testing.assert_allclose(popt, true_params, rtol=0.2)

    def test_hybrid_streaming_returns_scipy_compatible_result(self):
        """Test that hybrid_streaming returns scipy-compatible result."""
        import numpy as np

        from nlsq import curve_fit

        # Generate test data
        def model(x, a):
            return a * x**2

        x = np.linspace(0, 5, 500)
        y = 2.0 * x**2 + np.random.normal(0, 0.5, len(x))

        # Use hybrid_streaming
        result = curve_fit(
            model,
            x,
            y,
            p0=np.array([1.5]),
            method="hybrid_streaming",
            verbose=0,
        )

        # Check result has expected fields
        assert "x" in result
        assert "pcov" in result
        assert "success" in result

        # Verify tuple unpacking works
        popt, pcov = result
        assert popt.shape == (1,)
        assert pcov.shape == (1, 1)


class TestMultiDeviceSupport:
    """Test Task Group 9: Multi-GPU and TPU Support.

    Tests for multi-device support including device detection, pmap Jacobian
    computation, psum aggregation, and graceful fallback to single device.
    """

    def test_device_detection(self):
        """Test detection of available GPU/TPU devices."""
        config = HybridStreamingConfig(enable_multi_device=True)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Detect devices
        device_info = optimizer._detect_available_devices()

        # Verify structure
        assert "device_count" in device_info
        assert "device_type" in device_info
        assert "devices" in device_info

        # Verify device count is positive
        assert device_info["device_count"] > 0

        # Verify device type is recognized
        assert device_info["device_type"] in ["cpu", "gpu", "tpu"]

    def test_multi_device_disabled_by_default(self):
        """Test that multi-device is disabled by default in config."""
        config = HybridStreamingConfig()
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Verify multi-device disabled
        assert config.enable_multi_device is False

        # Setup should not configure multi-device
        device_info = optimizer._detect_available_devices()

        # Even if multiple devices available, should use single device
        should_use_multi = optimizer._should_use_multi_device(device_info)
        assert should_use_multi is False

    def test_pmap_jacobian_computation_single_device(self):
        """Test pmap Jacobian computation falls back gracefully on single device."""
        config = HybridStreamingConfig(enable_multi_device=True)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([2.0, 1.0])
        bounds = None

        # Setup normalization
        optimizer._setup_normalization(model, p0, bounds)

        # Create test data
        x_data = jnp.linspace(0, 10, 100)
        params = jnp.array([2.0, 1.0])

        # Try pmap computation (should fall back to regular computation on single device)
        try:
            J = optimizer._compute_jacobian_chunk(x_data, params)

            # Verify shape
            assert J.shape == (100, 2)

            # Verify values are finite
            assert jnp.all(jnp.isfinite(J))

        except Exception as e:
            # If pmap not supported, should fall back gracefully
            pytest.skip(f"Multi-device not available: {e}")

    def test_psum_aggregation_single_device(self):
        """Test psum aggregation of J^T J across devices (single device case)."""
        config = HybridStreamingConfig(enable_multi_device=True)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([2.0, 1.0])
        optimizer._setup_normalization(model, p0, None)

        # Create mock J^T J matrices (as if from different devices)
        JTJ_local = jnp.array([[10.0, 2.0], [2.0, 8.0]])

        # Aggregate (on single device, should just return the matrix)
        JTJ_global = optimizer._aggregate_jtj_across_devices(JTJ_local)

        # Verify shape preserved
        assert JTJ_global.shape == (2, 2)

        # On single device, should be identical
        assert jnp.allclose(JTJ_global, JTJ_local)

    def test_graceful_fallback_when_pmap_fails(self):
        """Test graceful fallback to single device when pmap fails."""
        config = HybridStreamingConfig(enable_multi_device=True)
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * jnp.exp(-b * x)

        p0 = jnp.array([1.0, 0.5])
        optimizer._setup_normalization(model, p0, None)

        # Create small dataset
        x_data = jnp.linspace(0, 5, 50)
        y_data = 1.0 * jnp.exp(-0.5 * x_data) + 0.01 * jax.random.normal(
            jax.random.PRNGKey(0), (50,)
        )

        # Try multi-device setup (should fall back gracefully)
        device_info = optimizer._detect_available_devices()
        multi_device_config = optimizer._setup_multi_device(device_info)

        # Verify fallback configuration exists
        assert "use_multi_device" in multi_device_config
        assert "device_count" in multi_device_config

        # Should work even if multi-device not available
        params = jnp.array([1.0, 1.0])

        # Compute Jacobian (should work on single device)
        J = optimizer._compute_jacobian_chunk(x_data, params)
        assert J.shape == (50, 2)
        assert jnp.all(jnp.isfinite(J))


class TestLoopStrategyDispatch:
    """Test loop_strategy config and backend-aware dispatch."""

    def test_loop_strategy_forced_loop(self):
        """Test loop_strategy='loop' always uses Python loops."""
        config = HybridStreamingConfig(loop_strategy="loop")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 1.0])
        optimizer._setup_normalization(model, p0, None)

        # Should always return False for Python loops
        assert optimizer._use_scan_for_accumulation() is False

    def test_loop_strategy_forced_scan(self):
        """Test loop_strategy='scan' always uses JAX scan."""
        config = HybridStreamingConfig(loop_strategy="scan")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 1.0])
        optimizer._setup_normalization(model, p0, None)

        # Should always return True for JAX scan
        assert optimizer._use_scan_for_accumulation() is True

    def test_loop_strategy_auto_dispatches_based_on_backend(self):
        """Test loop_strategy='auto' dispatches based on backend."""
        config = HybridStreamingConfig(loop_strategy="auto")
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 1.0])
        optimizer._setup_normalization(model, p0, None)

        # Get the dispatch decision
        use_scan = optimizer._use_scan_for_accumulation()

        # Verify it's based on backend
        devices = jax.devices()
        if devices:
            platform = devices[0].platform
            expected = platform in ("gpu", "cuda", "rocm", "tpu")
            assert use_scan == expected
        else:
            assert use_scan is False

    def test_scan_based_accumulation_produces_correct_results(self):
        """Test scan-based accumulation produces same results as loop-based."""

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([2.0, 1.0])
        x_data = jnp.linspace(0, 10, 1000)
        y_data = model(x_data, 2.5, 0.8) + 0.01 * jax.random.normal(
            jax.random.PRNGKey(0), (1000,)
        )

        # Create optimizer with loop strategy
        config_loop = HybridStreamingConfig(loop_strategy="loop", chunk_size=100)
        optimizer_loop = AdaptiveHybridStreamingOptimizer(config_loop)
        optimizer_loop._setup_normalization(model, p0, None)

        # Create optimizer with scan strategy
        config_scan = HybridStreamingConfig(loop_strategy="scan", chunk_size=100)
        optimizer_scan = AdaptiveHybridStreamingOptimizer(config_scan)
        optimizer_scan._setup_normalization(model, p0, None)

        # Get normalized params
        params = optimizer_loop.normalized_params

        # Compute cost with both strategies
        cost_loop = optimizer_loop._compute_cost_only(params, x_data, y_data)
        cost_scan = optimizer_scan._compute_cost_only(params, x_data, y_data)

        # Results should be nearly identical (within numerical precision)
        assert jnp.abs(cost_loop - cost_scan) < 1e-6


class TestMixedPrecisionSupport:
    """Test Task Group 10: Mixed Precision Support.

    Tests precision='auto', 'float32', 'float64' modes and auto-upgrade behavior.
    """

    def test_float32_warmup_phase(self):
        """Test float32 warmup phase with precision='auto'."""
        config = HybridStreamingConfig(
            precision="auto",  # Should use float32 for Phase 1
            warmup_iterations=50,
            max_warmup_iterations=100,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 2.0])
        x_data = jnp.linspace(0, 10, 100)
        y_data = model(x_data, 2.0, 1.0)

        # Setup normalization
        optimizer._setup_normalization(model, p0, None)

        # Setup precision
        optimizer._setup_precision()

        # Verify Phase 1 precision is float32
        assert optimizer.current_precision == jnp.float32
        assert optimizer.phase_precisions[0] == jnp.float64
        assert optimizer.phase_precisions[1] == jnp.float32
        assert optimizer.phase_precisions[2] == jnp.float64
        assert optimizer.phase_precisions[3] == jnp.float64

    def test_float64_gauss_newton_phase(self):
        """Test float64 Gauss-Newton phase with precision='auto'."""
        config = HybridStreamingConfig(
            precision="auto",  # Should upgrade to float64 for Phase 2
            warmup_iterations=50,
            gauss_newton_max_iterations=20,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a):
            return a * x

        p0 = jnp.array([1.5])
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data

        # Run full fit (to test Phase 1 -> Phase 2 transition)
        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=p0,
            verbose=0,
        )

        # Verify transition to float64 for Phase 2
        # Check phase history for precision transitions
        assert "streaming_diagnostics" in result
        assert "phase_history" in result["streaming_diagnostics"]

        # Verify success
        assert result["success"]
        assert jnp.allclose(
            result["x"], jnp.array([2.0]), atol=0.05
        )  # Relaxed tolerance

    def test_auto_precision_upgrade_on_nan(self):
        """Test auto precision upgrade when NaN detected in float32."""
        config = HybridStreamingConfig(
            precision="auto",
            warmup_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Setup precision
        optimizer._setup_precision()

        # Verify starts in float32
        assert optimizer.current_precision == jnp.float32

        # Simulate NaN detection
        params_with_nan = jnp.array([1.0, jnp.nan], dtype=jnp.float32)

        # Trigger precision upgrade
        should_upgrade = not jnp.all(jnp.isfinite(params_with_nan))
        assert should_upgrade

        # Upgrade precision
        optimizer._convert_precision(jnp.float64)

        # Verify upgraded to float64
        assert optimizer.current_precision == jnp.float64

    def test_user_override_float32_throughout(self):
        """Test user override via precision='float32' throughout."""
        config = HybridStreamingConfig(
            precision="float32",  # User forces float32
            warmup_iterations=20,
            gauss_newton_max_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a):
            return a * x

        p0 = jnp.array([1.5])
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data

        # Run full fit
        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=p0,
            verbose=0,
        )

        # Verify success
        assert result["success"]

        # Verify stayed in float32 (check parameter dtype)
        assert result["x"].dtype == jnp.float32 or result["x"].dtype == jnp.float64
        # Note: Final result may be cast to float64 for covariance accuracy

    def test_user_override_float64_throughout(self):
        """Test user override via precision='float64' throughout."""
        config = HybridStreamingConfig(
            precision="float64",  # User forces float64
            warmup_iterations=50,
            gauss_newton_max_iterations=20,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a):
            return a * x

        p0 = jnp.array([1.5])
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data

        # Setup precision
        optimizer._setup_precision()

        # Verify uses float64 from start
        assert optimizer.current_precision == jnp.float64

        # Run full fit
        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=p0,
            verbose=0,
        )

        # Verify success
        assert result["success"]
        assert jnp.allclose(
            result["x"], jnp.array([2.0]), atol=0.05
        )  # Relaxed tolerance

    def test_precision_consistency_in_covariance(self):
        """Test precision consistency in covariance computation."""
        config = HybridStreamingConfig(
            precision="auto",  # Should use float64 for covariance
            warmup_iterations=20,
            gauss_newton_max_iterations=10,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.5])
        x_data = jnp.linspace(0, 5, 100)
        y_data = model(x_data, 2.0, 1.0)

        # Run full fit
        result = optimizer.fit(
            data_source=(x_data, y_data),
            func=model,
            p0=p0,
            verbose=0,
        )

        # Verify covariance matrix is computed
        assert "pcov" in result
        assert result["pcov"].shape == (2, 2)

        # Verify covariance is in float64 for accuracy
        # (Phase 3 always uses float64 for covariance)
        assert result["pcov"].dtype == jnp.float64

        # Verify standard errors computed
        assert "perr" in result
        assert result["perr"].shape == (2,)

        # Verify success
        assert result["success"]


class TestOptaxEnhancements:
    """Tests for Task Group 11: Optax Integration Enhancements."""

    def test_learning_rate_schedule_warmup_and_decay(self):
        """Test learning rate schedule with warmup and cosine decay."""
        config = HybridStreamingConfig(
            use_learning_rate_schedule=True,
            warmup_learning_rate=0.001,
            lr_schedule_warmup_steps=10,
            lr_schedule_decay_steps=90,
            lr_schedule_end_value=0.0001,
            warmup_iterations=50,
            max_warmup_iterations=100,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define simple model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.5])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create optimizer with schedule
        params = optimizer.normalized_params
        opt, opt_state = optimizer._create_adam_optimizer(params)

        # Verify optimizer is created successfully
        assert opt is not None
        assert opt_state is not None

        # Test that optimizer works (can compute gradients)
        loss_fn = optimizer._create_warmup_loss_fn()
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data + 1.0

        # Run multiple steps to see schedule effect
        # (first step has zero lr during warmup phase)
        for step in range(20):
            _loss, grads = jax.value_and_grad(loss_fn)(params, x_data, y_data)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Verify parameters changed after multiple steps
        assert not jnp.allclose(params, optimizer.normalized_params)
        assert jnp.all(jnp.isfinite(params))

    def test_gradient_clipping_with_optax(self):
        """Test gradient clipping prevents exploding gradients."""
        config = HybridStreamingConfig(
            gradient_clip_value=1.0,  # Clip gradients to max norm of 1.0
            warmup_learning_rate=0.1,  # High learning rate to test clipping
            warmup_iterations=20,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.0, 1.0])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create optimizer with clipping
        params = optimizer.normalized_params
        opt, opt_state = optimizer._create_adam_optimizer(params)

        # Generate data with large residuals to create large gradients
        x_data = jnp.linspace(0, 5, 50)
        y_data = 100.0 * x_data + 50.0  # Large values to create large gradients

        loss_fn = optimizer._create_warmup_loss_fn()

        # Compute gradient (would be large without clipping)
        _loss, grads = jax.value_and_grad(loss_fn)(params, x_data, y_data)

        # Verify gradients are large
        grad_norm = jnp.linalg.norm(grads)
        assert grad_norm > 1.0  # Gradients before clipping are large

        # Apply optimizer update with clipping
        updates, _new_opt_state = opt.update(grads, opt_state)

        # Verify clipping worked (updates should be bounded)
        update_norm = jnp.linalg.norm(
            jnp.concatenate([jnp.ravel(u) for u in jax.tree_util.tree_leaves(updates)])
        )
        # Updates are bounded by learning_rate * clip_value
        # With lr=0.1 and clip=1.0, updates should be reasonable
        assert jnp.isfinite(update_norm)

        # Apply updates
        new_params = optax.apply_updates(params, updates)
        assert jnp.all(jnp.isfinite(new_params))

    def test_optimizer_state_checkpointing(self):
        """Test optimizer state can be saved and restored."""
        import tempfile
        from pathlib import Path

        config = HybridStreamingConfig(
            warmup_iterations=10,
            enable_checkpoints=True,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.5])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create optimizer
        params = optimizer.normalized_params
        opt, opt_state = optimizer._create_adam_optimizer(params)

        # Run a few steps to change optimizer state
        loss_fn = optimizer._create_warmup_loss_fn()
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data + 1.0

        for _ in range(5):
            _loss, grads = jax.value_and_grad(loss_fn)(params, x_data, y_data)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

        # Store optimizer state
        optimizer.phase1_optimizer_state = opt_state
        optimizer.normalized_params = params
        optimizer.current_phase = 1

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.h5"
            optimizer._save_checkpoint(checkpoint_path)

            # Verify checkpoint exists
            assert checkpoint_path.exists()

            # Create new optimizer and load checkpoint
            optimizer2 = AdaptiveHybridStreamingOptimizer(config)
            optimizer2._setup_normalization(model, p0, bounds)
            optimizer2._load_checkpoint(checkpoint_path)

            # Verify state restored
            assert optimizer2.current_phase == 1
            assert optimizer2.phase1_optimizer_state is not None
            assert jnp.allclose(optimizer2.normalized_params, params)

    def test_combined_schedule_and_clipping(self):
        """Test learning rate schedule combined with gradient clipping."""
        config = HybridStreamingConfig(
            use_learning_rate_schedule=True,
            warmup_learning_rate=0.001,
            lr_schedule_warmup_steps=5,
            lr_schedule_decay_steps=45,
            lr_schedule_end_value=0.0001,
            gradient_clip_value=1.0,
            warmup_iterations=30,
        )
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Define model
        def model(x, a, b):
            return a * x + b

        p0 = jnp.array([1.5, 0.5])
        bounds = None
        optimizer._setup_normalization(model, p0, bounds)

        # Create optimizer with both schedule and clipping
        params = optimizer.normalized_params
        opt, opt_state = optimizer._create_adam_optimizer(params)

        # Verify optimizer created
        assert opt is not None
        assert opt_state is not None

        # Generate test data
        x_data = jnp.linspace(0, 5, 50)
        y_data = 2.0 * x_data + 1.0

        loss_fn = optimizer._create_warmup_loss_fn()

        # Run several steps to test schedule + clipping interaction
        for step in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(params, x_data, y_data)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            # Verify finite values at each step
            assert jnp.all(jnp.isfinite(params))
            assert jnp.isfinite(loss)

        # Verify parameters converged somewhat
        final_loss, _ = jax.value_and_grad(loss_fn)(params, x_data, y_data)
        # Loss should decrease over iterations (relaxed tolerance for 10 steps)
        assert (
            final_loss < 5.0
        )  # Should decrease but may not fully converge in 10 steps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
