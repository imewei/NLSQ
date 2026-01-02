"""Tests for TRF dataclasses (TRFConfig, StepContext, BoundsContext, FallbackContext).

These dataclasses encapsulate TRF algorithm parameters and state.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nlsq.core.trf import (
    BoundsContext,
    FallbackContext,
    StepContext,
    TRFConfig,
)


class TestTRFConfig:
    """Tests for TRFConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TRFConfig()
        assert config.ftol == 1e-8
        assert config.xtol == 1e-8
        assert config.gtol == 1e-8
        assert config.max_nfev is None
        assert config.x_scale == "jac"
        assert config.loss == "linear"
        assert config.tr_solver == "exact"
        assert config.verbose == 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TRFConfig(
            ftol=1e-6,
            xtol=1e-7,
            gtol=1e-5,
            max_nfev=1000,
            x_scale="auto",
            loss="soft_l1",
            tr_solver="lsmr",
            verbose=2,
        )
        assert config.ftol == 1e-6
        assert config.xtol == 1e-7
        assert config.gtol == 1e-5
        assert config.max_nfev == 1000
        assert config.x_scale == "auto"
        assert config.loss == "soft_l1"
        assert config.tr_solver == "lsmr"
        assert config.verbose == 2

    def test_immutable(self):
        """Test that TRFConfig is immutable (frozen)."""
        config = TRFConfig()
        with pytest.raises(AttributeError):
            config.ftol = 1e-6

    def test_validation_negative_ftol(self):
        """Test validation rejects negative ftol."""
        with pytest.raises(ValueError, match="ftol must be positive"):
            TRFConfig(ftol=-1e-8)

    def test_validation_negative_xtol(self):
        """Test validation rejects negative xtol."""
        with pytest.raises(ValueError, match="xtol must be positive"):
            TRFConfig(xtol=-1e-8)

    def test_validation_negative_gtol(self):
        """Test validation rejects negative gtol."""
        with pytest.raises(ValueError, match="gtol must be positive"):
            TRFConfig(gtol=-1e-8)

    def test_validation_negative_max_nfev(self):
        """Test validation rejects negative max_nfev."""
        with pytest.raises(ValueError, match="max_nfev must be positive"):
            TRFConfig(max_nfev=-1)

    def test_validation_zero_max_nfev(self):
        """Test validation rejects zero max_nfev."""
        with pytest.raises(ValueError, match="max_nfev must be positive"):
            TRFConfig(max_nfev=0)

    def test_validation_invalid_loss(self):
        """Test validation rejects invalid loss function."""
        with pytest.raises(ValueError, match="loss must be one of"):
            TRFConfig(loss="invalid_loss")

    def test_validation_invalid_tr_solver(self):
        """Test validation rejects invalid tr_solver."""
        with pytest.raises(ValueError, match="tr_solver must be one of"):
            TRFConfig(tr_solver="invalid_solver")

    def test_valid_loss_functions(self):
        """Test all valid loss functions are accepted."""
        valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
        for loss in valid_losses:
            config = TRFConfig(loss=loss)
            assert config.loss == loss

    def test_valid_tr_solvers(self):
        """Test all valid tr_solvers are accepted."""
        valid_solvers = ["exact", "lsmr"]
        for solver in valid_solvers:
            config = TRFConfig(tr_solver=solver)
            assert config.tr_solver == solver


class TestStepContext:
    """Tests for StepContext dataclass."""

    @pytest.fixture
    def sample_step_context(self):
        """Create a sample StepContext for testing."""
        n_params = 3
        n_residuals = 10
        return StepContext(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.zeros(n_residuals),
            J=jnp.ones((n_residuals, n_params)),
            cost=0.5,
            g=jnp.array([0.1, 0.2, 0.3]),
            trust_radius=1.0,
            iteration=0,
            scale=jnp.ones(n_params),
            scale_inv=jnp.ones(n_params),
        )

    def test_creation(self, sample_step_context):
        """Test StepContext can be created."""
        ctx = sample_step_context
        assert ctx.x.shape == (3,)
        assert ctx.f.shape == (10,)
        assert ctx.J.shape == (10, 3)
        assert ctx.cost == 0.5
        assert ctx.trust_radius == 1.0
        assert ctx.iteration == 0
        assert ctx.alpha == 0.0  # default

    def test_mutable(self, sample_step_context):
        """Test that StepContext is mutable."""
        ctx = sample_step_context
        ctx.iteration = 5
        assert ctx.iteration == 5
        ctx.trust_radius = 2.0
        assert ctx.trust_radius == 2.0

    def test_default_alpha(self):
        """Test default alpha value."""
        ctx = StepContext(
            x=jnp.array([1.0]),
            f=jnp.array([0.0]),
            J=jnp.array([[1.0]]),
            cost=0.0,
            g=jnp.array([0.0]),
            trust_radius=1.0,
            iteration=0,
            scale=jnp.array([1.0]),
            scale_inv=jnp.array([1.0]),
        )
        assert ctx.alpha == 0.0

    def test_custom_alpha(self):
        """Test custom alpha value."""
        ctx = StepContext(
            x=jnp.array([1.0]),
            f=jnp.array([0.0]),
            J=jnp.array([[1.0]]),
            cost=0.0,
            g=jnp.array([0.0]),
            trust_radius=1.0,
            iteration=0,
            scale=jnp.array([1.0]),
            scale_inv=jnp.array([1.0]),
            alpha=0.5,
        )
        assert ctx.alpha == 0.5


class TestBoundsContext:
    """Tests for BoundsContext dataclass."""

    def test_creation(self):
        """Test BoundsContext can be created directly."""
        ctx = BoundsContext(
            lb=jnp.array([0.0, 0.0]),
            ub=jnp.array([1.0, 1.0]),
            x_scale=jnp.array([1.0, 1.0]),
            x_offset=jnp.array([0.0, 0.0]),
            lb_scaled=jnp.array([0.0, 0.0]),
            ub_scaled=jnp.array([1.0, 1.0]),
        )
        assert ctx.lb.shape == (2,)
        assert ctx.ub.shape == (2,)

    def test_immutable(self):
        """Test that BoundsContext is immutable (frozen)."""
        ctx = BoundsContext(
            lb=jnp.array([0.0]),
            ub=jnp.array([1.0]),
            x_scale=jnp.array([1.0]),
            x_offset=jnp.array([0.0]),
            lb_scaled=jnp.array([0.0]),
            ub_scaled=jnp.array([1.0]),
        )
        with pytest.raises(AttributeError):
            ctx.lb = jnp.array([0.5])

    def test_from_bounds_factory(self):
        """Test from_bounds factory method."""
        lb = jnp.array([0.0, -1.0, 0.5])
        ub = jnp.array([10.0, 1.0, 2.0])
        ctx = BoundsContext.from_bounds(lb, ub)

        assert ctx.lb.shape == (3,)
        assert ctx.ub.shape == (3,)
        assert jnp.allclose(ctx.lb, lb)
        assert jnp.allclose(ctx.ub, ub)

    def test_from_bounds_with_scale(self):
        """Test from_bounds with custom x_scale."""
        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([1.0, 1.0])
        x_scale = jnp.array([2.0, 0.5])
        ctx = BoundsContext.from_bounds(lb, ub, x_scale=x_scale)

        assert jnp.allclose(ctx.x_scale, x_scale)

    def test_from_bounds_infinite_bounds(self):
        """Test from_bounds with infinite bounds."""
        lb = jnp.array([-jnp.inf, 0.0])
        ub = jnp.array([jnp.inf, 1.0])
        ctx = BoundsContext.from_bounds(lb, ub)

        assert jnp.isinf(ctx.lb[0])
        assert jnp.isinf(ctx.ub[0])
        assert ctx.lb[1] == 0.0
        assert ctx.ub[1] == 1.0


class TestFallbackContext:
    """Tests for FallbackContext dataclass."""

    def test_default_values(self):
        """Test default FallbackContext values."""
        ctx = FallbackContext(original_dtype=jnp.float32)
        assert ctx.original_dtype == jnp.float32
        assert ctx.fallback_triggered is False
        assert ctx.fallback_reason == ""
        assert ctx.step_context is None

    def test_mutable(self):
        """Test that FallbackContext is mutable."""
        ctx = FallbackContext(original_dtype=jnp.float32)
        ctx.fallback_triggered = True
        ctx.fallback_reason = "Numerical instability"
        assert ctx.fallback_triggered is True
        assert ctx.fallback_reason == "Numerical instability"

    def test_with_step_context(self):
        """Test FallbackContext with step_context."""
        step_ctx = StepContext(
            x=jnp.array([1.0]),
            f=jnp.array([0.0]),
            J=jnp.array([[1.0]]),
            cost=0.0,
            g=jnp.array([0.0]),
            trust_radius=1.0,
            iteration=0,
            scale=jnp.array([1.0]),
            scale_inv=jnp.array([1.0]),
        )
        ctx = FallbackContext(
            original_dtype=jnp.float32,
            fallback_triggered=True,
            fallback_reason="SVD failed",
            step_context=step_ctx,
        )
        assert ctx.step_context is not None
        assert ctx.step_context.iteration == 0

    def test_float64_dtype(self):
        """Test FallbackContext with float64 dtype."""
        ctx = FallbackContext(original_dtype=jnp.float64)
        assert ctx.original_dtype == jnp.float64
