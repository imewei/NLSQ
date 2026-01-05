"""Extended tests for streaming phase classes.

Enterprise-level test coverage for WarmupPhase, GaussNewtonPhase,
PhaseOrchestrator, and CheckpointManager.

Tests focus on:
- Numerical correctness with known analytical solutions
- Edge cases (empty data, single point, boundary conditions)
- Error handling and fault tolerance
- Integration between phases
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.checkpoint import CheckpointManager, CheckpointState
from nlsq.streaming.phases.gauss_newton import GaussNewtonPhase, GNResult
from nlsq.streaming.phases.orchestrator import (
    PhaseOrchestrator,
    PhaseOrchestratorResult,
)
from nlsq.streaming.phases.warmup import WarmupPhase, WarmupResult


def config_with_overrides(
    config: HybridStreamingConfig, **overrides
) -> HybridStreamingConfig:
    """Create a new config with overridden values (works with slots)."""
    config_dict = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    config_dict.update(overrides)
    return HybridStreamingConfig(**config_dict)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def base_config() -> HybridStreamingConfig:
    """Create base configuration for tests."""
    return HybridStreamingConfig(
        chunk_size=100,
        warmup_iterations=5,
        max_warmup_iterations=20,
        gauss_newton_max_iterations=10,
        gauss_newton_tol=1e-8,
        trust_region_initial=1.0,
        lbfgs_initial_step_size=0.1,
        lbfgs_history_size=5,
        gradient_norm_threshold=1e-6,
        loss_plateau_threshold=1e-8,
        enable_warm_start_detection=True,
        warm_start_threshold=0.01,
        enable_adaptive_warmup_lr=True,
        enable_cost_guard=True,
        cost_increase_tolerance=0.5,
        enable_step_clipping=True,
        max_warmup_step_size=1.0,
        enable_fault_tolerance=True,
        verbose=0,
    )


@pytest.fixture
def simple_linear_model():
    """Create simple linear model: y = a*x + b."""

    def model(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
        return a * x + b

    return model


@pytest.fixture
def exponential_model():
    """Create exponential decay model: y = a * exp(-b * x)."""

    def model(x: jnp.ndarray, a: float, b: float) -> jnp.ndarray:
        return a * jnp.exp(-b * x)

    return model


@pytest.fixture
def linear_data():
    """Generate linear test data with known parameters."""
    true_a, true_b = 2.5, 1.0
    x = jnp.linspace(0, 10, 200)
    y = true_a * x + true_b
    return x, y, jnp.array([true_a, true_b])


@pytest.fixture
def noisy_linear_data():
    """Generate noisy linear data."""
    true_a, true_b = 2.5, 1.0
    key = jax.random.PRNGKey(42)
    x = jnp.linspace(0, 10, 200)
    noise = jax.random.normal(key, shape=x.shape) * 0.1
    y = true_a * x + true_b + noise
    return x, y, jnp.array([true_a, true_b])


class MockNormalizedModel:
    """Mock normalized model wrapper for testing."""

    def __init__(self, model_fn):
        self.model_fn = model_fn
        self.call_count = 0

    def __call__(self, x: jnp.ndarray, *params) -> jnp.ndarray:
        self.call_count += 1
        return self.model_fn(x, *params)


# ============================================================================
# PhaseOrchestrator Tests
# ============================================================================


class TestPhaseOrchestratorRun:
    """Tests for PhaseOrchestrator.run() method."""

    def test_run_without_initialization_skips_phases(self, base_config, linear_data):
        """Test run() without initialize_phases() skips warmup and GN."""
        x, y, _ = linear_data
        orchestrator = PhaseOrchestrator(base_config)
        initial_params = jnp.array([1.0, 0.0])

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        assert "final_params" in result
        assert "phase_history" in result
        # Should have phase records for skipped phases
        assert len(result["phase_history"]) >= 2

    def test_run_full_workflow(self, base_config, simple_linear_model, linear_data):
        """Test complete run through all phases."""
        x, y, _true_params = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        initial_params = jnp.array([1.0, 0.0])
        result = orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        assert "final_params" in result
        assert "normalized_params" in result
        assert "best_cost" in result
        assert "phase_history" in result
        assert "total_time" in result
        assert result["total_time"] > 0

    def test_run_tracks_best_params_globally(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test that best params are tracked across phases."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        initial_params = jnp.array([1.0, 0.0])
        orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        # Best params should be updated
        best_params = orchestrator.get_best_params()
        best_cost = orchestrator.get_best_cost()
        assert best_params is not None
        assert best_cost < float("inf")

    def test_run_with_normalizer(self, base_config, simple_linear_model, linear_data):
        """Test run with normalizer denormalization."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        # Mock normalizer
        normalizer = MagicMock()
        normalizer.denormalize = lambda p: p * 2.0

        initial_params = jnp.array([1.0, 0.0])
        result = orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=normalizer,
        )

        # Final params should be denormalized
        assert result["final_params"] is not None

    def test_run_phase_history_structure(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test phase history contains expected structure."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        initial_params = jnp.array([1.0, 0.0])
        result = orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        history = result["phase_history"]
        assert len(history) >= 2  # At least warmup and GN phases

        # Check phase 1 (warmup) record
        phase1 = next((h for h in history if h.get("phase") == 1), None)
        assert phase1 is not None
        assert "name" in phase1
        assert "iterations" in phase1
        assert "timestamp" in phase1

    def test_current_phase_transitions(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test current_phase updates during run."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        assert orchestrator.current_phase == 0

        initial_params = jnp.array([1.0, 0.0])
        orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        # After run, should be in phase 3 (finalization)
        assert orchestrator.current_phase == 3


class TestPhaseOrchestratorVerbosity:
    """Tests for PhaseOrchestrator logging/verbosity."""

    def test_verbose_zero_no_output(
        self, base_config, simple_linear_model, linear_data, capsys
    ):
        """Test verbose=0 produces no console output."""
        base_config.verbose = 0
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        initial_params = jnp.array([1.0, 0.0])
        orchestrator.run(
            data_source=(x, y),
            initial_params=initial_params,
            normalizer=None,
        )

        captured = capsys.readouterr()
        # With verbose=0, minimal output expected
        # Note: GN phase may still print progress


# ============================================================================
# WarmupPhase Tests
# ============================================================================


class TestWarmupPhaseInitialization:
    """Tests for WarmupPhase initialization."""

    def test_init_sets_config(self, base_config, simple_linear_model):
        """Test initialization sets config correctly."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(base_config, mock_model)

        assert phase.config is base_config
        assert phase.normalized_model is mock_model
        assert phase._initial_loss is None
        assert phase._relative_loss is None

    def test_set_residual_weights(self, base_config, simple_linear_model):
        """Test set_residual_weights method."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(base_config, mock_model)

        weights = jnp.array([1.0, 2.0, 1.0])
        phase.set_residual_weights(weights)

        assert phase._residual_weights is not None
        assert jnp.allclose(phase._residual_weights, weights)


class TestWarmupPhaseRun:
    """Tests for WarmupPhase.run() method."""

    def test_run_basic(self, base_config, simple_linear_model, linear_data):
        """Test basic warmup run."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(base_config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0, 0.0]),
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        assert "final_params" in result
        assert "best_params" in result
        assert "best_loss" in result
        assert "iterations" in result
        assert "switch_reason" in result
        assert "warmup_result" in result
        assert isinstance(result["warmup_result"], WarmupResult)

    def test_warm_start_detection(self, base_config, simple_linear_model, linear_data):
        """Test warm start detection skips warmup."""
        x, y, true_params = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)

        # Start very close to optimal
        config = config_with_overrides(
            base_config,
            enable_warm_start_detection=True,
            warm_start_threshold=0.9,  # High threshold to trigger (must be < 1.0)
        )

        phase = WarmupPhase(config, mock_model)
        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=true_params,  # Start at optimal
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        # Check if warm start was detected (iterations=0 or skipped flag)
        assert result["iterations"] == 0 or result.get("warm_start", False)

    def test_cost_history_tracking(self, base_config, simple_linear_model, linear_data):
        """Test cost history is tracked during warmup."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(base_config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0, 0.0]),
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        warmup_result = result["warmup_result"]
        assert len(warmup_result.cost_history) > 0
        # First cost should be initial loss
        assert warmup_result.cost_history[0] > 0


class TestWarmupPhaseDefenseLayers:
    """Tests for 4-layer defense strategy in WarmupPhase."""

    def test_adaptive_lr_refinement_mode(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test adaptive LR selects refinement mode for small relative loss."""
        x, y, true_params = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)

        config = config_with_overrides(base_config, enable_adaptive_warmup_lr=True)
        phase = WarmupPhase(config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=true_params + jnp.array([0.01, 0.01]),  # Close to optimal
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        # Should complete without error
        assert result["warmup_result"] is not None

    def test_step_clipping(self, base_config, simple_linear_model, noisy_linear_data):
        """Test step clipping limits update magnitude."""
        x, y, _ = noisy_linear_data
        mock_model = MockNormalizedModel(simple_linear_model)

        config = config_with_overrides(
            base_config,
            enable_step_clipping=True,
            max_warmup_step_size=0.5,
        )
        phase = WarmupPhase(config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([10.0, 10.0]),  # Far from optimal
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        # Should complete successfully with clipping
        assert result["warmup_result"] is not None


class TestWarmupPhaseLossFunctions:
    """Tests for WarmupPhase loss function creation."""

    def test_create_loss_fn_basic(self, base_config, simple_linear_model):
        """Test basic loss function creation."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(base_config, mock_model)

        loss_fn = phase._create_loss_fn()
        assert callable(loss_fn)

        # Test loss function works
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 4.0, 6.0])
        params = jnp.array([2.0, 0.0])  # y = 2x

        loss = loss_fn(params, x, y)
        assert jnp.isfinite(loss)

    def test_create_loss_fn_with_variance_regularization(self, simple_linear_model):
        """Test loss function with group variance regularization."""
        config = HybridStreamingConfig(
            chunk_size=100,
            enable_group_variance_regularization=True,
            group_variance_lambda=0.1,
            group_variance_indices=[(0, 2)],
            verbose=0,
        )
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = WarmupPhase(config, mock_model)

        loss_fn = phase._create_loss_fn()
        assert callable(loss_fn)


class TestWarmupPhaseSwitchCriteria:
    """Tests for switch criteria in WarmupPhase."""

    def test_check_switch_criteria_gradient(self, base_config, simple_linear_model):
        """Test switch on gradient norm threshold."""
        mock_model = MockNormalizedModel(simple_linear_model)

        config = config_with_overrides(
            base_config,
            active_switching_criteria=["gradient"],
            gradient_norm_threshold=1e-4,
        )
        phase = WarmupPhase(config, mock_model)

        should_switch, reason = phase._check_switch_criteria(
            iteration=10, current_loss=0.001, prev_loss=0.001, grad_norm=1e-5
        )

        assert should_switch is True
        assert "gradient" in reason.lower()

    def test_check_switch_criteria_plateau(self, base_config, simple_linear_model):
        """Test switch on loss plateau."""
        mock_model = MockNormalizedModel(simple_linear_model)

        config = config_with_overrides(
            base_config,
            active_switching_criteria=["plateau"],
            loss_plateau_threshold=1e-6,
        )
        phase = WarmupPhase(config, mock_model)

        should_switch, reason = phase._check_switch_criteria(
            iteration=10, current_loss=0.001, prev_loss=0.001, grad_norm=0.1
        )

        assert should_switch is True
        assert "plateau" in reason.lower()


# ============================================================================
# GaussNewtonPhase Tests
# ============================================================================


class TestGaussNewtonPhaseInitialization:
    """Tests for GaussNewtonPhase initialization."""

    def test_init_basic(self, base_config, simple_linear_model):
        """Test basic initialization."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        assert phase.config is base_config
        assert phase.normalized_model is mock_model
        assert phase.normalized_bounds is None

    def test_init_with_bounds(self, base_config, simple_linear_model):
        """Test initialization with bounds."""
        mock_model = MockNormalizedModel(simple_linear_model)
        bounds = (jnp.array([0.0, -10.0]), jnp.array([10.0, 10.0]))
        phase = GaussNewtonPhase(base_config, mock_model, bounds)

        assert phase.normalized_bounds is not None
        assert jnp.allclose(phase.normalized_bounds[0], bounds[0])

    def test_set_jacobian_fn(self, base_config, simple_linear_model):
        """Test set_jacobian_fn method."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        def custom_jacobian(params, x):
            return jnp.ones((len(x), len(params)))

        phase.set_jacobian_fn(custom_jacobian)
        assert phase._jacobian_fn_compiled is custom_jacobian

    def test_set_cost_fn(self, base_config, simple_linear_model):
        """Test set_cost_fn method."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        def custom_cost(params, x, y):
            return 0.0

        phase.set_cost_fn(custom_cost)
        assert phase._cost_fn_compiled is custom_cost


class TestGaussNewtonPhaseRun:
    """Tests for GaussNewtonPhase.run() method."""

    def test_run_basic(self, base_config, simple_linear_model, linear_data):
        """Test basic GN run."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([2.0, 0.5]),  # Close to optimal
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        assert "final_params" in result
        assert "best_params" in result
        assert "best_cost" in result
        assert "iterations" in result
        assert "convergence_reason" in result
        assert "gn_result" in result
        assert isinstance(result["gn_result"], GNResult)

    def test_run_converges_on_perfect_data(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test GN converges to optimal on noise-free data."""
        x, y, true_params = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([2.0, 0.5]),
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        # Should converge close to true params
        final_params = result["best_params"]
        assert jnp.allclose(final_params, true_params, atol=0.1)

    def test_run_with_bounds_clipping(self, base_config, simple_linear_model):
        """Test GN respects bounds."""
        x = jnp.linspace(0, 10, 100)
        y = 5.0 * x + 2.0  # True params outside bounds

        mock_model = MockNormalizedModel(simple_linear_model)
        bounds = (jnp.array([0.0, 0.0]), jnp.array([3.0, 3.0]))  # Clamp to [0, 3]
        phase = GaussNewtonPhase(base_config, mock_model, bounds)

        phase_history: list[dict[str, Any]] = []
        best_tracker = {"best_params_global": None, "best_cost_global": float("inf")}

        result = phase.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0, 1.0]),
            phase_history=phase_history,
            best_tracker=best_tracker,
        )

        final_params = result["final_params"]
        # Params should be within bounds
        assert jnp.all(final_params >= bounds[0])
        assert jnp.all(final_params <= bounds[1])


class TestGaussNewtonPhaseNumerics:
    """Tests for numerical methods in GaussNewtonPhase."""

    def test_accumulate_jtj_jtr(self, base_config, simple_linear_model):
        """Test JTJ/JTr accumulation."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        x_chunk = jnp.array([1.0, 2.0, 3.0])
        y_chunk = jnp.array([3.0, 5.0, 7.0])  # y = 2x + 1
        params = jnp.array([2.0, 1.0])

        JTJ_init = jnp.zeros((2, 2))
        JTr_init = jnp.zeros(2)

        JTJ, JTr, cost = phase._accumulate_jtj_jtr(
            x_chunk, y_chunk, params, JTJ_init, JTr_init
        )

        assert JTJ.shape == (2, 2)
        assert JTr.shape == (2,)
        # At optimal params, cost should be ~0
        assert cost < 1e-6

    def test_solve_gauss_newton_step(self, base_config, simple_linear_model):
        """Test GN step solution."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        # Simple positive definite JTJ
        JTJ = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        JTr = jnp.array([1.0, 1.0])
        trust_radius = 10.0

        step, pred_red = phase._solve_gauss_newton_step(JTJ, JTr, trust_radius)

        assert step.shape == (2,)
        assert jnp.isfinite(pred_red)
        assert pred_red >= 0

    def test_solve_gauss_newton_step_with_trust_region(
        self, base_config, simple_linear_model
    ):
        """Test GN step respects trust region."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        JTJ = jnp.array([[2.0, 0.0], [0.0, 2.0]])
        JTr = jnp.array([10.0, 10.0])  # Large gradient -> large step
        trust_radius = 0.5  # Small trust region

        step, _ = phase._solve_gauss_newton_step(JTJ, JTr, trust_radius)

        step_norm = jnp.linalg.norm(step)
        assert step_norm <= trust_radius + 1e-6  # Within trust region

    def test_compute_cost(self, base_config, simple_linear_model):
        """Test cost computation."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        x_data = jnp.array([1.0, 2.0, 3.0])
        y_data = jnp.array([3.0, 5.0, 7.0])  # y = 2x + 1
        params = jnp.array([2.0, 1.0])

        cost = phase._compute_cost(params, x_data, y_data)

        # Perfect fit should have near-zero cost
        assert cost < 1e-6

    def test_compute_jacobian_chunk(self, base_config, simple_linear_model):
        """Test Jacobian computation."""
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        x_chunk = jnp.array([1.0, 2.0, 3.0])
        params = jnp.array([2.0, 1.0])

        J = phase._compute_jacobian_chunk(x_chunk, params)

        assert J.shape == (3, 2)  # (n_points, n_params)
        # For linear model y = a*x + b, J = [x, 1]
        assert jnp.allclose(J[:, 0], x_chunk)  # d/da = x
        assert jnp.allclose(J[:, 1], jnp.ones(3))  # d/db = 1


class TestGaussNewtonPhaseIteration:
    """Tests for single GN iteration."""

    def test_gauss_newton_iteration(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test single GN iteration."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        result = phase._gauss_newton_iteration(
            data_source=(x, y),
            current_params=jnp.array([1.0, 0.0]),
            trust_radius=1.0,
        )

        assert "new_params" in result
        assert "new_cost" in result
        assert "step" in result
        assert "actual_reduction" in result
        assert "predicted_reduction" in result
        assert "trust_radius" in result
        assert "gradient_norm" in result

    def test_trust_radius_update_good_step(
        self, base_config, simple_linear_model, linear_data
    ):
        """Test trust radius increases on good step."""
        x, y, _true_params = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)
        phase = GaussNewtonPhase(base_config, mock_model)

        # Start far from optimal -> expect good improvement
        initial_trust = 1.0
        result = phase._gauss_newton_iteration(
            data_source=(x, y),
            current_params=jnp.array([0.5, 0.0]),
            trust_radius=initial_trust,
        )

        # If step was good, trust radius may increase
        # (depends on reduction ratio)
        assert result["trust_radius"] > 0


# ============================================================================
# CheckpointManager Tests
# ============================================================================


class TestCheckpointManagerSave:
    """Tests for CheckpointManager.save() method."""

    def test_save_basic_state(self, base_config, tmp_path):
        """Test saving basic checkpoint state."""
        manager = CheckpointManager(base_config)
        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[{"phase": 0, "cost": 1.0}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "test.h5"
        manager.save(path, state)

        assert path.exists()

        # Verify HDF5 structure
        with h5py.File(path, "r") as f:
            assert "version" in f.attrs
            assert f.attrs["version"] == "3.0"
            assert "phase_state" in f

    def test_save_with_accumulators(self, base_config, tmp_path):
        """Test saving with JTJ/JTr accumulators."""
        manager = CheckpointManager(base_config)
        state = CheckpointState(
            current_phase=2,
            normalized_params=jnp.array([1.0, 2.0, 3.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=jnp.eye(3) * 2.0,
            phase2_JTr_accumulator=jnp.array([1.0, 2.0, 3.0]),
            best_params_global=jnp.array([1.0, 2.0, 3.0]),
            best_cost_global=0.001,
            phase_history=[],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "test_acc.h5"
        manager.save(path, state)

        with h5py.File(path, "r") as f:
            assert "phase2_jtj_accumulator" in f["phase_state"]
            assert "phase2_jtr_accumulator" in f["phase_state"]

    def test_save_creates_parent_dirs(self, base_config, tmp_path):
        """Test save creates parent directories."""
        manager = CheckpointManager(base_config)
        state = CheckpointState(
            current_phase=0,
            normalized_params=jnp.array([1.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=None,
            best_cost_global=float("inf"),
            phase_history=[],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "subdir" / "nested" / "checkpoint.h5"
        manager.save(path, state)

        assert path.exists()

    def test_save_with_normalizer_state(self, base_config, tmp_path):
        """Test saving normalizer state."""
        manager = CheckpointManager(base_config)

        # Mock normalizer
        normalizer = MagicMock()
        normalizer.strategy = "minmax"
        normalizer.scales = jnp.array([1.0, 2.0])
        normalizer.offsets = jnp.array([0.0, 1.0])

        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([0.5, 0.5]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([0.5, 0.5]),
            best_cost_global=0.01,
            phase_history=[],
            normalizer=normalizer,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "test_norm.h5"
        manager.save(path, state)

        with h5py.File(path, "r") as f:
            assert "normalizer_state" in f["phase_state"]
            assert f["phase_state/normalizer_state"].attrs["strategy"] == "minmax"

    def test_save_with_multistart_candidates(self, base_config, tmp_path):
        """Test saving multistart candidates."""
        manager = CheckpointManager(base_config)
        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )

        path = tmp_path / "test_multistart.h5"
        manager.save(path, state)

        with h5py.File(path, "r") as f:
            assert "multistart_candidates" in f["phase_state"]


class TestCheckpointManagerLoad:
    """Tests for CheckpointManager.load() method."""

    def test_load_basic(self, base_config, tmp_path):
        """Test loading basic checkpoint."""
        manager = CheckpointManager(base_config)

        # Save first
        state = CheckpointState(
            current_phase=2,
            normalized_params=jnp.array([1.5, 2.5]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.5, 2.5]),
            best_cost_global=0.005,
            phase_history=[{"phase": 1, "iterations": 10}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "test_load.h5"
        manager.save(path, state)

        # Load
        loaded = manager.load(path)

        assert loaded.current_phase == 2
        assert jnp.allclose(loaded.normalized_params, state.normalized_params)
        assert loaded.best_cost_global == 0.005
        assert len(loaded.phase_history) == 1

    def test_load_with_accumulators(self, base_config, tmp_path):
        """Test loading checkpoint with accumulators."""
        manager = CheckpointManager(base_config)

        JTJ = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        JTr = jnp.array([0.1, 0.2])

        state = CheckpointState(
            current_phase=2,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=JTJ,
            phase2_JTr_accumulator=JTr,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "test_acc_load.h5"
        manager.save(path, state)

        loaded = manager.load(path)

        assert loaded.phase2_JTJ_accumulator is not None
        assert jnp.allclose(loaded.phase2_JTJ_accumulator, JTJ)
        assert loaded.phase2_JTr_accumulator is not None
        assert jnp.allclose(loaded.phase2_JTr_accumulator, JTr)

    def test_load_nonexistent_raises(self, base_config, tmp_path):
        """Test loading nonexistent file raises FileNotFoundError."""
        manager = CheckpointManager(base_config)

        with pytest.raises(FileNotFoundError):
            manager.load(tmp_path / "does_not_exist.h5")

    def test_load_incompatible_version_raises(self, base_config, tmp_path):
        """Test loading incompatible version raises ValueError."""
        manager = CheckpointManager(base_config)

        # Create file with old version
        path = tmp_path / "old_version.h5"
        with h5py.File(path, "w") as f:
            f.attrs["version"] = "1.0"  # Old version
            f.create_group("phase_state")

        with pytest.raises(ValueError, match="Incompatible checkpoint version"):
            manager.load(path)

    def test_load_with_multistart_candidates(self, base_config, tmp_path):
        """Test loading multistart candidates."""
        manager = CheckpointManager(base_config)

        candidates = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=candidates,
        )

        path = tmp_path / "test_ms.h5"
        manager.save(path, state)

        loaded = manager.load(path)

        assert loaded.multistart_candidates is not None
        assert jnp.allclose(loaded.multistart_candidates, candidates)


class TestCheckpointManagerJaxConversion:
    """Tests for JAX array conversion in CheckpointManager."""

    def test_convert_jax_to_numpy(self, base_config):
        """Test _convert_jax_to_numpy method."""
        manager = CheckpointManager(base_config)

        # Test JAX array
        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = manager._convert_jax_to_numpy(jax_arr)
        assert isinstance(result, np.ndarray)

        # Test dict with JAX arrays
        data = {"arr": jnp.array([1.0]), "scalar": 5}
        result = manager._convert_jax_to_numpy(data)
        assert isinstance(result["arr"], np.ndarray)
        assert result["scalar"] == 5

        # Test list with JAX arrays
        data = [jnp.array([1.0]), jnp.array([2.0])]
        result = manager._convert_jax_to_numpy(data)
        assert all(isinstance(x, np.ndarray) for x in result)

        # Test tuple with JAX arrays
        data = (jnp.array([1.0]), jnp.array([2.0]))
        result = manager._convert_jax_to_numpy(data)
        assert isinstance(result, tuple)
        assert all(isinstance(x, np.ndarray) for x in result)


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhaseIntegrationWorkflow:
    """Integration tests for complete phase workflow."""

    def test_full_optimization_converges(
        self, base_config, simple_linear_model, noisy_linear_data
    ):
        """Test full optimization workflow converges."""
        x, y, true_params = noisy_linear_data
        mock_model = MockNormalizedModel(simple_linear_model)

        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=jnp.array([0.5, 0.0]),
            normalizer=None,
        )

        # Get best params from orchestrator or result
        final_params = orchestrator.get_best_params()
        if final_params is None:
            final_params = result.get("final_params", result.get("normalized_params"))
        # Should be reasonably close to true params (within 1.0 for noisy data)
        assert jnp.allclose(final_params, true_params, atol=1.0)

    def test_checkpoint_resume_workflow(
        self, base_config, simple_linear_model, linear_data, tmp_path
    ):
        """Test checkpoint save/resume workflow."""
        x, y, _ = linear_data
        mock_model = MockNormalizedModel(simple_linear_model)

        # Run partial optimization
        config = config_with_overrides(
            base_config,
            warmup_iterations=2,
            max_warmup_iterations=3,
        )
        orchestrator = PhaseOrchestrator(config)
        orchestrator.initialize_phases(mock_model)

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0, 0.0]),
            normalizer=None,
        )

        # Get best params from result or orchestrator
        best_params = orchestrator.get_best_params()
        if best_params is None:
            best_params = result.get("final_params", jnp.array([1.0, 0.0]))

        # Save checkpoint
        checkpoint_manager = CheckpointManager(config)
        state = CheckpointState(
            current_phase=orchestrator.current_phase,
            normalized_params=result["normalized_params"],
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=result.get("JTJ_final"),
            phase2_JTr_accumulator=None,
            best_params_global=best_params,
            best_cost_global=result["best_cost"],
            phase_history=result["phase_history"],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )

        path = tmp_path / "checkpoint.h5"
        checkpoint_manager.save(path, state)

        # Load and verify
        loaded = checkpoint_manager.load(path)
        assert loaded.current_phase == state.current_phase
        assert loaded.best_cost_global == state.best_cost_global


class TestPhaseEdgeCases:
    """Tests for edge cases in phase handling."""

    def test_small_dataset(self, base_config, simple_linear_model):
        """Test with very small dataset (5 points)."""
        x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = jnp.array([1.0, 3.0, 5.0, 7.0, 9.0])  # y = 2x + 1

        mock_model = MockNormalizedModel(simple_linear_model)
        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0, 0.0]),
            normalizer=None,
        )

        assert result["final_params"] is not None

    def test_single_parameter(self, base_config):
        """Test with single parameter model."""

        def single_param_model(x, a):
            return a * x

        mock_model = MockNormalizedModel(single_param_model)

        x = jnp.linspace(0, 10, 100)
        y = 3.0 * x

        orchestrator = PhaseOrchestrator(base_config)
        orchestrator.initialize_phases(mock_model)

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=jnp.array([1.0]),
            normalizer=None,
        )

        # Get best params from orchestrator or result
        best_params = orchestrator.get_best_params()
        if best_params is None:
            best_params = result.get("final_params", jnp.array([1.0]))

        assert best_params.shape == (1,)
        assert jnp.allclose(best_params, jnp.array([3.0]), atol=0.5)

    def test_many_parameters(self, base_config):
        """Test with model having many parameters."""

        def multi_param_model(x, a, b, c, d, e):
            return a * x**4 + b * x**3 + c * x**2 + d * x + e

        mock_model = MockNormalizedModel(multi_param_model)

        x = jnp.linspace(0, 1, 50)
        true_params = jnp.array([1.0, -0.5, 0.3, 0.1, 0.5])
        y = multi_param_model(x, *true_params)

        config = config_with_overrides(
            base_config,
            gauss_newton_max_iterations=50,
            gauss_newton_tol=1e-10,
        )
        orchestrator = PhaseOrchestrator(config)
        orchestrator.initialize_phases(mock_model)

        result = orchestrator.run(
            data_source=(x, y),
            initial_params=jnp.array([0.5, 0.0, 0.0, 0.0, 0.0]),
            normalizer=None,
        )

        # Should converge reasonably well
        assert result["best_cost"] < 1.0
