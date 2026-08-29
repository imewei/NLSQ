"""Tests for streaming phase classes.

Tests for WarmupPhase, GaussNewtonPhase, PhaseOrchestrator, and CheckpointManager.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases import (
    CheckpointManager,
    CheckpointState,
    GaussNewtonPhase,
    GNResult,
    PhaseOrchestrator,
    PhaseOrchestratorResult,
    WarmupPhase,
    WarmupResult,
)


class TestWarmupResult:
    """Tests for WarmupResult dataclass."""

    def test_creation(self):
        """Test WarmupResult can be created."""
        result = WarmupResult(
            params=jnp.array([1.0, 2.0]),
            cost=0.01,
            iterations=100,
            converged=True,
            cost_history=[0.5, 0.1, 0.01],
        )
        assert result.params.shape == (2,)
        assert result.cost == 0.01
        assert result.iterations == 100
        assert result.converged is True
        assert len(result.cost_history) == 3

    def test_not_converged(self):
        """Test WarmupResult with converged=False."""
        result = WarmupResult(
            params=jnp.array([1.0]),
            cost=0.5,
            iterations=10,
            converged=False,
            cost_history=[1.0, 0.5],
        )
        assert result.converged is False


class TestGNResult:
    """Tests for GNResult dataclass."""

    def test_creation(self):
        """Test GNResult can be created."""
        result = GNResult(
            params=jnp.array([1.0, 2.0, 3.0]),
            cost=0.001,
            iterations=50,
            converged=True,
        )
        assert result.params.shape == (3,)
        assert result.cost == 0.001
        assert result.iterations == 50
        assert result.converged is True
        assert result.jacobian is None
        assert result.cov is None

    def test_with_jacobian_and_cov(self):
        """Test GNResult with jacobian and covariance."""
        result = GNResult(
            params=jnp.array([1.0, 2.0]),
            cost=0.01,
            iterations=10,
            converged=True,
            jacobian=jnp.eye(2),
            cov=jnp.eye(2) * 0.01,
        )
        assert result.jacobian is not None
        assert result.cov is not None
        assert result.jacobian.shape == (2, 2)


class TestPhaseOrchestratorResult:
    """Tests for PhaseOrchestratorResult dataclass."""

    def test_creation(self):
        """Test PhaseOrchestratorResult can be created."""
        result = PhaseOrchestratorResult(
            params=jnp.array([1.0, 2.0]),
            normalized_params=jnp.array([0.5, 0.5]),
            cost=0.001,
            warmup_result=None,
            gn_result=None,
            phase_history=[{"phase": 0}],
            total_time=10.5,
        )
        assert result.params.shape == (2,)
        assert result.cost == 0.001
        assert result.total_time == 10.5

    def test_with_phase_results(self):
        """Test PhaseOrchestratorResult with phase results."""
        warmup = WarmupResult(
            params=jnp.array([1.0]),
            cost=0.1,
            iterations=10,
            converged=True,
            cost_history=[0.5, 0.1],
        )
        gn = GNResult(
            params=jnp.array([1.0]),
            cost=0.01,
            iterations=5,
            converged=True,
        )
        result = PhaseOrchestratorResult(
            params=jnp.array([1.0]),
            normalized_params=jnp.array([0.5]),
            cost=0.01,
            warmup_result=warmup,
            gn_result=gn,
            phase_history=[],
            total_time=5.0,
        )
        assert result.warmup_result is not None
        assert result.gn_result is not None


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_creation(self):
        """Test CheckpointState can be created."""
        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[{"phase": 0, "name": "setup"}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )
        assert state.current_phase == 1
        assert state.normalized_params.shape == (2,)
        assert state.best_cost_global == 0.01

    def test_with_accumulators(self):
        """Test CheckpointState with JTJ/JTr accumulators."""
        n_params = 3
        state = CheckpointState(
            current_phase=2,
            normalized_params=jnp.array([1.0, 2.0, 3.0]),
            phase1_optimizer_state={"step": 100},
            phase2_JTJ_accumulator=jnp.eye(n_params),
            phase2_JTr_accumulator=jnp.zeros(n_params),
            best_params_global=jnp.array([1.0, 2.0, 3.0]),
            best_cost_global=0.005,
            phase_history=[{"phase": 0}, {"phase": 1}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )
        assert state.phase2_JTJ_accumulator is not None
        assert state.phase2_JTJ_accumulator.shape == (3, 3)


class TestWarmupPhase:
    """Tests for WarmupPhase class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=50,
            max_warmup_iterations=100,
            lbfgs_initial_step_size=0.1,
            gradient_norm_threshold=1e-6,
            verbose=0,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock normalized model."""
        model = MagicMock()
        model.return_value = jnp.array([1.0, 2.0, 3.0])
        return model

    def test_initialization(self, config, mock_model):
        """Test WarmupPhase initialization."""
        phase = WarmupPhase(config, mock_model)
        assert phase.config is config
        assert phase.normalized_model is mock_model

    def test_create_loss_fn(self, config, mock_model):
        """Test _create_loss_fn creates callable."""
        phase = WarmupPhase(config, mock_model)
        loss_fn = phase._create_loss_fn()
        assert callable(loss_fn)

    def test_group_variance_regularization_unequal_trailing_group(self):
        """Trailing group with unequal size must not pull values from the
        previous group.

        Regression test: dynamic_slice(params, (start,), (max_group_size,))
        clamps its start index to stay in bounds, so a trailing group whose
        start + max_group_size overruns len(params) silently slides the
        window into the previous group instead of raising. Padding params
        with max_group_size zeros (mirrors the fix already applied in
        adaptive_hybrid.py's _create_warmup_loss_fn) avoids the clamp.
        """
        config = HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=50,
            max_warmup_iterations=100,
            lbfgs_initial_step_size=0.1,
            gradient_norm_threshold=1e-6,
            verbose=0,
            enable_group_variance_regularization=True,
            group_variance_lambda=1.0,
            group_variance_indices=[(0, 4), (4, 6)],
        )

        def identity_model(x_batch, *params):
            # Zero residuals so the loss is purely the variance penalty.
            return jnp.zeros_like(x_batch)

        phase = WarmupPhase(config, identity_model)
        loss_fn = phase._create_loss_fn()

        params = jnp.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0])
        x_batch = jnp.zeros(5)
        y_batch = jnp.zeros(5)

        loss = loss_fn(params, x_batch, y_batch)

        # Group (0,4): [1,2,3,4] -> population variance 1.25
        # Group (4,6): [10,20]   -> population variance 25.0
        # With the clamp bug, the trailing group's window shifts into the
        # first group, giving a variance far below the correct 25.0.
        expected = 1.25 + 25.0
        assert float(loss) == pytest.approx(expected, rel=1e-5)

    def test_best_tracker_cost_global_is_ssr_not_mse(self):
        """best_tracker['best_cost_global'] must be stored on the same SSR
        (sum of squared residuals) scale that GaussNewtonPhase uses, not
        raw MSE.

        Regression test: storing Phase 1's MSE loss directly meant Phase
        2's `new_cost < best_cost_global` (SSR vs MSE) was essentially
        never true for datasets beyond a few points, silently freezing
        best_params_global at wherever Phase 1 left it. Mirrors the fix
        already applied in adaptive_hybrid.py's _update_global_best.
        """
        n = 5
        config = HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=1,
            max_warmup_iterations=1,
            lbfgs_initial_step_size=0.01,
            gradient_norm_threshold=1e-12,
            verbose=0,
            enable_warm_start_detection=False,
        )

        def model(x_batch, *params):
            return jnp.full_like(x_batch, params[0])

        phase = WarmupPhase(config, model)

        x_data = jnp.zeros(n)
        y_data = jnp.ones(n)
        best_tracker: dict = {}

        result = phase.run(
            data_source=(x_data, y_data),
            initial_params=jnp.array([0.0]),
            phase_history=[],
            best_tracker=best_tracker,
        )

        assert "best_cost_global" in best_tracker
        expected_ssr = result["final_loss"] * n
        assert best_tracker["best_cost_global"] == pytest.approx(expected_ssr, rel=1e-6)
        # The bug stored raw MSE (== final_loss) instead of the SSR-scaled
        # value -- fail loudly if this regresses.
        assert best_tracker["best_cost_global"] != pytest.approx(result["final_loss"])


class TestGaussNewtonPhase:
    """Tests for GaussNewtonPhase class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=50,
            max_warmup_iterations=100,
            gauss_newton_max_iterations=50,
            gauss_newton_tol=1e-8,
            trust_region_initial=1.0,
            verbose=0,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock normalized model."""
        model = MagicMock()
        model.return_value = jnp.array([1.0, 2.0, 3.0])
        return model

    def test_initialization(self, config, mock_model):
        """Test GaussNewtonPhase initialization."""
        phase = GaussNewtonPhase(config, mock_model)
        assert phase.config is config
        assert phase.normalized_model is mock_model
        assert phase.normalized_bounds is None

    def test_initialization_with_bounds(self, config, mock_model):
        """Test GaussNewtonPhase initialization with bounds."""
        bounds = (jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0]))
        phase = GaussNewtonPhase(config, mock_model, bounds)
        assert phase.normalized_bounds is not None

    def test_rejected_first_step_does_not_falsely_converge(self, config):
        """A rejected first Gauss-Newton iteration must not report false
        'Cost change below tolerance' convergence.

        Regression test: on the first iteration, prev_cost is still inf, so
        the pre-fix fallback `cost_before_step = new_cost` made
        `cost_change = abs(cost_before_step - new_cost)` exactly zero
        whenever that first step was rejected -- reporting convergence
        after one rejected step with zero real progress, regardless of how
        far new_cost actually was from the true initial cost.
        """

        def model(x, a):
            return a * x

        phase = GaussNewtonPhase(config, model)

        # _gauss_newton_iteration is mocked to a value (-999.0) that can
        # never equal a real sum-of-squared-residuals (always >= 0), so a
        # correct cost_before_step (the true pre-loop SSR) makes cost_change
        # large and non-zero, while the pre-fix fallback (new_cost itself)
        # would make it exactly zero.
        phase._gauss_newton_iteration = (
            lambda data_source, params, trust_radius: {
                "new_params": params,
                "new_cost": -999.0,
                "gradient_norm": 10.0,  # above gauss_newton_tol: no gradient-based convergence
                "actual_reduction": -1.0,  # rejected step
                "trust_radius": trust_radius,
            }
        )

        x_data = jnp.array([1.0, 2.0, 3.0, 4.0])
        y_data = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = phase.run(
            data_source=(x_data, y_data),
            initial_params=jnp.array([1.0]),
            phase_history=[],
            best_tracker={},
        )

        assert result["convergence_reason"] != "Cost change below tolerance"
        assert result["convergence_reason"] == "Maximum iterations reached"
        assert result["iterations"] == config.gauss_newton_max_iterations


class TestPhaseOrchestrator:
    """Tests for PhaseOrchestrator class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=5,
            max_warmup_iterations=10,
            gauss_newton_max_iterations=10,
            verbose=0,
        )

    def test_initialization(self, config):
        """Test PhaseOrchestrator initialization."""
        orchestrator = PhaseOrchestrator(config)
        assert orchestrator.config is config
        assert orchestrator.warmup_phase is None
        assert orchestrator.gn_phase is None
        assert orchestrator.phase_history == []
        assert orchestrator.current_phase == 0

    def test_get_phase_history_empty(self, config):
        """Test get_phase_history returns empty list initially."""
        orchestrator = PhaseOrchestrator(config)
        assert orchestrator.get_phase_history() == []

    def test_get_best_params_none(self, config):
        """Test get_best_params returns None initially."""
        orchestrator = PhaseOrchestrator(config)
        assert orchestrator.get_best_params() is None

    def test_get_best_cost_inf(self, config):
        """Test get_best_cost returns inf initially."""
        orchestrator = PhaseOrchestrator(config)
        assert orchestrator.get_best_cost() == float("inf")


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return HybridStreamingConfig(
            chunk_size=1000,
            warmup_iterations=50,
            max_warmup_iterations=100,
            enable_checkpoints=True,
            checkpoint_frequency=10,
            verbose=0,
        )

    def test_initialization(self, config):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(config)
        assert manager.config is config

    def test_save_checkpoint_creates_file(self, config, tmp_path):
        """Test save_checkpoint creates a file."""
        manager = CheckpointManager(config)
        state = CheckpointState(
            current_phase=1,
            normalized_params=jnp.array([1.0, 2.0]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=None,
            phase2_JTr_accumulator=None,
            best_params_global=jnp.array([1.0, 2.0]),
            best_cost_global=0.01,
            phase_history=[{"phase": 0, "name": "setup"}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )
        checkpoint_path = tmp_path / "test_checkpoint.h5"
        manager.save(checkpoint_path, state)
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, config, tmp_path):
        """Test loading a saved checkpoint."""
        manager = CheckpointManager(config)
        state = CheckpointState(
            current_phase=2,
            normalized_params=jnp.array([1.5, 2.5, 3.5]),
            phase1_optimizer_state=None,
            phase2_JTJ_accumulator=jnp.eye(3),
            phase2_JTr_accumulator=jnp.zeros(3),
            best_params_global=jnp.array([1.5, 2.5, 3.5]),
            best_cost_global=0.005,
            phase_history=[{"phase": 0}, {"phase": 1}],
            normalizer=None,
            tournament_selector=None,
            multistart_candidates=None,
        )
        checkpoint_path = tmp_path / "test_checkpoint.h5"
        manager.save(checkpoint_path, state)

        loaded_state = manager.load(checkpoint_path)
        assert loaded_state.current_phase == 2
        assert jnp.allclose(loaded_state.normalized_params, state.normalized_params)
        assert loaded_state.best_cost_global == 0.005

    def test_load_nonexistent_raises(self, config, tmp_path):
        """Test loading nonexistent checkpoint raises error."""
        manager = CheckpointManager(config)
        with pytest.raises(FileNotFoundError):
            manager.load(tmp_path / "nonexistent.h5")


class TestPhaseIntegration:
    """Integration tests for phase classes working together."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model function."""

        def model(x, a, b):
            return a * x + b

        return model

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        x = jnp.linspace(0, 10, 100)
        y = 2.0 * x + 1.0 + 0.01 * jnp.sin(x)
        return x, y

    def test_orchestrator_initialize_phases(self, simple_model):
        """Test PhaseOrchestrator.initialize_phases."""
        config = HybridStreamingConfig(
            chunk_size=50,
            warmup_iterations=3,
            max_warmup_iterations=5,
            gauss_newton_max_iterations=5,
            verbose=0,
        )
        orchestrator = PhaseOrchestrator(config)

        # Create a mock normalized model
        mock_model = MagicMock()
        mock_model.return_value = jnp.array([1.0])

        orchestrator.initialize_phases(mock_model)
        assert orchestrator.warmup_phase is not None
        assert orchestrator.gn_phase is not None
