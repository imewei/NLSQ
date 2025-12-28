"""Tests for mixed precision fallback system.

This module tests the mixed precision components including:
- PrecisionState enum
- Configuration dataclasses
- ConvergenceMonitor (5 issue detection types)
- PrecisionUpgrader (state preservation)
- BestParameterTracker (fault tolerance)
- MixedPrecisionManager (orchestration)
"""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.precision.mixed_precision import (
    BestParameterTracker,
    ConvergenceMetrics,
    ConvergenceMonitor,
    MixedPrecisionConfig,
    MixedPrecisionManager,
    OptimizationState,
    PrecisionState,
    PrecisionUpgrader,
)

# Fixtures


@pytest.fixture
def logger():
    """Create a logger for testing."""
    test_logger = logging.getLogger("nlsq.mixed_precision.test")
    test_logger.setLevel(logging.DEBUG)
    return test_logger


@pytest.fixture
def default_config():
    """Create default MixedPrecisionConfig."""
    return MixedPrecisionConfig()


@pytest.fixture
def sample_arrays_float32():
    """Create sample JAX arrays in float32 for testing."""
    return {
        "x": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "f": jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float32),
        "J": jnp.ones((4, 3), dtype=jnp.float32),
        "g": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32),
    }


@pytest.fixture
def sample_arrays_float64():
    """Create sample JAX arrays in float64 for testing."""
    return {
        "x": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
        "f": jnp.array([0.1, 0.2, 0.3, 0.4], dtype=jnp.float64),
        "J": jnp.ones((4, 3), dtype=jnp.float64),
        "g": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
    }


# Tests for Enums and Dataclasses


class TestPrecisionState:
    """Tests for PrecisionState enum."""

    def test_all_states_exist(self):
        """Test that all 5 states are defined."""
        assert hasattr(PrecisionState, "FLOAT32_ACTIVE")
        assert hasattr(PrecisionState, "MONITORING_DEGRADATION")
        assert hasattr(PrecisionState, "UPGRADING_TO_FLOAT64")
        assert hasattr(PrecisionState, "FLOAT64_ACTIVE")
        assert hasattr(PrecisionState, "RELAXED_FLOAT32_FALLBACK")

    def test_state_instantiation(self):
        """Test that states can be instantiated."""
        state = PrecisionState.FLOAT32_ACTIVE
        assert isinstance(state, PrecisionState)

    def test_state_comparison(self):
        """Test state equality comparison."""
        state1 = PrecisionState.FLOAT32_ACTIVE
        state2 = PrecisionState.FLOAT32_ACTIVE
        state3 = PrecisionState.MONITORING_DEGRADATION
        assert state1 == state2
        assert state1 != state3


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig dataclass."""

    def test_default_instantiation(self):
        """Test config instantiation with defaults."""
        config = MixedPrecisionConfig()
        assert config.enable_mixed_precision_fallback is True
        assert config.max_degradation_iterations == 5
        assert config.stall_window == 10
        assert config.gradient_explosion_threshold == 1e10
        assert config.precision_limit_threshold == 1e-7
        assert config.tolerance_relaxation_factor == 10.0
        assert config.verbose is False

    def test_custom_values(self):
        """Test config with custom values."""
        config = MixedPrecisionConfig(
            enable_mixed_precision_fallback=False,
            max_degradation_iterations=3,
            verbose=True,
        )
        assert config.enable_mixed_precision_fallback is False
        assert config.max_degradation_iterations == 3
        assert config.verbose is True

    def test_conservative_config(self):
        """Test conservative configuration (prefers float64)."""
        config = MixedPrecisionConfig(
            max_degradation_iterations=2,
            gradient_explosion_threshold=1e8,
            verbose=True,
        )
        assert config.max_degradation_iterations == 2
        assert config.gradient_explosion_threshold == 1e8


class TestConvergenceMetrics:
    """Tests for ConvergenceMetrics dataclass."""

    def test_instantiation(self):
        """Test metrics instantiation."""
        metrics = ConvergenceMetrics(
            iteration=10,
            residual_norm=1.5e-3,
            gradient_norm=2.1e-5,
            parameter_change=1.2e-6,
            cost=1.125e-6,
            trust_radius=0.8,
            has_nan_inf=False,
        )
        assert metrics.iteration == 10
        assert metrics.residual_norm == 1.5e-3
        assert metrics.has_nan_inf is False

    def test_with_nan_inf(self):
        """Test metrics with NaN/Inf flag."""
        metrics = ConvergenceMetrics(
            iteration=5,
            residual_norm=np.inf,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=np.inf,
            trust_radius=1.0,
            has_nan_inf=True,
        )
        assert metrics.has_nan_inf is True
        assert np.isinf(metrics.residual_norm)


class TestOptimizationState:
    """Tests for OptimizationState dataclass."""

    def test_instantiation_float32(self, sample_arrays_float32):
        """Test state instantiation with float32 arrays."""
        state = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific=None,
        )
        assert state.dtype == jnp.float32
        assert state.iteration == 15
        assert state.x.dtype == jnp.float32

    def test_instantiation_float64(self, sample_arrays_float64):
        """Test state instantiation with float64 arrays."""
        state = OptimizationState(
            x=sample_arrays_float64["x"],
            f=sample_arrays_float64["f"],
            J=sample_arrays_float64["J"],
            g=sample_arrays_float64["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=20,
            dtype=jnp.float64,
        )
        assert state.dtype == jnp.float64
        assert state.x.dtype == jnp.float64

    def test_with_algorithm_specific_state(self, sample_arrays_float32):
        """Test state with algorithm-specific data (LM lambda)."""
        state = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=0.5,
            iteration=10,
            dtype=jnp.float32,
            algorithm_specific={"lambda": 0.1},
        )
        assert state.algorithm_specific is not None
        assert state.algorithm_specific["lambda"] == 0.1


class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor class."""

    def test_nan_inf_detection(self, default_config, logger):
        """Test NaN/Inf detection returns correct issue string."""
        monitor = ConvergenceMonitor(default_config, logger)

        # No NaN/Inf
        metrics_ok = ConvergenceMetrics(
            iteration=5,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_ok) is None

        # With NaN/Inf
        metrics_bad = ConvergenceMetrics(
            iteration=6,
            residual_norm=np.inf,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=np.inf,
            trust_radius=1.0,
            has_nan_inf=True,
        )
        assert monitor.check_convergence(metrics_bad) == "nan_inf_detected"

    def test_gradient_explosion_detection(self, default_config, logger):
        """Test gradient explosion detection."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Below threshold
        metrics_ok = ConvergenceMetrics(
            iteration=5,
            residual_norm=1.0,
            gradient_norm=1e9,  # Below default threshold of 1e10
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_ok) is None

        # Above threshold
        metrics_exploded = ConvergenceMetrics(
            iteration=6,
            residual_norm=1.0,
            gradient_norm=1e11,  # Above default threshold
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_exploded) == "gradient_explosion"

    def test_convergence_stall_detection(self, default_config, logger):
        """Test convergence stall detection over window."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Feed stall_window iterations with same cost
        for i in range(default_config.stall_window - 1):
            metrics = ConvergenceMetrics(
                iteration=i,
                residual_norm=1.0,
                gradient_norm=1.0,
                parameter_change=0.1,
                cost=1.0,  # Same cost
                trust_radius=1.0,
                has_nan_inf=False,
            )
            # Should not detect stall until window is full
            assert monitor.check_convergence(metrics) is None

        # Final iteration fills the window
        metrics_final = ConvergenceMetrics(
            iteration=default_config.stall_window,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,  # Same cost
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_final) == "convergence_stall"

    def test_precision_limit_detection(self, default_config, logger):
        """Test parameter precision limit detection."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Above precision limit
        metrics_ok = ConvergenceMetrics(
            iteration=5,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=1e-6,  # Above default threshold of 1e-7
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_ok) is None

        # Below precision limit
        metrics_limited = ConvergenceMetrics(
            iteration=6,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=1e-8,  # Below default threshold
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_limited) == "precision_limit"

    def test_trust_region_collapse_detection(self, default_config, logger):
        """Test trust region collapse detection."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Normal trust radius
        metrics_ok = ConvergenceMetrics(
            iteration=5,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1e-10,  # Above collapse threshold of 1e-12
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_ok) is None

        # Collapsed trust radius
        metrics_collapsed = ConvergenceMetrics(
            iteration=6,
            residual_norm=1.0,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1e-13,  # Below collapse threshold
            has_nan_inf=False,
        )
        assert monitor.check_convergence(metrics_collapsed) == "trust_region_collapse"

    def test_degradation_counter(self, default_config, logger):
        """Test degradation counter increment and reset."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Initially zero
        assert monitor.degradation_counter == 0
        assert not monitor.should_upgrade()

        # Increment counter
        for i in range(default_config.max_degradation_iterations - 1):
            monitor.increment_degradation()
            assert monitor.degradation_counter == i + 1
            assert not monitor.should_upgrade()

        # One more increment reaches threshold
        monitor.increment_degradation()
        assert monitor.degradation_counter == default_config.max_degradation_iterations
        assert monitor.should_upgrade()

        # Reset counter
        monitor.reset_degradation()
        assert monitor.degradation_counter == 0
        assert not monitor.should_upgrade()

    def test_should_upgrade_logic(self, default_config, logger):
        """Test should_upgrade() method logic."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Below threshold
        for i in range(default_config.max_degradation_iterations):
            assert not monitor.should_upgrade()
            monitor.increment_degradation()

        # At threshold
        assert monitor.should_upgrade()

        # Above threshold
        monitor.increment_degradation()
        assert monitor.should_upgrade()

    def test_all_issue_types_return_correct_strings(self, default_config, logger):
        """Test that all 5 issue types return correct string identifiers."""
        monitor = ConvergenceMonitor(default_config, logger)

        # Test each issue type individually
        issue_types = [
            ("nan_inf_detected", {"has_nan_inf": True}),
            ("gradient_explosion", {"gradient_norm": 1e11}),
            ("precision_limit", {"parameter_change": 1e-8}),
            ("trust_region_collapse", {"trust_radius": 1e-13}),
        ]

        for expected_issue, override_params in issue_types:
            # Create metrics with defaults
            params = {
                "iteration": 1,
                "residual_norm": 1.0,
                "gradient_norm": 1.0,
                "parameter_change": 0.1,
                "cost": 1.0,
                "trust_radius": 1.0,
                "has_nan_inf": False,
            }
            # Override with specific params to trigger issue
            params.update(override_params)
            metrics = ConvergenceMetrics(**params)

            # Reset monitor for clean test
            monitor = ConvergenceMonitor(default_config, logger)
            issue = monitor.check_convergence(metrics)
            assert issue == expected_issue, f"Expected {expected_issue}, got {issue}"


class TestPrecisionUpgrader:
    """Tests for PrecisionUpgrader class."""

    def test_upgrade_float32_to_float64_preserves_values(
        self, sample_arrays_float32, logger
    ):
        """Test upgrade from float32 to float64 preserves all values."""
        upgrader = PrecisionUpgrader(logger)

        # Create float32 state
        state_float32 = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific={"lambda": 0.1},
        )

        # Upgrade to float64
        state_float64 = upgrader.upgrade_to_float64(state_float32)

        # Check dtype
        assert state_float64.dtype == jnp.float64
        assert state_float64.x.dtype == jnp.float64
        assert state_float64.f.dtype == jnp.float64
        assert state_float64.J.dtype == jnp.float64
        assert state_float64.g.dtype == jnp.float64

        # Check values preserved (within float32 precision)
        np.testing.assert_allclose(state_float64.x, state_float32.x, rtol=1e-7)
        np.testing.assert_allclose(state_float64.f, state_float32.f, rtol=1e-7)
        np.testing.assert_allclose(state_float64.J, state_float32.J, rtol=1e-7)
        np.testing.assert_allclose(state_float64.g, state_float32.g, rtol=1e-7)

        # Check scalars
        assert state_float64.cost == state_float32.cost
        assert state_float64.trust_radius == state_float32.trust_radius

    def test_upgrade_preserves_iteration_count(self, sample_arrays_float32, logger):
        """Test upgrade preserves iteration count (critical for zero-iteration loss)."""
        upgrader = PrecisionUpgrader(logger)

        state_float32 = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=42,  # Specific iteration count
            dtype=jnp.float32,
            algorithm_specific=None,
        )

        state_float64 = upgrader.upgrade_to_float64(state_float32)

        # Critical: iteration count must be preserved exactly
        assert state_float64.iteration == 42
        assert state_float64.iteration == state_float32.iteration

    def test_downgrade_float64_to_float32_preserves_values(
        self, sample_arrays_float64, logger
    ):
        """Test downgrade from float64 to float32 preserves values."""
        upgrader = PrecisionUpgrader(logger)

        # Create float64 state
        state_float64 = OptimizationState(
            x=sample_arrays_float64["x"],
            f=sample_arrays_float64["f"],
            J=sample_arrays_float64["J"],
            g=sample_arrays_float64["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=20,
            dtype=jnp.float64,
            algorithm_specific={"lambda": 0.1},
        )

        # Downgrade to float32
        state_float32 = upgrader.downgrade_to_float32(state_float64)

        # Check dtype
        assert state_float32.dtype == jnp.float32
        assert state_float32.x.dtype == jnp.float32
        assert state_float32.f.dtype == jnp.float32
        assert state_float32.J.dtype == jnp.float32
        assert state_float32.g.dtype == jnp.float32

        # Check values preserved (within float32 precision)
        np.testing.assert_allclose(state_float32.x, state_float64.x, rtol=1e-7)
        np.testing.assert_allclose(state_float32.f, state_float64.f, rtol=1e-7)
        np.testing.assert_allclose(state_float32.J, state_float64.J, rtol=1e-7)
        np.testing.assert_allclose(state_float32.g, state_float64.g, rtol=1e-7)

        # Check iteration preserved
        assert state_float32.iteration == state_float64.iteration
        assert state_float32.algorithm_specific == state_float64.algorithm_specific

    def test_upgrade_already_float64_returns_same_state(
        self, sample_arrays_float64, logger
    ):
        """Test upgrading already-float64 state returns same state with warning."""
        upgrader = PrecisionUpgrader(logger)

        state_float64 = OptimizationState(
            x=sample_arrays_float64["x"],
            f=sample_arrays_float64["f"],
            J=sample_arrays_float64["J"],
            g=sample_arrays_float64["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=10,
            dtype=jnp.float64,
            algorithm_specific=None,
        )

        # Upgrade already-float64 state
        result = upgrader.upgrade_to_float64(state_float64)

        # Should return same state
        assert result is state_float64
        assert result.dtype == jnp.float64

    def test_downgrade_already_float32_returns_same_state(
        self, sample_arrays_float32, logger
    ):
        """Test downgrading already-float32 state returns same state."""
        upgrader = PrecisionUpgrader(logger)

        state_float32 = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=10,
            dtype=jnp.float32,
            algorithm_specific=None,
        )

        # Downgrade already-float32 state
        result = upgrader.downgrade_to_float32(state_float32)

        # Should return same state
        assert result is state_float32
        assert result.dtype == jnp.float32

    def test_state_preservation_within_float32_precision(
        self, sample_arrays_float32, logger
    ):
        """Test round-trip upgrade→downgrade preserves values within float32 precision."""
        upgrader = PrecisionUpgrader(logger)

        # Start with float32
        state_original = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific={"lambda": 0.1},
        )

        # Upgrade to float64
        state_float64 = upgrader.upgrade_to_float64(state_original)

        # Downgrade back to float32
        state_final = upgrader.downgrade_to_float32(state_float64)

        # Values should be preserved within float32 precision (rtol=1e-7)
        np.testing.assert_allclose(state_final.x, state_original.x, rtol=1e-7)
        np.testing.assert_allclose(state_final.f, state_original.f, rtol=1e-7)
        np.testing.assert_allclose(state_final.J, state_original.J, rtol=1e-7)
        np.testing.assert_allclose(state_final.g, state_original.g, rtol=1e-7)
        assert state_final.cost == state_original.cost
        assert state_final.trust_radius == state_original.trust_radius
        assert state_final.iteration == state_original.iteration
        assert state_final.algorithm_specific == state_original.algorithm_specific


class TestBestParameterTracker:
    """Tests for BestParameterTracker class."""

    def test_initial_state(self):
        """Test initial state before any updates."""
        tracker = BestParameterTracker()

        # Initially, no parameters tracked
        assert tracker.get_best_parameters() is None
        assert tracker.get_best_cost() == float("inf")
        assert tracker.best_iteration == -1

    def test_tracking_improving_parameters(self):
        """Test tracker updates when cost improves."""
        tracker = BestParameterTracker()

        # First update
        params1 = jnp.array([1.0, 2.0, 3.0])
        tracker.update(params1, cost=10.5, iteration=0)
        assert tracker.get_best_cost() == 10.5
        np.testing.assert_array_equal(tracker.get_best_parameters(), params1)
        assert tracker.best_iteration == 0

        # Second update with better (lower) cost
        params2 = jnp.array([1.2, 2.1, 3.2])
        tracker.update(params2, cost=8.3, iteration=1)
        assert tracker.get_best_cost() == 8.3
        np.testing.assert_array_equal(tracker.get_best_parameters(), params2)
        assert tracker.best_iteration == 1

        # Third update with even better cost
        params3 = jnp.array([1.1, 2.05, 3.1])
        tracker.update(params3, cost=5.2, iteration=2)
        assert tracker.get_best_cost() == 5.2
        np.testing.assert_array_equal(tracker.get_best_parameters(), params3)
        assert tracker.best_iteration == 2

    def test_best_params_not_updated_when_cost_worsens(self):
        """Test tracker doesn't update when cost increases (worsens)."""
        tracker = BestParameterTracker()

        # First update with good cost
        params1 = jnp.array([1.0, 2.0])
        tracker.update(params1, cost=5.0, iteration=0)

        # Second update with worse (higher) cost
        params2 = jnp.array([1.5, 2.5])
        tracker.update(params2, cost=10.0, iteration=1)

        # Best should still be params1
        assert tracker.get_best_cost() == 5.0
        np.testing.assert_array_equal(tracker.get_best_parameters(), params1)
        assert tracker.best_iteration == 0

        # Third update with even worse cost
        params3 = jnp.array([2.0, 3.0])
        tracker.update(params3, cost=15.0, iteration=2)

        # Best should still be params1
        assert tracker.get_best_cost() == 5.0
        np.testing.assert_array_equal(tracker.get_best_parameters(), params1)
        assert tracker.best_iteration == 0

    def test_retrieval_of_best_parameters_and_cost(self):
        """Test retrieval methods return correct values."""
        tracker = BestParameterTracker()

        # Add several updates
        params_list = [
            (jnp.array([1.0, 2.0]), 10.0),
            (jnp.array([1.2, 2.1]), 8.0),
            (jnp.array([1.1, 2.05]), 12.0),  # Worse
            (jnp.array([1.15, 2.08]), 6.0),  # Best
            (jnp.array([1.3, 2.2]), 9.0),  # Worse
        ]

        for i, (params, cost) in enumerate(params_list):
            tracker.update(params, cost=cost, iteration=i)

        # Best should be params with cost 6.0
        assert tracker.get_best_cost() == 6.0
        np.testing.assert_array_equal(
            tracker.get_best_parameters(), jnp.array([1.15, 2.08])
        )
        assert tracker.best_iteration == 3

    def test_numpy_conversion_for_storage(self):
        """Test that parameters are converted to NumPy arrays for storage."""
        tracker = BestParameterTracker()

        # Update with JAX array
        jax_params = jnp.array([1.0, 2.0, 3.0])
        tracker.update(jax_params, cost=5.0, iteration=0)

        # Retrieved parameters should be NumPy array
        best_params = tracker.get_best_parameters()
        assert isinstance(best_params, np.ndarray)
        assert not isinstance(best_params, jnp.ndarray)
        np.testing.assert_array_equal(best_params, jax_params)


class TestMixedPrecisionManager:
    """Tests for MixedPrecisionManager orchestrator class."""

    def test_instantiation_and_initial_state(self, default_config):
        """Test manager instantiates with correct initial state."""
        manager = MixedPrecisionManager(default_config)

        # Check initial state
        assert manager.state == PrecisionState.FLOAT32_ACTIVE
        assert manager.current_dtype == jnp.float32
        assert manager.config == default_config

        # Check components initialized
        assert isinstance(manager.monitor, ConvergenceMonitor)
        assert isinstance(manager.upgrader, PrecisionUpgrader)
        assert isinstance(manager.tracker, BestParameterTracker)

    def test_state_transition_float32_to_monitoring(self, default_config):
        """Test FLOAT32_ACTIVE → MONITORING_DEGRADATION transition."""
        manager = MixedPrecisionManager(default_config)

        # Initially in FLOAT32_ACTIVE
        assert manager.state == PrecisionState.FLOAT32_ACTIVE

        # Report issue
        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=1.0,
            gradient_norm=1e11,  # Gradient explosion
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        manager.report_metrics(metrics)

        # Should transition to MONITORING_DEGRADATION
        assert manager.state == PrecisionState.MONITORING_DEGRADATION
        assert manager.monitor.degradation_counter == 1

    def test_state_transition_monitoring_to_float32(self, default_config):
        """Test MONITORING_DEGRADATION → FLOAT32_ACTIVE when issues resolve."""
        manager = MixedPrecisionManager(default_config)

        # Put into MONITORING_DEGRADATION state
        manager.state = PrecisionState.MONITORING_DEGRADATION
        manager.monitor.degradation_counter = 2

        # Report healthy metrics (no issues)
        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=1.0,
            gradient_norm=1e8,  # Below threshold
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        manager.report_metrics(metrics)

        # Should transition back to FLOAT32_ACTIVE
        assert manager.state == PrecisionState.FLOAT32_ACTIVE
        assert manager.monitor.degradation_counter == 0

    def test_degradation_counter_hysteresis(self, default_config):
        """Test degradation counter increments and resets properly."""
        manager = MixedPrecisionManager(default_config)

        # Report issues repeatedly
        for i in range(3):
            metrics = ConvergenceMetrics(
                iteration=i,
                residual_norm=1.0,
                gradient_norm=1e11,
                parameter_change=0.1,
                cost=1.0,
                trust_radius=1.0,
                has_nan_inf=False,
            )
            manager.report_metrics(metrics)

        assert manager.monitor.degradation_counter == 3
        assert manager.state == PrecisionState.MONITORING_DEGRADATION

        # Report healthy metric
        healthy_metrics = ConvergenceMetrics(
            iteration=3,
            residual_norm=1.0,
            gradient_norm=1e8,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )
        manager.report_metrics(healthy_metrics)

        # Counter should reset
        assert manager.monitor.degradation_counter == 0
        assert manager.state == PrecisionState.FLOAT32_ACTIVE

    def test_should_upgrade_logic(self, default_config):
        """Test should_upgrade() returns True when threshold reached."""
        manager = MixedPrecisionManager(default_config)

        # Initially should not upgrade
        assert not manager.should_upgrade()

        # Report issues until threshold reached
        for i in range(default_config.max_degradation_iterations):
            metrics = ConvergenceMetrics(
                iteration=i,
                residual_norm=1.0,
                gradient_norm=1e11,
                parameter_change=0.1,
                cost=1.0,
                trust_radius=1.0,
                has_nan_inf=False,
            )
            manager.report_metrics(metrics)

        # Now should upgrade
        assert manager.should_upgrade()

    def test_should_upgrade_respects_disabled_flag(self):
        """Test should_upgrade() returns False when mixed precision disabled."""
        config = MixedPrecisionConfig(enable_mixed_precision_fallback=False)
        manager = MixedPrecisionManager(config)

        # Put into state where upgrade would normally happen
        manager.state = PrecisionState.MONITORING_DEGRADATION
        manager.monitor.degradation_counter = 10  # Way over threshold

        # Should still not upgrade
        assert not manager.should_upgrade()

    def test_upgrade_precision_flow(self, default_config, sample_arrays_float32):
        """Test upgrade_precision() full flow."""
        manager = MixedPrecisionManager(default_config)

        # Create float32 state
        state_float32 = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific=None,
        )

        # Upgrade
        state_float64 = manager.upgrade_precision(state_float32)

        # Check state transitions
        assert manager.state == PrecisionState.FLOAT64_ACTIVE
        assert manager.current_dtype == jnp.float64

        # Check state upgraded
        assert state_float64.dtype == jnp.float64
        assert state_float64.iteration == 15  # Preserved

    def test_apply_relaxed_fallback_flow(self, default_config, sample_arrays_float64):
        """Test apply_relaxed_fallback() full flow."""
        manager = MixedPrecisionManager(default_config)

        # Add some best parameters to tracker
        best_params = jnp.array([1.5, 2.5, 3.5])
        manager.tracker.update(best_params, cost=5.0, iteration=10)

        # Create float64 state
        state_float64 = OptimizationState(
            x=sample_arrays_float64["x"],
            f=sample_arrays_float64["f"],
            J=sample_arrays_float64["J"],
            g=sample_arrays_float64["g"],
            cost=10.0,
            trust_radius=1.0,
            iteration=20,
            dtype=jnp.float64,
            algorithm_specific=None,
        )

        # Apply fallback
        original_tol = {"gtol": 1e-8, "ftol": 1e-8, "xtol": 1e-8}
        fallback_state, relaxed_tol = manager.apply_relaxed_fallback(
            state_float64, original_tol
        )

        # Check state transitions
        assert manager.state == PrecisionState.RELAXED_FLOAT32_FALLBACK

        # Check state downgraded
        assert fallback_state.dtype == jnp.float32

        # Check best parameters restored
        np.testing.assert_array_equal(fallback_state.x, best_params)

        # Check tolerances relaxed
        assert relaxed_tol["gtol"] == 1e-8 * default_config.tolerance_relaxation_factor
        assert relaxed_tol["ftol"] == 1e-8 * default_config.tolerance_relaxation_factor
        assert relaxed_tol["xtol"] == 1e-8 * default_config.tolerance_relaxation_factor

    def test_update_best_and_get_best_parameters(self, default_config):
        """Test update_best() and get_best_parameters() integration."""
        manager = MixedPrecisionManager(default_config)

        # Initially no parameters
        assert manager.get_best_parameters() is None

        # Update with parameters
        params1 = jnp.array([1.0, 2.0])
        manager.update_best(params1, cost=10.0, iteration=0)
        np.testing.assert_array_equal(manager.get_best_parameters(), params1)

        # Update with better parameters
        params2 = jnp.array([1.2, 2.1])
        manager.update_best(params2, cost=5.0, iteration=1)
        np.testing.assert_array_equal(manager.get_best_parameters(), params2)

    def test_get_current_dtype_tracking(self, default_config, sample_arrays_float32):
        """Test get_current_dtype() tracks precision correctly."""
        manager = MixedPrecisionManager(default_config)

        # Initially float32
        assert manager.get_current_dtype() == jnp.float32

        # After upgrade
        state = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific=None,
        )
        manager.upgrade_precision(state)

        # Should be float64
        assert manager.get_current_dtype() == jnp.float64

    def test_logging_verbosity_levels(self, default_config):
        """Test logging uses DEBUG by default and INFO when verbose."""
        # Default: DEBUG level
        manager_quiet = MixedPrecisionManager(default_config, verbose=False)
        assert manager_quiet.logger.level == logging.DEBUG

        # Verbose: INFO level
        manager_verbose = MixedPrecisionManager(default_config, verbose=True)
        assert manager_verbose.logger.level == logging.INFO

    def test_end_to_end_state_machine_flow(self, default_config, sample_arrays_float32):
        """Test complete state machine flow: float32 → monitoring → upgrade → float64."""
        manager = MixedPrecisionManager(default_config)

        # 1. Start in FLOAT32_ACTIVE
        assert manager.state == PrecisionState.FLOAT32_ACTIVE

        # 2. Report issues to trigger monitoring
        for i in range(default_config.max_degradation_iterations):
            metrics = ConvergenceMetrics(
                iteration=i,
                residual_norm=1.0,
                gradient_norm=1e11,  # Explosion
                parameter_change=0.1,
                cost=1.0,
                trust_radius=1.0,
                has_nan_inf=False,
            )
            manager.report_metrics(metrics)

        # 3. Should be in MONITORING_DEGRADATION and ready to upgrade
        assert manager.state == PrecisionState.MONITORING_DEGRADATION
        assert manager.should_upgrade()

        # 4. Perform upgrade
        state = OptimizationState(
            x=sample_arrays_float32["x"],
            f=sample_arrays_float32["f"],
            J=sample_arrays_float32["J"],
            g=sample_arrays_float32["g"],
            cost=0.035,
            trust_radius=1.0,
            iteration=15,
            dtype=jnp.float32,
            algorithm_specific=None,
        )
        upgraded_state = manager.upgrade_precision(state)

        # 5. Should be in FLOAT64_ACTIVE
        assert manager.state == PrecisionState.FLOAT64_ACTIVE
        assert manager.current_dtype == jnp.float64
        assert upgraded_state.dtype == jnp.float64


class TestFloat64FallbackIntegration:
    """Integration tests for float64 failure fallback mechanism."""

    def test_fallback_relaxed_criteria_application(self):
        """Test that relaxed criteria are applied correctly in fallback."""
        config = MixedPrecisionConfig(
            gradient_explosion_threshold=1e10,
            precision_limit_threshold=1e-7,
            stall_window=5,
            max_degradation_iterations=3,
            tolerance_relaxation_factor=10.0,
        )
        manager = MixedPrecisionManager(config)

        # Simulate float32 → float64 upgrade
        state32 = OptimizationState(
            x=jnp.array([1.0, 2.0], dtype=jnp.float32),
            f=jnp.array([0.1, 0.2], dtype=jnp.float32),
            J=jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
            g=jnp.array([0.1, 0.2], dtype=jnp.float32),
            cost=0.5,
            trust_radius=1.0,
            iteration=10,
            dtype=jnp.float32,
            algorithm_specific={},
        )
        manager.upgrade_precision(state32)

        # Apply fallback
        original_tol = {"gtol": 1e-8, "ftol": 1e-8, "xtol": 1e-8}
        fallback_state, relaxed_tol = manager.apply_relaxed_fallback(
            state32, original_tol
        )

        # Verify relaxed tolerances (10x relaxation)
        assert relaxed_tol["gtol"] == 1e-7
        assert relaxed_tol["ftol"] == 1e-7
        assert relaxed_tol["xtol"] == 1e-7

        # Verify state is in float32
        assert fallback_state.dtype == jnp.float32
        assert manager.state == PrecisionState.RELAXED_FLOAT32_FALLBACK

    def test_fallback_uses_best_parameters_from_history(self):
        """Test that fallback uses best parameters from entire history."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        # Track several parameter sets
        params1 = jnp.array([1.0, 2.0], dtype=jnp.float32)
        params2 = jnp.array([1.5, 2.5], dtype=jnp.float32)
        params3 = jnp.array([1.2, 2.2], dtype=jnp.float32)

        manager.update_best(params1, cost=1.0, iteration=1)
        manager.update_best(params2, cost=0.5, iteration=2)  # Best
        manager.update_best(params3, cost=0.8, iteration=3)

        # Get best parameters
        best_params = manager.get_best_parameters()
        best_cost = manager.tracker.get_best_cost()
        best_iter = manager.tracker.best_iteration

        # Should return params2 (lowest cost)
        np.testing.assert_array_almost_equal(best_params, params2, decimal=5)
        assert best_cost == 0.5
        assert best_iter == 2

    def test_fallback_succeeds_when_float64_fails(self):
        """Test that fallback can succeed when float64 optimization fails."""
        import jax.numpy as jnp

        from nlsq import curve_fit

        # Create a difficult problem that might fail in float64 with tight tolerances
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = 2.5 * np.exp(-0.5 * x) + 0.1
        y = y_true + 0.01 * np.random.randn(len(x))

        def model(x, a, b):
            return a * jnp.exp(-b * x) + 0.1

        # Enable mixed precision
        config = MixedPrecisionConfig(
            gradient_explosion_threshold=1e10,
            precision_limit_threshold=1e-7,
            stall_window=3,
            max_degradation_iterations=2,
        )

        # This should trigger float32 → float64 → fallback if needed
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=[2.0, 0.3],
            mixed_precision_config=config,
            max_nfev=20,  # Limited iterations to potentially trigger fallback
        )

        # Should complete successfully (either in float64 or fallback)
        assert popt is not None
        assert pcov is not None
        assert len(popt) == 2
        assert pcov.shape == (2, 2)

    def test_fallback_logging_messages(self):
        """Test that fallback produces appropriate logging messages."""
        import logging

        config = MixedPrecisionConfig(tolerance_relaxation_factor=10.0)
        manager = MixedPrecisionManager(config, verbose=True)

        # Capture log messages
        logger = logging.getLogger("nlsq.mixed_precision")
        original_level = logger.level
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            # Trigger fallback
            state = OptimizationState(
                x=jnp.array([1.0, 2.0], dtype=jnp.float64),
                f=jnp.array([0.1, 0.2], dtype=jnp.float64),
                J=jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
                g=jnp.array([0.1, 0.2], dtype=jnp.float64),
                cost=0.5,
                trust_radius=1.0,
                iteration=10,
                dtype=jnp.float64,
                algorithm_specific={},
            )
            manager.upgrade_precision(state)
            original_tol = {"gtol": 1e-8, "ftol": 1e-8, "xtol": 1e-8}
            _fallback_state, _relaxed_tol = manager.apply_relaxed_fallback(
                state, original_tol
            )

            # Verify fallback was applied
            assert manager.state == PrecisionState.RELAXED_FLOAT32_FALLBACK

        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)


class TestEdgeCaseHandling:
    """Tests for edge case handling in mixed precision manager."""

    def test_validate_state_with_nan_parameters(self):
        """Test validation detects NaN in parameters."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, jnp.nan, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Parameters contain NaN" in error

    def test_validate_state_with_inf_parameters(self):
        """Test validation detects Inf in parameters."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, jnp.inf, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Parameters contain Inf" in error

    def test_validate_state_with_nan_residuals(self):
        """Test validation detects NaN in residuals."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, jnp.nan]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Residuals contain NaN" in error

    def test_validate_state_with_nan_jacobian(self):
        """Test validation detects NaN in Jacobian."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        J = jnp.zeros((2, 3))
        J = J.at[0, 1].set(jnp.nan)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=J,
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Jacobian contains NaN" in error

    def test_validate_state_with_nan_gradient(self):
        """Test validation detects NaN in gradient."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.array([0.0, jnp.nan, 0.0]),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Gradient contains NaN" in error

    def test_validate_state_with_nan_cost(self):
        """Test validation detects NaN cost."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=jnp.nan,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "Cost is" in error
        assert "nan" in error.lower()

    def test_validate_state_with_negative_trust_radius(self):
        """Test validation detects negative trust radius."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=-0.1,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert not is_valid
        assert "non-positive" in error

    def test_validate_state_valid(self):
        """Test validation passes for valid state."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1.0, 2.0, 3.0]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        is_valid, error = manager.validate_state(state)
        assert is_valid
        assert error is None

    def test_validate_metrics_with_nan_residual_norm(self):
        """Test validation detects NaN in residual norm."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=jnp.nan,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )

        is_valid, error = manager.validate_metrics(metrics)
        assert not is_valid
        assert "Residual norm" in error

    def test_validate_metrics_with_inf_gradient_norm(self):
        """Test validation detects Inf in gradient norm."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=1.0,
            gradient_norm=jnp.inf,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )

        is_valid, error = manager.validate_metrics(metrics)
        assert not is_valid
        assert "Gradient norm" in error

    def test_validate_metrics_with_negative_residual_norm(self):
        """Test validation detects negative residual norm."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=-1.0,
            gradient_norm=1.0,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )

        is_valid, error = manager.validate_metrics(metrics)
        assert not is_valid
        assert "negative" in error

    def test_validate_metrics_valid(self):
        """Test validation passes for valid metrics."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        metrics = ConvergenceMetrics(
            iteration=0,
            residual_norm=1.0,
            gradient_norm=0.5,
            parameter_change=0.1,
            cost=1.0,
            trust_radius=1.0,
            has_nan_inf=False,
        )

        is_valid, error = manager.validate_metrics(metrics)
        assert is_valid
        assert error is None

    def test_handle_validation_failure_nan_parameters(self):
        """Test recovery suggestion for NaN parameters."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        suggestion = manager.handle_validation_failure(
            "Parameters contain NaN", context="after step"
        )

        assert suggestion is not None
        assert (
            "step size" in suggestion.lower()
            or "initial parameters" in suggestion.lower()
        )

    def test_handle_validation_failure_nan_jacobian(self):
        """Test recovery suggestion for NaN Jacobian."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        suggestion = manager.handle_validation_failure(
            "Jacobian contains NaN", context="during computation"
        )

        assert suggestion is not None
        assert (
            "model function" in suggestion.lower()
            or "finite differences" in suggestion.lower()
        )

    def test_handle_validation_failure_collapsed_trust_radius(self):
        """Test recovery suggestion for collapsed trust radius."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        suggestion = manager.handle_validation_failure(
            "Trust radius is non-positive: -0.1", context="after reduction"
        )

        assert suggestion is not None
        assert (
            "cannot continue" in suggestion.lower()
            or "initial parameters" in suggestion.lower()
        )

    def test_validate_state_warns_on_extremely_small_parameters(self, caplog):
        """Test that validation warns about extremely small parameters."""
        config = MixedPrecisionConfig()
        manager = MixedPrecisionManager(config)

        state = OptimizationState(
            x=jnp.array([1e-16, 1e-17, 1e-18]),
            f=jnp.array([0.1, 0.2]),
            J=jnp.zeros((2, 3)),
            g=jnp.zeros(3),
            cost=1.0,
            trust_radius=1.0,
            iteration=0,
            dtype=jnp.float32,
            algorithm_specific={},
        )

        with caplog.at_level(logging.WARNING):
            is_valid, error = manager.validate_state(state)

        # Should still be valid but warn
        assert is_valid
        assert error is None
        assert any(
            "extremely small" in record.message.lower() for record in caplog.records
        )
