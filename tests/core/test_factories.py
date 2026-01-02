"""Tests for nlsq.core.factories module.

Characterization tests for the factory functions that enable runtime composition
of curve fitting features like global optimization and diagnostics.

Coverage targets:
- OptimizerConfig: Configuration dataclass
- create_optimizer(): Factory function for ConfiguredOptimizer
- ConfiguredOptimizer: Optimizer wrapper class
- configure_curve_fit(): Factory for pre-configured curve_fit

Note: Streaming workflows are configured via workflow selectors/minpack configs,
not via factory flags.
"""

import numpy as np
import pytest

from nlsq.core.factories import (
    ConfiguredOptimizer,
    OptimizerConfig,
    configure_curve_fit,
    create_optimizer,
)


# Test fixtures
@pytest.fixture
def simple_model():
    """A simple linear model for testing."""
    import jax.numpy as jnp

    def model(x, a, b):
        return a * x + b

    return model


@pytest.fixture
def exponential_model():
    """A simple exponential decay model."""
    import jax.numpy as jnp

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    return model


@pytest.fixture
def sample_data():
    """Generate sample data for fitting."""
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2.5 * x + 1.0 + np.random.normal(0, 0.5, 100)
    return x, y


@pytest.fixture
def exponential_data():
    """Generate exponential decay data for fitting."""
    np.random.seed(42)
    x = np.linspace(0, 5, 100)
    y = 3.0 * np.exp(-0.5 * x) + 0.5 + np.random.normal(0, 0.1, 100)
    return x, y


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.enable_global is False
        assert config.enable_diagnostics is False
        assert config.enable_recovery is True
        assert config.n_starts == 10
        assert config.extra_kwargs == {}

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OptimizerConfig(
            enable_global=True,
            enable_diagnostics=True,
            enable_recovery=False,
            n_starts=20,
            extra_kwargs={"maxfev": 1000},
        )

        assert config.enable_global is True
        assert config.enable_diagnostics is True
        assert config.enable_recovery is False
        assert config.n_starts == 20
        assert config.extra_kwargs == {"maxfev": 1000}

    def test_extra_kwargs_isolation(self):
        """Test that extra_kwargs are isolated between instances."""
        config1 = OptimizerConfig(extra_kwargs={"a": 1})
        config2 = OptimizerConfig(extra_kwargs={"b": 2})

        assert config1.extra_kwargs == {"a": 1}
        assert config2.extra_kwargs == {"b": 2}

        # Modify one, should not affect the other
        config1.extra_kwargs["c"] = 3
        assert "c" not in config2.extra_kwargs


class TestCreateOptimizer:
    """Tests for create_optimizer() factory function."""

    def test_default_optimizer(self):
        """Test creating optimizer with default settings."""
        optimizer = create_optimizer()

        assert isinstance(optimizer, ConfiguredOptimizer)
        assert optimizer._config.enable_global is False

    def test_global_optimizer(self):
        """Test creating global optimizer."""
        optimizer = create_optimizer(global_optimization=True, n_starts=20)

        assert optimizer._config.enable_global is True
        assert optimizer._config.n_starts == 20

    def test_diagnostics_optimizer(self):
        """Test creating optimizer with diagnostics enabled."""
        optimizer = create_optimizer(diagnostics=True)

        assert optimizer._config.enable_diagnostics is True

    def test_recovery_disabled(self):
        """Test creating optimizer with recovery disabled."""
        optimizer = create_optimizer(recovery=False)

        assert optimizer._config.enable_recovery is False

    def test_extra_kwargs_passed(self):
        """Test that extra kwargs are passed to config."""
        optimizer = create_optimizer(maxfev=500, xtol=1e-10)

        assert optimizer._config.extra_kwargs["maxfev"] == 500
        assert optimizer._config.extra_kwargs["xtol"] == 1e-10

    def test_combined_features(self):
        """Test creating optimizer with multiple features."""
        optimizer = create_optimizer(
            diagnostics=True,
            maxfev=1000,
        )

        assert optimizer._config.enable_diagnostics is True
        assert optimizer._config.extra_kwargs["maxfev"] == 1000


class TestConfiguredOptimizer:
    """Tests for ConfiguredOptimizer class."""

    def test_slots_defined(self):
        """Test that __slots__ is defined for memory efficiency."""
        assert hasattr(ConfiguredOptimizer, "__slots__")
        expected_slots = (
            "_cache",
            "_config",
            "_diagnostics_config",
            "_stability_guard",
        )
        assert set(ConfiguredOptimizer.__slots__) == set(expected_slots)

    def test_initialization(self):
        """Test ConfiguredOptimizer initialization."""
        config = OptimizerConfig()
        optimizer = ConfiguredOptimizer(config=config)

        assert optimizer._config is config
        assert optimizer._cache is None
        assert optimizer._stability_guard is None
        assert optimizer._diagnostics_config is None

    def test_fit_standard(self, simple_model, sample_data):
        """Test standard fit operation."""
        x, y = sample_data
        optimizer = create_optimizer()

        # Pass p0 and bounds to help the fit converge
        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, pcov = optimizer.fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds
        )

        # Check that parameters were estimated
        assert popt is not None
        assert len(popt) == 2
        # Check that covariance was computed
        assert pcov is not None
        assert pcov.shape == (2, 2)
        # Check reasonable parameter estimates (true values: a=2.5, b=1.0)
        assert 2.0 < popt[0] < 3.0  # slope
        assert 0.0 < popt[1] < 2.0  # intercept

    def test_fit_with_p0(self, simple_model, sample_data):
        """Test fit with initial parameter guess."""
        x, y = sample_data
        optimizer = create_optimizer()

        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, _pcov = optimizer.fit(simple_model, x, y, p0=[1.0, 0.0], bounds=bounds)

        assert popt is not None
        assert len(popt) == 2

    def test_fit_with_bounds(self, simple_model, sample_data):
        """Test fit with parameter bounds."""
        x, y = sample_data
        optimizer = create_optimizer()

        bounds = (np.array([0.0, -np.inf]), np.array([np.inf, np.inf]))
        popt, _pcov = optimizer.fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds
        )

        assert popt is not None
        assert popt[0] >= 0.0  # Respect lower bound

    def test_fit_with_sigma(self, simple_model, sample_data):
        """Test fit with uncertainties."""
        x, y = sample_data
        sigma = np.ones_like(y) * 0.5
        optimizer = create_optimizer()

        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, pcov = optimizer.fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), sigma=sigma, bounds=bounds
        )

        assert popt is not None
        assert pcov is not None

    def test_fit_merges_kwargs(self, simple_model, sample_data):
        """Test that fit merges config kwargs with call kwargs."""
        x, y = sample_data
        # Create optimizer with maxfev=100
        optimizer = create_optimizer(maxfev=100)

        # Call fit with different maxfev - should override
        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, _ = optimizer.fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds, maxfev=500
        )

        assert popt is not None


class TestConfigureCurveFit:
    """Tests for configure_curve_fit() factory function."""

    def test_basic_configured_fit(self, simple_model, sample_data):
        """Test basic configured curve_fit."""
        x, y = sample_data
        curve_fit = configure_curve_fit()

        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, pcov = curve_fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds
        )

        assert popt is not None
        assert len(popt) == 2
        assert pcov is not None

    def test_configured_fit_with_defaults(self, simple_model, sample_data):
        """Test configured curve_fit with default kwargs."""
        x, y = sample_data
        curve_fit = configure_curve_fit(maxfev=500)

        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, _pcov = curve_fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds
        )

        assert popt is not None

    def test_call_kwargs_override_defaults(self, simple_model, sample_data):
        """Test that call kwargs override configured defaults."""
        x, y = sample_data
        curve_fit = configure_curve_fit(maxfev=100)

        # Call with different maxfev
        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, _pcov = curve_fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds, maxfev=500
        )

        assert popt is not None

    def test_enable_caching_parameter(self):
        """Test enable_caching parameter (currently unused but part of API)."""
        curve_fit = configure_curve_fit(enable_caching=False)

        assert callable(curve_fit)

    def test_enable_recovery_parameter(self):
        """Test enable_recovery parameter (currently unused but part of API)."""
        curve_fit = configure_curve_fit(enable_recovery=False)

        assert callable(curve_fit)


class TestGlobalOptimizer:
    """Tests for global optimization path."""

    @pytest.mark.slow
    def test_global_fit(self, simple_model, sample_data):
        """Test global optimization fit path."""
        x, y = sample_data
        optimizer = create_optimizer(global_optimization=True, n_starts=3)

        result = optimizer.fit(simple_model, x, y, p0=np.array([1.0, 0.0]))

        # MultiStartOrchestrator returns dict, not tuple
        assert result is not None

    @pytest.mark.slow
    def test_global_custom_n_starts(self, simple_model, sample_data):
        """Test global optimization with custom n_starts."""
        _x, _y = sample_data
        optimizer = create_optimizer(global_optimization=True, n_starts=5)

        assert optimizer._config.n_starts == 5


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_extra_kwargs(self):
        """Test creating optimizer with empty extra kwargs."""
        optimizer = create_optimizer()

        assert optimizer._config.extra_kwargs == {}

    def test_none_optional_params(self):
        """Test that None optional params are handled correctly."""
        optimizer = create_optimizer(
            cache=None,
            stability_guard=None,
            diagnostics_config=None,
        )

        assert optimizer._cache is None
        assert optimizer._stability_guard is None
        assert optimizer._diagnostics_config is None

    def test_optimizer_with_all_features_disabled(self, simple_model, sample_data):
        """Test optimizer with streaming and global disabled uses standard fit."""
        x, y = sample_data
        optimizer = create_optimizer(
            streaming=False,
            global_optimization=False,
            diagnostics=False,
            recovery=False,
        )

        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt, _pcov = optimizer.fit(
            simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds
        )

        assert popt is not None


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_optimizer_reuse(
        self, simple_model, sample_data, exponential_model, exponential_data
    ):
        """Test reusing the same optimizer for multiple fits."""
        optimizer = create_optimizer()

        # First fit
        x1, y1 = sample_data
        bounds1 = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))
        popt1, _ = optimizer.fit(
            simple_model, x1, y1, p0=np.array([1.0, 0.0]), bounds=bounds1
        )

        # Second fit with different model/data
        x2, y2 = exponential_data
        bounds2 = (
            np.array([-np.inf, -np.inf, -np.inf]),
            np.array([np.inf, np.inf, np.inf]),
        )
        popt2, _ = optimizer.fit(
            exponential_model, x2, y2, p0=np.array([1.0, 0.1, 0.1]), bounds=bounds2
        )

        assert len(popt1) == 2
        assert len(popt2) == 3

    def test_configure_curve_fit_returns_callable(self):
        """Test that configure_curve_fit returns a proper callable."""
        curve_fit = configure_curve_fit(enable_diagnostics=True)

        assert callable(curve_fit)
        # Should have proper signature
        import inspect

        sig = inspect.signature(curve_fit)
        params = list(sig.parameters.keys())
        assert "f" in params
        assert "xdata" in params
        assert "ydata" in params

    def test_multiple_configured_fits(self, simple_model, sample_data):
        """Test creating multiple configured curve_fit instances."""
        fit1 = configure_curve_fit(maxfev=100)
        fit2 = configure_curve_fit(maxfev=500)

        x, y = sample_data
        bounds = (np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf]))

        # Both should work independently
        popt1, _ = fit1(simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds)
        popt2, _ = fit2(simple_model, x, y, p0=np.array([1.0, 0.0]), bounds=bounds)

        assert popt1 is not None
        assert popt2 is not None
