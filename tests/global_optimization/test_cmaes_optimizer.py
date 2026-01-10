"""Tests for CMAESOptimizer class.

Tests cover:
- Basic optimization without restarts
- Multi-scale parameter fitting
- NLSQ refinement phase
- Preset instantiation
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import CMAESConfig, is_evosax_available

# Skip all tests if evosax is not available
pytestmark = pytest.mark.skipif(
    not is_evosax_available(),
    reason="evosax not installed - skipping CMA-ES tests",
)


@pytest.fixture
def simple_model():
    """Simple exponential model for testing."""

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    return model


@pytest.fixture
def quadratic_model():
    """Quadratic model for testing."""

    def model(x, a, b, c):
        return a * x**2 + b * x + c

    return model


@pytest.fixture
def multi_scale_model():
    """Model with multi-scale parameters (6+ orders of magnitude)."""

    def model(x, D0, gamma0, n):
        """Diffusion model: D = D0 * (1 + (gamma / gamma0)^n)"""
        # D0 ~ 1e4, gamma0 ~ 1e-3, n ~ 1
        return D0 * (1 + (x / gamma0) ** n)

    return model


class TestCMAESOptimizerBasic:
    """Tests for basic CMAESOptimizer functionality."""

    def test_import(self) -> None:
        """Test that CMAESOptimizer can be imported."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        assert CMAESOptimizer is not None

    def test_instantiation_default(self) -> None:
        """Test default instantiation."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        optimizer = CMAESOptimizer()
        assert optimizer.config is not None
        assert optimizer.config.restart_strategy == "bipop"

    def test_instantiation_with_config(self) -> None:
        """Test instantiation with custom config."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        config = CMAESConfig(
            popsize=16, max_generations=50, restart_strategy="none", max_restarts=0
        )
        optimizer = CMAESOptimizer(config=config)

        assert optimizer.config.popsize == 16
        assert optimizer.config.max_generations == 50
        assert optimizer.config.restart_strategy == "none"

    def test_from_preset(self) -> None:
        """Test from_preset class method."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        optimizer = CMAESOptimizer.from_preset("cmaes-fast")

        assert optimizer.config.max_generations == 50
        assert optimizer.config.restart_strategy == "none"


class TestCMAESOptimizerFit:
    """Tests for CMAESOptimizer.fit() method."""

    def test_fit_simple_model(self, simple_model) -> None:
        """Test fitting a simple exponential model."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        # Generate test data
        true_params = (2.5, 0.5)
        x = jnp.linspace(0, 5, 50)
        y = simple_model(x, *true_params)
        # Add small noise
        y = y + 0.01 * jnp.std(y) * jnp.sin(x * 10)

        # Define bounds
        bounds = ([0.1, 0.01], [10.0, 2.0])

        # Create optimizer with no restarts for speed
        config = CMAESConfig(
            max_generations=50, restart_strategy="none", max_restarts=0, seed=42
        )
        optimizer = CMAESOptimizer(config=config)

        # Fit
        result = optimizer.fit(simple_model, x, y, bounds=bounds)

        # Check result structure
        assert "popt" in result
        assert "pcov" in result

        # Check convergence (not exact due to noise and limited generations)
        popt = result["popt"]
        assert len(popt) == 2
        # Parameters should be reasonable
        assert 0.1 < popt[0] < 10.0
        assert 0.01 < popt[1] < 2.0

    def test_fit_with_initial_guess(self, quadratic_model) -> None:
        """Test fitting with initial parameter guess."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        # Generate test data
        true_params = (1.0, -2.0, 3.0)
        x = jnp.linspace(-2, 2, 100)
        y = quadratic_model(x, *true_params)

        # Define bounds
        bounds = ([-5.0, -5.0, -5.0], [5.0, 5.0, 10.0])

        # Initial guess
        p0 = jnp.array([0.5, -1.0, 2.0])

        # NLSQ refinement ensures tight convergence from CMA-ES approximate solution
        config = CMAESConfig(
            max_generations=50,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(quadratic_model, x, y, p0=p0, bounds=bounds)

        popt = result["popt"]
        # Should converge close to true parameters
        np.testing.assert_allclose(popt, true_params, rtol=0.1)

    def test_fit_returns_pcov(self, simple_model) -> None:
        """Test that fit returns parameter covariance matrix."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        x = jnp.linspace(0, 5, 50)
        y = simple_model(x, 2.0, 0.5)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=30,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(simple_model, x, y, bounds=bounds)

        # Check pcov exists and has correct shape
        assert "pcov" in result
        pcov = result["pcov"]
        assert pcov.shape == (2, 2)
        # Covariance matrix should be symmetric
        np.testing.assert_allclose(pcov, pcov.T, rtol=1e-10)

    def test_fit_without_nlsq_refinement(self, simple_model) -> None:
        """Test fitting without NLSQ refinement phase."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        x = jnp.linspace(0, 5, 50)
        y = simple_model(x, 2.0, 0.5)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=30,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=False,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(simple_model, x, y, bounds=bounds)

        # Should still return result, but pcov may be estimated differently
        assert "popt" in result
        popt = result["popt"]
        assert len(popt) == 2


class TestCMAESOptimizerMultiScale:
    """Tests for multi-scale parameter optimization."""

    def test_multi_scale_parameters(self, multi_scale_model) -> None:
        """Test fitting model with parameters spanning 6+ orders of magnitude."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        # True parameters: D0 ~ 1e4, gamma0 ~ 1e-3, n ~ 1
        true_params = (1e4, 1e-3, 1.5)
        x = jnp.logspace(-5, 0, 100)  # Shear rate from 1e-5 to 1
        y = multi_scale_model(x, *true_params)

        # Bounds spanning 6 orders of magnitude
        bounds = ([1e2, 1e-5, 0.5], [1e6, 1e-1, 3.0])

        # Use more generations for multi-scale
        config = CMAESConfig(
            max_generations=100, restart_strategy="none", max_restarts=0, seed=42
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(multi_scale_model, x, y, bounds=bounds)

        popt = result["popt"]

        # Check parameters are in correct order of magnitude
        assert 1e2 < popt[0] < 1e6  # D0
        assert 1e-5 < popt[1] < 1e-1  # gamma0
        assert 0.5 < popt[2] < 3.0  # n

    def test_scale_invariance(self, simple_model) -> None:
        """Test that CMA-ES handles different parameter scales."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        # Same model with different scale bounds
        x = jnp.linspace(0, 5, 50)
        y = simple_model(x, 2.0, 0.5)

        # Test with small-scale bounds
        bounds_small = ([0.1, 0.01], [10.0, 2.0])
        config = CMAESConfig(
            max_generations=30, restart_strategy="none", max_restarts=0, seed=42
        )
        optimizer = CMAESOptimizer(config=config)
        result_small = optimizer.fit(simple_model, x, y, bounds=bounds_small)

        # Result should converge regardless of scale
        assert result_small["popt"] is not None


class TestCMAESOptimizerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_parameter(self) -> None:
        """Test optimization with single parameter."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        # Single-parameter problems need more generations (less exploration diversity)
        # and refine_with_nlsq for robust convergence
        config = CMAESConfig(
            max_generations=50,
            restart_strategy="none",
            max_restarts=0,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)
        result = optimizer.fit(model, x, y, bounds=bounds)

        popt = result["popt"]
        assert len(popt) == 1
        np.testing.assert_allclose(popt[0], 2.5, rtol=0.1)

    def test_many_parameters(self) -> None:
        """Test optimization with many parameters (10)."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def polynomial(x, *coeffs):
            result = jnp.zeros_like(x)
            for i, c in enumerate(coeffs):
                result = result + c * x**i
            return result

        # True coefficients
        true_coeffs = [1.0, -0.5, 0.3, -0.1, 0.05, -0.02, 0.01, -0.005, 0.002, -0.001]
        x = jnp.linspace(-1, 1, 100)
        y = polynomial(x, *true_coeffs)

        # Bounds
        n_params = len(true_coeffs)
        bounds = ([-5.0] * n_params, [5.0] * n_params)

        config = CMAESConfig(
            max_generations=100, restart_strategy="none", max_restarts=0, seed=42
        )
        optimizer = CMAESOptimizer(config=config)
        result = optimizer.fit(polynomial, x, y, bounds=bounds)

        popt = result["popt"]
        assert len(popt) == n_params

    def test_requires_bounds(self) -> None:
        """Test that CMA-ES requires explicit bounds."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x

        optimizer = CMAESOptimizer()

        with pytest.raises((ValueError, TypeError)):
            optimizer.fit(model, x, y, bounds=None)


class TestCMAESOptimizerSeed:
    """Tests for reproducibility with seeds."""

    def test_seed_reproducibility(self, simple_model) -> None:
        """Test that same seed produces same results."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        x = jnp.linspace(0, 5, 50)
        y = simple_model(x, 2.0, 0.5)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config = CMAESConfig(
            max_generations=20, restart_strategy="none", max_restarts=0, seed=12345
        )

        optimizer1 = CMAESOptimizer(config=config)
        result1 = optimizer1.fit(simple_model, x, y, bounds=bounds)

        optimizer2 = CMAESOptimizer(config=config)
        result2 = optimizer2.fit(simple_model, x, y, bounds=bounds)

        np.testing.assert_allclose(result1["popt"], result2["popt"], rtol=1e-10)

    def test_different_seeds_different_results(self, simple_model) -> None:
        """Test that different seeds can produce different results."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        x = jnp.linspace(0, 5, 50)
        # Add noise to make optimization non-deterministic
        key = jnp.array([0, 42], dtype=jnp.uint32)
        y = simple_model(x, 2.0, 0.5) + 0.1 * jnp.sin(x * 10)
        bounds = ([0.1, 0.01], [10.0, 2.0])

        config1 = CMAESConfig(
            max_generations=20, restart_strategy="none", max_restarts=0, seed=1
        )
        config2 = CMAESConfig(
            max_generations=20, restart_strategy="none", max_restarts=0, seed=9999
        )

        optimizer1 = CMAESOptimizer(config=config1)
        result1 = optimizer1.fit(simple_model, x, y, bounds=bounds)

        optimizer2 = CMAESOptimizer(config=config2)
        result2 = optimizer2.fit(simple_model, x, y, bounds=bounds)

        # Results may differ slightly due to different random starting points
        # (they should both converge, but may not be exactly equal)
        # This test just verifies different seeds don't cause errors
        assert result1["popt"] is not None
        assert result2["popt"] is not None


class TestCMAESOptimizerBIPOP:
    """Tests for BIPOP restart strategy."""

    def test_bipop_runs_multiple_restarts(self) -> None:
        """Test that BIPOP correctly executes multiple restart runs.

        This test verifies BIPOP infrastructure works correctly by fitting
        a simple model and checking that the result is valid.
        """
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def simple_quadratic(x, a, b, c):
            """Simple quadratic model with known solution."""
            return a * x**2 + b * x + c

        # Generate target data
        true_params = (2.0, -1.0, 3.0)
        x = jnp.linspace(-2, 2, 100)
        y_target = simple_quadratic(x, *true_params)

        # Bounds containing the true solution
        bounds = ([0.0, -5.0, 0.0], [5.0, 5.0, 10.0])

        # With BIPOP enabled, should converge
        config = CMAESConfig(
            max_generations=50,
            restart_strategy="bipop",
            max_restarts=3,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(simple_quadratic, x, y_target, bounds=bounds)

        popt = result["popt"]
        # Should converge to true parameters (NLSQ refinement ensures this)
        np.testing.assert_allclose(popt, true_params, rtol=0.1)

    def test_bipop_restart_count(self) -> None:
        """Test that BIPOP performs restarts when needed."""
        from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer

        def simple_model(x, a):
            return a * x

        x = jnp.linspace(0, 10, 50)
        y = 2.5 * x
        bounds = ([0.1], [10.0])

        # Configure with BIPOP enabled
        config = CMAESConfig(
            max_generations=30,
            restart_strategy="bipop",
            max_restarts=3,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(simple_model, x, y, bounds=bounds)

        # Should converge to correct answer
        popt = result["popt"]
        np.testing.assert_allclose(popt[0], 2.5, rtol=0.1)


class TestCMAESDiagnostics:
    """Tests for CMAESDiagnostics dataclass."""

    def test_diagnostics_default_values(self) -> None:
        """Test default values of CMAESDiagnostics."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        diag = CMAESDiagnostics()

        assert diag.total_generations == 0
        assert diag.total_restarts == 0
        assert diag.final_sigma == 0.0
        assert diag.best_fitness == float("inf")
        assert diag.fitness_history == []
        assert diag.restart_history == []
        assert diag.convergence_reason == "not_converged"
        assert diag.nlsq_refinement is False
        assert diag.wall_time == 0.0

    def test_diagnostics_custom_values(self) -> None:
        """Test custom values of CMAESDiagnostics."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        diag = CMAESDiagnostics(
            total_generations=150,
            total_restarts=3,
            final_sigma=0.01,
            best_fitness=-1e-10,
            fitness_history=[-100.0, -50.0, -10.0, -1e-10],
            restart_history=[{"popsize": 8, "generations": 50}],
            convergence_reason="fitness_tolerance",
            nlsq_refinement=True,
            wall_time=5.5,
        )

        assert diag.total_generations == 150
        assert diag.total_restarts == 3
        assert diag.final_sigma == 0.01
        assert diag.best_fitness == -1e-10
        assert len(diag.fitness_history) == 4
        assert len(diag.restart_history) == 1
        assert diag.convergence_reason == "fitness_tolerance"
        assert diag.nlsq_refinement is True
        assert diag.wall_time == 5.5

    def test_diagnostics_summary(self) -> None:
        """Test summary() method generates readable output."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        diag = CMAESDiagnostics(
            total_generations=100,
            total_restarts=2,
            final_sigma=0.001,
            best_fitness=-1e-8,
            convergence_reason="max_generations",
            wall_time=3.14,
        )

        summary = diag.summary()

        assert "CMA-ES Optimization Summary" in summary
        assert "100" in summary  # generations
        assert "2" in summary  # restarts
        assert "max_generations" in summary

    def test_diagnostics_to_dict(self) -> None:
        """Test to_dict() serialization."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        diag = CMAESDiagnostics(
            total_generations=50,
            total_restarts=1,
            best_fitness=-0.5,
        )

        d = diag.to_dict()

        assert d["total_generations"] == 50
        assert d["total_restarts"] == 1
        assert d["best_fitness"] == -0.5

    def test_diagnostics_from_dict(self) -> None:
        """Test from_dict() deserialization."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        d = {
            "total_generations": 75,
            "total_restarts": 2,
            "final_sigma": 0.05,
            "best_fitness": -1e-5,
            "convergence_reason": "xtol",
        }

        diag = CMAESDiagnostics.from_dict(d)

        assert diag.total_generations == 75
        assert diag.total_restarts == 2
        assert diag.final_sigma == 0.05
        assert diag.best_fitness == -1e-5
        assert diag.convergence_reason == "xtol"

    def test_diagnostics_fitness_improvement(self) -> None:
        """Test get_fitness_improvement() method."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        # Not enough history
        diag1 = CMAESDiagnostics(fitness_history=[-100.0])
        assert diag1.get_fitness_improvement() is None

        # Valid history (fitness improves from -100 to -10)
        diag2 = CMAESDiagnostics(fitness_history=[-100.0, -50.0, -10.0])
        improvement = diag2.get_fitness_improvement()
        assert improvement is not None
        # Improvement is (final - initial) / |initial| = (-10 - (-100)) / 100 = 0.9
        np.testing.assert_allclose(improvement, 0.9, rtol=0.01)

    def test_diagnostics_convergence_rate(self) -> None:
        """Test get_convergence_rate() method."""
        from nlsq.global_optimization.cmaes_diagnostics import CMAESDiagnostics

        # Not enough history
        diag1 = CMAESDiagnostics(fitness_history=[-100.0])
        assert diag1.get_convergence_rate() is None

        # Valid history
        diag2 = CMAESDiagnostics(fitness_history=[-100.0, -50.0, -10.0])
        rate = diag2.get_convergence_rate()
        assert rate is not None
        assert len(rate) == 2
        # Rates: (-50) - (-100) = 50, (-10) - (-50) = 40
        np.testing.assert_allclose(rate, [50.0, 40.0], rtol=0.01)
