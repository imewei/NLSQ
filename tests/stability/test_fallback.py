"""
Tests for Automatic Fallback Strategies
========================================

Tests the fallback orchestrator and individual strategies for robust optimization.
"""

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.stability.fallback import (
    AddParameterBoundsStrategy,
    AdjustTolerancesStrategy,
    FallbackOrchestrator,
    FallbackResult,
    FallbackStrategy,
    PerturbInitialGuessStrategy,
    RescaleProblemStrategy,
    UseRobustLossStrategy,
)


# Test models
def exponential_decay(x, a, b, c):
    """Exponential decay model."""
    return a * jnp.exp(-b * x) + c


def linear(x, a, b):
    """Linear model."""
    return a * x + b


class TestFallbackStrategy:
    """Test base FallbackStrategy class."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = FallbackStrategy(
            name="test", description="Test strategy", priority=5
        )
        assert strategy.name == "test"
        assert strategy.description == "Test strategy"
        assert strategy.priority == 5
        assert strategy.attempts == 0
        assert strategy.successes == 0

    def test_apply_not_implemented(self):
        """Test that base class raises NotImplementedError."""
        strategy = FallbackStrategy(name="test", description="Test", priority=0)
        with pytest.raises(NotImplementedError):
            strategy.apply({})


class TestPerturbInitialGuessStrategy:
    """Test initial guess perturbation strategy."""

    def test_perturbation_applied(self):
        """Test that p0 is perturbed."""
        strategy = PerturbInitialGuessStrategy(perturbation_scale=0.1)
        p0_original = np.array([1.0, 2.0, 3.0])
        kwargs = {"p0": p0_original.copy()}

        modified = strategy.apply(kwargs)

        # Should be different but close
        assert not np.allclose(modified["p0"], p0_original)
        assert np.allclose(modified["p0"], p0_original, rtol=0.2)

    def test_perturbation_respects_bounds(self):
        """Test that perturbed p0 stays within bounds."""
        strategy = PerturbInitialGuessStrategy(perturbation_scale=0.5)
        p0_original = np.array([1.0, 2.0])
        bounds = (np.array([0.5, 1.0]), np.array([1.5, 3.0]))
        kwargs = {"p0": p0_original, "bounds": bounds}

        modified = strategy.apply(kwargs)

        # Should stay within bounds
        assert np.all(modified["p0"] >= bounds[0])
        assert np.all(modified["p0"] <= bounds[1])

    def test_max_perturbations(self):
        """Test that perturbations are limited."""
        strategy = PerturbInitialGuessStrategy(max_perturbations=2)
        kwargs = {"p0": np.array([1.0])}

        # First two should perturb
        strategy.apply(kwargs)
        strategy.apply(kwargs)

        # Third should not change (max reached)
        assert strategy.perturbation_count == 2


class TestAdjustTolerancesStrategy:
    """Test tolerance adjustment strategy."""

    def test_tolerances_relaxed(self):
        """Test that tolerances are relaxed."""
        strategy = AdjustTolerancesStrategy(relaxation_factor=10.0)
        kwargs = {"ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8}

        modified = strategy.apply(kwargs)

        assert modified["ftol"] == 1e-7
        assert modified["xtol"] == 1e-7
        assert modified["gtol"] == 1e-7

    def test_cumulative_relaxation(self):
        """Test that relaxation is cumulative."""
        strategy = AdjustTolerancesStrategy(relaxation_factor=10.0)
        kwargs = {"ftol": 1e-8}

        modified1 = strategy.apply(kwargs)
        assert modified1["ftol"] == 1e-7

        modified2 = strategy.apply(kwargs)
        assert modified2["ftol"] == 1e-6  # 10x again


class TestAddParameterBoundsStrategy:
    """Test parameter bounds inference strategy."""

    def test_bounds_added_when_missing(self):
        """Test that bounds are inferred when not provided."""
        strategy = AddParameterBoundsStrategy()
        xdata = np.linspace(0, 10, 100)
        ydata = np.sin(xdata)
        kwargs = {"p0": np.array([1.0, 1.0]), "_xdata": xdata, "_ydata": ydata}

        modified = strategy.apply(kwargs)

        assert "bounds" in modified
        assert len(modified["bounds"]) == 2
        assert len(modified["bounds"][0]) == 2  # Lower bounds
        assert len(modified["bounds"][1]) == 2  # Upper bounds

    def test_existing_bounds_preserved(self):
        """Test that existing bounds are not overwritten."""
        strategy = AddParameterBoundsStrategy()
        original_bounds = ([0, 0], [10, 5])
        kwargs = {
            "p0": np.array([1.0, 1.0]),
            "bounds": original_bounds,
            "_xdata": np.linspace(0, 10, 100),
            "_ydata": np.ones(100),
        }

        modified = strategy.apply(kwargs)

        assert modified["bounds"] == original_bounds


class TestUseRobustLossStrategy:
    """Test robust loss function strategy."""

    def test_loss_functions_cycled(self):
        """Test that different loss functions are tried."""
        strategy = UseRobustLossStrategy()
        kwargs = {}

        # First call: soft_l1
        modified1 = strategy.apply(kwargs)
        assert modified1["loss"] == "soft_l1"

        # Second call: huber
        modified2 = strategy.apply(kwargs)
        assert modified2["loss"] == "huber"

        # Third call: cauchy
        modified3 = strategy.apply(kwargs)
        assert modified3["loss"] == "cauchy"


class TestRescaleProblemStrategy:
    """Test problem rescaling strategy."""

    def test_scaling_factors_stored(self):
        """Test that scaling information is stored."""
        strategy = RescaleProblemStrategy()
        xdata = np.linspace(0, 100, 50)
        ydata = np.linspace(10, 20, 50)
        kwargs = {"p0": np.array([1.0]), "_xdata": xdata, "_ydata": ydata}

        modified = strategy.apply(kwargs)

        assert "_x_scale" in modified
        assert "_y_scale" in modified
        assert "_x_offset" in modified
        assert "_y_offset" in modified
        assert modified["_scaled"] is True


class TestFallbackOrchestrator:
    """Test FallbackOrchestrator class."""

    def test_initialization_default_strategies(self):
        """Test initialization with default strategies."""
        orchestrator = FallbackOrchestrator()

        assert len(orchestrator.strategies) > 0
        assert orchestrator.max_attempts == 10
        assert orchestrator.total_attempts == 0

    def test_initialization_custom_strategies(self):
        """Test initialization with custom strategies."""
        custom_strategies = [PerturbInitialGuessStrategy()]
        orchestrator = FallbackOrchestrator(strategies=custom_strategies)

        assert len(orchestrator.strategies) == 1
        assert isinstance(orchestrator.strategies[0], PerturbInitialGuessStrategy)

    def test_strategies_sorted_by_priority(self):
        """Test that strategies are sorted by priority."""
        orchestrator = FallbackOrchestrator()

        priorities = [s.priority for s in orchestrator.strategies]
        assert priorities == sorted(priorities, reverse=True)

    def test_successful_fit_no_fallback(self):
        """Test that successful fit doesn't use fallback."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(50)

        orchestrator = FallbackOrchestrator(verbose=False)
        result = orchestrator.fit_with_fallback(
            exponential_decay,
            x,
            y,
            p0=[2, 0.5, 1],  # Good p0
        )

        assert isinstance(result, FallbackResult)
        assert result.fallback_strategy_used is None
        assert result.fallback_attempts == 1
        assert orchestrator.total_attempts == 1

    def test_fallback_with_bad_p0(self):
        """Test that fallback can handle bad initial guess."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(50)

        orchestrator = FallbackOrchestrator(verbose=False, max_attempts=5)
        result = orchestrator.fit_with_fallback(
            exponential_decay,
            x,
            y,
            p0=[100, 10, 50],  # Very bad p0
        )

        assert isinstance(result, FallbackResult)
        # TRF is robust, so it may succeed without fallback
        # Just verify we got a result
        assert result.x is not None
        assert result.fallback_attempts >= 1

    def test_all_strategies_fail_raises_error(self):
        """Test that RuntimeError is raised when all strategies fail."""
        # Impossible problem: incompatible data
        x = np.linspace(0, 10, 10)
        y = np.array([np.nan] * 10)  # NaN data

        orchestrator = FallbackOrchestrator(verbose=False, max_attempts=3)

        with pytest.raises(RuntimeError, match=r"All .* fallback attempts failed"):
            orchestrator.fit_with_fallback(exponential_decay, x, y, p0=[1, 0.5, 1])

    def test_get_statistics(self):
        """Test statistics collection."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(50)

        orchestrator = FallbackOrchestrator(verbose=False)
        _ = orchestrator.fit_with_fallback(exponential_decay, x, y, p0=[2, 0.5, 1])

        stats = orchestrator.get_statistics()

        assert "total_attempts" in stats
        assert "strategies" in stats
        assert stats["total_attempts"] == 1
        assert len(stats["strategies"]) > 0

    def test_verbose_output(self, capsys):
        """Test verbose output."""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(50)

        orchestrator = FallbackOrchestrator(verbose=True)
        _ = orchestrator.fit_with_fallback(exponential_decay, x, y, p0=[2, 0.5, 1])

        captured = capsys.readouterr()
        assert "Attempt 1" in captured.out
        assert "Success" in captured.out or "✅" in captured.out

    def test_max_attempts_respected(self):
        """Test that max_attempts limit is respected."""
        x = np.linspace(0, 10, 10)
        y = np.ones(10)  # Degenerate problem

        orchestrator = FallbackOrchestrator(verbose=False, max_attempts=3)

        with contextlib.suppress(RuntimeError):
            _ = orchestrator.fit_with_fallback(exponential_decay, x, y, p0=[1, 0.5, 1])

        assert orchestrator.total_attempts <= 3


class TestFallbackResult:
    """Test FallbackResult class."""

    def test_initialization(self):
        """Test FallbackResult initialization."""

        # Mock result object
        class MockResult:
            x = np.array([1, 2, 3])
            cost = 0.5
            success = True

        mock_result = MockResult()
        fb_result = FallbackResult(mock_result, strategy_used="perturb_p0", attempts=3)

        assert fb_result.result is mock_result
        assert fb_result.fallback_strategy_used == "perturb_p0"
        assert fb_result.fallback_attempts == 3

    def test_attribute_delegation(self):
        """Test that attributes are delegated to underlying result."""

        class MockResult:
            x = np.array([1, 2, 3])
            cost = 0.5
            success = True

        mock_result = MockResult()
        fb_result = FallbackResult(mock_result)

        # Should delegate to underlying result
        assert np.array_equal(fb_result.x, mock_result.x)
        assert fb_result.cost == mock_result.cost
        assert fb_result.success == mock_result.success


class TestFallbackIntegration:
    """Integration tests for fallback system."""

    def test_recovery_from_poor_initial_guess(self):
        """Test that fallback can recover from very poor initial guess."""
        np.random.seed(42)
        x = np.linspace(0, 5, 100)
        y = 3.0 * x + 2.0 + 0.5 * np.random.randn(100)

        orchestrator = FallbackOrchestrator(verbose=False)
        result = orchestrator.fit_with_fallback(
            linear,
            x,
            y,
            p0=[1000, -500],  # Terrible p0
        )

        # Should still converge reasonably close
        assert abs(result.x[0] - 3.0) < 0.5
        assert abs(result.x[1] - 2.0) < 0.5

    def test_recovery_from_outliers(self):
        """Test that robust loss strategies handle outliers."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(100)

        # Add some severe outliers
        outlier_indices = [10, 30, 50, 70, 90]
        y[outlier_indices] += 10.0

        orchestrator = FallbackOrchestrator(verbose=False)
        result = orchestrator.fit_with_fallback(exponential_decay, x, y, p0=[2, 0.5, 1])

        # Should still get reasonable fit despite outliers
        assert result.x[0] > 1.0  # a should be positive
        assert result.x[1] > 0.0  # b should be positive
        assert result.x[2] > 0.0  # c should be positive

    @pytest.mark.slow
    def test_statistics_tracking(self):
        """Test that strategy statistics are tracked correctly."""
        np.random.seed(42)
        orchestrator = FallbackOrchestrator(verbose=False)

        # Run multiple fits with varying difficulty
        for i in range(5):
            x = np.linspace(0, 10, 50)
            y = 2.5 * np.exp(-0.5 * x) + 1.0 + 0.1 * np.random.randn(50)

            # Vary initial guess quality
            p0_scale = 1 + i * 0.5
            p0 = [2 * p0_scale, 0.5 / p0_scale, 1 * p0_scale]

            with contextlib.suppress(RuntimeError):
                _ = orchestrator.fit_with_fallback(exponential_decay, x, y, p0=p0)

        stats = orchestrator.get_statistics()
        assert stats["total_attempts"] >= 5  # At least one per fit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
