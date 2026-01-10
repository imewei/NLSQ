"""Tests for BIPOPRestarter class.

Tests cover:
- Population sizing alternation (large/small)
- Restart condition detection
- Budget tracking
- State management
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization.cmaes_config import is_evosax_available

# Skip all tests if evosax is not available
pytestmark = pytest.mark.skipif(
    not is_evosax_available(),
    reason="evosax not installed - skipping BIPOP tests",
)


class TestBIPOPRestarterBasic:
    """Tests for basic BIPOPRestarter functionality."""

    def test_import(self) -> None:
        """Test that BIPOPRestarter can be imported."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        assert BIPOPRestarter is not None

    def test_instantiation(self) -> None:
        """Test basic instantiation."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)
        assert restarter.base_popsize == 8
        assert restarter.n_params == 3
        assert restarter.max_restarts == 9

    def test_instantiation_with_config(self) -> None:
        """Test instantiation with custom max_restarts."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=16, n_params=5, max_restarts=5)
        assert restarter.max_restarts == 5


class TestBIPOPRestarterPopulationSizing:
    """Tests for BIPOP population sizing logic."""

    def test_initial_population_is_large(self) -> None:
        """Test that first run uses large population."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)
        popsize = restarter.get_next_popsize()

        # First run should be large (doubled base)
        assert popsize == 16  # 8 * 2

    def test_alternating_population_sizes(self) -> None:
        """Test that population sizes alternate between large and small."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3, max_restarts=5)

        # First: large
        pop1 = restarter.get_next_popsize()
        restarter.register_restart()

        # Second: small (random between base/2 and base)
        pop2 = restarter.get_next_popsize()
        restarter.register_restart()

        # Third: large again
        pop3 = restarter.get_next_popsize()

        assert pop1 == 16  # Large
        assert 4 <= pop2 <= 8  # Small (base/2 to base)
        assert pop3 == 16  # Large again

    def test_small_population_bounds(self) -> None:
        """Test that small population respects minimum size."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        # Use small base to test minimum
        restarter = BIPOPRestarter(base_popsize=4, n_params=2, max_restarts=3)

        # First large run
        restarter.get_next_popsize()
        restarter.register_restart()

        # Small run should still be at least 4
        pop_small = restarter.get_next_popsize()
        assert pop_small >= 4


class TestBIPOPRestarterRestartConditions:
    """Tests for restart condition detection."""

    def test_should_restart_on_stagnation(self) -> None:
        """Test that should_restart returns True when conditions are met."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)

        # Simulate stagnation with very similar fitness values
        fitness = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Create mock state with small spread (stagnation)
        # The actual check uses evosax conditions
        should_restart = restarter.check_stagnation(fitness_spread=1e-15)
        assert should_restart is True

    def test_should_not_restart_when_improving(self) -> None:
        """Test that should_restart returns False when fitness is improving."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)

        # Simulate diverse fitness (still improving)
        should_restart = restarter.check_stagnation(fitness_spread=1.0)
        assert should_restart is False


class TestBIPOPRestarterBudget:
    """Tests for evaluation budget tracking."""

    def test_exhausted_restarts(self) -> None:
        """Test that restarter reports exhausted correctly."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3, max_restarts=2)

        assert not restarter.exhausted

        # Exhaust restarts
        restarter.register_restart()
        restarter.register_restart()

        assert restarter.exhausted

    def test_restart_counter(self) -> None:
        """Test that restart counter increments correctly."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)

        assert restarter.restart_count == 0

        restarter.register_restart()
        assert restarter.restart_count == 1

        restarter.register_restart()
        assert restarter.restart_count == 2


class TestBIPOPRestarterBestSolution:
    """Tests for best solution tracking across restarts."""

    def test_update_best_solution(self) -> None:
        """Test that best solution is tracked across restarts."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=3)

        # First solution
        solution1 = jnp.array([1.0, 2.0, 3.0])
        fitness1 = -10.0  # Lower fitness (CMA-ES maximizes, so more negative is worse)

        restarter.update_best(solution1, fitness1)
        assert np.allclose(restarter.best_solution, solution1)
        assert restarter.best_fitness == fitness1

        # Better solution
        solution2 = jnp.array([1.5, 2.5, 3.5])
        fitness2 = -5.0  # Better fitness

        restarter.update_best(solution2, fitness2)
        assert np.allclose(restarter.best_solution, solution2)
        assert restarter.best_fitness == fitness2

        # Worse solution (should not update)
        solution3 = jnp.array([0.0, 0.0, 0.0])
        fitness3 = -20.0  # Worse fitness

        restarter.update_best(solution3, fitness3)
        assert np.allclose(restarter.best_solution, solution2)  # Still solution2
        assert restarter.best_fitness == fitness2

    def test_get_best_after_restarts(self) -> None:
        """Test that get_best returns the global best after multiple restarts."""
        from nlsq.global_optimization.bipop import BIPOPRestarter

        restarter = BIPOPRestarter(base_popsize=8, n_params=2, max_restarts=3)

        # Simulate multiple restarts with different best solutions
        solutions = [
            (jnp.array([1.0, 1.0]), -15.0),
            (jnp.array([2.0, 2.0]), -8.0),  # Best
            (jnp.array([3.0, 3.0]), -12.0),
        ]

        for sol, fit in solutions:
            restarter.update_best(sol, fit)
            restarter.register_restart()

        best_sol, best_fit = restarter.get_best()
        assert np.allclose(best_sol, [2.0, 2.0])
        assert best_fit == -8.0
