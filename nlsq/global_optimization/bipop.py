"""BIPOP restart strategy for CMA-ES.

Implements the Bi-Population restart strategy where large and small
population runs alternate to balance exploration and exploitation.

References
----------
Hansen, N. (2009). Benchmarking a BI-Population CMA-ES on the BBOB-2009
Function Testbed. GECCO Workshop on Black-Box Optimization Benchmarking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

__all__ = ["BIPOPRestarter"]

logger = logging.getLogger(__name__)


@dataclass
class BIPOPRestarter:
    """BIPOP restart manager for CMA-ES optimization.

    Manages alternating large/small population restarts following the BIPOP
    strategy. Large populations explore broadly while small populations
    exploit local regions more intensively.

    Parameters
    ----------
    base_popsize : int
        Base population size (typically 4 + floor(3 * ln(n))).
    n_params : int
        Number of parameters being optimized.
    max_restarts : int, optional
        Maximum number of restarts before giving up. Default is 9.
    min_fitness_spread : float, optional
        Minimum fitness spread threshold for stagnation detection.
        Default is 1e-12.

    Attributes
    ----------
    restart_count : int
        Number of restarts performed so far.
    exhausted : bool
        True if max_restarts has been reached.
    best_solution : jax.Array | None
        Best solution found across all restarts.
    best_fitness : float
        Best fitness found across all restarts.

    Examples
    --------
    >>> restarter = BIPOPRestarter(base_popsize=8, n_params=3)
    >>> popsize = restarter.get_next_popsize()
    >>> # ... run CMA-ES with popsize ...
    >>> if restarter.check_stagnation(fitness_spread=1e-15):
    ...     restarter.register_restart()
    ...     popsize = restarter.get_next_popsize()
    """

    base_popsize: int
    n_params: int
    max_restarts: int = 9
    min_fitness_spread: float = 1e-12

    # Internal state
    restart_count: int = field(default=0, init=False)
    _use_large_pop: bool = field(default=True, init=False)
    _best_solution: jax.Array | None = field(default=None, init=False)
    _best_fitness: float = field(default=-float("inf"), init=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(), init=False
    )

    @property
    def exhausted(self) -> bool:
        """Whether maximum restarts have been reached."""
        return self.restart_count >= self.max_restarts

    @property
    def best_solution(self) -> jax.Array | None:
        """Best solution found across all restarts."""
        return self._best_solution

    @property
    def best_fitness(self) -> float:
        """Best fitness found across all restarts."""
        return self._best_fitness

    def get_next_popsize(self) -> int:
        """Get population size for the next run.

        Returns
        -------
        int
            Population size to use for next CMA-ES run.
            Large runs use 2x base_popsize, small runs use base_popsize/2 to base_popsize.
        """
        if self._use_large_pop:
            # Large population: doubled base
            popsize = self.base_popsize * 2
            logger.debug(f"BIPOP: Using large population (popsize={popsize})")
        else:
            # Small population: random between base/2 and base
            min_pop = max(4, self.base_popsize // 2)  # At least 4
            max_pop = self.base_popsize
            popsize = int(self._rng.integers(min_pop, max_pop + 1))
            logger.debug(f"BIPOP: Using small population (popsize={popsize})")

        return popsize

    def register_restart(self) -> None:
        """Register that a restart has occurred.

        Call this after a restart to update internal state.
        """
        self.restart_count += 1
        self._use_large_pop = not self._use_large_pop  # Alternate

        logger.debug(
            f"BIPOP: Restart {self.restart_count}/{self.max_restarts}, "
            f"next run: {'large' if self._use_large_pop else 'small'}"
        )

    def check_stagnation(self, fitness_spread: float) -> bool:
        """Check if optimization has stagnated.

        Parameters
        ----------
        fitness_spread : float
            Difference between max and min fitness in current population.

        Returns
        -------
        bool
            True if stagnation detected (fitness spread below threshold).
        """
        return fitness_spread < self.min_fitness_spread

    def update_best(self, solution: jax.Array, fitness: float) -> None:
        """Update best solution if the new one is better.

        Parameters
        ----------
        solution : jax.Array
            Candidate solution.
        fitness : float
            Fitness value (CMA-ES maximizes, so higher is better).
        """
        if fitness > self._best_fitness:
            self._best_solution = solution
            self._best_fitness = fitness
            logger.debug(f"BIPOP: New best fitness: {fitness:.6e}")

    def get_best(self) -> tuple[jax.Array | None, float]:
        """Get the best solution found across all restarts.

        Returns
        -------
        tuple[jax.Array | None, float]
            Best solution and its fitness, or (None, -inf) if none found.
        """
        return self._best_solution, self._best_fitness

    def reset(self) -> None:
        """Reset the restarter to initial state."""
        self.restart_count = 0
        self._use_large_pop = True
        self._best_solution = None
        self._best_fitness = -float("inf")
