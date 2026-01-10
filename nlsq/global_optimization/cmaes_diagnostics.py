"""CMA-ES Diagnostics dataclass for monitoring optimization.

Provides diagnostic information collected during CMA-ES optimization,
including generation counts, restart history, and convergence metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["CMAESDiagnostics"]


@dataclass(slots=True)
class CMAESDiagnostics:
    """Diagnostic information from CMA-ES optimization.

    Attributes
    ----------
    total_generations : int
        Total number of CMA-ES generations across all restarts.
    total_restarts : int
        Number of BIPOP restarts performed.
    final_sigma : float
        Final step size (standard deviation) at convergence.
    best_fitness : float
        Best fitness value found (negative SSR, higher is better).
    fitness_history : list[float]
        History of best fitness values per generation.
    restart_history : list[dict[str, Any]]
        Information about each restart (popsize, generations, reason).
    convergence_reason : str
        Reason for convergence or termination.
    nlsq_refinement : bool
        Whether NLSQ refinement was applied.
    wall_time : float
        Total wall-clock time in seconds.

    Examples
    --------
    >>> diagnostics = CMAESDiagnostics(
    ...     total_generations=150,
    ...     total_restarts=3,
    ...     final_sigma=0.01,
    ...     best_fitness=-1e-10,
    ...     convergence_reason="fitness_tolerance",
    ... )
    >>> print(diagnostics.summary())
    """

    # Core metrics
    total_generations: int = 0
    total_restarts: int = 0
    final_sigma: float = 0.0
    best_fitness: float = float("inf")

    # Histories
    fitness_history: list[float] = field(default_factory=list)
    restart_history: list[dict[str, Any]] = field(default_factory=list)

    # Convergence info
    convergence_reason: str = "not_converged"
    nlsq_refinement: bool = False

    # Timing
    wall_time: float = 0.0

    def summary(self) -> str:
        """Generate a human-readable summary of the diagnostics.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = [
            "CMA-ES Optimization Summary",
            "-" * 28,
            f"Total generations: {self.total_generations}",
            f"Total restarts: {self.total_restarts}",
            f"Final sigma: {self.final_sigma:.6e}",
            f"Best fitness (neg SSR): {self.best_fitness:.6e}",
            f"Convergence reason: {self.convergence_reason}",
            f"NLSQ refinement: {self.nlsq_refinement}",
            f"Wall time: {self.wall_time:.3f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert diagnostics to a dictionary.

        Returns
        -------
        dict
            Dictionary representation for serialization.
        """
        return {
            "total_generations": self.total_generations,
            "total_restarts": self.total_restarts,
            "final_sigma": self.final_sigma,
            "best_fitness": self.best_fitness,
            "fitness_history": list(self.fitness_history),
            "restart_history": list(self.restart_history),
            "convergence_reason": self.convergence_reason,
            "nlsq_refinement": self.nlsq_refinement,
            "wall_time": self.wall_time,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CMAESDiagnostics:
        """Create diagnostics from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary with diagnostics values.

        Returns
        -------
        CMAESDiagnostics
            Diagnostics instance.
        """
        return cls(
            total_generations=d.get("total_generations", 0),
            total_restarts=d.get("total_restarts", 0),
            final_sigma=d.get("final_sigma", 0.0),
            best_fitness=d.get("best_fitness", float("inf")),
            fitness_history=list(d.get("fitness_history", [])),
            restart_history=list(d.get("restart_history", [])),
            convergence_reason=d.get("convergence_reason", "not_converged"),
            nlsq_refinement=d.get("nlsq_refinement", False),
            wall_time=d.get("wall_time", 0.0),
        )

    def get_fitness_improvement(self) -> float | None:
        """Calculate fitness improvement from first to last generation.

        Returns
        -------
        float | None
            Relative fitness improvement, or None if not enough history.
        """
        if len(self.fitness_history) < 2:
            return None

        initial = self.fitness_history[0]
        final = self.fitness_history[-1]

        if initial == 0:
            return None

        # Fitness is negative SSR, so improvement is (final - initial) / |initial|
        # Higher (less negative) fitness is better
        return (final - initial) / abs(initial)

    def get_convergence_rate(self) -> NDArray[np.floating[Any]] | None:
        """Compute per-generation convergence rate.

        Returns
        -------
        NDArray | None
            Array of per-generation fitness changes, or None if not enough history.
        """
        if len(self.fitness_history) < 2:
            return None

        history = np.array(self.fitness_history)
        return np.diff(history)
