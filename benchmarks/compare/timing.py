"""Timing and accuracy-gate helpers."""

import time
from collections.abc import Callable

import numpy as np

from .problems import Problem
from .types import FitOutcome, Outcomes, Timing


def time_fit(run: Callable[[], Outcomes], repeats: int = 7) -> tuple[Timing, Outcomes]:
    """First call is cold; then `repeats` warm calls. Returns the last outcome."""
    t0 = time.perf_counter()
    out = run()
    cold = time.perf_counter() - t0
    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = run()
        warm.append(time.perf_counter() - t0)
    q25, q75 = np.percentile(warm, [25, 75])
    return Timing(cold, float(np.median(warm)), float(q75 - q25)), out


def residual_cost(problem: Problem, x, y, params) -> float:
    """0.5 * sum of squared residuals, recomputed from returned parameters."""
    r = np.asarray(y) - problem.model(np, np.asarray(x), *np.asarray(params))
    return float(0.5 * np.sum(r**2))


def case_cost(problem: Problem, x, Y, outcomes: Outcomes) -> float:
    """Single-fit cost, or mean cost over a batch (`Y` is (batch, n))."""
    if isinstance(outcomes, FitOutcome):
        return residual_cost(problem, x, Y, outcomes.params)
    return float(
        np.mean(
            [
                residual_cost(problem, x, y, o.params)
                for y, o in zip(Y, outcomes, strict=True)
            ]
        )
    )


def all_converged(outcomes: Outcomes) -> bool:
    if isinstance(outcomes, FitOutcome):
        return outcomes.converged
    return all(o.converged for o in outcomes)


def reaches_optimum(cost: float, best_cost: float, rtol: float = 1e-6) -> bool:
    tiny = 1e-300
    return cost <= best_cost + rtol * max(abs(best_cost), tiny)
