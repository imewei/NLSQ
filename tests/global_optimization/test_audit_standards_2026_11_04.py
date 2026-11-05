"""Fit parameter inference at global optimization boundaries."""

import numpy as np

from nlsq.global_optimization import GlobalOptimizationConfig, MultiStartOrchestrator
from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer


def _slope(x, a, **options):
    return a * x


def test_multistart_infers_positional_parameters_with_kwargs():
    x = np.linspace(0, 1, 12)
    optimizer = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=2))
    result = optimizer.fit(_slope, x, 2 * x)
    np.testing.assert_allclose(result.popt, [2.0], rtol=1e-6)
    assert result.pcov.shape == (1, 1)


def test_cmaes_scalar_bounds_infer_positional_parameters_with_kwargs():
    x = np.linspace(0, 1, 12)
    optimizer = CMAESOptimizer(
        CMAESConfig(
            max_generations=1,
            restart_strategy="none",
            max_restarts=0,
            refine_with_nlsq=False,
            popsize=4,
        )
    )
    result = optimizer.fit(_slope, x, 2 * x, bounds=(0.0, 4.0))
    assert np.asarray(result["popt"]).shape == (1,)
    assert np.isfinite(result["popt"]).all()
