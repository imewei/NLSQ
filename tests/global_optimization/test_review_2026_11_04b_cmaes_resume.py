"""A finished (xtol-converged) CMA-ES job resubmitted must be a no-op."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import CMAESConfig, CMAESOptimizer


def _f(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.mark.serial
def test_converged_run_resubmission_is_stable(tmp_path):
    x = np.linspace(0, 5, 50)
    y = np.asarray(_f(x, 2.5, 0.5))
    gens = []
    for _ in range(3):
        cfg = CMAESConfig(
            restart_strategy="none",
            seed=17,
            model_id="m",
            max_generations=200,
            tol_x=1e-3,
            checkpoint_dir=str(tmp_path),
            checkpoint_interval=5,
            run_id="r",
        )
        r = CMAESOptimizer(config=cfg).fit(
            _f, x, y, bounds=([0, 0], [10, 2]), refine_with_nlsq=False
        )
        dg = r["cmaes_diagnostics"]
        gens.append(dg["total_generations"])
        assert dg["convergence_reason"] == "xtol"
    assert gens[0] == gens[1] == gens[2]
