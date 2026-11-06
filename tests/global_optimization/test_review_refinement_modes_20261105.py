"""CMA-ES refinement must not discard its result for tuple output flags."""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.global_optimization.cmaes_optimizer as module
from nlsq.global_optimization.cmaes_config import CMAESConfig


@pytest.mark.parametrize("flag", ["full_output", "return_eval", "timeit"])
def test_direct_cmaes_refinement_keeps_fit_when_output_flag_given(monkeypatch, flag):
    monkeypatch.setattr(module, "is_evosax_available", lambda: True)
    optimizer = module.CMAESOptimizer(
        CMAESConfig(max_generations=1, restart_strategy="none")
    )
    monkeypatch.setattr(
        optimizer, "_run_cmaes", lambda *args, **kwargs: (jnp.zeros(1), 10.0, 1)
    )
    x = np.linspace(0, 2, 20)
    y = 3 * x
    result = optimizer.fit(
        lambda x, a: a * x, x, y, p0=[1.0], bounds=([0.0], [4.0]), **{flag: True}
    )
    np.testing.assert_allclose(result["popt"], [3.0], atol=1e-5)
    assert np.isfinite(result["pcov"]).all()
