"""R9: a p0 exactly on a bound used to pin CMA-ES there (flat sigmoid)."""

import jax.numpy as jnp
import numpy as np

from nlsq.global_optimization import CMAESConfig, CMAESOptimizer


def _f(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def test_p0_on_lower_bound_still_converges():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 3, 300)
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.01 * rng.standard_normal(300)
    cfg = CMAESConfig(
        seed=1, max_generations=60, restart_strategy="none", refine_with_nlsq=False
    )
    r = CMAESOptimizer(cfg).fit(
        _f, x, y, p0=[1, 1, 0], bounds=([0, 0, 0], [10, 10, 10])
    )
    np.testing.assert_allclose(np.asarray(r["popt"]), [2.5, 1.3, 0.5], atol=0.1)
