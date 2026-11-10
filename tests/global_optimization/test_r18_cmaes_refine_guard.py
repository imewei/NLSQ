import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

pytest.importorskip("evosax")


def _model(x, d0, g0, n):
    return d0 * (1.0 + jnp.power(x / g0, n))


def test_refinement_never_returns_worse_ssr_than_cmaes_point():
    """TRF stops at nfev=1 after nudging a bound-hugging param (tiny y, abs gtol)."""
    np.random.seed(42)
    x = jnp.logspace(-1, 3, 50)
    yt = _model(x, 1e-10, 1e-3, 0.5)
    y = yt + 0.02 * yt * np.random.randn(50)
    bounds = ([1e-12, 1e-5, 0.1], [1e-8, 1e-1, 2.0])

    def ssr(p):
        return float(np.sum((np.asarray(_model(x, *p)) - np.asarray(y)) ** 2))

    def run(refine):
        cfg = CMAESConfig(
            max_generations=100,
            restart_strategy="bipop",
            max_restarts=5,
            seed=42,
            refine_with_nlsq=refine,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return CMAESOptimizer(cfg).fit(_model, x, y, bounds=bounds)

    raw, refined = run(False), run(True)
    assert ssr(refined["popt"]) <= ssr(raw["popt"]) * (1 + 1e-6)


def test_refinement_guard_does_not_reject_robust_loss():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + 1.0 + 0.05 * rng.standard_normal(100)
    y[::10] += 30.0  # outliers: robust loss deliberately raises the plain SSR

    def line(x, a, b):
        return a * x + b

    cfg = CMAESConfig(max_generations=60, seed=0, refine_with_nlsq=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = CMAESOptimizer(cfg).fit(
            line, x, y, bounds=([0, -5], [5, 10]), loss="soft_l1"
        )
    assert res["cmaes_diagnostics"]["nlsq_refinement"] is True
    assert np.all(np.isfinite(res["pcov"]))
