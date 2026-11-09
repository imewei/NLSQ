import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def test_hybrid_handles_jacobian_columns_of_very_different_scale():
    rng = np.random.default_rng(0)

    def line(x, a, b):
        return a * x + b

    x = np.linspace(0, 1e6, 20000)
    y = 3e-6 * x + 2e6 + 0.5 * rng.standard_normal(x.size)
    # closed-form OLS optimum
    ref = np.linalg.lstsq(np.c_[x, np.ones_like(x)], y, rcond=None)[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig()).fit(
            (x, y), line, np.array([1e-6, 1e6]), verbose=0
        )
    np.testing.assert_allclose(res["x"][0], ref[0], rtol=1e-3)


def test_gauss_newton_step_ignores_near_zero_jacobian_column():
    """Regularisation must not dominate after Jacobi scaling of a ~0 column."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    m = 1000
    J = np.column_stack(
        [rng.normal(size=m), rng.normal(size=m), 1e-12 * rng.normal(size=m)]
    )
    r = J[:, :2] @ np.array([0.3, -0.2]) + 1e-3 * rng.normal(size=m)
    step, _ = AdaptiveHybridStreamingOptimizer._solve_gauss_newton_step(
        None, jnp.asarray(J.T @ J), jnp.asarray(J.T @ r), trust_radius=10.0
    )
    oracle = np.linalg.lstsq(J[:, :2], r, rcond=None)[0]
    np.testing.assert_allclose(np.asarray(step)[:2], oracle, rtol=1e-3)


@pytest.mark.parametrize("scale", [1e-6, 1e-9])
def test_hybrid_streaming_is_invariant_to_small_y_scale(scale):
    """Absolute epsilons/tolerances must not stop the fit when y is tiny."""
    from nlsq import curve_fit_large

    def model(x, a, b):
        return a * jnp.exp(-b * x)

    x = np.linspace(0, 5, 20000)
    rng = np.random.default_rng(0)
    y = scale * (2.0 * np.exp(-1.3 * x) + 0.01 * rng.standard_normal(x.size))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt = np.asarray(
            curve_fit_large(
                model,
                x,
                y,
                p0=[scale, 1.0],
                method="hybrid_streaming",
                bounds=([0, 0], [scale * 10, 10]),
            )[0]
        )
    np.testing.assert_allclose(popt / [scale, 1.0], [2.0, 1.3], rtol=2e-2)
