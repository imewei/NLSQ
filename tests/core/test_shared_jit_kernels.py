"""Stateless JIT kernels are shared across instances, and results are unchanged.

curve_fit() builds a new CurveFit (and CommonJIT) per call; per-instance
`@jit` closures made JAX re-compile them on every call.
"""

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit
from nlsq.common_jax import CommonJIT
from nlsq.core.minpack import CurveFit

_COMMON = (
    "scale_for_robust_loss_function",
    "js_dot",
    "js0_dot",
    "evaluate_quadratic1",
    "evaluate_quadratic_diagonal1",
    "evaluate_quadratic2",
    "evaluate_quadratic_diagonal2",
    "jac_sum_func",
)
_CURVEFIT = ("sigma_transform1d", "sigma_transform2d", "covariance_svd")


def test_kernels_are_shared_between_instances():
    a, b = CommonJIT(), CommonJIT()
    for name in _COMMON:
        assert getattr(a, name) is getattr(b, name), name
    c, d = CurveFit(), CurveFit()
    for name in _CURVEFIT:
        assert getattr(c, name) is getattr(d, name), name


def test_quadratic_and_covariance_kernels_values():
    rng = np.random.default_rng(1)
    J = jnp.asarray(rng.standard_normal((20, 3)))
    g = jnp.asarray(rng.standard_normal(3))
    s = jnp.asarray(rng.standard_normal(3))
    cj = CommonJIT()
    Js = np.asarray(J) @ np.asarray(s)
    expected = 0.5 * Js @ Js + np.asarray(s) @ np.asarray(g)
    np.testing.assert_allclose(cj.evaluate_quadratic(J, g, s), expected, rtol=1e-12)
    sv, VT = CurveFit().covariance_svd(J)
    np.testing.assert_allclose(sv, np.linalg.svd(np.asarray(J))[1], rtol=1e-12)
    assert VT.shape == (3, 3)


def test_repeated_fits_are_bit_identical():
    x = np.linspace(0.0, 4.0, 300)
    y = 2.5 * np.exp(-1.3 * x) + 0.5
    y = y + 0.05 * np.random.default_rng(0).standard_normal(x.size)
    sigma = np.full_like(x, 0.05)

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    popt1, pcov1 = curve_fit(model, x, y, p0=[1.0, 1.0, 0.0], sigma=sigma)
    popt2, pcov2 = curve_fit(model, x, y, p0=[1.0, 1.0, 0.0], sigma=sigma)
    np.testing.assert_array_equal(popt1, popt2)
    np.testing.assert_array_equal(pcov1, pcov2)
