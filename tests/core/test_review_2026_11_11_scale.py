"""Regression: CG trust-region solver at extreme y scales (review 2026-11-11).

Unpreconditioned CG on the normal equations failed when Jacobian columns
differed by ~1e8 (y scaled by 1e8 or 1e-8), so ``tr_solver="lsmr"`` (the auto
choice for m*n >= 10000) returned p0 unchanged via ``xtol``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit

# SciPy trf solution of the unscaled problem (scaled by s for amplitude/offset)
POPT_UNIT = np.array([2.99427067, 1.29670459, 0.49900105])
POPT_SMALL = np.array([2.99425158, 1.29668251, 0.49900014])  # s=1e-8: gtol stop
# s=1e-8 with finite bounds: SciPy trf/exact also stops early on gtol here
POPT_SMALL_BOUNDED = np.array([2.99393013, 1.29647185, 0.49899683])


def _model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
@pytest.mark.parametrize("tr_solver", ["exact", "lsmr"])
@pytest.mark.parametrize("x_scale", [1.0, "jac"])
@pytest.mark.parametrize("bounded", [False, True])
def test_extreme_y_scale_converges(scale, tr_solver, x_scale, bounded):
    x = np.linspace(0, 5, 5000)
    y = 3 * np.exp(-1.3 * x) + 0.5
    y = y + 0.05 * np.random.default_rng(1).standard_normal(5000)
    kw = {"bounds": ([-np.inf, 0, -np.inf], [np.inf, 10, np.inf])} if bounded else {}
    popt, _ = curve_fit(
        _model,
        x,
        y * scale,
        p0=[2 * scale, 1, 0],
        tr_solver=tr_solver,
        x_scale=x_scale,
        **kw,
    )
    if scale == 1e-8:
        ref = POPT_SMALL_BOUNDED if bounded else POPT_SMALL
    else:
        ref = POPT_UNIT
    expected = ref * np.array([scale, 1, scale])
    np.testing.assert_allclose(popt, expected, rtol=1e-4)
