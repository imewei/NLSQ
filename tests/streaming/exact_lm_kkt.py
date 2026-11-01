"""KKT check for box-bounded exact_lm results, recomputed independently in NumPy float64."""

import jax
import numpy as np


def assert_kkt(result, model, x, y, lb, ub, tau=1e-6):
    """Assert first-order optimality of ``result["x"]`` for ``min 0.5*|y - model|^2`` on a box.

    ``g = J^T r`` (``r = y - f``, so ``grad F = -g``), ``H = J^T J``, ``F = 0.5*sum r^2`` are
    recomputed here in float64 (the Jacobian column by column with JAX forward mode, then
    plain NumPy). Scaled gradient ``g_i/(sqrt(H_ii)*sqrt(2F))``, 0 if ``H_ii = 0`` or
    ``F = 0``; for ``0 < F <= 1e-24*sum(y^2)`` the test is coordinatewise on
    ``g_i`` against ``1e-12*sqrt(sum y^2)*sqrt(H_ii)`` instead. Free: ``|g_i| <= tol``; at
    the lower bound: ``g_i <= tol``; at the upper bound: ``g_i >= -tol``. Feasibility
    ``lb <= theta <= ub`` is exact.
    """
    theta = np.asarray(result["x"], dtype=np.float64)
    p = theta.size
    lb = np.broadcast_to(np.asarray(lb, dtype=np.float64), (p,))
    ub = np.broadcast_to(np.asarray(ub, dtype=np.float64), (p,))
    assert np.all(lb <= theta) and np.all(theta <= ub), (theta, lb, ub)

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(model(x, *theta), dtype=np.float64)
    J = np.asarray(
        jax.jacfwd(lambda q: model(x, *q))(np.asarray(theta)), dtype=np.float64
    )
    r = y - f
    F = 0.5 * float(r @ r)
    g = J.T @ r
    Hd = np.einsum("ij,ij->j", J, J)
    sy2 = float(y @ y)

    if F == 0:
        return
    if 1e-24 * sy2 >= F:  # near-exact: coordinatewise, no division by sqrt(2F)
        gs, tol = g, 1e-12 * np.sqrt(sy2) * np.sqrt(Hd)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            gs = np.where(Hd > 0, g / (np.sqrt(Hd) * np.sqrt(2.0 * F)), 0.0)
        tol = np.full(p, tau)
    at_lo, at_hi = theta <= lb, theta >= ub
    free = ~(at_lo | at_hi)
    assert np.all(np.abs(gs[free]) <= tol[free]), ("free", gs, theta)
    assert np.all(gs[at_lo] <= tol[at_lo]), ("lower", gs, theta)
    assert np.all(gs[at_hi] >= -tol[at_hi]), ("upper", gs, theta)
