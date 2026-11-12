"""Regression: CG/lsmr trust-region step must reach the trust-region boundary.

The regularised CG step used one damped solve with alpha = ||g||/Delta, which
lands well inside the region; the radius then never grew (growth needs the step
on the boundary) and the default ``curve_fit`` (solver="auto" -> lsmr for
m*n >= 10000) stalled at max_nfev with a wrong popt.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq.core.trf_jit import _solve_tr_subproblem_cg, _solve_tr_subproblem_cg_bounds


def _exact_tr_step(A, b, Delta):
    """argmin ||A p + b|| s.t. ||p|| <= Delta, via SVD + bisection on alpha."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    c = U.T @ b

    def step(alpha):
        return -Vt.T @ (s * c / (s**2 + alpha))

    lo, hi = 0.0, 1.0
    while np.linalg.norm(step(hi)) > Delta:
        hi *= 10
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.linalg.norm(step(mid)) > Delta else (lo, mid)
    return step(hi)


def _problem():
    rng = np.random.default_rng(3)
    J = rng.standard_normal((40, 4)) * np.array([10.0, 1.0, 0.1, 0.01])
    f = rng.standard_normal(40)
    d = np.array([1.0, 2.0, 0.5, 1.0])
    return J, f, d


def test_cg_step_lies_on_boundary_and_matches_exact():
    J, f, d = _problem()
    Delta = 0.5
    p = np.asarray(
        _solve_tr_subproblem_cg(jnp.array(J), jnp.array(f), jnp.array(d), Delta)
    )
    ref = _exact_tr_step(J * d, f, Delta)
    assert np.linalg.norm(p) == pytest.approx(Delta, rel=1e-6)
    np.testing.assert_allclose(p, ref, rtol=0.05, atol=0.05 * Delta)


def test_cg_bounds_step_lies_on_boundary_and_matches_exact():
    J, f, d = _problem()
    J_diag = np.diag([0.3, 0.0, 1.0, 0.2])
    Delta = 0.5
    p = np.asarray(
        _solve_tr_subproblem_cg_bounds(
            jnp.array(J),
            jnp.array(f),
            jnp.array(d),
            jnp.array(J_diag),
            jnp.zeros(4),
            Delta,
        )
    )
    ref = _exact_tr_step(
        np.vstack([J * d, J_diag]), np.concatenate([f, np.zeros(4)]), Delta
    )
    assert np.linalg.norm(p) == pytest.approx(Delta, rel=1e-6)
    np.testing.assert_allclose(p, ref, rtol=0.05, atol=0.05 * Delta)


def test_cg_step_inside_region_is_gauss_newton():
    J, f, d = _problem()
    p = np.asarray(
        _solve_tr_subproblem_cg(jnp.array(J), jnp.array(f), jnp.array(d), 1e6)
    )
    ref = np.linalg.lstsq(J * d, -f, rcond=None)[0]
    np.testing.assert_allclose(p, ref, rtol=1e-6)


def _exp_model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


@pytest.mark.parametrize("tr_solver", [None, "lsmr"])
def test_default_curve_fit_on_cg_path_converges(tr_solver):
    # M = 3334 -> m*n = 10002 >= 10000, so solver="auto" picks the CG kernel.
    M = 3334
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, M)
    y = 2500 * np.exp(-1.3 * x) + 0.0005 + 5 * rng.standard_normal(M)
    kw = {} if tr_solver is None else {"tr_solver": tr_solver}
    res = nlsq.curve_fit(_exp_model, x, y, p0=[1000, 1, 0], **kw)
    ref = nlsq.curve_fit(_exp_model, x, y, p0=[1000, 1, 0], tr_solver="exact")
    assert res.success
    np.testing.assert_allclose(res[0], ref[0], rtol=1e-4, atol=1e-2)
    assert res.nfev <= 30


def test_bounded_curve_fit_on_cg_path_converges():
    M = 3334
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, M)
    y = 2500 * np.exp(-1.3 * x) + 0.0005 + 5 * rng.standard_normal(M)
    kw = {"p0": [1000, 1, 0], "bounds": (-1e10, 1e10)}
    res = nlsq.curve_fit(_exp_model, x, y, tr_solver="lsmr", **kw)
    ref = nlsq.curve_fit(_exp_model, x, y, tr_solver="exact", **kw)
    assert res.success
    np.testing.assert_allclose(res[0], ref[0], rtol=1e-4, atol=1e-2)
    assert res.nfev <= 30
