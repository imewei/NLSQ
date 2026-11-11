"""Gate G-cov: exact_lm_fit's pcov on both routes against the full-Jacobian SVD reference.

For every case the fit's ``pcov`` (``covariance_method`` "normal" or "tsqr") is compared
with ``pcov_from_jacobian(full_jac(popt), ..., _covariance_svd)`` at the same ``popt``:
scale-aware ``max |dC_ij| / sqrt(C_ii C_jj) <= 1e-8`` and the same rank decision: the
count of directions the route actually retains (singular values of the TSQR ``R`` above
``pcov_from_jacobian``'s cutoff, or eigenvalues of the accumulated ``H`` above
``pcov_from_normal``'s) equals the reference's count on the full ``J`` and the rank the
case was built with. The route taken is asserted too, so a case cannot silently test the other route. Controlled
cases use a linear pointwise model ``f(x, *theta) = B[x] @ theta`` whose singular values,
hence ``kappa(J)``, are exact by construction. Arrays stay below 100K elements.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.compare.problems import PROBLEMS
from nlsq.core.covariance import normal_condition, pcov_from_jacobian
from nlsq.core.minpack import _covariance_svd
from nlsq.result import OptimizeWarning
from nlsq.streaming import exact_lm
from nlsq.streaming.exact_lm import exact_lm_fit
from nlsq.streaming.normal_equations import make_chunk_stats, run_pass

pytestmark = pytest.mark.slow

GATE = 1e-8
TIGHT = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
P = 4


def _basis(n, s, seed=0):
    """``B = U diag(s) V^T``: ``U`` the Q of a seeded Gaussian, ``V`` seeded orthogonal."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, len(s))))
    V, _ = np.linalg.qr(rng.standard_normal((len(s), len(s))))
    B = (U * s) @ V.T
    np.testing.assert_allclose(
        np.linalg.svd(B, compute_uv=False), np.sort(s)[::-1], rtol=1e-12, atol=0
    )
    return B


def _linear(B):
    Bj = jnp.asarray(B)

    def model(x, *theta):
        return Bj[x.astype(jnp.int32)] @ jnp.stack(theta)

    return model


def _full_jac(model, x, theta, sigma=None):
    J = np.asarray(
        jax.jacfwd(lambda q: model(jnp.asarray(x), *q))(jnp.asarray(theta, float))
    )
    return J if sigma is None else J / np.asarray(sigma)[:, None]


def _svd_retained(A, n):
    """Count kept by ``pcov_from_jacobian``: ``s > eps * max(n, p) * s[0]``."""
    s = np.asarray(_covariance_svd(np.asarray(A))[0])
    return int(np.sum(s > np.finfo(float).eps * max(n, A.shape[1]) * s[0]))


def _eig_retained(H):
    """Count kept by ``pcov_from_normal``: eigenvalues of the power-of-two scaled ``H``
    above ``p * eps * lambda_max``."""
    s = np.ldexp(1.0, int(np.frexp(np.max(np.abs(H)))[1]) - 1)
    w = np.linalg.eigh((H / s + (H / s).T) / 2)[0]
    return int(np.sum(w > H.shape[0] * np.finfo(np.float64).eps * w[-1]))


def _route_retained(res, model, x, y, sigma, chunk):
    """Directions retained by the route ``res`` took, from its own factor at ``res.x``."""
    p = res.x.size
    src, w = exact_lm._open_source(x, y, sigma, chunk, p, None, None)
    if res.covariance_method == "tsqr":
        R = exact_lm.tsqr_r_factor(model, res.x, src, chunk, weighted=w)
        return _svd_retained(R, x.size)
    return _eig_retained(
        run_pass(make_chunk_stats(model, weighted=w), res.x, src, chunk).H
    )


def _check(
    model,
    x,
    y,
    p0,
    *,
    sigma=None,
    absolute_sigma=False,
    bounds=None,
    chunk=700,
    tol=TIGHT,
    rank=None,
):
    """Fit, compare with the reference and return ``(route, err, kappa)``.

    ``rank`` is the rank the case was built with (default: full, ``len(p0)``).
    """
    kw = {} if bounds is None else {"bounds": bounds}
    res = exact_lm_fit(
        model,
        x,
        y,
        p0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        chunk_size=chunk,  # a padded tail on every n used here
        **kw,
        **tol,
    )
    assert res.success, res.message
    J = _full_jac(model, x, res.x, sigma)
    kappa = normal_condition(J.T @ J)
    ref, _ = pcov_from_jacobian(
        J, res.cost, x.size, len(p0), absolute_sigma, _covariance_svd
    )
    d = np.sqrt(np.outer(np.diag(ref), np.diag(ref)))
    err = float(np.max(np.abs(res.pcov - ref) / d))
    print(f"route={res.covariance_method} kappa={kappa:.4g} err={err:.3g}")
    expected = len(p0) if rank is None else rank
    got = _route_retained(res, model, x, y, sigma, chunk)
    assert got == _svd_retained(J, x.size) == expected, (got, expected)
    assert err <= GATE, (res.covariance_method, kappa, err)
    return res.covariance_method, err, kappa


def _controlled(kappa, n=2000, seed=0):
    B = _basis(n, np.geomspace(kappa, 1.0, P), seed)
    x = np.arange(n, dtype=float)
    theta = np.array([1.0, -2.0, 0.5, 3.0])
    y = B @ theta + 1e-2 * np.random.default_rng(seed + 1).standard_normal(n)
    return _linear(B), x, y


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        # the TSQR-route notice above COND_WARN; asserted in test_exact_lm.py
        warnings.filterwarnings("ignore", "normal-equation Hessian", UserWarning)
        yield


@pytest.mark.parametrize("kappa", [1, 10, 100, 500, 900, 1100, 2000])
def test_controlled_kappa(kappa):
    model, x, y = _controlled(kappa)
    route, _, measured = _check(model, x, y, np.zeros(P))
    assert measured == pytest.approx(kappa, rel=1e-6)
    assert route == ("normal" if kappa <= exact_lm.TSQR_KAPPA else "tsqr")


@pytest.mark.parametrize("name", ["two_exp", "gaussian"])
def test_realism(name):
    prob = PROBLEMS[name]
    x, y = prob.data(20_000, seed=0)

    def model(x, *q):
        return prob.model(jnp, x, *q)

    route, _, kappa = _check(model, x, y, np.asarray(prob.p0, float))
    assert route == ("normal" if kappa <= exact_lm.TSQR_KAPPA else "tsqr")


def test_correlated_parameters():
    # quadratic on t in [1, 2]: |corr| > 0.99 between the coefficients
    t = np.linspace(1.0, 2.0, 2000)
    B = np.vander(t, 3, increasing=True)
    x = np.arange(t.size, dtype=float)
    y = B @ np.array([1.0, -2.0, 0.5]) + 1e-2 * np.random.default_rng(
        0
    ).standard_normal(t.size)
    route, _, kappa = _check(_linear(B), x, y, np.zeros(3))
    assert route == ("normal" if kappa <= exact_lm.TSQR_KAPPA else "tsqr")


def test_column_scales_1e_minus6_to_1e6():
    # J = U diag(s) V^T diag(c), c in 1e-6..1e6: kappa(J) ~ 1e12, kept by the SVD cutoff
    # (eps * n * s0 at n = 300 is 7e-14 relative)
    n = 300
    B = _basis(n, np.geomspace(2.0, 1.0, P)) * np.geomspace(1e-6, 1e6, P)
    x = np.arange(n, dtype=float)
    theta = np.array([1e6, -2.0, 0.5, 3e-6])
    y = B @ theta + 1e-2 * np.random.default_rng(1).standard_normal(n)
    route, _, _ = _check(_linear(B), x, y, np.zeros(P), chunk=128)
    assert route == "tsqr"


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("kappa", [100, 2000])
def test_sigma(kappa, absolute_sigma):
    model, x, y = _controlled(kappa)
    sigma = 0.5 + np.random.default_rng(2).random(x.size)
    route, _, measured = _check(
        model, x, y, np.zeros(P), sigma=sigma, absolute_sigma=absolute_sigma
    )
    assert route == ("normal" if measured <= exact_lm.TSQR_KAPPA else "tsqr")


def test_active_bound():
    model, x, y = _controlled(100)
    hi = np.array([0.5, np.inf, np.inf, np.inf])  # theta_0 = 1 unconstrained: active
    route, _, _ = _check(model, x, y, np.zeros(P), bounds=(-np.inf, hi))
    assert route == "normal"


def test_rank_deficient():
    # s = (10, 5, 1, 0): rank 3. Rounding leaves lambda_min of H tiny but positive
    # (normal_condition ~ 1.7e8, not inf), still far above TSQR_KAPPA: the TSQR route
    _, x, y = _controlled(1)  # data only; the model below is singular
    B = _basis(2000, np.array([10.0, 5.0, 1.0, 1.0]))
    U, s, VT = np.linalg.svd(B, full_matrices=False)
    B = (U[:, :3] * s[:3]) @ VT[:3]
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        # default tolerances: at 1e-12 the LM stalls on the flat null direction
        route, _, _ = _check(_linear(B), x, y, np.zeros(P), tol={}, rank=3)
    assert route == "tsqr"
