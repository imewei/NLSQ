"""Unit tests for the Stage 3 conditioning-study helpers."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares

mpmath = pytest.importorskip("mpmath")

from benchmarks.studies.conditioning_helpers import (
    eigh_rank,
    error_norms,
    failed_row_eligible,
    mp_gn_step,
    mp_reference,
    reference_conflict,
    svd_covariance,
    tsqr_solve,
    verdict,
)


def exp_decay(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = 2.5 * np.exp(-1.3 * x) + 0.01 * rng.standard_normal(n)
    return x, y


def _chunks(x, y, sizes):
    edges = np.cumsum([0, *sizes])
    assert edges[-1] == len(x)
    return [(x[s:e], y[s:e]) for s, e in itertools.pairwise(edges)]


def _normal_equations(x, y, theta):
    a, b = theta
    e = np.exp(-b * x)
    J = np.column_stack([e, -a * x * e])
    r = y - a * e
    return J, r


def test_mp_reference_matches_scipy_on_well_conditioned_problem():
    x, y = _data()
    theta, cost, cov, rank = mp_reference(
        lambda xi, a, b: a * mpmath.exp(-b * xi), x, y, np.array([2.0, 1.0])
    )
    sp = least_squares(
        lambda q: q[0] * np.exp(-q[1] * x) - y,
        [2.0, 1.0],
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
    )
    np.testing.assert_allclose(theta, sp.x, rtol=1e-10)
    assert cost == pytest.approx(sp.cost, rel=1e-10)
    assert rank == 2
    J = sp.jac
    cov_sp = np.linalg.inv(J.T @ J) * (2 * sp.cost / (len(x) - 2))
    np.testing.assert_allclose(cov, cov_sp, rtol=1e-6)


def test_tsqr_step_equals_normal_equation_step_when_well_conditioned():
    x, y = _data()
    theta = np.array([2.0, 1.0])
    J, r = _normal_equations(x, y, theta)
    H, g = J.T @ J, J.T @ r
    d = np.diag(H).copy()
    lam = 1e-3
    step, F = tsqr_solve(_chunks(x, y, [90, 2, 108]), theta, exp_decay, lam, d)
    np.testing.assert_allclose(
        step, np.linalg.solve(H + lam * np.diag(d), g), rtol=1e-8
    )
    assert pytest.approx(0.5 * r @ r, rel=1e-12) == F


def test_tsqr_step_uses_supplied_damping():
    x, y = _data()
    theta = np.array([2.0, 1.0])
    J, r = _normal_equations(x, y, theta)
    H, g = J.T @ J, J.T @ r
    lam = 0.5
    d1 = np.diag(H).copy()
    d2 = d1 * np.array([4.0, 25.0])  # history-inflated running maximum
    chunks = _chunks(x, y, [50, 150])
    s1, _ = tsqr_solve(chunks, theta, exp_decay, lam, d1)
    s2, _ = tsqr_solve(chunks, theta, exp_decay, lam, d2)
    np.testing.assert_allclose(s1, np.linalg.solve(H + lam * np.diag(d1), g), rtol=1e-8)
    np.testing.assert_allclose(s2, np.linalg.solve(H + lam * np.diag(d2), g), rtol=1e-8)
    assert not np.allclose(s1, s2, rtol=1e-3)


def test_tsqr_cost_includes_column_space_residual():
    # linear model: at theta = 0 the whole residual y = 3 + 2x lies in span(J)
    x = np.linspace(0.0, 1.0, 50)
    y = 3.0 + 2.0 * x

    def line(x, a, b):
        return a + b * x

    step, F = tsqr_solve(_chunks(x, y, [20, 1, 29]), np.zeros(2), line, 0.0, np.ones(2))
    assert pytest.approx(0.5 * y @ y, rel=1e-12) == F
    np.testing.assert_allclose(step, [3.0, 2.0], rtol=1e-12)


def test_error_norms_zero_reference_covariance():
    cov = np.array([[1e-3, 0.0], [0.0, 2e-3]])
    out = error_norms(np.ones(2), np.ones(2), cov, np.zeros((2, 2)), rank=2, rank_ref=2)
    assert np.isfinite(out["cov_err"])
    assert out["cov_err"] == pytest.approx(np.linalg.norm(cov))
    assert out["param_err"] == 0.0
    assert out["rank_equal"] is True


def test_error_norms_values_on_synthetic_inputs():
    theta = np.array([1.0 + 1e-7, 0.0 + 5e-13, -2.0])
    theta_ref = np.array([1.0, 0.0, -2.0 - 4e-6])
    cov_ref = np.diag([4.0, 0.0, 0.0])
    cov = np.diag([4.0, 3.0, 0.0])
    out = error_norms(theta, theta_ref, cov, cov_ref, rank=3, rank_ref=1)
    # component errors: 1e-7, 5e-13/1e-12 = 0.5, 4e-6/2
    assert out["param_err"] == pytest.approx(0.5)
    assert out["cov_err"] == pytest.approx(3.0 / 4.0)
    assert out["rank_equal"] is False


def _row(kappa, pe=1e-9, ce=1e-6, rank_equal=True):
    return {
        "kappa": kappa,
        "normal_equations": {
            "param_err": pe,
            "cov_err": ce,
            "rank_equal": rank_equal,
        },
    }


def test_verdict_rule():
    passing = [_row(10.0), _row(1e4), _row(1e3, pe=1e-6, ce=1e-3)]
    assert verdict(passing) == "normal_equations_stay"
    assert verdict([*passing, _row(1e4, ce=2e-3)]) == "tsqr_default"
    assert verdict([*passing, _row(1e2, pe=2e-6)]) == "tsqr_default"
    assert verdict([*passing, _row(1e2, rank_equal=False)]) == "tsqr_default"
    assert verdict([*passing, _row(1e2, ce=float("nan"))]) == "tsqr_default"
    # rows above kappa = 1e4 are outside the rule
    assert (
        verdict([*passing, _row(1.0001e4, pe=1.0, ce=1.0, rank_equal=False)])
        == "normal_equations_stay"
    )
    with pytest.raises(ValueError):
        verdict([_row(1e6)])


def test_verdict_rejects_nan_kappa():
    with pytest.raises(ValueError, match="NaN"):
        verdict([_row(10.0), _row(float("nan"))])


def test_svd_covariance_and_eigh_rank():
    rng = np.random.default_rng(1)
    J = rng.standard_normal((40, 3))
    cov, rank = svd_covariance(J, cost=2.0, n=40)
    np.testing.assert_allclose(cov, np.linalg.inv(J.T @ J) * (4.0 / 37), rtol=1e-12)
    assert rank == 3 and eigh_rank(J.T @ J) == 3
    Jc = np.column_stack([J[:, :2], J[:, 0]])  # exactly collinear third column
    assert svd_covariance(Jc, cost=2.0, n=40)[1] == 2
    assert eigh_rank(Jc.T @ Jc) == 2


def test_mp_reference_analytic_jacobian_hook():
    x, y = _data(n=50)

    def f(xi, a, b):
        return a * mpmath.exp(-b * xi)

    def jac(xi, a, b):
        e = mpmath.exp(-b * xi)
        return [e, -a * xi * e]

    p0 = np.array([2.4, 1.2])
    out_diff = mp_reference(f, x, y, p0)
    out_jac = mp_reference(f, x, y, p0, jac_mp=jac)
    np.testing.assert_allclose(out_jac[0], out_diff[0], rtol=1e-14)
    np.testing.assert_allclose(out_jac[2], out_diff[2], rtol=1e-12)
    assert out_jac[3] == out_diff[3] == 2


def test_mp_gn_step_matches_float64_gauss_newton_step():
    x, y = _data(n=60)
    theta = np.array([2.3, 1.2])
    J, r = _normal_equations(x, y, theta)

    def f(xi, a, b):
        return a * mpmath.exp(-b * xi)

    def jac(xi, a, b):
        e = mpmath.exp(-b * xi)
        return [e, -a * xi * e]

    expected = np.linalg.lstsq(J, r, rcond=None)[0]
    np.testing.assert_allclose(mp_gn_step(f, jac, x, y, theta), expected, rtol=1e-10)
    np.testing.assert_allclose(mp_gn_step(f, None, x, y, theta), expected, rtol=1e-10)


def test_failed_row_eligible():
    assert failed_row_eligible([1e16, 8.5e3, 2e5])  # one solver inside the range
    assert failed_row_eligible([1e16, float("nan"), 2e5])  # unknown cannot exclude
    assert not failed_row_eligible([1e16, 1.0001e4, 2e5])


def test_reference_conflict():
    t = np.array([1.0, 2.0])
    assert not reference_conflict([(t, 1.0)])
    assert not reference_conflict([(t, 1.0), (t * (1 + 1e-6), 1.0 + 1e-12)])
    assert reference_conflict([(t, 1.0), (t, 1.0 + 1e-9)])  # cost
    assert reference_conflict([(t, 1.0), (t * (1 + 1e-3), 1.0)])  # parameters
