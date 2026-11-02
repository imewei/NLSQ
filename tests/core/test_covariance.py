"""Unit tests for the shared covariance helpers (``nlsq.core.covariance``).

Behavior preservation against the old call sites is pinned by
``test_covariance_characterization.py``; these tests cover the helpers' own contract.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from nlsq.core.covariance import pcov_from_jacobian, pcov_from_normal
from nlsq.core.minpack import CurveFit
from nlsq.result import OptimizeWarning
from nlsq.streaming import exact_lm

N = 12
X = np.linspace(0.0, 3.0, N)
COST = 0.0002887827887097562  # 0.5 * sum(r^2)


def _jac():
    e = np.exp(-1.3 * X)
    return np.stack([e, -2.0 * X * e], axis=1)


def _svd(J):
    _, s, VT = np.linalg.svd(J, full_matrices=False)
    return s, VT


def _J_pcov(J, cost, n, p, absolute_sigma):
    return pcov_from_jacobian(J, cost, n, p, absolute_sigma, _svd)


def _H_pcov(J, cost, n, p, absolute_sigma):
    return pcov_from_normal(J.T @ J, cost, n, p, absolute_sigma)


FUNCS = [pytest.param(_J_pcov, id="jacobian"), pytest.param(_H_pcov, id="normal")]


def test_jacobian_and_normal_agree_when_well_conditioned():
    J = _jac()
    for absolute_sigma in (False, True):
        a, fa = _J_pcov(J, COST, N, 2, absolute_sigma)
        b, fb = _H_pcov(J, COST, N, 2, absolute_sigma)
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=0)
        assert not fa
        assert not fb


@pytest.mark.parametrize("fn", FUNCS)
def test_duplicate_column_returns_finite_pseudoinverse(fn):
    J = _jac()
    J3 = np.column_stack([J[:, 0], J[:, 0], J[:, 1]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # exact flags singular H
        pcov, fired = fn(J3, COST, N, 3, False)
    assert np.isfinite(pcov).all()
    assert not fired
    assert np.linalg.matrix_rank(pcov) == 2
    np.testing.assert_allclose(pcov @ np.array([1.0, -1.0, 0.0]), 0.0, atol=1e-12)


@pytest.mark.parametrize("fn", FUNCS)
def test_fully_degenerate_returns_inf_and_warns_once(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pcov, fired = fn(np.zeros((N, 2)), COST, N, 2, False)
    assert np.isinf(pcov).all()
    assert fired
    assert [w.category for w in caught] == [OptimizeWarning]
    assert caught[0].filename == __file__


@pytest.mark.parametrize("fn", FUNCS)
def test_absolute_sigma_skips_scaling(fn):
    J = _jac()
    scaled, _ = fn(J, COST, N, 2, False)
    absolute, _ = fn(J, COST, N, 2, True)
    np.testing.assert_allclose(scaled, absolute * (2.0 * COST / (N - 2)), rtol=1e-12)
    # cost is never read under absolute_sigma, even with n <= p
    pcov, fired = fn(J, float("nan"), 2, 2, True)
    assert np.isfinite(pcov).all()
    assert not fired


def test_exact_compute_covariance_still_returns_ndarray():
    H = _jac().T @ _jac()
    out = exact_lm.compute_covariance(H, COST, N, 2, False)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, pcov_from_normal(H, COST, N, 2, False)[0])


class _RecordingLogger:
    def __init__(self):
        self.calls = []

    def warning(self, msg, **fields):
        self.calls.append((msg, fields))


@pytest.mark.parametrize(
    ("J", "ysize", "reason"),
    [
        (np.zeros((N, 2)), N, "singular_jacobian"),
        (_jac(), 2, "insufficient_data"),
    ],
)
def test_fallback_logs_through_instance_logger(J, ysize, reason):
    fit = CurveFit()
    fit.logger = _RecordingLogger()
    res = SimpleNamespace(jac=J, cost=COST)
    with pytest.warns(OptimizeWarning):
        pcov, warn_cov = fit._compute_covariance(res, ysize, np.zeros(2), False)
    assert warn_cov
    assert np.isinf(pcov).all()
    assert fit.logger.calls == [
        ("Covariance could not be estimated", {"reason": reason})
    ]


def test_compute_covariance_uses_instance_svd_hook():
    J = _jac()
    calls = []

    def stub(jac):
        calls.append(jac)
        return _svd(np.asarray(jac) * 2.0)  # distinguishable from the default

    fit = CurveFit()
    fit.covariance_svd = stub
    res = SimpleNamespace(jac=J, cost=COST)
    pcov, warn_cov = fit._compute_covariance(res, N, np.zeros(2), True)
    assert len(calls) == 1
    assert calls[0] is J
    assert not warn_cov
    np.testing.assert_allclose(pcov, _J_pcov(J, COST, N, 2, True)[0] / 4.0, rtol=1e-10)


def test_pcov_from_normal_cond_warn_switch():
    """sqrt(cond(H)) = 1e5 warns by default; cond_warn=False suppresses only that."""
    H = np.diag([1.0, 1e-10])
    with pytest.warns(UserWarning, match="ill-conditioned"):
        loud, fired = pcov_from_normal(H, COST, N, 2, False)
    assert not fired
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        quiet, fired_quiet = pcov_from_normal(H, COST, N, 2, False, cond_warn=False)
    assert not fired_quiet
    np.testing.assert_array_equal(loud, quiet)
    # the fallback warning is not governed by the switch
    with pytest.warns(OptimizeWarning):
        pcov_from_normal(np.zeros((2, 2)), COST, N, 2, False, cond_warn=False)
