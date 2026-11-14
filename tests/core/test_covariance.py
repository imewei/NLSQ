"""Unit tests for the shared covariance helpers (``nlsq.core.covariance``).

Behavior preservation against the old call sites is pinned by
``test_covariance_characterization.py``; these tests cover the helpers' own contract.
"""

from __future__ import annotations

import contextlib
import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.core import minpack
from nlsq.core.covariance import pcov_from_jacobian, pcov_from_normal
from nlsq.core.minpack import CurveFit, _covariance_svd
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


@pytest.mark.parametrize(
    "h,cost,expected",
    [
        (1.0, 1e308, (1e308 / 3) * 2),
        (1e-300, 1e-300, 2 / 3),
        (1e300, 1e300, 2 / 3),
        (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0), 2 / 3),
        (1e-300, 0.0, 0.0),
        (1e300, 0.0, 0.0),
    ],
)
def test_normal_relative_covariance_avoids_intermediate_overflow(h, cost, expected):
    pcov, fired = pcov_from_normal(np.array([[h]]), cost, 4, 1, False)
    assert not fired
    assert pcov[0, 0] == pytest.approx(expected, rel=1e-12, abs=0)


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_normal_covariance_is_invariant_to_parameter_units(absolute_sigma):
    """Rescaling a Jacobian column rescales pcov and must not drop directions."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(50, 3))
    scale = np.array([1.0, 1e9, 1.0])
    ref, _ = _H_pcov(J, 1.3, 50, 3, absolute_sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # ill-conditioned H warning
        got, fired = _H_pcov(J * scale, 1.3, 50, 3, absolute_sigma)
        jac, _ = _J_pcov(J * scale, 1.3, 50, 3, absolute_sigma)
    assert not fired
    np.testing.assert_allclose(got * np.outer(scale, scale), ref, rtol=1e-6, atol=0)
    np.testing.assert_allclose(got, jac, rtol=1e-6, atol=0)


def test_jacobian_covariance_handles_extreme_singular_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pcov, fired = _J_pcov(np.array([[1e160], [2e160]]), 1, 2, 1, True)
    assert not fired
    assert pcov[0, 0] > 0
    assert pcov[0, 0] == pytest.approx(2e-321, rel=0.01, abs=0)

    with pytest.warns(OptimizeWarning):
        pcov, fired = _J_pcov(np.array([[1e-200], [2e-200]]), 1, 2, 1, True)
    assert fired
    assert np.isinf(pcov).all()


def test_float32_jacobian_covariance_does_not_overflow_float64():
    J = np.array([[1e-20], [2e-20]], dtype=np.float32)
    fit = CurveFit()
    pcov, fired = fit._compute_covariance(
        SimpleNamespace(jac=J, cost=1.0), 2, np.zeros(1), True
    )
    assert not fired
    assert pcov[0, 0] == pytest.approx(2e39, rel=1e-6)


@pytest.mark.parametrize(
    ("jac_scale", "cost", "expected"),
    [
        (1e200, 5e199, 2e-201),
        (1e-200, 5e-201, 2e199),
        (1e-200, 0, 0),
        (1.0, 1.5e308, 6e307),
    ],
)
def test_relative_covariance_preserves_extreme_scale(jac_scale, cost, expected):
    pcov, fired = _J_pcov(np.array([[jac_scale], [2 * jac_scale]]), cost, 2, 1, False)
    assert not fired
    assert pcov[0, 0] == pytest.approx(expected, rel=1e-12, abs=0)


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
    # the hook now sees the column-equilibrated Jacobian
    np.testing.assert_allclose(
        calls[0], np.asarray(J) / np.linalg.norm(np.asarray(J), axis=0)
    )
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


def _svd_ref(J):
    _, s, VT = np.linalg.svd(np.asarray(J), full_matrices=False)
    return s, VT


def _rel(a, b):
    d = np.sqrt(np.outer(np.diag(b), np.diag(b)))
    return np.max(np.abs(a - b) / np.where(d > 0, d, 1.0))


@pytest.mark.parametrize("shape,rank", [((200, 3), 3), ((200, 5), 3), ((50, 3), 2)])
def test_qr_covariance_matches_svd(shape, rank):
    rng = np.random.default_rng(0)
    J = jnp.asarray(
        rng.normal(size=(shape[0], rank)) @ rng.normal(size=(rank, shape[1]))
    )
    deficient = rank < shape[1]

    def ctx():  # a fresh context per use: a pytest.warns object is single-use
        return (
            pytest.warns(OptimizeWarning, match="rank-deficient")
            if deficient
            else contextlib.nullcontext()
        )

    with ctx():
        got, _ = pcov_from_jacobian(J, 0.7, shape[0], shape[1], False, _covariance_svd)
    with ctx():
        ref, _ = pcov_from_jacobian(J, 0.7, shape[0], shape[1], False, _svd_ref)
    assert _rel(got, ref) <= 1e-12


def test_qr_covariance_wide_absolute_sigma():
    rng = np.random.default_rng(1)
    J = jnp.asarray(rng.normal(size=(2, 4)))
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        got, _ = pcov_from_jacobian(J, 0.7, 2, 4, True, _covariance_svd)
    with pytest.warns(OptimizeWarning, match="rank-deficient"):
        ref, _ = pcov_from_jacobian(J, 0.7, 2, 4, True, _svd_ref)
    assert _rel(got, ref) <= 1e-12


def test_qr_covariance_wide_relative_sigma_is_inf():
    J = jnp.asarray(np.random.default_rng(2).normal(size=(2, 4)))
    with pytest.warns(OptimizeWarning):
        got, fired = pcov_from_jacobian(J, 0.7, 2, 4, False, _covariance_svd)
    assert fired and np.all(np.isinf(got))


def test_qr_covariance_kernel_uses_r_factor(monkeypatch):
    calls, real = [], jnp.linalg.qr
    monkeypatch.setattr(
        jnp.linalg,
        "qr",
        lambda a, mode="reduced": calls.append(mode) or real(a, mode=mode),
    )
    minpack._covariance_svd_body(jnp.ones((10, 2)) + jnp.eye(10, 2))
    assert calls == ["r"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "+inf", "-inf"])
def test_nan_jacobian_still_gives_inf_pcov(bad):
    J = jnp.asarray(np.array([[1.0, 0.0], [bad, 1.0], [0.0, 2.0]]))
    with pytest.warns(OptimizeWarning):
        pcov, fired = pcov_from_jacobian(J, 0.5, 3, 2, False, _covariance_svd)
    assert fired and np.all(np.isinf(pcov))


def _rank_and_pcov(kernel, J, n):
    """``(rank-deficiency warned, pcov)`` of ``J`` through ``kernel``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pcov, _ = pcov_from_jacobian(J, 0.7, n, J.shape[1], False, kernel)
    return any(issubclass(w.category, OptimizeWarning) for w in rec), pcov


def test_qr_covariance_all_zero_column_matches_svd():
    J = np.random.default_rng(3).normal(size=(100, 4))
    J[:, 2] = 0.0
    qr_warned, got = _rank_and_pcov(_covariance_svd, jnp.asarray(J), 100)
    ref_warned, ref = _rank_and_pcov(_svd_ref, jnp.asarray(J), 100)
    assert qr_warned and ref_warned  # same rank decision: rank 3 < 4
    np.testing.assert_array_equal(
        got[2], 0.0
    )  # the unidentifiable direction is dropped
    np.testing.assert_array_equal(got == 0.0, ref == 0.0)
    assert _rel(got, ref) <= 1e-12


def test_qr_covariance_trailing_zero_padded_rows_match_svd():
    J = np.random.default_rng(4).normal(size=(80, 3))
    padded = np.vstack([J, np.zeros((20, 3))])
    qr_warned, got = _rank_and_pcov(_covariance_svd, jnp.asarray(padded), 100)
    ref_warned, ref = _rank_and_pcov(_svd_ref, jnp.asarray(padded), 100)
    assert not qr_warned and not ref_warned  # padding does not change the rank
    assert _rel(got, ref) <= 1e-12
    # zero rows add nothing to J^T J: same pcov as the unpadded Jacobian at the same n
    _, unpadded = _rank_and_pcov(_covariance_svd, jnp.asarray(J), 100)
    assert _rel(got, unpadded) <= 1e-12


def test_pcov_from_normal_singular_h_keeps_moore_penrose_semantics():
    """Equilibration must not change the null space of a rank-deficient H."""
    H = np.array([[1.0, 1e-9], [1e-9, 1e-18]])
    pcov, _ = pcov_from_normal(H, 1.0, 10, 2, True, False)
    np.testing.assert_allclose(pcov, np.linalg.pinv(H), rtol=1e-6, atol=1e-30)


class TestColumnNorms:
    """``_column_norms`` replaced a host-side NumPy reduction that copied all of a GPU J."""

    def test_matches_numpy_and_does_not_overflow(self):
        import jax.numpy as jnp
        import numpy as np

        from nlsq.core.covariance import _column_norms

        rng = np.random.default_rng(0)
        J = rng.normal(size=(50, 4)) * np.array([1.0, 1e-8, 1e8, 1.0])
        J[:, 3] = 0.0  # zero column stays zero (the caller then leaves it unscaled)
        np.testing.assert_allclose(
            np.asarray(_column_norms(jnp.asarray(J))),
            np.linalg.norm(J, axis=0),
            rtol=1e-12,
        )
        big = np.full((4, 1), 1e200)  # a plain sum of squares would overflow
        np.testing.assert_allclose(
            np.asarray(_column_norms(jnp.asarray(big))), [2e200], rtol=1e-12
        )
