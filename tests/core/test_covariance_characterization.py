"""Characterization tests for the four covariance sites.

These pin the CURRENT behavior of the four places that turn a fit into ``pcov`` so
a later refactor that routes them through shared helpers can prove it preserved
behavior:

* standard: ``CurveFit._compute_covariance`` (SVD of J, ``nlsq/core/minpack.py``)
* exact:    ``exact_lm.compute_covariance`` (eigh of H = J^T J)
* hybrid:   ``AdaptiveHybridStreamingOptimizer._run_phase3_finalize`` (bare ``pinv``)
* GLS:      ``LargeDatasetFitter._compute_covariance_from_history``

Golden constants were captured once from the unmodified code (see ``_problems``
below for the exact inputs) and pasted as ``repr`` output. Do not regenerate them
to make a failing test pass: a mismatch means behavior changed. Goldens use
``rtol=1e-8, atol=0`` (stricter than rtol 1e-8 / atol 1e-10: every entry is
~1e-5..1, so an absolute floor of 1e-10 would hide real changes); the two
end-to-end fits use rtol 1e-6 because they pass through an iterative optimizer.

Several of these behaviors are deliberate differences between sites and are
preserved on purpose; the docstrings say so.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import curve_fit
from nlsq.core.minpack import CurveFit
from nlsq.result import OptimizeWarning
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.exact_lm import compute_covariance, exact_lm_fit
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import LargeDatasetFitter

RTOL = 1e-8
RTOL_E2E = 1e-6
N = 12
X = np.linspace(0.0, 3.0, N)
Y = 2.0 * np.exp(-1.3 * X) + 0.01 * np.sin(7.0 * X + 1.0)

# Well-conditioned problem: a*exp(-b*x) at (2.0, 1.3).
P_WC = np.array([2.0, 1.3])
COST_WC = 0.0002887827887097562  # 0.5 * sum(r^2) at P_WC
# Partially rank-deficient problem: (a + b)*exp(-c*x) at (1, 1, 1.3); columns of a, b
# of the Jacobian are identical, so direction (1, -1, 0) is in the null space.
P_DUP = np.array([1.0, 1.0, 1.3])
COST_DUP = 0.0002887827887097562
NULL_DIR = np.array([1.0, -1.0, 0.0])

# --- goldens (captured from the unmodified code) --------------------------------------
STD_E2E_POPT = [2.0059225161952785, 1.3042978115396613]
STD_E2E_PCOV = [
    [4.044686860201271e-05, 2.5477667066871243e-05],
    [2.5477667066871247e-05, 4.840830625599563e-05],
]
STD_E2E_PCOV_ABS = [
    [0.7616734798361435, 0.47978160000165926],
    [0.4797816000016593, 0.9115989532288085],
]
EX_E2E_X = [2.005921963939399, 1.3042967370729526]
EX_E2E_COST = 0.00026551317440889183
EX_E2E_PCOV = [
    [4.044685444582692e-05, 2.547764706101839e-05],
    [2.5477647061018383e-05, 4.840822653662568e-05],
]
EX_E2E_PCOV_ABS = [
    [0.7616732114305282, 0.4797812221133454],
    [0.4797812221133453, 0.9115974498138598],
]
# Fixed J / H at P_WC (direct helper calls)
STD_WC = [
    [4.3929509205406857e-05, 2.7674464192993217e-05],
    [2.7674464192993217e-05, 5.2498473011368475e-05],
]
STD_WC_ABS = [
    [0.7605977731858288, 0.4791570909859121],
    [0.4791570909859121, 0.9089612515677405],
]
EX_WC = [
    [4.392950920540686e-05, 2.7674464192993234e-05],
    [2.767446419299324e-05, 5.249847301136849e-05],
]
EX_WC_ABS = [
    [0.7605977731858289, 0.47915709098591236],
    [0.47915709098591247, 0.9089612515677408],
]
# Fixed J / H at P_DUP (rank 2 of 3)
STD_DUP = [
    [1.2202641445946341e-05, 1.220264144594635e-05, 1.5374702329440667e-05],
    [1.220264144594635e-05, 1.2202641445946361e-05, 1.537470232944069e-05],
    [1.5374702329440667e-05, 1.537470232944069e-05, 5.833163667929832e-05],
]
STD_DUP_ABS = [
    [0.19014944329645708, 0.19014944329645722, 0.2395785454929559],
    [0.19014944329645722, 0.1901494432964574, 0.23957854549295626],
    [0.2395785454929559, 0.23957854549295626, 0.9089612515677408],
]
EX_DUP = [
    [1.2202641445946358e-05, 1.2202641445946356e-05, 1.5374702329440694e-05],
    [1.2202641445946356e-05, 1.2202641445946351e-05, 1.537470232944068e-05],
    [1.5374702329440694e-05, 1.5374702329440677e-05, 5.833163667929832e-05],
]
EX_DUP_ABS = [
    [0.19014944329645733, 0.1901494432964573, 0.23957854549295635],
    [0.1901494432964573, 0.19014944329645725, 0.2395785454929561],
    [0.23957854549295635, 0.23957854549295604, 0.9089612515677408],
]
# Hybrid: J^T J is fed in "normalized" space with D = diag(2.0, 1.3) (WC) / diag(1, 1, 1.3).
HYB_WC = [
    [0.00017571803682162753, 7.195360690178243e-05],
    [7.195360690178243e-05, 8.872241938921277e-05],
]
HYB_WC_PERR = [0.013255868014642706, 0.009419257900132726]
HYB_DUP = [
    [1.220264144594635e-05, 1.2202641445946353e-05, 1.9987113028272875e-05],
    [1.2202641445946353e-05, 1.2202641445946356e-05, 1.9987113028272885e-05],
    [1.9987113028272875e-05, 1.9987113028272885e-05, 9.858046598801415e-05],
]
# GLS: three chunk fits accumulated through _update_parameters_convergence.
GLS_THETAS = [np.array([2.01, 1.29]), np.array([1.99, 1.31]), np.array([2.0, 1.3])]
GLS_PCOVS = [
    np.array([[4e-2, 2e-2], [2e-2, 5e-2]]),
    np.array([[3e-2, 1e-2], [1e-2, 6e-2]]),
    np.array([[5e-2, 0.0], [0.0, 4e-2]]),
]
GLS_SSRS = [1.0e-4, 1.5e-4, 0.8e-4]
GLS_INFO = [
    [86.54411764705883, -18.38235294117647],
    [-18.38235294117647, 67.64705882352942],
]
GLS_REL = [
    [1.794656856022981e-05, 4.876784934845057e-06],
    [4.876784934845057e-06, 2.295990347325053e-05],
]
GLS_ABS = [
    [0.01226257914028657, 0.0033322225924691763],
    [0.0033322225924691763, 0.015688103965344882],
]


# --- helpers ---------------------------------------------------------------------------
def model(x, a, b):
    return a * jnp.exp(-b * x)


def model_dup(x, a, b, c):
    return (a + b) * jnp.exp(-c * x)


def _jac(m, p):
    return np.asarray(jax.jacfwd(lambda q: m(jnp.asarray(X), *q))(jnp.asarray(p)))


J_WC = _jac(model, P_WC)
J_DUP = _jac(model_dup, P_DUP)


def _record(fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, [warning, ...])`` with every warning recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, caught


def _assert_golden(actual, golden, rtol=RTOL):
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(golden), rtol=rtol, atol=0
    )


def _assert_no_warnings(caught):
    assert [str(w.message) for w in caught] == []


def _assert_single(caught, category):
    """Exactly one warning, of exactly ``category`` (OptimizeWarning vs UserWarning)."""
    assert [w.category for w in caught] == [category], [str(w.message) for w in caught]


def _assert_all_inf(pcov, p):
    assert pcov.shape == (p, p)
    assert np.all(np.isposinf(pcov))


def _assert_rank_deficient_pinv(pcov):
    """Finite, symmetric, rank 2 of 3, and the dropped direction is exactly zeroed."""
    pcov = np.asarray(pcov)
    scale = np.abs(pcov).max()
    assert np.isfinite(pcov).all()
    np.testing.assert_allclose(pcov, pcov.T, rtol=1e-12, atol=0)
    # A zeroed dropped direction annihilates the null vector (finite != zeroed).
    assert np.abs(pcov @ NULL_DIR).max() <= 1e-12 * scale
    eig = np.linalg.eigvalsh((pcov + pcov.T) / 2)
    assert int(np.sum(eig > 1e-9 * eig.max())) == 2
    assert eig.min() >= -1e-12 * scale  # pseudo-inverse is PSD


def _std_cov(jac, cost, ysize, p0, absolute_sigma):
    res = SimpleNamespace(jac=jnp.asarray(jac), cost=cost)
    return CurveFit()._compute_covariance(res, ysize, np.asarray(p0), absolute_sigma)


def _hybrid(m, p0):
    cfg = HybridStreamingConfig(chunk_size=5, verbose=0, enable_checkpoints=False)
    opt = AdaptiveHybridStreamingOptimizer(config=cfg)
    opt._setup_normalization(m, jnp.asarray(p0), None)
    return opt, opt.normalizer.normalize(jnp.asarray(p0))


@pytest.fixture(scope="module")
def hybrid_wc():
    return _hybrid(model, P_WC)


@pytest.fixture(scope="module")
def hybrid_dup():
    return _hybrid(model_dup, P_DUP)


def _gls_fitter(absolute_sigma):
    fit = LargeDatasetFitter(memory_limit_gb=1.0)
    fit._absolute_sigma = absolute_sigma
    for theta, pcov, ssr in zip(GLS_THETAS, GLS_PCOVS, GLS_SSRS, strict=True):
        fit._update_parameters_convergence(
            None, theta, pcov, [], 1.0, 0, 3, chunk_ssr=ssr, chunk_n=4
        )
    return fit


# --- standard site ---------------------------------------------------------------------
def test_standard_golden():
    for absolute_sigma, golden in ((False, STD_WC), (True, STD_WC_ABS)):
        (pcov, warn_cov), caught = _record(
            _std_cov, J_WC, COST_WC, N, P_WC, absolute_sigma
        )
        assert warn_cov is False
        _assert_no_warnings(caught)
        _assert_golden(pcov, golden)

    # End-to-end through curve_fit (iterative: looser rtol).
    for absolute_sigma, golden in ((False, STD_E2E_PCOV), (True, STD_E2E_PCOV_ABS)):
        (popt, pcov), caught = _record(
            curve_fit, model, X, Y, p0=[1.5, 1.0], absolute_sigma=absolute_sigma
        )
        _assert_no_warnings(caught)
        _assert_golden(popt, STD_E2E_POPT, RTOL_E2E)
        _assert_golden(pcov, golden, RTOL_E2E)


def test_standard_partial_rank_deficient_contract():
    """Duplicate column: finite pseudo-inverse, dropped direction zeroed, no warning."""
    for absolute_sigma, golden in ((False, STD_DUP), (True, STD_DUP_ABS)):
        (pcov, warn_cov), caught = _record(
            _std_cov, J_DUP, COST_DUP, N, P_DUP, absolute_sigma
        )
        assert warn_cov is False
        _assert_no_warnings(caught)
        _assert_rank_deficient_pinv(pcov)
        _assert_golden(pcov, golden)


def test_standard_fully_degenerate_contract():
    """All-zero J, non-finite J, or n <= p without absolute_sigma: all-inf + one warning."""
    cases = {
        "zero_jacobian": (np.zeros((N, 3)), COST_DUP, N, P_DUP, False),
        "zero_jacobian_absolute": (np.zeros((N, 3)), COST_DUP, N, P_DUP, True),
        "nan_jacobian": (
            np.where(np.eye(N, 2) > 0, np.nan, J_WC),
            COST_WC,
            N,
            P_WC,
            False,
        ),
        "n_equals_p": (J_WC, COST_WC, 2, P_WC, False),
    }
    for name, args in cases.items():
        (pcov, warn_cov), caught = _record(_std_cov, *args)
        assert warn_cov is True, name
        _assert_single(caught, OptimizeWarning)
        _assert_all_inf(pcov, args[0].shape[1])

    # n <= p WITH absolute_sigma is a legitimate finite covariance and does not warn.
    (pcov, warn_cov), caught = _record(_std_cov, J_WC, COST_WC, 2, P_WC, True)
    assert warn_cov is False
    _assert_no_warnings(caught)
    _assert_golden(pcov, STD_WC_ABS)


def test_standard_nonfinite_cost_gives_nan_no_warning():
    """Differs from exact on purpose: finite J + NaN cost -> NaN pcov, NO warning."""
    (pcov, warn_cov), caught = _record(_std_cov, J_WC, float("nan"), N, P_WC, False)
    assert warn_cov is False
    _assert_no_warnings(caught)
    assert pcov.shape == (2, 2)
    assert np.isnan(pcov).all()
    # absolute_sigma never reads the cost, so the same input is finite.
    (pcov, _), caught = _record(_std_cov, J_WC, float("nan"), N, P_WC, True)
    _assert_no_warnings(caught)
    _assert_golden(pcov, STD_WC_ABS)


# --- exact site ------------------------------------------------------------------------
def test_exact_golden():
    H = J_WC.T @ J_WC
    for absolute_sigma, golden in ((False, EX_WC), (True, EX_WC_ABS)):
        pcov, caught = _record(compute_covariance, H, COST_WC, N, 2, absolute_sigma)
        assert isinstance(pcov, np.ndarray)
        _assert_no_warnings(caught)
        _assert_golden(pcov, golden)

    # End-to-end through exact_lm_fit (iterative: looser rtol).
    for absolute_sigma, golden in ((False, EX_E2E_PCOV), (True, EX_E2E_PCOV_ABS)):
        res, caught = _record(
            exact_lm_fit,
            model,
            X,
            Y,
            np.array([1.5, 1.0]),
            absolute_sigma=absolute_sigma,
        )
        _assert_no_warnings(caught)
        _assert_golden(res.x, EX_E2E_X, RTOL_E2E)
        _assert_golden(res.cost, EX_E2E_COST, RTOL_E2E)
        _assert_golden(res.pcov, golden, RTOL_E2E)


def test_standard_and_exact_agree_when_well_conditioned():
    """The two sites are numerically the same estimator on a well-conditioned problem."""
    H = J_WC.T @ J_WC
    for absolute_sigma in (False, True):
        std, _ = _std_cov(J_WC, COST_WC, N, P_WC, absolute_sigma)
        ex = compute_covariance(H, COST_WC, N, 2, absolute_sigma)
        np.testing.assert_allclose(std, ex, rtol=1e-9, atol=0)


def test_exact_partial_rank_deficient_contract():
    """Duplicate column: finite pseudo-inverse, dropped direction zeroed.

    Unlike standard it also emits exactly one UserWarning (ill-conditioned H), and
    that warning is not an OptimizeWarning.
    """
    H = J_DUP.T @ J_DUP
    for absolute_sigma, golden in ((False, EX_DUP), (True, EX_DUP_ABS)):
        pcov, caught = _record(compute_covariance, H, COST_DUP, N, 3, absolute_sigma)
        _assert_single(caught, UserWarning)
        assert "ill-conditioned" in str(caught[0].message)
        _assert_rank_deficient_pinv(pcov)
        _assert_golden(pcov, golden)


def test_exact_ill_conditioned_warning_threshold():
    """UserWarning iff sqrt(cond(H)) > 1e4 (or H is numerically singular); pcov finite."""
    pcov, caught = _record(compute_covariance, np.diag([1.0, 1e-6]), 1.0, N, 2, False)
    _assert_no_warnings(caught)  # sqrt(cond) = 1e3
    np.testing.assert_allclose(pcov, np.diag([1.0, 1e6]) * (2.0 / (N - 2)), rtol=RTOL)

    pcov, caught = _record(compute_covariance, np.diag([1.0, 1e-10]), 1.0, N, 2, False)
    _assert_single(caught, UserWarning)  # sqrt(cond) = 1e5
    np.testing.assert_allclose(pcov, np.diag([1.0, 1e10]) * (2.0 / (N - 2)), rtol=RTOL)


def test_exact_fully_degenerate_contract():
    """Zero H, or n <= p without absolute_sigma: all-inf + exactly one OptimizeWarning."""
    H = J_WC.T @ J_WC
    cases = {
        "zero_hessian": (np.zeros((3, 3)), 1.0, N, 3, False),
        "zero_hessian_absolute": (np.zeros((3, 3)), 1.0, N, 3, True),
        "n_equals_p": (H, COST_WC, 2, 2, False),
    }
    for args in cases.values():
        pcov, caught = _record(compute_covariance, *args)
        _assert_single(caught, OptimizeWarning)
        _assert_all_inf(pcov, args[3])

    # n <= p WITH absolute_sigma is a legitimate finite covariance and does not warn.
    pcov, caught = _record(compute_covariance, H, COST_WC, 2, 2, True)
    _assert_no_warnings(caught)
    _assert_golden(pcov, EX_WC_ABS)


def test_exact_nonfinite_input_gives_inf_and_warns():
    """Differs from standard on purpose: any non-finite H or final pcov -> all-inf + warn."""
    H = J_WC.T @ J_WC
    bad_h = {
        "nan_hessian": np.array([[np.nan, 0.0], [0.0, 1.0]]),
        "inf_hessian": np.array([[np.inf, 0.0], [0.0, 1.0]]),
    }
    for Hbad in bad_h.values():
        pcov, caught = _record(compute_covariance, Hbad, 1.0, N, 2, False)
        _assert_single(caught, OptimizeWarning)
        _assert_all_inf(pcov, 2)

    # NaN cost poisons the scaled pcov -> all-inf (standard returns NaN, silently).
    pcov, caught = _record(compute_covariance, H, float("nan"), N, 2, False)
    _assert_single(caught, OptimizeWarning)
    _assert_all_inf(pcov, 2)
    # absolute_sigma never reads the cost, so the same input is finite.
    pcov, caught = _record(compute_covariance, H, float("nan"), N, 2, True)
    _assert_no_warnings(caught)
    _assert_golden(pcov, EX_WC_ABS)


# --- hybrid site -----------------------------------------------------------------------
def test_hybrid_golden(hybrid_wc):
    opt, p_norm = hybrid_wc
    r3, caught = _record(
        opt._run_phase3_finalize, p_norm, jnp.asarray(J_WC.T @ J_WC), 2 * COST_WC, N
    )
    _assert_no_warnings(caught)
    _assert_golden(r3["pcov"], HYB_WC)
    _assert_golden(r3["perr"], HYB_WC_PERR)
    _assert_golden(r3["popt"], P_WC)
    _assert_golden(r3["sigma_sq"], 2 * COST_WC / (N - 2))
    assert np.asarray(r3["pcov"]).dtype == np.float64


def test_hybrid_partial_rank_deficient_contract(hybrid_dup):
    """Bare pinv: finite, dropped direction zeroed, and (unlike exact) no warning."""
    opt, p_norm = hybrid_dup
    r3, caught = _record(
        opt._run_phase3_finalize,
        p_norm,
        jnp.asarray(J_DUP.T @ J_DUP),
        2 * COST_DUP,
        N,
    )
    _assert_no_warnings(caught)
    pcov = np.asarray(r3["pcov"])
    _assert_rank_deficient_pinv(pcov)
    _assert_golden(pcov, HYB_DUP)


def test_hybrid_fully_degenerate_contract(hybrid_wc):
    """Pinned deliberately (not endorsed): hybrid never returns all-inf for a bad JTJ.

    * all-zero J^T J -> finite all-ZERO pcov ("perfectly determined"), no warning
    * non-finite J^T J -> all-NaN pcov, no warning
    * n <= p -> all-inf pcov via the residual-variance step, no warning
    """
    opt, p_norm = hybrid_wc
    zero, caught = _record(
        opt._run_phase3_finalize, p_norm, jnp.zeros((2, 2)), 2 * COST_WC, N
    )
    _assert_no_warnings(caught)
    np.testing.assert_array_equal(np.asarray(zero["pcov"]), np.zeros((2, 2)))

    nan, caught = _record(
        opt._run_phase3_finalize, p_norm, jnp.full((2, 2), jnp.nan), 2 * COST_WC, N
    )
    _assert_no_warnings(caught)
    assert np.isnan(np.asarray(nan["pcov"])).all()

    few, caught = _record(
        opt._run_phase3_finalize, p_norm, jnp.asarray(J_WC.T @ J_WC), 2 * COST_WC, 2
    )
    _assert_no_warnings(caught)
    _assert_all_inf(np.asarray(few["pcov"]), 2)


# --- GLS site --------------------------------------------------------------------------
def test_gls_golden():
    fit = _gls_fitter(absolute_sigma=False)
    _assert_golden(fit._accum_information, GLS_INFO)
    pcov, caught = _record(
        fit._compute_covariance_from_history, [np.array([2.0, 1.3])], GLS_THETAS[-1]
    )
    _assert_no_warnings(caught)
    _assert_golden(pcov, GLS_REL)

    fit_abs = _gls_fitter(absolute_sigma=True)
    pcov, caught = _record(
        fit_abs._compute_covariance_from_history,
        [np.array([2.0, 1.3])],
        GLS_THETAS[-1],
    )
    _assert_no_warnings(caught)
    _assert_golden(pcov, GLS_ABS)


def test_gls_degenerate_contract():
    """n <= p -> all-inf via logger only (no warning); bad information -> history proxy."""
    fit = _gls_fitter(absolute_sigma=False)
    fit._accum_n = 2
    pcov, caught = _record(
        fit._compute_covariance_from_history, [np.array([2.0, 1.3])], GLS_THETAS[-1]
    )
    _assert_no_warnings(caught)
    _assert_all_inf(pcov, 2)

    # Ill-conditioned information (cond 1e13 > 1e12) falls back to the covariance of
    # the last parameter estimates, not an inverse.
    ill = LargeDatasetFitter(memory_limit_gb=1.0)
    ill._accum_information = np.diag([1.0, 1e-13])
    ill._accum_info_vector = np.array([1.0, 1e-13])
    ill._accum_n = 10
    ill._absolute_sigma = True
    history = [np.array([1.0, 2.0]), np.array([1.1, 2.2]), np.array([0.9, 1.8])]
    pcov, caught = _record(ill._compute_covariance_from_history, history, history[-1])
    _assert_no_warnings(caught)
    _assert_golden(pcov, np.cov(np.asarray(history).T))
    _assert_golden(pcov, [[0.01, 0.02], [0.02, 0.04]])

    # No usable information and no history: diagonal 1% proxy.
    none = LargeDatasetFitter(memory_limit_gb=1.0)
    pcov, caught = _record(
        none._compute_covariance_from_history, [], np.array([2.0, 1.3])
    )
    _assert_no_warnings(caught)
    _assert_golden(pcov, [[0.021, 0.0], [0.0, 0.014]])


def test_package_import_enables_float64():
    """The goldens above assume float64 (importing nlsq enables x64)."""
    assert nlsq is not None
    assert jnp.asarray(1.0).dtype == jnp.float64
