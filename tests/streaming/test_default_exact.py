"""curve_fit_large routes large inputs to the exact streaming LM by default (Change 1).

Covers ``Auto.exact``, the eligibility check (engine reasons warn and fall back to
chunked, data reasons raise), the settings transfer, and scalar and signed ``sigma`` in
the exact solver.
"""

import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit, curve_fit_large
from nlsq.streaming import large_fit
from nlsq.streaming.exact_lm import ExactSolverError
from nlsq.streaming.large_fit import fit_with_strategy
from nlsq.streaming.normal_equations import FactorySource
from nlsq.streaming.specs import Auto, Chunked, Exact

EXP = PROBLEMS["exp_decay"]
N = 20_000
TH = 10_000
P0 = EXP.p0
HINT = re.escape("strategy=Chunked()")


def m(x, a, b, c):
    return EXP.model(jnp, x, a, b, c)


@pytest.fixture(scope="module")
def data():
    return EXP.data(N)


@pytest.fixture
def dispatch(monkeypatch):
    """Record which path body fit_with_strategy runs (and still run it)."""
    calls = []
    for name, tag in (
        ("_fit_in_memory", "in_memory"),
        ("_fit_chunked", "chunked"),
        ("_fit_exact", "exact"),
    ):
        real = getattr(large_fit, name)

        def wrapper(*args, _real=real, _tag=tag, **kwargs):
            calls.append(_tag)
            return _real(*args, **kwargs)

        monkeypatch.setattr(large_fit, name, wrapper)
    return calls


@pytest.fixture
def spy(monkeypatch):
    """Record every exact_lm_fit call's keyword arguments."""
    calls = []
    real = lm.exact_lm_fit

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(lm, "exact_lm_fit", wrapper)
    return calls


def _scale_aware(C, C_ref):
    d = np.sqrt(np.outer(np.diag(C_ref), np.diag(C_ref)))
    return float(np.max(np.abs(C - C_ref) / d))


def _match_curve_fit(popt, pcov, x, y, sigma_abs, absolute_sigma, **tol):
    ref_p, ref_C = curve_fit(
        m, x, y, p0=P0, sigma=sigma_abs, absolute_sigma=absolute_sigma, **tol
    )[:2]
    np.testing.assert_allclose(popt, ref_p, rtol=1e-6)
    assert _scale_aware(pcov, np.asarray(ref_C)) <= 1e-6


# --- routing ------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,expected", [(9_999, "in_memory"), (10_000, "exact"), (10_001, "exact")]
)
def test_threshold_routing(dispatch, data, n, expected):
    x, y = data
    curve_fit_large(m, x[:n], y[:n], p0=P0, size_threshold=TH)
    assert dispatch == [expected]


def test_auto_size_detection_false_runs_exact(dispatch, data):
    x, y = data
    curve_fit_large(m, x[:5_000], y[:5_000], p0=P0, auto_size_detection=False)
    assert dispatch == ["exact"]


def test_auto_exact_none_is_old_behaviour(data):
    x, y = data
    popt, pcov = curve_fit_large(
        m, x, y, p0=P0, strategy=Auto(size_threshold=TH, exact=None)
    )
    ref_p, ref_C = curve_fit_large(m, x, y, p0=P0, strategy=Chunked())
    np.testing.assert_array_equal(popt, ref_p)
    np.testing.assert_array_equal(pcov, ref_C)


# --- engine reasons: warn and fall back --------------------------------------


@pytest.mark.parametrize(
    "kw,reason",
    [
        ({"method": "trf", "size_threshold": TH}, "sets method"),
        ({"multistart": True, "size_threshold": TH}, "multistart"),
        ({"loss": "soft_l1", "size_threshold": TH}, "not supported by exact"),
        (
            {
                "strategy": Auto(
                    size_threshold=TH, chunked=Chunked(options={"loss": "soft_l1"})
                )
            },
            "not supported by exact",
        ),
    ],
)
def test_configured_chunked_falls_back(dispatch, data, kw, reason):
    x, y = data
    with pytest.warns(UserWarning, match=reason):
        curve_fit_large(m, x, y, p0=P0, **kw)
    assert dispatch == ["chunked"]


def test_engine_fallback_warning_text(data):
    x, y = data
    with pytest.warns(UserWarning, match=r"exact streaming not used") as rec:
        curve_fit_large(m, x, y, p0=P0, method="trf", size_threshold=TH)
    msg = next(str(w.message) for w in rec if "exact streaming" in str(w.message))
    assert "not full-batch equivalent" in msg
    assert "strategy=Chunked()" in msg


# --- data reasons: raise ------------------------------------------------------


def test_inf_sigma_unchecked_raises(dispatch, data):
    x, y = data
    sigma = np.ones(N)
    sigma[5] = np.inf
    with pytest.raises(ValueError, match=HINT):
        curve_fit_large(
            m, x, y, p0=P0, sigma=sigma, check_finite=False, size_threshold=TH
        )
    assert dispatch == []


def test_2d_xdata_raises(dispatch, data):
    x, y = data
    X = np.vstack([x, x])
    with pytest.raises(ValueError, match="multi-feature") as e:
        curve_fit_large(
            lambda X, a, b, c: m(X[0], a, b, c), X, y, p0=P0, size_threshold=TH
        )
    assert "strategy=Chunked()" in str(e.value)
    assert dispatch == []


def test_2d_xdata_with_explicit_chunked_runs(dispatch, data):
    x, y = data
    X = np.vstack([x, x])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        curve_fit_large(
            lambda X, a, b, c: m(X[0], a, b, c), X, y, p0=P0, strategy=Chunked()
        )
    assert dispatch == ["chunked"]
    assert not [w for w in rec if "exact streaming" in str(w.message)]


def test_non_pointwise_model_raises(dispatch, data):
    x, y = data
    with pytest.raises(ValueError, match="not pointwise") as e:
        curve_fit_large(
            lambda x, a: a * (x - jnp.mean(x)), x, y, p0=[1.0], size_threshold=TH
        )
    assert "strategy=Chunked()" in str(e.value)
    assert dispatch == []


def test_non_pointwise_beyond_probe_raises(data):
    x, y = data

    def sneaky(x, a):
        return a * x + jnp.where(x.size > 4096, jnp.mean(x), 0.0)

    with pytest.raises(ValueError, match=r"^model is not pointwise") as e:
        curve_fit_large(sneaky, x, y, p0=[1.0], size_threshold=TH)
    assert "strategy=Chunked()" in str(e.value)
    assert isinstance(e.value.__cause__, ValueError)


def test_stall_raises_exact_solver_error(data):
    x, y = data
    with pytest.raises(ExactSolverError) as e:
        curve_fit_large(m, x, y, p0=P0, max_nfev=1, size_threshold=TH)
    assert "strategy=Chunked()" in str(e.value)
    assert e.value.result["termination"] == "max_nfev"


# --- sigma --------------------------------------------------------------------


def test_zero_sigma_raises(dispatch, data):
    x, y = data
    sigma = np.ones(N)
    sigma[3] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        curve_fit_large(m, x, y, p0=P0, sigma=sigma, size_threshold=TH)


def test_2d_sigma_raises(dispatch):
    # 2-D sigma at n=2_000: a (20k, 20k) matrix would take 3.2 GB
    x, y = EXP.data(2_000)
    with pytest.raises(ValueError, match=r"shape \(2000, 2000\)"):
        curve_fit_large(m, x, y, p0=P0, sigma=np.eye(2_000), size_threshold=1_000)
    # chunked's error names the shape too
    with pytest.raises(ValueError, match=r"shape \(2000, 2000\)"):
        curve_fit_large(m, x, y, p0=P0, sigma=np.eye(2_000), strategy=Chunked())


def test_default_route_2d_sigma_error_is_neutral(dispatch):
    """The user never chose method='exact_streaming': say so neutrally, name the hint."""
    x, y = EXP.data(2_000)
    with pytest.raises(ValueError) as exc:
        curve_fit_large(m, x, y, p0=P0, sigma=np.eye(2_000), size_threshold=1_000)
    msg = str(exc.value)
    assert "exact_streaming" not in msg
    assert "sigma must be 1-D" in msg and "(2000, 2000)" in msg
    assert "strategy=Chunked()" in msg
    # the explicit route keeps its text
    with pytest.raises(
        ValueError, match=r"sigma must be 1-D for method='exact_streaming'"
    ):
        curve_fit_large(m, x, y, p0=P0, sigma=np.eye(2_000), strategy=Exact())


@pytest.mark.parametrize("absolute_sigma", [False, True])
@pytest.mark.parametrize("sigma", [2.0, -np.ones(N), -2.0], ids=["2", "neg1d", "-2"])
def test_scalar_and_signed_sigma_run_exact(dispatch, data, sigma, absolute_sigma):
    x, y = data
    popt, pcov = curve_fit_large(
        m,
        x,
        y,
        p0=P0,
        sigma=sigma,
        absolute_sigma=absolute_sigma,
        size_threshold=TH,
    )
    assert dispatch == ["exact"]
    sig_abs = np.abs(np.broadcast_to(sigma, (N,)))
    _match_curve_fit(popt, pcov, x, y, sig_abs, absolute_sigma)


def test_explicit_exact_accepts_scalar_sigma(data):
    x, y = data
    popt, pcov = curve_fit_large(m, x, y, p0=P0, sigma=2.0, method="exact_streaming")
    _match_curve_fit(popt, pcov, x, y, np.full(N, 2.0), False)


def _full_jac(x, theta, sigma_abs):
    J = np.asarray(jax.jacfwd(lambda q: m(jnp.asarray(x), *q))(jnp.asarray(theta)))
    return J / sigma_abs[:, None]


@pytest.mark.slow
@pytest.mark.parametrize("sigma", [2.0, "signed"])
def test_sigma_variants_padded_tail_and_factory(sigma):
    n, chunk = 3001, 700  # 3001 = 4 * 700 + 201: padded tail
    x, y = EXP.data(n)
    if sigma == "signed":
        rng = np.random.default_rng(1)
        sig = (0.5 + rng.random(n)) * np.where(np.arange(n) % 2, -1.0, 1.0)
        sig_abs = np.abs(sig)
    else:
        sig, sig_abs = sigma, np.full(n, sigma)

    def block(s, e):
        return sig if np.ndim(sig) == 0 else sig[s:e]

    def factory(c):
        for s in range(0, n, c):
            e = min(s + c, n)
            yield x[s:e], y[s:e], block(s, e)

    # both solvers run to 1e-12 so the comparison is of the optimum, not the stopping rule
    tol = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    exact = {"method": "exact_streaming", "chunk_size": chunk, **tol}
    out_arr = curve_fit_large(m, x, y, p0=P0, sigma=sig, **exact)
    out_fac = curve_fit_large(m, factory, None, p0=P0, **exact)
    for popt, pcov in (out_arr, out_fac):
        _match_curve_fit(popt, pcov, x, y, sig_abs, False, **tol)

    popt = out_arr[0]
    s_ref = np.linalg.svd(_full_jac(x, popt, sig_abs), compute_uv=False)
    src, weighted = lm._open_source(x, y, np.asarray(sig), chunk, 3, None, None)
    assert weighted
    for source in (src, FactorySource(factory, chunk, n)):
        R = lm.tsqr_r_factor(m, popt, source, chunk, weighted=True)
        np.testing.assert_allclose(
            np.linalg.svd(R, compute_uv=False), s_ref, rtol=1e-10
        )


def test_valid_sigma_longer_than_probe_runs_exact(dispatch, data):
    x, y = data
    sigma = 0.5 + np.random.default_rng(2).random(N)
    curve_fit_large(m, x, y, p0=P0, sigma=sigma, size_threshold=TH)
    assert dispatch == ["exact"]


def test_absolute_sigma_pcov_matches_curve_fit(data):
    x, y = data
    sigma = 0.5 + np.random.default_rng(3).random(N)
    popt, pcov = curve_fit_large(
        m, x, y, p0=P0, sigma=sigma, absolute_sigma=True, size_threshold=TH
    )
    _match_curve_fit(popt, pcov, x, y, sigma, True)


# --- settings transfer --------------------------------------------------------


def test_auto_chunked_settings_reach_exact(spy, data):
    x, y = data
    curve_fit_large(
        m,
        x,
        y,
        p0=P0,
        strategy=Auto(
            size_threshold=TH,
            chunked=Chunked(chunk_size=500, options={"ftol": 1e-10}),
        ),
    )
    (kw,) = spy
    assert kw["chunk_size"] == 500
    assert kw["ftol"] == 1e-10


def test_explicit_auto_exact_fields_win(spy, data):
    x, y = data
    curve_fit_large(
        m,
        x,
        y,
        p0=P0,
        strategy=Auto(
            size_threshold=TH,
            exact=Exact(ftol=1e-11),
            chunked=Chunked(options={"ftol": 1e-10}),
        ),
    )
    (kw,) = spy
    assert kw["ftol"] == 1e-11


def _fws(strategy, x, y, **kwargs):
    return fit_with_strategy(
        strategy,
        m,
        x,
        y,
        P0,
        sigma=None,
        absolute_sigma=False,
        bounds=(-np.inf, np.inf),
        check_finite=True,
        memory_limit_gb=None,
        show_progress=False,
        **kwargs,
    )


@pytest.mark.slow
def test_fit_with_strategy_call_kwargs_reach_exact(spy, dispatch, data):
    x, y = data
    _fws(Auto(size_threshold=TH), x, y, ftol=1e-10)
    assert dispatch == ["exact"]
    assert spy[0]["ftol"] == 1e-10
    with pytest.warns(UserWarning, match="not supported by exact"):
        _fws(Auto(size_threshold=TH), x, y, loss="soft_l1")
    assert dispatch == ["exact", "chunked"]


def test_ignored_options_do_not_raise(dispatch, data):
    x, y = data
    curve_fit_large(
        m,
        x,
        y,
        p0=P0,
        size_threshold=TH,
        stability="auto",
        rescale_data=False,
        max_jacobian_elements_for_svd=10,
    )
    assert dispatch == ["exact"]


def test_tolerances_reach_exact(spy, data):
    x, y = data
    curve_fit_large(m, x, y, p0=P0, size_threshold=TH, ftol=1e-10, chunk_size=500)
    (kw,) = spy
    assert kw["ftol"] == 1e-10
    assert kw["chunk_size"] == 500


def test_compute_covariance_false_reaches_exact(dispatch, data):
    x, y = data
    _popt, pcov = curve_fit_large(
        m, x, y, p0=P0, size_threshold=TH, compute_covariance=False
    )
    assert dispatch == ["exact"]
    assert pcov is None


# --- review focus -------------------------------------------------------------


def test_nan_y_raises_value_error(data):
    x, y = data
    y2 = y.copy()
    y2[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        curve_fit_large(m, x, y2, p0=P0, size_threshold=TH)


def test_p0_none_fits(dispatch):
    x, y = EXP.data(N, noise=1e-3)
    popt, _ = curve_fit_large(m, x, y, size_threshold=TH)
    assert dispatch == ["exact"]
    np.testing.assert_allclose(popt, EXP.p_true, rtol=1e-3)


# x up to 2e4 with b = 1e-4 is ill-conditioned: the TSQR-route warning is expected
@pytest.mark.filterwarnings("ignore:normal-equation Hessian:UserWarning")
def test_docstring_good_model_fits(dispatch):
    def good_model(xdata, a, b):
        return a * jnp.exp(-b * xdata.astype(jnp.int32))

    x = np.arange(N, dtype=float)
    y = 2.0 * np.exp(-1e-4 * x)
    popt, _ = curve_fit_large(good_model, x, y, p0=[1.0, 5e-5], size_threshold=TH)
    assert dispatch == ["exact"]
    np.testing.assert_allclose(popt, [2.0, 1e-4], rtol=1e-6)


def test_legacy_options_reach_the_default_exact():
    """The shim leaves ``Auto.exact`` at its default; the effective Exact inherits."""
    from nlsq import _legacy_to_strategy
    from nlsq.streaming.large_fit import _effective_exact

    auto = _legacy_to_strategy(ftol=1e-10, chunk_size=500)
    assert isinstance(auto, Auto)
    assert auto.exact == Exact()
    assert _effective_exact(auto.exact, auto, {"ftol": 1e-10}) == Exact(
        chunk_size=500, ftol=1e-10
    )


def test_chunked_path_logs_when_it_drops_pcov(data, caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger="nlsq.streaming.large_fit")
    x, y = data
    _, pcov = curve_fit_large(
        m, x, y, p0=P0, strategy=Chunked(chunk_size=5_000), compute_covariance=False
    )
    assert pcov is None
    assert [r for r in caplog.records if "compute_covariance=False" in r.getMessage()]
