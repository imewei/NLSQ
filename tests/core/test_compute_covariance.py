"""``curve_fit(..., compute_covariance=False)`` returns ``pcov=None`` on every route."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import OptimizeWarning, curve_fit
from nlsq.core import minpack
from nlsq.core.least_squares import LeastSquares
from nlsq.global_optimization import CMAESConfig

x = np.linspace(0, 4, 200)
y = 2.5 * np.exp(-1.3 * x) + 0.01 * np.random.default_rng(0).standard_normal(200)
p0 = [1.0, 1.0]


def model(x, a, b):
    return a * jnp.exp(-b * x)


def test_default_unchanged():
    _, pcov = curve_fit(model, x, y, p0=p0)
    assert pcov.shape == (2, 2)


def test_skip_returns_none_and_no_svd(monkeypatch):
    ref_popt = curve_fit(model, x, y, p0=p0)[0]  # before patching
    calls = []
    real = minpack._covariance_svd_jit
    monkeypatch.setattr(
        minpack, "_covariance_svd_jit", lambda J: calls.append(1) or real(J)
    )
    res = curve_fit(model, x, y, p0=p0, compute_covariance=False)
    popt, pcov = res
    assert pcov is None and res.pcov is None and calls == []
    np.testing.assert_allclose(popt, ref_popt, rtol=1e-12)


def test_skip_full_output_5tuple():
    _, pcov, info, _, ier = curve_fit(
        model, x, y, p0=p0, full_output=True, compute_covariance=False
    )
    assert pcov is None and "fvec" in info and ier in (1, 2, 3, 4)


def test_skip_return_eval_and_timeit():
    _, pcov, feval = curve_fit(
        model, x, y, p0=p0, return_eval=True, compute_covariance=False
    )
    assert pcov is None and feval.shape == x.shape
    _, pcov, *_ = curve_fit(model, x, y, p0=p0, timeit=True, compute_covariance=False)
    assert pcov is None


def test_skip_no_covariance_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        curve_fit(model, x, y, p0=p0, compute_covariance=False)


@pytest.mark.parametrize(
    "call",
    [
        lambda r: r.confidence_intervals(),
        lambda r: r.confidence_band(),
        lambda r: r.prediction_interval(),
    ],
)
def test_dependent_members_raise(call):
    res = curve_fit(model, x, y, p0=p0, compute_covariance=False)
    with pytest.raises(AttributeError, match="compute_covariance=False"):
        call(res)


def test_summary_without_pcov():
    curve_fit(model, x, y, p0=p0, compute_covariance=False).summary()  # no exception


def test_not_forwarded_to_least_squares(monkeypatch):
    seen = {}
    orig = LeastSquares.least_squares
    monkeypatch.setattr(
        LeastSquares,
        "least_squares",
        lambda self, *a, **k: seen.update(k) or orig(self, *a, **k),
    )
    curve_fit(model, x, y, p0=p0, compute_covariance=False)
    assert "compute_covariance" not in seen


# Per-branch behaviour. Honouring routes compute no covariance and warn about none;
# rejecting routes raise ValueError naming the flag.

LB_UB = ([0, 0, 0], [10, 10, 10])
P0_RD = [1.0, 1.0, 1.0]


def rank_deficient_model(x, a, b, c):
    """``c`` is unused, so the Jacobian is rank-deficient (covariance warns)."""
    return a * jnp.exp(-b * x)


HONOURING_ROUTES = {
    "trf": {},
    "cmaes": {
        "method": "cmaes",
        "bounds": LB_UB,
        "cmaes_config": CMAESConfig(max_generations=10, seed=0),
    },
    "multi-start": {"method": "multi-start", "bounds": LB_UB, "n_starts": 3},
    "multistart": {"multistart": True, "bounds": LB_UB, "n_starts": 3},
    "global_search": {"global_search": True, "bounds": LB_UB},
    "fallback": {"fallback": True},
    "compute_diagnostics": {"compute_diagnostics": True},
    "auto_bounds": {"auto_bounds": True},
    "auto_standard": {"method": "auto"},
}


@pytest.fixture
def cov_spies(monkeypatch):
    """Count calls to the covariance SVD kernel and to ``pcov_from_jacobian``."""
    calls = {"svd": 0, "pcov": 0}
    real_svd, real_pcov = minpack._covariance_svd_jit, minpack.pcov_from_jacobian

    def svd(J):
        calls["svd"] += 1
        return real_svd(J)

    def pcov(*a, **k):
        calls["pcov"] += 1
        return real_pcov(*a, **k)

    monkeypatch.setattr(minpack, "_covariance_svd_jit", svd)
    monkeypatch.setattr(minpack, "pcov_from_jacobian", pcov)
    return calls


def _fit_recording(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = curve_fit(rank_deficient_model, x, y, p0=P0_RD, **kw)
    return out, [m for m in w if issubclass(m.category, OptimizeWarning)]


@pytest.mark.parametrize("route", list(HONOURING_ROUTES))
def test_honouring_route_skips_covariance(route, cov_spies):
    kw = HONOURING_ROUTES[route]
    # Control: the spies and the warning are live on this route with the flag on.
    _fit_recording(**kw)
    assert cov_spies["pcov"] > 0 and cov_spies["svd"] > 0
    cov_spies["svd"] = cov_spies["pcov"] = 0

    res, caught = _fit_recording(compute_covariance=False, **kw)
    assert res.pcov is None and res.popt.shape == (3,)
    assert cov_spies == {"svd": 0, "pcov": 0}
    assert not [m for m in caught if "rank-deficient" in str(m.message)]
    assert not [m for m in caught if "could not be estimated" in str(m.message)]


# multistart suppresses per-start OptimizeWarnings, so only the spy covers it.
@pytest.mark.parametrize("route", ["trf", "cmaes", "fallback"])
def test_honouring_route_warns_with_flag_on(route):
    """The rank-deficient model really does warn when the covariance is computed."""
    _, caught = _fit_recording(**HONOURING_ROUTES[route])
    assert [m for m in caught if "rank-deficient" in str(m.message)]


@pytest.mark.parametrize("route", ["cmaes", "multistart", "fallback"])
def test_honouring_route_full_output(route):
    out, _ = _fit_recording(
        full_output=True, compute_covariance=False, **HONOURING_ROUTES[route]
    )
    assert out[1] is None


def test_honouring_cmaes_without_refinement():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        method="cmaes",
        cmaes_config=CMAESConfig(max_generations=5, refine_with_nlsq=False, seed=0),
        compute_covariance=False,
    )
    assert res.pcov is None


def test_cmaes_without_refinement_logs_covariance_skipped(caplog):
    """CMA-ES drops pcov outside _final_covariance; it logs the same DEBUG line."""
    with caplog.at_level("DEBUG", logger="nlsq.core.minpack"):
        curve_fit(
            model,
            x,
            y,
            p0=p0,
            bounds=([0, 0], [10, 10]),
            method="cmaes",
            cmaes_config=CMAESConfig(max_generations=5, refine_with_nlsq=False, seed=0),
            compute_covariance=False,
        )
    assert "covariance skipped" in caplog.text


def test_default_true_is_explicit_noop():
    _, pcov = curve_fit(model, x, y, p0=p0, compute_covariance=True)
    assert pcov.shape == (2, 2)
    _, pcov = curve_fit(
        model, x, y, p0=p0, multistart=True, n_starts=2, compute_covariance=True
    )
    assert pcov.shape == (2, 2)


def _force_strategy(monkeypatch, strategy):
    from nlsq.core.workflow import MemoryBudgetSelector
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    monkeypatch.setattr(
        MemoryBudgetSelector,
        "select",
        lambda self, **k: (strategy, HybridStreamingConfig()),
    )


def test_reject_hybrid_streaming():
    with pytest.raises(ValueError, match="compute_covariance"):
        curve_fit(
            model,
            x,
            y,
            p0=p0,
            method="hybrid_streaming",
            verbose=0,
            compute_covariance=False,
        )


@pytest.mark.parametrize("strategy", ["streaming", "chunked"])
@pytest.mark.parametrize("extra", [{}, {"timeit": True}, {"return_eval": True}])
def test_reject_auto_streaming_and_chunked(monkeypatch, strategy, extra):
    _force_strategy(monkeypatch, strategy)
    with pytest.raises(ValueError, match="compute_covariance"):
        curve_fit(
            model,
            x,
            y,
            p0=p0,
            method="auto",
            verbose=0,
            compute_covariance=False,
            **extra,
        )
