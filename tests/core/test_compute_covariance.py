"""``curve_fit(..., compute_covariance=False)`` returns ``pcov=None`` on every route."""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
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


# Per-branch behaviour: every branch honours the flag (pcov is None).


def _assert_skipped(res):
    assert res.pcov is None
    assert res.popt.shape == (2,)
    np.testing.assert_allclose(res.popt, [2.5, 1.3], rtol=0.1)


def test_branch_cmaes():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        method="cmaes",
        cmaes_config=CMAESConfig(max_generations=20, seed=0),
        compute_covariance=False,
    )
    _assert_skipped(res)


def test_branch_cmaes_full_output():
    out = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        method="cmaes",
        cmaes_config=CMAESConfig(max_generations=20, seed=0),
        full_output=True,
        compute_covariance=False,
    )
    assert out[1] is None


def test_branch_multi_start_method():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        method="multi-start",
        n_starts=3,
        compute_covariance=False,
    )
    _assert_skipped(res)


def test_branch_multistart_flag():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        multistart=True,
        n_starts=3,
        compute_covariance=False,
    )
    _assert_skipped(res)


def test_branch_multistart_full_output():
    out = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        multistart=True,
        n_starts=3,
        full_output=True,
        compute_covariance=False,
    )
    assert out[1] is None


def test_branch_global_search():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=([0, 0], [10, 10]),
        global_search=True,
        compute_covariance=False,
    )
    _assert_skipped(res)


def test_branch_hybrid_streaming():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        method="hybrid_streaming",
        verbose=0,
        compute_covariance=False,
    )
    assert res.pcov is None and "perr" not in res
    res2 = curve_fit(
        model,
        x,
        y,
        p0=p0,
        method="hybrid_streaming",
        verbose=0,
        full_output=True,
        compute_covariance=False,
    )
    assert res2[1] is None


def test_branch_fallback():
    res = curve_fit(model, x, y, p0=p0, fallback=True, compute_covariance=False)
    _assert_skipped(res)
    out = curve_fit(
        model, x, y, p0=p0, fallback=True, full_output=True, compute_covariance=False
    )
    assert out[1] is None


def test_branch_compute_diagnostics():
    res = curve_fit(
        model, x, y, p0=p0, compute_diagnostics=True, compute_covariance=False
    )
    _assert_skipped(res)
    assert res.diagnostics is not None


def test_branch_auto_bounds():
    res = curve_fit(model, x, y, p0=p0, auto_bounds=True, compute_covariance=False)
    _assert_skipped(res)


def test_branch_auto_standard():
    res = curve_fit(model, x, y, p0=p0, method="auto", compute_covariance=False)
    _assert_skipped(res)


def test_branch_auto_chunked():
    res = curve_fit(
        model,
        x,
        y,
        p0=p0,
        method="auto",
        memory_limit_gb=0.0001,
        verbose=0,
        compute_covariance=False,
    )
    assert res.pcov is None


def test_branch_auto_streaming(monkeypatch):
    from nlsq.core.workflow import MemoryBudgetSelector
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    monkeypatch.setattr(
        MemoryBudgetSelector,
        "select",
        lambda self, **k: ("streaming", HybridStreamingConfig()),
    )
    res = curve_fit(
        model, x, y, p0=p0, method="auto", verbose=0, compute_covariance=False
    )
    assert res.pcov is None


def test_default_true_is_explicit_noop():
    _, pcov = curve_fit(model, x, y, p0=p0, compute_covariance=True)
    assert pcov.shape == (2, 2)
    _, pcov = curve_fit(
        model, x, y, p0=p0, multistart=True, n_starts=2, compute_covariance=True
    )
    assert pcov.shape == (2, 2)
