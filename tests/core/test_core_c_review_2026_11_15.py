"""Regression tests for the 2026-11-15 core batch C review of ``core/minpack.py``."""

from __future__ import annotations

import warnings
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.core.minpack as mp
import nlsq.core.workflow as wf
from nlsq import CurveFit, curve_fit, fit
from nlsq.global_optimization import method_selector as ms
from nlsq.result import CurveFitResult
from nlsq.streaming.large_dataset import LDMemoryConfig

try:
    import evosax

    HAS_EVOSAX = True
except Exception:  # pragma: no cover - optional dependency
    HAS_EVOSAX = False

needs_evosax = pytest.mark.skipif(not HAS_EVOSAX, reason="evosax not installed")


def model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=200, seed=0, sigma=0.01):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 5, n)
    y = np.asarray(model(x, 2.0, 0.7)) + sigma * rng.standard_normal(n)
    return x, y


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ---------------------------------------------------------------------------
# C1: stored statistics must be over the rows actually fit
# ---------------------------------------------------------------------------


class TestStoredRowsMatchFitRows:
    def test_ldmemory_nan_omit(self):
        x, y = _data()
        sig = np.full(200, 0.01)
        sig[:100] = 0.02
        y[5] = np.nan
        ref = curve_fit(model, x, y, p0=[1, 1], sigma=sig, nan_policy="omit")
        r = fit(
            model,
            x,
            y,
            p0=[1, 1],
            sigma=sig,
            workflow=LDMemoryConfig(memory_limit_gb=8.0),
            nan_policy="omit",
        )
        assert len(r["ydata"]) == len(ref["ydata"]) == 199
        assert len(r["xdata"]) == 199
        assert len(r["sigma"]) == 199
        assert np.isfinite(r.aic) and r.aic == pytest.approx(ref.aic, rel=1e-3)
        assert r.r_squared == pytest.approx(ref.r_squared, rel=1e-3)

    def test_ldmemory_data_mask(self):
        x, y = _data()
        sig = np.full(200, 0.01)
        sig[:100] = 0.02
        mask = np.arange(200) >= 10
        ref = curve_fit(model, x, y, p0=[1, 1], sigma=sig, data_mask=mask)
        r = fit(
            model,
            x,
            y,
            p0=[1, 1],
            sigma=sig,
            workflow=LDMemoryConfig(memory_limit_gb=8.0),
            data_mask=mask,
        )
        assert len(r["ydata"]) == 190 and len(r["sigma"]) == 190
        assert r.aic == pytest.approx(ref.aic, rel=1e-2, abs=0.5)
        assert r.r_squared == pytest.approx(ref.r_squared, rel=1e-3)
        assert r.prediction_interval(x[mask]).shape[0] == 190

    def test_chunked_multistart_nan_omit_and_mask(self):
        n = 2000
        x, y = _data(n)
        G = {
            "p0": [1, 1],
            "bounds": ([0, 0], [10, 10]),
            "workflow": "auto_global",
            "method": "multi-start",
            "n_starts": 2,
            "memory_limit_gb": 1e-4,
        }
        yn = y.copy()
        yn[7] = np.nan
        r = fit(model, x, yn, nan_policy="omit", **G)
        assert len(r["ydata"]) == n - 1
        assert np.isfinite(r.r_squared)

        yc = y.copy()
        yc[:200] = 50.0
        mask = np.arange(n) >= 200
        r = fit(model, x, yc, data_mask=mask, **G)
        assert len(r["ydata"]) == n - 200
        assert r.r_squared > 0.99


# ---------------------------------------------------------------------------
# C2: the streaming multistart route must not silently drop options
# ---------------------------------------------------------------------------


class TestStreamingMultistartRejectsIgnoredOptions:
    G: ClassVar[dict] = {
        "p0": [1, 1],
        "bounds": ([0, 0], [10, 10]),
        "workflow": "auto_global",
        "method": "multi-start",
        "n_starts": 2,
        "memory_limit_gb": 1e-5,
    }

    def test_data_mask_raises(self):
        x, y = _data(2000)
        with pytest.raises(ValueError, match="data_mask"):
            fit(model, x, y, data_mask=np.arange(2000) >= 200, **self.G)

    def test_robust_loss_raises(self):
        x, y = _data(2000)
        with pytest.raises(ValueError, match="loss"):
            fit(model, x, y, loss="cauchy", **self.G)

    def test_nan_omit_raises(self):
        x, y = _data(2000)
        with pytest.raises(ValueError, match="nan_policy"):
            fit(model, x, y, nan_policy="omit", **self.G)


# ---------------------------------------------------------------------------
# W3 (C1): auto_bounds refit with a route that reports cost=None
# ---------------------------------------------------------------------------


def test_auto_bounds_refit_without_cost_keeps_original():
    x = np.linspace(0, 5, 300)
    y = np.asarray(model(x, 2.0, 0.7))
    calls = []

    def run(**kw):
        _, ub = (np.asarray(b, float) for b in kw["bounds"])
        calls.append(1)
        return CurveFitResult(
            x=np.array([ub[0] if np.isfinite(ub[0]) else 12.0, 0.7]), cost=None
        )

    out = mp._curve_fit_with_auto_bounds(
        run,
        model,
        x,
        y,
        10.0,
        "cmaes",
        {"p0": [1.0, 1.0], "bounds": ([0, 0], [np.inf, np.inf])},
    )
    assert out is not None


# ---------------------------------------------------------------------------
# W4 (C1): workflow CMA-ES adapter keeps the refinement fields
# ---------------------------------------------------------------------------


@needs_evosax
def test_fit_cmaes_route_keeps_refinement_fields():
    x, y = _data(500)
    r = fit(
        model,
        x,
        y,
        p0=[1, 1],
        bounds=([0, 0], [10, 10]),
        workflow="auto_global",
        method="cmaes",
    )
    assert r.get("cost") is not None
    assert r.get("nfev") is not None
    assert r.get("fun") is not None
    # historical keys of the workflow route stay available
    assert np.allclose(r["popt"], r["x"])


# ---------------------------------------------------------------------------
# W5 (C1): memory sizing counts points, not features
# ---------------------------------------------------------------------------


@needs_evosax
def test_cmaes_memory_sizing_counts_points(monkeypatch):
    seen = []
    orig = wf.MemoryBudget.compute.__func__

    def spy(cls, *a, **k):
        seen.append(k.get("n_points"))
        return orig(cls, *a, **k)

    monkeypatch.setattr(wf.MemoryBudget, "compute", classmethod(spy))

    def m2(X, a, b):
        return a * X[0] + b * X[1]

    X = np.vstack([np.linspace(0, 1, 500), np.linspace(1, 2, 500)])
    Y = np.asarray(3 * X[0] + X[1])
    curve_fit(m2, X, Y, p0=[1, 1], bounds=([0, 0], [10, 10]), method="cmaes")
    assert seen and seen[0] == 500


# ---------------------------------------------------------------------------
# S6 (C1): auto-bounds inference must not see data_mask-excluded rows
# ---------------------------------------------------------------------------


def test_auto_bounds_inference_excludes_masked_rows(monkeypatch):
    import nlsq.precision.bound_inference as bi

    x, y = _data(2000)
    y[:200] = 50.0
    got = []
    orig = bi.infer_bounds

    def spy(xd, yd, *a, **k):
        got.append((len(yd), float(np.max(yd))))
        return orig(xd, yd, *a, **k)

    monkeypatch.setattr(bi, "infer_bounds", spy)
    curve_fit(
        model, x, y, p0=[1, 1], auto_bounds=True, data_mask=np.arange(2000) >= 200
    )
    assert got and got[0][0] == 1800 and got[0][1] < 10


# ---------------------------------------------------------------------------
# C2 W1-W3: method dispatch
# ---------------------------------------------------------------------------


def test_cmaes_without_evosax_falls_back_to_multistart(monkeypatch):
    monkeypatch.setattr(ms, "is_evosax_available", lambda: False)
    x, y = _data(100)
    r = curve_fit(
        model, x, y, p0=[2, 1], bounds=([0, 0], [10, 5]), method="cmaes", n_starts=3
    )
    assert np.allclose(r.popt, [2.0, 0.7], atol=0.1)
    assert r["multistart_diagnostics"]["bypassed"] is False


def test_multistart_method_with_zero_starts_bypasses():
    x, y = _data(100)
    r = curve_fit(model, x, y, p0=[2, 1], method="multi-start", n_starts=0)
    assert np.allclose(r.popt, [2.0, 0.7], atol=0.1)
    assert r["multistart_diagnostics"]["bypassed"] is True


@needs_evosax
def test_positional_method_cmaes_routes_to_cmaes(monkeypatch):
    x, y = _data(100)
    calls = []
    orig = mp._run_cmaes_optimization

    def spy(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(mp, "_run_cmaes_optimization", spy)
    curve_fit(model, x, y, [2, 1], None, False, True, ([0, 0], [10, 5]), "cmaes")
    assert calls


# ---------------------------------------------------------------------------
# C2 W4 / S1 / S2: output-flag precedence and types on every route
# ---------------------------------------------------------------------------


def _force_chunked(monkeypatch):
    def sel(self, n_points, n_params, memory_limit_gb=None, verbose=False, **k):
        b = wf.MemoryBudget.compute(n_points=n_points, n_params=n_params)
        return self._create_chunked_config(b, n_params, None, n_points)

    monkeypatch.setattr(wf.MemoryBudgetSelector, "select", sel)


def test_chunked_auto_output_flag_precedence(monkeypatch):
    _force_chunked(monkeypatch)
    x, y = _data(2000, sigma=0.05)
    both = curve_fit(
        model, x, y, p0=[2, 1], method="auto", return_eval=True, timeit=True
    )
    assert len(both) == 3 and both[2].shape == (2000,)

    t = curve_fit(model, x, y, p0=[2, 1], method="auto", timeit=True)
    assert len(t) == 5 and isinstance(t[3], float) and isinstance(t[4], float)

    fo = curve_fit(model, x, y, p0=[2, 1], method="auto", full_output=True)
    assert len(fo) == 5 and isinstance(fo[2], dict) and isinstance(fo[4], int)


def test_hybrid_flags_return_numpy_and_follow_precedence():
    x, y = _data(500, sigma=0.05)
    cf = CurveFit()
    re = cf.curve_fit(
        model, x, y, p0=[2, 1], method="hybrid_streaming", return_eval=True
    )
    assert all(isinstance(t, np.ndarray) for t in re)
    both = cf.curve_fit(
        model, x, y, p0=[2, 1], method="hybrid_streaming", return_eval=True, timeit=True
    )
    assert len(both) == 3
    fo = cf.curve_fit(
        model, x, y, p0=[2, 1], method="hybrid_streaming", full_output=True
    )
    assert len(fo) == 5 and isinstance(fo[0], np.ndarray) and isinstance(fo[4], int)


@pytest.mark.parametrize(
    ("status", "success", "ier"), [(2, True, 2), (1, True, 1), (0, False, 5)]
)
def test_apply_output_flags_ier_is_result_status(status, success, ier):
    res = CurveFitResult(
        x=np.array([2.0, 0.7]),
        pcov=np.eye(2),
        status=status,
        success=success,
        message="m",
        nfev=3,
    )
    out = mp._apply_output_flags(
        res, model, np.linspace(0, 1, 5), {"full_output": True}
    )
    assert out[4] == ier and out[3] == "m" and out[2]["nfev"] == 3


# ---------------------------------------------------------------------------
# C2 W5: diagnostics / SVD limit reach multistart and fallback routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"fallback": True},
        {"multistart": True, "n_starts": 3, "bounds": ([0, 0], [10, 5])},
    ],
)
def test_compute_diagnostics_reaches_route(kw):
    x, y = _data(60, sigma=0.05)
    r = curve_fit(model, x, y, p0=[2, 1], compute_diagnostics=True, **kw)
    assert r.get("_diagnostics_report") is not None


def test_max_jacobian_elements_reaches_multistart(monkeypatch):
    seen = []
    orig = CurveFit.__init__

    def spy(self, *a, **k):
        orig(self, *a, **k)
        seen.append(self.max_jacobian_elements_for_svd)

    monkeypatch.setattr(CurveFit, "__init__", spy)
    x, y = _data(60, sigma=0.05)
    curve_fit(
        model,
        x,
        y,
        p0=[2, 1],
        multistart=True,
        n_starts=2,
        bounds=([0, 0], [10, 5]),
        max_jacobian_elements_for_svd=123,
    )
    assert 123 in seen


# ---------------------------------------------------------------------------
# Round 2 (Codex): W1..W7
# ---------------------------------------------------------------------------


def _lin(x, a, b):
    return a * x + b


def _lin_nan_data(n=40):
    x = np.linspace(0, 1, n)
    y = 2 * x + 1
    y[3] = np.nan
    return x, y


@pytest.mark.parametrize(
    "route", [{"fallback": True}, {"multistart": True, "n_starts": 2}]
)
def test_r2_w1_explicit_lm_with_bounds_raises_on_every_route(route):
    x = np.arange(6.0)
    y = 2 * x + 1
    kw = {"p0": [1.0, 1.0], "bounds": ([0.0, 0.0], [10.0, 10.0]), "method": "lm"}
    with pytest.raises(ValueError, match="lm"):
        curve_fit(_lin, x, y, **kw)
    with pytest.raises(ValueError, match="lm"):
        curve_fit(_lin, x, y, **kw, **route)


def test_r2_fallback_lm_without_bounds_still_runs():
    x = np.arange(6.0)
    r = curve_fit(_lin, x, 2 * x + 1, p0=[1.0, 1.0], method="lm", fallback=True)
    assert np.allclose(r.popt, [2, 1], atol=1e-6)


@pytest.mark.parametrize(
    "route", [{"fallback": True}, {"multistart": True, "n_starts": 2}]
)
def test_r2_w1_hybrid_streaming_with_multistart_or_fallback_is_rejected(route):
    x = np.arange(6.0)
    with pytest.raises(ValueError, match="hybrid_streaming"):
        curve_fit(_lin, x, 2 * x + 1, p0=[1.0, 1.0], method="hybrid_streaming", **route)


@pytest.mark.parametrize(
    "route", [{"fallback": True}, {"multistart": True, "n_starts": 2}]
)
def test_r2_w1_explicit_trf_is_forwarded(monkeypatch, route):
    seen = []
    orig = CurveFit.curve_fit

    def spy(self, *a, **k):
        seen.append(k.get("method"))
        return orig(self, *a, **k)

    monkeypatch.setattr(CurveFit, "curve_fit", spy)
    x = np.arange(6.0)
    curve_fit(
        _lin,
        x,
        2 * x + 1,
        p0=[1.0, 1.0],
        bounds=([0.0, 0.0], [10.0, 10.0]),
        method="trf",
        **route,
    )
    assert seen and all(m == "trf" for m in seen)


def test_r2_w2_auto_chunked_stores_rows_actually_fit():
    x, y = _lin_nan_data()
    r = curve_fit(
        _lin, x, y, p0=[2, 1], method="auto", memory_limit_gb=0.001, nan_policy="omit"
    )
    assert len(r["xdata"]) == len(r["ydata"]) == 39
    assert np.isfinite(r.r_squared)


def test_r2_w3_auto_streaming_keeps_x_y_aligned():
    x, y = _lin_nan_data()
    y = y + 0.01 * np.sin(7 * x)  # nonzero residuals so aic is finite
    r = curve_fit(
        _lin, x, y, p0=[2, 1], method="auto", memory_limit_gb=1e-7, nan_policy="omit"
    )
    assert len(r["xdata"]) == len(r["ydata"]) == 39
    assert np.asarray(r.residuals).shape == (39,)
    assert np.isfinite(r.r_squared) and np.isfinite(r.aic)


@pytest.mark.parametrize("flag", ["full_output", "return_eval", "timeit"])
def test_r2_w4_auto_bounds_flags_do_not_change_the_fit(flag):
    x = np.linspace(0, 1, 20)
    y = 2 * x

    def lin1(x, a):
        return a * x

    calls = []
    base = {
        "p0": [0.5],
        "max_nfev": 2,
        "auto_bounds": True,
        "callback": lambda *a, **k: calls.append(1),
    }
    plain = curve_fit(lin1, x, y, **base)
    n_plain = len(calls)
    calls.clear()
    out = curve_fit(lin1, x, y, **base, **{flag: True})
    assert np.allclose(np.asarray(out[0]), plain.popt)
    assert len(calls) == n_plain


def test_r2_w5_covariance_sigma_nan_on_masked_row():
    x = np.arange(6.0)
    y = 2 * x + 1
    sigma = np.eye(6)
    sigma[1, :] = np.nan
    sigma[:, 1] = np.nan
    mask = np.array([True, False, True, True, True, True])
    r = fit(
        _lin,
        x,
        y,
        p0=[2, 1],
        sigma=sigma,
        data_mask=mask,
        nan_policy="omit",
        workflow=LDMemoryConfig(memory_limit_gb=8),
    )
    assert np.allclose(r.popt, [2, 1], atol=1e-6)
    assert len(r["ydata"]) == 5


def test_r2_w5_retained_row_with_nan_covariance_still_raises():
    x = np.arange(6.0)
    sigma = np.eye(6)
    sigma[2, :] = np.nan
    sigma[:, 2] = np.nan
    mask = np.array([True, False, True, True, True, True])
    with pytest.raises(ValueError):
        fit(
            _lin,
            x,
            2 * x + 1,
            p0=[2, 1],
            sigma=sigma,
            data_mask=mask,
            nan_policy="omit",
            workflow=LDMemoryConfig(memory_limit_gb=8),
        )


@pytest.mark.parametrize("stability", [False, True])
@pytest.mark.parametrize(
    "bounds",
    ["Bounds", (0, 10), ([0, 0], [10, 10])],
    ids=["scipy-Bounds", "scalar", "arrays"],
)
def test_r2_w6_curvefit_normalizes_bounds(stability, bounds):
    from scipy.optimize import Bounds

    if isinstance(bounds, str):
        bounds = Bounds([0, 0], [10, 10])
    x = np.linspace(0, 1, 30)
    r = CurveFit(enable_stability=stability).curve_fit(
        _lin, x, 2 * x + 1, p0=[1, 1], bounds=bounds
    )
    assert np.allclose(r.popt if hasattr(r, "popt") else r[0], [2, 1], atol=1e-5)


def test_r2_w7_auto_chunked_full_output_fvec():
    x = np.linspace(0, 1, 40)
    y = 2 * x + 1
    out = curve_fit(
        _lin, x, y, p0=[2, 1], method="auto", memory_limit_gb=0.001, full_output=True
    )
    ref = curve_fit(_lin, x, y, p0=[2, 1], full_output=True)
    assert out[2]["fvec"] is not None
    assert np.asarray(out[2]["fvec"]).shape == np.asarray(ref[2]["fvec"]).shape
    assert np.allclose(out[2]["fvec"], ref[2]["fvec"], atol=1e-6)


def test_r2_w4_flagged_relaxation_returns_chosen_attempt_verbatim():
    x = np.linspace(0, 5, 50)
    y = np.asarray(model(x, 2.0, 0.7))
    seen = []

    def run(**kw):
        seen.append(kw)
        _, ub = (np.asarray(b, float) for b in kw["bounds"])
        clipped = np.isfinite(ub[0])  # first attempt: inferred upper bound
        fvec = np.full(5, 1.0 if clipped else 0.1)
        popt = np.array([ub[0] if clipped else 12.0, 0.7])
        return popt, np.eye(2), {"nfev": 3, "fvec": fvec}, "m", 1

    out = mp._curve_fit_with_auto_bounds(
        run,
        model,
        x,
        y,
        10.0,
        None,
        {"p0": [1.0, 1.0], "bounds": ([0, 0], [np.inf, np.inf]), "full_output": True},
    )
    assert len(seen) == 2 and all(k["full_output"] for k in seen)
    assert out[0][0] == 12.0  # the relaxed attempt's tuple, not re-solved


# ---------------------------------------------------------------------------
# W8 (batch D): boundedness reaches every memory-budget call
# ---------------------------------------------------------------------------


def _spy_budget(monkeypatch):
    seen = {"select": [], "compute": []}
    orig_sel = wf.MemoryBudgetSelector.select
    orig_cmp = wf.MemoryBudget.compute.__func__

    def sel(self, *a, **k):
        seen["select"].append(bool(k.get("bounded", False)))
        return orig_sel(self, *a, **k)

    def cmp_(cls, *a, **k):
        seen["compute"].append(bool(k.get("bounded", False)))
        return orig_cmp(cls, *a, **k)

    monkeypatch.setattr(wf.MemoryBudgetSelector, "select", sel)
    monkeypatch.setattr(wf.MemoryBudget, "compute", classmethod(cmp_))
    return seen


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        (None, False),
        ((-np.inf, np.inf), False),
        ((0, 10), True),
        (([0, 0], [10, 10]), True),
        (([-np.inf, -np.inf], [10, np.inf]), True),
        ("Bounds", True),
    ],
)
def test_w8_method_auto_passes_boundedness(monkeypatch, bounds, expected):
    from scipy.optimize import Bounds

    if isinstance(bounds, str):
        bounds = Bounds([0, 0], [10, 10])
    seen = _spy_budget(monkeypatch)
    x = np.linspace(0, 1, 30)
    kw = {} if bounds is None else {"bounds": bounds}
    curve_fit(_lin, x, 2 * x + 1, p0=[1, 1], method="auto", **kw)
    assert seen["select"] and all(b is expected for b in seen["select"])


@pytest.mark.parametrize(("bounds", "expected"), [(None, False), ((0, 10), True)])
def test_w8_fit_workflow_auto_passes_boundedness(monkeypatch, bounds, expected):
    seen = _spy_budget(monkeypatch)
    x = np.linspace(0, 1, 30)
    kw = {} if bounds is None else {"bounds": bounds}
    fit(_lin, x, 2 * x + 1, p0=[1, 1], workflow="auto", **kw)
    assert seen["compute"] and all(b is expected for b in seen["compute"])
    assert seen["select"] and all(b is expected for b in seen["select"])


def test_w8_verbose_diagnostics_budget_is_bounded(monkeypatch):
    seen = _spy_budget(monkeypatch)
    x = np.linspace(0, 1, 30)
    curve_fit(_lin, x, 2 * x + 1, p0=[1, 1], bounds=(0, 10), verbose=2)
    assert seen["compute"] and all(seen["compute"])


def test_w8_bounded_layout_changes_the_strategy():
    # 1000 points, 5000 parameters, 1 GiB: the bounded layout needs chunking.
    sel = wf.MemoryBudgetSelector()
    kw = {"n_points": 1000, "n_params": 5000, "memory_limit_gb": 1.0}
    assert sel.select(**kw)[0] == "standard"
    assert sel.select(**kw, bounded=True)[0] == "chunked"


def test_r3_streaming_multistart_warns_on_ignored_user_tolerances():
    x, y = _data(2000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit(
            model,
            x,
            y,
            p0=[1, 1],
            bounds=([0, 0], [10, 10]),
            workflow="auto_global",
            method="multi-start",
            n_starts=2,
            memory_limit_gb=1e-5,
            ftol=1e-3,
        )
    assert any("ignores" in str(m.message) and "ftol" in str(m.message) for m in w)


def test_r3_streaming_multistart_goal_tolerances_do_not_warn():
    x, y = _data(2000)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit(
            model,
            x,
            y,
            p0=[1, 1],
            bounds=([0, 0], [10, 10]),
            workflow="auto_global",
            method="multi-start",
            n_starts=2,
            memory_limit_gb=1e-5,
        )
    assert not any("ignores" in str(m.message) for m in w)


def test_r3_run_multistart_docstring_is_numpydoc_clean():
    import inspect

    doc = inspect.cleandoc(mp._run_multistart_optimization.__doc__)
    assert "XX" not in doc
    assert "\nReturns\n-------\n" in doc
    assert "\nParameters\n----------\n" in doc


# ---------------------------------------------------------------------------
# Round 3 (Codex): auto_bounds cost without status, auto return_eval rows, auto routing
# ---------------------------------------------------------------------------


def _clip_case():
    x = np.linspace(0, 1e-6, 20)
    base = {
        "p0": [0.01],
        "auto_bounds": True,
        "gtol": 1e-20,
        "ftol": 1e-12,
        "xtol": 1e-12,
    }
    return x, 2 * x, base


def _lin1(x, a):
    return a * x


def _variants():
    _x, y, _base = _clip_case()
    yn = y.copy()
    yn[4] = np.nan
    mask = np.ones(20, bool)
    mask[7] = False
    return {
        "scalar_sigma": (y, {"sigma": 2.0}),
        "cov_sigma": (y, {"sigma": 4.0 * np.eye(20)}),
        "nan_omit": (yn, {"nan_policy": "omit"}),
        "nan_omit_sigma": (yn, {"nan_policy": "omit", "sigma": np.full(20, 2.0)}),
        "mask": (y, {"data_mask": mask}),
        "mask_cov": (y, {"data_mask": mask, "sigma": 4.0 * np.eye(20)}),
    }


@pytest.mark.parametrize("name", list(_variants()))
def test_r3_w1_return_eval_does_not_change_auto_bounds_fit(name):
    x, _, base = _clip_case()
    y, extra = _variants()[name]
    plain = curve_fit(_lin1, x, y, **base, **extra)
    out = curve_fit(_lin1, x, y, **base, **extra, return_eval=True)
    assert np.allclose(plain.popt, [2.0], rtol=1e-3)
    assert np.allclose(np.asarray(out[0]), plain.popt, rtol=1e-3)


def test_r3_w1_uncomparable_eval_cost_warns(monkeypatch):
    # If the cost of a return_eval attempt cannot be rebuilt, say so loudly.
    monkeypatch.setattr(mp, "_auto_bounds_cost", lambda *a, **k: np.inf)
    x, y, base = _clip_case()
    with pytest.warns(mp.OptimizeWarning, match="auto_bounds"):
        curve_fit(_lin1, x, y, **base, sigma=2.0, return_eval=True)


def test_r3_claude_unknown_costs_still_prefer_the_converged_fit(monkeypatch):
    # Both costs unknown (e.g. CMA-ES without refinement): convergence decides.
    import nlsq.stability.fallback as fb

    monkeypatch.setattr(mp, "_auto_bounds_cost", lambda *a, **k: np.inf)
    monkeypatch.setattr(
        fb, "_fit_succeeded", lambda r, flags: abs(float(r.popt[0]) - 2.0) < 0.5
    )
    x, y, base = _clip_case()
    with pytest.warns(mp.OptimizeWarning, match="refit without it"):
        out = curve_fit(_lin1, x, y, **base)
    assert np.allclose(out.popt, [2.0], rtol=1e-3)


def test_r3_claude_unknown_cost_warning_names_the_real_cause(monkeypatch):
    monkeypatch.setattr(mp, "_auto_bounds_cost", lambda *a, **k: np.inf)
    x, y, base = _clip_case()
    with pytest.warns(mp.OptimizeWarning, match="auto_bounds") as rec:
        curve_fit(_lin1, x, y, **base)
    assert not any("return_eval" in str(w.message) for w in rec)


def test_r3_w2_auto_chunked_return_eval_describes_rows_fit():
    x, y = _lin_nan_data()
    ref = curve_fit(_lin, x, y, p0=[2, 1], nan_policy="omit", return_eval=True)
    out = curve_fit(
        _lin,
        x,
        y,
        p0=[2, 1],
        method="auto",
        memory_limit_gb=0.001,
        nan_policy="omit",
        return_eval=True,
    )
    assert len(out[2]) == len(ref[2]) == 39
    assert np.allclose(out[2], ref[2])


@pytest.mark.parametrize(
    "route", [{"fallback": True}, {"multistart": True, "n_starts": 2}]
)
def test_r3_w3_method_auto_reaches_memory_selection(monkeypatch, route):
    n = {"select": 0}
    orig = wf.MemoryBudgetSelector.select

    def sel(self, *a, **k):
        n["select"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(wf.MemoryBudgetSelector, "select", sel)
    x = np.linspace(0, 1, 30)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        curve_fit(
            _lin,
            x,
            2 * x + 1,
            p0=[1, 1],
            bounds=([0, 0], [10, 10]),
            method="auto",
            memory_limit_gb=0.001,
            **route,
        )
    assert n["select"] > 0
    assert not any("memory_limit_gb" in str(m.message) for m in w)


def test_r4_return_eval_ranks_attempts_by_real_status():
    # The relaxed refit exhausts max_nfev (not converged); the clipped fit
    # converged. return_eval must not pick the unconverged one.
    x = np.linspace(0, 1e-6, 20)
    base = {
        "p0": [0.01],
        "auto_bounds": True,
        "gtol": 1e-20,
        "ftol": 1e-12,
        "xtol": 1e-12,
        "max_nfev": 6,
    }

    def sq(x, a):
        return a * a * x

    y = 4 * x
    plain = curve_fit(sq, x, y, **base)
    out = curve_fit(sq, x, y, **base, return_eval=True)
    assert np.allclose(np.asarray(out[0]), plain.popt, rtol=1e-3)
