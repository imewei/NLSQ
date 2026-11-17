"""Regression tests for the 2026-11-16 nlsq/utils round-1 review findings."""

from __future__ import annotations

import csv
import json
import logging

import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

import nlsq.utils.logging as nlsq_logging
from nlsq.utils.diagnostics import OptimizationDiagnostics
from nlsq.utils.logging import NLSQLogger
from nlsq.utils.profiler import PerformanceProfiler, ProfileMetrics
from nlsq.utils.profiler_visualization import ProfilerVisualization, ProfilingDashboard
from nlsq.utils.safe_serialize import SafeSerializationError, safe_loads
from nlsq.utils.validators import InputValidator


# --------------------------------------------------------------------------
# logging.py
# --------------------------------------------------------------------------
def test_save_iteration_data_keeps_columns_absent_from_first_entry(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    lg = NLSQLogger("r1_union_cols")
    lg.optimization_step(0, 5.0, gradient_norm=1.0)  # no step at iteration 0
    lg.optimization_step(1, 3.0, gradient_norm=0.5, step_size=0.25)
    lg.optimization_step(2, 2.0, gradient_norm=0.1, step_size=0.125)
    lg.save_iteration_data()

    (path,) = list(tmp_path.glob("optimization_history_*r1_union_cols_*.npz"))
    with np.load(path) as npz:
        assert "step" in npz.files
        step = npz["step"]
    assert np.isnan(step[0])
    np.testing.assert_array_equal(step[1:], [0.25, 0.125])


def test_save_iteration_data_warns_when_history_truncated(tmp_path, monkeypatch):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    lg = NLSQLogger("r1_trunc_hist")
    n = lg.optimization_history.maxlen
    for i in range(n + 5):
        lg.optimization_history.append({"iter": i, "cost": 1.0})
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _H(logging.WARNING)
    lg.logger.addHandler(h)
    try:
        lg.save_iteration_data()
    finally:
        lg.logger.removeHandler(h)
    assert any("truncat" in r.getMessage().lower() for r in records)


def test_performance_does_not_format_when_disabled():
    lg = NLSQLogger("r1_perf_guard", level=logging.ERROR)
    calls = []
    lg._format_message = lambda *a, **k: calls.append(1) or ""  # type: ignore[method-assign]
    lg.performance("x", a=1)
    assert calls == []


def test_enable_debug_mode_keeps_user_handlers(tmp_path, monkeypatch):
    monkeypatch.setenv("NLSQ_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("NLSQ_DEBUG", "0")
    root = logging.getLogger("nlsq")
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_init = getattr(root, "_nlsq_initialized", False)
    saved_global = nlsq_logging._global_level
    user_handler = logging.StreamHandler()
    user_handler.setLevel(logging.ERROR)
    root.addHandler(user_handler)
    NLSQLogger("r1_debug_handlers")
    try:
        nlsq_logging.enable_debug_mode()
        assert user_handler in root.handlers
        assert user_handler.level == logging.ERROR  # level untouched too
    finally:
        for h in list(root.handlers):
            if h not in saved_handlers:
                root.removeHandler(h)
                h.close()
        for h in saved_handlers:
            if h not in root.handlers:
                root.addHandler(h)
        root.setLevel(saved_level)
        root._nlsq_initialized = saved_init  # type: ignore[attr-defined]
        nlsq_logging._global_level = saved_global
        monkeypatch.delenv("NLSQ_DEBUG", raising=False)


# --------------------------------------------------------------------------
# diagnostics.py
# --------------------------------------------------------------------------
def test_diagnostics_summary_exact_beyond_window():
    d = OptimizationDiagnostics(verbosity=0)
    d.start_optimization(np.zeros(2), "r1_window")
    x = np.zeros(2)
    n = 12_000
    for i in range(1, n + 1):
        d.record_iteration(i, x, cost=1000.0 / i, gradient=np.array([1.0 / i, 0.0]))
    s = d.get_summary_statistics()
    assert s["total_iterations"] == n
    assert s["initial_cost"] == 1000.0
    assert s["final_cost"] == pytest.approx(1000.0 / n)
    assert s["max_cost"] == 1000.0
    assert s["min_cost"] == pytest.approx(1000.0 / n)
    assert s["initial_gradient_norm"] == pytest.approx(1.0)
    assert s["history_truncated"] is True
    assert any("truncat" in w.lower() for w in s["warnings_issued"])


def test_diagnostics_summary_not_truncated_flag_false_for_short_runs():
    d = OptimizationDiagnostics(verbosity=0)
    d.start_optimization(np.zeros(1), "r1_short")
    for i in range(5):
        d.record_iteration(i, np.zeros(1), cost=10.0 - i)
    s = d.get_summary_statistics()
    assert s["history_truncated"] is False
    assert s["total_iterations"] == 5
    assert s["initial_cost"] == 10.0


def test_plot_convergence_closes_figure(tmp_path):
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    d = OptimizationDiagnostics(verbosity=0)
    d.start_optimization(np.zeros(1), "r1_plot")
    for i in range(3):
        d.record_iteration(i, np.zeros(1), cost=3.0 - i)
    plt.close("all")
    d.plot_convergence(save_path=str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").exists()
    assert plt.get_fignums() == []


# --------------------------------------------------------------------------
# validators.py
# --------------------------------------------------------------------------
def _lin(x, a, b):
    return a * x + b


def test_validator_accepts_bool_xdata():
    v = InputValidator(fast_mode=True)
    out = v.validate_curve_fit_inputs(
        _lin,
        np.array([True, False, True, False]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    )
    assert out is not None  # no TypeError from np.ptp on bool


def test_validator_identical_x_is_warning_not_error():
    v = InputValidator(fast_mode=True)
    errors, warns, _, _ = v.validate_curve_fit_inputs(
        _lin, np.full(6, 2.0), np.arange(6.0)
    )
    assert not any("identical" in e for e in errors)
    assert any("x values are identical" in w for w in warns)


def test_stability_curve_fit_accepts_constant_x():
    from nlsq.core.minpack import CurveFit

    def const(x, c):
        return c + 0 * x

    popt, _ = CurveFit(enable_stability=True).curve_fit(
        const, np.full(10, 2.0), np.arange(10.0), p0=[1.0]
    )
    assert float(popt[0]) == pytest.approx(4.5)


# --------------------------------------------------------------------------
# safe_serialize.py
# --------------------------------------------------------------------------
def test_safe_loads_rejects_legacy_type_marker():
    payload = b'{"__type__":"ndarray","dtype":"float64","shape":[2],"data":[1,2]}'
    with pytest.raises(SafeSerializationError, match=r"(?i)legacy"):
        safe_loads(payload)


def test_safe_loads_plain_dict_with_unrelated_type_key_still_loads():
    assert safe_loads(b'{"__type__":"widget","n":1}') == {
        "__type__": "widget",
        "n": 1,
    }


# --------------------------------------------------------------------------
# profiler.py
# --------------------------------------------------------------------------
def test_profile_metrics_speedup_is_labelled_estimate():
    d = ProfileMetrics(backend="gpu", n_data_points=100).to_dict()
    assert "speedup_vs_scipy" not in d  # not exported as if measured
    assert "estimated_speedup_vs_scipy" in d
    assert "estimate" in (ProfileMetrics.speedup_vs_scipy.__doc__ or "").lower()


# --------------------------------------------------------------------------
# profiler_visualization.py
# --------------------------------------------------------------------------
def _profiler_with(name="p", **kw):
    prof = PerformanceProfiler()
    with prof.profile(name):
        prof.update_current(**kw)
    return prof


def test_export_csv_preserves_small_values_and_quotes(tmp_path):
    prof = _profiler_with(
        final_cost=1.234e-9,
        initial_cost=5e-7,
        final_gradient_norm=3e-12,
        method="trf,x",
    )
    out = tmp_path / "o.csv"
    ProfilerVisualization(prof).export_csv("p", out)
    with out.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["final_cost"]) == 1.234e-9
    assert float(row["initial_cost"]) == 5e-7
    assert float(row["final_gradient_norm"]) == 3e-12
    assert row["method"] == "trf,x"
    assert None not in row  # no overflow columns


def test_export_json_handles_numpy_scalars(tmp_path):
    prof = _profiler_with(n_iterations=np.int64(7), success=np.bool_(True))
    out = tmp_path / "o.json"
    ProfilerVisualization(prof).export_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "p" in json.dumps(data)


def test_save_dashboard_sanitizes_profile_names(tmp_path):
    prof = _profiler_with(name="fit/x", n_iterations=3)
    with prof.profile("../escaped"):
        prof.update_current(n_iterations=2)
    dash = ProfilingDashboard(prof)
    dash.add_profile("fit/x")
    dash.add_profile("../escaped")
    out = tmp_path / "dash"
    dash.save_dashboard(out)
    assert not (tmp_path / "escaped_series.png").exists()
    for p in tmp_path.rglob("*"):
        assert p.resolve().is_relative_to(tmp_path.resolve())
    pngs = sorted(q.name for q in out.glob("*_series.png"))
    assert len(pngs) == 2  # distinct names, both inside output_dir


def test_save_dashboard_tolerates_tracked_profile_without_data(tmp_path):
    prof = _profiler_with(name="ok")
    dash = ProfilingDashboard(prof)
    dash.add_profile("ok")
    dash.add_profile("missing")  # no data: must not abort halfway
    out = tmp_path / "dash"
    dash.save_dashboard(out)
    assert (out / "dashboard.html").exists()


def test_html_report_escapes_profile_names():
    prof = _profiler_with(name="<script>alert(1)</script>")
    html = ProfilerVisualization(prof).generate_html_report()
    assert "<script>alert(1)</script>" not in html


def test_import_does_not_switch_matplotlib_backend():
    import subprocess
    import sys

    code = (
        "import matplotlib; matplotlib.use('pdf');"
        "import nlsq.utils.profiler_visualization;"
        "print(matplotlib.get_backend().lower())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**__import__("os").environ, "MPLBACKEND": ""},
    )
    assert out.stdout.strip().splitlines()[-1] == "pdf"


# --------------------------------------------------------------------------
# profiling.py / error_messages.py
# --------------------------------------------------------------------------
def test_profile_transfers_runtime_does_not_claim_zero_transfers():
    from nlsq.utils.profiling import profile_transfers_runtime

    _res, stats = profile_transfers_runtime(lambda x: jnp.sum(x**2), jnp.arange(3.0))
    assert stats["transfer_count"] is None
    assert stats["transfer_bytes"] is None


def test_error_message_docs_describe_max_nfev_as_warning():
    from nlsq.utils import error_messages

    assert "OptimizeWarning" in (error_messages.__doc__ or "")
    assert "OptimizeWarning" in (error_messages.OptimizationError.__doc__ or "")


# --------------------------------------------------------------------------
# cheap suggestions
# --------------------------------------------------------------------------
def test_optimization_error_tolerates_gtol_none():
    from types import SimpleNamespace

    from nlsq.utils.error_messages import OptimizationError

    res = SimpleNamespace(
        x=np.array([1.0]), cost=1.0, grad=np.array([0.5]), nfev=3, nit=2, message="m"
    )
    err = OptimizationError(res, None, 1e-8, 1e-8, 10)  # type: ignore[arg-type]
    assert "Gradient norm" in str(err)


def test_update_current_does_not_overwrite_methods():
    prof = PerformanceProfiler()
    with prof.profile("m"):
        prof.update_current(speedup_vs_scipy=2.5)  # a method name: metadata, no crash
    assert prof.get_metrics("m")[0].metadata["speedup_vs_scipy"] == 2.5


def test_max_condition_number_ignores_nan_entries():
    d = OptimizationDiagnostics(verbosity=1)
    d.start_optimization(np.zeros(1), "r1_nan_cond")
    d.record_iteration(0, np.zeros(1), cost=1.0, jacobian=np.full((3, 1), np.nan))
    d.record_iteration(1, np.zeros(1), cost=0.5, jacobian=np.array([[1.0], [2.0]]))
    assert d.get_summary_statistics()["max_condition_number"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Agy review gaps
# --------------------------------------------------------------------------
def test_validator_rejects_2d_p0_with_matching_len():
    v = InputValidator(fast_mode=True)
    errors, _, _, _ = v.validate_curve_fit_inputs(
        _lin, np.arange(5.0), np.arange(5.0), p0=np.zeros((2, 2))
    )
    assert any("at most 1 dimension" in e for e in errors)


def test_validator_scalar_p0_still_ok():
    v = InputValidator(fast_mode=True)
    errors, _, _, _ = v.validate_curve_fit_inputs(
        lambda x, a: a * x, np.arange(5.0), np.arange(5.0), p0=1.0
    )
    assert not any("p0" in e for e in errors)


def test_summary_without_record_iteration_history_does_not_raise():
    d = OptimizationDiagnostics(verbosity=0)
    d.start_optimization(np.zeros(1), "r1_injected")
    d.iteration_data.append({"iteration": 0, "cost": 4.0, "elapsed_time": 0.1})
    d.iteration_data.append({"iteration": 1, "cost": 1.0, "elapsed_time": 0.2})
    s = d.get_summary_statistics()
    assert s["initial_cost"] == 4.0
    assert s["cost_reduction"] == 3.0
    assert s["total_iterations"] == 2


@pytest.mark.parametrize(
    "val", [1 + 2j, np.complex64(1.5 - 2j), np.complex128(3j), complex(np.nan, np.inf)]
)
def test_safe_serialize_complex_scalar_roundtrip(val):
    from nlsq.utils.safe_serialize import safe_dumps

    out = safe_loads(safe_dumps({"z": val, "t": (val,)}))
    assert isinstance(out["z"], complex)
    expected = complex(val)
    np.testing.assert_array_equal(np.asarray(out["z"]), np.asarray(expected))
    assert out["t"] == (out["z"],) or np.isnan(out["z"].real)


def test_degenerate_range_checks_skip_complex_data():
    import warnings

    v = InputValidator(fast_mode=True)
    x = np.arange(5.0) + 0j
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a ComplexWarning must fail the test
        assert v._check_degenerate_x_values(x) == ([], [])
        assert v._check_degenerate_y_values(np.arange(5.0) + 1j * np.arange(5.0)) == []


# --------------------------------------------------------------------------
# Codex round-1 findings
# --------------------------------------------------------------------------
def test_w1_metadata_cannot_overwrite_exported_fields_or_inject_html():
    prof = PerformanceProfiler()
    with prof.profile("p"):
        prof.update_current(n_iterations=3)
    m = prof.get_metrics("p")[0]
    m.metadata = {"n_iterations": "<script>alert(1)</script>"}
    d = m.to_dict()
    assert d["n_iterations"] == 3
    assert d["metadata"]["n_iterations"] == "<script>alert(1)</script>"
    html = ProfilerVisualization(prof).generate_html_report()
    assert "<script>alert(1)</script>" not in html


def _ndarray_payload(data, shape, dtype="float64"):
    return json.dumps(
        {"__nlsq_type__": "ndarray", "dtype": dtype, "shape": shape, "data": data}
    ).encode()


def test_w2_ndarray_actual_data_checked_before_allocation(monkeypatch):
    import nlsq.utils.safe_serialize as ss

    monkeypatch.setattr(ss, "_MAX_DESERIALIZE_ELEMENTS", 3)

    def boom(*a, **k):
        raise AssertionError("data was processed before the size check")

    monkeypatch.setattr(ss, "_unsanitize_array_data", boom)
    with pytest.raises(SafeSerializationError, match="shape"):
        safe_loads(_ndarray_payload([1.0, 2.0, 3.0, 4.0], [1]))


def test_w2_valid_arrays_still_roundtrip():
    from nlsq.utils.safe_serialize import safe_dumps

    for arr in (
        np.arange(6.0).reshape(2, 3),
        np.array(5.0),
        np.zeros((2, 0)),
        np.array([1 + 2j, np.nan]),
        np.array([True, False]),
    ):
        out = safe_loads(safe_dumps(arr))
        np.testing.assert_array_equal(out, arr)
        assert out.shape == arr.shape


def test_w3_dict_resembling_legacy_marker_rejected_on_dump():
    from nlsq.utils.safe_serialize import safe_dumps

    with pytest.raises(SafeSerializationError):
        safe_dumps({"__type__": "float", "value": "normal application data"})
    # unrelated __type__ dicts remain fine
    assert safe_loads(safe_dumps({"__type__": "widget", "value": 1})) == {
        "__type__": "widget",
        "value": 1,
    }


def test_w3_reserved_nlsq_type_key_rejected_on_dump():
    from nlsq.utils.safe_serialize import safe_dumps

    with pytest.raises(SafeSerializationError):
        safe_dumps({"__nlsq_type__": "float", "value": "x"})


def test_w4_underdetermined_trf_fit_is_not_an_error():
    f = lambda x, a, b, c: a * x * x + b * x + c
    v = InputValidator(fast_mode=True)
    errors, warns, _, _ = v.validate_curve_fit_inputs(
        f,
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
        p0=[1, 1, 1],
        bounds=([-10] * 3, [10] * 3),
    )
    assert not errors
    assert any("fewer data points" in w for w in warns)


def test_w4_lm_still_requires_enough_points():
    f = lambda x, a, b, c: a * x * x + b * x + c
    v = InputValidator(fast_mode=True)
    errors, _, _, _ = v.validate_curve_fit_inputs(
        f, np.array([0.0, 1.0]), np.array([1.0, 3.0]), p0=[1, 1, 1], method="lm"
    )
    assert any("as many data points" in e for e in errors)


def test_w4_stability_curve_fit_accepts_underdetermined_trf():
    from nlsq.core.minpack import CurveFit

    f = lambda x, a, b, c: a * x * x + b * x + c
    popt, _ = CurveFit(enable_stability=True).curve_fit(
        f,
        np.array([0.0, 1.0]),
        np.array([1.0, 3.0]),
        p0=[1.0, 1.0, 1.0],
        bounds=([-10.0] * 3, [10.0] * 3),
    )
    assert np.all(np.isfinite(popt))


def test_w5_indefinite_covariance_sigma_rejected():
    v = InputValidator(fast_mode=True)
    sigma = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    errors, _ = v._validate_sigma(sigma, np.arange(3.0))
    assert any("positive definite" in e for e in errors)


def test_w5_valid_covariance_sigma_accepted():
    v = InputValidator(fast_mode=True)
    sigma = np.array([[2.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    errors, _ = v._validate_sigma(sigma, np.arange(3.0))
    assert errors == []


def test_w6_history_is_per_thread(tmp_path):
    import threading

    lg = NLSQLogger("r1_w6_threads")
    ready = threading.Barrier(2)
    saved: dict[str, np.ndarray] = {}

    def fit(tag, cost):
        lg.optimization_history.clear()
        lg.logger.setLevel(logging.INFO)
        lg.optimization_step(0, cost)
        ready.wait()  # both fits have recorded before either saves
        d = tmp_path / tag
        lg.save_iteration_data(str(d))
        (f,) = d.glob("*.npz")
        with np.load(f) as z:
            saved[tag] = z["cost"]

    import os

    os.environ["NLSQ_SAVE_ITERATIONS"] = str(tmp_path)
    try:
        ts = [
            threading.Thread(target=fit, args=("a", 11.0)),
            threading.Thread(target=fit, args=("b", 22.0)),
        ]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        os.environ.pop("NLSQ_SAVE_ITERATIONS", None)
    assert list(saved["a"]) == [11.0]
    assert list(saved["b"]) == [22.0]


def test_w7_profile_state_is_per_thread():
    import threading

    prof = PerformanceProfiler()
    entered = threading.Barrier(2)
    updated = threading.Barrier(2)

    def run_a():
        with prof.profile("A"):
            entered.wait()
            prof.update_current(n_iterations=11)
            updated.wait()

    def run_b():
        with prof.profile("B"):
            entered.wait()
            updated.wait()

    ts = [threading.Thread(target=run_a), threading.Thread(target=run_b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert prof.get_metrics("A")[0].n_iterations == 11
    assert prof.get_metrics("B")[0].n_iterations == 0


def test_w8_sample_in_bounds_wide_finite_box_not_treated_as_unbounded():
    from nlsq.utils.bounds_sampling import sample_in_bounds

    out = sample_in_bounds([[0.0], [0.5], [1.0]], [-1e308], [1e308], [0.0], [1.0])
    assert out[0, 0] == -1e308
    assert out[1, 0] == pytest.approx(0.0, abs=1e292)
    assert out[2, 0] == 1e308
    assert np.all(np.isfinite(out))


def test_w9_least_squares_validator_rejects_dogbox():
    v = InputValidator(fast_mode=True)
    errors, _, _ = v.validate_least_squares_inputs(
        lambda x: x, np.array([1.0]), method="dogbox"
    )
    assert any("method must be" in e for e in errors)
    for ok in ("trf", "lm"):
        errors, _, _ = v.validate_least_squares_inputs(
            lambda x: x, np.array([1.0]), method=ok
        )
        assert not any("method must be" in e for e in errors)


def test_s1_error_message_does_not_recommend_dogbox():
    from types import SimpleNamespace

    from nlsq.utils.error_messages import analyze_failure

    res = SimpleNamespace(nfev=10, x=np.array([1.0]), cost=1.0, grad=None)
    _, recs = analyze_failure(res, 1e-8, 1e-8, 1e-8, 10)
    assert not any("dogbox" in r for r in recs)


# --------------------------------------------------------------------------
# utils round-2 review
# --------------------------------------------------------------------------
def _summary_for(costs):
    d = OptimizationDiagnostics(verbosity=0)
    d.start_optimization(np.zeros(2), "r2_nan")
    for i, c in enumerate(costs):
        d.record_iteration(i, np.zeros(2), cost=c)
    return d.get_summary_statistics()


def test_r2_w1_all_nan_costs_report_nan_not_sentinels():
    s = _summary_for([np.nan] * 3)
    assert np.isnan(s["min_cost"]) and np.isnan(s["max_cost"])


@pytest.mark.parametrize("costs", [[np.nan, 2.0, 1.0], [2.0, np.nan, 1.0], [1.0, 2.0]])
def test_r2_w1_nan_semantics_match_builtin_min_max(costs):
    s = _summary_for(costs)
    # old behaviour: Python min/max over the cost list
    exp_min, exp_max = min(costs), max(costs)
    assert (np.isnan(s["min_cost"]) and np.isnan(exp_min)) or s["min_cost"] == exp_min
    assert (np.isnan(s["max_cost"]) and np.isnan(exp_max)) or s["max_cost"] == exp_max


def test_r2_w2_end_profile_from_another_thread_stores_profile():
    import threading

    p = PerformanceProfiler()
    m = p.start_profile("x")
    t = threading.Thread(target=lambda: p.end_profile(m))
    t.start()
    t.join()
    assert p.get_metrics("x") == [m]
    assert m.total_time > 0
    assert p._current_profile is None  # starter's stale state cleaned lazily
    assert len(p._context_stack) == 0


def test_r2_s2_no_truncation_warning_when_exactly_full(tmp_path, monkeypatch):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    lg = NLSQLogger("r2_exact_full")
    for i in range(lg.optimization_history.maxlen):
        lg.optimization_history.append({"iter": i, "cost": 1.0})
    records: list[logging.LogRecord] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record)

    h = _H(logging.WARNING)
    lg.logger.addHandler(h)
    try:
        lg.save_iteration_data()
    finally:
        lg.logger.removeHandler(h)
    assert not any("truncat" in r.getMessage().lower() for r in records)


def test_r2_s4_covariance_pd_check_only_in_validation_path():
    v = InputValidator(fast_mode=True)
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])
    assert v._validate_sigma(sigma, np.arange(2.0), strict=False)[0] == []
    assert any(
        "positive definite" in e for e in v._validate_sigma(sigma, np.arange(2.0))[0]
    )


def test_r2_s5_validate_inputs_decorator_forwards_method():
    from nlsq.utils.validators import validate_inputs

    @validate_inputs("curve_fit")
    def fit(f, x, y, **kw):
        return True

    f3 = lambda x, a, b, c: a * x * x + b * x + c
    with pytest.raises(ValueError, match="as many data points"):
        fit(f3, np.array([0.0, 1.0]), np.array([1.0, 3.0]), p0=[1, 1, 1], method="lm")


# --------------------------------------------------------------------------
# utils round-3 review
# --------------------------------------------------------------------------
def test_r3_enable_debug_mode_recreates_nlsq_handlers_beside_app_handler(
    tmp_path, monkeypatch
):
    import io

    monkeypatch.setenv("NLSQ_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("NLSQ_DEBUG", "0")
    root = logging.getLogger("nlsq")
    py_root = logging.getLogger()
    saved = list(root.handlers), root.level, getattr(root, "_nlsq_initialized", False)
    saved_root_handlers = list(py_root.handlers)
    saved_global = nlsq_logging._global_level
    py_root.handlers[:] = []  # unconfigured application root: NLSQ adds a console
    for h in list(root.handlers):
        root.removeHandler(h)
    root._nlsq_initialized = False  # type: ignore[attr-defined]
    lg = nlsq_logging.get_logger("r3_debug_handlers")
    app_stream = io.StringIO()
    app = logging.StreamHandler(app_stream)
    root.addHandler(app)
    try:
        nlsq_logging.enable_debug_mode()
        nlsq_logging.enable_debug_mode()  # repeat: no duplicates
        owned = [h for h in root.handlers if getattr(h, "_nlsq_owned", False)]
        assert app in root.handlers
        assert sum(isinstance(h, logging.FileHandler) for h in owned) == 1
        assert sum(not isinstance(h, logging.FileHandler) for h in owned) == 1
        lg.debug("r3-debug-record")
        for h in root.handlers:
            h.flush()
        assert "r3-debug-record" in app_stream.getvalue()
        assert any(
            "r3-debug-record" in p.read_text()
            for p in tmp_path.glob("nlsq_debug_*.log")
        )
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in saved[0]:
            root.addHandler(h)
        root.setLevel(saved[1])
        root._nlsq_initialized = saved[2]  # type: ignore[attr-defined]
        py_root.handlers[:] = saved_root_handlers
        nlsq_logging._global_level = saved_global


def test_r3_s1_prune_stack_uses_identity_not_equality():
    p = PerformanceProfiler()
    m1 = p.start_profile("same")
    m2 = p.start_profile("same")
    m1.metadata["arr"] = np.arange(3)  # numpy metadata: == on these is ambiguous
    m2.metadata["arr"] = np.arange(3)
    del p._active[id(m2)]  # m2 was ended by another thread
    assert [e[1] for e in p._context_stack] == [m1]


# --------------------------------------------------------------------------
# utils round-2 Codex findings
# --------------------------------------------------------------------------
def test_r2c_w1_empty_shape_with_many_empty_containers_rejected(monkeypatch):
    import nlsq.utils.safe_serialize as ss

    monkeypatch.setattr(ss, "_MAX_DESERIALIZE_ELEMENTS", 3)
    with pytest.raises(SafeSerializationError, match="shape"):
        safe_loads(_ndarray_payload([[] for _ in range(1000)], [0, 0]))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0,), (2, 0, 2)])
def test_r2c_w1_empty_arrays_still_roundtrip(shape):
    from nlsq.utils.safe_serialize import safe_dumps

    arr = np.zeros(shape)
    out = safe_loads(safe_dumps(arr))
    assert out.shape == shape


def test_r2c_w1_ragged_and_overdeep_nesting_rejected():
    with pytest.raises(SafeSerializationError):
        safe_loads(_ndarray_payload([[1.0, 2.0], [3.0]], [2, 2]))
    deep = 1.0
    for _ in range(200):
        deep = [deep]
    with pytest.raises(SafeSerializationError):
        safe_loads(_ndarray_payload(deep, [1] * 200))


def test_r2c_w2_open_interval_overflow_saturates_finite():
    from nlsq.utils.bounds_sampling import sample_in_bounds

    out = sample_in_bounds([[1.0]], [-np.inf], [np.inf], [1e308], [1e308])
    assert np.all(np.isfinite(out))
    assert out[0, 0] == np.finfo(np.float64).max


def test_r2c_w2_unused_branch_not_evaluated_under_raising_errstate():
    from nlsq.utils.bounds_sampling import sample_in_bounds

    with np.errstate(all="raise"):
        out = sample_in_bounds([[1.0]], [0.0], [1.0], [1e308], [1e308])
    assert out[0, 0] == 1.0


# --------------------------------------------------------------------------
# utils round-3 Codex finding: marker payloads must not force traversal
# --------------------------------------------------------------------------
def _count_unsanitize_calls(monkeypatch):
    import nlsq.utils.safe_serialize as ss

    calls = []
    orig = ss._unsanitize_array_data

    def counting(data):
        calls.append(1)
        return orig(data)

    monkeypatch.setattr(ss, "_unsanitize_array_data", counting)
    return calls


def test_r3c_complex_marker_with_nested_parts_rejected_without_traversal(monkeypatch):
    import nlsq.utils.safe_serialize as ss

    monkeypatch.setattr(ss, "_MAX_DESERIALIZE_ELEMENTS", 3)
    calls = _count_unsanitize_calls(monkeypatch)
    marker = {
        "__nlsq_type__": "complex",
        "real": [[] for _ in range(1000)],
        "imag": 0,
    }
    with pytest.raises(SafeSerializationError):
        safe_loads(_ndarray_payload([marker], [1], dtype="complex128"))
    assert len(calls) <= 3


def test_r3c_nested_dict_with_bad_complex_marker_rejected_without_traversal(
    monkeypatch,
):
    calls = _count_unsanitize_calls(monkeypatch)
    deep = [[] for _ in range(1000)]
    payload = json.dumps(
        {"a": {"b": {"__nlsq_type__": "complex", "real": deep, "imag": deep}}}
    ).encode()
    with pytest.raises(SafeSerializationError):
        safe_loads(payload)
    assert len(calls) <= 3


def test_r3c_markers_still_roundtrip():
    from nlsq.utils.safe_serialize import safe_dumps

    obj = {
        "z": complex(np.nan, -np.inf),
        "t": (1 + 2j, (3, float("inf"))),
        "arr": np.array([1 + 1j, complex(np.inf, np.nan)]),
        "d": {"k": complex(0, 1)},
    }
    out = safe_loads(safe_dumps(obj))
    assert np.isnan(out["z"].real) and out["z"].imag == -np.inf
    assert out["t"] == (1 + 2j, (3, float("inf")))
    np.testing.assert_array_equal(out["arr"], obj["arr"])
    assert out["d"] == {"k": 1j}
