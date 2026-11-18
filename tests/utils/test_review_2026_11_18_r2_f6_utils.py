"""Regression tests for the 2026-11-18 round-2 review (fixer F6: utils/)."""

import json
import pickle

import numpy as np
import pytest

from nlsq.result import OptimizeResult
from nlsq.utils import logging as nlsq_logging
from nlsq.utils.diagnostics import ConvergenceMonitor, OptimizationDiagnostics
from nlsq.utils.error_messages import OptimizationError
from nlsq.utils.logging import get_logger
from nlsq.utils.profiler import PerformanceProfiler
from nlsq.utils.profiler_visualization import ProfilerVisualization, _safe_filename
from nlsq.utils.profiling import analyze_source_transfers
from nlsq.utils.validators import InputValidator, validate_inputs


def test_p09_optimization_error_pickles():
    res = OptimizeResult(x=np.ones(1), cost=1.0, nfev=3, nit=2, message="m", status=0)
    err = OptimizationError(res, 1e-8, 1e-8, 1e-8, 10)
    back = pickle.loads(pickle.dumps(err))
    assert str(back) == str(err) and back.reasons == err.reasons


def test_p09_decorator_applies_wrapped_defaults():
    @validate_inputs("curve_fit")
    def fit(f, xdata, ydata, check_finite=False):
        return xdata

    out = fit(lambda x, a: a * x, np.array([np.nan, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(out[0])


def test_p09_zero_tolerance_is_accepted():
    errs, _, _ = InputValidator().validate_least_squares_inputs(
        lambda p: p, [1.0], ftol=0.0
    )
    assert errs == []


def test_p09_max_nfev_rejects_bool_and_inf():
    v = InputValidator()
    assert v._validate_max_nfev(True, 1)[0]
    assert v._validate_max_nfev(float("inf"), 1)[0]
    assert not v._validate_max_nfev(10, 1)[0]


def test_p09_bounds_rank_and_none_side_checked():
    v = InputValidator()
    assert v._validate_bounds((np.array([[0.0]]), np.array([[2.0]])), 1, [1.0])[0]
    assert v._validate_bounds((None, [1.0, 2.0]), 1, [1.0])[0]


def test_p09_model_probe_rechecked_per_p0_and_sees_full_data():
    v = InputValidator()

    def f(x, a):
        if a < 0:
            raise ValueError("negative a")
        return a * x

    x = np.arange(5.0)
    assert not v._check_function_callable(f, x, x, [1.0], 1)[0]
    assert v._check_function_callable(f, x, x, [-1.0], 1)[0]

    n = np.arange(12.0)
    g = lambda x, a: a * x + n
    assert not v._check_function_callable(g, n, n, [1.0], 1)[0]


def test_c09_validate_inputs_forwards_args_kwargs_to_residual():
    @validate_inputs("least_squares")
    def ls(fun, x0, args=(), kwargs=None):
        return x0

    ls(lambda x, a, b=0: x - a - b, [1.0], args=(2.0,), kwargs={"b": 1.0})


def test_p09_diagnostics_use_optimizer_counters():
    d = OptimizationDiagnostics()
    d.start_optimization(n_params=1)
    d.record_iteration(0, np.ones(1), 1.0, nfev=5, njev=2)
    stats = d.get_summary_statistics()
    assert stats["function_evaluations"] == 5 and stats["jacobian_evaluations"] == 2


def test_p09_relative_reduction_and_rate_are_scale_free():
    def run(scale):
        d = OptimizationDiagnostics()
        d.start_optimization(n_params=1)
        for i in range(5):
            d.record_iteration(i, np.ones(1), scale * 2.0**-i)
        return d.get_summary_statistics()

    small, big = run(1e-40), run(1.0)
    assert small["relative_cost_reduction"] == pytest.approx(0.9375)
    assert small["convergence_rate"] == pytest.approx(big["convergence_rate"])
    assert small["convergence_rate"] == pytest.approx(np.log(2.0))


def test_p09_profile_context_reentrant():
    p = PerformanceProfiler()
    ctx = p.profile("same")
    with ctx, ctx:
        pass
    assert len(p.profiles["same"]) == 2 and not p._active


def test_p09_profiler_json_is_strict(tmp_path):
    p = PerformanceProfiler()
    with p.profile("run") as m:
        m.final_cost = float("inf")
    out = tmp_path / "p.json"
    ProfilerVisualization(p).export_json(out)
    assert "Infinity" not in out.read_text()
    json.loads(out.read_text())


def test_p09_long_profile_name_fits_a_filename():
    assert len(_safe_filename("x" * 261 + "é").encode()) <= 120


def test_p09_parameter_update_reserved_names():
    get_logger("r2f6").parameter_update(np.array([1.0, 2.0]), ["message", "self"])


def test_p09_empty_summaries_do_not_raise():
    lg = get_logger("r2f6")
    lg.matrix_info("J", np.empty((0, 3)))
    lg.data_summary(np.empty(0), np.empty(0))
    lg.data_summary(np.float64(1.0), np.float64(2.0))


def test_p09_jax_compilation_logged_under_performance_tracking(monkeypatch):
    import logging

    monkeypatch.setenv("NLSQ_TRACE_JAX", "1")
    lg = get_logger("r2f6_jit")
    records = []

    class H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = H()
    lg.logger.addHandler(h)
    old = lg.logger.level
    lg.logger.setLevel(nlsq_logging.LogLevel.PERFORMANCE)
    try:
        lg.jax_compilation("f", (3,), 0.2)
    finally:
        lg.logger.removeHandler(h)
        lg.logger.setLevel(old)
    assert any("JAX compilation" in r for r in records)


def test_p09_static_transfer_count_ignores_comments_and_lookalikes():
    src = "x = np.asarray (y)  # np.array(\ns = 'np.array('\nz = anp.array(1)\n"
    out = analyze_source_transfers(src)
    assert out["np_array_calls"] == 0 and out["np_asarray_calls"] == 1


def test_c09_saved_text_labels_keep_original_strings(tmp_path, monkeypatch):
    lg = get_logger("r2f6_save")
    lg.optimization_history.clear()
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", str(tmp_path))
    for phase in ("1", "warmup", "nan"):
        lg.optimization_history.append({"iteration": 0, "phase": phase})
    lg.optimization_history.append({"iteration": 1})
    lg.save_iteration_data(str(tmp_path))
    f = next(tmp_path.glob("optimization_history_*.npz"))
    assert list(np.load(f)["phase"]) == ["1", "warmup", "nan", ""]


def test_v1_successive_halving_at_tiny_scale_is_not_stagnation():
    m = ConvergenceMonitor()
    for i in range(5):
        m.update(1e-40 * 2.0**-i, np.ones(1), np.array([1e-12]))
    assert m.get_convergence_rate() == pytest.approx(np.log(2.0))
    assert m.detect_stagnation()[0] is False


def test_v1_max_nfev_rejects_fractional():
    v = InputValidator()
    assert v._validate_max_nfev(2.5, 1)[0]
    assert v._validate_max_nfev(-1, 1)[0]
    assert not v._validate_max_nfev(3, 1)[0]
