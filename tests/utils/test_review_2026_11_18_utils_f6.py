"""Regression tests for the 2026-11-18 three-brain review (nlsq/utils, F6)."""

import json

import numpy as np
import pytest

from nlsq.utils import logging as nlsq_logging
from nlsq.utils.diagnostics import ConvergenceMonitor
from nlsq.utils.profiler import PerformanceProfiler
from nlsq.utils.safe_serialize import SafeSerializationError, safe_dumps, safe_loads
from nlsq.utils.validators import InputValidator, validate_inputs


def test_stagnation_check_survives_squared_mean_underflow():
    diag = ConvergenceMonitor()
    for _ in range(12):
        diag.update(1e-170, np.array([1.0, 2.0]))
    # 1e-170**2 == 0.0 in float64; must not raise ZeroDivisionError
    stagnant, score = diag.detect_stagnation()
    assert stagnant and score == pytest.approx(1.0)


def test_lm_with_infinite_bounds_is_valid():
    v = InputValidator()
    f = lambda x: x
    errs = v._validate_bounds_and_x0((-np.inf, np.inf), np.array([1.0]), "lm")
    assert errs == []
    errs = v._validate_bounds_and_x0((0.0, np.inf), np.array([1.0]), "lm")
    assert any("does not support bounds" in e for e in errs)
    _, _, _ = v.validate_least_squares_inputs(
        f, np.array([1.0]), (-np.inf, np.inf), "lm"
    )


def test_nan_bounds_rejected_by_preflight():
    v = InputValidator()
    errs = v._validate_bounds_and_x0((np.nan, np.inf), np.array([1.0]), "trf")
    assert any("NaN" in e for e in errs)


def test_profiler_report_uses_one_snapshot(monkeypatch):
    prof = PerformanceProfiler()
    with prof.profile("run"):
        pass
    # Clearing between the detail fetch and the summary used to give KeyError('n_runs')
    orig = prof.get_summary
    monkeypatch.setattr(prof, "get_summary", lambda name="default": orig("nothing"))
    assert "Runs: 1" in prof.get_report("run", detailed=True)


def test_html_report_survives_empty_summary(monkeypatch):
    from nlsq.utils.profiler_visualization import ProfilerVisualization

    prof = PerformanceProfiler()
    with prof.profile("run"):
        pass
    monkeypatch.setattr(prof, "get_summary", lambda name="default": {})
    html = ProfilerVisualization(prof).generate_html_report()
    assert "Total Runs" in html


def test_safe_loads_rejects_duplicate_members():
    with pytest.raises(SafeSerializationError, match="Duplicate"):
        safe_loads(b'{"cost":1,"cost":2}')
    assert safe_loads(safe_dumps({"cost": 1})) == {"cost": 1}


def test_safe_loads_rejects_non_integer_data_in_int_array():
    payload = {
        "__nlsq_type__": "ndarray",
        "dtype": "int64",
        "shape": [2],
        "data": [1.5, 2.9],
    }
    with pytest.raises(SafeSerializationError):
        safe_loads(json.dumps(payload).encode())
    arr = np.array([1, 2], dtype=np.int64)
    np.testing.assert_array_equal(safe_loads(safe_dumps(arr)), arr)


def test_slow_validation_scalar_p0_and_zero_d_probe_not_cached():
    v = InputValidator(fast_mode=False)
    x = np.array([1.0, 2.0])
    errs, _ = v._check_function_callable(lambda x, a: a * x, x, x, 1.0, 1)
    assert errs == []
    # a failing probe must not be cached as "tested"
    calls = []

    def bad(x, a):
        calls.append(1)
        raise RuntimeError("boom")

    for _ in range(2):
        errs, _ = v._check_function_callable(bad, x, x, [1.0], 1)
        assert errs and "boom" in errs[0]
    assert len(calls) == 2


def test_validate_inputs_binds_keywords_and_positional_options():
    @validate_inputs("least_squares")
    def fit(fun, x0, bounds=None):
        return x0, bounds

    x0, _ = fit(fun=lambda x: x, x0=[1.0])
    assert list(np.ravel(x0)) == [1.0]
    with pytest.raises(ValueError, match="outside bounds"):
        fit(lambda x: x, [1.0], (2.0, 3.0))  # positional bounds are validated


def test_save_iteration_data_on_flag_and_text_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", " On ")
    monkeypatch.setenv("NLSQ_LOG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path / "..")
    lg = nlsq_logging.NLSQLogger("f6_flag")
    lg.optimization_step(0, 2.0, phase="warmup")
    lg.save_iteration_data()
    files = list(tmp_path.glob("optimization_history_*.npz"))
    assert len(files) == 1
    with np.load(files[0]) as z:
        assert z["phase"].tolist() == ["warmup"]


def test_get_logger_fast_path_survives_concurrent_clear(monkeypatch):
    class Racy(dict):
        # another thread clears the cache right after a membership test succeeds
        def __contains__(self, key):
            hit = super().__contains__(key)
            self.clear()
            return hit

    cache = Racy()
    monkeypatch.setattr(nlsq_logging, "_loggers", cache)
    dict.__setitem__(cache, "nlsq.f6_race", nlsq_logging.NLSQLogger("f6_race"))
    assert nlsq_logging.get_logger("nlsq.f6_race") is not None


def test_validate_inputs_keyword_only_defaults_and_varargs_forms():
    @validate_inputs("least_squares")
    def kw_only(*, fun, x0):
        return x0

    assert list(np.ravel(kw_only(fun=lambda x: x, x0=[1.0]))) == [1.0]

    @validate_inputs("least_squares")
    def mixed(fun, x0, *, bounds=None, method="trf"):
        return x0, bounds, method

    x0, _, method = mixed(lambda x: x, x0=[1.0], bounds=(-np.inf, np.inf))
    assert list(np.ravel(x0)) == [1.0] and method == "trf"
    with pytest.raises(ValueError, match="outside bounds"):
        mixed(fun=lambda x: x, x0=[1.0], bounds=(2.0, 3.0))

    @validate_inputs("least_squares")
    def varargs(*args, **kwargs):
        return args, kwargs

    args, _ = varargs(lambda x: x, [1.0])
    assert len(args) == 2
    with pytest.raises(ValueError, match="at least 2 arguments"):
        varargs(lambda x: x)
