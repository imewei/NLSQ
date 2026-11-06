"""Regression tests for the 2026-11-04 sweep (logging, profiling, histogram)."""

import logging
import sys
import time

import numpy as np
import pytest

from nlsq.utils import logging as nlsq_logging
from nlsq.utils.logging import NLSQLogger
from nlsq.utils.profiler import PerformanceProfiler
from nlsq.utils.profiling import profile_transfers_runtime


@pytest.fixture
def fresh_nlsq_logging(monkeypatch):
    """Uninitialised 'nlsq' logger and a root logger with no handlers."""
    root_nlsq = logging.getLogger("nlsq")
    root = logging.getLogger()
    saved = (root_nlsq.handlers[:], root_nlsq.level, root.handlers[:])
    for k in ("NLSQ_DEBUG", "NLSQ_VERBOSE", "NLSQ_SAVE_ITERATIONS", "NLSQ_LOG_DIR"):
        monkeypatch.delenv(k, raising=False)
    root_nlsq.handlers[:] = []
    root.handlers[:] = []
    root_nlsq._nlsq_initialized = False
    yield root_nlsq
    root_nlsq.handlers[:] = saved[0]
    root_nlsq.setLevel(saved[1])
    root.handlers[:] = saved[2]
    root_nlsq._nlsq_initialized = bool(saved[0])


def _no_root_handlers():
    # pytest re-attaches its capture handlers to root at the start of each phase.
    logging.getLogger().handlers[:] = []


def test_console_handler_writes_to_stderr(fresh_nlsq_logging):
    """SWM1: stdout is reserved for program output (nlsq fit --stdout)."""
    _no_root_handlers()
    NLSQLogger("swm1")
    (handler,) = fresh_nlsq_logging.handlers
    assert handler.stream is sys.stderr


def test_no_duplicate_handler_when_root_configured(fresh_nlsq_logging):
    """SWM8: records propagate to the app's root handler; don't print twice."""
    _no_root_handlers()
    logging.getLogger().addHandler(logging.NullHandler())
    NLSQLogger("swm8")
    assert fresh_nlsq_logging.handlers == []


def test_enable_verbose_mode_lowers_handler_level(fresh_nlsq_logging):
    """SWM7: verbose mode must actually let INFO reach the console handler."""
    _no_root_handlers()
    NLSQLogger("swm7")
    (handler,) = fresh_nlsq_logging.handlers
    assert handler.level == logging.WARNING
    nlsq_logging.enable_verbose_mode()
    assert handler.level == logging.INFO


def test_save_iterations_flag_is_not_a_directory(
    fresh_nlsq_logging, tmp_path, monkeypatch
):
    """SWM7: NLSQ_SAVE_ITERATIONS=1 (flag) must not create ./1/."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NLSQ_SAVE_ITERATIONS", "1")
    monkeypatch.setenv("NLSQ_LOG_DIR", str(tmp_path / "logs"))
    lg = NLSQLogger("swm7b")
    lg.optimization_step(1, 0.5)
    lg.save_iteration_data()
    assert not (tmp_path / "1").exists()
    assert list((tmp_path / "logs").glob("optimization_history_*.npz"))


def test_profile_transfers_does_not_rerun_failing_function(tmp_path):
    """SWM6: the user's exception propagates; the function runs once."""
    calls = []

    def f():
        calls.append(1)
        raise RuntimeError("user bug")

    with pytest.raises(RuntimeError, match="user bug"):
        profile_transfers_runtime(f, trace_dir=str(tmp_path))
    assert len(calls) == 1


def test_nested_profiles_keep_own_timing():
    """Nested profile() must not overwrite the outer start time/metrics."""
    p = PerformanceProfiler()
    with p.profile("outer"):
        time.sleep(0.1)
        with p.profile("inner"):
            time.sleep(0.02)
        p.record_timing("optimization", 1.0)
    outer = p.get_metrics("outer")[0]
    assert outer.total_time >= 0.12
    assert outer.optimization_time == 1.0
    assert p.get_metrics("inner")[0].total_time < outer.total_time


def test_histogram_ignores_nonfinite_residuals():
    """SWM9: NaN/Inf residuals are filtered, not raised on."""
    import matplotlib.pyplot as plt

    from nlsq.cli.visualization import FitVisualizer

    res = np.array([0.1, -0.2, np.nan, 0.3, np.inf, -0.1, 0.05])
    fig = FitVisualizer()._create_histogram_figure(res, {"histogram": {}})
    plt.close(fig)
