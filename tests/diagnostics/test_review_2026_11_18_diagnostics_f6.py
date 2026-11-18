"""Regression tests for the 2026-11-18 three-brain review (nlsq/diagnostics, F6)."""

import gc
import io
import threading

import numpy as np
import pytest

from nlsq.callbacks import IterationLogger
from nlsq.diagnostics.gradient_health import GradientMonitor
from nlsq.diagnostics.plugin import PluginRegistry
from nlsq.diagnostics.types import DiagnosticsConfig
from nlsq.diagnostics.types import _validate_fim as validate_fim


def test_rejected_params_leave_monitor_untouched():
    m = GradientMonitor(DiagnosticsConfig())
    with pytest.raises(ValueError):
        m.record_gradient([3.0], 1.0, [1.0, 2.0])
    assert m.iteration_count == 0 and len(m._gradient_norm_history) == 0
    m.record_gradient([4.0], 1.0, [1.0])
    assert m.get_report().n_iterations == 1
    assert m._gradient_norm_history[-1] == pytest.approx(4.0)


def test_int_gradient_magnitude_does_not_wrap():
    m = GradientMonitor(DiagnosticsConfig())
    m.record_gradient(np.array([-(2**31)], dtype=np.int32), 1.0)
    assert float(m._param_means[0]) == 2.0**31


def test_stagnation_window_larger_than_gradient_window_is_rejected():
    with pytest.raises(ValueError, match="stagnation_window"):
        DiagnosticsConfig(gradient_window_size=5, stagnation_window=10)
    with pytest.warns(UserWarning, match="GRAD-001"):
        DiagnosticsConfig(gradient_window_size=4, stagnation_window=4)


def test_fim_validation_tolerance_is_relative_to_largest_eigenvalue():
    # Documented limit: round-off is judged against the largest eigenvalue, so a
    # valid mixed-scale FIM is accepted and a roundoff-sized negative diagonal
    # next to a unit-scale matrix is noise, not an indefinite block.
    ok = np.array([[1e20, 0, 0], [0, 2, 1], [0, 1, 2]], dtype=float)
    assert validate_fim(ok) is None
    assert validate_fim(np.diag([1.0, 1.0, -1e-17])) is None
    assert "positive semi-definite" in validate_fim(np.diag([1.0, 1.0, -1e-3]))


def test_unregister_with_registry_touching_finalizer_does_not_deadlock():
    class P:
        name = "f6-finalizer"

        def __del__(self):
            PluginRegistry.get("f6-finalizer")

        def analyze(self, *a, **k):  # pragma: no cover
            return None

    PluginRegistry.register(P())
    done = threading.Event()

    def work():
        PluginRegistry.unregister("f6-finalizer")
        PluginRegistry.register(P())
        PluginRegistry.clear()
        gc.collect()
        done.set()

    t = threading.Thread(target=work, daemon=True)
    t.start()
    assert done.wait(10), "registry deadlocked on plugin finalizer"


def test_iteration_logger_provided_file_single_footer_and_closed_file():
    buf = io.StringIO()
    lg = IterationLogger(file=buf)
    lg(1, 1.0, np.array([1.0]), {"nfev": 1})
    lg.close()
    lg.close()
    assert buf.getvalue().count("Optimization completed") == 1
    buf.close()
    lg.close()  # must not write to a closed file
    del lg


def test_rejected_call_leaves_all_monitor_state_untouched():
    m = GradientMonitor(DiagnosticsConfig())
    with pytest.raises(ValueError):
        m.record_gradient([np.nan], 1.0, [1.0, 2.0])
    assert m._has_numerical_issues is False
    assert m.iteration_count == 0 and m._n_params == 0
    m.record_gradient([1.0], 1.0, [1.0])
    assert m._has_numerical_issues is False
    # same after the first call established the parameter count
    with pytest.raises(ValueError):
        m.record_gradient([np.inf], 1.0, [1.0, 2.0])
    with pytest.raises(ValueError):
        m.record_gradient([np.inf, 1.0], 1.0)
    assert m._has_numerical_issues is False and m.iteration_count == 1
