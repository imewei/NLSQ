"""Regression tests for the 2026-11-18 round-2 review (fixer F6: diagnostics/)."""

import warnings

import numpy as np
import pytest

from nlsq.diagnostics.gradient_health import GradientMonitor
from nlsq.diagnostics.plugin import PluginRegistry, run_plugins
from nlsq.diagnostics.types import DiagnosticsConfig, PluginResult, _json_safe


@pytest.fixture(autouse=True)
def _clean_registry():
    PluginRegistry.clear()
    yield
    PluginRegistry.clear()


class _Plugin:
    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def analyze(self, jacobian, parameters, residuals, **context):
        return self._fn(jacobian, parameters, residuals, **context)


def _ok(name):
    return PluginResult(plugin_name=name, available=True, data={}, issues=[])


def test_p07_plugin_cannot_mutate_caller_arrays():
    y = np.array([2.0, 4.0, 6.0])
    r = np.array([1.0, 2.0, 3.0])

    def evil(J, p, res, ydata=None, **kw):
        ydata -= ydata.mean()
        return _ok("evil")

    PluginRegistry.register(_Plugin("evil", evil))
    with pytest.warns(UserWarning):
        results = run_plugins(np.eye(3), np.ones(1), r, ydata=y)
    assert not results["evil"].available
    np.testing.assert_array_equal(y, [2.0, 4.0, 6.0])


def test_p07_warnings_as_errors_keeps_plugin_isolation():
    def bad(*a, **k):
        raise ValueError("boom")

    PluginRegistry.register(_Plugin("bad", bad))
    PluginRegistry.register(_Plugin("good", lambda *a, **k: _ok("good")))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        results = run_plugins(np.eye(2), np.ones(1), np.ones(2))
    assert not results["bad"].available and results["good"].available


def test_p07_array_availability_is_isolated():
    bad = PluginResult(
        plugin_name="arr", available=np.array([True, False]), data={}, issues=[]
    )
    PluginRegistry.register(_Plugin("arr", lambda *a, **k: bad))
    with pytest.warns(UserWarning):
        results = run_plugins(np.eye(2), np.ones(1), np.ones(2))
    assert results["arr"].available is False


@pytest.mark.parametrize("g", [np.array(1.0), np.ones((2, 1))])
def test_p07_malformed_first_gradient_leaves_monitor_untouched(g):
    m = GradientMonitor(DiagnosticsConfig())
    with pytest.raises(ValueError):
        m.record_gradient(g, 1.0)
    assert m.iteration_count == 0 and not m._gradient_norm_history


def test_p07_float32_gradient_norm_does_not_overflow():
    m = GradientMonitor(DiagnosticsConfig())
    m.record_gradient(np.array([1e20], dtype=np.float32), 1.0)
    assert m.get_report().final_gradient_norm == pytest.approx(1e20)


def test_p07_mean_gradient_norm_covers_all_iterations():
    m = GradientMonitor(DiagnosticsConfig(gradient_window_size=5, stagnation_window=5))
    for v in [10.0] * 5 + [1.0] * 5:
        m.record_gradient(np.array([v]), 1.0)
    assert m.get_report().mean_gradient_norm == pytest.approx(5.5)


def test_p07_config_rejects_one_sample_stagnation_window_and_big_sloppy():
    with pytest.raises(ValueError, match="stagnation_window"):
        DiagnosticsConfig(stagnation_window=1)
    with pytest.raises(ValueError, match="sloppy_threshold must be positive"):
        DiagnosticsConfig(sloppy_threshold=2.0)


def test_p07_longdouble_metrics_serialize():
    out = _json_safe({"e": np.longdouble("1.5"), "a": np.array([1.5], np.longdouble)})
    assert out == {"e": 1.5, "a": [1.5]}


def test_p07_summary_agrees_with_analyzer_at_condition_threshold():
    from nlsq.diagnostics.health_report import create_health_report
    from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer
    from nlsq.diagnostics.types import _format_model_health_summary

    config = DiagnosticsConfig(condition_threshold=1.0)
    ident = IdentifiabilityAnalyzer(config).analyze(np.eye(2))
    report = create_health_report(identifiability=ident, config=config)
    assert "Practically identifiable: Yes" in _format_model_health_summary(report)
