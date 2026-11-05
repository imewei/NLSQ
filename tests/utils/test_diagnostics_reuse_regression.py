"""A fresh optimization must not inherit a previous fit's diagnostics."""

import numpy as np

from nlsq.utils.diagnostics import OptimizationDiagnostics


def test_start_optimization_clears_previous_convergence_history(monkeypatch):
    diagnostics = OptimizationDiagnostics()
    monkeypatch.setattr(diagnostics, "_get_memory_usage", lambda: 100.0)
    diagnostics.start_optimization(x0=np.array([1.0]))
    diagnostics.record_iteration(
        1, np.array([1.0]), 100.0, gradient=np.array([3.0]), step_size=1.0
    )
    monkeypatch.setattr(diagnostics, "_get_memory_usage", lambda: 20.0)
    diagnostics.start_optimization(x0=np.array([2.0]))

    monitor = diagnostics.convergence_monitor
    assert not monitor.cost_history
    assert not monitor.param_history
    assert not monitor.gradient_history
    assert not monitor.step_size_history
    assert diagnostics.peak_memory == 20.0

    diagnostics.record_iteration(1, np.array([2.0]), 1.0)
    assert monitor.get_convergence_rate() is None
    summary = diagnostics.get_summary_statistics()
    assert summary["initial_cost"] == 1.0
    assert summary["memory_increase_mb"] == 0.0
