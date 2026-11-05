"""Regression tests for the 2026-11-04 review round 3 (CLI + diagnostics)."""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.commands import batch as batch_mod
from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import CLIError, CLILogger, DataLoadError
from nlsq.cli.model_registry import ModelRegistry
from nlsq.utils.diagnostics import ConvergenceMonitor

MULTI = textwrap.dedent("""
    import jax.numpy as jnp

    def first_model(x, a, b):
        return a * x + b

    def estimate_p0(xdata, ydata):
        return [1.0, 2.0]

    def bounds():
        return ([0.0, 0.0], [5.0, 5.0])

    def second_model(x, a, b):
        return a * jnp.exp(-b * x)

    def third_model(x, a, b):
        return a + b * x

    def estimate_third_model_p0(xdata, ydata):
        return [9.0, 9.0]

    def third_model_bounds():
        return ([-1.0, -1.0], [1.0, 1.0])
""")


def _load(tmp_path, monkeypatch, fn):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / "multi.py"
    f.write_text(MULTI)
    cfg = {"type": "custom", "path": str(f), "function": fn}
    return ModelRegistry().get_model(str(f), cfg)


def test_bare_helpers_not_applied_to_other_model(tmp_path, monkeypatch):
    """UCL1: bare estimate_p0/bounds belong to the first model only."""
    other = _load(tmp_path, monkeypatch, "second_model")
    assert not hasattr(other, "estimate_p0")
    assert not hasattr(other, "bounds")
    first = _load(tmp_path, monkeypatch, "first_model")
    assert first.estimate_p0(None, None) == [1.0, 2.0]
    assert first.bounds() == ([0.0, 0.0], [5.0, 5.0])


def test_per_function_helpers_resolved(tmp_path, monkeypatch):
    """UCL1: estimate_<fn>_p0 / <fn>_bounds are attached to that function."""
    third = _load(tmp_path, monkeypatch, "third_model")
    assert third.estimate_p0(None, None) == [9.0, 9.0]
    assert third.bounds() == ([-1.0, -1.0], [1.0, 1.0])


def test_handle_fit_nonconvergence_exit_code(monkeypatch):
    """UCL2: result['success'] False -> exit 1."""
    from nlsq.cli.commands import fit
    from nlsq.cli.main import handle_fit

    args = argparse.Namespace(workflow="w.yaml", output=None, style=None, stdout=False)
    monkeypatch.setattr(fit, "run_fit", lambda **kw: {"success": False, "message": "m"})
    assert handle_fit(args) == 1
    monkeypatch.setattr(fit, "run_fit", lambda **kw: {"success": True})
    assert handle_fit(args) == 0


def test_batch_counts_nonconverged_as_failed(tmp_path, monkeypatch):
    """UCL2: non-converged fit is a failed batch entry."""
    from nlsq.cli.commands import fit

    monkeypatch.setattr(fit, "run_fit", lambda **kw: {"success": False, "message": "m"})
    out = batch_mod._run_single_workflow("w.yaml")
    assert out["status"] == "failed"


def _batch_fixture(tmp_path, monkeypatch):
    from nlsq.cli.commands import fit

    ran = []

    def fake(workflow_path, **kw):
        ran.append(Path(workflow_path).name)
        if "bad" in workflow_path:
            raise CLIError("boom")
        return {"success": True}

    monkeypatch.setattr(fit, "run_fit", fake)
    paths = []
    for n in ("bad.yaml", "ok.yaml"):
        (tmp_path / n).write_text("x: 1\n")
        paths.append(str(tmp_path / n))
    return paths, ran


def test_no_continue_on_error_stops_after_runtime_failure(tmp_path, monkeypatch):
    """UCL3: runtime failure stops the batch when continue_on_error=False."""
    paths, ran = _batch_fixture(tmp_path, monkeypatch)
    results = batch_mod.run_batch(paths, max_workers=1, continue_on_error=False)
    assert ran == ["bad.yaml"]
    assert [r["status"] for r in results] == ["failed", "skipped"]


def test_continue_on_error_runs_all(tmp_path, monkeypatch):
    paths, ran = _batch_fixture(tmp_path, monkeypatch)
    results = batch_mod.run_batch(paths, max_workers=1)
    assert sorted(ran) == ["bad.yaml", "ok.yaml"]
    assert len(results) == 2


def test_batch_zero_workers_is_cli_error(tmp_path, monkeypatch):
    """UCL5: -w 0 raises CLIError, not a ThreadPoolExecutor ValueError."""
    paths, _ = _batch_fixture(tmp_path, monkeypatch)
    with pytest.raises(CLIError):
        batch_mod.run_batch(paths, max_workers=0)


def test_divergence_detects_nan_costs():
    """UCL6: a NaN cost is divergence, not healthy."""
    m = ConvergenceMonitor()
    for c in [10.0, 5.0, 4.0, 3.0, 5.0, 50.0, float("nan"), float("nan")]:
        m.cost_history.append(c)
    assert m.detect_divergence() == (True, 1.0)
    assert m.detect_stagnation() == (False, 0.0)


def test_cli_logger_accepts_exc_info():
    """UCL7: exc_info must reach Logger.error as a real kwarg."""
    log = CLILogger("nlsq.test_ucl7")
    try:
        raise ValueError("x")
    except ValueError:
        log.error("failed", exc_info=True, detail=1)


def test_loader_rejects_nonpositive_sigma(tmp_path):
    """UCL4 (loader): sigma <= 0 is rejected."""
    x = np.arange(5.0)
    loader = DataLoader()
    with pytest.raises(DataLoadError, match="sigma"):
        loader._validate_data(
            x, x, np.array([1.0, 1.0, 0.0, 1.0, 1.0]), tmp_path / "f.csv", {}
        )
    loader._validate_data(x, x, np.ones(5), tmp_path / "f.csv", {})
