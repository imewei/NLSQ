"""Regression tests for the 2026-11-04 GUI review findings."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.mark.serial
def test_file_loader_keeps_all_columns(qapp, tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    f = tmp_path / "d.csv"
    f.write_text("x,y,z,s\n1,2,3,0.1\n2,4,6,0.2\n3,6,9,0.3\n")
    page = DataLoadingPage(AppState())
    page._load_file(f)
    assert page._raw_data.shape == (3, 4)
    assert page._column_names == ["x", "y", "z", "s"]


def _exp_state(bounds=None):
    import jax.numpy as jnp
    import numpy as np

    from nlsq.gui_qt.session_state import SessionState

    x = np.linspace(0, 5, 200)
    y = 3 * np.exp(-0.7 * x) + 0.01 * np.random.default_rng(0).normal(size=200)
    s = SessionState(xdata=x, ydata=y, model_func=lambda x, a, b: a * jnp.exp(-b * x))
    s.p0 = [1.0, 1.0]
    s.bounds = bounds
    return s


def _spy_fit(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa

    captured: dict = {}
    real = fa.fit

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(fa, "fit", spy)
    return captured


def test_fit_adapter_forwards_global_settings(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa

    captured = _spy_fit(monkeypatch)
    s = _exp_state(bounds=([0.0, 0.0], [10.0, 10.0]))
    s.workflow = "auto_global"
    s.n_starts, s.sampler, s.scale_factor, s.center_on_p0 = 3, "sobol", 2.0, False
    s.max_iterations = 50
    fa.run_fit(s)
    assert captured["workflow"] == "auto_global"
    assert captured["n_starts"] == 3
    assert captured["sampler"] == "sobol"
    assert captured["scale_factor"] == 2.0
    assert captured["center_on_p0"] is False
    assert captured["max_nfev"] == 50


def test_auto_global_without_bounds_is_reported_not_silent(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa

    captured = _spy_fit(monkeypatch)
    s = _exp_state(bounds=None)
    s.workflow = "auto_global"
    s.chunk_size = 1234
    r = fa.run_fit(s)
    assert captured["workflow"] == "auto"
    notes = " ".join(r["gui_notes"])
    assert "finite bounds" in notes
    assert "chunk size" in notes


def _results_page(state):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.results import ResultsPage

    aps = AppState()
    aps._state = state
    return ResultsPage(aps)


@pytest.mark.serial
def test_results_page_handles_2d_surface_data(qapp):
    import numpy as np

    from nlsq.gui_qt.adapters.fit_adapter import run_fit
    from nlsq.gui_qt.session_state import SessionState

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (2, 100))
    y = 2 * X[0] + 3 * X[1] + 0.1 * rng.normal(size=100)
    st = SessionState(
        xdata=X, ydata=y, model_func=lambda x, a, b: a * x[0] + b * x[1], data_mode="2d"
    )
    page = _results_page(st)
    page.update_results(run_fit(st))
    assert page.has_results()
    assert "Fit completed" in page._status_label.text()
    assert page._export_btn.isEnabled()


@pytest.mark.serial
def test_results_page_model_failure_is_not_perfect_fit(qapp):
    import numpy as np

    from nlsq.result import CurveFitResult

    def broken(x, a):
        raise RuntimeError("boom")

    from nlsq.gui_qt.session_state import SessionState

    x = np.linspace(0, 1, 20)
    st = SessionState(xdata=x, ydata=x * 2.0, model_func=broken)
    page = _results_page(st)
    page.update_results(
        CurveFitResult(popt=np.array([1.0]), pcov=None, success=True, nfev=3)
    )
    assert "statistics unavailable" in page._status_label.text()
    assert page._fit_statistics._r2_card._value_label.text() == "nan"


@pytest.mark.serial
def test_results_page_unexpected_error_shows_error_state(qapp, monkeypatch):
    import numpy as np

    from nlsq.gui_qt.session_state import SessionState
    from nlsq.result import CurveFitResult

    x = np.linspace(0, 1, 20)
    st = SessionState(xdata=x, ydata=2 * x, model_func=lambda x, a: a * x)
    page = _results_page(st)
    monkeypatch.setattr(
        page, "_update_plots", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad"))
    )
    page.update_results(
        CurveFitResult(popt=np.array([2.0]), pcov=None, success=True, nfev=3)
    )
    assert "Could not display" in page._status_label.text()
    assert not page._export_btn.isEnabled()


@pytest.mark.serial
def test_export_script_and_csv_carry_sigma_and_tolerances(qapp, tmp_path):
    import jax.numpy as jnp
    import numpy as np

    from nlsq.gui_qt.adapters.fit_adapter import run_fit
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.export import ExportPage

    aps = AppState()
    st = aps.state
    x = np.linspace(0, 5, 30)
    sig = np.linspace(0.01, 1, 30)
    y = 3 * np.exp(-0.7 * x) + sig * np.random.default_rng(1).normal(size=30)
    st.xdata, st.ydata, st.sigma = x, y, sig
    st.model_func = lambda x, a, b: a * jnp.exp(-b * x)
    st.model_type = "custom"
    st.model_config = {
        "code": "import jax.numpy as jnp\ndef model(x, a, b):\n    return a*jnp.exp(-b*x)",
        "function": "model",
    }
    st.p0 = [1.0, 1.0]
    st.gtol = st.ftol = st.xtol = 1e-4
    st.loss = "soft_l1"
    st.fit_result = run_fit(st)

    page = ExportPage(aps)
    code = page.generate_python_code()
    for needle in (
        "sigma=sigma",
        "gtol=0.0001",
        "loss='soft_l1'",
        "max_nfev=",
        "skiprows=1",
    ):
        assert needle in code
    lines = page._data_csv_lines()
    assert lines[0] == "x,y,y_fit,residuals,sigma"
    assert float(lines[1].split(",")[-1]) == sig[0]

    ns: dict = {"__x": x, "__y": y, "__s": sig}
    code = code.replace(
        "xdata = np.linspace(0, 10, 30)  # Replace with your data", "xdata = __x"
    ).replace("ydata = np.zeros(30)  # Replace with your data", "ydata = __y")
    code = code.replace("sigma = np.ones(30)  # Replace with your sigma", "sigma = __s")
    exec(code, ns)
    assert np.allclose(ns["popt"], np.asarray(st.fit_result.popt), rtol=1e-6)
