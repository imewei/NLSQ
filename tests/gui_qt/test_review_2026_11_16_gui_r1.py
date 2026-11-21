"""Regression tests for the 2026-11-16 GUI review (round 1)."""

from __future__ import annotations

import logging
import os
import threading
import warnings

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _exp3(x, a, b, c):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x) + c


def _fit_state(x, y, model, p0, sigma=None, absolute_sigma=False):
    from nlsq.gui_qt.session_state import SessionState

    s = SessionState(xdata=np.asarray(x, float), ydata=np.asarray(y, float))
    s.sigma = None if sigma is None else np.asarray(sigma, float)
    s.model_func = model
    s.p0 = p0
    s.absolute_sigma = absolute_sigma
    return s


def _installed(app_state, state, result):
    app_state._state = state
    app_state.set_fit_result(result, state.snapshot_for_fit())


# --- W1: library warnings / log records reach the GUI ------------------------


def test_underdetermined_fit_warning_reaches_gui_notes():
    import nlsq.gui_qt.adapters.fit_adapter as fa

    s = _fit_state([0.0, 1.0], [2.0, 1.0], _exp3, [1.0, 1.0, 0.0])
    r = fa.run_fit(s)
    assert any("ovariance" in n for n in r["gui_notes"])


def test_rank_deficient_log_record_reaches_gui_notes():
    import nlsq.gui_qt.adapters.fit_adapter as fa

    s = _fit_state(
        [1.0] * 20,
        np.linspace(1, 2, 20),
        lambda x, a, b, c: a + b * x + c * x**2,
        [1.0, 1.0, 1.0],
    )
    r = fa.run_fit(s)
    assert any("rank" in n.lower() for n in r["gui_notes"])


def test_diagnostics_capture_is_per_thread():
    from nlsq.gui_qt.adapters.fit_adapter import capture_fit_diagnostics
    from nlsq.result import OptimizeWarning

    other_done = threading.Event()

    def other():
        warnings.warn("from another thread", OptimizeWarning, stacklevel=1)
        logging.getLogger("nlsq.core.test").warning("log from another thread")
        other_done.set()

    # The other thread's warning is delegated to the previous hook, which
    # pytest.warns records; that keeps it off stderr and out of the summary.
    with (
        pytest.warns(OptimizeWarning, match="from another thread"),
        capture_fit_diagnostics() as msgs,
    ):
        warnings.warn("mine", OptimizeWarning, stacklevel=1)
        logging.getLogger("nlsq.core.test").warning("my log record")
        t = threading.Thread(target=other)
        t.start()
        t.join()
    assert other_done.is_set()
    text = " ".join(msgs)
    assert "mine" in text and "my log record" in text
    assert "another thread" not in text


@pytest.mark.serial
def test_results_page_flags_underdetermined_fit(app_state, qtbot):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from nlsq.gui_qt.pages.results import ResultsPage

    s = _fit_state([0.0, 1.0], [2.0, 1.0], _exp3, [1.0, 1.0, 0.0])
    r = fa.run_fit(s)
    app_state.set_data(s.xdata, s.ydata)
    app_state.set_model("custom", model_func=_exp3)
    _installed(app_state, app_state.state, r)

    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(r)

    assert "underdetermined" in page._status_label.text().lower()
    assert "#FF9800" in page._status_label.styleSheet()
    # dof <= 0: the interval is undefined, never a fabricated dof=1 t-interval
    ci_text = page._param_results._table.item(0, 3).text()
    assert "nan" in ci_text.lower() or "inf" in ci_text.lower()


# --- Codex 1: intervals follow the result (known sigma, dof rules) ----------


def _known_sigma_result(app_state):
    import nlsq.gui_qt.adapters.fit_adapter as fa

    s = _fit_state(
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
        lambda x, a: a + 0 * x,
        [0.5],
        sigma=[1.0, 1.0, 1.0],
        absolute_sigma=True,
    )
    r = fa.run_fit(s)
    app_state.set_data(s.xdata, s.ydata, sigma=s.sigma)
    app_state.set_model("custom", model_func=s.model_func)
    app_state.state.absolute_sigma = True
    _installed(app_state, app_state.state, r)
    return r


@pytest.mark.serial
def test_json_export_intervals_match_result_with_known_sigma(app_state, qtbot):
    import json

    from nlsq.gui_qt.pages.export import ExportPage

    r = _known_sigma_result(app_state)
    expected = np.asarray(r.confidence_intervals())[0]
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    p = next(iter(json.loads(page._generate_json())["parameters"].values()))
    assert p["ci_lower"] == pytest.approx(expected[0], rel=1e-9)
    assert p["ci_upper"] == pytest.approx(expected[1], rel=1e-9)


@pytest.mark.serial
def test_csv_export_intervals_match_result_with_known_sigma(app_state, qtbot, tmp_path):
    from nlsq.gui_qt.pages.export import ExportPage

    r = _known_sigma_result(app_state)
    expected = np.asarray(r.confidence_intervals())[0]
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    out = tmp_path / "p.csv"
    page.export_csv(str(out))
    cells = out.read_text().splitlines()[1].split(",")
    assert float(cells[3]) == pytest.approx(expected[0], rel=1e-9)
    assert float(cells[4]) == pytest.approx(expected[1], rel=1e-9)


@pytest.mark.serial
def test_results_table_intervals_match_result_with_known_sigma(app_state, qtbot):
    from nlsq.gui_qt.pages.results import ResultsPage

    r = _known_sigma_result(app_state)
    lo, hi = np.asarray(r.confidence_intervals())[0]
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(r)
    assert page._param_results._table.item(0, 3).text() == f"[{lo:.6g}, {hi:.6g}]"


# --- Codex 2 / S4: bands delegate to the result ------------------------------


def _sqrt_result(app_state, pcov, model=None, a=1e-8):
    import jax.numpy as jnp

    from nlsq.result import CurveFitResult

    def sqrt_model(x, a):
        return jnp.sqrt(a) + 0 * x

    model = model or sqrt_model
    x = np.arange(3.0)
    app_state.set_data(x, np.full(3, 1e-4))
    app_state.set_model("custom", model_func=model)
    r = CurveFitResult(
        popt=np.array([a]),
        pcov=pcov,
        success=True,
        model=model,
        xdata=x,
        ydata=np.full(3, 1e-4),
    )
    _installed(app_state, app_state.state, r)
    return r


@pytest.mark.serial
def test_confidence_band_stays_in_model_domain(app_state, qtbot):
    from nlsq.gui_qt.pages.results import ResultsPage

    r = _sqrt_result(app_state, np.array([[1e-16]]))
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    page.update_results(r)
    upper = page._fit_plot._conf_upper
    assert upper is not None and np.all(np.isfinite(upper))


@pytest.mark.serial
def test_confidence_band_has_no_runtime_warning_for_infinite_pcov(app_state, qtbot):
    from nlsq.gui_qt.pages.results import ResultsPage

    # dy/da = x is 0 at x = 0, so 0 * inf in J @ pcov raises "invalid value"
    r = _sqrt_result(app_state, np.full((1, 1), np.inf), lambda x, a: a * x, 1.0)
    page = ResultsPage(app_state)
    qtbot.addWidget(page)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        page.update_results(r)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]


# --- Codex 3: imported auto_bounds is honoured --------------------------------


def test_auto_bounds_is_forwarded_to_fit(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from tests.gui_qt.test_review_2026_11_04_gui import _exp_state

    captured: dict = {}
    real = fa.fit

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(fa, "fit", spy)
    s = _exp_state()
    s.auto_bounds = True
    fa.run_fit(s)
    assert captured.get("auto_bounds") is True


@pytest.mark.serial
def test_export_script_carries_auto_bounds(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage
    from tests.gui_qt.test_review_2026_11_04_gui import _exp_state

    s = _exp_state()
    s.auto_bounds = True
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    assert "auto_bounds=True" in page.generate_python_code()


# --- Codex 4: recovery replaces the whole data tuple --------------------------


def test_recovery_does_not_keep_old_sigma(app_state):
    from nlsq.gui_qt.autosave import AutosaveManager

    app_state.set_data(np.arange(3.0), np.arange(3.0), sigma=np.full(3, 9.0))
    epoch = app_state.input_epoch
    AutosaveManager(app_state)._restore_state(
        {"xdata": [10.0, 11.0, 12.0], "ydata": [1.0, 2.0, 3.0]}
    )
    assert app_state.state.sigma is None
    assert app_state.input_epoch > epoch


def test_invalid_recovery_clears_old_data(app_state):
    from nlsq.gui_qt.autosave import AutosaveManager

    app_state.set_data(np.arange(3.0), np.arange(3.0), sigma=np.full(3, 9.0))
    AutosaveManager(app_state)._restore_state(
        {"xdata": [1.0, float("nan"), 3.0], "ydata": [1.0, 2.0, 3.0]}
    )
    s = app_state.state
    assert s.xdata is None and s.ydata is None and s.sigma is None


# --- S1: system-aware theme ---------------------------------------------------


class _FakeSignal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _FakeHints:
    def __init__(self, scheme):
        self.scheme = scheme
        self.set_calls = []
        self.colorSchemeChanged = _FakeSignal()

    def colorScheme(self):
        return self.scheme

    def setColorScheme(self, scheme):
        self.set_calls.append(scheme)
        self.scheme = scheme


class _FakePalette:
    def __init__(self, lightness):
        self.lightness_value = lightness

    def window(self):
        return self

    def color(self):
        return self

    def lightness(self):
        return self.lightness_value


class _FakeApp:
    def __init__(self, scheme, lightness=30):
        self.hints = _FakeHints(scheme)
        self._lightness = lightness

    def palette(self):
        return _FakePalette(self._lightness)

    def styleHints(self):
        return self.hints

    def setPalette(self, palette):  # ThemeManager applies a theme palette
        self.applied_palette = palette


def test_theme_follows_system_scheme_until_user_chooses():
    from PySide6.QtCore import Qt

    from nlsq.gui_qt.theme import ThemeManager

    app = _FakeApp(Qt.ColorScheme.Light)
    tm = ThemeManager(app)
    assert tm.current_theme.name == "light"
    assert app.hints.set_calls == []  # nothing forced

    seen = []
    tm.theme_changed.connect(lambda t: seen.append(t.name))
    app.hints.scheme = Qt.ColorScheme.Dark
    for slot in app.hints.colorSchemeChanged.slots:
        slot(Qt.ColorScheme.Dark)
    assert tm.current_theme.name == "dark" and seen == ["dark"]

    tm.set_theme("light")  # explicit choice wins over the OS from now on
    for slot in app.hints.colorSchemeChanged.slots:
        slot(Qt.ColorScheme.Dark)
    assert tm.current_theme.name == "light"
    assert tm.explicit


def test_theme_unknown_scheme_defaults_to_dark():
    from PySide6.QtCore import Qt

    from nlsq.gui_qt.theme import ThemeManager

    assert ThemeManager(_FakeApp(Qt.ColorScheme.Unknown)).current_theme.name == "dark"


# --- S2: a stale fit's error / abort is dropped --------------------------------


@pytest.mark.serial
def test_stale_fit_error_and_abort_are_not_reported(app_state, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    shown = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a))
    )
    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    aborted = []
    page.fit_aborted.connect(lambda: aborted.append(1))

    app_state.set_fit_running(True)
    page._fit_epoch = app_state.input_epoch
    app_state.input_epoch += 1  # data or model changed mid-fit

    page._on_fit_error("ValueError: boom")
    page._on_fit_aborted()

    assert shown == [] and aborted == []
    assert not app_state.state.fit_running
    assert page._run_btn.isEnabled()


@pytest.mark.serial
def test_current_fit_error_still_reported(app_state, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    shown = []
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a))
    )
    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    page._fit_epoch = app_state.input_epoch
    page._on_fit_error("ValueError: boom")
    assert len(shown) == 1


# --- S3: clipboard parser shares the page's cleaning --------------------------


def test_load_from_clipboard_strips_bom_and_comments():
    from nlsq.gui_qt.adapters.data_adapter import load_from_clipboard

    text = "﻿# exported by lab tool\n1\t2\n\n3\t4  # inline note\n"
    x, y, sigma = load_from_clipboard(text, {"columns": {"x": 0, "y": 1}})
    assert x.tolist() == [1.0, 3.0] and y.tolist() == [2.0, 4.0] and sigma is None


# --- W2: approximate chunked route is disclosed -------------------------------


def test_chunked_route_disclosure_reaches_gui_notes(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from tests.gui_qt.test_review_2026_11_04_gui import _exp_state

    real = fa.fit

    def chunked_like(**kw):
        logging.getLogger("nlsq.streaming.large_dataset").warning(
            "Chunked fitting is not equivalent to a full-batch fit"
        )
        return real(**kw)

    monkeypatch.setattr(fa, "fit", chunked_like)
    r = fa.run_fit(_exp_state())
    assert any("not equivalent" in n for n in r["gui_notes"])
