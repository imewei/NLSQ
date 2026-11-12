"""Regression tests: stale code export, null bounds on YAML reload, empty model combo."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.serial


def _fit_state(app_state, name):
    x = np.linspace(0.0, 2.0, 20)
    app_state.set_data(x, 2.0 * np.exp(-x))
    app_state.set_model("builtin", {"name": name})


def test_export_code_regenerates_after_model_change(
    app_state, qtbot, tmp_path, monkeypatch
):
    from PySide6.QtWidgets import QFileDialog

    from nlsq.gui_qt.pages.export import ExportPage

    page = ExportPage(app_state)
    qtbot.addWidget(page)
    _fit_state(app_state, "exponential_decay")
    page._on_generate_code()
    assert "exponential_decay" in page._code_preview.toPlainText()

    _fit_state(app_state, "gaussian")  # switch model, "refit"
    out = tmp_path / "fit_script.py"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    page._on_export_code()

    text = out.read_text(encoding="utf-8")
    assert "import gaussian" in text
    assert "exponential_decay" not in text
    assert "import gaussian" in page._code_preview.toPlainText()


def test_copy_code_regenerates_after_model_change(app_state, qtbot):
    from PySide6.QtGui import QGuiApplication

    from nlsq.gui_qt.pages.export import ExportPage

    page = ExportPage(app_state)
    qtbot.addWidget(page)
    _fit_state(app_state, "exponential_decay")
    page._on_generate_code()
    _fit_state(app_state, "gaussian")
    page._on_copy_code()
    assert "import gaussian" in QGuiApplication.clipboard().text()


def test_yaml_roundtrip_unbounded_sides_are_inf_not_nan():
    from nlsq.gui_qt.adapters.config_adapter import (
        export_yaml_config,
        load_yaml_config,
    )
    from nlsq.gui_qt.session_state import SessionState

    s = SessionState(bounds=([0, 0, 0], [None, None, None]), auto_p0=True)
    r = load_yaml_config(export_yaml_config(s))
    lower, upper = r.bounds
    assert lower == [0, 0, 0]
    assert upper == [np.inf] * 3
    assert not np.isnan(np.asarray(r.bounds, dtype=float)).any()


def test_yaml_flat_bounds_null_lower_is_minus_inf():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    r = load_yaml_config("fitting:\n  bounds: [null, 5]\n")
    assert r.bounds == ([-np.inf], [5.0])


def test_recovered_data_only_session_applies_builtin_model(
    qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

    monkeypatch.setattr(autosave, "AUTOSAVE_DIR", tmp_path)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", tmp_path / "autosave.json")

    src = AppState()
    x = np.linspace(0.0, 1.0, 10)
    src.set_data(x, 2.0 * x)  # data only: no model chosen
    autosave.AutosaveManager(src).save_now()

    fresh = AppState()
    page = ModelSelectionPage(fresh)
    qtbot.addWidget(page)
    assert autosave.AutosaveManager(fresh).recover()

    from PySide6.QtWidgets import QMessageBox

    # a modal dialog would block the offscreen run; critical == apply failed
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(
        QMessageBox, "critical", lambda *a, **k: pytest.fail(f"apply failed: {a[1:]}")
    )
    assert page._builtin_combo.currentIndex() >= 0
    page._on_apply()
    assert fresh.state.model_func is not None
