"""Round-2 GUI regressions: Guided/Advanced leakage, varargs models, CSV trailing
delimiter, and Clear not clearing AppState."""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QMessageBox

pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: None)


# --- 1. Advanced-mode edits must not leak into Guided runs ------------------


def _fitting_page(app_state, qtbot, monkeypatch):
    from nlsq.gui_qt.adapters.model_adapter import get_model
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    x = np.arange(6.0)
    app_state.set_data(x, 2 * x + 3)
    app_state.set_model(
        "builtin", {"name": "linear"}, get_model("builtin", {"name": "linear"})
    )
    monkeypatch.setattr(QThread, "start", lambda self: None)
    return page


def test_guided_run_ignores_advanced_only_settings(app_state, qtbot, monkeypatch):
    page = _fitting_page(app_state, qtbot, monkeypatch)
    page._advanced_radio.click()
    page._advanced_options._max_fev_spin.setValue(2)
    assert app_state.state.max_function_evals == 2
    page._guided_radio.click()
    page._preset_combo.setCurrentText("Fast")
    page.run_fit()
    assert app_state.state.max_function_evals == 2000


def test_reset_restores_advanced_defaults(app_state, qtbot, monkeypatch):
    page = _fitting_page(app_state, qtbot, monkeypatch)
    page._advanced_radio.click()
    page._advanced_options._max_fev_spin.setValue(2)
    page._guided_radio.click()
    page.reset()  # preset already "Robust": must still restore defaults
    assert app_state.state.max_function_evals == 2000
    assert page._advanced_options.get_options()["max_function_evals"] == 2000


# --- 2. varargs-only custom models are rejected ------------------------------


def test_varargs_custom_model_rejected(app_state, qtbot, monkeypatch):
    from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

    page = ModelSelectionPage(app_state)
    qtbot.addWidget(page)
    errors = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *a, **k: errors.append(a[2] if len(a) > 2 else ""),
    )
    page._custom_radio.click()
    page._code_editor.set_code("def model(x, *p):\n    return p[0] * x + p[1]\n")
    page._on_apply()
    assert app_state.state.model_func is None
    assert errors and "explicit parameters" in errors[-1]


# --- 3. trailing delimiter in CSV --------------------------------------------


def test_csv_trailing_delimiter_drops_blank_column(app_state, qtbot, monkeypatch):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)
    page._clipboard_text.setPlainText("x,y,\n1,2,\n2,4,\n3,6,\n4,8,\n")
    page._on_parse_clipboard()
    assert page._column_names == ["x", "y"]
    assert page._column_selector.get_selection().get("sigma") is None
    assert page._apply_btn.isEnabled()


def test_named_all_nan_column_is_kept(app_state, qtbot):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)
    page._clipboard_text.setPlainText("x,y,note\n1,2,\n2,4,\n3,6,\n")
    page._on_parse_clipboard()
    assert page._column_names == ["x", "y", "note"]


# --- 4. Clear must clear AppState -------------------------------------------


def test_clear_button_clears_app_state(app_state, qtbot):
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    page = DataLoadingPage(app_state)
    qtbot.addWidget(page)
    page._clipboard_text.setPlainText("1,2\n2,4\n3,6\n")
    page._on_parse_clipboard()
    page._on_apply()
    assert app_state.state.xdata is not None
    page._clear_btn.click()
    assert app_state.state.xdata is None
    assert app_state.state.ydata is None
