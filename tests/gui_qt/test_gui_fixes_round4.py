"""Round-4 GUI regressions: Advanced-mode clamping, ignored streaming/defense
controls, keyword-only custom model parameters."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


# --- 1. Advanced mode must round-trip restored settings ---------------------


@pytest.mark.serial
def test_advanced_restore_roundtrips_exactly(app_state, qtbot):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    s = app_state.state
    s.mode = "advanced"
    s.workflow = "hpc"
    s.scale_factor = 0.125
    s.gtol = 1e-16
    s.n_starts = 1
    s.max_function_evals = 5_000_000

    page._restore_session_controls()
    page._sync_options_to_state(page._advanced_options.get_options())

    assert s.workflow == "hpc"
    assert s.scale_factor == 0.125
    assert s.gtol == 1e-16
    assert s.n_starts == 1
    assert s.max_function_evals == 5_000_000


@pytest.mark.serial
def test_workflow_checkbox_toggles_between_auto_and_global(qtbot):
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    w.set_options({"workflow": "hpc"})
    assert w.get_options()["workflow"] == "hpc"
    w._multistart_check.setChecked(True)
    assert w.get_options()["workflow"] == "auto_global"
    w._multistart_check.setChecked(False)
    assert w.get_options()["workflow"] == "auto"


# --- 2. Streaming / Defense controls: noted, and "relaxed" selectable -------


@pytest.mark.serial
def test_relaxed_defense_preset_selectable(qtbot):
    from nlsq.gui_qt.presets import PRESETS
    from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget

    w = AdvancedOptionsWidget()
    qtbot.addWidget(w)
    w.set_options({"defense_preset": "relaxed"})
    assert w.get_options()["defense_preset"] == "relaxed"
    wanted = {p["defense_preset"] for p in PRESETS.values() if "defense_preset" in p}
    items = {w._defense_preset.itemText(i) for i in range(w._defense_preset.count())}
    assert wanted <= items


def _run(**overrides):
    import nlsq.gui_qt.adapters.fit_adapter as fa
    from tests.gui_qt.test_review_2026_11_04_gui import _exp_state

    s = _exp_state()
    for k, v in overrides.items():
        setattr(s, k, v)
    return fa.run_fit(s)


def test_layer_off_is_noted_on_result():
    r = _run(layer3_enabled=False)
    notes = " ".join(r["gui_notes"])
    assert "hybrid" in notes and "defense" in notes.lower()


def test_streaming_controls_are_noted_on_result():
    r = _run(normalize=False, warmup_iterations=50, defense_preset="relaxed")
    notes = " ".join(r["gui_notes"])
    for key in ("normalize", "warmup", "defense preset"):
        assert key in notes


def test_default_streaming_controls_add_no_note():
    r = _run()
    assert "gui_notes" not in r or not any("hybrid" in n for n in r["gui_notes"])


# --- 3. Keyword-only custom-model parameters are rejected -------------------


def test_keyword_only_model_is_rejected():
    from nlsq.gui_qt.adapters.model_adapter import parse_custom_model_string

    code = "def f(x, a, *, b=1.0):\n    return a * x + b\n"
    with pytest.raises(ValueError, match="keyword-only"):
        parse_custom_model_string(code, "f")


@pytest.mark.serial
def test_keyword_only_model_sets_no_model_on_page(qtbot, app_state, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))
    page = ModelSelectionPage(app_state)
    qtbot.addWidget(page)
    page._code_editor.set_code("def f(x, a, *, b=1.0):\n    return a * x + b\n")
    page._function_combo.clear()
    page._function_combo.addItem("f")
    page._on_function_selected(0)
    assert page._current_model is None
    assert "keyword-only" in (page._custom_error or "")
