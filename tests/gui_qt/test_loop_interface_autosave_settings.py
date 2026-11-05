"""Crash recovery preserves the configuration chosen before the crash."""

import pytest

from nlsq.gui_qt.app_state import AppState
from nlsq.gui_qt.autosave import AutosaveManager

pytestmark = pytest.mark.serial


@pytest.mark.parametrize(
    "model_type,config",
    [("polynomial", {"degree": 2}), ("custom", {"code": "def model(x,a): return a*x"})],
)
def test_autosave_roundtrip_restores_model_and_fit_settings(model_type, config):
    source = AppState()
    source.set_model(model_type, config)
    source.state.loss = "soft_l1"
    source.state.gtol = 1e-5
    source.state.workflow = "auto_global"
    source.state.n_starts = 7
    source.state.max_function_evals = 432
    source.state.fit_running = True
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = source
    serialized = manager._serialize_state()

    recovered = AppState()
    manager._app_state = recovered
    manager._restore_state(serialized)

    assert recovered.state.model_type == model_type
    assert recovered.state.model_config == config
    for key in ("loss", "gtol", "workflow", "n_starts", "max_function_evals"):
        assert getattr(recovered.state, key) == getattr(source.state, key)
    if model_type == "custom":
        assert recovered.state.model_func is None
    else:
        assert callable(recovered.state.model_func)
    assert recovered.state.fit_running is False


def test_main_window_recovery_hydrates_fit_controls(app_state, qtbot, monkeypatch):
    import numpy as np
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.main_window import MainWindow

    monkeypatch.setenv("NLSQ_SAFE_MODE", "1")
    window = MainWindow(app_state)
    qtbot.addWidget(window)
    source = AppState()
    source.set_model("polynomial", {"degree": 1})
    source.set_data(np.arange(6.0), 2 * np.arange(6.0) + 3)
    source.state.mode = "advanced"
    source.state.gtol = 1e-5
    source.state.max_function_evals = 432
    source.state.auto_p0 = False
    source.state.p0 = [2.0, 3.0]
    source.state.bounds = ([0.0, 0.0], [5.0, 5.0])
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = source
    payload = manager._serialize_state()
    monkeypatch.setattr(window._autosave, "check_recovery", lambda: True)
    monkeypatch.setattr(
        window._autosave,
        "recover",
        lambda: (window._autosave._restore_state(payload), True)[1],
    )
    monkeypatch.setattr(window._autosave, "clear_recovery", lambda: None)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    window._check_recovery()
    page = window._pages["fitting_options"]
    assert app_state.state.model_func is not None
    assert page._advanced_radio.isChecked()
    assert page._advanced_options.get_options()["gtol"] == 1e-5
    assert page._param_config.get_values()[0] == [2, 3]
    monkeypatch.setattr(QThread, "start", lambda self: None)
    page.run_fit()
    assert app_state.state.gtol == 1e-5
    assert app_state.state.max_function_evals == 432
    assert app_state.state.p0 == [2, 3]
    page._cleanup_thread()
    app_state.set_fit_aborted()


def test_custom_recovery_waits_for_explicit_apply(app_state, qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.adapters import model_adapter
    from nlsq.gui_qt.main_window import MainWindow

    monkeypatch.setenv("NLSQ_SAFE_MODE", "1")
    window = MainWindow(app_state)
    qtbot.addWidget(window)
    code = "def model(x, a, b):\n    return a * x + b\n"
    original_parser = model_adapter.parse_custom_model_string
    calls = []

    def parse(*args, **kwargs):
        calls.append(args)
        return original_parser(*args, **kwargs)

    monkeypatch.setattr(model_adapter, "parse_custom_model_string", parse)
    window._autosave._restore_state(
        {
            "model_config": {"code": code, "function": "model"},
            "settings": {"model_type": "custom", "auto_p0": False},
            "p0": [2.0, 3.0],
        }
    )
    assert not calls
    assert app_state.state.model_func is None
    model_page = window._pages["model_selection"]
    assert model_page._code_editor.get_code() == code
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    model_page._on_apply()
    assert len(calls) == 1
    assert callable(app_state.state.model_func)
    params = window._pages["fitting_options"]._param_config
    assert params.get_values()[0] == [2, 3]


def test_fit_mode_and_preset_changes_reach_autosave(app_state, qtbot):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    page._advanced_radio.click()
    page._preset_combo.setCurrentText("Fast")
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = app_state
    settings = manager._serialize_state()["settings"]
    assert settings["mode"] == "advanced"
    assert settings["preset"] == "Fast"


def test_optional_workflow_settings_survive_recovery():
    source = AppState()
    source.state.checkpoint_dir = "/tmp/checkpoints"
    source.state.defense_preset = "relaxed"
    source.state.batch_max_workers = 3
    source.state.transforms = {"amplitude": "log"}
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = source
    data = manager._serialize_state()
    target = AppState()
    manager._app_state = target
    manager._restore_state(data)
    for key in ("checkpoint_dir", "defense_preset", "batch_max_workers", "transforms"):
        assert getattr(target.state, key) == getattr(source.state, key)


def test_advanced_edits_before_fit_reach_autosave(app_state, qtbot):
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    page._advanced_radio.click()
    page._advanced_options._max_fev_spin.setValue(432)
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = app_state
    assert manager._serialize_state()["settings"]["max_function_evals"] == 432


def test_parameter_edits_before_fit_reach_autosave(app_state, qtbot):
    from nlsq.gui_qt.adapters.model_adapter import get_model
    from nlsq.gui_qt.pages.fitting_options import FittingOptionsPage

    page = FittingOptionsPage(app_state)
    qtbot.addWidget(page)
    app_state.set_model(
        "builtin", {"name": "linear"}, get_model("builtin", {"name": "linear"})
    )
    page._param_config._auto_p0_check.setChecked(False)
    page._param_config._table.cellWidget(0, 1).setValue(2.5)
    manager = AutosaveManager.__new__(AutosaveManager)
    manager._app_state = app_state
    assert manager._serialize_state()["p0"] == [2.5, 1.0]
