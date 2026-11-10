"""Round-22 GUI regressions."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.mark.serial
def test_export_custom_model_without_function_name(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.export import ExportPage

    st = AppState()
    st.set_data(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    st.set_model("custom", {"code": "def my_line(x, a):\n    return a * x\n"})
    code = ExportPage(st).generate_python_code()
    assert "my_line" in code


def test_yaml_import_coerces_string_scalars():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    st = load_yaml_config(
        """
model:
  type: builtin
  name: exponential_decay
  parameters:
    - name: a
      initial: 2
    - name: b
      initial: 5e-1
      bounds: [1e-3, 1e1]
"""
    )
    assert all(isinstance(v, float) for v in st.p0)
    assert st.bounds[0][1] == pytest.approx(1e-3)
    assert all(isinstance(v, float) for v in st.bounds[1])


def test_yaml_import_non_numeric_p0_is_named():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    with pytest.raises(ValueError, match="numeric"):
        load_yaml_config(
            "model:\n  type: builtin\n  name: exponential_decay\n"
            "  parameters:\n    - name: a\n      initial: abc\n"
        )


@pytest.mark.serial
def test_autosave_not_started_before_recovery_decision(qapp, monkeypatch):
    from nlsq.gui_qt.main_window import MainWindow

    monkeypatch.delenv("NLSQ_SAFE_MODE", raising=False)
    started = []
    from nlsq.gui_qt.autosave import AutosaveManager

    monkeypatch.setattr(AutosaveManager, "start", lambda self: started.append(1))
    monkeypatch.setattr(AutosaveManager, "check_recovery", lambda self: False)
    from nlsq.gui_qt.app_state import AppState

    w = MainWindow(AppState())
    assert started == []  # deferred until _check_recovery runs
    w._check_recovery()
    assert started == [1]
    w.close()


@pytest.mark.serial
def test_non_bytes_window_geometry_does_not_break_startup(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.main_window import MainWindow

    w = MainWindow(AppState())
    w._settings.setValue("geometry", "garbage")
    w._settings.setValue("windowState", 5)
    w._restore_state()  # must not raise
    w._settings.remove("geometry")
    w._settings.remove("windowState")
    w.close()


@pytest.mark.serial
def test_autosave_not_restarted_after_close(qapp, monkeypatch):
    """A deferred recovery check firing after closeEvent must not restart autosave."""
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.autosave import AutosaveManager
    from nlsq.gui_qt.main_window import MainWindow

    monkeypatch.delenv("NLSQ_SAFE_MODE", raising=False)
    started = []
    monkeypatch.setattr(AutosaveManager, "start", lambda self: started.append(1))
    monkeypatch.setattr(AutosaveManager, "check_recovery", lambda self: False)
    w = MainWindow(AppState())
    w.close()
    w._check_recovery()  # the singleShot(0) that was still pending
    assert started == []


def test_builtin_dropdown_only_lists_existing_models():
    import nlsq.core.functions as F
    from nlsq.gui_qt.pages.model_selection import BUILTIN_MODELS

    for _label, name in BUILTIN_MODELS:
        assert callable(getattr(F, name, None)), name


@pytest.mark.serial
def test_model_page_reset_then_apply_has_model(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.model_selection import ModelSelectionPage

    page = ModelSelectionPage(AppState())
    page.reset()
    assert page._current_model is not None
    assert page._current_model_type == "builtin"


@pytest.mark.serial
def test_restored_2d_session_uses_surface_column_layout(qapp):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.pages.data_loading import DataLoadingPage

    st = AppState()
    page = DataLoadingPage(st)
    rng = np.random.default_rng(0)
    x = rng.random((2, 30))
    z = x[0] + 2 * x[1]
    st.set_data(x, z, None)
    page._on_app_state_data_changed()
    sel = page._column_selector.get_selection()
    assert (sel["x"], sel["y"], sel["z"]) == (0, 1, 2)
    assert page._column_names[:3] == ["x", "y", "z"]
