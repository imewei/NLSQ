"""Regression tests for the 2026-11-18 round-2 three-brain review (F7: adapters/state/autosave/export)."""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml


def _model(x, a):
    return a * x


# --- autosave ---------------------------------------------------------------


@pytest.mark.serial
def test_simultaneous_first_autosaves_do_not_overwrite(
    app_state, qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.app_state import AppState

    primary = tmp_path / "session.json"
    monkeypatch.setattr(autosave, "AUTOSAVE_DIR", tmp_path)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", primary)
    # one process stands in for two: a snapshot with another token is "live"
    monkeypatch.setattr(
        autosave,
        "_owned_by_live_session",
        lambda data, token: (
            bool(data) and (data.get("owner") or {}).get("token") != token
        ),
    )
    other = AppState()
    app_state.set_data(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    other.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    a = autosave.AutosaveManager(app_state)
    b = autosave.AutosaveManager(other)
    # both pick their target before either has published
    assert a._target_file() == primary == b._target_file()
    a.save_now()
    b.save_now()
    assert json.loads(primary.read_text())["ydata"] == [1.0, 2.0]
    assert json.loads(b._target_file().read_text())["ydata"] == [3.0, 4.0]
    assert b._target_file() != primary
    b.clear_recovery()
    assert primary.exists()  # B's close must not delete A's snapshot


@pytest.mark.serial
def test_crash_hook_marks_this_sessions_file(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    import nlsq.gui_qt as gui
    from nlsq.gui_qt import autosave

    shared = tmp_path / "session.json"
    mine = tmp_path / "session.tok.json"
    shared.write_text('{"timestamp": "other", "ydata": [9], "owner": {"token": "o"}}')
    mine.write_text('{"timestamp": "mine", "ydata": [1], "owner": {"token": "me"}}')
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", shared)
    monkeypatch.setattr(autosave, "_active_target", mine)
    monkeypatch.setattr(autosave, "_active_token", "me")
    monkeypatch.setattr(
        QApplication, "instance", lambda: MagicMock(activeWindow=lambda: None)
    )
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    gui._exception_hook(RuntimeError, RuntimeError("boom"), None)
    assert "crash_info" not in json.loads(shared.read_text())
    assert json.loads(mine.read_text())["crash_info"]["error"] == "boom"


# --- app state --------------------------------------------------------------


@pytest.mark.serial
def test_set_model_without_callable_drops_old_callable(app_state):
    app_state.set_model("builtin", {"name": "linear"}, _model)
    app_state.set_model("builtin", {"name": "gaussian"})
    assert app_state.state.model_func is None


# --- data adapter -----------------------------------------------------------


def test_quoted_header_separators_are_not_counted():
    from nlsq.gui_qt.adapters.data_adapter import parse_table_text

    rows, header = parse_table_text('"time; seconds; source; 1",signal\n1,2\n2,4')
    assert rows == [[1.0, 2.0], [2.0, 4.0]]
    assert header == ["time; seconds; source; 1", "signal"]


def test_clipboard_accepts_multichar_delimiter():
    from nlsq.gui_qt.adapters.data_adapter import load_from_clipboard

    x, y, _ = load_from_clipboard("1||10\n2||20", {"delimiter": "||"})
    assert x.tolist() == [1.0, 2.0]
    assert y.tolist() == [10.0, 20.0]


def test_comment_char_inside_quoted_header_is_kept():
    from nlsq.gui_qt.adapters.data_adapter import parse_table_text

    rows, header = parse_table_text('x,"signal #1"  # trailing comment\n1,2')
    assert header == ["x", "signal #1"]
    assert rows == [[1.0, 2.0]]


def test_load_from_unnamed_tempfile_handle():
    from nlsq.gui_qt.adapters.data_adapter import load_from_file

    with tempfile.TemporaryFile() as fh:
        fh.write(b"1,2\n2,4\n3,6\n")
        fh.seek(0)
        x, y, _ = load_from_file(fh, {"format": "csv", "columns": {"x": 0, "y": 1}})
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [2.0, 4.0, 6.0]


# --- config adapter / session state -----------------------------------------


def test_validate_yaml_rejects_unknown_workflow():
    from nlsq.gui_qt.adapters.config_adapter import validate_yaml_config

    ok, err = validate_yaml_config("fitting: {workflow: bogus}")
    assert not ok
    assert "bogus" in err


def test_quoted_false_auto_bounds_rejected():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    with pytest.raises(ValueError, match="auto_bounds"):
        load_yaml_config('model: {auto_bounds: "false"}')


def test_yaml_export_scalar_bounds_broadcast():
    from nlsq.gui_qt.adapters.config_adapter import (
        export_yaml_config,
        load_yaml_config,
    )

    state = load_yaml_config("fitting: {bounds: {lower: 0, upper: 5}}")
    state.p0 = [1.0, 1.0]
    state.bounds = ([0.0], [5.0])
    params = yaml.safe_load(export_yaml_config(state))["model"]["parameters"]
    assert [p["bounds"] for p in params] == [[0.0, 5.0], [0.0, 5.0]]


def test_yaml_export_keeps_transform_names_without_applied_model():
    from nlsq.gui_qt.adapters.config_adapter import (
        export_yaml_config,
        load_yaml_config,
    )

    state = load_yaml_config(
        "model:\n  type: builtin\n  name: linear\n  parameters:\n"
        "    - {name: a, initial: 1.0, transform: log}\n"
        "    - {name: b, initial: 0.5}\n"
    )
    state.model_func = None
    out = yaml.safe_load(export_yaml_config(state))["model"]["parameters"]
    assert out[0]["transform"] == "log"


# --- model adapter ----------------------------------------------------------


def test_underscore_prefixed_model_can_be_loaded():
    from nlsq.gui_qt.adapters.model_adapter import (
        list_functions_in_module,
        parse_custom_model_string,
    )

    code = "def _model(x, a):\n    return a * x\n"
    assert list_functions_in_module(code) == ["_model"]
    func, names = parse_custom_model_string(code, "_model")
    assert names == ["a"]
    assert func(2.0, 3.0) == 6.0


# --- fit adapter / export ---------------------------------------------------


def test_hpc_without_finite_bounds_falls_back_like_auto_global():
    from nlsq.gui_qt.adapters.fit_adapter import _resolve_workflow

    notes: list[str] = []
    assert _resolve_workflow("hpc", None, notes) == "auto"
    assert notes


def _page(app_state, qtbot, state):
    from nlsq.gui_qt.pages.export import ExportPage

    app_state._state = state
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    return page


@pytest.mark.serial
def test_export_script_model_named_fit_does_not_shadow_solver(app_state, qtbot):
    from nlsq.gui_qt.session_state import SessionState

    x = np.array([0.0, 1.0, 2.0, 3.0])
    s = SessionState(xdata=x, ydata=2 * x + 1, model_func=_model)
    s.model_type = "custom"
    s.model_config = {
        "code": "def fit(x, a, b):\n    return a * x + b\n",
        "function": "fit",
    }
    s.p0 = [1.0, 1.0]
    code = _page(app_state, qtbot, s).generate_python_code()
    ns: dict = {}
    # the user's `fit` is defined after the import and must not replace the solver
    exec(compile(code.split("# Run curve fitting")[0], "<script>", "exec"), ns)
    assert ns["nlsq_fit"] is not ns["fit"]
    assert "result = nlsq_fit(" in code


@pytest.mark.serial
def test_export_script_includes_checkpoint_dir(app_state, qtbot):
    from nlsq.gui_qt.session_state import SessionState

    x = np.linspace(0, 1, 5)
    s = SessionState(xdata=x, ydata=2 * x, model_func=_model, workflow="hpc")
    s.p0 = [1.0]
    s.bounds = ([0.0], [10.0])
    s.n_starts = 3
    s.enable_checkpoints = True
    s.checkpoint_dir = "/some/dir"
    assert (
        "checkpoint_dir='/some/dir',"
        in _page(app_state, qtbot, s).generate_python_code()
    )


# --- wave 3: no-hardlink filesystems, crash before first save ---------------


@pytest.mark.serial
def test_first_autosaves_without_hardlinks_use_separate_files(
    app_state, qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.app_state import AppState

    primary = tmp_path / "session.json"
    monkeypatch.setattr(autosave, "AUTOSAVE_DIR", tmp_path)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", primary)

    def no_link(*args, **kwargs):
        raise OSError(95, "Operation not supported")

    monkeypatch.setattr(autosave.os, "link", no_link)
    other = AppState()
    app_state.set_data(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    other.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    a = autosave.AutosaveManager(app_state)
    b = autosave.AutosaveManager(other)
    assert a._target_file() == primary == b._target_file()
    a.save_now()
    b.save_now()
    assert a._target_file() != b._target_file()
    assert json.loads(a._target_file().read_text())["ydata"] == [1.0, 2.0]
    assert json.loads(b._target_file().read_text())["ydata"] == [3.0, 4.0]
    b.clear_recovery()
    assert a._target_file().exists()  # B's close leaves A's snapshot
    # a later autosave keeps using the private file
    a.save_now()
    assert not primary.exists() or json.loads(primary.read_text())["ydata"] != [
        3.0,
        4.0,
    ]


@pytest.mark.serial
def test_crash_hook_before_first_save_touches_no_file(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    import nlsq.gui_qt as gui
    from nlsq.gui_qt import autosave

    shared = tmp_path / "session.json"
    original = '{"timestamp": "other", "ydata": [9], "owner": {"pid": 1, "token": "x"}}'
    shared.write_text(original)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", shared)
    monkeypatch.setattr(autosave, "_active_target", None)
    monkeypatch.setattr(
        QApplication, "instance", lambda: MagicMock(activeWindow=lambda: None)
    )
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    gui._exception_hook(RuntimeError, RuntimeError("boom"), None)
    assert shared.read_text() == original


# --- wave 4: two sessions racing for a STALE shared snapshot ------------------


def _stale_primary(tmp_path, monkeypatch):
    from nlsq.gui_qt import autosave

    primary = tmp_path / "session.json"
    monkeypatch.setattr(autosave, "AUTOSAVE_DIR", tmp_path)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", primary)
    stale = '{"timestamp": "old", "ydata": [7], "owner": {"pid": 2147483646, "token": "dead"}}'
    primary.write_text(stale)
    return primary, stale


@pytest.mark.serial
def test_two_sessions_do_not_both_claim_a_stale_snapshot(
    app_state, qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.app_state import AppState

    primary, stale = _stale_primary(tmp_path, monkeypatch)
    other = AppState()
    app_state.set_data(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    other.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    a = autosave.AutosaveManager(app_state)
    b = autosave.AutosaveManager(other)
    # both chose the shared path before either published
    assert a._target_file() == primary == b._target_file()
    a.save_now()
    b.save_now()
    ta, tb = a._target_file(), b._target_file()
    assert ta != tb
    assert primary.read_text() == stale  # nobody replaced the stale snapshot
    assert json.loads(ta.read_text())["ydata"] == [1.0, 2.0]
    assert json.loads(tb.read_text())["ydata"] == [3.0, 4.0]
    b.clear_recovery()  # loser closes
    assert primary.read_text() == stale
    assert ta.exists()
    assert not tb.exists()


@pytest.mark.serial
def test_close_before_first_save_never_deletes_another_sessions_file(
    app_state, qtbot, tmp_path, monkeypatch
):
    from nlsq.gui_qt import autosave

    primary, stale = _stale_primary(tmp_path, monkeypatch)
    mgr = autosave.AutosaveManager(app_state)
    assert mgr._target_file() == primary  # chosen, never published
    mgr.clear_recovery()
    assert primary.read_text() == stale


@pytest.mark.serial
def test_crash_hook_checks_owner_token(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    import nlsq.gui_qt as gui
    from nlsq.gui_qt import autosave

    shared = tmp_path / "session.json"
    original = '{"timestamp": "w", "ydata": [9], "owner": {"token": "winner"}}'
    shared.write_text(original)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", shared)
    monkeypatch.setattr(autosave, "_active_target", shared)  # loser's cached path
    monkeypatch.setattr(autosave, "_active_token", "loser")
    monkeypatch.setattr(
        QApplication, "instance", lambda: MagicMock(activeWindow=lambda: None)
    )
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    gui._exception_hook(RuntimeError, RuntimeError("boom"), None)
    assert shared.read_text() == original
