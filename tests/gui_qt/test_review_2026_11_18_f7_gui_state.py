"""Regression tests for the 2026-11-18 three-brain review (adapters/state/autosave/export)."""

from __future__ import annotations

import csv
import io
import json
import os

import numpy as np
import pytest

# --- adapters ---------------------------------------------------------------


def _spy_fit(monkeypatch):
    import nlsq.gui_qt.adapters.fit_adapter as fa

    captured: dict = {}

    def spy(**kw):
        captured.update(kw)
        return {"popt": [1.0]}

    monkeypatch.setattr(fa, "fit", spy)
    return captured


def _model(x, a):
    return a * x


def test_hpc_workflow_forwards_multistart_controls(monkeypatch):
    from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit

    captured = _spy_fit(monkeypatch)
    cfg = FitConfig(
        p0=[1.0],
        bounds=([0.0], [10.0]),
        workflow="hpc",
        n_starts=3,
        sampler="sobol",
        center_on_p0=False,
        scale_factor=4.0,
    )
    execute_fit(np.arange(1.0, 5.0), np.arange(1.0, 5.0), None, _model, cfg, None)
    assert captured["workflow"] == "hpc"
    assert captured["n_starts"] == 3
    assert captured["sampler"] == "sobol"
    assert captured["center_on_p0"] is False
    assert captured["scale_factor"] == 4.0


def test_checkpoint_dir_forwarded_for_hpc_and_noted_otherwise(monkeypatch):
    from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit

    captured = _spy_fit(monkeypatch)
    cfg = FitConfig(
        p0=[1.0],
        bounds=([0.0], [10.0]),
        workflow="hpc",
        enable_checkpoints=True,
        checkpoint_dir="/some/dir",
    )
    execute_fit(np.arange(1.0, 5.0), np.arange(1.0, 5.0), None, _model, cfg, None)
    assert captured["checkpoint_dir"] == "/some/dir"

    captured.clear()
    cfg = FitConfig(p0=[1.0], workflow="auto", enable_checkpoints=True)
    result = execute_fit(
        np.arange(1.0, 5.0), np.arange(1.0, 5.0), None, _model, cfg, None
    )
    assert "checkpoint_dir" not in captured
    assert any("checkpointing" in n for n in result["gui_notes"])


def test_create_fit_config_carries_checkpoint_settings():
    from nlsq.gui_qt.adapters.fit_adapter import create_fit_config_from_state
    from nlsq.gui_qt.session_state import SessionState

    s = SessionState(enable_checkpoints=True, checkpoint_dir="/c")
    cfg = create_fit_config_from_state(s)
    assert cfg.enable_checkpoints is True
    assert cfg.checkpoint_dir == "/c"


def test_one_iteration_fit_takes_a_step():
    from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit

    x = np.arange(1.0, 11.0)
    cfg = FitConfig(p0=[1.0], max_iterations=1, max_function_evals=100)
    result = execute_fit(x, 2 * x, None, _model, cfg, None)
    assert result.nfev >= 2
    assert abs(result.popt[0] - 1.0) > 0.1  # moved off p0


def test_custom_model_excludes_var_keyword_from_parameters():
    from nlsq.gui_qt.adapters.model_adapter import parse_custom_model_string

    _, names = parse_custom_model_string(
        "def m(x, a, **kwargs):\n    return a * x\n", "m"
    )
    assert names == ["a"]


def test_list_functions_ignores_nested_definitions():
    from nlsq.gui_qt.adapters.model_adapter import list_functions_in_module

    code = "def outer(x, a):\n    def inner(x):\n        return x\n    return a * x\n"
    assert list_functions_in_module(code) == ["outer"]


def test_quoted_csv_cells_parse():
    from nlsq.gui_qt.adapters.data_adapter import parse_table_text

    rows, header = parse_table_text('"x","y"\n"1","2"\n"2","4"\n')
    assert header == ["x", "y"]
    assert rows == [[1.0, 2.0], [2.0, 4.0]]


def test_validate_yaml_rejects_non_mapping_section():
    from nlsq.gui_qt.adapters.config_adapter import validate_yaml_config

    ok, err = validate_yaml_config("model: 8")
    assert not ok
    assert "model" in err


def test_yaml_fractional_max_iterations_rejected():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config

    with pytest.raises(ValueError, match="max_iterations"):
        load_yaml_config("fitting:\n  termination:\n    max_iterations: 2.8\n")
    state = load_yaml_config("fitting:\n  termination:\n    max_iterations: 7\n")
    assert state.max_iterations == 7


def test_yaml_export_keeps_parameter_names_and_transforms():
    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config
    from nlsq.gui_qt.session_state import SessionState, get_current_config

    s = SessionState(model_func=_model)
    s.p0 = [2.0]
    s.transforms = {"a": "log"}
    param = get_current_config(s)["model"]["parameters"][0]
    assert param["name"] == "a"
    assert param["transform"] == "log"

    import yaml

    back = load_yaml_config(yaml.safe_dump(get_current_config(s)))
    assert back.transforms == {"a": "log"}


# --- autosave ---------------------------------------------------------------


@pytest.fixture
def autosave_dir(tmp_path, monkeypatch):
    from nlsq.gui_qt import autosave

    monkeypatch.setattr(autosave, "AUTOSAVE_DIR", tmp_path)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", tmp_path / "session.json")
    return tmp_path


@pytest.mark.serial
def test_concurrent_sessions_keep_separate_snapshots(app_state, autosave_dir):
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    # session A is "another running process": same file format, live foreign pid
    other = AppState()
    other.set_data(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    a = AutosaveManager(other)
    a.save_now()
    data = json.loads(primary.read_text())
    data["owner"]["pid"] = os.getppid()  # a different, running process
    primary.write_text(json.dumps(data))

    app_state.set_data(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    b = AutosaveManager(app_state)
    b.save_now()
    assert json.loads(primary.read_text())["ydata"] == [1.0, 2.0]  # A untouched
    assert b._target_file() != primary
    assert json.loads(b._target_file().read_text())["ydata"] == [3.0, 4.0]
    assert b.check_recovery() is False  # A is running: nothing to recover

    b.clear_recovery()  # clean close of B
    assert primary.exists()  # A's only snapshot survives
    assert not b._target_file().exists()


@pytest.mark.serial
def test_recovered_snapshot_survives_until_first_autosave(app_state, autosave_dir):
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    primary.write_text(
        json.dumps({"timestamp": "t", "xdata": [1.0, 2.0], "ydata": [3.0, 4.0]})
    )
    mgr = AutosaveManager(app_state)
    assert mgr.check_recovery()
    assert mgr.recover()
    assert primary.exists()  # still the only copy of the recovered session
    mgr.save_now()  # first autosave goes to this session's own file
    # the stale shared file is never replaced in place; it is removed (identity
    # checked) once the new snapshot is on disk
    assert json.loads(mgr._target_file().read_text())["ydata"] == [3.0, 4.0]
    assert not primary.exists()


@pytest.mark.serial
def test_recovery_prompt_keeps_snapshot_when_accepted(
    app_state, qtbot, autosave_dir, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox

    from nlsq.gui_qt.main_window import MainWindow

    primary = autosave_dir / "session.json"
    primary.write_text(
        json.dumps({"timestamp": "t", "xdata": [1.0, 2.0], "ydata": [3.0, 4.0]})
    )
    monkeypatch.setenv("NLSQ_SAFE_MODE", "1")
    window = MainWindow(app_state)
    qtbot.addWidget(window)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    window._offer_recovery()
    assert list(app_state.state.ydata) == [3.0, 4.0]
    assert primary.exists()
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No
    )
    window._offer_recovery()  # declining discards the snapshot
    assert not primary.exists()


@pytest.mark.serial
def test_malformed_p0_does_not_half_restore(app_state, autosave_dir):
    from nlsq.gui_qt.autosave import AutosaveManager

    app_state.set_data(np.array([1.0]), np.array([4.0]))
    mgr = AutosaveManager(app_state)
    payload = {
        "timestamp": "t",
        "xdata": [1.0],
        "ydata": [6.0],
        "p0": [10**400],
    }
    (autosave_dir / "session.json").write_text(json.dumps(payload))
    mgr.recover()  # must not raise OverflowError
    assert app_state.state.p0 is None  # the bad value was ignored, not committed


def test_autosave_write_is_compact_json_and_fsynced(autosave_dir, monkeypatch):
    from nlsq.gui_qt import autosave

    synced = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    autosave._write_autosave({"a": [1, 2]})
    assert synced
    assert (autosave_dir / "session.json").read_text() == '{"a": [1, 2]}'


# --- export -----------------------------------------------------------------


def _state_with_model(model, p0, x, y, **kw):
    from nlsq.gui_qt.session_state import SessionState

    s = SessionState(xdata=x, ydata=y, model_func=model, **kw)
    s.p0 = p0
    return s


@pytest.mark.serial
def test_export_script_hpc_includes_multistart_controls(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 1, 5)
    s = _state_with_model(_model, [1.0], x, 2 * x, workflow="hpc")
    s.bounds = ([0.0], [10.0])
    s.n_starts = 3
    s.sampler = "sobol"
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    code = page.generate_python_code()
    assert "workflow = 'hpc'" in code
    assert "n_starts=3," in code
    assert "sampler='sobol'," in code


@pytest.mark.serial
def test_export_script_absolute_sigma_without_sigma(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 1, 5)
    app_state._state = _state_with_model(_model, [1.0], x, 2 * x, absolute_sigma=True)
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    assert "absolute_sigma=True," in page.generate_python_code()


@pytest.mark.serial
def test_export_script_polynomial_uses_least_squares_p0(app_state, qtbot):
    from nlsq.core.functions import polynomial
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(-2, 2, 21)
    s = _state_with_model(polynomial(2), None, x, 4 * x * x - 7 * x + 8)
    s.model_type = "polynomial"
    s.model_config = {"degree": 2}
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    code = page.generate_python_code()
    assert "p0 = _poly(2).estimate_p0(xdata, ydata)" in code
    assert "[1.0] * 3" not in code


@pytest.mark.serial
def test_export_script_max_nfev_matches_gui_fit(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 1, 5)
    s = _state_with_model(_model, [1.0], x, 2 * x, max_iterations=5)
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    assert "max_nfev=6," in page.generate_python_code()


@pytest.mark.serial
def test_export_data_csv_converts_model_output_to_numpy_once(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    class LazyArray:
        """Like a JAX array: cheap __array__, very slow per-element iteration."""

        def __init__(self, values):
            self._values = values

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self._values, dtype=dtype)

        def __iter__(self):
            raise AssertionError("iterated element by element")

        def __len__(self):
            return len(self._values)

    x = np.linspace(0, 1, 6)
    s = _state_with_model(_model, [2.0], x, 2 * x)
    s.fit_result = type("R", (), {"popt": np.array([2.0])})()
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    page._compute_y_fit = lambda: LazyArray(2.0 * x)
    rows = list(csv.reader(io.StringIO("\n".join(page._data_csv_lines()))))
    assert rows[0] == ["x", "y", "y_fit", "residuals"]
    assert float(rows[2][2]) == pytest.approx(0.4)


@pytest.mark.serial
def test_json_export_includes_covariance(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 1, 6)
    s = _state_with_model(_model, [2.0], x, 2 * x)
    s.fit_result = type(
        "R",
        (),
        {"popt": np.array([2.0, 1.0]), "pcov": np.array([[0.2, -0.3], [-0.3, np.inf]])},
    )()
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    data = json.loads(page._generate_json())
    assert data["covariance"] == [[0.2, -0.3], [-0.3, None]]


# --- wave 2 (Agy) -------------------------------------------------------------


def test_scalar_yaml_bounds_broadcast_to_every_parameter():
    import jax.numpy as jnp

    from nlsq.gui_qt.adapters.config_adapter import load_yaml_config
    from nlsq.gui_qt.adapters.fit_adapter import FitConfig, execute_fit

    state = load_yaml_config("fitting:\n  bounds:\n    lower: -10\n    upper: 10\n")
    assert state.bounds == ([-10.0], [10.0])  # single entry from a scalar bound

    def gauss(x, a, b, c):
        return a * jnp.exp(-((x - b) ** 2) / (2 * c**2))

    x = np.linspace(-5, 5, 50)
    cfg = FitConfig(p0=[1.0, 0.5, 1.5], bounds=state.bounds)
    result = execute_fit(x, np.asarray(gauss(x, 2.0, 0.0, 1.0)), None, gauss, cfg, None)
    np.testing.assert_allclose(result.popt, [2.0, 0.0, 1.0], atol=1e-5)


@pytest.mark.serial
def test_export_script_scalar_bounds_are_scalars(app_state, qtbot):
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.linspace(0, 1, 5)
    s = _state_with_model(lambda x, a, b: a * x + b, [1.0, 1.0], x, 2 * x)
    s.bounds = ([-10.0], [10.0])
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    assert "bounds = (-10.0, 10.0)" in page.generate_python_code()


@pytest.mark.serial
def test_export_script_polynomial_p0_survives_rank_deficient_design(app_state, qtbot):
    from nlsq.core.functions import polynomial
    from nlsq.gui_qt.pages.export import ExportPage

    x = np.zeros(5)
    y = np.arange(5.0)
    s = _state_with_model(polynomial(2), None, x, y)
    s.model_type = "polynomial"
    s.model_config = {"degree": 2}
    app_state._state = s
    page = ExportPage(app_state)
    qtbot.addWidget(page)
    code = page.generate_python_code()
    namespace: dict = {}
    exec(code.split("# Run curve fitting")[0], namespace)  # data + p0 section only
    assert namespace["p0"] == polynomial(2).estimate_p0(x, y) == [0.0, 0.0, 2.0]


# --- wave 3 -------------------------------------------------------------------


def _snapshot(path, owner, ydata, timestamp="2026-01-01T00:00:00"):
    body = {"timestamp": timestamp, "xdata": [1.0, 2.0], "ydata": ydata}
    if owner is not None:
        body["owner"] = owner
    path.write_text(json.dumps(body))


@pytest.mark.serial
def test_malformed_owner_does_not_block_other_candidates(app_state, autosave_dir):
    from nlsq.gui_qt.autosave import AutosaveManager

    _snapshot(autosave_dir / "session.json", "broken", [9.0, 9.0], "2026-01-02")
    _snapshot(autosave_dir / "session.good.json", None, [3.0, 4.0], "2026-01-01")
    mgr = AutosaveManager(app_state)
    assert mgr.check_recovery()
    assert mgr.recover()


@pytest.mark.serial
def test_stale_recovery_prompt_does_not_delete_swapped_snapshot(
    app_state, autosave_dir
):
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    _snapshot(primary, None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()  # instance A opens the prompt on the stale snapshot
    # instance B (another live process) declines it and writes its own snapshot
    primary.unlink()
    _snapshot(primary, {"pid": os.getppid(), "token": "b"}, [7.0, 8.0])
    a.clear_recovery()  # A declines the earlier prompt
    assert json.loads(primary.read_text())["ydata"] == [7.0, 8.0]  # B's file kept
    assert a.recover() is False  # and A cannot adopt the swapped file either


# --- wave 5 -------------------------------------------------------------------


@pytest.mark.parametrize(
    "pid", [10**100, -1, 0, True, 1.0, "123", None, 2**31], ids=repr
)
def test_liveness_probe_is_total_for_malformed_pids(pid):
    from nlsq.gui_qt.autosave import _owned_by_live_session

    assert _owned_by_live_session({"owner": {"pid": pid, "token": "x"}}, "me") is False


def test_liveness_probe_handles_psutil_errors(monkeypatch):
    import psutil

    from nlsq.gui_qt.autosave import _owned_by_live_session

    def boom(pid):
        raise OverflowError

    monkeypatch.setattr(psutil, "pid_exists", boom)
    assert (
        _owned_by_live_session({"owner": {"pid": os.getppid(), "token": "x"}}, "m")
        is False
    )


@pytest.mark.serial
def test_huge_pid_snapshot_does_not_abort_discovery(app_state, autosave_dir):
    from nlsq.gui_qt.autosave import AutosaveManager

    _snapshot(autosave_dir / "session.json", {"pid": 10**100, "token": "t"}, [9.0, 9.0])
    _snapshot(autosave_dir / "session.good.json", None, [3.0, 4.0], "2025-01-01")
    mgr = AutosaveManager(app_state)
    assert mgr.check_recovery()


@pytest.mark.serial
def test_unlink_race_between_check_and_remove_keeps_live_snapshot(
    app_state, autosave_dir, monkeypatch
):
    """A live instance installs a new snapshot after the identity check passed."""
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    _snapshot(primary, None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()

    real_rename = os.rename

    def racing_rename(src, dst):
        # B atomically replaces the file right before A's rename takes it
        tmp = autosave_dir / "b.tmp"
        _snapshot(tmp, {"pid": os.getppid(), "token": "b"}, [7.0, 8.0])
        os.replace(tmp, primary)  # os.rename refuses an existing target on Windows
        return real_rename(src, dst)

    monkeypatch.setattr(autosave.os, "rename", racing_rename)
    a.clear_recovery()
    monkeypatch.undo()
    assert json.loads(primary.read_text())["ydata"] == [7.0, 8.0]  # restored
    assert not list(autosave_dir.glob("*.rm"))


@pytest.mark.serial
def test_unlink_race_when_destination_retaken_leaves_tombstone(
    app_state, autosave_dir, monkeypatch
):
    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    _snapshot(primary, None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()

    real_rename, real_link = os.rename, os.link

    def racing_rename(src, dst):
        tmp = autosave_dir / "b.tmp"
        _snapshot(tmp, {"pid": os.getppid(), "token": "b"}, [7.0, 8.0])
        os.replace(tmp, primary)  # os.rename refuses an existing target on Windows
        return real_rename(src, dst)

    def racing_link(src, dst):
        # B autosaves again before A can restore: the path is taken
        _snapshot(primary, {"pid": os.getppid(), "token": "b"}, [10.0, 11.0])
        return real_link(src, dst)

    monkeypatch.setattr(autosave.os, "rename", racing_rename)
    monkeypatch.setattr(autosave.os, "link", racing_link)
    a.clear_recovery()
    monkeypatch.undo()
    assert json.loads(primary.read_text())["ydata"] == [10.0, 11.0]  # newest kept
    assert len(list(autosave_dir.glob("*.rm"))) == 1  # older B file, ignored
    assert a.check_recovery() is False  # B is live; tombstone never offered


@pytest.mark.serial
def test_matching_snapshot_is_removed_without_leftovers(app_state, autosave_dir):
    from nlsq.gui_qt.autosave import AutosaveManager

    primary = autosave_dir / "session.json"
    _snapshot(primary, None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()
    a.clear_recovery()
    assert not primary.exists()
    assert not list(autosave_dir.glob("*.rm"))


# --- wave 6 -------------------------------------------------------------------


def _race_with_live_writer(autosave, autosave_dir, monkeypatch, link):
    """B atomically replaces the file right before A's rename; patch os.link."""
    primary = autosave_dir / "session.json"
    real_rename = os.rename

    def racing_rename(src, dst):
        if str(dst).endswith(".rm"):
            tmp = autosave_dir / "b.tmp"
            _snapshot(tmp, {"pid": os.getppid(), "token": "b"}, [7.0, 8.0])
            os.replace(tmp, primary)  # os.rename refuses an existing target on Windows
        return real_rename(src, dst)

    monkeypatch.setattr(autosave.os, "rename", racing_rename)
    monkeypatch.setattr(autosave.os, "link", link)
    return primary


@pytest.mark.serial
def test_restore_falls_back_to_rename_without_hard_links(
    app_state, autosave_dir, monkeypatch
):
    import errno

    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.autosave import AutosaveManager

    _snapshot(autosave_dir / "session.json", None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()

    def no_link(src, dst):
        raise OSError(errno.EOPNOTSUPP, "no hard links")

    primary = _race_with_live_writer(autosave, autosave_dir, monkeypatch, no_link)
    a.clear_recovery()
    assert json.loads(primary.read_text())["ydata"] == [7.0, 8.0]  # B's file restored
    assert not list(autosave_dir.glob("*.rm"))


@pytest.mark.serial
def test_failed_restore_leaves_tombstone_that_discovery_adopts(
    app_state, autosave_dir, monkeypatch
):
    import errno

    from nlsq.gui_qt import autosave
    from nlsq.gui_qt.autosave import AutosaveManager

    _snapshot(autosave_dir / "session.json", None, [3.0, 4.0])
    a = AutosaveManager(app_state)
    assert a.check_recovery()

    real_rename, real_link = os.rename, os.link

    def no_link(src, dst):
        raise OSError(errno.EOPNOTSUPP, "no hard links")

    primary = _race_with_live_writer(autosave, autosave_dir, monkeypatch, no_link)
    racing = autosave.os.rename

    def rename_then_fail(src, dst):  # the restoring rename itself fails
        if str(src).endswith(".rm"):
            raise OSError(errno.EIO, "io error")
        return racing(src, dst)

    monkeypatch.setattr(autosave.os, "rename", rename_then_fail)
    a.clear_recovery()
    assert not primary.exists()
    assert len(list(autosave_dir.glob("*.rm"))) == 1  # never deleted
    monkeypatch.setattr(autosave.os, "rename", real_rename)
    monkeypatch.setattr(autosave.os, "link", real_link)

    # B crashes; the next startup finds its snapshot through the tombstone
    _snapshot_owner = json.loads(next(autosave_dir.glob("*.rm")).read_text())
    _snapshot_owner["owner"]["pid"] = 2**30  # no such process
    next(autosave_dir.glob("*.rm")).write_text(json.dumps(_snapshot_owner))
    fresh = AutosaveManager(app_state)
    assert fresh.check_recovery()
    assert fresh.recover()
    assert list(app_state.state.ydata) == [7.0, 8.0]
    assert not list(autosave_dir.glob("*.rm"))
