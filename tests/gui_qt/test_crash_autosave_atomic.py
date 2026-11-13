import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.serial


def test_failed_crash_marker_write_preserves_recovery(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    import nlsq.gui_qt as gui
    from nlsq.gui_qt import autosave

    recovery = tmp_path / "session.json"
    original = '{"timestamp": "saved", "ydata": [1, 2]}'
    recovery.write_text(original)
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", recovery)
    monkeypatch.setattr(
        QApplication, "instance", lambda: MagicMock(activeWindow=lambda: None)
    )
    monkeypatch.setattr(gui.sys, "platform", "darwin")

    def partial_write(path, *args, **kwargs):
        with path.open("w") as stream:
            stream.write("{")
        raise OSError("disk full")

    def failed_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_write)
    monkeypatch.setattr(autosave.os, "replace", failed_replace)
    gui._exception_hook(RuntimeError, RuntimeError("failure"), None)
    assert json.loads(recovery.read_text()) == json.loads(original)


def test_crash_marker_keeps_saved_state(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    import nlsq.gui_qt as gui
    from nlsq.gui_qt import autosave

    recovery = tmp_path / "session.json"
    recovery.write_text('{"timestamp": "saved", "ydata": [1, 2]}')
    monkeypatch.setattr(autosave, "AUTOSAVE_FILE", recovery)
    monkeypatch.setattr(
        QApplication, "instance", lambda: MagicMock(activeWindow=lambda: None)
    )
    monkeypatch.setattr(gui.sys, "platform", "darwin")
    gui._exception_hook(RuntimeError, RuntimeError("failure"), None)
    saved = json.loads(recovery.read_text())
    assert saved["ydata"] == [1, 2]
    assert saved["crash_info"]["error"] == "failure"
