import os
import stat
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.serial


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.parametrize("existing", [False, True])
def test_bundle_export_is_private(tmp_path, existing):
    from nlsq.gui_qt.pages.export import ExportPage

    target = tmp_path / "session.zip"
    if existing:
        target.write_text("old")
        target.chmod(0o644)
    page = MagicMock()
    page._app_state.state.xdata = None
    page._generate_json.return_value = "{}"
    page.generate_python_code.return_value = ""
    page._generate_config_yaml.return_value = ""
    ExportPage.export_session_bundle(page, str(target))
    assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0


def test_failed_json_export_preserves_existing_file(tmp_path, monkeypatch):
    from nlsq.cli import result_exporter
    from nlsq.gui_qt.pages.export import ExportPage

    target = tmp_path / "results.json"
    target.write_text("existing result")

    def partial_write(path, *args, **kwargs):
        with path.open("w") as stream:
            stream.write("{")
        raise OSError("disk full")

    def failed_replace(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", partial_write)
    monkeypatch.setattr(result_exporter.os, "replace", failed_replace)
    page = MagicMock()
    page._generate_json.return_value = '{"new": true}'
    with pytest.raises(OSError, match="disk full"):
        ExportPage.export_json(page, str(target))
    assert target.read_text() == "existing result"


def test_json_export_follows_existing_symlink(tmp_path):
    from nlsq.gui_qt.pages.export import ExportPage

    target = tmp_path / "actual.json"
    target.write_text("old")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    page = MagicMock()
    page._generate_json.return_value = '{"new": true}'
    ExportPage.export_json(page, str(link))
    assert link.is_symlink()
    assert target.read_text() == '{"new": true}'
