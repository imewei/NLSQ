from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.serial


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
