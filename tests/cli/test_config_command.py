"""Fast tests for CLI config command branches."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlsq.cli.commands import config as config_module


def test_run_config_copies_both_templates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """run_config should copy both templates by default."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        copied = config_module.run_config()

    names = {p.name for p in copied}
    assert "workflow_config.yaml" in names
    assert "custom_model.py" in names
    assert (tmp_path / "workflow_config.yaml").exists()
    assert (tmp_path / "custom_model.py").exists()

    output = capsys.readouterr().out
    assert "Created 2 template file" in output


def test_run_config_single_output(tmp_path: Path) -> None:
    """Custom output should be allowed when copying a single template."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        copied = config_module.run_config(workflow=True, output="my_workflow.yaml")

    assert copied == [tmp_path / "my_workflow.yaml"]
    assert (tmp_path / "my_workflow.yaml").exists()


def test_run_config_rejects_output_with_multiple_templates(tmp_path: Path) -> None:
    """Output should raise when copying multiple templates."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Cannot use --output"):
            config_module.run_config(output="both.txt")


def test_run_config_file_exists_error(tmp_path: Path) -> None:
    """Existing destination should raise when force=False."""
    target = tmp_path / "workflow_config.yaml"
    target.write_text("existing")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileExistsError, match="already exists"):
            config_module.run_config(workflow=True)
