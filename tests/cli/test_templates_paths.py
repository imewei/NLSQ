"""Tests for CLI template path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlsq.cli import templates


def test_get_template_path_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_template_path should return the correct file path when present."""
    target = tmp_path / "custom_model_template.py"
    target.write_text("# template")
    monkeypatch.setattr(templates, "TEMPLATES_DIR", tmp_path)

    assert templates.get_template_path("custom_model_template.py") == target


def test_get_template_path_error_lists_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing templates should raise with available template names listed."""
    (tmp_path / "workflow_config_template.yaml").write_text("# template")
    (tmp_path / "custom_model_template.py").write_text("# template")
    monkeypatch.setattr(templates, "TEMPLATES_DIR", tmp_path)

    with pytest.raises(FileNotFoundError) as excinfo:
        templates.get_template_path("missing.txt")

    msg = str(excinfo.value)
    assert "workflow_config_template.yaml" in msg
    assert "custom_model_template.py" in msg


def test_get_custom_and_workflow_templates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """get_custom_model_template/get_workflow_template should delegate to helper."""
    custom = tmp_path / "custom_model_template.py"
    workflow = tmp_path / "workflow_config_template.yaml"

    def _get_template_path(name: str) -> Path:
        return custom if name == "custom_model_template.py" else workflow

    monkeypatch.setattr(templates, "get_template_path", _get_template_path)

    assert templates.get_custom_model_template() == custom
    assert templates.get_workflow_template() == workflow
