import logging
import os
import stat

import pytest
import yaml

from nlsq.cli.commands.batch import _warn_output_collisions
from nlsq.cli.errors import CLIError
from nlsq.cli.result_exporter import _atomic_write

pytestmark = pytest.mark.serial


def test_batch_protects_relative_custom_model(tmp_path):
    model = tmp_path / "model.py"
    model.write_text("keep")
    first = tmp_path / "first.yaml"
    first.write_text(yaml.safe_dump({"model": {"type": "custom", "path": model.name}}))
    second = tmp_path / "second.yaml"
    second.write_text(yaml.safe_dump({"export": {"results_file": str(model)}}))
    with pytest.raises(CLIError, match="input"):
        _warn_output_collisions([str(first), str(second)], logging.getLogger(__name__))
    assert model.read_text() == "keep"


def test_preflight_leaves_invalid_utf8_for_workflow_to_report(tmp_path):
    workflow = tmp_path / "bad.yaml"
    workflow.write_bytes(b"\xff")
    _warn_output_collisions([str(workflow)], logging.getLogger(__name__))


def test_atomic_write_never_exposes_existing_private_content(tmp_path, monkeypatch):
    target = tmp_path / "private.json"
    target.write_text("old")
    target.chmod(0o600)
    original = os.fdopen

    def checked(fd, *args, **kwargs):
        assert stat.S_IMODE(os.fstat(fd).st_mode) & 0o077 == 0
        return original(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", checked)
    _atomic_write(target, "secret")
    assert target.read_text() == "secret"
