"""CG2: unsupported export.format fails before the fit runs."""

import pytest

from nlsq.cli.errors import CLIError


def test_unsupported_export_format_rejected_before_fit(tmp_path, monkeypatch):
    """CG2: a bad export.format must not cost a full fit."""
    from nlsq.cli.workflow_runner import WorkflowRunner

    runner = WorkflowRunner()
    monkeypatch.setattr(
        runner, "_load_data", lambda c: pytest.fail("fit work ran before validation")
    )
    config = {
        "data": {"input_file": "d.txt", "format": "ascii"},
        "model": {"type": "builtin", "name": "linear"},
        "export": {"results_file": str(tmp_path / "o.out"), "format": "xml"},
    }
    with pytest.raises(CLIError, match="Unsupported export format"):
        runner.run(config)
    assert not (tmp_path / "o.out").exists()
