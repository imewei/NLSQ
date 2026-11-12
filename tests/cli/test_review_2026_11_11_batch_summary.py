"""Batch summary must be written atomically and tolerate non-JSON values."""

import json

import numpy as np
import pytest

from nlsq.cli.commands import batch


def _workflow(tmp_path):
    x = np.linspace(0, 1, 20)
    np.savetxt(tmp_path / "good.txt", np.column_stack([x, 1 + 2 * x + x**2]))
    wf = tmp_path / "w2.yaml"
    wf.write_text(
        "data: {input_file: good.txt, format: ascii}\n"
        "model: {type: polynomial, degree: 2024-01-01}\n"
        "export: {results_file: out2.json}\n"
    )
    return wf


def test_summary_survives_unserializable_error_context(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wf = _workflow(tmp_path)
    summary = tmp_path / "summary.json"
    summary.write_text('{"previous": "summary"}')

    batch.run_batch([str(wf)], summary_file=str(summary))

    data = json.loads(summary.read_text())  # complete, valid JSON
    assert "previous" not in data
    assert "2024-01-01" in json.dumps(data)


def test_existing_summary_not_truncated_when_serialization_fails(tmp_path, monkeypatch):
    summary = tmp_path / "summary.json"
    summary.write_text('{"previous": "summary"}')

    def boom(*a, **k):
        raise TypeError("forced")

    monkeypatch.setattr(batch.json, "dumps", boom)
    with pytest.raises(TypeError):
        batch._write_batch_summary({"a": 1}, str(summary), False, None)

    assert summary.read_text() == '{"previous": "summary"}'
    assert [p.name for p in tmp_path.iterdir()] == ["summary.json"]
