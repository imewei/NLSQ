"""Batch: summary-path validation up front, clean Ctrl-C, duplicate-input warning."""

import json
import logging

import pytest

from nlsq.cli.commands import batch
from nlsq.cli.errors import CLIError
from nlsq.cli.main import main as cli_main


def _wf(tmp_path, name="w1.yaml"):
    p = tmp_path / name
    p.write_text("data: {input_file: d.txt}\nmodel: {type: linear}\n")
    return str(p)


pytestmark = pytest.mark.serial


@pytest.fixture
def count_runs(monkeypatch):
    calls = []

    def fake(path, verbose=False):
        calls.append(path)
        return {"status": "success", "workflow_path": path, "result": {}, "error": None}

    monkeypatch.setattr(batch, "_run_single_workflow", fake)
    return calls


# ---- M4 ---------------------------------------------------------------------


def test_empty_summary_path_rejected_before_any_fit(tmp_path, count_runs):
    with pytest.raises(CLIError, match="summary"):
        batch.run_batch([_wf(tmp_path)], summary_file="")
    assert count_runs == []


def test_directory_summary_path_rejected_before_any_fit(tmp_path, count_runs):
    with pytest.raises(CLIError, match="directory"):
        batch.run_batch([_wf(tmp_path)], summary_file=str(tmp_path))
    assert count_runs == []


def test_unwritable_summary_parent_rejected_before_any_fit(tmp_path, count_runs):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        import os

        if os.access(ro, os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        with pytest.raises(CLIError, match="writable"):
            batch.run_batch([_wf(tmp_path)], summary_file=str(ro / "s.json"))
        assert count_runs == []
    finally:
        ro.chmod(0o755)


def test_summary_parent_that_is_a_file_rejected(tmp_path, count_runs):
    (tmp_path / "f").write_text("x")
    with pytest.raises(CLIError):
        batch.run_batch([_wf(tmp_path)], summary_file=str(tmp_path / "f" / "s.json"))
    assert count_runs == []


def test_cli_bad_summary_exits_nonzero_without_traceback(tmp_path, count_runs, capsys):
    rc = cli_main(["batch", _wf(tmp_path), "-s", ""])
    assert rc == 1
    assert "Traceback" not in capsys.readouterr().err
    assert count_runs == []


def test_final_write_failure_prints_summary_and_raises_clierror(
    tmp_path, count_runs, monkeypatch, capsys
):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(batch, "_write_batch_summary", boom)
    with pytest.raises(CLIError, match="disk full"):
        batch.run_batch([_wf(tmp_path)], summary_file=str(tmp_path / "s.json"))
    err = capsys.readouterr().err
    assert '"succeeded": 1' in err  # in-memory summary not lost


# ---- M5 ---------------------------------------------------------------------


def test_keyboard_interrupt_cancels_remaining_and_writes_summary(
    tmp_path, monkeypatch, capsys
):
    wfs = [_wf(tmp_path, f"w{i}.yaml") for i in (1, 2, 3)]
    ran = []

    def fake(path, verbose=False):
        ran.append(path)
        if path == wfs[1]:
            raise KeyboardInterrupt
        return {"status": "success", "workflow_path": path, "result": {}, "error": None}

    monkeypatch.setattr(batch, "_run_single_workflow", fake)
    summary = tmp_path / "s.json"

    rc = cli_main(["batch", *wfs, "-w", "1", "-s", str(summary)])

    assert rc == 130
    assert ran == wfs[:2]  # workflow 3 never started
    data = json.loads(summary.read_text())
    assert data["succeeded"] == 1
    assert data["cancelled"] == 2
    assert data["cancelled_workflows"] == wfs[1:]
    err = capsys.readouterr().err
    assert "Interrupted: 1 of 3 workflows completed" in err
    assert "Traceback" not in err


def test_run_batch_returns_cancelled_statuses_on_interrupt(tmp_path, monkeypatch):
    wfs = [_wf(tmp_path, f"w{i}.yaml") for i in (1, 2)]

    def fake(path, verbose=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(batch, "_run_single_workflow", fake)
    results = batch.run_batch(wfs, max_workers=1)
    assert sorted(r["status"] for r in results) == ["cancelled", "cancelled"]


# ---- LOW --------------------------------------------------------------------


def test_duplicate_workflow_input_warns(tmp_path, count_runs, capsys):
    wf = _wf(tmp_path)
    batch.run_batch([wf, wf], max_workers=1)
    assert "are the same workflow" in capsys.readouterr().err


# ---- collision guard reads raw YAML: reuse the shared visualization helpers ---


def _vis_wf(tmp_path, name, vis, input_file="d.txt"):
    p = tmp_path / name
    p.write_text(
        f"data: {{input_file: {input_file}}}\nmodel: {{type: linear}}\n"
        f"visualization: {vis}\n"
    )
    return str(p)


def test_collision_guard_treats_bare_string_formats_as_one_format(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "p_combined.pdf").write_text("input")
    wf = _vis_wf(
        tmp_path,
        "a.yaml",
        "{enabled: true, filename_prefix: p, formats: pdf}",
        input_file="figures/p_combined.pdf",
    )
    with pytest.raises(CLIError, match="overwrite"):
        batch._warn_output_collisions([wf], logging.getLogger("t"))


def test_collision_guard_skips_traversal_prefix(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "figures").mkdir()
    (tmp_path / "in_combined.pdf").write_text("input")
    wf = _vis_wf(
        tmp_path,
        "a.yaml",
        "{enabled: true, filename_prefix: ../in, formats: [pdf]}",
        input_file="in_combined.pdf",
    )
    batch._warn_output_collisions([wf], logging.getLogger("t"))  # must not raise
