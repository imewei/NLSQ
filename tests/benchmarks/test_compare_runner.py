"""Tests for the comparison-harness runner and report."""

import json
from typing import get_args

import pytest

from benchmarks.compare import run_library_comparison as runner
from benchmarks.compare.report import RECORD_FIELDS, rank_case, to_markdown, write_json
from benchmarks.compare.run_library_comparison import main
from benchmarks.compare.types import Status


def _rec(library, status="ok", warm=1.0, reason=None, mode="end_to_end", **kw):
    rec = dict.fromkeys(RECORD_FIELDS)
    rec.update(
        library=library,
        mode=mode,
        problem="exp_decay",
        n=200,
        backend="cpu",
        status=status,
        reason=reason,
        warm_median_s=warm,
        **kw,
    )
    return rec


def test_rank_case_excludes_early_stoppers():
    recs = [
        _rec("slow", warm=3.0),
        _rec("cheater", status="did_not_reach_optimum", warm=0.1),
        _rec("fast", warm=2.0),
    ]
    assert [r["library"] for r in rank_case(recs)] == ["fast", "slow"]


def test_jitted_core_cannot_win_the_headline():
    recs = [
        _rec("resident", warm=0.1, mode="jitted_core"),
        _rec("e2e", warm=2.0),
    ]
    assert [r["library"] for r in rank_case(recs)] == ["e2e"]
    md = to_markdown(recs, {"backend": "cpu"})
    head, _, resident = md.partition("resident data (jitted_core)")
    assert "| e2e |" in head
    assert "| resident |" not in head
    assert "| resident |" in resident


def test_markdown_excluded_line_names_the_batch():
    rec = _rec("jaxopt", status="unavailable", reason="boom", batch=256)
    assert "batch=256" in to_markdown([rec], {"backend": "cpu"})


def _case(*costs):
    return [_rec(f"lib{i}", cost=c) for i, c in enumerate(costs)]


def test_finalize_ignores_nan_when_finding_the_best_cost():
    case = _case(float("nan"), 1.0)
    runner._finalize(case)
    assert case[0]["status"] == "did_not_reach_optimum"
    assert case[0]["reason"] == "non-finite cost"
    assert case[0]["cost_gap"] is None
    assert case[1]["status"] == "ok"
    assert case[1]["cost_gap"] == 0.0


def test_finalize_ranks_nothing_when_every_cost_is_non_finite():
    case = _case(float("inf"))
    runner._finalize(case)
    assert case[0]["status"] == "did_not_reach_optimum"
    assert rank_case(case) == []


def test_finalize_exact_fit_has_zero_best_cost():
    case = _case(0.0, 1e-3)
    runner._finalize(case)
    assert [r["status"] for r in case] == ["ok", "did_not_reach_optimum"]
    assert case[0]["cost_gap"] == 0.0


def test_write_json_is_strict(tmp_path):
    path = tmp_path / "r.json"
    write_json([_rec("x", cost=float("nan"), cost_gap=float("inf"))], path)
    rec = json.loads(path.read_text(), parse_constant=pytest.fail)[0]
    assert rec["cost"] is None
    assert rec["cost_gap"] is None


def test_spawn_survives_a_truncated_record(monkeypatch):
    class Done:
        stdout = runner.PREFIX + '{"library": "nl'
        stderr = ""
        returncode = 0

    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: Done())
    rec = runner._spawn("nlsq", "end_to_end", "exp_decay", 10, 0, 1)
    assert rec["status"] == "unavailable"
    assert "unparsable" in rec["reason"]


@pytest.mark.parametrize("flag", ["--libraries", "--problems"])
def test_unknown_names_are_rejected(flag, tmp_path, capsys):
    assert main([flag, "no_such_name", "--out", str(tmp_path)]) == 2
    assert "no_such_name" in capsys.readouterr().err


def test_markdown_lists_unavailable_library_with_reason():
    md = to_markdown(
        [_rec("jaxopt", status="unavailable", reason="ImportError: boom")],
        {"backend": "cpu"},
    )
    assert "boom" in md
    assert "jaxopt" in md


def test_markdown_carries_required_caveats():
    md = to_markdown([_rec("nlsq")], {"backend": "tpu-sentinel"})
    assert "no longer maintained" in md
    assert "wraps SciPy" in md
    assert "tpu-sentinel" in md


@pytest.mark.serial
def test_quick_run_writes_valid_json_and_markdown(tmp_path, monkeypatch):
    # main() sets these; monkeypatch restores them afterwards
    monkeypatch.setenv("NLSQ_DISABLE_PERSISTENT_CACHE", "1")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    assert main(["--quick", "--no-batch", "--out", str(tmp_path)]) == 0
    records = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert records
    assert any(r["status"] == "ok" for r in records)
    for rec in records:
        assert tuple(rec) == RECORD_FIELDS
        assert rec["status"] in get_args(Status)
    md = next(tmp_path.glob("*.md")).read_text()
    for lib in ("nlsq", "scipy", "lmfit", "jaxopt", "optimistix"):
        assert lib in md
