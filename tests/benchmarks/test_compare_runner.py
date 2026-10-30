"""Tests for the comparison-harness runner and report."""

import json
from typing import get_args

import pytest

from benchmarks.compare.report import RECORD_FIELDS, rank_case, to_markdown
from benchmarks.compare.run_library_comparison import main
from benchmarks.compare.types import Status


def _rec(library, status="ok", warm=1.0, reason=None):
    rec = dict.fromkeys(RECORD_FIELDS)
    rec.update(
        library=library,
        mode="end_to_end",
        problem="exp_decay",
        n=200,
        backend="cpu",
        status=status,
        reason=reason,
        warm_median_s=warm,
    )
    return rec


def test_rank_case_excludes_early_stoppers():
    recs = [
        _rec("slow", warm=3.0),
        _rec("cheater", status="did_not_reach_optimum", warm=0.1),
        _rec("fast", warm=2.0),
    ]
    assert [r["library"] for r in rank_case(recs)] == ["fast", "slow"]


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
    for rec in records:
        assert tuple(rec) == RECORD_FIELDS
        assert rec["status"] in get_args(Status)
    md = next(tmp_path.glob("*.md")).read_text()
    for lib in ("nlsq", "scipy", "lmfit", "jaxopt", "optimistix"):
        assert lib in md
