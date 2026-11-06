"""Tests for the Phase 0 profiling script and the interleaved A/B runner."""

import json

import pytest

from benchmarks.compare.report import RECORD_FIELDS
from benchmarks.studies import ab_nlsq
from benchmarks.studies.profile_fit_paths import main as profile_main


@pytest.mark.serial
def test_profile_fit_paths_smoke(tmp_path):
    out = tmp_path / "fit.json"
    assert profile_main(["--case", "fit", "--n", "200", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert {"case", "n", "backend", "wall_ms", "top"} <= data.keys()
    assert data["case"] == "fit" and data["n"] == 200
    assert data["wall_ms"] > 0
    assert data["top"]
    assert {"func", "tottime_s", "cumtime_s", "ncalls"} <= data["top"][0].keys()


def test_ab_ratio_math():
    assert ab_nlsq.ratio([2, 2, 2], [1, 1, 1]) == 0.5
    assert ab_nlsq.ratio([1, 3, 2], [4, 6, 5]) == 2.5


def test_ab_spread():
    assert ab_nlsq.spread([1, 1, 1]) == 0
    assert ab_nlsq.spread([1, 2, 3]) == 1.0


def _rec(warm=0.1, cost=1.0, status="ok", library="nlsq", mode="end_to_end", n=1000):
    rec = dict.fromkeys(RECORD_FIELDS)
    rec.update(
        library=library,
        mode=mode,
        problem="exp_decay",
        n=n,
        batch=None,
        backend="cpu",
        status=status,
        warm_median_s=warm,
        cost=cost,
    )
    return rec


def _write(path, records):
    path.write_text(json.dumps(records))
    return path


def test_ab_parses_harness_json(tmp_path):
    paths = [
        _write(
            tmp_path / f"r{i}.json",
            [
                _rec(warm=w),
                _rec(warm=10 * w, n=5000),
                _rec(warm=99.0, mode="warm_core"),  # non-headline mode: ignored
            ],
        )
        for i, w in enumerate([0.1, 0.3, 0.2])
    ]
    med = ab_nlsq.median_per_case(paths)
    assert med == {
        ("nlsq", "exp_decay", 1000): pytest.approx(0.2),
        ("nlsq", "exp_decay", 5000): pytest.approx(2.0),
    }


def _run_main(monkeypatch, tmp_path, base_recs, cand_recs, rounds=2):
    """Run `ab_nlsq.main` with the harness subprocess replaced by fixtures."""
    by_checkout = {"base": base_recs, "cand": cand_recs}

    def fake_harness(checkout, out_dir, args):
        return _write(out_dir / "library-comparison-x.json", by_checkout[checkout.name])

    monkeypatch.setattr(ab_nlsq, "_run_harness", fake_harness)
    monkeypatch.setattr(ab_nlsq, "_check_import", lambda checkout: None)
    argv = ["--base", str(tmp_path / "base"), "--cand", str(tmp_path / "cand")]
    argv += ["--libraries", "nlsq", "--sizes", "1000", "--problems", "exp_decay"]
    argv += ["--rounds", str(rounds), "--out-root", str(tmp_path / "runs")]
    return ab_nlsq.main(argv)


def test_ab_correctness_gate(monkeypatch, tmp_path, capsys):
    base = [_rec(warm=0.1, cost=1.0)]
    assert _run_main(monkeypatch, tmp_path, base, [_rec(warm=0.05, cost=1.0)]) == 0
    assert "0.500" in capsys.readouterr().out

    failed = [_rec(status="failed", cost=None)]
    bad = {  # name: (base records, cand records)
        "status mismatch": (base, failed),
        "missing record": (base, []),
        "cost gap 1e-5": (base, [_rec(cost=1.00001)]),
        "not ok in both": (failed, failed),
    }
    for name, (b, c) in bad.items():
        assert _run_main(monkeypatch, tmp_path, b, c) != 0, name

    # within tolerance still passes
    assert _run_main(monkeypatch, tmp_path, base, [_rec(cost=1.0 + 1e-8)]) == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("arm", ["base", "cand"])
def test_ab_gate_rejects_non_finite_cost(monkeypatch, tmp_path, arm, bad):
    good = [_rec(cost=1.0)]
    broken = [_rec(cost=bad)]
    b, c = (broken, good) if arm == "base" else (good, broken)
    assert _run_main(monkeypatch, tmp_path, b, c) != 0


def test_ab_gate_rejects_non_finite_cost_in_both_arms(monkeypatch, tmp_path):
    nan = [_rec(cost=float("nan"))]
    assert _run_main(monkeypatch, tmp_path, nan, nan) != 0


def test_require_gpu_backend_decision():
    assert ab_nlsq.backend_matches("gpu", "gpu")
    assert ab_nlsq.backend_matches("gpu", "cuda")
    assert not ab_nlsq.backend_matches("gpu", "cpu")
    assert ab_nlsq.backend_matches("cpu", "cpu")
    assert ab_nlsq.backend_matches("cpu", "gpu")  # cpu is forced via env, not checked
