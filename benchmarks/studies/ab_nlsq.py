"""Interleaved A/B runner for before/after `curve_fit` / `curve_fit_large` timings.

Wraps ``benchmarks.compare.run_library_comparison`` for two checkouts (base, cand),
alternating one harness run per arm per round, each in a fresh ``--out`` directory (the
harness names its file by date, so a shared directory would be overwritten).

Usage::

    python -m benchmarks.studies.ab_nlsq --base ../NLSQ-base --cand . \\
        --libraries nlsq --sizes 1000 --problems exp_decay --rounds 5 --repeats 7

Acceptance rule used by every perf task: a speedup counts only if
``ratio < 1 - max(spread_base, spread_cand)`` and ratio < the task's bound; a slowdown
check passes if ``ratio < 1 + max(spread)`` and ratio < its bound.

The harness's own cost gate is relative to the best library in the same run, so with one
library it is vacuous. The gate here is therefore explicit: per round, base and cand
must both be ``ok``, report the same case set (nothing expected is missing) and agree on
final ``cost`` within rtol 1e-6. Any violation returns exit code 1.

``--cost-gate no-worse`` relaxes the cost rule for default-path changes that make the fit
more accurate: the candidate may be cheaper, never costlier; a case where base is the
costlier arm is listed separately and not timed.
"""

import argparse
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

COST_RTOL = 1e-6
Case = tuple[str, str, int]  # (library, problem, n)


def ratio(base_ms: Sequence[float], cand_ms: Sequence[float]) -> float:
    """``median(cand) / median(base)``; below 1 means the candidate is faster."""
    return statistics.median(cand_ms) / statistics.median(base_ms)


def spread(ms: Sequence[float]) -> float:
    """``(max - min) / median`` of one arm's per-round values (in-session noise)."""
    return (max(ms) - min(ms)) / statistics.median(ms)


def load_end_to_end(path: Path) -> dict[Case, dict]:
    """Headline (``mode == "end_to_end"``) records of one harness JSON, by case."""
    return {
        (r["library"], r["problem"], r["n"]): r
        for r in json.loads(Path(path).read_text())
        if r["mode"] == "end_to_end"
    }


def median_per_case(paths: Sequence[Path]) -> dict[Case, float]:
    """Median over harness files of ``warm_median_s`` for every case."""
    per_case: dict[Case, list[float]] = {}
    for p in paths:
        for case, r in load_end_to_end(p).items():
            per_case.setdefault(case, []).append(r["warm_median_s"])
    return {case: statistics.median(v) for case, v in per_case.items()}


def check_correctness(
    expected: Sequence[Case],
    base: dict[Case, dict],
    cand: dict[Case, dict],
    cost_gate: str = "equal",
) -> tuple[list[str], set[Case]]:
    """Return ``(violations, skipped)`` between one base run and one candidate run.

    ``cost_gate="equal"``: costs must agree within ``COST_RTOL``; ``skipped`` is empty.
    ``cost_gate="no-worse"``: both arms must be ``ok`` with a finite cost; the candidate
    may not be costlier than base beyond ``COST_RTOL``. When base is the costlier arm
    (it stopped short of the optimum) the case is ``skipped``: timing it would compare
    a less accurate fit against a more accurate one.
    """
    bad: list[str] = []
    skipped: set[Case] = set()
    for case in expected:
        b, c = base.get(case), cand.get(case)
        if b is None or c is None:
            bad.append(f"{case}: missing record (base={b is None}, cand={c is None})")
        elif b["status"] != c["status"]:
            bad.append(f"{case}: status base={b['status']!r} cand={c['status']!r}")
        elif b["status"] != "ok":
            bad.append(f"{case}: status {b['status']!r} in both arms, nothing to time")
        elif b["cost"] is None or c["cost"] is None:
            bad.append(f"{case}: missing cost on an ok record")
        elif not (math.isfinite(b["cost"]) and math.isfinite(c["cost"])):
            bad.append(f"{case}: non-finite cost base={b['cost']!r} cand={c['cost']!r}")
        elif cost_gate == "equal":
            if abs(c["cost"] - b["cost"]) > COST_RTOL * max(abs(b["cost"]), 1e-300):
                bad.append(f"{case}: cost base={b['cost']!r} cand={c['cost']!r}")
        elif c["cost"] > b["cost"] * (1 + COST_RTOL):
            bad.append(f"{case}: cand cost worse base={b['cost']!r} cand={c['cost']!r}")
        elif b["cost"] > c["cost"] * (1 + COST_RTOL):
            skipped.add(case)
    return bad, skipped


def _env(checkout: Path, platform: str = "") -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(checkout), *filter(None, [env.get("PYTHONPATH")])]
    )
    if platform == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    return env


def _check_import(checkout: Path) -> None:
    """Abort unless ``import nlsq`` resolves inside ``checkout`` (editable-install guard)."""
    out = subprocess.run(
        [sys.executable, "-c", "import nlsq; print(nlsq.__file__)"],
        cwd=checkout,
        env=_env(checkout),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not Path(out).resolve().is_relative_to(checkout.resolve()):
        raise SystemExit(f"nlsq imported from {out}, outside checkout {checkout}")


def backend_matches(platform: str, backend: str) -> bool:
    """``--platform gpu`` needs a GPU backend; cpu is forced via ``JAX_PLATFORMS``."""
    return platform != "gpu" or backend in ("gpu", "cuda")


def _check_backend(checkout: Path, platform: str) -> None:
    """Abort unless JAX under the harness env really uses the requested platform."""
    if platform != "gpu":
        return
    out = subprocess.run(
        [sys.executable, "-c", "import jax; print(jax.default_backend())"],
        cwd=checkout,
        env=_env(checkout, platform),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not backend_matches(platform, out):
        raise SystemExit(f"--platform gpu requested but JAX backend is {out!r}")


def _run_harness(checkout: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    cmd = [sys.executable, "-m", "benchmarks.compare.run_library_comparison"]
    cmd += ["--no-batch", "--repeats", str(args.repeats)]
    cmd += ["--libraries", *args.libraries, "--sizes", *map(str, args.sizes)]
    cmd += ["--problems", *args.problems, "--out", str(out_dir)]
    subprocess.run(
        cmd, cwd=checkout, env=_env(checkout, args.platform), check=True, text=True
    )
    (path,) = out_dir.glob("library-comparison-*.json")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--libraries", nargs="+", default=["nlsq"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000])
    ap.add_argument("--problems", nargs="+", default=["exp_decay"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--platform", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--out-root", type=Path, help="keep raw harness JSON here")
    ap.add_argument(
        "--cost-gate",
        choices=["equal", "no-worse"],
        default="equal",
        help="equal: costs agree within rtol; no-worse: cand not costlier than base "
        "(cases where base is costlier are listed and not timed)",
    )
    args = ap.parse_args(argv)

    arms = {"base": args.base.resolve(), "cand": args.cand.resolve()}
    for checkout in arms.values():
        _check_import(checkout)
        _check_backend(checkout, args.platform)

    root = args.out_root or Path(tempfile.mkdtemp(prefix="ab_nlsq_"))
    paths: dict[str, list[Path]] = {"base": [], "cand": []}
    for rnd in range(args.rounds):
        for arm, checkout in arms.items():
            out_dir = root / f"round{rnd}-{arm}"
            out_dir.mkdir(parents=True, exist_ok=True)
            paths[arm].append(_run_harness(checkout, out_dir, args))

    expected = list(itertools.product(args.libraries, args.problems, args.sizes))
    violations = []
    skipped: set[Case] = set()
    for rnd, (pb, pc) in enumerate(zip(paths["base"], paths["cand"], strict=True)):
        bad, skip = check_correctness(
            expected, load_end_to_end(pb), load_end_to_end(pc), args.cost_gate
        )
        violations += [f"round {rnd}: {v}" for v in bad]
        skipped |= skip
    timed = [case for case in expected if case not in skipped]
    series = {
        arm: [
            {c: r["warm_median_s"] for c, r in load_end_to_end(p).items()} for p in ps
        ]
        for arm, ps in paths.items()
    }
    for case in timed:
        for arm, rounds in series.items():
            for rnd, values in enumerate(rounds):
                t = values.get(case)
                if not (isinstance(t, int | float) and math.isfinite(t)):
                    violations.append(
                        f"round {rnd}: {case}: missing/non-finite warm_median_s "
                        f"in {arm} ({t!r})"
                    )
    if violations:
        print("CORRECTNESS GATE FAILED", *violations, sep="\n  ", file=sys.stderr)
        return 1

    print(
        f"{'case':<34}{'base_s':>10}{'cand_s':>10}{'ratio':>8}{'spr_b':>8}{'spr_c':>8}"
    )
    for case in timed:
        b = [rnd[case] for rnd in series["base"]]
        c = [rnd[case] for rnd in series["cand"]]
        print(
            f"{'/'.join(map(str, case)):<34}{statistics.median(b):>10.4f}"
            f"{statistics.median(c):>10.4f}{ratio(b, c):>8.3f}"
            f"{spread(b):>8.3f}{spread(c):>8.3f}"
        )
    if skipped:
        print(
            "SKIPPED (base cost worse than cand beyond rtol; not timed, not in verdict)"
        )
        for case in expected:
            if case in skipped:
                print(f"  {'/'.join(map(str, case))}")
    print(f"raw harness JSON: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
