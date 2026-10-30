"""Library comparison runner.

Each measured case runs in a fresh worker process so the cold time is the first
call of that process (spec section B). The parent only schedules workers and
turns their raw records into ranked, gated results.

    uv run python -m benchmarks.compare.run_library_comparison --out benchmarks/results
"""

import argparse
import json
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from .report import RECORD_FIELDS

BATCH_PROBLEM, BATCH_N, BATCH_SIZE = "exp_decay", 200, 256
WORKER_TIMEOUT_S = 900
PREFIX = "RECORD_JSON:"  # library log noise on stdout must not corrupt parsing
REPO_ROOT = Path(__file__).resolve().parents[2]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        NLSQ_DISABLE_PERSISTENT_CACHE="1",
        JAX_ENABLE_COMPILATION_CACHE="false",
        NLSQ_SKIP_GPU_CHECK="1",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",  # parent and workers share a GPU
        PYTHONPATH=os.pathsep.join(
            [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
        ),
    )
    return env


def _blank(library, mode, problem, n, batch, status, reason) -> dict:
    rec = dict.fromkeys(RECORD_FIELDS)
    rec.update(
        library=library,
        mode=mode,
        problem=problem,
        n=n,
        batch=batch,
        status=status,
        reason=reason,
    )
    return rec


def _param_error(problem, outcomes) -> float:
    import numpy as np

    truth = np.asarray(problem.p_true)
    scale = np.where(truth == 0, 1.0, np.abs(truth))  # abs error where truth is 0
    outs = outcomes if isinstance(outcomes, list) else [outcomes]
    return float(
        max(np.max(np.abs(np.asarray(o.params) - truth) / scale) for o in outs)
    )


def _worker(library, mode, problem_name, n, batch, repeats) -> dict:
    """Measure one case in this process and return its raw record."""
    batch_arg = batch or None
    rec = _blank(library, mode, problem_name, n, batch_arg, "ok", None)
    try:
        import jax
        import jax.numpy as jnp

        jax.block_until_ready(jnp.ones(3) + 1)  # device init is not a library's cost
        rec["backend"] = jax.default_backend()
        rec["device"] = ", ".join(sorted({d.device_kind for d in jax.devices()}))

        from .adapters import ADAPTERS
        from .problems import PROBLEMS
        from .timing import all_converged, case_cost, time_fit

        problem, adapter = PROBLEMS[problem_name], ADAPTERS[library]
        if batch:
            x, data = problem.batch_data(n, batch)
            run = adapter.prepare_batch(problem, x, data, mode)
        else:
            x, data = problem.data(n)
            run = adapter.prepare(problem, x, data, mode)
        timing, out = time_fit(run, repeats)
        nfevs = [o.nfev for o in (out if batch else [out]) if o.nfev is not None]
        rec.update(
            converged=all_converged(out),
            cold_s=timing.cold_s,
            warm_median_s=timing.warm_median_s,
            warm_iqr_s=timing.warm_iqr_s,
            nfev=round(sum(nfevs) / len(nfevs)) if nfevs else None,
            cost=case_cost(problem, x, data, out),
            max_param_error_vs_truth=_param_error(problem, out),
        )
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # SciPy/NLSQ curve_fit raise RuntimeError when they fail to converge
        if isinstance(e, RuntimeError) and any(
            s in msg.lower() for s in ("converge", "not found")
        ):
            msg = f"did not converge: {e}"
        rec.update(status="unavailable", reason=msg)
    return rec


def _spawn(library, mode, problem, n, batch, repeats) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.compare.run_library_comparison",
        "--worker",
        library,
        mode,
        problem,
        str(n),
        str(batch),
        str(repeats),
    ]
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_S,
            env=_child_env(),
            cwd=REPO_ROOT,
        )
    except subprocess.TimeoutExpired:
        reason = f"timed out after {WORKER_TIMEOUT_S} s"
        return _blank(library, mode, problem, n, batch or None, "unavailable", reason)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(PREFIX):
            try:
                return json.loads(line[len(PREFIX) :])
            except json.JSONDecodeError as e:
                reason = f"unparsable worker record: {e}"
                return _blank(
                    library, mode, problem, n, batch or None, "unavailable", reason
                )
    tail = (p.stderr.strip().splitlines() or ["no output"])[-1]
    reason = f"worker exited {p.returncode}: {tail}"
    return _blank(library, mode, problem, n, batch or None, "unavailable", reason)


def _finalize(case: list[dict]) -> None:
    """Gate measured records against the best finite cost of the case, in place."""
    from .timing import reaches_optimum

    measured = [r for r in case if r["status"] == "ok" and r["cost"] is not None]
    finite = [r["cost"] for r in measured if math.isfinite(r["cost"])]
    best = min(finite) if finite else None
    for r in measured:
        if best is None or not math.isfinite(r["cost"]):
            r.update(
                status="did_not_reach_optimum", reason="non-finite cost", cost_gap=None
            )
            continue
        r["cost_gap"] = (r["cost"] - best) / max(abs(best), 1e-300)
        if not reaches_optimum(r["cost"], best):
            r["status"] = "did_not_reach_optimum"


def run(
    problems: Sequence[str],
    sizes: Sequence[int],
    repeats: int,
    include_batch: bool,
    libraries: Sequence[str] | None = None,
) -> list[dict]:
    from .adapters import ADAPTERS
    from .problems import PROBLEMS

    cases = [(p, n, 0) for p in problems for n in sizes]
    if include_batch:
        cases.append((BATCH_PROBLEM, BATCH_N, BATCH_SIZE))
    names = list(libraries) if libraries else list(ADAPTERS)
    records: list[dict] = []
    for problem, n, batch in cases:
        case: list[dict] = []
        for name in names:
            adapter = ADAPTERS[name]
            av = adapter.availability()
            ok, why = adapter.supports(PROBLEMS[problem])
            skip = None if av.ok else ("unavailable", av.reason)
            skip = skip or (None if ok else ("unsupported", why))
            if skip:
                case.append(_blank(name, "-", problem, n, batch or None, *skip))
                continue
            case += [
                _spawn(name, mode, problem, n, batch, repeats) for mode in adapter.modes
            ]
        _finalize(case)
        records += case
    return records


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worker", nargs=6, help=argparse.SUPPRESS)
    ap.add_argument("--quick", action="store_true", help="n=200, one repeat")
    ap.add_argument("--problems", nargs="+")
    ap.add_argument("--sizes", nargs="+", type=int, default=[1000, 100_000])
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--no-batch", action="store_true")
    ap.add_argument("--libraries", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = ap.parse_args(argv)

    # before adapters import JAX/nlsq in this process; workers get the full set
    os.environ["NLSQ_DISABLE_PERSISTENT_CACHE"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.worker:
        lib, mode, problem, n, batch, repeats = args.worker
        rec = _worker(lib, mode, problem, int(n), int(batch), int(repeats))
        print(PREFIX + json.dumps(rec))
        return 0

    from .adapters import ADAPTERS
    from .problems import PROBLEMS
    from .report import environment, to_markdown, write_json

    for flag, given, known in (
        ("--problems", args.problems, PROBLEMS),
        ("--libraries", args.libraries, ADAPTERS),
    ):
        bad = [g for g in given or () if g not in known]
        if bad:
            print(
                f"error: unknown {flag} {bad}; choose from {sorted(known)}",
                file=sys.stderr,
            )
            return 2

    sizes, repeats = ([200], 1) if args.quick else (args.sizes, args.repeats)
    records = run(
        args.problems or list(PROBLEMS),
        sizes,
        repeats,
        not args.no_batch,
        args.libraries or list(ADAPTERS),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"library-comparison-{date.today().isoformat()}"
    write_json(records, args.out / f"{stem}.json")
    (args.out / f"{stem}.md").write_text(to_markdown(records, environment()))
    print(f"wrote {args.out / stem}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
