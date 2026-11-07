"""Phase 0 profiling of the `curve_fit` / `curve_fit_large` paths.

Usage::

    python -m benchmarks.studies.profile_fit_paths --case fit --n 1000 --out fit.json

Cases: fit, large, bounded (cProfile of the fit, top functions by tottime), svd (CPU vs
default device factorizations of an n x p Jacobian), accept (one TRF step-acceptance
call split into model, Jacobian and the rest), cond (per-chunk cond(J) of the default
`curve_fit_large` path). Output JSON: ``{case, n, backend, wall_ms, top, ...}``.
Model and data follow the spec: ``a*exp(-b*x)+c``, ``linspace(0, 4, n)``, noise 0.05,
seed 0, ``p0 = [1, 1, 0]``.
"""

import argparse
import cProfile
import json
import pstats
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import nlsq

CASES = ("fit", "large", "bounded", "svd", "accept", "cond")
DEFAULT_N = {"fit": 1000, "bounded": 1000, "accept": 1000, "large": 100_000}
DEFAULT_N |= {"svd": 100_000, "cond": 100_000}
BOUNDS = ([0, 0, -1], [10, 10, 1])
P0 = [1.0, 1.0, 0.0]
WARMUPS, REPEATS, TOP = 3, 20, 25


def _model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def _data(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, 4, n)
    rng = np.random.default_rng(0)
    return x, 2.5 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.standard_normal(n)


def _ms(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return (time.perf_counter() - t0) * 1e3


def _top(prof: cProfile.Profile) -> list[dict]:
    st = pstats.Stats(prof)
    rows = sorted(st.stats.items(), key=lambda kv: kv[1][2], reverse=True)[:TOP]
    return [
        {
            "func": f"{Path(f).name}:{line}({name})",
            "tottime_s": tt,
            "cumtime_s": ct,
            "ncalls": nc,
        }
        for (f, line, name), (_, nc, tt, ct, _) in rows
    ]


def _profile_fit(case: str, n: int) -> dict:
    x, y = _data(n)
    if case == "large":
        run, repeats = (lambda: nlsq.curve_fit_large(_model, x, y, p0=P0)), 1
    else:
        kw = {"bounds": BOUNDS} if case == "bounded" else {}
        run, repeats = (lambda: nlsq.curve_fit(_model, x, y, p0=P0, **kw)), REPEATS
    for _ in range(WARMUPS):
        run()
    wall = statistics.median(_ms(run) for _ in range(repeats))
    prof = cProfile.Profile()
    prof.runcall(run)
    return {"wall_ms": wall, "top": _top(prof)}


def _profile_svd(n: int) -> dict:
    cpu = jax.devices("cpu")[0]

    def svd(j):
        return jnp.linalg.svd(j, full_matrices=False)

    def qr_svd(j):
        return jnp.linalg.svd(jnp.linalg.qr(j)[1], full_matrices=False)

    def gram_eigh(j):
        return jnp.linalg.eigh(j.T @ j)

    rows, t_all = [], time.perf_counter()
    for p in (3, 6):
        j_default = jnp.asarray(np.random.default_rng(p).standard_normal((n, p)))
        for device, put in (
            (jax.default_backend(), lambda a: a),
            ("cpu", lambda a: jax.device_put(a, cpu)),  # transfer is part of the cost
        ):
            for name, fn in (
                ("svd", svd),
                ("qr_svd", qr_svd),
                ("gram_eigh", gram_eigh),
            ):
                run = lambda fn=fn, put=put, j=j_default: fn(put(j))  # noqa: E731
                for _ in range(WARMUPS):
                    jax.block_until_ready(run())
                ms = statistics.median(_ms(run) for _ in range(7))
                rows.append({"p": p, "device": device, "method": name, "ms": ms})
    return {"wall_ms": (time.perf_counter() - t_all) * 1e3, "top": [], "rows": rows}


def _profile_accept(n: int) -> dict:
    from nlsq.core.trf import TrustRegionReflective

    orig, captured = TrustRegionReflective._evaluate_step_acceptance, {}

    def grab(self, **kw):
        captured.setdefault("call", (self, kw))
        return orig(self, **kw)

    x, y = _data(n)
    TrustRegionReflective._evaluate_step_acceptance = grab
    try:
        nlsq.curve_fit(_model, x, y, p0=P0)
    finally:
        TrustRegionReflective._evaluate_step_acceptance = orig
    self_, kw = captured["call"]

    spent = {"fun": 0.0, "jac": 0.0}

    def timed(key, fn):
        def wrapper(*a):
            t0 = time.perf_counter()
            out = jax.block_until_ready(fn(*a))
            spent[key] += time.perf_counter() - t0
            return out

        return wrapper

    kw = {**kw, "fun": timed("fun", kw["fun"]), "jac": timed("jac", kw["jac"])}
    for _ in range(WARMUPS):
        orig(self_, **kw)
    fun_ms, jac_ms, total_ms = [], [], []
    for _ in range(REPEATS):
        spent.update(fun=0.0, jac=0.0)
        t0 = time.perf_counter()
        orig(self_, **kw)
        total_ms.append((time.perf_counter() - t0) * 1e3)
        fun_ms.append(spent["fun"] * 1e3)
        jac_ms.append(spent["jac"] * 1e3)
    total, fun, jac = map(statistics.median, (total_ms, fun_ms, jac_ms))
    return {
        "wall_ms": total,
        "top": [],
        "fun_ms": fun,
        "jac_ms": jac,
        "scalar_and_updates_ms": total - fun - jac,  # residual: everything else
    }


def _profile_cond(n: int, problem_name: str) -> dict:
    from benchmarks.compare.problems import PROBLEMS
    from nlsq.core import minpack

    problem = PROBLEMS[problem_name]
    x, y = problem.data(n)
    model = lambda x, *p: problem.model(jnp, x, *p)  # noqa: E731
    conds: list[float] = []
    orig = minpack._covariance_svd

    def recorder(jac):
        conds.append(float(np.linalg.cond(np.asarray(jac))))
        return orig(jac)

    # CurveFit.__init__ binds the module-level name per instance, so patch the module
    # before curve_fit_large builds its fitter.
    minpack._covariance_svd = recorder
    try:
        t0 = time.perf_counter()
        nlsq.curve_fit_large(model, x, y, p0=list(problem.p0))
        wall = (time.perf_counter() - t0) * 1e3
    finally:
        minpack._covariance_svd = orig
    return {
        "wall_ms": wall,
        "top": [],
        "problem": problem_name,
        "cond_per_chunk": conds,
        "cond_min": min(conds, default=None),
        "cond_median": statistics.median(conds) if conds else None,
        "cond_max": max(conds, default=None),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", choices=CASES, required=True)
    ap.add_argument("--n", type=int)
    ap.add_argument("--problem", default="exp_decay", help="cond case problem name")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    n = args.n or DEFAULT_N[args.case]
    if args.case in ("fit", "large", "bounded"):
        res = _profile_fit(args.case, n)
    elif args.case == "svd":
        res = _profile_svd(n)
    elif args.case == "accept":
        res = _profile_accept(n)
    else:
        res = _profile_cond(n, args.problem)
    res = {"case": args.case, "n": n, "backend": jax.default_backend(), **res}
    text = json.dumps(res, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
