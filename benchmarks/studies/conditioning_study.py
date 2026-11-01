"""Stage 3 conditioning study: normal equations versus TSQR (pre-registered rule).

Usage::

    python -m benchmarks.studies.conditioning_study --out <dir> [--seeds 0 1 2 3 4]
        [--problems ...] [--n-acc 2000] [--n-scale 20000] [--only accuracy|scale]
        [--resume] [--chunk 512] [--tol 1e-12]

One JSON row per (set, problem, seed) is written to ``<dir>`` as soon as it is computed
(``--resume`` skips rows already on disk). Every row of one (problem, seed, set) is fitted
on the same float64 observations: the mpmath reference (accuracy set only, started from
the SciPy solution, inputs converted exactly to mpf), the normal-equations candidate
(``nlsq.streaming.exact_lm.exact_lm_fit``), the TSQR candidate (an LM loop in this file on
:func:`~benchmarks.studies.conditioning_helpers.tsqr_step`) and the SciPy baseline. Both
candidates read the data in ``--chunk`` sized chunks. At the end the per-problem worst
errors and ``verdict`` over the accuracy rows with a valid reference are printed.
"""

import argparse
import json
import sys
import time
import warnings
import zlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from benchmarks.studies.conditioning_helpers import (
    KAPPA_MAX,
    eigh_rank,
    error_norms,
    mp_reference,
    svd_covariance,
    tsqr_factor,
    tsqr_step,
    verdict,
)
from nlsq.streaming.exact_lm import (
    ATTEMPT_CAP,
    LAM0,
    LAMBDA_MAX,
    _damping,
    exact_lm_fit,
    make_chunk_stats,
    run_pass,
    step_termination,
)

TOL = 1e-12  # candidate ftol = xtol = gtol; --tol overrides (process-wide)
NOISE = 1e-3
P0_SPREAD = 0.05  # p0 = truth * (1 + 0.05 * N(0, 1))
EPS = float(np.finfo(np.float64).eps)


class Problem:
    def __init__(
        self,
        name: str,
        model: Callable[..., Any],
        model_mp: Callable[..., Any],
        jac_mp: Callable[..., Sequence[Any]],
        truth: Sequence[float],
        x_range: tuple[float, float],
    ) -> None:
        self.name, self.model, self.model_mp, self.jac_mp = (
            name,
            model,
            model_mp,
            jac_mp,
        )
        self.truth = np.asarray(truth, dtype=np.float64)
        self.x_range = x_range


def _mp() -> Any:
    try:
        import mpmath  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit("mpmath is required for the study") from None
    return mpmath


def _two_exp(delta: float) -> Problem:
    mp = _mp()

    def model(x: Any, a1: Any, k1: Any, a2: Any, k2: Any) -> Any:
        return a1 * jnp.exp(-k1 * x) + a2 * jnp.exp(-k2 * x)

    def model_mp(x: Any, a1: Any, k1: Any, a2: Any, k2: Any) -> Any:
        return a1 * mp.exp(-k1 * x) + a2 * mp.exp(-k2 * x)

    def jac_mp(x: Any, a1: Any, k1: Any, a2: Any, k2: Any) -> list[Any]:
        e1, e2 = mp.exp(-k1 * x), mp.exp(-k2 * x)
        return [e1, -a1 * x * e1, e2, -a2 * x * e2]

    return Problem(
        f"two_exp_{delta:.0e}".replace("e-0", "e-"),
        model,
        model_mp,
        jac_mp,
        [1.0, 1.0, 1.0, 1.0 + delta],
        (0.0, 5.0),
    )


def _poly(deg: int) -> Problem:
    def model(x: Any, *c: Any) -> Any:
        return sum(ck * x**k for k, ck in enumerate(c))

    def jac_mp(x: Any, *c: Any) -> list[Any]:
        return [x**k for k in range(len(c))]

    truth = [(-1.0) ** k / (k + 1) for k in range(deg + 1)]
    return Problem(f"poly{deg}", model, model, jac_mp, truth, (0.0, 1.0))


def _gauss3() -> Problem:
    mp = _mp()

    def model(x: Any, *q: Any) -> Any:
        return sum(
            q[i] * jnp.exp(-((x - q[i + 1]) ** 2) / (2 * q[i + 2] ** 2))
            for i in (0, 3, 6)
        )

    def model_mp(x: Any, *q: Any) -> Any:
        return sum(
            q[i] * mp.exp(-((x - q[i + 1]) ** 2) / (2 * q[i + 2] ** 2))
            for i in (0, 3, 6)
        )

    def jac_mp(x: Any, *q: Any) -> list[Any]:
        out = []
        for i in (0, 3, 6):
            a, m, w = q[i : i + 3]
            e = mp.exp(-((x - m) ** 2) / (2 * w**2))
            out += [e, a * e * (x - m) / w**2, a * e * (x - m) ** 2 / w**3]
        return out

    truth = [1.0, 0.4, 0.08, 0.8, 0.5, 0.08, 1.2, 0.6, 0.08]
    return Problem("gauss3", model, model_mp, jac_mp, truth, (0.0, 1.0))


def problems() -> dict[str, Problem]:
    probs = [_two_exp(10.0**-k) for k in range(1, 6)]
    probs += [_poly(8), _poly(12), _gauss3()]
    return {p.name: p for p in probs}


def make_data(
    prob: Problem, seed: int, n: int, set_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic float64 ``(x, y, p0)``; the entropy names seed, problem and set."""
    tag = 0 if set_name == "accuracy" else 1
    rng = np.random.default_rng([seed, zlib.crc32(prob.name.encode()), tag])
    x = np.linspace(*prob.x_range, n)
    f = np.asarray(prob.model(jnp.asarray(x), *jnp.asarray(prob.truth)))
    y = f + NOISE * rng.standard_normal(n)
    p0 = prob.truth * (1 + P0_SPREAD * rng.standard_normal(prob.truth.size))
    return x, y, p0


def jacobian(model: Callable[..., Any], x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    xj = jnp.asarray(x)
    return np.asarray(
        jax.jacfwd(lambda q: model(xj, *q))(jnp.asarray(theta, dtype=jnp.float64))
    )


def fit_scipy(prob: Problem, x: np.ndarray, y: np.ndarray, p0: np.ndarray) -> dict:
    xj = jnp.asarray(x)
    fun = jax.jit(lambda q: prob.model(xj, *q) - y)
    jac = jax.jit(jax.jacfwd(lambda q: prob.model(xj, *q)))
    t = time.perf_counter()
    sp = least_squares(
        lambda q: np.asarray(fun(q)),
        p0,
        jac=lambda q: np.asarray(jac(q)),
        method="trf",
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        x_scale="jac",
    )
    cov, rank = svd_covariance(jacobian(prob.model, x, sp.x), float(sp.cost), len(x))
    return {
        "x": sp.x,
        "cost": float(sp.cost),
        "cov": cov,
        "rank": rank,
        "status": int(sp.status),
        "time_s": time.perf_counter() - t,
    }


def fit_normal_equations(
    prob: Problem, x: np.ndarray, y: np.ndarray, p0: np.ndarray, chunk: int
) -> dict:
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = exact_lm_fit(
            prob.model, x, y, p0, chunk_size=chunk, ftol=TOL, xtol=TOL, gtol=TOL
        )
    elapsed = time.perf_counter() - t
    # rank is not returned: recompute H at res.x with the module's own pass and cutoff
    H = run_pass(make_chunk_stats(prob.model), res.x, (x, y, None), chunk).H
    return {
        "x": np.asarray(res.x),
        "cost": float(res.cost),
        "cov": np.asarray(res.pcov),
        "rank": eigh_rank(H),
        "termination": res.termination,
        "nfev": int(res.nfev),
        "warnings": sorted({str(w.message) for w in caught}),
        "time_s": elapsed,
    }


def fit_tsqr(
    prob: Problem,
    x: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    chunk: int,
    max_nfev: int | None = None,
) -> dict:
    """LM on TSQR factors, mirroring ``exact_lm_fit``'s unbounded loop.

    Same ``LAM0``, Nielsen update, running-max damping ``_damping``, attempt cap, λ
    ceiling, ``step_termination`` and ``max_nfev = 100*p`` default. Stationarity: ``gtol``
    on ``g = R^T c``; the Newton certificate uses the undamped decrease ``0.5*|c|^2``,
    issued only when the Jacobi-scaled ``R`` keeps every direction (as the module).
    """
    t0 = time.perf_counter()
    chunks = [(x[s : s + chunk], y[s : s + chunk]) for s in range(0, len(x), chunk)]
    p = len(p0)
    max_nfev = 100 * p if max_nfev is None else max_nfev

    def factor(th: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        R, c, rho = tsqr_factor(chunks, th, prob.model)
        return R, c, 0.5 * float(c @ c + rho * rho)

    def stationary(R: np.ndarray, c: np.ndarray, F: float) -> str | None:
        g = R.T @ c
        if F == 0 or np.max(np.abs(g)) < TOL:
            return "gtol"
        cn = np.linalg.norm(R, axis=0)
        if np.all(cn > 0):
            s = np.linalg.svd(R / cn, compute_uv=False)
            if s[-1] ** 2 > p * EPS * s[0] ** 2 and 0.5 * float(c @ c) <= TOL * F:
                return "ftol"
        return None

    theta = np.asarray(p0, dtype=np.float64)
    R, c, F = factor(theta)
    nfev, lam, nu = 1, LAM0, 2.0
    D = np.zeros(p)
    while True:
        termination = stationary(R, c, F)
        if termination:
            break
        if nfev >= max_nfev:
            termination = "max_nfev"
            break
        g = R.T @ c
        D = _damping(D, R.T @ R)
        accepted = out_of_budget = False
        attempts = 0
        while attempts < ATTEMPT_CAP and lam <= LAMBDA_MAX:
            if nfev >= max_nfev:
                out_of_budget = True
                break
            attempts += 1
            s = tsqr_step(R, c, lam, D)
            if not np.isfinite(s).all():
                lam, nu = lam * nu, nu * 2
                continue
            new = theta + s
            Rs = R @ s
            pred = s @ g - 0.5 * Rs @ Rs
            R_new, c_new, F_new = factor(new)
            nfev += 1
            actual = F - F_new
            if np.isfinite(F_new) and actual > 0 and pred > 0:
                rho = actual / pred
                lam = min(
                    max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-14), LAMBDA_MAX
                )
                nu = 2.0
                accepted = True
                break
            lam, nu = lam * nu, nu * 2
        if not accepted:
            termination = "max_nfev" if out_of_budget else "stalled"
            break
        step = step_termination(
            actual, F, rho, np.linalg.norm(s), np.linalg.norm(theta), TOL, TOL
        )
        theta, R, c, F = new, R_new, c_new, F_new
        if step:
            termination = step
            break
    cov, rank = svd_covariance(R, F, len(x))
    return {
        "x": theta,
        "cost": F,
        "cov": cov,
        "rank": rank,
        "termination": termination,
        "nfev": nfev,
        "time_s": time.perf_counter() - t0,
    }


def _info(fit: dict) -> dict:
    """JSON-safe fit summary (parameters, cost, rank and bookkeeping, no covariance)."""
    out = {k: v for k, v in fit.items() if k not in ("x", "cov")}
    out["x"] = [float(v) for v in fit["x"]]
    return out


def run_row(prob: Problem, seed: int, set_name: str, n: int, chunk: int) -> dict:
    x, y, p0 = make_data(prob, seed, n, set_name)
    row: dict[str, Any] = {
        "set": set_name,
        "problem": prob.name,
        "seed": seed,
        "n": n,
        "p": len(p0),
        "chunk": chunk,
        "tol": TOL,
        "p0": [float(v) for v in p0],
    }
    sp = fit_scipy(prob, x, y, p0)
    ne = fit_normal_equations(prob, x, y, p0, chunk)
    ts = fit_tsqr(prob, x, y, p0, chunk)
    row["kappa_scipy"] = float(np.linalg.cond(jacobian(prob.model, x, sp["x"])))
    if set_name == "scale":
        ref = sp
        row["kappa"] = row["kappa_scipy"]
    else:
        t = time.perf_counter()
        try:
            theta_r, cost_r, cov_r, rank_r = mp_reference(
                prob.model_mp, x, y, sp["x"], jac_mp=prob.jac_mp
            )
        except Exception as exc:  # recorded per row; the summary prints every failure
            row.update(
                status="reference_failed",
                error=repr(exc),
                ref_time_s=time.perf_counter() - t,
                info={
                    "scipy": _info(sp),
                    "normal_equations": _info(ne),
                    "tsqr": _info(ts),
                },
            )
            return row
        ref = {"x": theta_r, "cost": cost_r, "cov": cov_r, "rank": rank_r}
        row["ref_time_s"] = time.perf_counter() - t
        row["reference"] = {
            "x": [float(v) for v in theta_r],
            "cost": cost_r,
            "rank": rank_r,
        }
        row["kappa"] = float(np.linalg.cond(jacobian(prob.model, x, theta_r)))
    cands = {"normal_equations": ne, "tsqr": ts}
    if set_name == "accuracy":
        cands["scipy"] = sp
    else:
        row["baseline_scipy"] = _info(sp)
    for name, fit in cands.items():
        errs = error_norms(
            fit["x"], ref["x"], fit["cov"], ref["cov"], fit["rank"], ref["rank"]
        )
        row[name] = {**errs, **_info(fit)}
    row["status"] = "ok"
    return row


def _path(out: Path, set_name: str, prob: str, seed: int) -> Path:
    return out / f"{set_name}__{prob}__seed{seed}.json"


def summarize(rows: list[dict], missing: list[str]) -> None:
    bad = [r for r in rows if r["status"] != "ok"]
    if missing or bad:
        print("\n" + "!" * 78)
        if missing:
            print(
                f"!! {len(missing)} ROW(S) MISSING (not computed): {', '.join(missing)}"
            )
        for r in bad:
            print(
                f"!! {r['set']} {r['problem']} seed {r['seed']}: {r['status']}: "
                f"{r.get('error', '')}  (kappa at SciPy optimum {r['kappa_scipy']:.3g})"
            )
        print("!! THESE ROWS ARE EXCLUDED FROM THE VERDICT")
        print("!" * 78)
    for set_name in ("accuracy", "scale"):
        ok = [r for r in rows if r["set"] == set_name and r["status"] == "ok"]
        if not ok:
            continue
        vs = "mpmath reference" if set_name == "accuracy" else "SciPy"
        print(f"\n== {set_name} set: worst errors versus {vs} ==")
        print(
            f"{'problem':14s} {'rows':>4s} {'kappa max':>10s}  "
            f"{'candidate':16s} {'param_err':>10s} {'cov_err':>10s} {'rank!=':>6s} "
            "terminations"
        )
        for prob in dict.fromkeys(r["problem"] for r in ok):
            pr = [r for r in ok if r["problem"] == prob]
            kmax = max(r["kappa"] for r in pr)
            for cand in ("normal_equations", "tsqr", "scipy"):
                es = [r[cand] for r in pr if cand in r]
                if not es:
                    continue
                terms = sorted({str(e.get("termination", e.get("status"))) for e in es})
                print(
                    f"{prob:14s} {len(pr):4d} {kmax:10.3g}  {cand:16s} "
                    f"{max(e['param_err'] for e in es):10.3g} "
                    f"{max(e['cov_err'] for e in es):10.3g} "
                    f"{sum(not e['rank_equal'] for e in es):6d} {','.join(terms)}"
                )
    acc = [r for r in rows if r["set"] == "accuracy" and r["status"] == "ok"]
    if acc:
        n_in = sum(r["kappa"] <= KAPPA_MAX for r in acc)
        try:
            v: str = verdict(acc)
        except ValueError as exc:
            v = f"undecided ({exc})"
        print(
            f"\nverdict over {len(acc)} accuracy rows ({n_in} with kappa <= 1e4): {v}"
        )
        if missing or bad:
            print(
                f"VERDICT EXCLUDES {len(missing)} missing and {len(bad)} failed rows "
                "(listed above)"
            )


def main(argv: list[str] | None = None) -> list[dict]:
    global TOL  # noqa: PLW0603 - one CLI setting read by both candidate fits
    _mp()
    probs = problems()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--problems", nargs="+", default=list(probs), choices=list(probs))
    ap.add_argument("--n-acc", type=int, default=2000)
    ap.add_argument("--n-scale", type=int, default=20000)
    ap.add_argument("--only", choices=["accuracy", "scale"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-12, help="candidate ftol=xtol=gtol")
    args = ap.parse_args(argv)
    TOL = args.tol
    args.out.mkdir(parents=True, exist_ok=True)
    sets = [s for s in ("accuracy", "scale") if args.only in (None, s)]
    rows, missing = [], []
    for set_name in sets:
        n = args.n_acc if set_name == "accuracy" else args.n_scale
        for name in args.problems:
            for seed in args.seeds:
                path = _path(args.out, set_name, name, seed)
                if args.resume and path.exists():
                    row = json.loads(path.read_text())
                    if (row["n"], row.get("tol")) != (n, TOL):
                        raise SystemExit(f"{path}: n, tol differ from this run")
                    rows.append(row)
                    continue
                t = time.perf_counter()
                try:
                    row = run_row(probs[name], seed, set_name, n, args.chunk)
                except Exception as exc:  # candidate or baseline crash: keep going
                    print(f"{set_name} {name} seed {seed}: CRASHED {exc!r}", flush=True)
                    missing.append(f"{set_name}/{name}/seed{seed}")
                    continue
                path.write_text(json.dumps(row, indent=1))
                rows.append(row)
                print(
                    f"{set_name} {name} seed {seed}: {row['status']} "
                    f"kappa={row.get('kappa', float('nan')):.3g} "
                    f"({time.perf_counter() - t:.1f} s)",
                    flush=True,
                )
    summarize(rows, missing)
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
