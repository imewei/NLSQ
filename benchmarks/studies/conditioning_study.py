"""Stage 3 conditioning study: normal equations versus TSQR (pre-registered rule).

Usage::

    python -m benchmarks.studies.conditioning_study --out <dir> [--seeds 0 1 2 3 4]
        [--problems ...] [--n-acc 2000] [--n-scale 20000] [--only accuracy|scale]
        [--resume] [--chunk 512] [--tol 1e-12]

One JSON row per (set, problem, seed) is written atomically to ``<dir>`` as soon as it is
computed (``--resume`` skips rows already on disk, after checking that n, tol, chunk and
the problem definition match). Every solver of one (problem, seed, set) sees the same
float64 observations: the normal-equations candidate (``exact_lm_fit``), the TSQR
candidate (an LM loop in this file on ``tsqr_step``), the SciPy baseline and, on the
accuracy set, the mpmath reference, started from the normal-equations solution (the
candidate being judged) with inputs converted exactly to mpf. Both candidates read the
data in ``--chunk`` sized chunks. Parameters are compared after sorting the terms of
two_exp (by rate) and gauss3 (by centre), so a label swap is not an error.

Rank agreement is vacuous inside the rule's range: ``kappa(J) <= 1e4`` gives
``kappa(H) <= 1e8``, far below the ``1/(p*eps)`` at which the cutoff drops a direction.

Each accuracy row also records tolerance-independent diagnostics: the undamped step and
covariance of both candidates' algebra at the reference optimum (``same_theta_ne``,
``same_theta_tsqr``), and their steps at the normal-equations end point against one
extended-precision Gauss-Newton step (``mp_step_at_ne_theta``).
"""

import argparse
import json
import os
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
    COV_TOL,
    KAPPA_MAX,
    PARAM_TOL,
    eigh_rank,
    error_norms,
    mp_gn_step,
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
    compute_covariance,
    exact_lm_fit,
    make_chunk_stats,
    run_pass,
    step_termination,
)

NOISE = 1e-3
P0_SPREAD = 0.05  # p0 = truth * (1 + 0.05 * N(0, 1))
EPS = float(np.finfo(np.float64).eps)
BASIN_COST_RTOL = 1e-10  # basin_ok: |cost - cost_ref| <= 1e-10 * cost_ref
BASIN_PARAM_TOL = 1e-2  # basin_ok: param_err < 1e-2 (after the term sort)
CANDIDATES = ("normal_equations", "tsqr", "scipy")
# SciPy is a float64 baseline, not a judged candidate: its basin must not gate the rule.
JUDGED = ("normal_equations", "tsqr")

Chunks = list[tuple[np.ndarray, np.ndarray]]


class Problem:
    def __init__(
        self,
        name: str,
        model: Callable[..., Any],
        model_mp: Callable[..., Any],
        jac_mp: Callable[..., Sequence[Any]],
        truth: Sequence[float],
        x_range: tuple[float, float],
        group: int = 0,
    ) -> None:
        self.name, self.model, self.model_mp, self.jac_mp = (
            name,
            model,
            model_mp,
            jac_mp,
        )
        self.truth = np.asarray(truth, dtype=np.float64)
        self.x_range = x_range
        self.group = group  # term size; terms sorted by their second entry (0: no sort)

    def perm(self, theta: np.ndarray) -> np.ndarray:
        """Parameter order with the terms sorted by rate (two_exp) or centre (gauss3)."""
        if self.group == 0:
            return np.arange(theta.size)
        order = np.argsort(theta.reshape(-1, self.group)[:, 1], kind="stable")
        return (order[:, None] * self.group + np.arange(self.group)).ravel()

    def fingerprint(self) -> dict[str, Any]:
        return {
            "truth": self.truth.tolist(),
            "x_range": list(self.x_range),
            "noise": NOISE,
            "p0_spread": P0_SPREAD,
        }


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
        group=2,
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
    return Problem("gauss3", model, model_mp, jac_mp, truth, (0.0, 1.0), group=3)


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
    # 1e-15 with x_scale='jac' as a strong float64 baseline; NOT tolerance-matched to the
    # candidates (which run at --tol), so it is not a like-for-like comparator
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
    prob: Problem, x: np.ndarray, y: np.ndarray, p0: np.ndarray, chunk: int, tol: float
) -> dict:
    t = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = exact_lm_fit(
            prob.model, x, y, p0, chunk_size=chunk, ftol=tol, xtol=tol, gtol=tol
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
    tol: float,
    max_nfev: int | None = None,
) -> dict:
    """LM on TSQR factors, mirroring ``exact_lm_fit``'s unbounded loop.

    Same ``LAM0``, Nielsen update, running-max damping ``_damping``, attempt cap, λ
    ceiling, ``step_termination`` and ``max_nfev = 100*p`` default. Stationarity: ``gtol``
    on ``g = R^T c``; the Newton certificate uses the undamped decrease ``0.5*|c|^2``,
    issued only when the Jacobi-scaled ``R`` keeps every direction (as the module).
    """
    t0 = time.perf_counter()
    chunks = _chunks(x, y, chunk)
    p = len(p0)
    max_nfev = 100 * p if max_nfev is None else max_nfev

    def factor(th: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        R, c, rho = tsqr_factor(chunks, th, prob.model)
        return R, c, 0.5 * float(c @ c + rho * rho)

    def stationary(R: np.ndarray, c: np.ndarray, F: float) -> str | None:
        g = R.T @ c
        if F == 0 or np.max(np.abs(g)) < tol:
            return "gtol"
        cn = np.linalg.norm(R, axis=0)
        if np.all(cn > 0):
            s = np.linalg.svd(R / cn, compute_uv=False)
            if s[-1] ** 2 > p * EPS * s[0] ** 2 and 0.5 * float(c @ c) <= tol * F:
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
            actual, F, rho, np.linalg.norm(s), np.linalg.norm(theta), tol, tol
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


def _chunks(x: np.ndarray, y: np.ndarray, chunk: int) -> Chunks:
    return [(x[s : s + chunk], y[s : s + chunk]) for s in range(0, len(x), chunk)]


def _guard(fit: Callable[[], dict], p: int) -> dict:
    """Run one solver; an exception becomes that solver's recorded failure."""
    t = time.perf_counter()
    try:
        return fit()
    except Exception as exc:  # recorded as the solver's failure, never a missing row
        return {
            "x": np.full(p, np.nan),
            "cost": float("nan"),
            "cov": np.full((p, p), np.nan),
            "rank": -1,
            "termination": "crashed",
            "error": repr(exc),
            "time_s": time.perf_counter() - t,
        }


def _info(fit: dict) -> dict:
    """JSON-safe copy of a fit (parameters and covariance as lists)."""
    out = {k: v for k, v in fit.items() if k not in ("x", "cov")}
    out["x"] = np.asarray(fit["x"], dtype=float).tolist()
    out["cov"] = np.asarray(fit["cov"], dtype=float).tolist()
    return out


def _passes(e: dict) -> bool:
    """The rule's criteria on one candidate's errors (NaN or inf fails)."""
    return bool(
        e["param_err"] <= PARAM_TOL and e["cov_err"] <= COV_TOL and e["rank_equal"]
    )


def compare(prob: Problem, fit: dict, ref: dict) -> dict:
    """Errors versus ``ref`` after the term sort, ``basin_ok`` and the fit itself."""
    if "error" in fit:
        errs: dict[str, Any] = {
            "param_err": float("inf"),
            "cov_err": float("inf"),
            "rank_equal": False,
        }
    else:
        pf, pr = prob.perm(fit["x"]), prob.perm(ref["x"])
        errs = error_norms(
            fit["x"][pf],
            ref["x"][pr],
            fit["cov"][np.ix_(pf, pf)],
            ref["cov"][np.ix_(pr, pr)],
            fit["rank"],
            ref["rank"],
        )
    errs["basin_ok"] = bool(
        abs(fit["cost"] - ref["cost"]) <= BASIN_COST_RTOL * abs(ref["cost"])
        and errs["param_err"] < BASIN_PARAM_TOL
    )
    return {**errs, **_info(fit)}


def _steps_at(
    prob: Problem,
    x: np.ndarray,
    y: np.ndarray,
    chunk: int,
    theta: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """``(pass stats, delta_NE, R, delta_TSQR)``: undamped steps of both algebras."""
    st = run_pass(make_chunk_stats(prob.model), theta, (x, y, None), chunk)
    try:
        d_ne = np.linalg.solve(st.H, st.g)
    except np.linalg.LinAlgError:
        d_ne = np.full(theta.size, np.nan)
    R, c, _ = tsqr_factor(_chunks(x, y, chunk), theta, prob.model)
    d_ts = tsqr_step(R, c, 0.0, np.zeros(theta.size))
    return st, d_ne, R, d_ts


def _rel(d: np.ndarray, theta: np.ndarray) -> float:
    return float(np.max(np.abs(d) / np.maximum(np.abs(theta), 1e-12)))


def same_theta(
    prob: Problem,
    x: np.ndarray,
    y: np.ndarray,
    chunk: int,
    theta: np.ndarray,
    cost: float,
    cov_ref: np.ndarray,
) -> dict:
    """Both algebras at the reference optimum: step size and covariance error."""
    n, p = len(x), theta.size
    st, d_ne, R, d_ts = _steps_at(prob, x, y, chunk, theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # its ill-conditioning warning is expected here
        cov_ne = compute_covariance(st.H, cost, n, p, False)
    cov_ts, _ = svd_covariance(R, cost, n)
    out = {}
    for key, d, cov in (
        ("same_theta_ne", d_ne, cov_ne),
        ("same_theta_tsqr", d_ts, cov_ts),
    ):
        cov_err = error_norms(theta, theta, cov, cov_ref, 0, 0)["cov_err"]
        out[key] = {"step_rel": _rel(d, theta), "cov_err": cov_err}
    return out


def mp_step_at(
    prob: Problem, x: np.ndarray, y: np.ndarray, chunk: int, theta: np.ndarray
) -> dict:
    """Both algebras' undamped steps at ``theta`` against one 50-digit GN step."""
    d_mp = mp_gn_step(prob.model_mp, prob.jac_mp, x, y, theta)
    _, d_ne, _, d_ts = _steps_at(prob, x, y, chunk, theta)
    return {
        "mp_step_rel": _rel(d_mp, theta),
        "ne_step_err": _rel(d_ne - d_mp, theta),
        "tsqr_step_err": _rel(d_ts - d_mp, theta),
        "note": "steps only; a single mp step gives no reference covariance",
    }


def run_row(
    prob: Problem, seed: int, set_name: str, n: int, chunk: int, tol: float
) -> dict:
    x, y, p0 = make_data(prob, seed, n, set_name)
    p = len(p0)
    row: dict[str, Any] = {
        "set": set_name,
        "problem": prob.name,
        "seed": seed,
        "n": n,
        "p": p,
        "chunk": chunk,
        "tol": tol,
        "problem_def": prob.fingerprint(),
        "p0": p0.tolist(),
    }
    sp = _guard(lambda: fit_scipy(prob, x, y, p0), p)
    ne = _guard(lambda: fit_normal_equations(prob, x, y, p0, chunk, tol), p)
    ts = _guard(lambda: fit_tsqr(prob, x, y, p0, chunk, tol), p)
    row["kappa_scipy"] = (
        float("nan")
        if "error" in sp
        else float(np.linalg.cond(jacobian(prob.model, x, sp["x"])))
    )
    cands = {"normal_equations": ne, "tsqr": ts}
    if set_name == "scale":
        if "error" in sp:
            raise RuntimeError(f"SciPy baseline failed: {sp['error']}")
        ref = sp
        row["kappa"] = row["kappa_scipy"]
        row["baseline_scipy"] = _info(sp)
    else:
        cands["scipy"] = sp
        start, row["ref_start"] = (
            (ne["x"], "normal_equations") if "error" not in ne else (sp["x"], "scipy")
        )
        t = time.perf_counter()
        try:
            theta_r, cost_r, cov_r, rank_r = mp_reference(
                prob.model_mp, x, y, start, jac_mp=prob.jac_mp
            )
        except Exception as exc:  # recorded per row; the summary prints every failure
            row.update(
                status="reference_failed",
                error=repr(exc),
                ref_time_s=time.perf_counter() - t,
                info={k: _info(v) for k, v in (*cands.items(), ("scipy", sp))},
            )
            return row
        ref = {"x": theta_r, "cost": cost_r, "cov": cov_r, "rank": rank_r}
        row["ref_time_s"] = time.perf_counter() - t
        row["reference"] = _info(ref)
        row["kappa"] = float(np.linalg.cond(jacobian(prob.model, x, theta_r)))
        row.update(same_theta(prob, x, y, chunk, theta_r, cost_r, cov_r))
        if "error" in ne:
            row["mp_step_at_ne_theta"] = {"note": "skipped: normal equations crashed"}
        else:
            row["mp_step_at_ne_theta"] = mp_step_at(prob, x, y, chunk, ne["x"])
    for name, fit in cands.items():
        row[name] = compare(prob, fit, ref)
    row["basin_ok"] = all(row[name]["basin_ok"] for name in JUDGED)
    row["status"] = "ok"
    return row


def _path(out: Path, set_name: str, prob: str, seed: int) -> Path:
    return out / f"{set_name}__{prob}__seed{seed}.json"


def _write(path: Path, row: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(row, indent=1, default=float))
    os.replace(tmp, path)


def _in_range(kappa: float) -> bool:
    """Inside the rule's range, or unknown (NaN counts as inside: it cannot be ruled out)."""
    return not kappa > KAPPA_MAX


def _tables(rows: list[dict], acc: list[dict]) -> None:
    """Worst end-to-end errors per problem, then per-row same-theta diagnostics."""
    for set_name in ("accuracy", "scale"):
        ok = [r for r in rows if r["set"] == set_name and r["status"] == "ok"]
        if not ok:
            continue
        vs = "mpmath reference" if set_name == "accuracy" else "SciPy"
        print(f"\n== {set_name} set: worst end-to-end errors versus {vs} ==")
        print(
            f"{'problem':14s} {'rows':>4s} {'kappa max':>10s}  {'candidate':16s} "
            f"{'param_err':>10s} {'cov_err':>10s} {'rank!=':>6s} {'basin!':>6s} "
            "terminations"
        )
        for prob in dict.fromkeys(r["problem"] for r in ok):
            pr = [r for r in ok if r["problem"] == prob]
            kmax = max(r["kappa"] for r in pr)
            for cand in CANDIDATES:
                es = [r[cand] for r in pr if cand in r]
                if not es:
                    continue
                terms = sorted({str(e.get("termination", e.get("status"))) for e in es})
                print(
                    f"{prob:14s} {len(pr):4d} {kmax:10.3g}  {cand:16s} "
                    f"{max(e['param_err'] for e in es):10.3g} "
                    f"{max(e['cov_err'] for e in es):10.3g} "
                    f"{sum(not e['rank_equal'] for e in es):6d} "
                    f"{sum(not e['basin_ok'] for e in es):6d} {','.join(terms)}"
                )
    if acc:
        print(
            "\n== accuracy rows: end-to-end errors next to tolerance-independent "
            "diagnostics =="
        )
        print(
            f"{'problem':14s} {'seed':>4s} {'kappa':>9s} | {'NE p/cov':>19s} | "
            f"{'TSQR p/cov':>19s} | {'sameθ NE step/cov':>19s} | "
            f"{'sameθ TSQR step/cov':>19s} | mp-step err NE/TSQR"
        )
        for r in acc:
            ne, ts = r["normal_equations"], r["tsqr"]
            sn, st = r["same_theta_ne"], r["same_theta_tsqr"]
            mp_s = r["mp_step_at_ne_theta"]
            mp_txt = (
                f"{mp_s['ne_step_err']:.2e}/{mp_s['tsqr_step_err']:.2e}"
                if "ne_step_err" in mp_s
                else mp_s["note"]
            )
            print(
                f"{r['problem']:14s} {r['seed']:4d} {r['kappa']:9.3g} | "
                f"{ne['param_err']:9.2e}/{ne['cov_err']:9.2e} | "
                f"{ts['param_err']:9.2e}/{ts['cov_err']:9.2e} | "
                f"{sn['step_rel']:9.2e}/{sn['cov_err']:9.2e} | "
                f"{st['step_rel']:9.2e}/{st['cov_err']:9.2e} | {mp_txt}"
            )


def summarize(rows: list[dict], missing: list[str]) -> None:
    failed = [r for r in rows if r["status"] != "ok"]
    acc = [r for r in rows if r["set"] == "accuracy" and r["status"] == "ok"]
    for r in acc:  # recompute so rows stored by an older run use the same definition
        r["basin_ok"] = all(r[c]["basin_ok"] for c in JUDGED)
    basin_bad = [r for r in acc if not r["basin_ok"]]
    blocking = [
        *(f"missing {m}" for m in missing),
        *(
            f"{r['problem']} seed {r['seed']} reference failed"
            for r in failed
            if _in_range(r["kappa_scipy"])
        ),
        *(
            f"{r['problem']} seed {r['seed']} basin mismatch"
            for r in basin_bad
            if _in_range(r["kappa"])
        ),
    ]
    if missing or failed or any(_in_range(r["kappa"]) for r in basin_bad):
        print("\n" + "!" * 78)
        for m in missing:
            print(f"!! MISSING (crashed, not computed): {m}")
        for r in failed:
            print(
                f"!! {r['set']} {r['problem']} seed {r['seed']}: {r['status']}: "
                f"{r.get('error', '')} (kappa at SciPy optimum {r['kappa_scipy']:.3g})"
            )
        for r in basin_bad:
            if _in_range(r["kappa"]):
                off = [c for c in JUDGED if not r[c]["basin_ok"]]
                print(
                    f"!! accuracy {r['problem']} seed {r['seed']}: BASIN MISMATCH "
                    f"({', '.join(off)}) at kappa {r['kappa']:.3g} <= 1e4"
                )
        print(
            "!! rows above are EXCLUDED from the rule; in-range ones make it UNDECIDED"
        )
        print("!" * 78)
    out_basin = [r for r in basin_bad if not _in_range(r["kappa"])]
    if out_basin:
        print(f"\n{len(out_basin)} accuracy row(s) above kappa 1e4 with basin mismatch")
    _tables(rows, acc)
    covered = [r for r in acc if _in_range(r["kappa"])]
    if covered:
        ts_ok = sum(_passes(r["tsqr"]) for r in covered)
        print(
            f"\npremise: TSQR meets the same criteria on {ts_ok}/{len(covered)} rows "
            "with kappa <= 1e4"
        )
    try:
        v = verdict(acc) if acc else None
        reason = "" if acc else "no accuracy row with a valid reference"
    except ValueError as exc:
        v, reason = None, str(exc)
    if blocking:
        reason = "; ".join(blocking)
        v = None
    if v is None:
        print(f"verdict = UNDECIDED ({reason})")
        return
    both = [
        f"{r['problem']} seed {r['seed']}"
        for r in covered
        if not _passes(r["normal_equations"]) and not _passes(r["tsqr"])
    ]
    if v == "tsqr_default" and both:
        print(
            "verdict = tsqr_default -- stopping-limited / undecided (TSQR fails the "
            f"same rows: {', '.join(both)})"
        )
    else:
        print(f"verdict = {v}")


def main(argv: list[str] | None = None) -> list[dict]:
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
                    want = (n, args.tol, args.chunk, probs[name].fingerprint())
                    have = tuple(
                        row.get(k) for k in ("n", "tol", "chunk", "problem_def")
                    )
                    if have != want:
                        raise SystemExit(
                            f"{path}: n/tol/chunk/problem definition differ from this run"
                        )
                    rows.append(row)
                    continue
                t = time.perf_counter()
                try:
                    row = run_row(probs[name], seed, set_name, n, args.chunk, args.tol)
                except Exception as exc:  # e.g. scale-set SciPy crash: keep going
                    print(f"{set_name} {name} seed {seed}: CRASHED {exc!r}", flush=True)
                    missing.append(f"{set_name}/{name}/seed{seed}")
                    continue
                _write(path, row)
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
