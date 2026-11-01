"""Exact chunk-accumulated Levenberg-Marquardt prototype (not wired into nlsq).

One pass over the data per trial point accumulates (0.5*sum r^2, J^T r, J^T J) in float64, so
the state is O(p^2) and the result equals a full-batch fit regardless of chunking. The
accepted trial's statistics are reused as the next iterate's, so a pass is never wasted on
re-evaluating an accepted point.

    uv run python -m benchmarks.prototypes.exact_accumulated_lm --selftest
    python -m benchmarks.prototypes.exact_accumulated_lm --bench exp_decay 100000000 [--host]

shortcuts: normal equations square cond(J) (upgrade to chunked QR/TSQR if two_exp-like
problems lose accuracy); no sigma/weights; bounds by active-set projection, not full TRF;
tail chunk is edge-padded and masked, so every chunk shares one compiled shape.
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from nlsq.result.optimize_result import OptimizeResult

jax.config.update("jax_enable_x64", True)

CHUNK = 1 << 22  # 4M points: J chunk is 4M x p float64 (about 130 MB for p=4)
LAMBDA_MAX = 1e14  # damping ceiling
ATTEMPT_CAP = 20  # solve attempts per iteration
DAMPING_FLOOR = 1e-10  # D_ii >= DAMPING_FLOOR * max_j D_jj


def make_stats(model):
    """Jitted (theta, x, y, w) -> (0.5*sum w r^2, J^T r, J^T J, data_finite); r = y - model."""

    def stats(theta, x, y, w):
        def f(p):
            return model(x, *p)

        r = (y - f(theta)) * w
        J = jax.jacfwd(f)(theta) * w[:, None]
        finite = jnp.all(
            jnp.isfinite(x) & jnp.isfinite(y)
        )  # padded rows copy real rows
        return 0.5 * jnp.sum(r * r), J.T @ r, J.T @ J, finite

    return jax.jit(stats)


def _chunks(x, y, chunk):
    """Yield equal-shape (x, y, mask) chunks; the tail is edge-padded and masked out."""
    n = len(x)
    ones = jnp.ones(chunk)
    for s in range(0, n, chunk):
        xc = jnp.asarray(x[s : s + chunk], dtype=jnp.float64)
        yc = jnp.asarray(y[s : s + chunk], dtype=jnp.float64)
        m = xc.shape[0]
        if m == chunk:
            yield xc, yc, ones
        else:
            pad = (0, chunk - m)
            yield (
                jnp.pad(xc, pad, mode="edge"),
                jnp.pad(yc, pad, mode="edge"),
                jnp.pad(jnp.ones(m), pad),
            )


def _pass(stats, theta, x, y, chunk):
    """One full pass: summed (cost, g, H) on the host; raises if x or y is non-finite."""
    th = jnp.asarray(theta, dtype=jnp.float64)
    acc = None
    for xc, yc, w in _chunks(x, y, chunk):
        out = stats(th, xc, yc, w)
        acc = (
            out
            if acc is None
            else (
                *(a + b for a, b in zip(acc[:3], out[:3], strict=True)),
                acc[3] & out[3],
            )
        )
    c, g, H, data_finite = jax.device_get(acc)
    if not data_finite:
        raise ValueError("x and y must be finite (found NaN or inf)")
    return float(c), np.asarray(g), np.asarray(H)


def step_termination(dF, F, rho, s_norm, x_norm, ftol, xtol):
    """Step-based criteria after an accepted step; F is the cost before the step."""
    if dF < ftol * F and rho > 0.25:
        return "ftol"
    if s_norm < xtol * (xtol + x_norm):
        return "xtol"
    return None


def _scaled_eig(g, H):
    """Jacobi-scaled (g_hat, eigvals, eigvecs, truncated) over columns with H_ii > 0."""
    d = np.diag(H)
    keep = d > 0
    s = 1.0 / np.sqrt(d[keep])
    Hs = H[np.ix_(keep, keep)] * s[:, None] * s[None, :]
    gs = g[keep] * s
    if gs.size == 0:
        return gs, gs, np.zeros((0, 0)), False
    w, U = np.linalg.eigh(Hs)
    truncated = not w[0] > len(g) * np.finfo(float).eps * w[-1]
    return gs, w, U, truncated


def newton_certificate(g, H, F, ftol, bounds_active=False):
    """Damping-independent stationarity: undamped GN step cannot gain ftol*F (spec)."""
    if F == 0:
        return True
    if bounds_active or not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
        return False
    gs, w, U, truncated = _scaled_eig(g, H)
    if truncated:
        return False
    delta_n = 0.5 * float(np.sum((U.T @ gs) ** 2 / w)) if gs.size else 0.0
    return delta_n <= ftol * F


def _stationary(cost, g, H, free, gtol, ftol):
    if cost == 0:  # exact fit: decided before any division
        return "gtol"
    if not free.any() or np.max(np.abs(g[free])) < gtol:
        return "gtol"
    if newton_certificate(g, H, cost, ftol, bounds_active=not free.all()):
        return "ftol"
    return None


_MESSAGES = {
    "gtol": "gradient below gtol (or exact fit)",
    "ftol": "cost decrease below ftol",
    "xtol": "step below xtol",
    "max_nfev": "maximum number of data passes reached",
    "stalled": (
        f"no further progress: trials rejected until damping reached {LAMBDA_MAX:g}"
        f" or the {ATTEMPT_CAP}-attempt cap"
    ),
}


def _damping(D_prev, H):
    """Moré running maximum of diag(H), floored; identity if every diagonal is zero."""
    D = np.maximum(D_prev, np.diag(H))
    top = D.max()
    return np.maximum(D, DAMPING_FLOOR * top) if top > 0 else np.ones_like(D)


def _check_lam0(lam0):
    if not 0 < lam0 <= LAMBDA_MAX:
        raise ValueError(f"lam0 must satisfy 0 < lam0 <= {LAMBDA_MAX:g}")


def _finite(c, g, H):
    return bool(np.isfinite(c) and np.isfinite(g).all() and np.isfinite(H).all())


def _solve(Hf, lam, Df, gf):
    """Damped step, or None if the solve fails or is non-finite."""
    try:
        d = np.linalg.solve(Hf + lam * np.diag(Df), gf)
    except np.linalg.LinAlgError:
        return None
    return d if np.isfinite(d).all() else None


def fit(
    model,
    x,
    y,
    p0,
    bounds=(-np.inf, np.inf),
    chunk=CHUNK,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_nfev=None,
    lam0=1e-3,
):
    """Exact full-data LM. `max_nfev` counts full-data passes (default 100*p).

    `lam0` is the initial damping (0 < lam0 <= 1e14); tests use it to place lambda near a limit.
    """
    _check_lam0(lam0)
    x_dev_ok = hasattr(x, "shape")
    if not x_dev_ok or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = len(x)
    p = len(p0)
    if max_nfev is None:
        max_nfev = 100 * p
    lo = np.broadcast_to(np.asarray(bounds[0], float), (p,))
    hi = np.broadcast_to(np.asarray(bounds[1], float), (p,))
    stats = make_stats(model)
    theta = np.clip(np.asarray(p0, np.float64), lo, hi)
    cost, g, H = _pass(stats, theta, x, y, chunk)
    if not _finite(cost, g, H):
        raise ValueError("model cost, gradient or Hessian is not finite at p0")
    nfev, lam, nu = 1, lam0, 2.0
    D = np.zeros(p)  # Moré running maximum of diag(H)
    while True:
        # g = J^T r is the descent direction; a variable at a bound with g pointing out is fixed
        free = ~(((theta <= lo) & (g <= 0)) | ((theta >= hi) & (g >= 0)))
        termination = _stationary(cost, g, H, free, gtol, ftol)
        if termination:
            break
        if nfev >= max_nfev:
            termination = "max_nfev"
            break
        Hf, gf = H[np.ix_(free, free)], g[free]
        D = _damping(D, H)
        accepted = out_of_budget = False
        attempts = 0
        while attempts < ATTEMPT_CAP and lam <= LAMBDA_MAX:
            if nfev >= max_nfev:
                out_of_budget = True
                break
            attempts += 1
            d = _solve(Hf, lam, D[free], gf)
            if d is None:  # failed solve: no data pass, still an attempt
                lam, nu = lam * nu, nu * 2
                continue
            new = theta.copy()
            new[free] += d
            new = np.clip(new, lo, hi)
            s = new - theta
            pred = s @ g - 0.5 * s @ H @ s  # quadratic-model decrease in 0.5*sum r^2
            c_new, g_new, H_new = _pass(stats, new, x, y, chunk)
            nfev += 1
            actual = cost - c_new
            if _finite(c_new, g_new, H_new) and actual > 0 and pred > 0:
                rho = actual / pred
                # cap: an acceptance must not leave lam above the ceiling (next iteration would skip its trials)
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
            actual, cost, rho, np.linalg.norm(s), np.linalg.norm(theta), ftol, xtol
        )
        theta, cost, g, H = new, c_new, g_new, H_new
        if step:
            termination = step
            break
    message = _MESSAGES[termination]
    if termination == "stalled" and _scaled_eig(g, H)[3]:
        message += "; rank cannot be certified from normal equations"
    s2 = 2.0 * cost / max(n - p, 1)
    try:
        pcov = s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        pcov = np.full((p, p), np.inf)
    return OptimizeResult(
        x=theta,
        cost=cost,
        pcov=pcov,
        nfev=nfev,
        success=termination in ("gtol", "ftol", "xtol"),
        message=message,
        termination=termination,
    )


def _selftest():
    from scipy.optimize import least_squares

    from benchmarks.compare.problems import PROBLEMS

    for name, pr in PROBLEMS.items():
        x, y = pr.data(3000)
        model = lambda x, *q, m=pr.model: m(jnp, x, *q)  # noqa: E731
        b = pr.bounds if pr.bounds is not None else (-np.inf, np.inf)
        tight = {
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
        }  # same as the SciPy reference
        small = fit(model, x, y, pr.p0, b, chunk=512, **tight)  # 6 chunks, padded tail
        whole = fit(model, x, y, pr.p0, b, chunk=4096, **tight)  # one padded chunk
        # exactness: result must not depend on how the data is chunked
        np.testing.assert_allclose(small["x"], whole["x"], rtol=1e-9, atol=1e-12)
        ref = least_squares(
            lambda q, pr=pr, x=x, y=y: y - pr.model(np, x, *q),
            pr.p0,
            bounds=b,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        assert small["cost"] <= ref.cost * (1 + 1e-9), (name, small["cost"], ref.cost)
        mine, theirs = small["x"], ref.x
        if name == "two_exp":  # a1,k1,a2,k2 is symmetric under swapping the two terms
            mine, theirs = (
                np.concatenate(sorted(q.reshape(2, 2).tolist(), key=lambda t: t[1]))
                for q in (mine, theirs)
            )
        np.testing.assert_allclose(mine, theirs, rtol=1e-5, atol=1e-6)
        print(f"{name:12s} ok  passes={small['nfev']:2d}  cost={small['cost']:.6f}")


def _module_fit(model, x, y, p0, b):
    """`--impl module`: the library engine with this prototype's chunk and tolerances."""
    from nlsq.streaming.exact_lm import exact_lm_fit

    # check_finite=False: the prototype has no finite-data pass, so time the solver alone
    return exact_lm_fit(model, x, y, p0, bounds=b, chunk_size=CHUNK, check_finite=False)


def _bench(problem, n, host, repeats=3, impl="prototype"):
    from benchmarks.compare.problems import PROBLEMS
    from benchmarks.compare.timing import residual_cost

    pr = PROBLEMS[problem]
    x, y = pr.data(n)
    model = lambda x, *q: pr.model(jnp, x, *q)  # noqa: E731
    b = pr.bounds if pr.bounds is not None else (-np.inf, np.inf)
    xd, yd = (x, y) if host else (jnp.asarray(x), jnp.asarray(y))
    jax.block_until_ready(jnp.ones(3) + 1)
    run = fit if impl == "prototype" else _module_fit
    times, out = [], None
    for _ in range(repeats + 1):  # first call is cold
        t = time.perf_counter()
        out = run(model, xd, yd, pr.p0, b)
        times.append(time.perf_counter() - t)
    cost = residual_cost(pr, x, y, out["x"])
    print(
        f"impl={impl} {problem} n={n:,} backend={jax.default_backend()} data={'host' if host else 'device'} "
        f"cold={times[0]:.2f}s warm_median={np.median(times[1:]):.2f}s passes={out['nfev']} "
        f"cost={cost!r} params={np.round(out['x'], 5).tolist()} ({out['message']})"
    )
    return {"impl": impl, "cost": cost, "nfev": out["nfev"], "x": out["x"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", nargs=2, metavar=("PROBLEM", "N"))
    ap.add_argument(
        "--host", action="store_true", help="keep data on host, stream chunks"
    )
    ap.add_argument(
        "--impl",
        choices=("prototype", "module"),
        default="prototype",
        help="--bench only: time this file's fit or nlsq.streaming.exact_lm.exact_lm_fit",
    )
    ap.add_argument("--repeats", type=int, default=3, help="--bench warm repeats")
    a = ap.parse_args(argv)
    if a.selftest:
        _selftest()
    if a.bench:
        return _bench(a.bench[0], int(a.bench[1]), a.host, a.repeats, a.impl)


if __name__ == "__main__":
    main()
