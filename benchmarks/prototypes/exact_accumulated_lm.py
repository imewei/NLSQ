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

jax.config.update("jax_enable_x64", True)

CHUNK = 1 << 22  # 4M points: J chunk is 4M x p float64 (about 130 MB for p=4)


def make_stats(model):
    """Jitted (theta, x, y, w) -> (0.5*sum w r^2, J^T r, J^T J); r = y - model."""

    def stats(theta, x, y, w):
        def f(p):
            return model(x, *p)

        r = (y - f(theta)) * w
        J = jax.jacfwd(f)(theta) * w[:, None]
        return 0.5 * jnp.sum(r * r), J.T @ r, J.T @ J

    return jax.jit(stats)


def _chunks(x, y, chunk):
    """Yield equal-shape (x, y, mask) chunks; the tail is edge-padded and masked out."""
    n = len(x)
    ones = jnp.ones(chunk)
    for s in range(0, n, chunk):
        xc, yc = jnp.asarray(x[s : s + chunk]), jnp.asarray(y[s : s + chunk])
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
    """One full pass: summed (cost, g, H) on the host."""
    th = jnp.asarray(theta)
    acc = None
    for xc, yc, w in _chunks(x, y, chunk):
        out = stats(th, xc, yc, w)
        acc = (
            out if acc is None else tuple(a + b for a, b in zip(acc, out, strict=True))
        )
    c, g, H = jax.device_get(acc)
    return float(c), np.asarray(g), np.asarray(H)


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
    max_nfev=100,
):
    """Exact full-data LM. Returns dict(x, cost, pcov, nfev=passes, success, message)."""
    x_dev_ok = hasattr(x, "shape")
    if not x_dev_ok or x.ndim != 1 or y.shape != x.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    n = len(x)
    p = len(p0)
    lo = np.broadcast_to(np.asarray(bounds[0], float), (p,))
    hi = np.broadcast_to(np.asarray(bounds[1], float), (p,))
    stats = make_stats(model)
    theta = np.clip(np.asarray(p0, float), lo, hi)
    cost, g, H = _pass(stats, theta, x, y, chunk)
    nfev, lam, nu = 1, 1e-3, 2.0
    g0 = max(np.max(np.abs(g)), 1e-300)
    message = "max_nfev reached"
    success = False
    while nfev < max_nfev:
        # g = J^T r is the descent direction; a variable at a bound with g pointing out is fixed
        free = ~(((theta <= lo) & (g <= 0)) | ((theta >= hi) & (g >= 0)))
        if not free.any() or np.max(np.abs(g[free])) <= gtol * g0:
            message, success = "gtol", True
            break
        Hf, gf = H[np.ix_(free, free)], g[free]
        accepted = False
        while nfev < max_nfev:
            try:
                d = np.linalg.solve(Hf + lam * np.diag(np.diag(Hf)), gf)
            except np.linalg.LinAlgError:
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
            if np.isfinite(c_new) and actual > 0 and pred > 0:
                rho = actual / pred
                lam = max(lam * max(1 / 3, 1 - (2 * rho - 1) ** 3), 1e-14)
                nu = 2.0
                accepted = True
                break
            lam, nu = lam * nu, nu * 2
            if lam > 1e14:
                break
        if not accepted:
            message = "no further progress" if lam > 1e14 else message
            success = lam > 1e14  # at the numerical floor, not a failure to run
            break
        small_step = np.linalg.norm(s) <= xtol * (xtol + np.linalg.norm(theta))
        small_cost = actual <= ftol * cost
        theta, cost, g, H = new, c_new, g_new, H_new
        if small_cost or small_step:
            message, success = ("ftol" if small_cost else "xtol"), True
            break
    s2 = 2.0 * cost / max(n - p, 1)
    try:
        pcov = s2 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        pcov = np.full((p, p), np.inf)
    return {
        "x": theta,
        "cost": cost,
        "pcov": pcov,
        "nfev": nfev,
        "success": success,
        "message": message,
    }


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


def _bench(problem, n, host, repeats=3):
    from benchmarks.compare.problems import PROBLEMS
    from benchmarks.compare.timing import residual_cost

    pr = PROBLEMS[problem]
    x, y = pr.data(n)
    model = lambda x, *q: pr.model(jnp, x, *q)  # noqa: E731
    b = pr.bounds if pr.bounds is not None else (-np.inf, np.inf)
    xd, yd = (x, y) if host else (jnp.asarray(x), jnp.asarray(y))
    jax.block_until_ready(jnp.ones(3) + 1)
    times, out = [], None
    for _ in range(repeats + 1):  # first call is cold
        t = time.perf_counter()
        out = fit(model, xd, yd, pr.p0, b)
        times.append(time.perf_counter() - t)
    cost = residual_cost(pr, x, y, out["x"])
    print(
        f"{problem} n={n:,} backend={jax.default_backend()} data={'host' if host else 'device'} "
        f"cold={times[0]:.2f}s warm_median={np.median(times[1:]):.2f}s passes={out['nfev']} "
        f"cost={cost!r} params={np.round(out['x'], 5).tolist()} ({out['message']})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", nargs=2, metavar=("PROBLEM", "N"))
    ap.add_argument(
        "--host", action="store_true", help="keep data on host, stream chunks"
    )
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    if a.bench:
        _bench(a.bench[0], int(a.bench[1]), a.host)
