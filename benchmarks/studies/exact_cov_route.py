"""One-off: exact_lm_fit covariance route at 10M and the TSQR pass cost at 10M / 100M.

Gate (route and cost) of docs/superpowers/specs/2026-11-10-close-jaxopt-gap-design.md.
Run on the GPU: ``uv run python -m benchmarks.studies.exact_cov_route``.
"""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.compare.problems import PROBLEMS
from nlsq.core.covariance import normal_condition
from nlsq.streaming import exact_lm
from nlsq.streaming.normal_equations import make_chunk_stats, run_pass


def jmodel(prob):
    return lambda x, *q: prob.model(jnp, x, *q)


def route_table(n: int) -> None:
    print(f"## route at n={n:,} ({jax.default_backend()})")
    for name, prob in PROBLEMS.items():
        x, y = prob.data(n, seed=0)
        model = jmodel(prob)
        kw = {} if prob.bounds is None else {"bounds": prob.bounds}
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            res = exact_lm.exact_lm_fit(model, x, y, prob.p0, **kw)
        dt = time.perf_counter() - t0
        chunk = exact_lm.derive_chunk_size(n, len(prob.p0), None, None)
        src, w = exact_lm._open_source(x, y, None, chunk, len(prob.p0), None, None)
        H = run_pass(make_chunk_stats(model, weighted=w), res.x, src, chunk).H
        print(
            f"| {name} | {res.covariance_method} | {normal_condition(H):.4g} | "
            f"{res.nfev} | {dt:.2f} | {res.success} | {len(rec)} |"
        )


def time_tsqr(n: int, reps: int = 3) -> None:
    prob = PROBLEMS["exp_decay"]
    x, y = prob.data(n, seed=0)
    model, p = jmodel(prob), len(prob.p0)
    chunk = exact_lm.derive_chunk_size(n, p, None, None)
    src, w = exact_lm._open_source(x, y, None, chunk, p, None, None)
    del x, y
    theta = np.asarray(prob.p_true)
    stats = make_chunk_stats(model, weighted=w)
    run_pass(stats, theta, src, chunk)  # warm-up (compile)
    exact_lm.tsqr_r_factor(model, theta, src, chunk, weighted=w)
    t_pass, t_qr = [], []
    for _ in range(reps):  # both return NumPy: host-synchronized
        t0 = time.perf_counter()
        run_pass(stats, theta, src, chunk)
        t_pass.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        exact_lm.tsqr_r_factor(model, theta, src, chunk, weighted=w)
        t_qr.append(time.perf_counter() - t0)
    on_device = isinstance(src[0], jax.Array)
    print(
        f"| {n:,} | {chunk:,} | {on_device} | {min(t_qr):.3f} | {np.median(t_qr):.3f} | "
        f"{min(t_pass):.3f} | {np.median(t_pass):.3f} |"
    )


if __name__ == "__main__":
    print(jax.devices())
    route_table(10_000_000)
    print("## TSQR pass vs one LM data pass, exp_decay at p_true (s)")
    print(
        "| n | chunk | data on device | tsqr min | tsqr median | pass min | pass median |"
    )
    for n in (10_000_000, 100_000_000):
        time_tsqr(n)
