"""Gate for the QR-then-SVD covariance kernel (Change 2 of the JAXopt-gap plan).

Modes (run each on the GPU machine, outside the sandbox)::

    python -m benchmarks.studies.bench_cov_kernel              # kernel microbenchmark
    python -m benchmarks.studies.bench_cov_kernel --crossover  # host- vs GPU-SVD sweep
    python -m benchmarks.studies.bench_cov_kernel --accuracy   # old vs new pcov

Kernel gate: new median < old median - old IQR on both shapes.
"""

import argparse
import statistics
import sys
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import svd as jax_svd

from benchmarks.compare.problems import PROBLEMS
from nlsq.core import minpack
from nlsq.core.covariance import pcov_from_jacobian


def old_body(J):
    """Pre-Task-1 kernel: SVD of J directly."""
    return jax_svd(J, full_matrices=False)[1:]


old_jit = jax.jit(old_body)
new_jit = jax.jit(minpack._covariance_svd_body)


def _time_ms(fn, *args) -> float:
    t = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return (time.perf_counter() - t) * 1e3


def _med_iqr(ms: list[float]) -> tuple[float, float]:
    q = statistics.quantiles(ms, n=4, method="inclusive")
    return statistics.median(ms), q[2] - q[0]


def kernel(rounds: int = 20) -> bool:
    ok = True
    for n, p in [(1_000_000, 3), (1_000_000, 6)]:
        J = jax.random.normal(jax.random.key(0), (n, p), dtype=jnp.float64)
        jax.block_until_ready(J)
        for _ in range(3):  # warm up both
            _time_ms(old_jit, J)
            _time_ms(new_jit, J)
        old, new = [], []
        for _ in range(rounds):  # interleaved
            old.append(_time_ms(old_jit, J))
            new.append(_time_ms(new_jit, J))
        (mo, io), (mn, i_n) = _med_iqr(old), _med_iqr(new)
        passed = mn < mo - io
        ok &= passed
        print(
            f"{jax.default_backend()} {n}x{p}: old {mo:.3f} ms (IQR {io:.3f}) | "
            f"new {mn:.3f} ms (IQR {i_n:.3f}) | new/old {mn / mo:.3f} | "
            f"{'PASS' if passed else 'FAIL'}"
        )
    print("KERNEL GATE", "PASS" if ok else "FAIL")
    return ok


def crossover(rounds: int = 15) -> None:
    cpu = jax.devices("cpu")[0]
    print(f"backend {jax.default_backend()}; host path includes device->host copy")
    print(f"{'n':>7}{'p':>3}{'n*p':>9}{'gpu_ms':>9}{'host_ms':>9}  winner")
    wins = {3: [], 6: []}
    for p in (3, 6):
        for n in (2_000, 5_000, 10_000, 20_000, 50_000):
            J = jax.random.normal(jax.random.key(1), (n, p), dtype=jnp.float64)
            jax.block_until_ready(J)

            def host(J=J):
                return minpack._covariance_svd_jit(jax.device_put(J, cpu))

            for _ in range(3):
                _time_ms(minpack._covariance_svd_jit, J)
                _time_ms(host)
            g, h = [], []
            for _ in range(rounds):
                g.append(_time_ms(minpack._covariance_svd_jit, J))
                h.append(_time_ms(host))
            mg, mh = statistics.median(g), statistics.median(h)
            wins[p].append((n * p, mh < mg))
            print(
                f"{n:>7}{p:>3}{n * p:>9}{mg:>9.3f}{mh:>9.3f}  "
                f"{'host' if mh < mg else 'gpu'}"
            )
    # largest n*p such that host wins at it and at every smaller size, in each sweep
    per_sweep = []
    for p, rows in wins.items():
        best = 0
        for size, host_wins in sorted(rows):
            if not host_wins:
                break
            best = size
        per_sweep.append(best)
    print(f"largest n*p with host winning throughout, per sweep: {per_sweep}")
    print(f"crossover (min over sweeps): {min(per_sweep)}")


def _rel(new, old) -> float:
    """Max over entries of |new-old| / sqrt(old_ii old_jj): scale-aware."""
    d = np.sqrt(np.abs(np.diag(old)))
    return float(np.max(np.abs(new - old) / np.outer(d, d)))


def _pcov(J, body, n):
    pcov, _ = pcov_from_jacobian(
        J, 1.0, n, J.shape[1], False, jax.jit(body), logger=None
    )
    return pcov


def accuracy(n: int = 1_000_000) -> bool:
    ok = True

    def check(name, model, x, params, tol):
        nonlocal ok
        jac = jax.jit(jax.jacfwd(lambda q: model(jnp, x, *q)))
        J = jac(jnp.asarray(params, dtype=jnp.float64))
        old = _pcov(J, old_body, n)
        new = _pcov(J, minpack._covariance_svd_body, n)
        s = np.linalg.svd(np.asarray(J), compute_uv=False)
        err = _rel(new, old)
        passed = err <= tol
        ok &= passed
        print(
            f"{name:<24} cond(J)~{s[0] / s[-1]:.2e}  max rel diff {err:.2e} "
            f"(tol {tol:.0e}) {'PASS' if passed else 'FAIL'}"
        )

    for name, pr in PROBLEMS.items():
        x = jnp.asarray(pr._x(n))
        check(name, pr.model, x, pr.p_true, 1e-12)
    pr = PROBLEMS["two_exp"]
    x = jnp.asarray(pr._x(n))
    # rates 1.0/1.05 give cond(J) ~ 5e5 (the plan's "around 1e6"); 1.0001 is beyond the
    # rank cutoff (~5e9), so both bodies return a truncated pseudo-inverse.
    check("two_exp k=1.0,1.05", pr.model, x, (2.0, 1.0, 1.5, 1.05), 1e-9)
    check("two_exp k=1.0,1.0001", pr.model, x, (2.0, 1.0, 1.5, 1.0001), 1e-9)
    print("ACCURACY GATE", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    warnings.simplefilter("ignore")  # rank-deficient OptimizeWarning is expected
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crossover", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    a = ap.parse_args()
    if a.crossover:
        crossover()
    elif a.accuracy:
        sys.exit(0 if accuracy() else 1)
    else:
        sys.exit(0 if kernel() else 1)
