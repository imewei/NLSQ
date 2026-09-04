"""
Batch Fitting With Optimistix vs a Hand-Written Solver

Companion to `batch_fitting_many_datasets.py`, which fits many small datasets
by vmapping a fixed-iteration Gauss-Newton step. Optimistix provides
production-grade JAX-native least-squares solvers, so the obvious question is
whether it is the better batch engine. On this problem it is, on both counts:

- It is faster. 500 datasets of 100 points, three parameters, on one RTX 4090
  in float64: Optimistix 11.4 us/dataset against 20.3 us for the hand solver
  at 30 iterations and 65.4 us at 100. Its per-step cost is higher (trust
  region plus a `lineax` solve), but it converges in a median of 8 steps and
  stops, where a fixed-iteration scan always pays its full budget.
- It is far more robust. From initial guesses off by 20x, the fixed-damping
  hand solver diverges on every one of the 500 datasets (parameters reaching
  1e32, finite rather than NaN), while Optimistix converges on all 500.

Note what "adaptive termination" does and does not buy under `vmap`. A batched
`while_loop` runs until the SLOWEST member converges, so the batch pays the
maximum step count, not the median. Optimistix still wins here because its
maximum (29) is well under a fixed budget chosen conservatively.

Section 3 covers the precision trap that makes all of this easy to get wrong.

Run this example:
    python examples/scripts/02_core_tutorials/batch_fitting_optimistix.py
"""

from __future__ import annotations

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

# NLSQ turns on float64 as a side effect of being imported (nlsq/config.py:183).
# This script never imports nlsq, so without this line JAX would silently run
# in float32 and every conclusion above would flip -- see section 3.
jax.config.update("jax_enable_x64", True)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
N_SETS = 20 if QUICK else 500
N_PTS = 40 if QUICK else 100
MAX_STEPS = 30 if QUICK else 100


def model(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def make_datasets(rng, p0_scale=1.0):
    """N_SETS datasets sharing one x grid, plus a shared starting guess.

    `p0_scale` deliberately spoils the initial guess so the two solvers can be
    compared on robustness rather than only on speed.
    """
    x = np.linspace(0, 10, N_PTS)
    n_pts = x.size  # not N_PTS: quick mode patches np.linspace to cap `num`
    true = np.stack(
        [
            rng.uniform(4, 8, N_SETS),
            rng.uniform(0.3, 0.8, N_SETS),
            rng.uniform(0.5, 1.5, N_SETS),
        ],
        axis=1,
    )
    y = (
        true[:, :1] * np.exp(-true[:, 1:2] * x)
        + true[:, 2:3]
        + rng.normal(0, 0.05, (N_SETS, n_pts))
    )
    p0 = np.tile(np.array([5.0, 0.5, 1.0]) * p0_scale, (N_SETS, 1))
    return jnp.asarray(x), jnp.asarray(y), jnp.asarray(p0), true


def residual(p, x, y):
    return model(x, *p) - y


def final_cost(p, x, y):
    r = residual(p, x, y)
    return 0.5 * (r @ r)


# =============================================================================
# Hand-written fixed-iteration Gauss-Newton (same solver as the sibling example)
# =============================================================================
def gauss_newton(p0, x, y, n_iter, lam=1e-3):
    def step(p, _):
        r = residual(p, x, y)
        J = jax.jacobian(residual)(p, x, y)
        JTJ = J.T @ J
        # Ridge floor scaled to the problem; an absolute 1e-12 is ~1e-15
        # relative to this JTJ and therefore does nothing.
        floor = 1e-12 * jnp.trace(JTJ) / p.size
        damped = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + floor * jnp.eye(p.size)
        return p + jnp.linalg.solve(damped, -J.T @ r), None

    p, _ = jax.lax.scan(step, p0, None, length=n_iter)
    return p, final_cost(p, x, y)


def make_gn(n_iter):
    return jax.jit(
        jax.vmap(lambda p, x, y: gauss_newton(p, x, y, n_iter), in_axes=(0, None, 0))
    )


# =============================================================================
# Optimistix Levenberg-Marquardt
# =============================================================================
def optx_residual(p, args):
    """Optimistix calls the residual as f(params, args)."""
    x, y = args
    return model(x, *p) - y


def optx_fit(p0, x, y, max_steps):
    """Solve one dataset.

    `throw=False` is required under vmap: the default raises a Python exception
    on non-convergence, which cannot happen inside traced code. Inspect
    `sol.result` instead.
    """
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(
        optx_residual, solver, p0, args=(x, y), max_steps=max_steps, throw=False
    )
    return sol.value, final_cost(sol.value, x, y), sol.stats["num_steps"]


def make_optx(max_steps):
    return jax.jit(
        jax.vmap(lambda p, x, y: optx_fit(p, x, y, max_steps), in_axes=(0, None, 0))
    )


def count_failures(popt, cost, y):
    """Flag datasets the solver did not actually fit.

    Testing for NaN is not enough. In float64 a diverging Gauss-Newton step
    overflows to enormous finite values (1e32 and beyond) rather than NaN, so a
    NaN-only check reports a clean sweep on a batch that failed completely.

    The test is absolute rather than relative to the batch. A threshold like
    "cost above ten times the batch median" looks reasonable and breaks exactly
    when it matters: from a 20x-wrong p0 every dataset here diverges, so the
    median is itself garbage and the check passes everything. Comparing the RMS
    residual against the spread of the data has no such blind spot -- a fit
    that predicts worse than a horizontal line through the data is not a fit.
    """
    popt, cost, y = np.asarray(popt), np.asarray(cost), np.asarray(y)
    rms_residual = np.sqrt(2.0 * np.abs(cost) / y.shape[1])
    non_finite = ~np.isfinite(popt).all(axis=1) | ~np.isfinite(cost)
    return non_finite | (rms_residual > y.std(axis=1))


def measure(label, fn, true, y):
    """Time a solver after compiling it on the real shapes."""
    out = jax.block_until_ready(fn())
    t = time.perf_counter()
    jax.block_until_ready(fn())
    dt = time.perf_counter() - t

    popt, cost = np.asarray(out[0]), np.asarray(out[1])
    failed = count_failures(popt, cost, y)
    good = np.abs(popt[~failed] - true[~failed])
    err = good.max() if good.size else float("nan")

    print(
        f"  {label:<34}{dt * 1e3:>8.2f} ms"
        f"{dt / len(true) * 1e6:>9.1f} us/set"
        f"   max|err|={err:>8.2e}   failed={failed.sum()}/{len(true)}"
    )
    return out


def main():
    print("=" * 78)
    print(
        f"Optimistix vs hand-written Gauss-Newton: {N_SETS} datasets x {N_PTS} points"
    )
    print(f"float64 enabled: {jax.config.jax_enable_x64}")
    print("=" * 78)

    rng = np.random.default_rng(0)
    x, Y, P0, true = make_datasets(rng)

    print("\n1. Sensible p0. Optimistix converges early; the scan cannot.")
    gn_short = make_gn(MAX_STEPS // 3)
    gn_full = make_gn(MAX_STEPS)
    ox = make_optx(MAX_STEPS)

    measure(
        f"hand GN ({MAX_STEPS // 3} fixed iters)", lambda: gn_short(P0, x, Y), true, Y
    )
    measure(f"hand GN ({MAX_STEPS} fixed iters)", lambda: gn_full(P0, x, Y), true, Y)
    out = measure("optimistix LevenbergMarquardt", lambda: ox(P0, x, Y), true, Y)

    steps = np.asarray(out[2])
    print(
        f"     optimistix steps: median={int(np.median(steps))} max={steps.max()}"
        f" of {MAX_STEPS} allowed"
    )

    # -------------------------------------------------------------------------
    # Robustness: the same problem from a 20x-wrong starting guess.
    # -------------------------------------------------------------------------
    print("\n2. Same data, initial guess off by 20x:")
    rng = np.random.default_rng(0)
    x, Y, P0_bad, true = make_datasets(rng, p0_scale=20.0)

    bad_gn = measure(
        f"hand GN ({MAX_STEPS} fixed iters)", lambda: gn_full(P0_bad, x, Y), true, Y
    )
    measure("optimistix LevenbergMarquardt", lambda: ox(P0_bad, x, Y), true, Y)

    worst = np.abs(np.asarray(bad_gn[0])).max()
    print(
        f"     the hand solver's worst parameter is {worst:.2e} --"
        " finite, so a NaN check would call it a success"
    )

    # -------------------------------------------------------------------------
    # 3. Why float64 matters here, and why float32 reverses the conclusion.
    # -------------------------------------------------------------------------
    print("\n3. Precision trap:")
    print("     Optimistix is asked for rtol=atol=1e-8. float32 eps is ~1.2e-7,")
    print("     so in float32 that tolerance is unreachable: the solver never")
    print("     converges, burns its full max_steps on every dataset, and looks")
    print("     several times slower than the fixed-iteration scan. Importing")
    print("     nlsq sets float64 globally; a script that does not import it")
    print("     must call jax.config.update('jax_enable_x64', True) itself, as")
    print("     this one does at the top.")


if __name__ == "__main__":
    main()
