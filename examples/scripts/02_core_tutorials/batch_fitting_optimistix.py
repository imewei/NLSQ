"""
Batch Fitting With Optimistix vs a Hand-Written Solver

Companion to `batch_fitting_many_datasets.py`, which fits many small datasets
by vmapping a fixed-iteration Gauss-Newton step. Optimistix provides
production-grade JAX-native least-squares solvers, so the obvious question is
whether it is the better batch engine. Measured on this problem:

- With a matched step budget it is ~1.4x slower per dataset than the hand
  solver, because each step does trust-region bookkeeping and a `lineax` solve
  where the hand version does one small `jnp.linalg.solve`.
- Its adaptive termination buys nothing under `vmap`. A batched `while_loop`
  runs until the SLOWEST member converges, so a batch whose median is 13 steps
  still pays for the worst-case dataset. Early exit only helps when fitting one
  dataset at a time.
- It wins decisively on robustness. With initial guesses off by 20x, the
  undamped fixed-iteration solver returns NaN for every dataset while
  Optimistix converges for all of them - it has a trust region and rejects bad
  steps, which a fixed lambda does not.

So: use the hand solver for many well-conditioned fits with a decent p0, and
Optimistix when the starting guesses are rough or the batch runs unattended.
The last section shows the hybrid that is usually the right answer - run the
fast solver, then re-fit only the rows it failed on.

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


# =============================================================================
# Hand-written fixed-iteration Gauss-Newton (same solver as the sibling example)
# =============================================================================
def gauss_newton(p0, x, y, n_iter, lam=1e-3):
    def step(p, _):
        r = residual(p, x, y)
        J = jax.jacobian(residual)(p, x, y)
        JTJ = J.T @ J
        damped = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + 1e-12 * jnp.eye(p.size)
        return p + jnp.linalg.solve(damped, -J.T @ r), None

    p, _ = jax.lax.scan(step, p0, None, length=n_iter)
    return p


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
    return sol.value, sol.stats["num_steps"], sol.result == optx.RESULTS.successful


def make_optx(max_steps):
    return jax.jit(
        jax.vmap(lambda p, x, y: optx_fit(p, x, y, max_steps), in_axes=(0, None, 0))
    )


def measure(label, fn, true):
    """Time a solver after compiling it on the real shapes."""
    out = jax.block_until_ready(fn())
    t = time.perf_counter()
    jax.block_until_ready(fn())
    dt = time.perf_counter() - t

    popt = np.asarray(out[0] if isinstance(out, tuple) else out)
    n_bad = int(np.isnan(popt).any(axis=1).sum())
    # max error over the rows that produced numbers at all
    finite = popt[~np.isnan(popt).any(axis=1)]
    err = (
        np.abs(finite - true[~np.isnan(popt).any(axis=1)]).max()
        if len(finite)
        else np.nan
    )

    print(
        f"  {label:<38}{dt * 1e3:>8.2f} ms"
        f"{dt / len(true) * 1e6:>10.1f} us/set"
        f"   max|err|={err:>8.2e}   NaN rows={n_bad}/{len(true)}"
    )
    return out


def main():
    print("=" * 78)
    print(
        f"Optimistix vs hand-written Gauss-Newton: {N_SETS} datasets x {N_PTS} points"
    )
    print("=" * 78)

    rng = np.random.default_rng(0)
    x, Y, P0, true = make_datasets(rng)

    print(f"\n1. Matched step budget ({MAX_STEPS} steps), sensible p0:")
    gn = make_gn(MAX_STEPS)
    ox = make_optx(MAX_STEPS)
    p_gn = measure("hand GN (fixed iterations)", lambda: gn(P0, x, Y), true)
    out = measure("optimistix LevenbergMarquardt", lambda: ox(P0, x, Y), true)

    steps = np.asarray(out[1])
    print(
        f"     optimistix steps: median={int(np.median(steps))} max={steps.max()}"
        f"  -> the whole batch pays the max, not the median"
    )
    print(
        f"     max|GN - optimistix| = {np.abs(np.asarray(p_gn) - np.asarray(out[0])).max():.2e}"
    )

    # -------------------------------------------------------------------------
    # Robustness: the same problem from a 20x-wrong starting guess.
    # -------------------------------------------------------------------------
    print("\n2. Same data, initial guess off by 20x:")
    rng = np.random.default_rng(0)
    x, Y, P0_bad, true = make_datasets(rng, p0_scale=20.0)

    p_gn_bad = measure("hand GN (fixed iterations)", lambda: gn(P0_bad, x, Y), true)
    out_bad = measure("optimistix LevenbergMarquardt", lambda: ox(P0_bad, x, Y), true)

    gn_failed = np.isnan(np.asarray(p_gn_bad)).any(axis=1)
    print(
        f"     hand GN diverged on {gn_failed.sum()}/{N_SETS};"
        f" optimistix converged on {int(np.asarray(out_bad[2]).sum())}/{N_SETS}"
    )

    # -------------------------------------------------------------------------
    # The practical answer: fast solver first, Optimistix only on the failures.
    # -------------------------------------------------------------------------
    print("\n3. Hybrid - fast solver, then Optimistix on the rows it lost:")
    popt = np.asarray(p_gn_bad).copy()
    failed = np.isnan(popt).any(axis=1)

    if failed.any():
        t = time.perf_counter()
        rescued, _, ok = ox(P0_bad[failed], x, Y[failed])
        rescued = np.asarray(jax.block_until_ready(rescued))
        dt = time.perf_counter() - t
        popt[failed] = rescued
        # The timing includes an XLA compile: the rescue batch is a new shape,
        # and that recompilation is a real cost of the hybrid, not an artifact.
        print(
            f"     re-fitted {failed.sum()} rows in {dt * 1e3:.2f} ms"
            f" (incl. compile for the new batch shape)"
            f"  ({int(np.asarray(ok).sum())} converged)"
        )
    print(f"     final max|fit - truth| = {np.abs(popt - true).max():.2e}")
    print(
        f"     final NaN rows         = {int(np.isnan(popt).any(axis=1).sum())}/{N_SETS}"
    )


if __name__ == "__main__":
    main()
