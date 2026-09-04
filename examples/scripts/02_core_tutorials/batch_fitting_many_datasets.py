"""
Batch Fitting Many Small Datasets From Python (No YAML)

`nlsq batch` runs one YAML workflow per dataset, which does not scale to
hundreds of small datasets. Two patterns work directly from the Python API:

1. Reuse a single `CurveFit` instance across datasets. Full NLSQ machinery -
   bounds, robust loss, covariance, diagnostics - at ~26 ms per fit here.
   Hoisting the instance out of the loop is worth ~14-18x on its own, because
   a bare `nlsq.curve_fit(...)` call builds a fresh instance every time and
   re-traces for each dataset.
2. `jax.vmap` one compiled solver over all datasets at once. ~800x faster
   (0.034 ms/dataset) because every dataset is solved inside a single XLA
   kernel, at the cost of writing the solver yourself and needing equal-length
   (padded) datasets.

Pattern 2 agrees with pattern 1 to 2.6e-07 on parameters and 1.9e-09 on
standard errors, so on a smooth unbounded model the speed costs no accuracy.
Prefer pattern 1 when you need bounds, robust loss, or NLSQ's trust-region
machinery; prefer pattern 2 for hundreds of small, well-behaved fits.

Run this example:
    python examples/scripts/02_core_tutorials/batch_fitting_many_datasets.py
"""

from __future__ import annotations

import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import nlsq

# NLSQ logs a line per fit; a few hundred fits would bury the output.
logging.disable(logging.CRITICAL)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
N_SETS = 20 if QUICK else 200
N_PTS = 40 if QUICK else 100


def model(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def make_datasets(rng):
    """N_SETS datasets sharing one x grid, each with its own true parameters."""
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
    return x, y, true


# =============================================================================
# Pattern 1: reuse one CurveFit instance
# =============================================================================
def fit_loop(x, Y, p0):
    """Fit each dataset with the full NLSQ optimizer.

    The instance is what makes this worth doing: `CurveFit` caches compiled
    functions across calls, and `flength` pins the padded data length so JAX
    never re-traces for a new dataset size.
    """
    fitter = nlsq.CurveFit(flength=Y.shape[1])
    popt, perr, ok, errors = [], [], [], []

    for y in Y:
        try:
            result = fitter.curve_fit(model, x, y, p0=p0)
            popt.append(result.popt)
            # pcov diagonal -> 1-sigma standard errors
            perr.append(np.sqrt(np.diag(result.pcov)))
            ok.append(bool(result.success))
        except Exception as exc:
            popt.append(np.full(len(p0), np.nan))
            perr.append(np.full(len(p0), np.nan))
            ok.append(False)
            errors.append(str(exc))

    return np.array(popt), np.array(perr), np.array(ok), errors


# =============================================================================
# Pattern 2: one vmapped solver over every dataset
# =============================================================================
def residual(p, x, y, w):
    """Weighted residual. w=0 masks a padded point out of the fit.

    `w` is used here as a 0/1 mask. Passing w = 1/sigma also works and weights
    the fit, but note the covariance below then follows scipy's
    `absolute_sigma=False` convention: the standard errors come out rescaled by
    the reduced chi-square rather than absolute.
    """
    return (model(x, *p) - y) * w


def gauss_newton(p0, x, y, w, n_iter=30, lam=1e-3):
    """Levenberg-damped Gauss-Newton over a fixed iteration count.

    A fixed `lax.scan` length rather than a convergence-based `while_loop`:
    under vmap a while_loop runs until the *slowest* dataset converges anyway,
    so early exit buys nothing and the fixed count keeps the kernel simple.
    Check the returned gradient norm rather than trusting the iteration count.
    """

    def step(p, _):
        r = residual(p, x, y, w)
        J = jax.jacobian(residual)(p, x, y, w)
        JTJ = J.T @ J
        # The ridge floor is scaled to the problem: an absolute 1e-12 is ~1e-15
        # relative to this JTJ (trace/n is O(100)) and so does nothing at all.
        floor = 1e-12 * jnp.trace(JTJ) / p.size
        damped = JTJ + lam * jnp.diag(jnp.diag(JTJ)) + floor * jnp.eye(p.size)
        return p + jnp.linalg.solve(damped, -J.T @ r), None

    p, _ = jax.lax.scan(step, p0, None, length=n_iter)

    # Covariance at the solution: s^2 * (J^T J)^-1, with s^2 = SSR / (n - n_params)
    r = residual(p, x, y, w)
    J = jax.jacobian(residual)(p, x, y, w)
    dof = jnp.maximum(jnp.sum(w > 0) - p.size, 1)
    # pinv, not inv: a rank-deficient JTJ makes inv return silent all-NaN
    # under jit, with no exception to notice.
    pcov = jnp.linalg.pinv(J.T @ J) * (r @ r) / dof
    return p, jnp.sqrt(jnp.diag(pcov)), jnp.linalg.norm(J.T @ r)


fit_vmapped = jax.jit(jax.vmap(gauss_newton, in_axes=(0, None, 0, 0)))


def fit_batched(x, Y, p0, weights=None):
    """Fit every dataset in one compiled call."""
    if weights is None:
        weights = np.ones_like(Y)
    P0 = np.tile(p0, (Y.shape[0], 1))
    out = fit_vmapped(
        jnp.asarray(P0), jnp.asarray(x), jnp.asarray(Y), jnp.asarray(weights)
    )
    popt, perr, grad = jax.block_until_ready(out)
    return np.asarray(popt), np.asarray(perr), np.asarray(grad)


def main():
    rng = np.random.default_rng(0)
    x, Y, true = make_datasets(rng)
    p0 = np.array([5.0, 0.5, 1.0])

    print("=" * 74)
    print(f"Batch fitting {Y.shape[0]} datasets x {Y.shape[1]} points, 3 parameters")
    print("=" * 74)

    t = time.perf_counter()
    popt_loop, perr_loop, ok, errors = fit_loop(x, Y, p0)
    t_loop = time.perf_counter() - t

    # Warm up on the SAME shapes: XLA recompiles per batch shape, so warming up
    # on a slice would leave compilation inside the timed region.
    fit_batched(x, Y, p0)
    t = time.perf_counter()
    popt_vmap, perr_vmap, grad = fit_batched(x, Y, p0)
    t_vmap = time.perf_counter() - t

    n_sets = Y.shape[0]
    print(
        f"\n1. CurveFit loop   {t_loop:8.2f} s  ({t_loop / n_sets * 1e3:6.1f} ms/dataset)"
        f"  converged {ok.sum()}/{n_sets}"
        + (f"  first error: {errors[0]}" if errors else "")
    )
    print(
        f"2. vmap solver     {t_vmap:8.2f} s  ({t_vmap / n_sets * 1e3:6.3f} ms/dataset)"
        f"  max|grad| {np.abs(grad).max():.1e}"
    )
    print(f"   speedup        {t_loop / t_vmap:8.0f}x")

    print("\nAgreement (both vs truth, and vs each other):")
    for name, popt in (("CurveFit loop", popt_loop), ("vmap solver", popt_vmap)):
        print(f"  {name:<16} max|fit - truth| = {np.abs(popt - true).max():.2e}")
    print(
        f"  {'difference':<16} max|loop - vmap| = "
        f"{np.abs(popt_loop - popt_vmap).max():.2e}"
    )
    print(
        f"  {'std errors':<16} max|loop - vmap| = "
        f"{np.abs(perr_loop - perr_vmap).max():.2e}"
    )

    print("\nFirst 3 datasets (vmap solver):")
    print(f"  {'#':<4}{'a':>18}{'b':>18}{'c':>18}")
    for i in range(3):
        cells = "".join(
            f"{v:>11.4f} +-{e:<5.3f}"
            for v, e in zip(popt_vmap[i], perr_vmap[i], strict=False)
        )
        print(f"  {i:<4}{cells}")

    # -------------------------------------------------------------------------
    # Ragged datasets: pad to a common length, weight the padding to zero.
    # -------------------------------------------------------------------------
    n_pts = Y.shape[1]
    lengths = rng.integers(n_pts * 6 // 10, n_pts + 1, n_sets)
    weights = (np.arange(n_pts)[None, :] < lengths[:, None]).astype(float)
    Y_padded = np.where(weights > 0, Y, 0.0)  # padded values are never read
    popt_ragged, perr_ragged, _ = fit_batched(x, Y_padded, p0, weights)

    print(
        f"\nRagged datasets ({lengths.min()}-{lengths.max()} points,"
        " zero-weighted padding):"
    )
    print(f"  max|fit - truth|  = {np.abs(popt_ragged - true).max():.2e}")
    print(
        f"  mean std error    = {perr_ragged.mean():.4f}"
        f"  (vs {perr_vmap.mean():.4f} with all {n_pts} points)"
    )


if __name__ == "__main__":
    main()
