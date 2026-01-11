#!/usr/bin/env python
"""CMA-ES Multi-Scale Parameter Fitting Example (v0.6.3).

This example demonstrates CMA-ES for fitting models with parameters spanning
many orders of magnitude - a scenario where traditional gradient-based
optimizers often struggle.

CMA-ES excels at multi-scale optimization because:
1. It adapts the covariance matrix to the local geometry
2. The sigmoid bound transformation normalizes all parameters
3. BIPOP restarts help escape local minima in complex landscapes

The fit(workflow='auto_global') API automatically selects CMA-ES when:
- Parameter scale ratio > 1000 AND evosax is available

Prerequisites:
    pip install "nlsq[global]"  # Installs evosax dependency

Run this example:
    python examples/scripts/07_global_optimization/06_cmaes_multiscale.py
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from nlsq import fit
from nlsq.global_optimization import MethodSelector, is_evosax_available


def diffusion_model(x, D0, gamma0, n):
    """Diffusion model: D = D0 * (1 + (x / gamma0)^n).

    Parameters:
    - D0: ~1e-10 (diffusion coefficient in m^2/s)
    - gamma0: ~1e-3 (critical shear rate in 1/s)
    - n: ~0.5 (power law exponent)

    Scale ratio: ~1e7 (7 orders of magnitude)
    """
    return D0 * (1.0 + jnp.power(x / gamma0, n))


def main():
    """Demonstrate multi-scale parameter fitting with CMA-ES."""
    print("=" * 60)
    print("CMA-ES Multi-Scale Parameter Fitting (v0.6.3)")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    x = jnp.logspace(-1, 3, 50)  # Shear rates from 0.1 to 1000 1/s

    # True parameters (span 7 orders of magnitude)
    true_D0 = 1e-10  # m^2/s
    true_gamma0 = 1e-3  # 1/s
    true_n = 0.5  # dimensionless

    y_true = diffusion_model(x, true_D0, true_gamma0, true_n)
    noise = 0.02 * y_true * np.random.randn(len(x))
    y = y_true + noise

    # Bounds spanning the expected parameter ranges
    bounds = (
        [1e-12, 1e-5, 0.1],  # Lower bounds
        [1e-8, 1e-1, 2.0],  # Upper bounds
    )

    print("\nTrue parameters:")
    print(f"  D0     = {true_D0:.2e} m^2/s")
    print(f"  gamma0 = {true_gamma0:.2e} 1/s")
    print(f"  n      = {true_n:.2f}")

    # =========================================================================
    # 1. Check scale ratio with MethodSelector
    # =========================================================================
    print("\n1. Check scale ratio with MethodSelector:")
    print("-" * 40)

    selector = MethodSelector()
    scale_ratio = selector.compute_scale_ratio(np.array(bounds[0]), np.array(bounds[1]))
    expected_method = selector.select("auto", np.array(bounds[0]), np.array(bounds[1]))

    print(f"  Scale ratio: {scale_ratio:.0f}x")
    print(f"  Threshold for CMA-ES: {selector.scale_threshold:.0f}x")
    print(f"  Expected method: {expected_method}")

    if scale_ratio > selector.scale_threshold:
        print("  -> Multi-scale problem detected, CMA-ES recommended")
    else:
        print("  -> Standard optimization sufficient")

    # =========================================================================
    # 2. Compare Multi-Start vs workflow='auto_global'
    # =========================================================================
    print("\n2. Fit with workflow='auto_global':")
    print("-" * 40)

    # Using auto_global which will select the appropriate method
    popt, pcov = fit(
        diffusion_model,
        x,
        y,
        p0=[1e-10, 1e-3, 0.5],
        bounds=bounds,
        workflow="auto_global",
        n_starts=10,
    )

    print("\n  Fitted parameters:")
    print(
        f"    D0     = {popt[0]:.2e} m^2/s (error: {abs(popt[0] - true_D0) / true_D0 * 100:.1f}%)"
    )
    print(
        f"    gamma0 = {popt[1]:.2e} 1/s (error: {abs(popt[1] - true_gamma0) / true_gamma0 * 100:.1f}%)"
    )
    print(
        f"    n      = {popt[2]:.2f} (error: {abs(popt[2] - true_n) / true_n * 100:.1f}%)"
    )

    # =========================================================================
    # 3. Direct CMA-ES (if evosax available)
    # =========================================================================
    print("\n3. Direct CMAESOptimizer (if evosax available):")
    print("-" * 40)

    if is_evosax_available():
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        config = CMAESConfig(
            max_generations=100,
            restart_strategy="bipop",
            max_restarts=5,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(diffusion_model, x, y, bounds=bounds)

        print("\n  CMA-ES fitted parameters:")
        print(
            f"    D0     = {result['popt'][0]:.2e} m^2/s (error: {abs(result['popt'][0] - true_D0) / true_D0 * 100:.1f}%)"
        )
        print(
            f"    gamma0 = {result['popt'][1]:.2e} 1/s (error: {abs(result['popt'][1] - true_gamma0) / true_gamma0 * 100:.1f}%)"
        )
        print(
            f"    n      = {result['popt'][2]:.2f} (error: {abs(result['popt'][2] - true_n) / true_n * 100:.1f}%)"
        )

        print("\n  Diagnostics:")
        diag = result["cmaes_diagnostics"]
        print(f"    Total generations: {diag['total_generations']}")
        print(f"    Total restarts: {diag['total_restarts']}")
        print(f"    Final sigma: {diag['final_sigma']:.6e}")
        print(f"    Convergence reason: {diag['convergence_reason']}")
        print(f"    Wall time: {diag['wall_time']:.3f}s")

        # Check restart history
        if diag["restart_history"]:
            print("\n  Restart history:")
            for i, restart in enumerate(diag["restart_history"]):
                print(
                    f"    Restart {i + 1}: popsize={restart['popsize']}, "
                    f"generations={restart['generations']}, "
                    f"best_fitness={restart['best_fitness']:.2e}"
                )
    else:
        print("  (Skipped - evosax not installed)")
        print("  Install with: pip install 'nlsq[global]'")

    # =========================================================================
    # 4. Scale invariance demonstration
    # =========================================================================
    print("\n4. Scale Invariance Demonstration:")
    print("-" * 40)
    print()
    print("  CMA-ES should give similar results regardless of parameter magnitudes")
    print("  because the sigmoid transformation normalizes everything.")

    # Compare with local optimization
    print("\n  Comparing with local optimization (workflow='auto'):")

    popt_local, _ = fit(
        diffusion_model,
        x,
        y,
        p0=[1e-10, 1e-3, 0.5],
        bounds=bounds,
        workflow="auto",  # Local optimization
    )

    print(
        f"    Local (auto): D0={popt_local[0]:.2e}, gamma0={popt_local[1]:.2e}, n={popt_local[2]:.2f}"
    )
    print(
        f"    Global (auto_global): D0={popt[0]:.2e}, gamma0={popt[1]:.2e}, n={popt[2]:.2f}"
    )

    # =========================================================================
    # 5. Visualization
    # =========================================================================
    print("\n5. Saving visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Fit on log-log scale
    ax1 = axes[0]
    ax1.loglog(x, y, "o", alpha=0.7, markersize=6, label="Data")
    ax1.loglog(x, y_true, "k--", linewidth=2, label="True")
    ax1.loglog(x, diffusion_model(x, *popt), "r-", linewidth=2, label="Fitted")
    ax1.set_xlabel("Shear rate (1/s)")
    ax1.set_ylabel("Diffusion coefficient (m²/s)")
    ax1.set_title("Multi-Scale Fit (log-log)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Relative residuals
    ax2 = axes[1]
    y_pred = diffusion_model(x, *popt)
    rel_residuals = (y - y_pred) / y_pred * 100
    ax2.semilogx(x, rel_residuals, "o", alpha=0.7, markersize=6)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Shear rate (1/s)")
    ax2.set_ylabel("Relative residual (%)")
    ax2.set_title("Relative Residuals")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_cmaes_multiscale.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '06_cmaes_multiscale.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("Summary - Multi-Scale Fitting (v0.6.3)")
    print("=" * 60)
    print()
    print("When to use CMA-ES:")
    print(f"  - Parameter scale ratio > {selector.scale_threshold}")
    print("  - Parameters spanning many orders of magnitude")
    print("  - Complex fitness landscapes with multiple local minima")
    print()
    print("Method selection (auto_global):")
    print("  scale_ratio = max(upper-lower) / min(upper-lower)")
    print(f"  - scale_ratio > {selector.scale_threshold} AND evosax -> CMA-ES")
    print("  - otherwise -> Multi-Start")
    print()
    print("Usage:")
    print("  # Auto method selection")
    print("  fit(model, x, y, bounds=bounds, workflow='auto_global')")
    print()
    print("  # Direct CMA-ES with configuration")
    print("  from nlsq.global_optimization import CMAESOptimizer, CMAESConfig")
    print("  config = CMAESConfig(max_generations=100, restart_strategy='bipop')")
    print("  optimizer = CMAESOptimizer(config=config)")
    print("  result = optimizer.fit(model, x, y, bounds=bounds)")


if __name__ == "__main__":
    main()
