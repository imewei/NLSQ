#!/usr/bin/env python
"""CMA-ES Basic Usage Example (v0.6.3).

This example demonstrates CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
usage via the unified fit() API and direct CMAESOptimizer.

CMA-ES is a gradient-free evolutionary algorithm particularly effective for:
- Multi-scale parameter problems (parameters spanning many orders of magnitude)
- Complex fitness landscapes with multiple local minima
- Problems where gradient information is unreliable or unavailable

The fit(workflow='auto_global') API automatically selects CMA-ES when:
- Parameter scale ratio > 1000 AND evosax is available

Prerequisites:
    pip install "nlsq[global]"  # Installs evosax dependency

Run this example:
    python examples/scripts/07_global_optimization/05_cmaes_basic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

from nlsq import fit
from nlsq.global_optimization import MethodSelector, is_evosax_available


def exponential_model(x, a, b):
    """Exponential decay: y = a * exp(-b * x)"""
    return a * jnp.exp(-b * x)


def main():
    """Demonstrate basic CMA-ES usage."""
    print("=" * 60)
    print("CMA-ES Basic Usage Example (v0.6.3)")
    print("=" * 60)

    # Check if evosax is available
    if not is_evosax_available():
        print("\nevosax is not installed. Install with: pip install 'nlsq[global]'")
        print("CMA-ES will fall back to Multi-Start without evosax.")
        print()

    # Generate synthetic data
    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    true_params = [2.5, 0.5]
    y_true = exponential_model(x, *true_params)
    noise = 0.05 * np.random.randn(len(x))
    y = y_true + noise

    # Define parameter bounds (required for global optimization)
    bounds = ([0.1, 0.01], [10.0, 2.0])

    print(f"\nTrue parameters: a={true_params[0]}, b={true_params[1]}")

    # =========================================================================
    # 1. Using fit(workflow='auto_global')
    # =========================================================================
    print("\n1. Using fit(workflow='auto_global'):")
    print("-" * 40)

    # Check what method will be selected
    selector = MethodSelector()
    scale_ratio = selector.compute_scale_ratio(np.array(bounds[0]), np.array(bounds[1]))
    expected_method = selector.select("auto", np.array(bounds[0]), np.array(bounds[1]))

    print(f"  Parameter scale ratio: {scale_ratio:.1f}")
    print(f"  Expected method: {expected_method}")

    popt_auto, pcov_auto = fit(
        exponential_model,
        x,
        y,
        p0=[1.0, 1.0],
        bounds=bounds,
        workflow="auto_global",
        n_starts=5,
    )

    print(f"  Fitted parameters: a={popt_auto[0]:.4f}, b={popt_auto[1]:.4f}")

    # =========================================================================
    # 2. Direct CMAESOptimizer usage (if evosax available)
    # =========================================================================
    print("\n2. Direct CMAESOptimizer usage:")
    print("-" * 40)

    if is_evosax_available():
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        # Create optimizer with default config (BIPOP enabled)
        config = CMAESConfig(
            max_generations=100,
            seed=42,
        )
        optimizer = CMAESOptimizer(config=config)

        # Run optimization
        result = optimizer.fit(exponential_model, x, y, bounds=bounds)

        print(
            f"  Fitted parameters: a={result['popt'][0]:.4f}, b={result['popt'][1]:.4f}"
        )
        print(
            f"  Diagnostics: {result['cmaes_diagnostics']['total_generations']} generations"
        )
    else:
        print("  (Skipped - evosax not installed)")

    # =========================================================================
    # 3. CMA-ES presets (if evosax available)
    # =========================================================================
    print("\n3. CMA-ES Presets:")
    print("-" * 40)

    if is_evosax_available():
        from nlsq.global_optimization import CMAESOptimizer

        # Fast preset (no restarts, fewer generations)
        print("\n  'cmaes-fast' preset:")
        optimizer_fast = CMAESOptimizer.from_preset("cmaes-fast")
        result_fast = optimizer_fast.fit(exponential_model, x, y, bounds=bounds)
        print(
            f"    Fitted: a={result_fast['popt'][0]:.4f}, b={result_fast['popt'][1]:.4f}"
        )
        print(
            f"    Generations: {result_fast['cmaes_diagnostics']['total_generations']}"
        )

        # Global preset (more generations, larger population)
        print("\n  'cmaes-global' preset:")
        optimizer_global = CMAESOptimizer.from_preset("cmaes-global")
        result_global = optimizer_global.fit(exponential_model, x, y, bounds=bounds)
        print(
            f"    Fitted: a={result_global['popt'][0]:.4f}, b={result_global['popt'][1]:.4f}"
        )
        print(
            f"    Generations: {result_global['cmaes_diagnostics']['total_generations']}"
        )
    else:
        print("  (Skipped - evosax not installed)")
        print()
        print("  Available presets when evosax is installed:")
        print("    - 'cmaes-fast':   Quick exploration (no restarts)")
        print("    - 'cmaes':        Default (BIPOP restarts)")
        print("    - 'cmaes-global': Thorough search (more restarts)")

    # =========================================================================
    # 4. Examining diagnostics (if evosax available)
    # =========================================================================
    print("\n4. CMA-ES Diagnostics:")
    print("-" * 40)

    if is_evosax_available():
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        config = CMAESConfig(
            max_generations=50,
            restart_strategy="bipop",
            max_restarts=3,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)
        result = optimizer.fit(exponential_model, x, y, bounds=bounds)

        diag = result["cmaes_diagnostics"]
        print(f"  Total generations: {diag['total_generations']}")
        print(f"  Total restarts: {diag['total_restarts']}")
        print(f"  Final sigma: {diag['final_sigma']:.6e}")
        print(f"  Best fitness (neg SSR): {diag['best_fitness']:.6e}")
        print(f"  Convergence reason: {diag['convergence_reason']}")
        print(f"  NLSQ refinement: {diag['nlsq_refinement']}")
        print(f"  Wall time: {diag['wall_time']:.3f}s")
    else:
        print("  (Skipped - evosax not installed)")

    # =========================================================================
    # 5. Visualization
    # =========================================================================
    print("\n5. Saving visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Fit comparison
    ax1 = axes[0]
    ax1.scatter(x, y, alpha=0.5, s=20, label="Data")
    ax1.plot(x, y_true, "k--", linewidth=2, label="True")
    ax1.plot(x, exponential_model(x, *popt_auto), "r-", linewidth=2, label="Fitted")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("CMA-ES Fit Result")
    ax1.legend()

    # Right: Residuals
    ax2 = axes[1]
    y_pred = exponential_model(x, *popt_auto)
    residuals = y - y_pred
    ax2.scatter(x, residuals, alpha=0.5, s=20)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Fit Residuals")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_cmaes_basic.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '05_cmaes_basic.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 60)
    print("Summary - CMA-ES Usage (v0.6.3)")
    print("=" * 60)
    print()
    print("Method Selection:")
    print(f"  scale_ratio > {selector.scale_threshold} AND evosax -> CMA-ES")
    print("  otherwise -> Multi-Start")
    print()
    print("Usage:")
    print("  # Via fit() with auto method selection")
    print("  fit(model, x, y, bounds=bounds, workflow='auto_global')")
    print()
    print("  # Direct CMAESOptimizer")
    print("  from nlsq.global_optimization import CMAESOptimizer")
    print("  optimizer = CMAESOptimizer.from_preset('cmaes')")
    print("  result = optimizer.fit(model, x, y, bounds=bounds)")
    print()
    print("CMA-ES Presets:")
    print("  - 'cmaes-fast':   Quick exploration, no restarts")
    print("  - 'cmaes':        Default, BIPOP restarts")
    print("  - 'cmaes-global': Thorough search, more restarts")


if __name__ == "__main__":
    main()
