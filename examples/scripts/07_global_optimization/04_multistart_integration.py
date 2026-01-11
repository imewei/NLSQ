"""
Multi-Start Integration with fit() Workflows (v0.6.3)

This script demonstrates how to use fit(workflow='auto_global') for
global optimization with various integration patterns.

Features demonstrated:
- Integration with fit() workflows
- Bounds handling with global optimization
- Practical workflow examples
- Large dataset handling

Run this example:
    python examples/scripts/07_global_optimization/04_multistart_integration.py
"""

import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def damped_oscillation(x, a, b, c, d):
    """Damped oscillation model.

    y = a * exp(-b * x) * cos(c * x + d)

    This model has many local minima due to the periodic cosine.
    """
    return a * jnp.exp(-b * x) * jnp.cos(c * x + d)


def exponential_model(x, a, b, c):
    """Exponential decay model."""
    return a * jnp.exp(-b * x) + c


def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2, baseline):
    """Two Gaussian peaks on a baseline."""
    peak1 = a1 * jnp.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    peak2 = a2 * jnp.exp(-((x - mu2) ** 2) / (2 * sigma2**2))
    return peak1 + peak2 + baseline


def main():
    print("=" * 70)
    print("Multi-Start Integration with fit() Workflows (v0.6.3)")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Basic global optimization with fit()
    # =========================================================================
    print("1. Basic Global Optimization with fit(workflow='auto_global'):")
    print("-" * 60)

    # Generate synthetic data
    n_samples = cap_samples(300)
    x_data = np.linspace(0, 10, n_samples)

    # True parameters
    true_a, true_b, true_c, true_d = 3.0, 0.3, 2.5, 0.5

    y_true = true_a * np.exp(-true_b * x_data) * np.cos(true_c * x_data + true_d)
    noise = 0.15 * np.random.randn(n_samples)
    y_data = y_true + noise

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}, d={true_d}")
    print(f"  Dataset: {n_samples} points")

    # Define bounds (required for auto_global)
    bounds = (
        [0.5, 0.01, 0.5, -np.pi],  # Lower bounds
        [10.0, 2.0, 5.0, np.pi],  # Upper bounds
    )

    # Poor initial guess
    p0 = [1.0, 0.1, 1.0, 0.0]

    # Local optimization with workflow='auto'
    popt_local, pcov_local = fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        workflow="auto",
    )

    print("\n  Local optimization (workflow='auto'):")
    print(
        f"    a={popt_local[0]:.4f}, b={popt_local[1]:.4f}, "
        f"c={popt_local[2]:.4f}, d={popt_local[3]:.4f}"
    )

    # Global optimization with workflow='auto_global'
    n_starts = 3 if QUICK else 10
    popt_global, pcov_global = fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        workflow="auto_global",
        n_starts=n_starts,
        sampler="lhs",
    )

    print(f"\n  Global optimization (workflow='auto_global', n_starts={n_starts}):")
    print(
        f"    a={popt_global[0]:.4f}, b={popt_global[1]:.4f}, "
        f"c={popt_global[2]:.4f}, d={popt_global[3]:.4f}"
    )

    # =========================================================================
    # 2. Bounds handling
    # =========================================================================
    print()
    print("2. Bounds Handling:")
    print("-" * 60)

    # Tight bounds
    tight_bounds = (
        [2.0, 0.1, 2.0, -0.5],
        [5.0, 0.8, 3.5, 1.5],
    )

    popt_tight, _ = fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=[3.0, 0.4, 2.5, 0.5],
        bounds=tight_bounds,
        workflow="auto_global",
        n_starts=3 if QUICK else 10,
    )

    print("  Result with tight bounds:")
    print(
        f"    a={popt_tight[0]:.4f}, b={popt_tight[1]:.4f}, "
        f"c={popt_tight[2]:.4f}, d={popt_tight[3]:.4f}"
    )

    # Verify bounds
    print("\n  Bounds verification:")
    for i, (name, val, lo, hi) in enumerate(
        zip(
            ["a", "b", "c", "d"],
            popt_tight,
            tight_bounds[0],
            tight_bounds[1],
            strict=False,
        )
    ):
        in_bounds = lo <= val <= hi
        print(
            f"    {name}: {lo:.2f} <= {val:.4f} <= {hi:.2f} : {'OK' if in_bounds else 'VIOLATION'}"
        )

    # =========================================================================
    # 3. Different sampler options
    # =========================================================================
    print()
    print("3. Sampler Options:")
    print("-" * 60)

    samplers = ["lhs", "sobol", "halton"]

    for sampler in samplers:
        popt, _ = fit(
            damped_oscillation,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            workflow="auto_global",
            n_starts=3 if QUICK else 8,
            sampler=sampler,
        )

        y_pred = damped_oscillation(x_data, *popt)
        ssr = float(jnp.sum((y_data - y_pred) ** 2))

        print(f"  {sampler}: SSR={ssr:.4f}")

    if QUICK:
        print("\n  Quick mode: skipping remaining sections.")
        return

    # =========================================================================
    # 4. Practical workflow: Peak fitting
    # =========================================================================
    print()
    print("4. Practical Workflow: Peak Fitting:")
    print("-" * 60)

    # Generate spectroscopy data
    n_spec = cap_samples(500)
    x_spec = np.linspace(0, 10, n_spec)

    true_params_spec = [3.0, 3.5, 0.5, 2.0, 5.0, 0.8, 0.5]
    y_spec_true = double_gaussian(x_spec, *true_params_spec)
    y_spec = y_spec_true + 0.1 * np.random.randn(n_spec)

    print("  True peak parameters:")
    print(
        f"    Peak 1: amplitude={true_params_spec[0]}, center={true_params_spec[1]}, width={true_params_spec[2]}"
    )
    print(
        f"    Peak 2: amplitude={true_params_spec[3]}, center={true_params_spec[4]}, width={true_params_spec[5]}"
    )

    # Define bounds
    peak_bounds = (
        [0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0],
        [10.0, 10.0, 3.0, 10.0, 10.0, 3.0, 2.0],
    )

    # Poor initial guess (peaks swapped)
    p0_spec = [1.5, 5.0, 0.8, 3.5, 3.5, 0.4, 0.3]

    # Local optimization
    popt_spec_local, _ = fit(
        double_gaussian,
        x_spec,
        y_spec,
        p0=p0_spec,
        bounds=peak_bounds,
        workflow="auto",
    )

    print("\n  Local optimization (workflow='auto'):")
    print(
        f"    Peak 1: a={popt_spec_local[0]:.3f}, mu={popt_spec_local[1]:.3f}, sigma={popt_spec_local[2]:.3f}"
    )
    print(
        f"    Peak 2: a={popt_spec_local[3]:.3f}, mu={popt_spec_local[4]:.3f}, sigma={popt_spec_local[5]:.3f}"
    )

    # Global optimization
    popt_spec_global, _ = fit(
        double_gaussian,
        x_spec,
        y_spec,
        p0=p0_spec,
        bounds=peak_bounds,
        workflow="auto_global",
        n_starts=20,
    )

    print("\n  Global optimization (workflow='auto_global', n_starts=20):")
    print(
        f"    Peak 1: a={popt_spec_global[0]:.3f}, mu={popt_spec_global[1]:.3f}, sigma={popt_spec_global[2]:.3f}"
    )
    print(
        f"    Peak 2: a={popt_spec_global[3]:.3f}, mu={popt_spec_global[4]:.3f}, sigma={popt_spec_global[5]:.3f}"
    )

    # =========================================================================
    # 5. Save visualizations
    # =========================================================================
    print()
    print("5. Saving visualizations...")

    # Peak fitting visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(x_spec, y_spec, alpha=0.3, s=10, label="Data")
    ax1.plot(x_spec, y_spec_true, "k--", linewidth=2, label="True")
    ax1.plot(
        x_spec,
        double_gaussian(x_spec, *popt_spec_local),
        "b-",
        linewidth=2,
        label="workflow='auto'",
    )
    ax1.plot(
        x_spec,
        double_gaussian(x_spec, *popt_spec_global),
        "r-",
        linewidth=2,
        label="workflow='auto_global'",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Double Gaussian Peak Fitting")
    ax1.legend()

    ax2 = axes[1]
    residuals_local = y_spec - double_gaussian(x_spec, *popt_spec_local)
    residuals_global = y_spec - double_gaussian(x_spec, *popt_spec_global)
    ax2.scatter(x_spec, residuals_local, alpha=0.5, s=10, label="workflow='auto'")
    ax2.scatter(
        x_spec, residuals_global, alpha=0.5, s=10, label="workflow='auto_global'"
    )
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Fit Residuals")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_peak_fitting.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_peak_fitting.png'}")

    # Comparison visualization
    y_pred_local = damped_oscillation(x_data, *popt_local)
    y_pred_global = damped_oscillation(x_data, *popt_global)

    ssr_local = float(jnp.sum((y_data - y_pred_local) ** 2))
    ssr_global = float(jnp.sum((y_data - y_pred_global) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(x_data, y_data, alpha=0.4, s=10, label="Data", color="gray")
    ax1.plot(x_data, y_true, "k--", linewidth=2, label="True", alpha=0.7)
    ax1.plot(
        x_data,
        y_pred_local,
        "b-",
        linewidth=2,
        label=f"workflow='auto' (SSR={ssr_local:.2f})",
    )
    ax1.plot(
        x_data,
        y_pred_global,
        "r-",
        linewidth=2,
        label=f"workflow='auto_global' (SSR={ssr_global:.2f})",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Damped Oscillation: Local vs Global")
    ax1.legend()

    ax2 = axes[1]
    params_true = np.array([true_a, true_b, true_c, true_d])
    params_local = np.array(popt_local)
    params_global = np.array(popt_global)

    x_pos = np.arange(4)
    width = 0.25

    ax2.bar(x_pos - width, params_true, width, label="True", color="green", alpha=0.7)
    ax2.bar(
        x_pos, params_local, width, label="workflow='auto'", color="blue", alpha=0.7
    )
    ax2.bar(
        x_pos + width,
        params_global,
        width,
        label="workflow='auto_global'",
        color="red",
        alpha=0.7,
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["a", "b", "c", "d"])
    ax2.set_xlabel("Parameter")
    ax2.set_ylabel("Value")
    ax2.set_title("Parameter Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_workflow_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_workflow_comparison.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary - Multi-Start Integration (v0.6.3)")
    print("=" * 70)
    print()
    print("The Three Workflows:")
    print("  workflow='auto'        : Local optimization (default)")
    print("  workflow='auto_global' : Global optimization (bounds required)")
    print("  workflow='hpc'         : auto_global + checkpointing")
    print()
    print("Global optimization patterns:")
    print("  # Basic global optimization")
    print("  fit(model, x, y, bounds=bounds, workflow='auto_global')")
    print()
    print("  # With sampler specification")
    print("  fit(..., workflow='auto_global', n_starts=10, sampler='lhs')")
    print()
    print("Recommended n_starts by problem complexity:")
    print("  - Simple (2-3 params): n_starts=5")
    print("  - Medium (4-6 params): n_starts=10")
    print("  - Complex (7+ params): n_starts=20+")


if __name__ == "__main__":
    main()
