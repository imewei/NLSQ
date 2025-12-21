"""
Converted from 05_multistart_integration.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- Integration with curve_fit() workflows
- Bounds handling with multi-start
- Combining with curve_fit_large() for large datasets
- Practical workflow examples

Run this example:
    python examples/scripts/07_global_optimization/05_multistart_integration.py
"""

import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import GlobalOptimizationConfig, curve_fit, curve_fit_large

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))
FIT_KWARGS = {"max_nfev": 200} if QUICK else {}


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
    print("Multi-Start Integration with curve_fit() Workflows")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Basic multi-start with curve_fit()
    # =========================================================================
    print("1. Basic Multi-Start with curve_fit():")
    print("-" * 50)

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

    # Define bounds
    bounds = (
        [0.5, 0.01, 0.5, -np.pi],  # Lower bounds
        [10.0, 2.0, 5.0, np.pi],  # Upper bounds
    )

    # Poor initial guess
    p0 = [1.0, 0.1, 1.0, 0.0]

    # Single-start fit
    popt_single, pcov_single = curve_fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        **FIT_KWARGS,
    )

    print("\n  Single-start result:")
    print(
        f"    a={popt_single[0]:.4f}, b={popt_single[1]:.4f}, c={popt_single[2]:.4f}, d={popt_single[3]:.4f}"
    )

    # Multi-start fit
    popt_multi, pcov_multi = curve_fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        multistart=True,  # Enable multi-start
        n_starts=2 if QUICK else 10,  # Number of starting points
        sampler="lhs",  # Latin Hypercube Sampling
        **FIT_KWARGS,
    )

    print("\n  Multi-start result:")
    print(
        f"    a={popt_multi[0]:.4f}, b={popt_multi[1]:.4f}, c={popt_multi[2]:.4f}, d={popt_multi[3]:.4f}"
    )

    # =========================================================================
    # 2. Bounds handling with multi-start
    # =========================================================================
    print()
    print("2. Bounds Handling:")
    print("-" * 50)

    # Tight bounds
    tight_bounds = (
        [2.0, 0.1, 2.0, -0.5],
        [5.0, 0.8, 3.5, 1.5],
    )

    popt_tight, _ = curve_fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=[3.0, 0.4, 2.5, 0.5],
        bounds=tight_bounds,
        multistart=True,
        n_starts=2 if QUICK else 10,
        sampler="lhs",
        **FIT_KWARGS,
    )

    print("  Result with tight bounds:")
    print(
        f"    a={popt_tight[0]:.4f}, b={popt_tight[1]:.4f}, c={popt_tight[2]:.4f}, d={popt_tight[3]:.4f}"
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
    # 3. Unbounded optimization
    # =========================================================================
    print()
    print("3. Unbounded Optimization:")
    print("-" * 50)

    # Generate data for exponential model
    x_exp = np.linspace(0, 5, cap_samples(200))
    y_exp_true = 2.5 * np.exp(-1.3 * x_exp) + 0.5
    y_exp = y_exp_true + 0.1 * np.random.randn(len(x_exp))

    # Unbounded multi-start
    popt_unbound, _ = curve_fit(
        exponential_model,
        x_exp,
        y_exp,
        p0=[2.0, 1.0, 0.0],
        multistart=True,
        n_starts=2 if QUICK else 8,
        sampler="lhs",
        **FIT_KWARGS,
    )

    print("  Unbounded multi-start result:")
    print(
        f"    a={popt_unbound[0]:.4f}, b={popt_unbound[1]:.4f}, c={popt_unbound[2]:.4f}"
    )
    print("  True values: a=2.5, b=1.3, c=0.5")

    # =========================================================================
    # 4. Using GlobalOptimizationConfig
    # =========================================================================
    print()
    print("4. GlobalOptimizationConfig:")
    print("-" * 50)

    config = GlobalOptimizationConfig(
        n_starts=4 if QUICK else 15,
        sampler="sobol",
        center_on_p0=True,
        scale_factor=0.8,
        elimination_rounds=2,
        elimination_fraction=0.5,
    )

    print(f"  n_starts: {config.n_starts}")
    print(f"  sampler: {config.sampler}")
    print(f"  center_on_p0: {config.center_on_p0}")
    print(f"  scale_factor: {config.scale_factor}")

    popt_config, _ = curve_fit(
        damped_oscillation,
        x_data,
        y_data,
        p0=p0,
        bounds=bounds,
        multistart=True,
        n_starts=config.n_starts,
        sampler=config.sampler,
        **FIT_KWARGS,
    )

    if QUICK:
        print("⏩ Quick mode: skipping large dataset and peak fitting sections.")
        return

    print(
        f"\n  Result: a={popt_config[0]:.4f}, b={popt_config[1]:.4f}, c={popt_config[2]:.4f}, d={popt_config[3]:.4f}"
    )

    # =========================================================================
    # 5. Integration with curve_fit_large()
    # =========================================================================
    print()
    print("5. Integration with curve_fit_large():")
    print("-" * 50)

    # Generate larger dataset
    n_large = cap_samples(10000 if QUICK else 50000)
    x_large = np.linspace(0, 10, n_large)

    y_large_true = (
        true_a * np.exp(-true_b * x_large) * np.cos(true_c * x_large + true_d)
    )
    y_large = y_large_true + 0.15 * np.random.randn(n_large)

    print(f"  Large dataset: {n_large:,} points")
    print(f"  Data memory: {y_large.nbytes / 1024**2:.2f} MB")

    popt_large, pcov_large = curve_fit_large(
        damped_oscillation,
        x_large,
        y_large,
        p0=p0,
        bounds=bounds,
        multistart=True,
        n_starts=2 if QUICK else 10,
        sampler="lhs",
        memory_limit_gb=1.0,
        **FIT_KWARGS,
    )

    print("\n  curve_fit_large with multi-start:")
    print(
        f"    a={popt_large[0]:.4f}, b={popt_large[1]:.4f}, c={popt_large[2]:.4f}, d={popt_large[3]:.4f}"
    )

    # =========================================================================
    # 6. Practical workflow: Peak fitting
    # =========================================================================
    print()
    print("6. Practical Workflow: Peak Fitting:")
    print("-" * 50)

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

    # Poor initial guess
    p0_spec = [1.5, 5.0, 0.8, 3.5, 3.5, 0.4, 0.3]

    # Single-start
    popt_spec_single, _ = curve_fit(
        double_gaussian,
        x_spec,
        y_spec,
        p0=p0_spec,
        bounds=peak_bounds,
        **FIT_KWARGS,
    )

    print("\n  Single-start result:")
    print(
        f"    Peak 1: a={popt_spec_single[0]:.3f}, mu={popt_spec_single[1]:.3f}, sigma={popt_spec_single[2]:.3f}"
    )
    print(
        f"    Peak 2: a={popt_spec_single[3]:.3f}, mu={popt_spec_single[4]:.3f}, sigma={popt_spec_single[5]:.3f}"
    )

    # Multi-start
    popt_spec_multi, _ = curve_fit(
        double_gaussian,
        x_spec,
        y_spec,
        p0=p0_spec,
        bounds=peak_bounds,
        multistart=True,
        n_starts=3 if QUICK else 20,
        sampler="lhs",
        **FIT_KWARGS,
    )

    print("\n  Multi-start result:")
    print(
        f"    Peak 1: a={popt_spec_multi[0]:.3f}, mu={popt_spec_multi[1]:.3f}, sigma={popt_spec_multi[2]:.3f}"
    )
    print(
        f"    Peak 2: a={popt_spec_multi[3]:.3f}, mu={popt_spec_multi[4]:.3f}, sigma={popt_spec_multi[5]:.3f}"
    )

    # =========================================================================
    # 7. Save visualizations
    # =========================================================================
    print()
    print("7. Saving visualizations...")

    # Peak fitting visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(x_spec, y_spec, alpha=0.3, s=10, label="Data")
    ax1.plot(x_spec, y_spec_true, "k--", linewidth=2, label="True")
    ax1.plot(
        x_spec,
        double_gaussian(x_spec, *popt_spec_single),
        "b-",
        linewidth=2,
        label="Single-start",
    )
    ax1.plot(
        x_spec,
        double_gaussian(x_spec, *popt_spec_multi),
        "r-",
        linewidth=2,
        label="Multi-start",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("Intensity")
    ax1.set_title("Double Gaussian Peak Fitting")
    ax1.legend()

    ax2 = axes[1]
    residuals_single = y_spec - double_gaussian(x_spec, *popt_spec_single)
    residuals_multi = y_spec - double_gaussian(x_spec, *popt_spec_multi)
    ax2.scatter(x_spec, residuals_single, alpha=0.5, s=10, label="Single-start")
    ax2.scatter(x_spec, residuals_multi, alpha=0.5, s=10, label="Multi-start")
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Fit Residuals")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_peak_fitting.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '05_peak_fitting.png'}")

    # Comparison visualization
    y_pred_single = damped_oscillation(x_data, *popt_single)
    y_pred_multi = damped_oscillation(x_data, *popt_multi)

    ssr_single = float(jnp.sum((y_data - y_pred_single) ** 2))
    ssr_multi = float(jnp.sum((y_data - y_pred_multi) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.scatter(x_data, y_data, alpha=0.4, s=10, label="Data", color="gray")
    ax1.plot(x_data, y_true, "k--", linewidth=2, label="True", alpha=0.7)
    ax1.plot(
        x_data,
        y_pred_single,
        "b-",
        linewidth=2,
        label=f"Single-start (SSR={ssr_single:.2f})",
    )
    ax1.plot(
        x_data,
        y_pred_multi,
        "r-",
        linewidth=2,
        label=f"Multi-start (SSR={ssr_multi:.2f})",
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Damped Oscillation: Single vs Multi-Start")
    ax1.legend()

    ax2 = axes[1]
    params_true = np.array([true_a, true_b, true_c, true_d])
    params_single = np.array(popt_single)
    params_multi = np.array(popt_multi)

    x_pos = np.arange(4)
    width = 0.25

    ax2.bar(x_pos - width, params_true, width, label="True", color="green", alpha=0.7)
    ax2.bar(x_pos, params_single, width, label="Single-start", color="blue", alpha=0.7)
    ax2.bar(
        x_pos + width, params_multi, width, label="Multi-start", color="red", alpha=0.7
    )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(["a", "b", "c", "d"])
    ax2.set_xlabel("Parameter")
    ax2.set_ylabel("Value")
    ax2.set_title("Parameter Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_multistart_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '05_multistart_comparison.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Multi-start integration patterns:")
    print()
    print("1. Basic: curve_fit(..., multistart=True, n_starts=10)")
    print("2. With sampler: curve_fit(..., multistart=True, sampler='lhs')")
    print("3. Large datasets: curve_fit_large(..., multistart=True)")
    print()
    print("Recommended settings by problem complexity:")
    print("  - Simple (2-3 params): n_starts=5")
    print("  - Medium (4-6 params): n_starts=10")
    print("  - Complex (7+ params): n_starts=20+")


if __name__ == "__main__":
    main()
