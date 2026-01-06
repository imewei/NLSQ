"""
NLSQ Unified fit() Entry Point - Quickstart

This script demonstrates the unified fit() entry point for curve fitting.

Features demonstrated:
- Using fit() with automatic memory-based strategy selection
- Applying preset configurations (fast, standard, quality)
- Using fit() with explicit multistart configuration
- Comparing fit(), curve_fit(), and curve_fit_large()

Run this example:
    python examples/scripts/08_workflow_system/01_fit_quickstart.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import OptimizationGoal, curve_fit, curve_fit_large, fit

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("Unified fit() Entry Point - Quickstart")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Generate synthetic data
    # =========================================================================
    print("1. Generating synthetic data...")

    n_samples = 500
    x_data = np.linspace(0, 5, n_samples)

    # True parameters
    true_a, true_b, true_c = 3.0, 1.2, 0.5

    # Generate noisy observations
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    noise = 0.15 * np.random.randn(n_samples)
    y_data = y_true + noise

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")
    print(f"  Dataset size: {n_samples} points")

    # =========================================================================
    # 2. Basic fit() usage with auto strategy selection
    # =========================================================================
    print()
    print("2. Basic fit() - automatic memory-based strategy selection...")

    popt, pcov = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        workflow="auto",  # Automatic strategy selection based on memory
    )

    print(f"  Fitted: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    print(f"  True:   a={true_a:.4f}, b={true_b:.4f}, c={true_c:.4f}")

    # =========================================================================
    # 3. Using presets
    # =========================================================================
    print()
    print("3. Using presets...")

    bounds = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])

    # Preset: 'fast'
    popt_fast, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="fast",
    )
    print(
        f"  workflow='fast':     a={popt_fast[0]:.4f}, b={popt_fast[1]:.4f}, c={popt_fast[2]:.4f}"
    )

    # Preset: 'standard'
    popt_standard, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="standard",
    )
    print(
        f"  workflow='standard': a={popt_standard[0]:.4f}, b={popt_standard[1]:.4f}, c={popt_standard[2]:.4f}"
    )

    # Preset: 'quality'
    popt_quality, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="quality",
    )
    print(
        f"  workflow='quality':  a={popt_quality[0]:.4f}, b={popt_quality[1]:.4f}, c={popt_quality[2]:.4f}"
    )

    # =========================================================================
    # 4. Using fit() with explicit multistart
    # =========================================================================
    print()
    print("4. Using fit() with explicit multistart configuration...")

    popt_ms, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        multistart=True,
        n_starts=15,
        sampler="lhs",
    )
    print(
        f"  multistart result: a={popt_ms[0]:.4f}, b={popt_ms[1]:.4f}, c={popt_ms[2]:.4f}"
    )

    # =========================================================================
    # 5. Comparison with curve_fit() and curve_fit_large()
    # =========================================================================
    print()
    print("5. Comparison with other APIs...")

    # curve_fit()
    popt_cf, _ = curve_fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
    )
    print(
        f"  curve_fit():       a={popt_cf[0]:.4f}, b={popt_cf[1]:.4f}, c={popt_cf[2]:.4f}"
    )

    # curve_fit_large() - uses standard curve_fit for small datasets
    # Note: curve_fit_large auto-detects dataset size and falls back to curve_fit
    # for small datasets. Here we just call it to show the API.
    popt_cfl, _ = curve_fit_large(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
    )
    print(
        f"  curve_fit_large(): a={popt_cfl[0]:.4f}, b={popt_cfl[1]:.4f}, c={popt_cfl[2]:.4f}"
    )

    # =========================================================================
    # 6. Visualization
    # =========================================================================
    print()
    print("6. Saving visualization...")

    y_pred = exponential_decay(x_data, *popt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Data and fit
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, alpha=0.4, s=10, label="Data")
    ax1.plot(x_data, y_true, "k--", linewidth=2, label="True function")
    ax1.plot(x_data, y_pred, "r-", linewidth=2, label="fit() result")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Exponential Decay Fit")
    ax1.legend()

    # Right: Residuals
    ax2 = axes[1]
    residuals = y_data - y_pred
    ax2.scatter(x_data, residuals, alpha=0.5, s=10)
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_fit_result.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '01_fit_result.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")
    print()
    print("Results from different approaches:")
    print(f"  fit() auto:        a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    print(
        f"  workflow='fast':   a={popt_fast[0]:.4f}, b={popt_fast[1]:.4f}, c={popt_fast[2]:.4f}"
    )
    print(
        f"  workflow='quality':a={popt_quality[0]:.4f}, b={popt_quality[1]:.4f}, c={popt_quality[2]:.4f}"
    )
    print(
        f"  curve_fit():       a={popt_cf[0]:.4f}, b={popt_cf[1]:.4f}, c={popt_cf[2]:.4f}"
    )
    print()
    print("Key takeaways:")
    print("  - fit() is the unified entry point with automatic strategy selection")
    print("  - Use workflow='auto' for memory-based automatic strategy selection")
    print("  - Use workflow presets (fast, standard, quality) for quick configuration")
    print("  - Use multistart=True with n_starts/sampler for global optimization")
    print("  - Choose API based on your needs: fit() for general use,")
    print("    curve_fit() for SciPy compatibility, curve_fit_large() for big data")


if __name__ == "__main__":
    main()
