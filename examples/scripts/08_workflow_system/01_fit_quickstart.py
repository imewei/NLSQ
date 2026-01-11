"""
NLSQ Unified fit() Entry Point - Quickstart

This script demonstrates the unified fit() entry point for curve fitting
with the new 3-workflow system (v0.6.3).

The Three Workflows:
- workflow="auto"        : Memory-aware local optimization (bounds optional)
- workflow="auto_global" : Memory-aware global optimization (bounds required)
- workflow="hpc"         : auto_global + checkpointing for HPC (bounds required)

Run this example:
    python examples/scripts/08_workflow_system/01_fit_quickstart.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit, curve_fit_large, fit

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("Unified fit() Entry Point - Quickstart (v0.6.3)")
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
    # 2. workflow='auto' - Local optimization with automatic memory strategy
    # =========================================================================
    print()
    print("2. workflow='auto' - Local optimization with automatic memory strategy...")
    print("   (Default workflow, bounds are optional)")

    popt_auto, pcov_auto = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        workflow="auto",  # Default: automatic memory-based strategy selection
    )

    print(f"  Fitted: a={popt_auto[0]:.4f}, b={popt_auto[1]:.4f}, c={popt_auto[2]:.4f}")
    print(f"  True:   a={true_a:.4f}, b={true_b:.4f}, c={true_c:.4f}")

    # =========================================================================
    # 3. workflow='auto' with bounds
    # =========================================================================
    print()
    print("3. workflow='auto' with optional bounds...")

    bounds = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])

    popt_bounded, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="auto",  # Bounds are optional for 'auto'
    )
    print(
        f"  Bounded fit: a={popt_bounded[0]:.4f}, b={popt_bounded[1]:.4f}, c={popt_bounded[2]:.4f}"
    )

    # =========================================================================
    # 4. workflow='auto_global' - Global optimization with automatic method selection
    # =========================================================================
    print()
    print("4. workflow='auto_global' - Global optimization (bounds required)...")
    print("   Automatically selects CMA-ES or Multi-Start based on parameter scales")

    popt_global, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="auto_global",  # Bounds required for global optimization
        n_starts=5,  # Number of multi-start runs (if Multi-Start is selected)
    )
    print(
        f"  Global fit: a={popt_global[0]:.4f}, b={popt_global[1]:.4f}, c={popt_global[2]:.4f}"
    )

    # =========================================================================
    # 5. Adjusting tolerances directly
    # =========================================================================
    print()
    print("5. Adjusting tolerances directly (not via presets)...")
    print("   Set gtol, ftol, xtol explicitly for precision control")

    # Looser tolerances for speed
    popt_fast, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="auto",
        gtol=1e-6,
        ftol=1e-6,
        xtol=1e-6,
    )
    print(
        f"  Fast (gtol=1e-6): a={popt_fast[0]:.4f}, b={popt_fast[1]:.4f}, c={popt_fast[2]:.4f}"
    )

    # Tighter tolerances for precision
    popt_precise, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="auto",
        gtol=1e-10,
        ftol=1e-10,
        xtol=1e-10,
    )
    print(
        f"  Precise (gtol=1e-10): a={popt_precise[0]:.4f}, b={popt_precise[1]:.4f}, c={popt_precise[2]:.4f}"
    )

    # =========================================================================
    # 6. Comparison with curve_fit() and curve_fit_large()
    # =========================================================================
    print()
    print("6. Comparison with other APIs...")

    # curve_fit() - SciPy-compatible API
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

    # curve_fit_large() - for large datasets
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
    # 7. Visualization
    # =========================================================================
    print()
    print("7. Saving visualization...")

    y_pred = exponential_decay(x_data, *popt_auto)

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
    print("Summary - The Three Workflows (v0.6.3)")
    print("=" * 70)
    print()
    print("Workflows:")
    print("  workflow='auto'        : Local optimization, bounds optional")
    print("                           Auto-selects: STANDARD / CHUNKED / STREAMING")
    print()
    print("  workflow='auto_global' : Global optimization, bounds required")
    print("                           Auto-selects: CMA-ES or Multi-Start")
    print(
        "                           Plus memory strategy: STANDARD / CHUNKED / STREAMING"
    )
    print()
    print("  workflow='hpc'         : auto_global + checkpointing for HPC")
    print("                           For long-running cluster jobs")
    print()
    print("Tolerance control (set directly, not via presets):")
    print("  gtol, ftol, xtol=1e-6  : Fast fitting, looser tolerances")
    print("  gtol, ftol, xtol=1e-10 : High precision fitting")
    print()
    print("Key takeaways:")
    print("  - fit() is the unified entry point with automatic strategy selection")
    print("  - Use workflow='auto' for standard local optimization (default)")
    print("  - Use workflow='auto_global' for global search (multi-modal problems)")
    print("  - Use workflow='hpc' for long-running HPC jobs with checkpointing")
    print("  - Set tolerances directly (gtol, ftol, xtol) for precision control")


if __name__ == "__main__":
    main()
