"""
Multi-Start Optimization Basics with NLSQ (v0.6.3)

This script demonstrates global optimization using the unified fit() API
with workflow='auto_global' for automatic method selection.

Features demonstrated:
- Local minima trap problem in nonlinear optimization
- fit(workflow='auto_global') for global optimization
- Comparison of single-start vs multi-start results
- Visualization of loss landscape and starting point distribution

Run this example:
    python examples/scripts/07_global_optimization/01_multistart_basics.py
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit
from nlsq.global_optimization import latin_hypercube_sample, scale_samples_to_bounds

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def multimodal_model(x, a, b, c, d):
    """Multimodal model: y = a * sin(b * x + c) + d

    This model has multiple local minima due to the periodicity of sin().
    Different combinations of (b, c) can produce similar fits.
    """
    return a * jnp.sin(b * x + c) + d


def main():
    print("=" * 70)
    print("Multi-Start Optimization Basics (v0.6.3)")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Generate synthetic data
    # =========================================================================
    print("1. Generating synthetic data...")

    n_samples = 200
    x_data = np.linspace(0, 4 * np.pi, n_samples)

    # True parameters
    true_a, true_b, true_c, true_d = 2.0, 1.5, 0.5, 1.0

    # Generate noisy observations
    y_true = true_a * np.sin(true_b * x_data + true_c) + true_d
    noise = 0.2 * np.random.randn(n_samples)
    y_data = y_true + noise

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}, d={true_d}")
    print(f"  Dataset: {n_samples} points")
    print()

    # Visualize data
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x_data, y_data, alpha=0.5, s=10, label="Noisy data")
    ax.plot(x_data, y_true, "r-", linewidth=2, label="True function")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Synthetic Data: Multimodal Sinusoidal Model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_data_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '01_data_visualization.png'}")

    # =========================================================================
    # 2. Single-start optimization with different initial guesses
    # =========================================================================
    print()
    print("2. Single-start optimization (showing sensitivity to initial guess)...")

    # Define bounds (required for auto_global)
    bounds = ([0.5, 0.5, -np.pi, -2.0], [5.0, 3.0, np.pi, 5.0])

    # Try several different initial guesses with workflow='auto'
    initial_guesses = [
        [1.0, 0.8, 0.0, 0.5],  # Poor guess 1
        [3.0, 2.5, 2.0, 2.0],  # Poor guess 2
        [1.5, 1.2, -1.0, 0.0],  # Poor guess 3
    ]

    single_start_results = []

    for i, p0 in enumerate(initial_guesses):
        try:
            # workflow='auto' uses local optimization (single-start)
            popt, pcov = fit(
                multimodal_model,
                x_data,
                y_data,
                p0=p0,
                bounds=bounds,
                workflow="auto",  # Local optimization
            )
            y_pred = multimodal_model(x_data, *popt)
            ssr = float(jnp.sum((y_data - y_pred) ** 2))
            single_start_results.append({"p0": p0, "popt": popt, "ssr": ssr})
            print(f"  Guess {i + 1}: p0={p0}")
            print(
                f"    Result: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}"
            )
            print(f"    SSR: {ssr:.4f}")
        except Exception as e:
            print(f"  Guess {i + 1}: Failed - {e}")
            single_start_results.append({"p0": p0, "popt": None, "ssr": float("inf")})

    # =========================================================================
    # 3. Global optimization with fit(workflow='auto_global')
    # =========================================================================
    print()
    print("3. Global optimization with fit(workflow='auto_global')...")

    # Use the first (poor) initial guess
    p0_poor = [1.0, 0.8, 0.0, 0.5]

    print("  workflow='auto_global' automatically selects:")
    print("    - Multi-Start (default) or CMA-ES based on parameter scale ratio")
    print("    - Memory strategy based on dataset size")
    print()

    # Fit with global optimization - auto-selects Multi-Start or CMA-ES
    popt_global, pcov_global = fit(
        multimodal_model,
        x_data,
        y_data,
        p0=p0_poor,
        bounds=bounds,
        workflow="auto_global",  # Global optimization with auto method selection
        n_starts=10,  # Number of multi-start runs
    )

    y_pred_global = multimodal_model(x_data, *popt_global)
    ssr_global = float(jnp.sum((y_data - y_pred_global) ** 2))

    print("  Global optimization result:")
    print(
        f"    Parameters: a={popt_global[0]:.3f}, b={popt_global[1]:.3f}, "
        f"c={popt_global[2]:.3f}, d={popt_global[3]:.3f}"
    )
    print(f"    SSR: {ssr_global:.4f}")

    # Compare with single-start from same initial guess
    popt_single, _ = fit(
        multimodal_model,
        x_data,
        y_data,
        p0=p0_poor,
        bounds=bounds,
        workflow="auto",  # Local optimization
    )
    y_pred_single = multimodal_model(x_data, *popt_single)
    ssr_single = float(jnp.sum((y_data - y_pred_single) ** 2))

    print()
    print("  Comparison (same initial guess):")
    print(f"    Single-start SSR: {ssr_single:.4f}")
    print(f"    Global (auto_global) SSR: {ssr_global:.4f}")
    if ssr_global < ssr_single:
        improvement = (1 - ssr_global / ssr_single) * 100
        print(f"    Improvement: {improvement:.1f}% lower SSR")

    # =========================================================================
    # 4. Comparison visualization
    # =========================================================================
    print()
    print("4. Saving comparison visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Data with both fits
    ax1 = axes[0]
    ax1.scatter(x_data, y_data, alpha=0.4, s=15, label="Data", color="gray")
    ax1.plot(x_data, y_true, "k--", linewidth=2, label="True function", alpha=0.7)
    ax1.plot(
        x_data,
        y_pred_single,
        "b-",
        linewidth=2,
        label=f"workflow='auto' (SSR={ssr_single:.2f})",
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
    ax1.set_title("Local vs Global Optimization Comparison")
    ax1.legend()

    # Right plot: Residuals comparison
    ax2 = axes[1]
    residuals_single = y_data - y_pred_single
    residuals_global = y_data - y_pred_global
    ax2.scatter(
        x_data, residuals_single, alpha=0.5, s=15, label="workflow='auto'", color="blue"
    )
    ax2.scatter(
        x_data,
        residuals_global,
        alpha=0.5,
        s=15,
        label="workflow='auto_global'",
        color="red",
    )
    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    ax2.set_xlabel("x")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residuals Comparison")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '01_comparison.png'}")

    # =========================================================================
    # 5. Loss landscape visualization
    # =========================================================================
    print()
    print("5. Generating loss landscape visualization...")

    b_range = np.linspace(0.5, 3.0, 50)
    c_range = np.linspace(-np.pi, np.pi, 50)
    B, C = np.meshgrid(b_range, c_range)

    loss_landscape = np.zeros_like(B)
    for i in range(len(c_range)):
        for j in range(len(b_range)):
            y_pred = true_a * np.sin(B[i, j] * x_data + C[i, j]) + true_d
            loss_landscape[i, j] = np.sum((y_data - y_pred) ** 2)

    loss_log = np.log10(loss_landscape + 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(B, C, loss_log, levels=30, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="log10(SSR + 1)")

    ax.scatter(
        [true_b],
        [true_c],
        color="white",
        marker="*",
        s=200,
        label="True parameters",
        edgecolors="black",
        linewidths=1,
    )
    ax.scatter(
        [popt_single[1]],
        [popt_single[2]],
        color="blue",
        marker="o",
        s=100,
        label="workflow='auto'",
        edgecolors="white",
        linewidths=1,
    )
    ax.scatter(
        [popt_global[1]],
        [popt_global[2]],
        color="red",
        marker="s",
        s=100,
        label="workflow='auto_global'",
        edgecolors="white",
        linewidths=1,
    )

    ax.set_xlabel("b (frequency)")
    ax.set_ylabel("c (phase)")
    ax.set_title(
        "Loss Landscape (a, d fixed at true values)\nMultiple local minima visible"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_loss_landscape.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '01_loss_landscape.png'}")

    # =========================================================================
    # 6. Starting point distribution
    # =========================================================================
    print()
    print("6. Generating starting point distribution visualization...")

    n_samples_viz = 20
    n_params = 4

    key = jax.random.PRNGKey(42)
    lhs_samples = latin_hypercube_sample(n_samples_viz, n_params, rng_key=key)

    lb = np.array([0.5, 0.5, -np.pi, -2.0])
    ub = np.array([5.0, 3.0, np.pi, 5.0])
    scaled_samples = scale_samples_to_bounds(lhs_samples, lb, ub)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LHS samples on loss landscape
    ax1 = axes[0]
    contour = ax1.contourf(B, C, loss_log, levels=30, cmap="viridis", alpha=0.7)
    ax1.scatter(
        scaled_samples[:, 1],
        scaled_samples[:, 2],
        color="yellow",
        marker="o",
        s=80,
        label="LHS starting points",
        edgecolors="black",
        linewidths=1,
    )
    ax1.scatter(
        [true_b],
        [true_c],
        color="white",
        marker="*",
        s=200,
        label="True parameters",
        edgecolors="black",
        linewidths=1,
    )
    ax1.set_xlabel("b (frequency)")
    ax1.set_ylabel("c (phase)")
    ax1.set_title("LHS Starting Points on Loss Landscape")
    ax1.legend()

    # Right: All 2D projections
    ax2 = axes[1]
    param_names = ["a", "b", "c", "d"]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    plot_idx = 0
    for i in range(n_params):
        for j in range(i + 1, n_params):
            ax2.scatter(
                scaled_samples[:, i],
                scaled_samples[:, j],
                alpha=0.6,
                s=30,
                color=colors[plot_idx],
                label=f"{param_names[i]} vs {param_names[j]}",
            )
            plot_idx += 1

    ax2.set_xlabel("Parameter value (normalized)")
    ax2.set_ylabel("Parameter value (normalized)")
    ax2.set_title("LHS Coverage: All 2D Projections")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_starting_points.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '01_starting_points.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary - The Three Workflows (v0.6.3)")
    print("=" * 70)
    print()
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}, d={true_d}")
    print()
    print("workflow='auto' (local optimization):")
    print(
        f"  Parameters: a={popt_single[0]:.3f}, b={popt_single[1]:.3f}, "
        f"c={popt_single[2]:.3f}, d={popt_single[3]:.3f}"
    )
    print(f"  SSR: {ssr_single:.4f}")
    print()
    print("workflow='auto_global' (global optimization, 10 starts):")
    print(
        f"  Parameters: a={popt_global[0]:.3f}, b={popt_global[1]:.3f}, "
        f"c={popt_global[2]:.3f}, d={popt_global[3]:.3f}"
    )
    print(f"  SSR: {ssr_global:.4f}")
    print()
    print("Key takeaways:")
    print("  - workflow='auto': Local optimization, good when you have a good guess")
    print("  - workflow='auto_global': Global optimization for multi-modal problems")
    print("  - workflow='hpc': auto_global + checkpointing for long HPC jobs")
    print()
    print("Global method auto-selection (auto_global):")
    print("  - Multi-Start: Default, explores multiple starting points")
    print("  - CMA-ES: Selected when scale_ratio > 1000 AND evosax available")


if __name__ == "__main__":
    main()
