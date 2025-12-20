"""
Converted from 01_multistart_basics.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- Local minima trap problem in nonlinear optimization
- GlobalOptimizationConfig configuration
- curve_fit() with global_optimization parameter
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

from nlsq import GlobalOptimizationConfig, curve_fit
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
    print("Multi-Start Optimization Basics")
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

    # Define bounds
    bounds = ([0.5, 0.5, -np.pi, -2.0], [5.0, 3.0, np.pi, 5.0])

    # Try several different initial guesses
    initial_guesses = [
        [1.0, 0.8, 0.0, 0.5],  # Poor guess 1
        [3.0, 2.5, 2.0, 2.0],  # Poor guess 2
        [1.5, 1.2, -1.0, 0.0],  # Poor guess 3
    ]

    single_start_results = []

    for i, p0 in enumerate(initial_guesses):
        try:
            popt, pcov = curve_fit(
                multimodal_model,
                x_data,
                y_data,
                p0=p0,
                bounds=bounds,
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
    # 3. Multi-start optimization
    # =========================================================================
    print()
    print("3. Multi-start optimization...")

    # Configure multi-start optimization
    global_config = GlobalOptimizationConfig(
        n_starts=10,
        sampler="lhs",
        center_on_p0=True,
        scale_factor=1.0,
    )

    print("  GlobalOptimizationConfig:")
    print(f"    n_starts: {global_config.n_starts}")
    print(f"    sampler: {global_config.sampler}")
    print(f"    center_on_p0: {global_config.center_on_p0}")
    print(f"    scale_factor: {global_config.scale_factor}")

    # Use the first (poor) initial guess
    p0_poor = [1.0, 0.8, 0.0, 0.5]

    # Fit with multi-start optimization
    popt_multi, pcov_multi = curve_fit(
        multimodal_model,
        x_data,
        y_data,
        p0=p0_poor,
        bounds=bounds,
        multistart=True,
        n_starts=10,
        sampler="lhs",
    )

    y_pred_multi = multimodal_model(x_data, *popt_multi)
    ssr_multi = float(jnp.sum((y_data - y_pred_multi) ** 2))

    print()
    print("  Multi-start result:")
    print(
        f"    Parameters: a={popt_multi[0]:.3f}, b={popt_multi[1]:.3f}, c={popt_multi[2]:.3f}, d={popt_multi[3]:.3f}"
    )
    print(f"    SSR: {ssr_multi:.4f}")

    # Compare with single-start from same initial guess
    popt_single, _ = curve_fit(
        multimodal_model,
        x_data,
        y_data,
        p0=p0_poor,
        bounds=bounds,
    )
    y_pred_single = multimodal_model(x_data, *popt_single)
    ssr_single = float(jnp.sum((y_data - y_pred_single) ** 2))

    print()
    print("  Comparison (same initial guess):")
    print(f"    Single-start SSR: {ssr_single:.4f}")
    print(f"    Multi-start SSR:  {ssr_multi:.4f}")
    if ssr_multi < ssr_single:
        improvement = (1 - ssr_multi / ssr_single) * 100
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
    ax1.set_title("Single-Start vs Multi-Start Comparison")
    ax1.legend()

    # Right plot: Residuals comparison
    ax2 = axes[1]
    residuals_single = y_data - y_pred_single
    residuals_multi = y_data - y_pred_multi
    ax2.scatter(
        x_data, residuals_single, alpha=0.5, s=15, label="Single-start", color="blue"
    )
    ax2.scatter(
        x_data, residuals_multi, alpha=0.5, s=15, label="Multi-start", color="red"
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
        label="Single-start result",
        edgecolors="white",
        linewidths=1,
    )
    ax.scatter(
        [popt_multi[1]],
        [popt_multi[2]],
        color="red",
        marker="s",
        s=100,
        label="Multi-start result",
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
    print("Summary")
    print("=" * 70)
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}, d={true_d}")
    print()
    print("Single-start result (poor initial guess):")
    print(
        f"  Parameters: a={popt_single[0]:.3f}, b={popt_single[1]:.3f}, c={popt_single[2]:.3f}, d={popt_single[3]:.3f}"
    )
    print(f"  SSR: {ssr_single:.4f}")
    print()
    print("Multi-start result (10 starts, LHS):")
    print(
        f"  Parameters: a={popt_multi[0]:.3f}, b={popt_multi[1]:.3f}, c={popt_multi[2]:.3f}, d={popt_multi[3]:.3f}"
    )
    print(f"  SSR: {ssr_multi:.4f}")
    print()
    print("Key takeaways:")
    print("  - Local minima are common in nonlinear optimization")
    print("  - Single-start optimization is sensitive to initial guess")
    print("  - Multi-start explores multiple starting points for global optimum")
    print("  - LHS provides stratified coverage of parameter space")


if __name__ == "__main__":
    main()
