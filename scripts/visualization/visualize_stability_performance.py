"""Visualize stability performance characteristics.

This script generates plots to understand:
1. SVD computation cost vs. Jacobian size
2. Stability overhead across different problem sizes
3. rescale_data impact on convergence

Requires: matplotlib, numpy
Run with: python scripts/visualization/visualize_stability_performance.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_svd_cost_scaling():
    """Plot theoretical SVD cost scaling with matrix size."""
    # Matrix configurations (m, n)
    configs = [
        # Vary m, fixed n
        ("Tall matrices (n=7)", [(1e3, 7), (1e4, 7), (1e5, 7), (1e6, 7), (1e7, 7)]),
        # Vary n, fixed m
        (
            "Wide matrices (m=100K)",
            [(1e5, 1), (1e5, 10), (1e5, 50), (1e5, 100), (1e5, 500)],
        ),
        # Square matrices
        ("Square matrices", [(10, 10), (100, 100), (1e3, 1e3), (1e4, 1e4)]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SVD Computational Cost Scaling: O(min(m,n)² × max(m,n))", fontsize=14)

    for idx, (title, sizes) in enumerate(configs):
        ax = axes[idx]

        m_vals = [m for m, n in sizes]
        n_vals = [n for m, n in sizes]
        elements = [m * n for m, n in sizes]
        svd_cost = [min(m, n) ** 2 * max(m, n) for m, n in sizes]

        # Normalize to seconds (assuming ~1e9 FLOPS)
        svd_time = [cost / 1e9 for cost in svd_cost]

        ax.loglog(elements, svd_time, "o-", linewidth=2, markersize=8, label="SVD cost")
        ax.axhline(y=0.5, color="r", linestyle="--", label="0.5s threshold")
        ax.axvline(x=1e7, color="g", linestyle="--", label="10M element threshold")

        ax.set_xlabel("Jacobian Elements (m × n)", fontsize=11)
        ax.set_ylabel("Estimated Time (seconds)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Add annotations for key points
        for i, (m, n) in enumerate(sizes):
            if elements[i] >= 1e6:  # Annotate large matrices
                ax.annotate(
                    f"{int(m):,}×{int(n)}",
                    (elements[i], svd_time[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    alpha=0.7,
                )

    plt.tight_layout()
    plt.savefig("stability_svd_cost_scaling.png", dpi=300, bbox_inches="tight")
    print("Saved: stability_svd_cost_scaling.png")


def plot_stability_overhead():
    """Plot stability mode overhead vs. problem size."""
    # Problem sizes
    problem_sizes = np.logspace(2, 7, 20)  # 100 to 10M points
    n_params = 7

    # Estimate overhead (based on analysis)
    def stability_overhead_old(m):
        """OLD per-iteration overhead."""
        # Per-iteration SVD: ~0.01s per iteration, assume 50 iterations
        return 50 * 0.01 * (m * n_params / 1e6)  # Scale with problem size

    def stability_overhead_new(m):
        """NEW init-only overhead."""
        # Single init check + per-iteration NaN/Inf
        init_cost = 0.01 if m * n_params < 1e7 else 0.001  # SVD or skip
        per_iter_cost = 0.001 * 50  # NaN/Inf check, 50 iterations
        return init_cost + per_iter_cost

    overhead_old = [stability_overhead_old(m) for m in problem_sizes]
    overhead_new = [stability_overhead_new(m) for m in problem_sizes]

    # Baseline optimization time (rough estimate)
    baseline_time = [0.1 + 0.001 * (m / 1e5) for m in problem_sizes]

    # Percentage overhead
    pct_old = [
        (oh / bt * 100) for oh, bt in zip(overhead_old, baseline_time, strict=False)
    ]
    pct_new = [
        (oh / bt * 100) for oh, bt in zip(overhead_new, baseline_time, strict=False)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Stability Mode Overhead: Before vs. After Fix (50 iterations)", fontsize=14
    )

    # Absolute overhead
    ax1.loglog(
        problem_sizes, overhead_old, "r-o", linewidth=2, label="OLD (per-iteration)"
    )
    ax1.loglog(problem_sizes, overhead_new, "g-s", linewidth=2, label="NEW (init-only)")
    ax1.axhline(y=0.01, color="gray", linestyle="--", alpha=0.5, label="0.01s")
    ax1.axhline(y=0.1, color="gray", linestyle="--", alpha=0.5, label="0.1s")
    ax1.set_xlabel("Problem Size (data points)", fontsize=11)
    ax1.set_ylabel("Stability Overhead (seconds)", fontsize=11)
    ax1.set_title("Absolute Overhead", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Percentage overhead
    ax2.semilogx(
        problem_sizes, pct_old, "r-o", linewidth=2, label="OLD (per-iteration)"
    )
    ax2.semilogx(problem_sizes, pct_new, "g-s", linewidth=2, label="NEW (init-only)")
    ax2.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="1% threshold")
    ax2.axhline(y=10.0, color="red", linestyle="--", alpha=0.5, label="10% threshold")
    ax2.set_xlabel("Problem Size (data points)", fontsize=11)
    ax2.set_ylabel("Stability Overhead (%)", fontsize=11)
    ax2.set_title("Percentage Overhead", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, max(*pct_old, *pct_new) * 1.2])

    plt.tight_layout()
    plt.savefig("stability_overhead_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved: stability_overhead_comparison.png")


def plot_threshold_sensitivity():
    """Plot SVD skip threshold sensitivity."""
    # Different threshold values
    thresholds = [1e6, 5e6, 1e7, 2e7, 5e7, 1e8]
    threshold_labels = ["1M", "5M", "10M", "20M", "50M", "100M"]

    # Test matrix configurations
    configs = [
        (1e5, 7, "100K×7"),
        (5e5, 7, "500K×7"),
        (1e6, 7, "1M×7"),
        (1e6, 10, "1M×10"),
        (1e6, 20, "1M×20"),
        (1e7, 5, "10M×5"),
    ]

    # For each config, determine if SVD is computed
    results = np.zeros((len(configs), len(thresholds)))
    for i, (m, n, label) in enumerate(configs):
        elements = m * n
        for j, threshold in enumerate(thresholds):
            results[i, j] = (
                1 if elements <= threshold else 0
            )  # 1 = computed, 0 = skipped

    _fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(results, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(configs)))
    ax.set_xticklabels(threshold_labels)
    ax.set_yticklabels([label for _, _, label in configs])

    ax.set_xlabel("SVD Skip Threshold (elements)", fontsize=11)
    ax.set_ylabel("Matrix Configuration", fontsize=11)
    ax.set_title("SVD Computation Decision Matrix", fontsize=13)

    # Add text annotations
    for i in range(len(configs)):
        for j in range(len(thresholds)):
            m, n, _ = configs[i]
            elements = m * n
            text = "✓" if results[i, j] == 1 else "✗"
            color = "black" if results[i, j] == 1 else "white"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Skipped", "Computed"])

    plt.tight_layout()
    plt.savefig("stability_threshold_sensitivity.png", dpi=300, bbox_inches="tight")
    print("Saved: stability_threshold_sensitivity.png")


def plot_rescale_impact():
    """Plot impact of data rescaling on convergence."""
    # Simulate different data conditioning scenarios
    conditions = [
        "Well-conditioned",
        "Moderate",
        "Ill-conditioned",
        "Very Ill-conditioned",
    ]
    condition_numbers = [1e3, 1e6, 1e10, 1e13]

    # Estimated iterations with/without rescaling
    # Well-conditioned: rescaling doesn't help much
    # Ill-conditioned: rescaling helps significantly
    iterations_no_rescale = [15, 25, 50, 100]  # More iterations without rescaling
    iterations_with_rescale = [15, 20, 22, 25]  # Fewer iterations with rescaling

    # Convergence time (assuming 0.01s per iteration base + overhead)
    time_no_rescale = [0.01 * n for n in iterations_no_rescale]
    time_with_rescale = [
        0.01 * n + 0.01 for n in iterations_with_rescale
    ]  # +rescale cost

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Impact of Data Rescaling (rescale_data parameter)", fontsize=14)

    # Iterations comparison
    x = np.arange(len(conditions))
    width = 0.35

    ax1.bar(
        x - width / 2,
        iterations_no_rescale,
        width,
        label="rescale_data=False",
        color="steelblue",
    )
    ax1.bar(
        x + width / 2,
        iterations_with_rescale,
        width,
        label="rescale_data=True",
        color="coral",
    )
    ax1.set_xlabel("Problem Conditioning", fontsize=11)
    ax1.set_ylabel("Iterations to Convergence", fontsize=11)
    ax1.set_title("Iteration Count", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=15, ha="right")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (no_rescale, with_rescale) in enumerate(
        zip(iterations_no_rescale, iterations_with_rescale, strict=False)
    ):
        ax1.text(
            i - width / 2, no_rescale + 2, str(no_rescale), ha="center", fontsize=9
        )
        ax1.text(
            i + width / 2, with_rescale + 2, str(with_rescale), ha="center", fontsize=9
        )

    # Time comparison
    ax2.bar(
        x - width / 2,
        time_no_rescale,
        width,
        label="rescale_data=False",
        color="steelblue",
    )
    ax2.bar(
        x + width / 2,
        time_with_rescale,
        width,
        label="rescale_data=True",
        color="coral",
    )
    ax2.set_xlabel("Problem Conditioning", fontsize=11)
    ax2.set_ylabel("Total Time (seconds)", fontsize=11)
    ax2.set_title("Total Convergence Time", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conditions, rotation=15, ha="right")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (no_rescale, with_rescale) in enumerate(
        zip(time_no_rescale, time_with_rescale, strict=False)
    ):
        ax2.text(
            i - width / 2,
            no_rescale + 0.02,
            f"{no_rescale:.2f}s",
            ha="center",
            fontsize=9,
        )
        ax2.text(
            i + width / 2,
            with_rescale + 0.02,
            f"{with_rescale:.2f}s",
            ha="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("stability_rescale_impact.png", dpi=300, bbox_inches="tight")
    print("Saved: stability_rescale_impact.png")


def generate_all_plots():
    """Generate all performance visualization plots."""
    print("Generating stability performance visualizations...")
    print()

    try:
        print("1. SVD Cost Scaling...")
        plot_svd_cost_scaling()
        print()

        print("2. Stability Overhead Comparison...")
        plot_stability_overhead()
        print()

        print("3. Threshold Sensitivity Analysis...")
        plot_threshold_sensitivity()
        print()

        print("4. Rescale Impact Analysis...")
        plot_rescale_impact()
        print()

        print("=" * 60)
        print("All plots generated successfully!")
        print("=" * 60)
        print()
        print("Files created:")
        print("  - stability_svd_cost_scaling.png")
        print("  - stability_overhead_comparison.png")
        print("  - stability_threshold_sensitivity.png")
        print("  - stability_rescale_impact.png")

    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    generate_all_plots()
