"""
Memory-Based Strategy Selection in NLSQ

This script demonstrates how NLSQ automatically selects the optimal fitting
strategy based on memory budget analysis.

Features demonstrated:
- Understanding the three strategies: standard, chunked, streaming
- MemoryBudget for computing memory requirements
- MemoryBudgetSelector for automatic strategy selection
- Manual strategy override via config objects
- Memory usage comparison across strategies

Run this example:
    python examples/scripts/08_workflow_system/02_workflow_tiers.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector
from nlsq.streaming.large_dataset import MemoryEstimator

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def estimate_memory_usage(n_points, n_params, strategy):
    """Estimate memory usage in GB for a given strategy."""
    bytes_per_point = 8 * (3 + n_params)  # x, y, residual + jacobian

    if strategy == "standard":
        # All data in memory
        return n_points * bytes_per_point / 1e9
    elif strategy == "chunked":
        # Chunk size typically 100K-1M
        chunk_size = min(1_000_000, n_points)
        return chunk_size * bytes_per_point / 1e9
    elif strategy == "streaming":
        # Batch size typically 50K
        batch_size = 50_000
        return batch_size * bytes_per_point / 1e9
    else:
        return 0


def main():
    print("=" * 70)
    print("Memory-Based Strategy Selection")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Overview of Strategies
    # =========================================================================
    print("1. Strategy Overview")
    print("-" * 50)

    strategy_info = {
        "standard": {
            "description": "Standard curve_fit() for small datasets",
            "dataset_size": "When peak memory fits in available memory",
            "memory": "O(N) - loads all data into memory",
        },
        "chunked": {
            "description": "LargeDatasetFitter with automatic chunking",
            "dataset_size": "When data fits but Jacobian doesn't",
            "memory": "O(chunk_size) - processes data in chunks",
        },
        "streaming": {
            "description": "AdaptiveHybridStreamingOptimizer for huge datasets",
            "dataset_size": "When even data arrays exceed memory",
            "memory": "O(batch_size) - mini-batch gradient descent",
        },
    }

    for strategy, info in strategy_info.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Description: {info['description']}")
        print(f"  Use case:    {info['dataset_size']}")
        print(f"  Memory:      {info['memory']}")

    # =========================================================================
    # 2. MemoryBudget for Computing Requirements
    # =========================================================================
    print()
    print()
    print("2. MemoryBudget - Computing Memory Requirements")
    print("-" * 50)

    # Compute memory budget for various dataset sizes
    test_cases = [
        (1_000, 5, "1K"),
        (100_000, 5, "100K"),
        (1_000_000, 5, "1M"),
        (10_000_000, 5, "10M"),
    ]

    print(
        f"\n{'Dataset':<10} {'Data (GB)':<12} {'Jacobian (GB)':<15} {'Peak (GB)':<12}"
    )
    print("-" * 50)

    for n_points, n_params, label in test_cases:
        budget = MemoryBudget.compute(n_points=n_points, n_params=n_params)
        print(
            f"{label:<10} {budget.data_gb:<12.4f} {budget.jacobian_gb:<15.4f} {budget.peak_gb:<12.4f}"
        )

    # Show current system memory
    available_memory = MemoryEstimator.get_available_memory_gb()
    print(f"\nCurrent system available memory: {available_memory:.1f} GB")

    # =========================================================================
    # 3. Memory Budget Details
    # =========================================================================
    print()
    print()
    print("3. Memory Budget Details for 5M Points")
    print("-" * 50)

    budget = MemoryBudget.compute(n_points=5_000_000, n_params=5, safety_factor=0.75)

    print(f"  Available memory:  {budget.available_gb:.1f} GB")
    print(f"  Threshold (75%):   {budget.threshold_gb:.1f} GB")
    print(f"  Data arrays:       {budget.data_gb:.3f} GB")
    print(f"  Jacobian matrix:   {budget.jacobian_gb:.3f} GB")
    print(f"  Peak estimate:     {budget.peak_gb:.3f} GB")
    print(f"  Fits in memory:    {budget.fits_in_memory}")
    print(f"  Data fits:         {budget.data_fits}")

    # =========================================================================
    # 4. Automatic Strategy Selection
    # =========================================================================
    print()
    print()
    print("4. Automatic Strategy Selection")
    print("-" * 50)

    selector = MemoryBudgetSelector(safety_factor=0.75)
    print(f"  Available memory: {available_memory:.1f} GB")
    print()

    test_sizes = [1_000, 50_000, 500_000, 5_000_000, 50_000_000]
    n_params = 5

    print(f"{'Dataset Size':<15} {'Strategy':<15}")
    print("-" * 30)

    for n_points in test_sizes:
        strategy, config = selector.select(n_points=n_points, n_params=n_params)

        if n_points >= 1_000_000:
            size_str = f"{n_points / 1_000_000:.0f}M"
        elif n_points >= 1_000:
            size_str = f"{n_points / 1_000:.0f}K"
        else:
            size_str = str(n_points)

        print(f"{size_str:<15} {strategy:<15}")

    # =========================================================================
    # 5. Decision Tree Visualization
    # =========================================================================
    print()
    print()
    print("5. Saving strategy selection decision tree...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5,
        9.5,
        "Memory-Based Strategy Selection Decision Tree",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Root node
    ax.add_patch(
        plt.Rectangle(
            (3.0, 7.8), 4, 1, fill=True, facecolor="lightblue", edgecolor="black"
        )
    )
    ax.text(5, 8.3, "Compute Memory Budget", ha="center", va="center", fontsize=11)
    ax.text(
        5,
        8.0,
        "MemoryBudget.compute()",
        ha="center",
        va="center",
        fontsize=9,
        style="italic",
    )

    # Level 1: data_fits check
    ax.plot([5, 5], [7.8, 6.8], "k-", linewidth=1)
    ax.add_patch(
        plt.Rectangle(
            (2.5, 5.8), 5, 1, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(5, 6.3, "data_gb > threshold_gb?", ha="center", va="center", fontsize=11)

    # Yes branch -> STREAMING
    ax.plot([5.8, 8, 8], [5.8, 5.2, 4.5], "k-", linewidth=1)
    ax.text(7.5, 5.5, "Yes", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (6.5, 3.5), 3, 1, fill=True, facecolor="salmon", edgecolor="black"
        )
    )
    ax.text(
        8, 4.0, "STREAMING", ha="center", va="center", fontsize=12, fontweight="bold"
    )
    ax.text(8, 3.7, "Mini-batch optimizer", ha="center", va="center", fontsize=8)

    # No branch -> check peak_fits
    ax.plot([4.2, 2, 2], [5.8, 5.2, 4.5], "k-", linewidth=1)
    ax.text(2.5, 5.5, "No", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (0.5, 3.5), 3, 1, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(2, 4.0, "peak_gb > threshold_gb?", ha="center", va="center", fontsize=10)

    # Yes branch -> CHUNKED
    ax.plot([2.8, 4, 4], [3.5, 2.8, 2.0], "k-", linewidth=1)
    ax.text(3.5, 3.0, "Yes", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (2.5, 1.0), 3, 1, fill=True, facecolor="orange", edgecolor="black"
        )
    )
    ax.text(4, 1.5, "CHUNKED", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(4, 1.2, "Memory-managed chunking", ha="center", va="center", fontsize=8)

    # No branch -> STANDARD
    ax.plot([1.2, 0.5, 0.5], [3.5, 2.8, 2.0], "k-", linewidth=1)
    ax.text(0.7, 3.0, "No", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (-0.5, 1.0), 3, 1, fill=True, facecolor="lightgreen", edgecolor="black"
        )
    )
    ax.text(
        1, 1.5, "STANDARD", ha="center", va="center", fontsize=12, fontweight="bold"
    )
    ax.text(1, 1.2, "Full in-memory fit", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_strategy_decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_strategy_decision_tree.png'}")

    # =========================================================================
    # 6. Using Different Memory Limits
    # =========================================================================
    print()
    print()
    print("6. Strategy Selection with Different Memory Limits")
    print("-" * 70)

    memory_limits = [8.0, 32.0, 64.0, 128.0]  # GB
    n_points = 5_000_000  # 5M points
    n_params = 5

    print(f"Dataset: {n_points / 1e6:.0f}M points, {n_params} parameters")
    print()

    for mem_limit in memory_limits:
        selector_fixed = MemoryBudgetSelector(safety_factor=0.75)
        strategy, config = selector_fixed.select(
            n_points=n_points, n_params=n_params, memory_limit_gb=mem_limit
        )

        config_type = type(config).__name__ if config else "None"
        print(f"  Memory limit: {mem_limit:>6.0f} GB -> {strategy:12s} ({config_type})")

    # =========================================================================
    # 7. Test Fit
    # =========================================================================
    print()
    print()
    print("7. Test Fit")
    print("-" * 50)

    n_samples = 1000
    x_data = np.linspace(0, 5, n_samples)
    true_a, true_b, true_c = 3.0, 1.2, 0.5
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    y_data = y_true + 0.1 * np.random.randn(n_samples)

    print(f"  Test dataset: {n_samples} points")
    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")

    popt, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        workflow="auto",
    )
    print(f"  Fitted: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # =========================================================================
    # 8. Memory Usage Comparison
    # =========================================================================
    print()
    print()
    print("8. Saving memory usage comparison...")

    dataset_sizes = np.logspace(3, 9, 50)  # 1K to 1B points
    n_params = 5

    memory_standard = [
        estimate_memory_usage(int(n), n_params, "standard") for n in dataset_sizes
    ]
    memory_chunked = [
        estimate_memory_usage(int(n), n_params, "chunked") for n in dataset_sizes
    ]
    memory_streaming = [
        estimate_memory_usage(int(n), n_params, "streaming") for n in dataset_sizes
    ]

    # Plot memory comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.loglog(dataset_sizes, memory_standard, "b-", linewidth=2, label="standard")
    ax.loglog(dataset_sizes, memory_chunked, "orange", linewidth=2, label="chunked")
    ax.loglog(dataset_sizes, memory_streaming, "r-", linewidth=2, label="streaming")

    # Add memory threshold lines
    ax.axhline(y=16, color="gray", linestyle="--", alpha=0.5, label="16 GB limit")
    ax.axhline(y=64, color="gray", linestyle=":", alpha=0.5, label="64 GB limit")

    ax.set_xlabel("Dataset Size (points)")
    ax.set_ylabel("Peak Memory Usage (GB)")
    ax.set_title("Memory Usage by Strategy")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e3, 1e9)
    ax.set_ylim(1e-3, 1e3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_memory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_memory_comparison.png'}")

    # =========================================================================
    # 9. Defense Layers for Streaming (v0.3.6+)
    # =========================================================================
    print()
    print()
    print("9. Defense Layers for Streaming (v0.3.6+)")
    print("-" * 50)
    print()
    print(
        "The streaming strategy uses AdaptiveHybridStreamingOptimizer, which includes"
    )
    print("a 4-layer defense strategy against L-BFGS warmup divergence:")
    print()
    print("  Layer 1 (Warm Start Detection):")
    print("    - Skips warmup if initial loss < 1% of data variance")
    print("    - Prevents overshooting when starting near the optimum")
    print()
    print("  Layer 2 (Adaptive Step Size):")
    print("    - Scales step size based on fit quality (1e-6 to 0.001)")
    print()
    print("  Layer 3 (Cost-Increase Guard):")
    print("    - Aborts warmup if loss increases > 5%")
    print("    - Triggers early switch to Gauss-Newton phase")
    print()
    print("  Layer 4 (Step Clipping):")
    print("    - Limits parameter update magnitude (max norm 0.1)")
    print("    - Prevents catastrophic parameter jumps")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Strategies:")
    print("  standard:  Full in-memory computation, fastest for small datasets")
    print("  chunked:   Memory-managed chunking for large datasets")
    print("  streaming: Mini-batch optimization with defense layers")
    print()
    print("Decision tree:")
    print("  1. data_gb > threshold_gb? → streaming (data doesn't fit)")
    print("  2. peak_gb > threshold_gb? → chunked (Jacobian doesn't fit)")
    print("  3. else → standard (everything fits)")
    print()
    print(f"Current system memory: {available_memory:.1f} GB")
    print()
    print("Key classes:")
    print("  MemoryBudget.compute()   - Compute memory requirements")
    print("  MemoryBudgetSelector()   - Automatic strategy selection")


if __name__ == "__main__":
    main()
