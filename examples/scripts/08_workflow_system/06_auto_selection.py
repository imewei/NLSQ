"""
Automatic Strategy Selection Based on Memory Budget

This script demonstrates how NLSQ automatically selects the fitting strategy
based on memory budget analysis.

Features demonstrated:
- MemoryBudgetSelector for automatic strategy selection
- Understanding the decision tree
- Effect of memory limits on strategy choice
- Using fit(workflow="auto") for automatic selection

Run this example:
    python examples/scripts/08_workflow_system/06_auto_selection.py
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


def main():
    print("=" * 70)
    print("Automatic Strategy Selection Based on Memory Budget")
    print("=" * 70)
    print()

    # Set random seed
    np.random.seed(42)

    # =========================================================================
    # 1. System Memory Detection
    # =========================================================================
    print("1. System Memory Detection:")
    print("-" * 50)

    available_memory = MemoryEstimator.get_available_memory_gb()
    print(f"  Detected available memory: {available_memory:.1f} GB")

    # =========================================================================
    # 2. MemoryBudget Computation
    # =========================================================================
    print()
    print("2. MemoryBudget Computation:")
    print("-" * 70)

    dataset_configs = [
        (100_000, 5, "100K"),
        (1_000_000, 5, "1M"),
        (10_000_000, 5, "10M"),
        (50_000_000, 5, "50M"),
        (100_000_000, 5, "100M"),
    ]

    print(
        f"{'Dataset':<10} {'Data GB':<12} {'Jacobian GB':<15} {'Peak GB':<12} {'Fits?':<8}"
    )
    print("-" * 70)

    for n_points, n_params, label in dataset_configs:
        budget = MemoryBudget.compute(
            n_points=n_points, n_params=n_params, safety_factor=0.75
        )
        fits = "Yes" if budget.fits_in_memory else "No"
        print(
            f"{label:<10} {budget.data_gb:<12.4f} {budget.jacobian_gb:<15.4f} "
            f"{budget.peak_gb:<12.4f} {fits:<8}"
        )

    # =========================================================================
    # 3. Decision Tree
    # =========================================================================
    print()
    print("3. Strategy Selection Decision Tree:")
    print("-" * 50)
    print()
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │       Compute MemoryBudget                      │")
    print("  │  (data_gb, jacobian_gb, peak_gb, threshold_gb)  │")
    print("  └─────────────────────┬───────────────────────────┘")
    print("                        │")
    print("                        ▼")
    print("             ┌──────────────────────┐")
    print("             │ data_gb > threshold? │")
    print("             └──────────┬───────────┘")
    print("                 Yes │      │ No")
    print("                     │      │")
    print("                     ▼      ▼")
    print("           ┌─────────────┐ ┌──────────────────────┐")
    print("           │  STREAMING  │ │ peak_gb > threshold? │")
    print("           │  Strategy   │ └──────────┬───────────┘")
    print("           └─────────────┘      Yes │      │ No")
    print("                                    │      │")
    print("                                    ▼      ▼")
    print("                          ┌─────────────┐ ┌─────────────┐")
    print("                          │   CHUNKED   │ │  STANDARD   │")
    print("                          │  Strategy   │ │  Strategy   │")
    print("                          └─────────────┘ └─────────────┘")

    # =========================================================================
    # 4. MemoryBudgetSelector
    # =========================================================================
    print()
    print()
    print("4. MemoryBudgetSelector Usage:")
    print("-" * 60)

    selector = MemoryBudgetSelector(safety_factor=0.75)

    print(f"\n  Available memory: {available_memory:.1f} GB")
    print(f"  Threshold (75%): {available_memory * 0.75:.1f} GB")
    print()

    # Test different dataset sizes
    test_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    n_params = 5

    print(f"  {'Dataset Size':<15} {'Strategy':<15} {'Config Type':<20}")
    print("  " + "-" * 50)

    for n_points in test_sizes:
        strategy, config = selector.select(n_points=n_points, n_params=n_params)
        config_type = type(config).__name__ if config else "None"
        size_str = f"{n_points:,}"
        print(f"  {size_str:<15} {strategy:<15} {config_type:<20}")

    # =========================================================================
    # 5. Manual Memory Limit Override
    # =========================================================================
    print()
    print("5. Manual Memory Limit Override:")
    print("-" * 60)

    n_points = 10_000_000  # 10M points
    n_params = 5

    memory_limits = [4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

    print("\n  Dataset: 10M points × 5 parameters")
    print()
    print(f"  {'Memory Limit':<15} {'Strategy':<15}")
    print("  " + "-" * 30)

    for mem_limit in memory_limits:
        strategy, _ = selector.select(
            n_points=n_points, n_params=n_params, memory_limit_gb=mem_limit
        )
        print(f"  {mem_limit:<15.1f} {strategy:<15}")

    # =========================================================================
    # 6. Strategy Selection Boundaries
    # =========================================================================
    print()
    print("6. Saving strategy boundaries visualization...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create a grid of dataset sizes and memory limits
    dataset_sizes = np.logspace(4, 9, 100)  # 10K to 1B
    memory_limits = np.linspace(4, 128, 50)

    n_params = 5
    strategy_map = np.zeros((len(memory_limits), len(dataset_sizes)))

    for i, mem_limit in enumerate(memory_limits):
        for j, n_points in enumerate(dataset_sizes):
            strategy, _ = selector.select(
                n_points=int(n_points), n_params=n_params, memory_limit_gb=mem_limit
            )
            if strategy == "streaming":
                strategy_map[i, j] = 2
            elif strategy == "chunked":
                strategy_map[i, j] = 1
            else:
                strategy_map[i, j] = 0

    # Plot
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(
        strategy_map,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[4, 9, 4, 128],
    )

    ax.set_xlabel("Dataset Size (log10)")
    ax.set_ylabel("Memory Limit (GB)")
    ax.set_title("Strategy Selection Boundaries (5 parameters)")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Standard", "Chunked", "Streaming"])

    # Add reference lines
    ax.axhline(y=available_memory, color="white", linestyle="--", linewidth=2)
    ax.text(
        9.05,
        available_memory,
        f"Current: {available_memory:.0f} GB",
        color="white",
        va="center",
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_strategy_boundaries.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '06_strategy_boundaries.png'}")

    # =========================================================================
    # 7. Using fit(workflow="auto")
    # =========================================================================
    print()
    print("7. Using fit(workflow='auto'):")
    print("-" * 50)

    # Generate test data
    n_samples = 1000
    x_data = np.linspace(0, 5, n_samples)

    true_a, true_b, true_c = 3.0, 1.2, 0.5
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    y_data = y_true + 0.1 * np.random.randn(n_samples)

    print(f"  Dataset: {n_samples} points")
    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")

    popt, pcov = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=([0.1, 0.1, -1.0], [10.0, 5.0, 2.0]),
        workflow="auto",  # Automatic strategy selection
    )

    print(f"  Fitted:  a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
    print(
        f"  Errors:  a_err={abs(popt[0] - true_a):.4f}, b_err={abs(popt[1] - true_b):.4f}, c_err={abs(popt[2] - true_c):.4f}"
    )

    # =========================================================================
    # 8. Effect of Parameters on Strategy Selection
    # =========================================================================
    print()
    print("8. Effect of Parameter Count on Strategy Selection:")
    print("-" * 60)

    n_points = 1_000_000  # 1M points fixed
    param_counts = [2, 5, 10, 20, 50, 100]

    print("\n  Dataset: 1M points, varying parameter count")
    print()
    print(f"  {'Parameters':<15} {'Peak GB':<12} {'Strategy':<15}")
    print("  " + "-" * 42)

    for n_params in param_counts:
        budget = MemoryBudget.compute(n_points=n_points, n_params=n_params)
        strategy, _ = selector.select(n_points=n_points, n_params=n_params)
        print(f"  {n_params:<15} {budget.peak_gb:<12.3f} {strategy:<15}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Automatic strategy selection:")
    print(f"  System memory: {available_memory:.1f} GB")
    print()
    print("Decision tree (in order):")
    print("  1. data_gb > threshold → STREAMING")
    print("  2. peak_gb > threshold → CHUNKED")
    print("  3. else → STANDARD")
    print()
    print("Key APIs:")
    print("  MemoryBudget.compute(n_points, n_params) - Compute memory requirements")
    print("  MemoryBudgetSelector().select(...)      - Get optimal strategy")
    print("  fit(model, x, y, workflow='auto')       - Automatic selection in fit()")
    print()
    print("Memory factors:")
    print("  - Data size: O(n_points)")
    print("  - Jacobian: O(n_points × n_params)")
    print("  - Peak: data + 1.3×jacobian + solver overhead")


if __name__ == "__main__":
    main()
