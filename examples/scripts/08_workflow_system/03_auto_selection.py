"""
Automatic Strategy and Method Selection in NLSQ (v0.6.3)

This script demonstrates how NLSQ automatically selects:
1. Memory strategy (standard/chunked/streaming) based on memory budget
2. Global optimization method (CMA-ES/Multi-Start) based on parameter scales

Features demonstrated:
- MemoryBudgetSelector for automatic memory strategy selection
- MethodSelector for CMA-ES vs Multi-Start selection
- Understanding the decision tree
- Using fit(workflow="auto") and fit(workflow="auto_global")

Run this example:
    python examples/scripts/08_workflow_system/06_auto_selection.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit
from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector
from nlsq.global_optimization.method_selector import MethodSelector
from nlsq.streaming.large_dataset import MemoryEstimator

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("Automatic Strategy and Method Selection (v0.6.3)")
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
    print(f"  Threshold (75%): {available_memory * 0.75:.1f} GB")

    # =========================================================================
    # 2. Memory Strategy Selection Decision Tree
    # =========================================================================
    print()
    print("2. Memory Strategy Selection Decision Tree:")
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
    # 3. MemoryBudgetSelector Examples
    # =========================================================================
    print()
    print()
    print("3. Memory Strategy Selection by Dataset Size:")
    print("-" * 60)

    selector = MemoryBudgetSelector(safety_factor=0.75)
    test_sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]
    n_params = 5

    print(f"\n  {'Dataset Size':<15} {'Peak GB':<12} {'Strategy':<15}")
    print("  " + "-" * 42)

    for n_points in test_sizes:
        budget = MemoryBudget.compute(n_points=n_points, n_params=n_params)
        strategy, _ = selector.select(n_points=n_points, n_params=n_params)
        size_str = f"{n_points:,}"
        print(f"  {size_str:<15} {budget.peak_gb:<12.4f} {strategy:<15}")

    # =========================================================================
    # 4. Global Method Selection (CMA-ES vs Multi-Start)
    # =========================================================================
    print()
    print()
    print("4. Global Method Selection (auto_global workflow):")
    print("-" * 60)
    print()
    print("  The MethodSelector chooses between CMA-ES and Multi-Start based on:")
    print("  - Parameter scale ratio: max(upper-lower) / min(upper-lower)")
    print("  - CMA-ES is selected when scale_ratio > 1000 AND evosax is available")
    print("  - Multi-Start is selected otherwise")

    method_selector = MethodSelector()

    # Test different bound configurations
    test_bounds = [
        ("Narrow bounds", np.array([0, 0, 0]), np.array([1, 1, 1])),
        ("Medium bounds", np.array([0, 0, 0]), np.array([10, 5, 1])),
        ("Wide bounds", np.array([0.001, 0, 0]), np.array([1000, 5, 1])),
        ("Very wide", np.array([1e-6, 0, 0]), np.array([1e6, 5, 1])),
    ]

    print()
    print(f"  {'Scenario':<20} {'Scale Ratio':<15} {'Method':<15}")
    print("  " + "-" * 50)

    for label, lb, ub in test_bounds:
        scale_range = ub - lb
        scale_ratio = (
            np.max(scale_range) / np.min(scale_range)
            if np.min(scale_range) > 0
            else 1.0
        )
        method = method_selector.select("auto", lb, ub)
        print(f"  {label:<20} {scale_ratio:<15.1f} {method:<15}")

    # =========================================================================
    # 5. Using fit(workflow="auto") - Local Optimization
    # =========================================================================
    print()
    print()
    print("5. Using fit(workflow='auto') - Local Optimization:")
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
        workflow="auto",  # Memory-aware local optimization
    )

    print(f"  Fitted:  a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # =========================================================================
    # 6. Using fit(workflow="auto_global") - Global Optimization
    # =========================================================================
    print()
    print()
    print("6. Using fit(workflow='auto_global') - Global Optimization:")
    print("-" * 50)

    print("  Global optimization automatically selects CMA-ES or Multi-Start")
    print("  based on parameter scale ratio.")
    print()

    bounds = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])

    popt_global, _ = fit(
        exponential_decay,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds,
        workflow="auto_global",  # Global optimization (bounds required)
        n_starts=5,  # Number of multi-start runs
    )

    print(
        f"  Fitted:  a={popt_global[0]:.4f}, b={popt_global[1]:.4f}, c={popt_global[2]:.4f}"
    )

    # =========================================================================
    # 7. Saving Strategy Boundaries Visualization
    # =========================================================================
    print()
    print("7. Saving strategy boundaries visualization...")

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
    ax.set_title("Memory Strategy Selection Boundaries (5 parameters)")

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
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary - The Three Workflows (v0.6.3)")
    print("=" * 70)
    print()
    print("Workflows:")
    print("  workflow='auto'        : Local optimization, bounds optional")
    print("  workflow='auto_global' : Global optimization, bounds required")
    print("  workflow='hpc'         : auto_global + checkpointing")
    print()
    print("Memory strategy selection (both auto and auto_global):")
    print("  1. data_gb > threshold → STREAMING")
    print("  2. peak_gb > threshold → CHUNKED")
    print("  3. else → STANDARD")
    print()
    print("Global method selection (auto_global only):")
    print("  - scale_ratio > 1000 AND evosax available → CMA-ES")
    print("  - otherwise → Multi-Start")
    print()
    print("Key APIs:")
    print("  MemoryBudgetSelector().select(...)  - Get memory strategy")
    print("  MethodSelector().select(...)        - Get global method")
    print("  fit(workflow='auto')                - Local optimization")
    print("  fit(workflow='auto_global')         - Global optimization")


if __name__ == "__main__":
    main()
