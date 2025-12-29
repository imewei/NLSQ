"""
Converted from 06_auto_selection.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- WorkflowSelector decision algorithm
- auto_select_workflow() convenience function
- DatasetSizeTier and MemoryTier classification
- Memory detection with get_total_available_memory_gb()
- Adaptive tolerance calculation

Run this example:
    python examples/scripts/08_workflow_system/06_auto_selection.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nlsq.core.workflow import (
    DatasetSizeTier,
    MemoryTier,
    OptimizationGoal,
    WorkflowSelector,
    WorkflowTier,
    auto_select_workflow,
    calculate_adaptive_tolerances,
)
from nlsq.streaming.large_dataset import (
    GPUMemoryEstimator,
    MemoryEstimator,
    get_memory_tier,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Automatic Workflow Selection Deep Dive")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Display the selection matrix
    # =========================================================================
    print("1. WorkflowSelector Decision Matrix")
    print("-" * 80)
    print()
    print(
        "Dataset Size    | Low (<16GB) | Medium (16-64GB) | High (64-128GB) | Very High (>128GB)"
    )
    print("-" * 80)
    print(
        "Small (<10K)    | standard    | standard         | standard        | standard+quality"
    )
    print(
        "Medium (10K-1M) | chunked     | standard         | standard+ms     | standard+ms"
    )
    print(
        "Large (1M-10M)  | streaming   | chunked          | chunked+ms      | chunked+ms"
    )
    print(
        "Huge (10M-100M) | stream+ckpt | streaming        | chunked         | chunked+ms"
    )
    print(
        "Massive (>100M) | stream+ckpt | streaming+ckpt   | streaming       | streaming+ms"
    )
    print()
    print("Legend:")
    print("  ms = multi-start enabled")
    print("  ckpt = checkpointing enabled")
    print("  streaming/stream+ckpt = 4-layer defense enabled (v0.3.6+)")

    # =========================================================================
    # 2. Dataset Size Tiers
    # =========================================================================
    print()
    print()
    print("2. DatasetSizeTier Classification")
    print("-" * 60)
    print(f"{'Tier':<12s} | {'Max Points':<15s} | {'Tolerance':<12s}")
    print("-" * 60)

    for tier in DatasetSizeTier:
        max_pts = tier.max_points
        tol = tier.tolerance

        if max_pts == float("inf"):
            max_str = "unlimited"
        elif max_pts >= 1_000_000:
            max_str = f"{max_pts / 1_000_000:.0f}M"
        elif max_pts >= 1_000:
            max_str = f"{max_pts / 1_000:.0f}K"
        else:
            max_str = str(max_pts)

        print(f"{tier.name:<12s} | {max_str:<15s} | {tol:.0e}")

    # Demonstrate tier classification
    test_sizes = [500, 5_000, 50_000, 500_000, 5_000_000, 50_000_000, 500_000_000]

    print()
    print("Automatic Size Classification:")
    for n_points in test_sizes:
        tier = DatasetSizeTier.from_n_points(n_points)

        if n_points >= 1_000_000:
            size_str = f"{n_points / 1_000_000:.0f}M"
        elif n_points >= 1_000:
            size_str = f"{n_points / 1_000:.0f}K"
        else:
            size_str = str(n_points)

        print(f"  {size_str:>8s} points -> {tier.name}")

    # =========================================================================
    # 3. Memory Tiers
    # =========================================================================
    print()
    print()
    print("3. MemoryTier Classification")
    print("-" * 60)
    print(f"{'Tier':<12s} | {'Max Memory':<12s} | {'Description'}")
    print("-" * 60)

    for tier in MemoryTier:
        max_mem = tier.max_memory_gb
        desc = tier.description

        if max_mem == float("inf"):
            max_str = "unlimited"
        else:
            max_str = f"{max_mem:.0f} GB"

        print(f"{tier.name:<12s} | {max_str:<12s} | {desc}")

    # Check current system memory
    cpu_memory = MemoryEstimator.get_available_memory_gb()
    total_memory = MemoryEstimator.get_total_available_memory_gb()
    current_tier = get_memory_tier(total_memory)

    print()
    print("Current System Memory:")
    print(f"  CPU available:   {cpu_memory:.1f} GB")

    gpu_estimator = GPUMemoryEstimator()
    if gpu_estimator.has_gpu():
        gpu_memory = gpu_estimator.get_available_gpu_memory_gb()
        print(f"  GPU available:   {gpu_memory:.1f} GB")
    else:
        print("  GPU available:   N/A (CPU only)")

    print(f"  Total available: {total_memory:.1f} GB")
    print(f"  Memory tier:     {current_tier.name}")

    # =========================================================================
    # 4. WorkflowSelector in Action
    # =========================================================================
    print()
    print()
    print("4. WorkflowSelector Decisions")
    print("-" * 80)

    selector = WorkflowSelector()
    print(f"Using system memory: {total_memory:.1f} GB")
    print()

    test_scenarios = [
        (1_000, 3, None),
        (100_000, 5, None),
        (1_000_000, 5, OptimizationGoal.FAST),
        (1_000_000, 5, OptimizationGoal.QUALITY),
        (10_000_000, 5, OptimizationGoal.ROBUST),
        (100_000_000, 5, OptimizationGoal.MEMORY_EFFICIENT),
    ]

    print(f"{'n_points':<12s} | {'n_params':<8s} | {'Goal':<18s} | {'Config Type'}")
    print("-" * 80)

    for n_points, n_params, goal in test_scenarios:
        config = selector.select(n_points, n_params, goal)
        config_type = type(config).__name__

        if n_points >= 1_000_000:
            size_str = f"{n_points / 1_000_000:.0f}M"
        elif n_points >= 1_000:
            size_str = f"{n_points / 1_000:.0f}K"
        else:
            size_str = str(n_points)

        goal_str = goal.name if goal else "None (ROBUST)"

        print(f"{size_str:<12s} | {n_params:<8d} | {goal_str:<18s} | {config_type}")

    # =========================================================================
    # 5. Selection with Different Memory Limits
    # =========================================================================
    print()
    print()
    print("5. Selection with Different Memory Limits")
    print("-" * 70)

    memory_limits = [8.0, 32.0, 96.0, 256.0]  # GB
    n_points = 5_000_000  # 5M points
    n_params = 5

    for mem_limit in memory_limits:
        selector_fixed = WorkflowSelector(memory_limit_gb=mem_limit)
        config = selector_fixed.select(n_points, n_params)
        config_type = type(config).__name__
        mem_tier = MemoryTier.from_available_memory_gb(mem_limit)

        print(f"  Memory: {mem_limit:>6.0f} GB ({mem_tier.name:<10s}) -> {config_type}")

    # =========================================================================
    # 6. auto_select_workflow() Convenience Function
    # =========================================================================
    print()
    print()
    print("6. auto_select_workflow() Examples")
    print("-" * 50)

    config1 = auto_select_workflow(n_points=5_000, n_params=5)
    print(f"  5K points: {type(config1).__name__}")

    config2 = auto_select_workflow(
        n_points=5_000_000,
        n_params=5,
        goal=OptimizationGoal.QUALITY,
    )
    print(f"  5M points + QUALITY: {type(config2).__name__}")

    config3 = auto_select_workflow(
        n_points=5_000_000,
        n_params=5,
        memory_limit_gb=8.0,
    )
    print(f"  5M points + 8GB limit: {type(config3).__name__}")

    # =========================================================================
    # 7. Adaptive Tolerances
    # =========================================================================
    print()
    print()
    print("7. Adaptive Tolerance Calculation")
    print("-" * 60)

    test_configs = [
        (1_000, None),
        (1_000_000, None),
        (1_000_000, OptimizationGoal.FAST),
        (1_000_000, OptimizationGoal.QUALITY),
        (100_000_000, OptimizationGoal.ROBUST),
    ]

    print(f"{'n_points':<12s} | {'Goal':<12s} | {'gtol':<12s} | {'ftol':<12s}")
    print("-" * 60)

    for n_points, goal in test_configs:
        tolerances = calculate_adaptive_tolerances(n_points, goal)

        if n_points >= 1_000_000:
            size_str = f"{n_points / 1_000_000:.0f}M"
        else:
            size_str = f"{n_points / 1_000:.0f}K"

        goal_str = goal.name if goal else "None"

        print(
            f"{size_str:<12s} | {goal_str:<12s} | {tolerances['gtol']:.0e} | {tolerances['ftol']:.0e}"
        )

    # =========================================================================
    # 8. Defense Layer Awareness (v0.3.6+)
    # =========================================================================
    print()
    print()
    print("8. Defense Layer Awareness (v0.3.6+)")
    print("-" * 70)
    print()
    print("When WorkflowSelector chooses STREAMING or STREAMING_CHECKPOINT tiers,")
    print("the returned HybridStreamingConfig automatically includes 4-layer defense.")
    print()
    print("Defense presets for streaming workflows:")
    print("  defense_strict()     - Warm-start refinement (previous fit as p0)")
    print("  defense_relaxed()    - Exploration (rough initial guesses)")
    print("  scientific_default() - Production scientific computing (default)")
    print("  defense_disabled()   - Pre-0.3.6 behavior (no protection)")
    print()
    print("The 4-layer defense strategy protects against Adam warmup divergence:")
    print("  Layer 1: Warm Start Detection - Skip warmup if near optimal")
    print("  Layer 2: Adaptive Learning Rate - Scale LR based on fit quality")
    print("  Layer 3: Cost-Increase Guard - Abort if loss increases > 5%")
    print("  Layer 4: Step Clipping - Limit parameter update magnitude")
    print()
    print("To use defense presets with auto_select_workflow:")
    print()
    print("  from nlsq import HybridStreamingConfig")
    print()
    print("  # Get the auto-selected config")
    print("  config = auto_select_workflow(n_points=50_000_000, n_params=5)")
    print()
    print("  # If it's a streaming config, apply defense preset")
    print("  if isinstance(config, HybridStreamingConfig):")
    print("      # For warm-start refinement scenarios")
    print("      config = HybridStreamingConfig.defense_strict()")

    # =========================================================================
    # 9. Visualization
    # =========================================================================
    print()
    print()
    print("9. Saving selection algorithm visualization...")

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Title
    ax.text(
        8,
        13.5,
        "WorkflowSelector Decision Algorithm",
        ha="center",
        fontsize=18,
        fontweight="bold",
    )

    # Step 1: Input
    ax.add_patch(
        plt.Rectangle(
            (0.5, 11.5),
            3,
            1.2,
            fill=True,
            facecolor="lightblue",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.text(2, 12.1, "INPUT", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(2, 11.7, "n_points, n_params, goal", ha="center", va="center", fontsize=9)

    # Arrow down
    ax.annotate(
        "",
        xy=(2, 10.3),
        xytext=(2, 11.5),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )

    # Step 2: Get Memory
    ax.add_patch(
        plt.Rectangle(
            (0.5, 9.3), 3, 1, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(
        2,
        9.8,
        "1. Get Memory",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(2, 9.5, "get_available_memory_gb()", ha="center", va="center", fontsize=8)

    # Arrow down
    ax.annotate(
        "",
        xy=(2, 8.1),
        xytext=(2, 9.3),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )

    # Step 3: Classify Memory
    ax.add_patch(
        plt.Rectangle(
            (0.5, 7.1), 3, 1, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(
        2,
        7.6,
        "2. Classify Memory",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(2, 7.3, "MemoryTier.from_...()", ha="center", va="center", fontsize=8)

    # Arrow down
    ax.annotate(
        "",
        xy=(2, 5.9),
        xytext=(2, 7.1),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )

    # Step 4: Classify Dataset
    ax.add_patch(
        plt.Rectangle(
            (0.5, 4.9), 3, 1, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(
        2,
        5.4,
        "3. Classify Dataset",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(2, 5.1, "DatasetSizeTier.from_...()", ha="center", va="center", fontsize=8)

    # Arrow down
    ax.annotate(
        "",
        xy=(2, 3.7),
        xytext=(2, 4.9),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )

    # Step 5: Apply Matrix
    ax.add_patch(
        plt.Rectangle(
            (0.5, 2.7),
            3,
            1,
            fill=True,
            facecolor="lightgreen",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.text(
        2,
        3.2,
        "4. Decision Matrix",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(2, 2.9, "tier, multistart, ckpt", ha="center", va="center", fontsize=8)

    # Arrow down
    ax.annotate(
        "",
        xy=(2, 1.5),
        xytext=(2, 2.7),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 2},
    )

    # Step 6: Output
    ax.add_patch(
        plt.Rectangle(
            (0.5, 0.5),
            3,
            1,
            fill=True,
            facecolor="lightsalmon",
            edgecolor="black",
            linewidth=2,
        )
    )
    ax.text(2, 1, "OUTPUT", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(2, 0.7, "ConfigType", ha="center", va="center", fontsize=9)

    # Memory Tier Reference (right side)
    ax.add_patch(
        plt.Rectangle((5, 8), 4.5, 4.5, fill=True, facecolor="white", edgecolor="black")
    )
    ax.text(7.25, 12, "MemoryTier", ha="center", fontsize=11, fontweight="bold")
    ax.text(5.2, 11.3, "LOW: < 16 GB", fontsize=9)
    ax.text(5.2, 10.6, "MEDIUM: 16-64 GB", fontsize=9)
    ax.text(5.2, 9.9, "HIGH: 64-128 GB", fontsize=9)
    ax.text(5.2, 9.2, "VERY_HIGH: > 128 GB", fontsize=9)
    ax.text(
        5.2,
        8.4,
        f"Current: {current_tier.name}",
        fontsize=9,
        fontweight="bold",
        color="blue",
    )

    # Dataset Size Reference
    ax.add_patch(
        plt.Rectangle(
            (10, 8), 5.5, 4.5, fill=True, facecolor="white", edgecolor="black"
        )
    )
    ax.text(12.75, 12, "DatasetSizeTier", ha="center", fontsize=11, fontweight="bold")
    ax.text(10.2, 11.3, "TINY: < 1K (tol=1e-12)", fontsize=9)
    ax.text(10.2, 10.7, "SMALL: 1K-10K (tol=1e-10)", fontsize=9)
    ax.text(10.2, 10.1, "MEDIUM: 10K-100K (tol=1e-9)", fontsize=9)
    ax.text(10.2, 9.5, "LARGE: 100K-1M (tol=1e-8)", fontsize=9)
    ax.text(10.2, 8.9, "VERY_LARGE: 1M-10M (tol=1e-7)", fontsize=9)
    ax.text(10.2, 8.3, "HUGE/MASSIVE: >10M", fontsize=9)

    # Config Types Reference
    ax.add_patch(
        plt.Rectangle(
            (5, 0.5), 10.5, 3, fill=True, facecolor="white", edgecolor="black"
        )
    )
    ax.text(
        10.25, 3, "Output Config Types", ha="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        5.2,
        2.3,
        "GlobalOptimizationConfig: STANDARD + multistart",
        fontsize=9,
        color="green",
    )
    ax.text(5.2, 1.7, "LDMemoryConfig: STANDARD or CHUNKED", fontsize=9, color="orange")
    ax.text(
        5.2,
        1.1,
        "HybridStreamingConfig: STREAMING or STREAMING_CHECKPOINT",
        fontsize=9,
        color="red",
    )

    # Goal modifiers
    ax.add_patch(
        plt.Rectangle(
            (5, 4.5), 10.5, 3, fill=True, facecolor="lavender", edgecolor="black"
        )
    )
    ax.text(10.25, 7, "Goal Modifiers", ha="center", fontsize=11, fontweight="bold")
    ax.text(5.2, 6.3, "FAST: Disable multistart, looser tolerances", fontsize=9)
    ax.text(5.2, 5.7, "ROBUST/GLOBAL: Enable multistart (if memory allows)", fontsize=9)
    ax.text(5.2, 5.1, "QUALITY: Enable multistart + tighter tolerances", fontsize=9)
    ax.text(5.2, 4.7, "MEMORY_EFFICIENT: Force streaming/chunking", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_selection_algorithm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '06_selection_algorithm.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Key Functions:")
    print("  auto_select_workflow(n_points, n_params, goal)")
    print("  WorkflowSelector().select(n_points, n_params, goal)")
    print("  DatasetSizeTier.from_n_points(n_points)")
    print("  MemoryTier.from_available_memory_gb(memory_gb)")
    print("  MemoryEstimator.get_total_available_memory_gb()")
    print("  get_memory_tier(memory_gb)")
    print()
    print("Current System:")
    print(f"  Total memory: {total_memory:.1f} GB")
    print(f"  Memory tier: {current_tier.name}")
    print()
    print("Key Takeaways:")
    print("  - WorkflowSelector uses a decision matrix based on size + memory")
    print("  - Memory is re-evaluated on each call (no caching)")
    print("  - auto_select_workflow() provides a simple convenience API")
    print("  - Override with memory_limit_gb for reproducible behavior")
    print("  - Streaming configs include 4-layer defense by default (v0.3.6+)")
    print()
    print("Defense presets for streaming (v0.3.6+):")
    print("  HybridStreamingConfig.defense_strict()     # Warm-start refinement")
    print("  HybridStreamingConfig.defense_relaxed()    # Exploration")
    print("  HybridStreamingConfig.scientific_default() # Production scientific")
    print("  HybridStreamingConfig.defense_disabled()   # Pre-0.3.6 behavior")


if __name__ == "__main__":
    main()
