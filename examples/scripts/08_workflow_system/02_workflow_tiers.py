"""
Converted from 02_workflow_tiers.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- Understanding the four workflow tiers
- Automatic tier selection based on dataset size and memory
- Manual tier override with WorkflowConfig
- Memory usage comparison across tiers

Run this example:
    python examples/scripts/08_workflow_system/02_workflow_tiers.py
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import OptimizationGoal, WorkflowConfig, WorkflowTier, fit
from nlsq.core.workflow import (
    DatasetSizeTier,
    MemoryTier,
    auto_select_workflow,
)
from nlsq.streaming.large_dataset import MemoryEstimator, get_memory_tier

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def estimate_memory_usage(n_points, n_params, tier):
    """Estimate memory usage in GB for a given tier."""
    bytes_per_point = 8 * (3 + n_params)  # x, y, residual + jacobian

    if tier == WorkflowTier.STANDARD:
        # All data in memory
        return n_points * bytes_per_point / 1e9
    elif tier == WorkflowTier.CHUNKED:
        # Chunk size typically 100K-1M
        chunk_size = min(1_000_000, n_points)
        return chunk_size * bytes_per_point / 1e9
    elif tier in (WorkflowTier.STREAMING, WorkflowTier.STREAMING_CHECKPOINT):
        # Batch size typically 50K
        batch_size = 50_000
        return batch_size * bytes_per_point / 1e9
    else:
        return 0


def main():
    print("=" * 70)
    print("Workflow Tiers: STANDARD, CHUNKED, STREAMING")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Overview of Workflow Tiers
    # =========================================================================
    print("1. Workflow Tiers Overview")
    print("-" * 50)

    tier_info = {
        WorkflowTier.STANDARD: {
            "description": "Standard curve_fit() for small datasets",
            "dataset_size": "< 10K points",
            "memory": "O(N) - loads all data into memory",
            "defense_layers": "N/A",
        },
        WorkflowTier.CHUNKED: {
            "description": "LargeDatasetFitter with automatic chunking",
            "dataset_size": "10K - 10M points",
            "memory": "O(chunk_size) - processes data in chunks",
            "defense_layers": "N/A",
        },
        WorkflowTier.STREAMING: {
            "description": "AdaptiveHybridStreamingOptimizer for huge datasets",
            "dataset_size": "10M - 100M points",
            "memory": "O(batch_size) - mini-batch gradient descent",
            "defense_layers": "4-layer defense enabled (v0.3.6+)",
        },
        WorkflowTier.STREAMING_CHECKPOINT: {
            "description": "Streaming with automatic checkpointing",
            "dataset_size": "> 100M points",
            "memory": "O(batch_size) + checkpoint storage",
            "defense_layers": "4-layer defense enabled (v0.3.6+)",
        },
    }

    for tier, info in tier_info.items():
        print(f"\n{tier.name}:")
        print(f"  Description: {info['description']}")
        print(f"  Dataset Size: {info['dataset_size']}")
        print(f"  Memory: {info['memory']}")
        print(f"  Defense Layers: {info['defense_layers']}")

    # =========================================================================
    # 2. Dataset Size Tiers
    # =========================================================================
    print()
    print()
    print("2. Dataset Size Tiers and Thresholds")
    print("-" * 50)

    for size_tier in DatasetSizeTier:
        max_pts = size_tier.max_points
        tol = size_tier.tolerance
        if max_pts == float("inf"):
            print(f"  {size_tier.name:12s}: > 100M points, tolerance = {tol:.0e}")
        else:
            print(
                f"  {size_tier.name:12s}: < {max_pts / 1e6:.0f}M points, tolerance = {tol:.0e}"
            )

    # =========================================================================
    # 3. Memory Tiers
    # =========================================================================
    print()
    print()
    print("3. Memory Tiers")
    print("-" * 50)

    for mem_tier in MemoryTier:
        print(f"  {mem_tier.name:10s}: {mem_tier.description}")

    # Check current system memory
    available_memory = MemoryEstimator.get_available_memory_gb()
    current_tier = get_memory_tier(available_memory)
    print(
        f"\n  Current system: {available_memory:.1f} GB available -> {current_tier.name}"
    )

    # =========================================================================
    # 4. Automatic Tier Selection
    # =========================================================================
    print()
    print()
    print("4. Automatic Tier Selection")
    print("-" * 50)
    print(f"  Available memory: {available_memory:.1f} GB")
    print()

    test_sizes = [1_000, 50_000, 500_000, 5_000_000, 50_000_000, 500_000_000]
    n_params = 5

    for n_points in test_sizes:
        config = auto_select_workflow(n_points, n_params)
        config_type = type(config).__name__

        # Determine tier from config type
        if "GlobalOptimization" in config_type:
            tier = "STANDARD (with multi-start)"
        elif "LDMemory" in config_type:
            tier = "STANDARD or CHUNKED"
        elif "HybridStreaming" in config_type:
            tier = "STREAMING or STREAMING_CHECKPOINT"
        else:
            tier = config_type

        if n_points >= 1_000_000:
            size_str = f"{n_points / 1_000_000:.0f}M"
        elif n_points >= 1_000:
            size_str = f"{n_points / 1_000:.0f}K"
        else:
            size_str = str(n_points)

        print(f"  {size_str:>8s} points -> {tier}")

    # =========================================================================
    # 5. Tier Selection Decision Tree Visualization
    # =========================================================================
    print()
    print()
    print("5. Saving tier selection decision tree...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(
        5,
        9.5,
        "Workflow Tier Selection Decision Tree",
        ha="center",
        fontsize=16,
        fontweight="bold",
    )

    # Root node
    ax.add_patch(
        plt.Rectangle(
            (3.5, 8.2), 3, 0.8, fill=True, facecolor="lightblue", edgecolor="black"
        )
    )
    ax.text(5, 8.6, "Dataset Size?", ha="center", va="center", fontsize=11)

    # Level 1 branches
    # Small
    ax.plot([4.2, 2, 2], [8.2, 7.5, 7.0], "k-", linewidth=1)
    ax.text(2.5, 7.7, "< 10K", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (0.5, 6.2), 3, 0.8, fill=True, facecolor="lightgreen", edgecolor="black"
        )
    )
    ax.text(
        2, 6.6, "STANDARD", ha="center", va="center", fontsize=10, fontweight="bold"
    )

    # Medium
    ax.plot([5, 5], [8.2, 7.0], "k-", linewidth=1)
    ax.text(5.3, 7.5, "10K - 10M", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (3.5, 6.2), 3, 0.8, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(5, 6.6, "Memory Check", ha="center", va="center", fontsize=10)

    # Large
    ax.plot([5.8, 8, 8], [8.2, 7.5, 7.0], "k-", linewidth=1)
    ax.text(7.2, 7.7, "> 10M", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (6.5, 6.2), 3, 0.8, fill=True, facecolor="lightyellow", edgecolor="black"
        )
    )
    ax.text(8, 6.6, "Memory Check", ha="center", va="center", fontsize=10)

    # Level 2 - Medium dataset branches
    ax.plot([4.2, 3, 3], [6.2, 5.5, 5.0], "k-", linewidth=1)
    ax.text(3.3, 5.6, "> 16GB", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (1.5, 4.2), 3, 0.8, fill=True, facecolor="lightgreen", edgecolor="black"
        )
    )
    ax.text(
        3, 4.6, "STANDARD", ha="center", va="center", fontsize=10, fontweight="bold"
    )

    ax.plot([5.8, 7, 7], [6.2, 5.5, 5.0], "k-", linewidth=1)
    ax.text(6.5, 5.6, "< 16GB", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (5.5, 4.2), 3, 0.8, fill=True, facecolor="orange", edgecolor="black"
        )
    )
    ax.text(7, 4.6, "CHUNKED", ha="center", va="center", fontsize=10, fontweight="bold")

    # Level 2 - Large dataset branches
    ax.plot([7.2, 6, 6], [6.2, 5.5, 3.0], "k-", linewidth=1)
    ax.text(6.3, 5.6, "> 64GB", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (4.5, 2.2), 3, 0.8, fill=True, facecolor="orange", edgecolor="black"
        )
    )
    ax.text(6, 2.6, "CHUNKED", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.plot([8.8, 9.5, 9.5], [6.2, 5.5, 3.0], "k-", linewidth=1)
    ax.text(9.2, 5.6, "< 64GB", fontsize=9)
    ax.add_patch(
        plt.Rectangle(
            (8, 2.2), 1.8, 0.8, fill=True, facecolor="salmon", edgecolor="black"
        )
    )
    ax.text(
        8.9, 2.6, "STREAMING", ha="center", va="center", fontsize=9, fontweight="bold"
    )

    # Additional note for massive datasets
    ax.add_patch(
        plt.Rectangle(
            (0.5, 0.5),
            9,
            1.2,
            fill=True,
            facecolor="lightgray",
            edgecolor="black",
            alpha=0.3,
        )
    )
    ax.text(
        5,
        1.1,
        "For > 100M points: STREAMING_CHECKPOINT (adds fault tolerance)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_tier_decision_tree.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_tier_decision_tree.png'}")

    # =========================================================================
    # 6. Manual Tier Override
    # =========================================================================
    print()
    print()
    print("6. Manual Tier Override")
    print("-" * 50)

    config_standard = WorkflowConfig(tier=WorkflowTier.STANDARD)
    config_chunked = WorkflowConfig(tier=WorkflowTier.CHUNKED)
    config_streaming = WorkflowConfig(tier=WorkflowTier.STREAMING)
    config_checkpoint = WorkflowConfig(tier=WorkflowTier.STREAMING_CHECKPOINT)

    print(f"  config_standard.tier = {config_standard.tier}")
    print(f"  config_chunked.tier = {config_chunked.tier}")
    print(f"  config_streaming.tier = {config_streaming.tier}")
    print(f"  config_checkpoint.tier = {config_checkpoint.tier}")

    # =========================================================================
    # 7. Test Fit with Default Tier
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
        estimate_memory_usage(int(n), n_params, WorkflowTier.STANDARD)
        for n in dataset_sizes
    ]
    memory_chunked = [
        estimate_memory_usage(int(n), n_params, WorkflowTier.CHUNKED)
        for n in dataset_sizes
    ]
    memory_streaming = [
        estimate_memory_usage(int(n), n_params, WorkflowTier.STREAMING)
        for n in dataset_sizes
    ]

    # Plot memory comparison
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.loglog(dataset_sizes, memory_standard, "b-", linewidth=2, label="STANDARD")
    ax.loglog(dataset_sizes, memory_chunked, "orange", linewidth=2, label="CHUNKED")
    ax.loglog(dataset_sizes, memory_streaming, "r-", linewidth=2, label="STREAMING")

    # Add memory threshold lines
    ax.axhline(y=16, color="gray", linestyle="--", alpha=0.5, label="16 GB limit")
    ax.axhline(y=64, color="gray", linestyle=":", alpha=0.5, label="64 GB limit")

    # Add tier transition zones
    ax.axvline(x=10_000, color="green", linestyle="--", alpha=0.3)
    ax.axvline(x=10_000_000, color="orange", linestyle="--", alpha=0.3)
    ax.axvline(x=100_000_000, color="red", linestyle="--", alpha=0.3)

    ax.text(3000, 100, "STANDARD\nzone", fontsize=9, ha="center")
    ax.text(300_000, 100, "CHUNKED\nzone", fontsize=9, ha="center")
    ax.text(30_000_000, 100, "STREAMING\nzone", fontsize=9, ha="center")

    ax.set_xlabel("Dataset Size (points)")
    ax.set_ylabel("Peak Memory Usage (GB)")
    ax.set_title("Memory Usage by Workflow Tier")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e3, 1e9)
    ax.set_ylim(1e-3, 1e3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_memory_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_memory_comparison.png'}")

    # =========================================================================
    # 9. Defense Layers for Streaming Tiers (v0.3.6+)
    # =========================================================================
    print()
    print()
    print("9. Defense Layers for Streaming Tiers (v0.3.6+)")
    print("-" * 50)
    print()
    print(
        "STREAMING and STREAMING_CHECKPOINT tiers use AdaptiveHybridStreamingOptimizer,"
    )
    print("which includes a 4-layer defense strategy against Adam warmup divergence:")
    print()
    print("  Layer 1 (Warm Start Detection):")
    print("    - Skips warmup if initial loss < 1% of data variance")
    print("    - Prevents overshooting when starting near the optimum")
    print()
    print("  Layer 2 (Adaptive Learning Rate):")
    print("    - Scales LR based on fit quality (1e-6 to 0.001)")
    print("    - lr_refinement=1e-6, lr_careful=1e-5, lr_exploration=0.001")
    print()
    print("  Layer 3 (Cost-Increase Guard):")
    print("    - Aborts warmup if loss increases > 5%")
    print("    - Triggers early switch to Gauss-Newton phase")
    print()
    print("  Layer 4 (Step Clipping):")
    print("    - Limits parameter update magnitude (max norm 0.1)")
    print("    - Prevents catastrophic parameter jumps")
    print()
    print("Defense Presets:")
    print("  - HybridStreamingConfig.defense_strict()     # Warm-start refinement")
    print("  - HybridStreamingConfig.defense_relaxed()    # Exploration")
    print("  - HybridStreamingConfig.scientific_default() # Production scientific")
    print("  - HybridStreamingConfig.defense_disabled()   # Pre-0.3.6 behavior")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Workflow Tiers:")
    print("  STANDARD:             < 10K points, full precision")
    print("  CHUNKED:              10K - 10M points, memory-managed")
    print("  STREAMING:            10M - 100M points, mini-batch + defense layers")
    print("  STREAMING_CHECKPOINT: > 100M points, fault-tolerant + defense layers")
    print()
    print("Override syntax:")
    print("  config = WorkflowConfig(tier=WorkflowTier.CHUNKED)")
    print()
    print(f"Current system memory: {available_memory:.1f} GB ({current_tier.name})")
    print()
    print("Key takeaways:")
    print("  - Automatic tier selection based on dataset size and memory")
    print("  - Override for specific memory constraints")
    print("  - STREAMING provides O(batch_size) memory for unlimited data")
    print("  - STREAMING tiers include 4-layer defense against warmup divergence")


if __name__ == "__main__":
    main()
