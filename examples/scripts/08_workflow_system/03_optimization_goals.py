"""
Converted from 03_optimization_goals.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- All 5 OptimizationGoal values and their behaviors
- Internal settings each goal applies
- Combining goals with WorkflowTier
- Visualization comparing goal performance

Run this example:
    python examples/scripts/08_workflow_system/03_optimization_goals.py
"""

import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import OptimizationGoal, WorkflowConfig, WorkflowTier, fit
from nlsq.core.workflow import DatasetSizeTier, calculate_adaptive_tolerances

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_decay(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("Goal-Driven Optimization")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. OptimizationGoal Overview
    # =========================================================================
    print("1. OptimizationGoal Values:")
    print("-" * 60)

    for goal in OptimizationGoal:
        print(f"  {goal.name:<20} = {goal.value}")

    # Goal descriptions
    goal_info = {
        OptimizationGoal.FAST: {
            "description": "Prioritize speed with local optimization only",
            "tolerances": "One tier looser",
            "multistart": "Disabled",
            "use_case": "Quick exploration, well-conditioned problems",
        },
        OptimizationGoal.ROBUST: {
            "description": "Standard tolerances with multi-start",
            "tolerances": "Dataset-appropriate",
            "multistart": "Enabled",
            "use_case": "Production use, unknown problem conditioning",
        },
        OptimizationGoal.GLOBAL: {
            "description": "Synonym for ROBUST (emphasizes global optimization)",
            "tolerances": "Dataset-appropriate",
            "multistart": "Enabled",
            "use_case": "Same as ROBUST, semantic clarity",
        },
        OptimizationGoal.MEMORY_EFFICIENT: {
            "description": "Minimize memory usage with standard tolerances",
            "tolerances": "Dataset-appropriate",
            "multistart": "Disabled",
            "use_case": "Memory-constrained environments",
        },
        OptimizationGoal.QUALITY: {
            "description": "Highest precision/accuracy as TOP PRIORITY",
            "tolerances": "One tier tighter",
            "multistart": "Enabled + validation passes",
            "use_case": "Publication-quality results",
        },
    }

    print("\nGoal Details:")
    print("-" * 80)

    for goal, info in goal_info.items():
        print(f"\n  {goal.name}:")
        print(f"    Description:  {info['description']}")
        print(f"    Tolerances:   {info['tolerances']}")
        print(f"    Multi-start:  {info['multistart']}")
        print(f"    Use case:     {info['use_case']}")

    # =========================================================================
    # 2. Adaptive Tolerances
    # =========================================================================
    print()
    print("2. Adaptive Tolerances by Dataset Size and Goal:")
    print("-" * 70)
    print(f"{'Dataset Size':<15} {'FAST':<15} {'ROBUST':<15} {'QUALITY':<15}")
    print("-" * 70)

    dataset_sizes = [500, 5_000, 50_000, 500_000, 5_000_000]
    goals_to_compare = [
        OptimizationGoal.FAST,
        OptimizationGoal.ROBUST,
        OptimizationGoal.QUALITY,
    ]

    for n_points in dataset_sizes:
        tols = {}
        for goal in goals_to_compare:
            tols[goal.name] = calculate_adaptive_tolerances(n_points, goal)["gtol"]

        print(
            f"{n_points:>12,}   {tols['FAST']:<15.0e} "
            f"{tols['ROBUST']:<15.0e} {tols['QUALITY']:<15.0e}"
        )

    # =========================================================================
    # 3. DatasetSizeTier Reference
    # =========================================================================
    print()
    print("3. DatasetSizeTier Reference:")
    print("-" * 50)
    print(f"{'Tier':<15} {'Max Points':<15} {'Base Tolerance'}")
    print("-" * 50)

    for tier in DatasetSizeTier:
        max_pts = tier.max_points
        if max_pts == float("inf"):
            max_pts_str = "unlimited"
        else:
            max_pts_str = f"{int(max_pts):,}"
        print(f"{tier.name:<15} {max_pts_str:<15} {tier.tolerance:.0e}")

    # =========================================================================
    # 4. Practical Comparison
    # =========================================================================
    print()
    print("4. Testing Goals on Exponential Decay Problem:")
    print("-" * 70)

    # Generate synthetic data
    n_samples = 1000
    x_data = np.linspace(0, 5, n_samples)

    # True parameters
    true_a, true_b, true_c = 3.0, 1.2, 0.5

    y_true = true_a * np.exp(-true_b * x_data) + true_c
    noise = 0.1 * np.random.randn(n_samples)
    y_data = y_true + noise

    # Initial guess and bounds
    p0 = [1.0, 0.5, 0.0]
    bounds = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")
    print(f"  Dataset size: {n_samples} points")

    results = {}
    goals_to_test = ["fast", "robust", "global"]

    for goal_name in goals_to_test:
        start_time = time.time()

        popt, pcov = fit(
            exponential_decay,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            preset=goal_name,
        )

        elapsed = time.time() - start_time

        y_pred = exponential_decay(x_data, *popt)
        ssr = float(jnp.sum((y_data - y_pred) ** 2))

        param_errors = [abs(popt[i] - [true_a, true_b, true_c][i]) for i in range(3)]

        results[goal_name] = {
            "popt": popt,
            "ssr": ssr,
            "time": elapsed,
            "errors": param_errors,
        }

        print(f"\n  {goal_name.upper()}:")
        print(f"    Time:       {elapsed:.4f}s")
        print(f"    SSR:        {ssr:.6f}")
        print(f"    Parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
        print(
            f"    Errors:     a_err={param_errors[0]:.4f}, "
            f"b_err={param_errors[1]:.4f}, c_err={param_errors[2]:.4f}"
        )

    # =========================================================================
    # 5. WorkflowConfig with Goals
    # =========================================================================
    print()
    print("5. WorkflowConfig with Different Goals:")
    print("-" * 60)

    for goal in [
        OptimizationGoal.FAST,
        OptimizationGoal.ROBUST,
        OptimizationGoal.QUALITY,
    ]:
        config = WorkflowConfig(goal=goal)
        print(f"\n  {goal.name}:")
        print(f"    tier:              {config.tier.name}")
        print(f"    gtol:              {config.gtol}")
        print(f"    enable_multistart: {config.enable_multistart}")

    # Combining goals with tiers
    print("\n  Quality + Streaming (combined):")
    config_combined = WorkflowConfig(
        goal=OptimizationGoal.QUALITY,
        tier=WorkflowTier.STREAMING,
        enable_multistart=True,
        n_starts=20,
    )
    print(f"    tier:              {config_combined.tier.name}")
    print(f"    goal:              {config_combined.goal.name}")
    print(f"    enable_multistart: {config_combined.enable_multistart}")
    print(f"    n_starts:          {config_combined.n_starts}")

    # =========================================================================
    # 6. GLOBAL vs ROBUST
    # =========================================================================
    print()
    print("6. GLOBAL and ROBUST Equivalence:")
    print("-" * 50)

    normalized = OptimizationGoal.normalize(OptimizationGoal.GLOBAL)
    print(f"  OptimizationGoal.GLOBAL normalizes to: {normalized.name}")

    tols_global = calculate_adaptive_tolerances(10000, OptimizationGoal.GLOBAL)
    tols_robust = calculate_adaptive_tolerances(10000, OptimizationGoal.ROBUST)

    print(f"  GLOBAL gtol: {tols_global['gtol']}")
    print(f"  ROBUST gtol: {tols_robust['gtol']}")
    print(f"  Same: {tols_global['gtol'] == tols_robust['gtol']}")

    # =========================================================================
    # 7. Visualization
    # =========================================================================
    print()
    print("7. Saving visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {"fast": "blue", "robust": "green", "global": "red"}

    # Top left: Tolerance comparison across dataset sizes
    ax1 = axes[0, 0]
    sizes = np.logspace(2, 8, 50).astype(int)

    for goal in [
        OptimizationGoal.FAST,
        OptimizationGoal.ROBUST,
        OptimizationGoal.QUALITY,
    ]:
        tols = [calculate_adaptive_tolerances(n, goal)["gtol"] for n in sizes]
        ax1.loglog(sizes, tols, label=goal.name, linewidth=2)

    ax1.set_xlabel("Dataset Size (points)")
    ax1.set_ylabel("gtol")
    ax1.set_title("Adaptive Tolerances by Goal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top right: SSR comparison
    ax2 = axes[0, 1]
    goal_names = list(results.keys())
    ssrs = [results[g]["ssr"] for g in goal_names]
    bars = ax2.bar(goal_names, ssrs, color=[colors[g] for g in goal_names])
    ax2.set_xlabel("Goal")
    ax2.set_ylabel("Sum of Squared Residuals")
    ax2.set_title("Fit Quality by Goal")
    for bar, ssr in zip(bars, ssrs, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ssr:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Bottom left: Time comparison
    ax3 = axes[1, 0]
    times = [results[g]["time"] for g in goal_names]
    bars = ax3.bar(goal_names, times, color=[colors[g] for g in goal_names])
    ax3.set_xlabel("Goal")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Computation Time by Goal")
    for bar, t in zip(bars, times, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Bottom right: Parameter errors
    ax4 = axes[1, 1]
    x_pos = np.arange(len(goal_names))
    width = 0.25

    for i, param in enumerate(["a", "b", "c"]):
        errors = [results[g]["errors"][i] for g in goal_names]
        ax4.bar(x_pos + i * width, errors, width, label=f"{param} error")

    ax4.set_xlabel("Goal")
    ax4.set_ylabel("Absolute Error")
    ax4.set_title("Parameter Errors by Goal")
    ax4.set_xticks(x_pos + width)
    ax4.set_xticklabels(goal_names)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_goal_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '03_goal_comparison.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")
    print()
    print("Goal recommendations:")
    print("  - Exploratory analysis:    OptimizationGoal.FAST")
    print("  - Production fitting:      OptimizationGoal.ROBUST")
    print("  - Global search emphasis:  OptimizationGoal.GLOBAL")
    print("  - Memory constraints:      OptimizationGoal.MEMORY_EFFICIENT")
    print("  - Publication quality:     OptimizationGoal.QUALITY")
    print()
    print("Key behaviors:")
    print("  - FAST: Looser tolerances, no multi-start")
    print("  - ROBUST/GLOBAL: Standard tolerances, multi-start enabled")
    print("  - MEMORY_EFFICIENT: Standard tolerances, streaming/chunking preferred")
    print("  - QUALITY: Tighter tolerances, multi-start + validation")


if __name__ == "__main__":
    main()
