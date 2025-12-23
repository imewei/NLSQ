"""
Converted from 04_workflow_presets.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- All entries in the WORKFLOW_PRESETS dictionary
- Using presets for common fitting scenarios
- Inspecting preset configurations
- Customizing presets as starting points

Run this example:
    python examples/scripts/08_workflow_system/04_workflow_presets.py
"""

import time
from pathlib import Path
from pprint import pprint

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import WORKFLOW_PRESETS, HybridStreamingConfig, WorkflowConfig, fit

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def exponential_model(x, a, b, c):
    """Exponential decay: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("WORKFLOW_PRESETS Guide")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Available Presets
    # =========================================================================
    print("1. Available WORKFLOW_PRESETS:")
    print("-" * 60)

    for preset_name in WORKFLOW_PRESETS:
        description = WORKFLOW_PRESETS[preset_name].get("description", "No description")
        print(f"  {preset_name:<20} - {description}")

    # =========================================================================
    # 2. Inspecting Presets
    # =========================================================================
    print()
    print("2. Inspecting Presets:")
    print("-" * 60)

    print("\n'standard' preset:")
    pprint(WORKFLOW_PRESETS["standard"])

    print("\n'quality' preset:")
    pprint(WORKFLOW_PRESETS["quality"])

    print("\n'streaming' preset:")
    pprint(WORKFLOW_PRESETS["streaming"])

    # =========================================================================
    # 3. Preset Comparison Table
    # =========================================================================
    print()
    print("3. Preset Comparison:")
    print("=" * 100)
    print(
        f"{'Preset':<18} {'Tier':<12} {'Goal':<16} "
        f"{'Multistart':<12} {'n_starts':<10} {'gtol':<12}"
    )
    print("-" * 100)

    for name, config in WORKFLOW_PRESETS.items():
        tier = config.get("tier", "STANDARD")
        goal = config.get("goal", "ROBUST")
        multistart = config.get("enable_multistart", False)
        n_starts = config.get("n_starts", 0)
        gtol = config.get("gtol", 1e-8)

        multistart_str = "Yes" if multistart else "No"

        print(
            f"{name:<18} {tier:<12} {goal:<16} "
            f"{multistart_str:<12} {n_starts:<10} {gtol:<12.0e}"
        )

    # =========================================================================
    # 4. Testing Presets
    # =========================================================================
    print()
    print("4. Testing Presets on Exponential Decay:")
    print("-" * 70)

    # Generate test data
    n_samples = 500
    x_data = np.linspace(0, 5, n_samples)

    # True parameters
    true_a, true_b, true_c = 3.0, 1.2, 0.5
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    noise = 0.15 * np.random.randn(n_samples)
    y_data = y_true + noise

    # Initial guess and bounds
    p0 = [1.0, 0.5, 0.0]
    bounds = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")

    presets_to_test = ["fast", "standard", "quality"]
    results = {}

    for preset_name in presets_to_test:
        start_time = time.time()

        popt, pcov = fit(
            exponential_model,
            x_data,
            y_data,
            p0=p0,
            bounds=bounds,
            preset=preset_name,
        )

        elapsed = time.time() - start_time

        y_pred = exponential_model(x_data, *popt)
        ssr = float(jnp.sum((y_data - y_pred) ** 2))

        results[preset_name] = {
            "popt": popt,
            "ssr": ssr,
            "time": elapsed,
        }

        print(f"\n  {preset_name.upper()}:")
        print(f"    Time:       {elapsed:.4f}s")
        print(f"    SSR:        {ssr:.6f}")
        print(f"    Parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")

    # =========================================================================
    # 5. Creating WorkflowConfig from Presets
    # =========================================================================
    print()
    print("5. WorkflowConfig from Presets:")
    print("-" * 50)

    config = WorkflowConfig.from_preset("quality")

    print("\nWorkflowConfig from 'quality' preset:")
    print(f"  tier:              {config.tier.name}")
    print(f"  goal:              {config.goal.name}")
    print(f"  enable_multistart: {config.enable_multistart}")
    print(f"  n_starts:          {config.n_starts}")
    print(f"  gtol:              {config.gtol}")

    # =========================================================================
    # 6. Customizing Presets
    # =========================================================================
    print()
    print("6. Customizing Presets:")
    print("-" * 50)

    base_config = WorkflowConfig.from_preset("quality")
    custom_config = base_config.with_overrides(
        n_starts=30,
        sampler="sobol",
        gtol=1e-12,
    )

    print("\nCustomized 'quality' preset:")
    print(f"  Original n_starts: {base_config.n_starts}")
    print(f"  Custom n_starts:   {custom_config.n_starts}")
    print(f"  Original sampler:  {base_config.sampler}")
    print(f"  Custom sampler:    {custom_config.sampler}")
    print(f"  Original gtol:     {base_config.gtol}")
    print(f"  Custom gtol:       {custom_config.gtol}")

    # =========================================================================
    # 7. Preset Documentation
    # =========================================================================
    print()
    print("7. Preset Use Case Guide:")
    print("=" * 80)

    preset_docs = {
        "standard": {
            "summary": "Default curve_fit() behavior",
            "best_for": "Well-conditioned problems with good initial guesses",
            "tradeoffs": "Balanced speed/accuracy, no global search",
        },
        "quality": {
            "summary": "Highest precision fitting",
            "best_for": "Publication results, parameter uncertainty estimation",
            "tradeoffs": "Slower due to multi-start and tight tolerances",
        },
        "fast": {
            "summary": "Speed-optimized fitting",
            "best_for": "Exploratory analysis, development, quick iterations",
            "tradeoffs": "May converge to local minima",
        },
        "large_robust": {
            "summary": "Chunked processing with multi-start",
            "best_for": "Large datasets (1M-100M points) needing global search",
            "tradeoffs": "Memory-efficient but slower than standard",
        },
        "streaming": {
            "summary": "Streaming for huge datasets",
            "best_for": "Datasets that exceed available memory (100M+ points)",
            "tradeoffs": "No covariance matrix, approximate convergence",
        },
        "hpc_distributed": {
            "summary": "Multi-GPU/node HPC clusters",
            "best_for": "PBS Pro, Slurm clusters with checkpoint recovery",
            "tradeoffs": "Requires HPC environment setup",
        },
        "memory_efficient": {
            "summary": "Minimize memory footprint",
            "best_for": "Memory-constrained systems, edge devices",
            "tradeoffs": "Smaller chunk sizes = more overhead",
        },
    }

    for name, doc in preset_docs.items():
        print(f"\n  {name.upper()}:")
        print(f"    Summary:    {doc['summary']}")
        print(f"    Best for:   {doc['best_for']}")
        print(f"    Tradeoffs:  {doc['tradeoffs']}")

    # =========================================================================
    # 8. Defense Layer Presets (v0.3.6+)
    # =========================================================================
    print()
    print()
    print("8. Defense Layer Presets (v0.3.6+):")
    print("-" * 70)
    print()
    print("For streaming workflows, HybridStreamingConfig provides defense presets")
    print("that protect against Adam warmup divergence when starting near optimal:")
    print()

    defense_presets = {
        "defense_strict": {
            "method": "HybridStreamingConfig.defense_strict()",
            "use_case": "Warm-start refinement (previous fit as p0)",
            "lr_range": "1e-6 to 1e-4",
        },
        "defense_relaxed": {
            "method": "HybridStreamingConfig.defense_relaxed()",
            "use_case": "Exploration (rough initial guesses)",
            "lr_range": "1e-4 to 0.01",
        },
        "scientific_default": {
            "method": "HybridStreamingConfig.scientific_default()",
            "use_case": "Production scientific computing",
            "lr_range": "1e-6 to 0.001 (balanced)",
        },
        "defense_disabled": {
            "method": "HybridStreamingConfig.defense_disabled()",
            "use_case": "Pre-0.3.6 behavior (no protection)",
            "lr_range": "Fixed at warmup_learning_rate",
        },
    }

    for name, info in defense_presets.items():
        print(f"  {name.upper()}:")
        print(f"    Method:   {info['method']}")
        print(f"    Use case: {info['use_case']}")
        print(f"    LR range: {info['lr_range']}")
        print()

    print("The 4-layer defense strategy:")
    print("  Layer 1: Warm Start Detection - Skip warmup if near optimal")
    print("  Layer 2: Adaptive Learning Rate - Scale LR based on fit quality")
    print("  Layer 3: Cost-Increase Guard - Abort if loss increases > 5%")
    print("  Layer 4: Step Clipping - Limit parameter update magnitude")

    # =========================================================================
    # 9. Visualization
    # =========================================================================
    print()
    print()
    print("9. Saving visualizations...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    preset_names = list(results.keys())
    colors = {"fast": "blue", "standard": "green", "quality": "red"}

    # SSR comparison
    ax1 = axes[0]
    ssrs = [results[p]["ssr"] for p in preset_names]
    bars = ax1.bar(preset_names, ssrs, color=[colors[p] for p in preset_names])
    ax1.set_xlabel("Preset")
    ax1.set_ylabel("Sum of Squared Residuals")
    ax1.set_title("Fit Quality by Preset")
    for bar, ssr in zip(bars, ssrs, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ssr:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Time comparison
    ax2 = axes[1]
    times = [results[p]["time"] for p in preset_names]
    bars = ax2.bar(preset_names, times, color=[colors[p] for p in preset_names])
    ax2.set_xlabel("Preset")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Computation Time by Preset")
    for bar, t in zip(bars, times, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.3f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Tolerance comparison
    ax3 = axes[2]
    tols = [WORKFLOW_PRESETS[p]["gtol"] for p in preset_names]
    bars = ax3.bar(preset_names, tols, color=[colors[p] for p in preset_names])
    ax3.set_xlabel("Preset")
    ax3.set_ylabel("gtol")
    ax3.set_title("Tolerance (gtol) by Preset")
    ax3.set_yscale("log")
    for bar, t in zip(bars, tols, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{t:.0e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_preset_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_preset_comparison.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Available presets:")
    for name in WORKFLOW_PRESETS:
        desc = WORKFLOW_PRESETS[name].get("description", "")
        print(f"  - {name}: {desc}")
    print()
    print("Quick usage:")
    print("  fit(model, x, y, preset='quality')")
    print("  WorkflowConfig.from_preset('quality')")
    print("  config.with_overrides(n_starts=30)")
    print()
    print("Defense presets for streaming (v0.3.6+):")
    print("  HybridStreamingConfig.defense_strict()     # Warm-start refinement")
    print("  HybridStreamingConfig.defense_relaxed()    # Exploration")
    print("  HybridStreamingConfig.scientific_default() # Production scientific")
    print("  HybridStreamingConfig.defense_disabled()   # Pre-0.3.6 behavior")


if __name__ == "__main__":
    main()
