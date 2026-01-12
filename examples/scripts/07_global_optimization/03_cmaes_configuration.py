"""
CMA-ES Configuration and Method Selection (v0.6.3)

This script demonstrates CMA-ES configuration options and how the
MethodSelector automatically chooses between CMA-ES and Multi-Start.

Features demonstrated:
- CMAESConfig parameters and presets
- MethodSelector for automatic method selection
- Parameter scale ratio analysis
- CMA-ES vs Multi-Start selection criteria

Prerequisites:
    pip install "nlsq[global]"  # Installs evosax dependency

Run this example:
    python examples/scripts/07_global_optimization/03_cmaes_configuration.py
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print("Quick mode: running abbreviated CMA-ES configuration demo.")

from nlsq import fit
from nlsq.global_optimization import MethodSelector, is_evosax_available


def exponential_model(x, a, b, c):
    """Exponential decay model."""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("CMA-ES Configuration and Method Selection (v0.6.3)")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Check evosax availability
    # =========================================================================
    print("1. Checking evosax availability...")
    print("-" * 50)

    evosax_available = is_evosax_available()
    print(f"  evosax installed: {evosax_available}")

    if not evosax_available:
        print("  Note: Install evosax with: pip install 'nlsq[global]'")
        print("  CMA-ES will fall back to Multi-Start without evosax.")
    print()

    # =========================================================================
    # 2. MethodSelector - Understanding auto method selection
    # =========================================================================
    print("2. MethodSelector - Automatic Method Selection:")
    print("-" * 50)
    print()
    print("  The MethodSelector chooses between CMA-ES and Multi-Start based on:")
    print("  - Parameter scale ratio: max(upper-lower) / min(upper-lower)")
    print("  - CMA-ES is selected when scale_ratio > 1000 AND evosax is available")
    print("  - Multi-Start is selected otherwise")
    print()

    selector = MethodSelector()
    print(f"  Scale ratio threshold: {selector.scale_threshold}")

    # Test different bound configurations
    test_bounds = [
        ("Narrow bounds", np.array([0, 0, 0]), np.array([1, 1, 1])),
        ("Medium bounds", np.array([0, 0, 0]), np.array([10, 5, 1])),
        ("Wide bounds", np.array([0.001, 0, 0]), np.array([1000, 5, 1])),
        ("Very wide (multi-scale)", np.array([1e-6, 0, 0]), np.array([1e6, 5, 1])),
    ]

    print()
    print(f"  {'Scenario':<25} {'Scale Ratio':<15} {'Method':<15}")
    print("  " + "-" * 55)

    for label, lb, ub in test_bounds:
        scale_ratio = selector.compute_scale_ratio(lb, ub)
        method = selector.select("auto", lb, ub)
        print(f"  {label:<25} {scale_ratio:<15.1f} {method:<15}")

    # =========================================================================
    # 3. CMAESConfig Overview (if evosax is available)
    # =========================================================================
    print()
    print("3. CMAESConfig Parameters:")
    print("-" * 50)

    if evosax_available:
        from nlsq.global_optimization import CMAESConfig

        default_config = CMAESConfig()
        print("  Default CMAESConfig:")
        print(f"    popsize:          {default_config.popsize} (auto if None)")
        print(f"    max_generations:  {default_config.max_generations}")
        print(f"    sigma:            {default_config.sigma}")
        print(f"    restart_strategy: {default_config.restart_strategy}")
        print(f"    max_restarts:     {default_config.max_restarts}")
        print(f"    refine_with_nlsq: {default_config.refine_with_nlsq}")
        print(f"    seed:             {default_config.seed}")
    else:
        print("  CMAESConfig not available (evosax not installed)")
        print("  CMA-ES will use defaults when invoked via workflow='auto_global'")

    # =========================================================================
    # 4. CMA-ES Presets (if evosax is available)
    # =========================================================================
    print()
    print("4. CMA-ES Presets:")
    print("-" * 50)

    if evosax_available:
        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        presets = ["cmaes-fast", "cmaes", "cmaes-global"]
        print()
        print(
            f"  {'Preset':<15} {'max_gen':<10} {'restarts':<10} {'strategy':<12} {'refine':<10}"
        )
        print("  " + "-" * 57)

        for preset in presets:
            config = CMAESConfig.from_preset(preset)
            print(
                f"  {preset:<15} {config.max_generations:<10} "
                f"{config.max_restarts:<10} {config.restart_strategy:<12} "
                f"{config.refine_with_nlsq!s:<10}"
            )
    else:
        print("  Presets not available (evosax not installed)")
        print()
        print("  Available CMA-ES presets when evosax is installed:")
        print("    - 'cmaes-fast':   Quick exploration (no restarts)")
        print("    - 'cmaes':        Default (BIPOP restarts)")
        print("    - 'cmaes-global': Thorough search (more restarts)")

    # =========================================================================
    # 5. Using fit() with auto_global workflow
    # =========================================================================
    print()
    print("5. Using fit(workflow='auto_global'):")
    print("-" * 50)

    np.random.seed(42)
    n_samples = 200
    x_data = np.linspace(0, 5, n_samples)
    true_a, true_b, true_c = 3.0, 1.2, 0.5
    y_true = true_a * np.exp(-true_b * x_data) + true_c
    y_data = y_true + 0.1 * np.random.randn(n_samples)

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}")

    # Narrow bounds - will use Multi-Start
    bounds_narrow = ([0.1, 0.1, -1.0], [10.0, 5.0, 2.0])
    print()
    print("  Narrow bounds (scale ratio < 1000):")

    popt_narrow, _ = fit(
        exponential_model,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds_narrow,
        workflow="auto_global",
        n_starts=5,
    )
    print(
        f"    Result: a={popt_narrow[0]:.4f}, b={popt_narrow[1]:.4f}, c={popt_narrow[2]:.4f}"
    )

    # Wide bounds - may use CMA-ES if evosax is available
    bounds_wide = ([1e-6, 0.001, -10.0], [1e6, 100.0, 10.0])
    print()
    print("  Wide bounds (scale ratio > 1000):")

    scale_ratio = selector.compute_scale_ratio(
        np.array(bounds_wide[0]), np.array(bounds_wide[1])
    )
    expected_method = selector.select(
        "auto", np.array(bounds_wide[0]), np.array(bounds_wide[1])
    )
    print(f"    Scale ratio: {scale_ratio:.0f}")
    print(f"    Expected method: {expected_method}")

    popt_wide, _ = fit(
        exponential_model,
        x_data,
        y_data,
        p0=[1.0, 1.0, 0.0],
        bounds=bounds_wide,
        workflow="auto_global",
        n_starts=5,
    )
    print(
        f"    Result: a={popt_wide[0]:.4f}, b={popt_wide[1]:.4f}, c={popt_wide[2]:.4f}"
    )

    # =========================================================================
    # 6. Direct CMAESOptimizer usage (if evosax available)
    # =========================================================================
    if evosax_available and not QUICK:
        print()
        print("6. Direct CMAESOptimizer usage:")
        print("-" * 50)

        from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

        config = CMAESConfig(
            max_generations=100,
            restart_strategy="bipop",
            max_restarts=3,
            seed=42,
            refine_with_nlsq=True,
        )
        optimizer = CMAESOptimizer(config=config)

        result = optimizer.fit(exponential_model, x_data, y_data, bounds=bounds_narrow)

        print("  CMAESOptimizer result:")
        print(
            f"    a={result['popt'][0]:.4f}, b={result['popt'][1]:.4f}, c={result['popt'][2]:.4f}"
        )
        print()
        print("  Diagnostics:")
        diag = result["cmaes_diagnostics"]
        print(f"    Total generations: {diag['total_generations']}")
        print(f"    Total restarts: {diag['total_restarts']}")
        print(f"    Convergence reason: {diag['convergence_reason']}")
        print(f"    Wall time: {diag['wall_time']:.3f}s")

    # =========================================================================
    # 7. Visualize method selection boundaries
    # =========================================================================
    print()
    print("7. Saving method selection visualization...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create range of scale ratios
    scale_ratios = np.logspace(0, 6, 100)

    # Determine method for each scale ratio
    methods = []
    for sr in scale_ratios:
        if sr > selector.scale_threshold and evosax_available:
            methods.append("CMA-ES")
        else:
            methods.append("Multi-Start")

    # Plot
    colors = ["blue" if m == "Multi-Start" else "red" for m in methods]
    ax.scatter(scale_ratios, [1] * len(scale_ratios), c=colors, s=100, alpha=0.7)
    ax.axvline(
        x=selector.scale_threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({selector.scale_threshold})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Scale Ratio")
    ax.set_title("Method Selection Based on Scale Ratio")
    ax.set_yticks([])

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="Multi-Start"),
        Patch(facecolor="red", label="CMA-ES"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.annotate(
        "Multi-Start\n(default)",
        xy=(10, 1),
        xytext=(10, 1.3),
        fontsize=12,
        ha="center",
    )
    if evosax_available:
        ax.annotate(
            "CMA-ES\n(scale-invariant)",
            xy=(100000, 1),
            xytext=(100000, 1.3),
            fontsize=12,
            ha="center",
        )

    # Note: bbox_inches="tight" handles layout; plt.tight_layout() can cause
    # warnings with annotations that extend beyond the axes
    plt.savefig(FIG_DIR / "03_method_selection.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '03_method_selection.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary - Method Selection (v0.6.3)")
    print("=" * 70)
    print()
    print("The Three Workflows:")
    print("  workflow='auto'        : Local optimization (bounds optional)")
    print("  workflow='auto_global' : Global optimization (bounds required)")
    print("  workflow='hpc'         : auto_global + checkpointing")
    print()
    print("Method Selection (auto_global only):")
    print("  scale_ratio = max(upper-lower) / min(upper-lower)")
    print(
        f"  - scale_ratio > {selector.scale_threshold} AND evosax available -> CMA-ES"
    )
    print("  - otherwise -> Multi-Start")
    print()
    print("CMA-ES Configuration (when evosax is available):")
    print("  - 'cmaes-fast':   Quick exploration, no restarts")
    print("  - 'cmaes':        Default, BIPOP restarts")
    print("  - 'cmaes-global': Thorough search, more restarts")
    print()
    print("Usage:")
    print("  # Auto method selection")
    print("  fit(model, x, y, bounds=bounds, workflow='auto_global')")
    print()
    print("  # Explicit CMA-ES")
    print("  from nlsq.global_optimization import CMAESOptimizer")
    print("  optimizer = CMAESOptimizer.from_preset('cmaes')")
    print("  result = optimizer.fit(model, x, y, bounds=bounds)")


if __name__ == "__main__":
    main()
