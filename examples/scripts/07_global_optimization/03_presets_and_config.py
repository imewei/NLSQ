"""
Converted from 03_presets_and_config.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- All GlobalOptimizationConfig parameters
- Built-in presets: 'fast', 'robust', 'global', 'thorough', 'streaming'
- Custom configuration creation
- Parameter exploration: n_starts, sampler, center_on_p0, scale_factor
- Tournament selection parameters for streaming
- Visualization comparing preset behaviors

Run this example:
    python examples/scripts/07_global_optimization/03_presets_and_config.py
"""

import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import GlobalOptimizationConfig, curve_fit
from nlsq.global_optimization import (
    center_samples_around_p0,
    latin_hypercube_sample,
)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


if QUICK:
    print("Quick mode: running abbreviated global optimization preset demo.")

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def multimodal_model(x, a, b, c, d):
    """Multimodal model with multiple local minima."""
    return a * jnp.sin(b * x + c) + d


def simulate_tournament(n_starts, elimination_rounds, elimination_fraction):
    """Simulate tournament elimination process."""
    candidates = n_starts
    rounds = [candidates]

    for _ in range(elimination_rounds):
        candidates = max(1, int(candidates * (1 - elimination_fraction)))
        rounds.append(candidates)

    return rounds


def main():
    print("=" * 70)
    print("GlobalOptimizationConfig Deep Dive")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. GlobalOptimizationConfig Overview
    # =========================================================================
    print("1. GlobalOptimizationConfig - Default Values:")
    print("-" * 50)

    default_config = GlobalOptimizationConfig()
    print(f"  n_starts:             {default_config.n_starts}")
    print(f"  sampler:              {default_config.sampler}")
    print(f"  center_on_p0:         {default_config.center_on_p0}")
    print(f"  scale_factor:         {default_config.scale_factor}")
    print(f"  elimination_rounds:   {default_config.elimination_rounds}")
    print(f"  elimination_fraction: {default_config.elimination_fraction}")
    print(f"  batches_per_round:    {default_config.batches_per_round}")

    # =========================================================================
    # 2. Built-in Presets
    # =========================================================================
    print()
    print("2. Built-in Presets:")
    print("-" * 80)
    print(
        f"{'Preset':<12} {'n_starts':<10} {'sampler':<8} {'center_on_p0':<14} {'elim_rounds':<12}"
    )
    print("-" * 80)

    presets = ["fast", "robust", "global", "thorough", "streaming"]

    for preset_name in presets:
        config = GlobalOptimizationConfig.from_preset(preset_name)
        print(
            f"{preset_name:<12} {config.n_starts:<10} {config.sampler:<8} "
            f"{config.center_on_p0!s:<14} {config.elimination_rounds:<12}"
        )

    # =========================================================================
    # 3. Custom Configuration
    # =========================================================================
    print()
    print("3. Custom Configuration Examples:")
    print("-" * 50)

    custom_config = GlobalOptimizationConfig(
        n_starts=15,
        sampler="sobol",
        center_on_p0=True,
        scale_factor=0.8,
        elimination_rounds=2,
        elimination_fraction=0.5,
        batches_per_round=75,
    )

    print("  Custom config:")
    print(f"    n_starts: {custom_config.n_starts}")
    print(f"    sampler:  {custom_config.sampler}")

    # Modify from preset
    config_from_preset = GlobalOptimizationConfig.from_preset("robust")
    modified_config = config_from_preset.with_overrides(
        n_starts=8,
        sampler="halton",
    )

    print()
    print("  Modified from 'robust' preset:")
    print(f"    n_starts: {config_from_preset.n_starts} -> {modified_config.n_starts}")
    print(f"    sampler:  {config_from_preset.sampler} -> {modified_config.sampler}")

    # =========================================================================
    # 4. Serialization
    # =========================================================================
    print()
    print("4. Serialization:")
    print("-" * 50)

    config = GlobalOptimizationConfig(n_starts=25, sampler="sobol")
    config_dict = config.to_dict()

    print("  Serialized to dict:")
    for key, value in config_dict.items():
        if not key.startswith("_"):
            print(f"    {key}: {value}")

    restored_config = GlobalOptimizationConfig.from_dict(config_dict)
    print(f"  Restored n_starts: {restored_config.n_starts}")

    # =========================================================================
    # 5. Test Presets on Multimodal Problem
    # =========================================================================
    print()
    print("5. Testing Presets on Multimodal Problem:")
    print("-" * 70)

    # Generate synthetic data
    n_samples = cap_samples(200)
    x_data = np.linspace(0, 4 * np.pi, n_samples)

    # True parameters
    true_a, true_b, true_c, true_d = 2.0, 1.5, 0.5, 1.0
    y_true = true_a * np.sin(true_b * x_data + true_c) + true_d
    noise = 0.2 * np.random.randn(n_samples)
    y_data = y_true + noise

    # Define bounds and initial guess
    bounds = ([0.5, 0.5, -np.pi, -2.0], [5.0, 3.0, np.pi, 5.0])
    p0 = [1.0, 0.8, 0.0, 0.5]  # Poor initial guess

    print(f"  True parameters: a={true_a}, b={true_b}, c={true_c}, d={true_d}")

    results = {}
    test_presets = ["fast", "robust", "global"]

    for preset_name in test_presets:
        config = GlobalOptimizationConfig.from_preset(preset_name)

        start_time = time.time()

        effective_n_starts = (
            min(config.n_starts, 5)
            if QUICK and config.n_starts > 0
            else config.n_starts
        )
        if effective_n_starts > 0:
            popt, pcov = curve_fit(
                multimodal_model,
                x_data,
                y_data,
                p0=p0,
                bounds=bounds,
                multistart=True,
                n_starts=effective_n_starts,
                sampler=config.sampler,
            )
        else:
            popt, pcov = curve_fit(
                multimodal_model,
                x_data,
                y_data,
                p0=p0,
                bounds=bounds,
            )

        elapsed = time.time() - start_time

        y_pred = multimodal_model(x_data, *popt)
        ssr = float(jnp.sum((y_data - y_pred) ** 2))

        results[preset_name] = {
            "popt": popt,
            "ssr": ssr,
            "time": elapsed,
            "n_starts": effective_n_starts,
        }

        print(f"\n  {preset_name.upper()}:")
        print(f"    n_starts:   {effective_n_starts}")
        print(f"    Time:       {elapsed:.3f}s")
        print(f"    SSR:        {ssr:.4f}")
        print(
            f"    Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}, d={popt[3]:.3f}"
        )

    # =========================================================================
    # 6. Save Visualizations
    # =========================================================================
    print()
    print("6. Saving visualizations...")

    # Preset comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Data with all fits
    ax1 = axes[0, 0]
    ax1.scatter(x_data, y_data, alpha=0.4, s=10, label="Data", color="gray")
    ax1.plot(x_data, y_true, "k--", linewidth=2, label="True function", alpha=0.7)

    colors = {"fast": "blue", "robust": "green", "global": "red"}
    for preset_name in test_presets:
        popt = results[preset_name]["popt"]
        y_pred = multimodal_model(x_data, *popt)
        ssr = results[preset_name]["ssr"]
        ax1.plot(
            x_data,
            y_pred,
            color=colors[preset_name],
            linewidth=2,
            label=f"{preset_name} (SSR={ssr:.2f})",
        )

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Fits from Different Presets")
    ax1.legend(loc="upper right")

    # Top right: SSR comparison
    ax2 = axes[0, 1]
    preset_names = list(results.keys())
    ssrs = [results[p]["ssr"] for p in preset_names]
    bars = ax2.bar(preset_names, ssrs, color=[colors[p] for p in preset_names])
    ax2.set_xlabel("Preset")
    ax2.set_ylabel("Sum of Squared Residuals (SSR)")
    ax2.set_title("Fit Quality by Preset")
    for bar, ssr in zip(bars, ssrs, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{ssr:.2f}",
            ha="center",
            va="bottom",
        )

    # Bottom left: Time comparison
    ax3 = axes[1, 0]
    times = [results[p]["time"] for p in preset_names]
    bars = ax3.bar(preset_names, times, color=[colors[p] for p in preset_names])
    ax3.set_xlabel("Preset")
    ax3.set_ylabel("Time (seconds)")
    ax3.set_title("Computation Time by Preset")
    for bar, t in zip(bars, times, strict=False):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{t:.3f}s",
            ha="center",
            va="bottom",
        )

    # Bottom right: n_starts comparison
    ax4 = axes[1, 1]
    n_starts_list = [results[p]["n_starts"] for p in preset_names]
    bars = ax4.bar(preset_names, n_starts_list, color=[colors[p] for p in preset_names])
    ax4.set_xlabel("Preset")
    ax4.set_ylabel("Number of Starting Points")
    ax4.set_title("Starting Points by Preset")
    for bar, n in zip(bars, n_starts_list, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(n),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_preset_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '03_preset_comparison.png'}")

    # =========================================================================
    # 7. Scale factor visualization
    # =========================================================================
    n_samples_demo = 50
    n_params = 2
    key = jax.random.PRNGKey(42)

    base_samples = latin_hypercube_sample(n_samples_demo, n_params, rng_key=key)

    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    p0_demo = np.array([5.0, 5.0])

    scale_factors = [0.3, 0.6, 1.0, 1.5]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, sf in zip(axes, scale_factors, strict=False):
        centered = center_samples_around_p0(
            base_samples, p0_demo, scale_factor=sf, lb=lb, ub=ub
        )

        ax.scatter(centered[:, 0], centered[:, 1], alpha=0.6, s=30)
        ax.scatter(
            [p0_demo[0]],
            [p0_demo[1]],
            color="red",
            s=100,
            marker="*",
            label="p0",
            zorder=5,
        )
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xlabel("Parameter 1")
        ax.set_ylabel("Parameter 2")
        ax.set_title(f"scale_factor = {sf}")
        ax.axhline(y=p0_demo[1], color="red", linestyle="--", alpha=0.3)
        ax.axvline(x=p0_demo[0], color="red", linestyle="--", alpha=0.3)
        ax.legend()

    plt.suptitle(
        "Effect of scale_factor on Exploration Range (center_on_p0=True)", y=1.02
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_scale_factor_effect.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '03_scale_factor_effect.png'}")

    # =========================================================================
    # 8. Tournament elimination visualization
    # =========================================================================
    configs_to_compare = [
        ("robust", 5, 2, 0.5),
        ("global", 20, 3, 0.5),
        ("thorough", 50, 4, 0.5),
        ("streaming", 10, 3, 0.5),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, n_starts, elim_rounds, elim_frac in configs_to_compare:
        rounds = simulate_tournament(n_starts, elim_rounds, elim_frac)
        ax.plot(
            range(len(rounds)),
            rounds,
            "o-",
            label=f"{name} (n={n_starts}, rounds={elim_rounds})",
            linewidth=2,
        )

    ax.set_xlabel("Tournament Round")
    ax.set_ylabel("Remaining Candidates")
    ax.set_title("Tournament Elimination Process by Preset")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_tournament_elimination.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '03_tournament_elimination.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Preset recommendations:")
    print("  - Quick exploration:      preset='fast'")
    print("  - Production use:         preset='robust'")
    print("  - Complex problems:       preset='global'")
    print("  - Critical applications:  preset='thorough'")
    print("  - Large datasets:         preset='streaming'")
    print()
    print("Customization options:")
    print("  - GlobalOptimizationConfig(n_starts=N, sampler='lhs')")
    print("  - config.with_overrides(n_starts=N)")
    print("  - GlobalOptimizationConfig.from_preset('robust').with_overrides(...)")
    print()
    print("Key parameters:")
    print("  - n_starts: Number of starting points (0 = disable multi-start)")
    print("  - sampler: 'lhs', 'sobol', or 'halton'")
    print("  - scale_factor: Exploration range when center_on_p0=True")
    print("  - center_on_p0: Focus exploration around initial guess")


if __name__ == "__main__":
    main()
