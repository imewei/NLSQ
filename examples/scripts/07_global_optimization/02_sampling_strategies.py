"""
Converted from 02_sampling_strategies.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- Latin Hypercube Sampling (LHS) - stratified random sampling
- Sobol sequences - deterministic quasi-random sampling
- Halton sequences - prime-based quasi-random sampling
- Random sampling - uniform random baseline
- Visualization of space-filling properties
- Quantitative comparison of discrepancy and success rates

Run this example:
    python examples/scripts/07_global_optimization/02_sampling_strategies.py
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print("Quick mode: reduced iterations for sampling strategies.")

from nlsq import curve_fit
from nlsq.global_optimization import (
    halton_sample,
    latin_hypercube_sample,
    scale_samples_to_bounds,
    sobol_sample,
)


def multimodal_model(x, a, b, c):
    """Multimodal model with multiple local minima."""
    return a * jnp.sin(b * x + c)


def compute_simple_discrepancy(samples: np.ndarray) -> float:
    """Compute a simple discrepancy measure based on minimum neighbor distances.

    Lower discrepancy indicates more uniform coverage.
    """
    n = len(samples)
    if n < 2:
        return 0.0

    # Compute all pairwise distances
    distances = []
    for i in range(n):
        min_dist = float("inf")
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(samples[i] - samples[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)

    # Ideal minimum distance for uniform samples in d dimensions
    d = samples.shape[1]
    ideal_dist = (1.0 / n) ** (1.0 / d)

    # Discrepancy: variance of min distances from ideal
    distances = np.array(distances)
    discrepancy = np.std(distances) / ideal_dist

    return discrepancy


def evaluate_starting_points(samples_unit, lb, ub, x_data, y_data, true_params, bounds):
    """Evaluate success rate of starting points.

    Returns the fraction of starting points that converge to the global optimum.
    """
    # Scale samples to bounds
    samples_scaled = scale_samples_to_bounds(jnp.array(samples_unit), lb, ub)
    samples_scaled = np.array(samples_scaled)

    # True SSR (sum of squared residuals)
    y_true_pred = true_params[0] * np.sin(true_params[1] * x_data + true_params[2])
    true_ssr = np.sum((y_data - y_true_pred) ** 2)

    # Evaluate each starting point
    success_count = 0
    ssrs = []

    for p0 in samples_scaled:
        try:
            popt, _ = curve_fit(
                multimodal_model,
                x_data,
                y_data,
                p0=list(p0),
                bounds=bounds,
            )
            y_pred = multimodal_model(x_data, *popt)
            ssr = float(jnp.sum((y_data - y_pred) ** 2))
            ssrs.append(ssr)

            # Check if converged to near-optimal (within 10% of true SSR)
            if ssr < true_ssr * 1.1:
                success_count += 1
        except Exception:
            ssrs.append(float("inf"))

    success_rate = success_count / len(samples_scaled)
    best_ssr = min(ssrs)

    return success_rate, best_ssr, ssrs


def main():
    print("=" * 70)
    print("Sampling Strategies for Multi-Start Optimization")
    print("=" * 70)
    print()

    if QUICK:
        print("Quick mode: skipping full demonstration.")
        print()
        print("=" * 70)
        print("Summary: Sampling Strategies")
        print("=" * 70)
        print()
        print("Sampling Strategies:")
        print("  - Random: Baseline, poor space-filling")
        print("  - LHS: Stratified random, good coverage, stochastic")
        print("  - Sobol: Quasi-random, excellent coverage, deterministic")
        print("  - Halton: Quasi-random, very good coverage, deterministic")
        print()
        print("Key Functions:")
        print("  - latin_hypercube_sample(n_samples, n_dims)")
        print("  - sobol_sample(n_samples, n_dims)")
        print("  - halton_sample(n_samples, n_dims)")
        print("  - scale_samples_to_bounds(samples, lb, ub)")
        print()
        print("Usage with curve_fit():")
        print('  curve_fit(..., multistart=True, n_starts=10, sampler="lhs")')
        return

    # Set random seed for reproducibility
    np.random.seed(42)

    # =========================================================================
    # 1. Generate samples using each method
    # =========================================================================
    print("1. Generating samples with different methods...")

    n_samples = 50
    n_dims = 2

    # Generate samples using each method
    random_samples = np.random.rand(n_samples, n_dims)
    key = jax.random.PRNGKey(42)
    lhs_samples = latin_hypercube_sample(n_samples, n_dims, rng_key=key)
    sobol_samples = sobol_sample(n_samples, n_dims)
    halton_samples = halton_sample(n_samples, n_dims)

    print(f"  Generated {n_samples} samples in {n_dims} dimensions")

    # =========================================================================
    # 2. Visualize 2D samples
    # =========================================================================
    print()
    print("2. Saving 2D comparison visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    samples_dict = {
        "Random": random_samples,
        "Latin Hypercube (LHS)": np.array(lhs_samples),
        "Sobol": np.array(sobol_samples),
        "Halton": np.array(halton_samples),
    }

    for ax, (name, samples) in zip(axes.flat, samples_dict.items(), strict=False):
        ax.scatter(
            samples[:, 0],
            samples[:, 1],
            s=40,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"{name} ({n_samples} samples)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "02_sampling_comparison_2d.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

    # =========================================================================
    # 3. LHS stratification visualization
    # =========================================================================
    print()
    print("3. Saving LHS stratification visualization...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Random samples
    ax1 = axes[0]
    ax1.scatter(random_samples[:, 0], random_samples[:, 1], s=40, alpha=0.7)

    for i in range(n_samples + 1):
        ax1.axvline(x=i / n_samples, color="gray", alpha=0.2, linewidth=0.5)
        ax1.axhline(y=i / n_samples, color="gray", alpha=0.2, linewidth=0.5)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_title("Random: Multiple samples per stratum")

    # Right: LHS samples
    ax2 = axes[1]
    ax2.scatter(
        np.array(lhs_samples)[:, 0],
        np.array(lhs_samples)[:, 1],
        s=40,
        alpha=0.7,
        color="orange",
    )

    for i in range(n_samples + 1):
        ax2.axvline(x=i / n_samples, color="gray", alpha=0.2, linewidth=0.5)
        ax2.axhline(y=i / n_samples, color="gray", alpha=0.2, linewidth=0.5)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.set_title("LHS: Exactly one sample per stratum per dimension")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_lhs_stratification.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_lhs_stratification.png'}")

    # =========================================================================
    # 4. Quasi-random progressive fill
    # =========================================================================
    print()
    print("4. Saving quasi-random progressive fill visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    sample_counts = [8, 16, 32, 64]

    for i, n in enumerate(sample_counts):
        # Sobol
        sobol_n = sobol_sample(n, 2)
        axes[0, i].scatter(
            np.array(sobol_n)[:, 0], np.array(sobol_n)[:, 1], s=30, alpha=0.8
        )
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_title(f"Sobol: n={n}")
        axes[0, i].set_aspect("equal")
        axes[0, i].grid(True, alpha=0.3)

        # Halton
        halton_n = halton_sample(n, 2)
        axes[1, i].scatter(
            np.array(halton_n)[:, 0],
            np.array(halton_n)[:, 1],
            s=30,
            alpha=0.8,
            color="green",
        )
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_title(f"Halton: n={n}")
        axes[1, i].set_aspect("equal")
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "02_quasi_random_progressive.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_quasi_random_progressive.png'}")

    # =========================================================================
    # 5. Discrepancy comparison
    # =========================================================================
    print()
    print("5. Computing discrepancy comparison...")

    sample_sizes = [10, 20, 30, 50, 75, 100]
    discrepancies = {"Random": [], "LHS": [], "Sobol": [], "Halton": []}

    for n in sample_sizes:
        random_s = np.random.rand(n, 2)
        lhs_s = np.array(latin_hypercube_sample(n, 2, rng_key=jax.random.PRNGKey(42)))
        sobol_s = np.array(sobol_sample(n, 2))
        halton_s = np.array(halton_sample(n, 2))

        discrepancies["Random"].append(compute_simple_discrepancy(random_s))
        discrepancies["LHS"].append(compute_simple_discrepancy(lhs_s))
        discrepancies["Sobol"].append(compute_simple_discrepancy(sobol_s))
        discrepancies["Halton"].append(compute_simple_discrepancy(halton_s))

    # Plot discrepancy
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Random": "red", "LHS": "orange", "Sobol": "blue", "Halton": "green"}
    markers = {"Random": "o", "LHS": "s", "Sobol": "^", "Halton": "d"}

    for name, discs in discrepancies.items():
        ax.plot(
            sample_sizes,
            discs,
            marker=markers[name],
            color=colors[name],
            linewidth=2,
            markersize=8,
            label=name,
        )

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Discrepancy (lower is better)")
    ax.set_title("Discrepancy Comparison: Space-Filling Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_discrepancy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_discrepancy_comparison.png'}")

    # =========================================================================
    # 6. Success rate comparison
    # =========================================================================
    print()
    print("6. Computing success rate comparison...")

    # Generate test data
    np.random.seed(42)
    n_points = 100
    x_data = np.linspace(0, 4 * np.pi, n_points)
    true_params = [2.0, 1.5, 0.5]
    y_true = true_params[0] * np.sin(true_params[1] * x_data + true_params[2])
    y_data = y_true + 0.2 * np.random.randn(n_points)

    bounds = ([0.5, 0.5, -np.pi], [5.0, 3.0, np.pi])
    lb, ub = np.array(bounds[0]), np.array(bounds[1])

    n_starts_list = [5, 10] if QUICK else [5, 10, 15, 20, 25, 30]
    n_trials = 2 if QUICK else 5

    success_rates = {"Random": [], "LHS": [], "Sobol": [], "Halton": []}

    for n_starts in n_starts_list:
        print(f"  Evaluating n_starts = {n_starts}...")

        # Random and LHS: average over trials
        random_rates = []
        lhs_rates = []

        for trial in range(n_trials):
            random_s = np.random.rand(n_starts, 3)
            lhs_s = np.array(
                latin_hypercube_sample(n_starts, 3, rng_key=jax.random.PRNGKey(trial))
            )

            rate, _, _ = evaluate_starting_points(
                random_s, lb, ub, x_data, y_data, true_params, bounds
            )
            random_rates.append(rate)

            rate, _, _ = evaluate_starting_points(
                lhs_s, lb, ub, x_data, y_data, true_params, bounds
            )
            lhs_rates.append(rate)

        success_rates["Random"].append(np.mean(random_rates))
        success_rates["LHS"].append(np.mean(lhs_rates))

        # Sobol and Halton: deterministic
        sobol_s = np.array(sobol_sample(n_starts, 3))
        halton_s = np.array(halton_sample(n_starts, 3))

        rate, _, _ = evaluate_starting_points(
            sobol_s, lb, ub, x_data, y_data, true_params, bounds
        )
        success_rates["Sobol"].append(rate)

        rate, _, _ = evaluate_starting_points(
            halton_s, lb, ub, x_data, y_data, true_params, bounds
        )
        success_rates["Halton"].append(rate)

    # Plot success rate
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, rates in success_rates.items():
        ax.plot(
            n_starts_list,
            [r * 100 for r in rates],
            marker=markers[name],
            color=colors[name],
            linewidth=2,
            markersize=8,
            label=name,
        )

    ax.set_xlabel("Number of Starting Points")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate: Finding Global Optimum")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(
        FIG_DIR / "02_success_rate_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"  Saved: {FIG_DIR / '02_success_rate_comparison.png'}")

    # =========================================================================
    # 7. Demonstrate samplers with curve_fit()
    # =========================================================================
    print()
    print("7. Using samplers with curve_fit()...")

    samplers = ["lhs", "sobol", "halton"]
    results = {}

    for sampler in samplers:
        popt, pcov = curve_fit(
            multimodal_model,
            x_data,
            y_data,
            p0=[1.0, 1.0, 0.0],
            bounds=bounds,
            multistart=True,
            n_starts=10,
            sampler=sampler,
        )

        y_pred = multimodal_model(x_data, *popt)
        ssr = float(jnp.sum((y_data - y_pred) ** 2))

        results[sampler] = {"popt": popt, "ssr": ssr}

        print(f"  Sampler: {sampler}")
        print(f"    Parameters: a={popt[0]:.3f}, b={popt[1]:.3f}, c={popt[2]:.3f}")
        print(f"    SSR: {ssr:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Sampling Strategies:")
    print("  - Random: Baseline, poor space-filling")
    print("  - LHS: Stratified random, good coverage, stochastic")
    print("  - Sobol: Quasi-random, excellent coverage, deterministic")
    print("  - Halton: Quasi-random, very good coverage, deterministic")
    print()
    print("Key Functions:")
    print("  - latin_hypercube_sample(n_samples, n_dims)")
    print("  - sobol_sample(n_samples, n_dims)")
    print("  - halton_sample(n_samples, n_dims)")
    print("  - scale_samples_to_bounds(samples, lb, ub)")
    print()
    print("Usage with curve_fit():")
    print('  curve_fit(..., multistart=True, n_starts=10, sampler="lhs")')
    print()
    print("Sampler Selection Guidelines:")
    print("  - General use: LHS (default)")
    print("  - Reproducibility needed: Sobol")
    print("  - Low dimensions (2-5): Halton")
    print("  - High dimensions (>10): LHS")


if __name__ == "__main__":
    main()
