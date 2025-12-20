"""
Converted from 04_tournament_selection.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.

Features demonstrated:
- TournamentSelector for progressive elimination
- Tournament selection for memory-efficient global optimization
- Configuration: elimination_rounds, elimination_fraction, batches_per_round
- Streaming candidate processing
- Visualization of tournament progression

Run this example:
    python examples/scripts/07_global_optimization/04_tournament_selection.py
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from nlsq import GlobalOptimizationConfig
from nlsq.global_optimization import (
    TournamentSelector,
    latin_hypercube_sample,
    scale_samples_to_bounds,
)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


if QUICK:
    print("Quick mode: running abbreviated tournament selection demo.")

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def multimodal_model(x, a, b, c):
    """Sinusoidal model with multiple local minima.

    y = a * sin(b * x) + c
    """
    return a * jnp.sin(b * x) + c


def create_data_batch_generator(n_batches=None, batch_size=None, noise_level=0.3):
    """Generator that yields streaming data batches.

    Simulates a streaming scenario where data arrives in batches
    rather than being available all at once.

    Parameters
    ----------
    n_batches : int
        Total number of batches to generate
    batch_size : int
        Number of points per batch
    noise_level : float
        Standard deviation of Gaussian noise

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        (x_batch, y_batch) data pairs
    """
    n_batches = cap_samples(100) if n_batches is None else cap_samples(int(n_batches))
    batch_size = (
        cap_samples(500) if batch_size is None else cap_samples(int(batch_size))
    )

    # True parameters
    true_a, true_b, true_c = 2.5, 1.8, 1.0

    for batch_idx in range(n_batches):
        # Generate random x values for this batch
        x_batch = np.random.uniform(0, 4 * np.pi, batch_size)

        # Generate y values with true parameters + noise
        y_true = true_a * np.sin(true_b * x_batch) + true_c
        noise = noise_level * np.random.randn(batch_size)
        y_batch = y_true + noise

        yield x_batch, y_batch


def main():
    print("=" * 70)
    print("Tournament Selection for Streaming Global Optimization")
    print("=" * 70)
    print()

    # Set random seed for reproducibility
    np.random.seed(42)

    # True parameters
    true_a, true_b, true_c = 2.5, 1.8, 1.0
    true_params = np.array([true_a, true_b, true_c])

    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")
    print()

    # =========================================================================
    # 1. Generate candidate starting points
    # =========================================================================
    print("1. Generating candidate starting points...")

    # Define parameter bounds
    lb = np.array([0.5, 0.5, -2.0])  # Lower bounds: a, b, c
    ub = np.array([5.0, 4.0, 5.0])  # Upper bounds: a, b, c

    # Number of candidate starting points
    n_candidates = 20
    n_params = 3

    # Generate candidates using LHS
    key = jax.random.PRNGKey(42)
    lhs_samples = latin_hypercube_sample(n_candidates, n_params, rng_key=key)
    candidates = scale_samples_to_bounds(lhs_samples, lb, ub)

    print(f"  Generated {n_candidates} candidates in {n_params}D parameter space")
    print("  First 5 candidates:")
    for i in range(5):
        print(
            f"    Candidate {i}: a={candidates[i, 0]:.2f}, b={candidates[i, 1]:.2f}, c={candidates[i, 2]:.2f}"
        )

    # =========================================================================
    # 2. Configure tournament selection
    # =========================================================================
    print()
    print("2. Configuring tournament selection...")

    config = GlobalOptimizationConfig(
        n_starts=n_candidates,
        sampler="lhs",
        elimination_rounds=3,  # 3 elimination rounds
        elimination_fraction=0.5,  # Eliminate 50% each round
        batches_per_round=10,  # Evaluate on 10 batches per round
    )

    print(f"  n_starts:             {config.n_starts}")
    print(f"  elimination_rounds:   {config.elimination_rounds}")
    print(f"  elimination_fraction: {config.elimination_fraction}")
    print(f"  batches_per_round:    {config.batches_per_round}")

    # Calculate expected progression
    expected_survivors = n_candidates
    print("\n  Expected tournament progression:")
    print(f"    Start: {expected_survivors} candidates")
    for r in range(config.elimination_rounds):
        expected_survivors = max(
            1, int(expected_survivors * (1 - config.elimination_fraction))
        )
        print(f"    After round {r + 1}: {expected_survivors} survivors")

    # =========================================================================
    # 3. Run tournament selection
    # =========================================================================
    print()
    print("3. Running tournament selection...")

    selector = TournamentSelector(candidates=candidates, config=config)

    # Need enough batches for all rounds
    total_batches_needed = config.elimination_rounds * config.batches_per_round + 10
    data_gen = create_data_batch_generator(
        n_batches=total_batches_needed, batch_size=500
    )

    # Run tournament and get top candidates
    best_candidates = selector.run_tournament(
        data_batch_iterator=data_gen,
        model=multimodal_model,
        top_m=3,  # Return top 3 candidates
    )

    print("\n  Tournament complete!")
    print("  Top 3 candidates:")
    for i, params in enumerate(best_candidates):
        print(f"    {i + 1}. a={params[0]:.3f}, b={params[1]:.3f}, c={params[2]:.3f}")

    # =========================================================================
    # 4. Tournament diagnostics
    # =========================================================================
    print()
    print("4. Tournament diagnostics:")

    diagnostics = selector.get_diagnostics()

    print(f"  Initial candidates: {diagnostics['n_candidates_initial']}")
    print(f"  Final survivors:    {diagnostics['n_survivors']}")
    print(f"  Elimination rate:   {diagnostics['elimination_rate']:.1%}")
    print(f"  Rounds completed:   {diagnostics['rounds_completed']}")
    print(f"  Total batches:      {diagnostics['total_batches_evaluated']}")
    print(f"  Numerical failures: {diagnostics['numerical_failures']}")

    if diagnostics["mean_survivor_loss"] is not None:
        print(f"  Mean survivor loss: {diagnostics['mean_survivor_loss']:.6f}")

    print()
    print("  Round History:")
    print("  " + "-" * 66)
    print(
        f"  {'Round':<8} {'Before':<10} {'After':<10} {'Eliminated':<12} {'Mean Loss':<12}"
    )
    print("  " + "-" * 66)

    for round_info in diagnostics["round_history"]:
        print(
            f"  {round_info['round']:<8} "
            f"{round_info['n_survivors_before']:<10} "
            f"{round_info['n_survivors_after']:<10} "
            f"{round_info['n_eliminated']:<12} "
            f"{round_info['mean_loss']:.6f}"
        )

    # =========================================================================
    # 5. Save tournament progression visualization
    # =========================================================================
    print()
    print("5. Saving visualizations...")

    # Extract round history for plotting
    rounds = [0] + [r["round"] + 1 for r in diagnostics["round_history"]]
    survivors = [diagnostics["n_candidates_initial"]] + [
        r["n_survivors_after"] for r in diagnostics["round_history"]
    ]
    mean_losses = [None] + [r["mean_loss"] for r in diagnostics["round_history"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Survivor count over rounds
    ax1 = axes[0]
    ax1.plot(rounds, survivors, "bo-", linewidth=2, markersize=10)
    ax1.fill_between(rounds, survivors, alpha=0.3)
    ax1.set_xlabel("Tournament Round")
    ax1.set_ylabel("Number of Candidates")
    ax1.set_title("Tournament Elimination: Candidate Survival")
    ax1.set_xticks(rounds)
    ax1.grid(True, alpha=0.3)

    # Add annotations
    for r, s in zip(rounds, survivors, strict=False):
        ax1.annotate(
            str(s), (r, s), textcoords="offset points", xytext=(0, 10), ha="center"
        )

    # Right: Mean loss evolution
    ax2 = axes[1]
    valid_rounds = [
        r for r, m in zip(rounds, mean_losses, strict=False) if m is not None
    ]
    valid_losses = [m for m in mean_losses if m is not None]

    if valid_losses:
        ax2.plot(valid_rounds, valid_losses, "ro-", linewidth=2, markersize=10)
        ax2.set_xlabel("Tournament Round")
        ax2.set_ylabel("Mean Survivor Loss")
        ax2.set_title("Tournament Elimination: Loss Improvement")
        ax2.set_xticks(valid_rounds)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_tournament_progression.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_tournament_progression.png'}")

    # =========================================================================
    # 6. Save candidate losses visualization
    # =========================================================================
    cumulative_losses = selector.cumulative_losses
    survival_mask = selector.survival_mask

    # Sort candidates by loss for visualization
    sorted_indices = np.argsort(cumulative_losses)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot bars for each candidate
    colors = ["green" if survival_mask[i] else "red" for i in sorted_indices]
    losses_sorted = [cumulative_losses[i] for i in sorted_indices]

    # Cap infinite values for visualization
    max_finite_loss = max(l for l in losses_sorted if np.isfinite(l)) * 1.5
    losses_capped = [min(l, max_finite_loss) for l in losses_sorted]

    ax.bar(range(len(sorted_indices)), losses_capped, color=colors, alpha=0.7)

    ax.set_xlabel("Candidate (sorted by loss)")
    ax.set_ylabel("Cumulative Loss")
    ax.set_title("Tournament Results: Cumulative Loss by Candidate")

    # Add legend
    legend_elements = [
        Patch(facecolor="green", alpha=0.7, label="Survivor"),
        Patch(facecolor="red", alpha=0.7, label="Eliminated"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_candidate_losses.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_candidate_losses.png'}")

    # =========================================================================
    # 7. Compare elimination strategies
    # =========================================================================
    print()
    print("6. Comparing elimination strategies...")

    elimination_fractions = [0.25, 0.5, 0.75]
    comparison_results = {}

    for elim_frac in elimination_fractions:
        # Configure with this elimination fraction
        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=3,
            elimination_fraction=elim_frac,
            batches_per_round=10,
        )

        # Create fresh candidates and selector
        key = jax.random.PRNGKey(42)
        lhs_samples = latin_hypercube_sample(n_candidates, n_params, rng_key=key)
        candidates_fresh = scale_samples_to_bounds(lhs_samples, lb, ub)

        selector = TournamentSelector(candidates=candidates_fresh, config=config)
        data_gen = create_data_batch_generator(n_batches=50, batch_size=500)

        best = selector.run_tournament(
            data_batch_iterator=data_gen,
            model=multimodal_model,
            top_m=1,
        )

        diag = selector.get_diagnostics()

        comparison_results[elim_frac] = {
            "best_params": best[0],
            "n_survivors": diag["n_survivors"],
            "total_batches": diag["total_batches_evaluated"],
            "round_history": diag["round_history"],
        }

        print(f"\n  elimination_fraction = {elim_frac}:")
        print(f"    Survivors: {diag['n_survivors']}")
        print(f"    Batches evaluated: {diag['total_batches_evaluated']}")
        print(
            f"    Best params: a={best[0][0]:.3f}, b={best[0][1]:.3f}, c={best[0][2]:.3f}"
        )

    # Visualize the comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    plot_colors = ["blue", "green", "orange"]

    # Plot survivor progression for each strategy
    ax1 = axes[0]
    for (elim_frac, data), color in zip(
        comparison_results.items(), plot_colors, strict=False
    ):
        rounds_plot = [0] + [r["round"] + 1 for r in data["round_history"]]
        survivors_plot = [n_candidates] + [
            r["n_survivors_after"] for r in data["round_history"]
        ]
        ax1.plot(
            rounds_plot,
            survivors_plot,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"elim_frac={elim_frac}",
        )

    ax1.set_xlabel("Tournament Round")
    ax1.set_ylabel("Survivors")
    ax1.set_title("Survivor Count by Elimination Fraction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart: Total batches evaluated
    ax2 = axes[1]
    fracs = list(comparison_results.keys())
    batches = [comparison_results[f]["total_batches"] for f in fracs]
    ax2.bar([str(f) for f in fracs], batches, color=plot_colors)
    ax2.set_xlabel("Elimination Fraction")
    ax2.set_ylabel("Total Batches Evaluated")
    ax2.set_title("Computational Cost")

    # Bar chart: Parameter error
    ax3 = axes[2]
    errors = []
    for f in fracs:
        best_p = comparison_results[f]["best_params"]
        error = np.linalg.norm(best_p - true_params)
        errors.append(error)

    ax3.bar([str(f) for f in fracs], errors, color=plot_colors)
    ax3.set_xlabel("Elimination Fraction")
    ax3.set_ylabel("Parameter Error (L2)")
    ax3.set_title("Best Candidate Accuracy")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_elimination_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR / '04_elimination_comparison.png'}")

    # =========================================================================
    # 8. Demonstrate checkpointing
    # =========================================================================
    print()
    print("7. Demonstrating checkpointing...")

    config = GlobalOptimizationConfig(
        n_starts=n_candidates,
        elimination_rounds=3,
        elimination_fraction=0.5,
        batches_per_round=10,
    )

    key = jax.random.PRNGKey(42)
    lhs_samples = latin_hypercube_sample(n_candidates, n_params, rng_key=key)
    candidates_ckpt = scale_samples_to_bounds(lhs_samples, lb, ub)

    selector = TournamentSelector(candidates=candidates_ckpt, config=config)

    # Save checkpoint
    checkpoint = selector.to_checkpoint()

    print("  Checkpoint contents:")
    for ckpt_key, value in checkpoint.items():
        if isinstance(value, np.ndarray):
            print(f"    {ckpt_key}: ndarray shape={value.shape}")
        elif isinstance(value, list):
            print(f"    {ckpt_key}: list length={len(value)}")
        else:
            print(f"    {ckpt_key}: {value}")

    # Restore from checkpoint
    restored_selector = TournamentSelector.from_checkpoint(checkpoint, config)

    print("\n  Restored selector:")
    print(f"    n_candidates: {restored_selector.n_candidates}")
    print(f"    n_survivors:  {restored_selector.n_survivors}")
    print(f"    current_round: {restored_selector.current_round}")

    # =========================================================================
    # Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")
    print()
    print("Tournament selection is ideal for:")
    print("  - Large datasets that exceed memory")
    print("  - Streaming data scenarios")
    print("  - High-dimensional parameter spaces")
    print()
    print("Key parameters in GlobalOptimizationConfig:")
    print("  - elimination_rounds: 2-4 (more = more filtering)")
    print("  - elimination_fraction: 0.25-0.75 (higher = faster)")
    print("  - batches_per_round: 10-100 (more = better ranking)")
    print()
    print("TournamentSelector methods:")
    print("  - run_tournament(): Execute full tournament")
    print("  - get_diagnostics(): Get detailed statistics")
    print("  - to_checkpoint() / from_checkpoint(): Fault tolerance")


if __name__ == "__main__":
    main()
