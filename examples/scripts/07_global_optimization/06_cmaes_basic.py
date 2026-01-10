#!/usr/bin/env python
"""CMA-ES Basic Usage Example.

This example demonstrates basic CMA-ES (Covariance Matrix Adaptation
Evolution Strategy) usage for global optimization in NLSQ.

CMA-ES is a gradient-free evolutionary algorithm particularly effective for:
- Multi-scale parameter problems (parameters spanning many orders of magnitude)
- Complex fitness landscapes with multiple local minima
- Problems where gradient information is unreliable or unavailable

Prerequisites:
    pip install "nlsq[global]"  # Installs evosax dependency
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

# Check if evosax is available
from nlsq.global_optimization import is_evosax_available

if not is_evosax_available():
    print("evosax is not installed. Install with: pip install 'nlsq[global]'")
    sys.exit(1)

from nlsq.global_optimization import CMAESConfig, CMAESOptimizer


def main():
    """Demonstrate basic CMA-ES usage."""
    print("=" * 60)
    print("CMA-ES Basic Usage Example")
    print("=" * 60)

    # Define a simple exponential decay model
    def model(x, a, b):
        """Exponential decay: y = a * exp(-b * x)"""
        return a * jnp.exp(-b * x)

    # Generate synthetic data
    np.random.seed(42)
    x = jnp.linspace(0, 5, 100)
    true_params = [2.5, 0.5]
    y_true = model(x, *true_params)
    noise = 0.05 * np.random.randn(len(x))
    y = y_true + noise

    # Define parameter bounds (required for CMA-ES)
    bounds = ([0.1, 0.01], [10.0, 2.0])

    print("\n1. Basic CMA-ES with default configuration")
    print("-" * 40)

    # Create optimizer with default config (BIPOP enabled)
    config = CMAESConfig(
        max_generations=100,
        seed=42,
    )
    optimizer = CMAESOptimizer(config=config)

    # Run optimization
    result = optimizer.fit(model, x, y, bounds=bounds)

    print(f"True parameters: a={true_params[0]}, b={true_params[1]}")
    print(f"Fitted parameters: a={result['popt'][0]:.4f}, b={result['popt'][1]:.4f}")
    print(
        f"Diagnostics: {result['cmaes_diagnostics']['total_generations']} generations"
    )

    print("\n2. Using presets")
    print("-" * 40)

    # Fast preset (no restarts, fewer generations)
    print("\n'cmaes-fast' preset:")
    optimizer_fast = CMAESOptimizer.from_preset("cmaes-fast")
    result_fast = optimizer_fast.fit(model, x, y, bounds=bounds)
    print(f"  Fitted: a={result_fast['popt'][0]:.4f}, b={result_fast['popt'][1]:.4f}")
    print(f"  Generations: {result_fast['cmaes_diagnostics']['total_generations']}")

    # Global preset (more generations, larger population)
    print("\n'cmaes-global' preset:")
    optimizer_global = CMAESOptimizer.from_preset("cmaes-global")
    result_global = optimizer_global.fit(model, x, y, bounds=bounds)
    print(
        f"  Fitted: a={result_global['popt'][0]:.4f}, b={result_global['popt'][1]:.4f}"
    )
    print(f"  Generations: {result_global['cmaes_diagnostics']['total_generations']}")

    print("\n3. Custom configuration")
    print("-" * 40)

    custom_config = CMAESConfig(
        popsize=20,  # Custom population size
        max_generations=50,  # Max generations per run
        sigma=0.3,  # Initial step size
        restart_strategy="bipop",  # Use BIPOP restarts
        max_restarts=3,  # Limit restarts
        refine_with_nlsq=True,  # Refine with Trust Region for pcov
        seed=123,  # For reproducibility
    )
    optimizer_custom = CMAESOptimizer(config=custom_config)
    result_custom = optimizer_custom.fit(model, x, y, bounds=bounds)

    print(
        f"Custom config result: a={result_custom['popt'][0]:.4f}, b={result_custom['popt'][1]:.4f}"
    )
    print(f"Restarts: {result_custom['cmaes_diagnostics']['total_restarts']}")

    print("\n4. Examining diagnostics")
    print("-" * 40)

    diag = result["cmaes_diagnostics"]
    print(f"Total generations: {diag['total_generations']}")
    print(f"Total restarts: {diag['total_restarts']}")
    print(f"Final sigma: {diag['final_sigma']:.6e}")
    print(f"Best fitness (neg SSR): {diag['best_fitness']:.6e}")
    print(f"Convergence reason: {diag['convergence_reason']}")
    print(f"NLSQ refinement: {diag['nlsq_refinement']}")
    print(f"Wall time: {diag['wall_time']:.3f}s")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
