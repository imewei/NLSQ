#!/usr/bin/env python
"""CMA-ES Multi-Scale Parameter Fitting Example.

This example demonstrates CMA-ES for fitting models with parameters spanning
many orders of magnitude - a scenario where traditional gradient-based
optimizers often struggle.

CMA-ES excels at multi-scale optimization because:
1. It adapts the covariance matrix to the local geometry
2. The sigmoid bound transformation normalizes all parameters to [-1, 1]
3. BIPOP restarts help escape local minima in complex landscapes

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

from nlsq.global_optimization import (
    CMAESConfig,
    CMAESDiagnostics,
    CMAESOptimizer,
    MethodSelector,
)


def main():
    """Demonstrate multi-scale parameter fitting with CMA-ES."""
    print("=" * 60)
    print("CMA-ES Multi-Scale Parameter Fitting")
    print("=" * 60)

    # Multi-scale model: parameters span 6+ orders of magnitude
    def diffusion_model(x, D0, gamma0, n):
        """Diffusion model: D = D0 * (1 + (x / gamma0)^n).

        Parameters:
        - D0: ~1e-10 (diffusion coefficient in m^2/s)
        - gamma0: ~1e-3 (critical shear rate in 1/s)
        - n: ~0.5 (power law exponent)

        Scale ratio: ~1e7 (7 orders of magnitude)
        """
        return D0 * (1.0 + jnp.power(x / gamma0, n))

    # Generate synthetic data
    np.random.seed(42)
    x = jnp.logspace(-1, 3, 50)  # Shear rates from 0.1 to 1000 1/s

    # True parameters (span 7 orders of magnitude)
    true_D0 = 1e-10  # m^2/s
    true_gamma0 = 1e-3  # 1/s
    true_n = 0.5  # dimensionless

    y_true = diffusion_model(x, true_D0, true_gamma0, true_n)
    noise = 0.02 * y_true * np.random.randn(len(x))
    y = y_true + noise

    # Bounds spanning the expected parameter ranges
    bounds = (
        [1e-12, 1e-5, 0.1],  # Lower bounds
        [1e-8, 1e-1, 2.0],  # Upper bounds
    )

    print("\n1. Check scale ratio with MethodSelector")
    print("-" * 40)

    selector = MethodSelector()
    scale_ratio = selector.compute_scale_ratio(bounds[0], bounds[1])
    print(f"Scale ratio: {scale_ratio:.0f}x")
    print(f"Threshold for CMA-ES: {selector.scale_threshold:.0f}x")

    if scale_ratio > selector.scale_threshold:
        print("=> Multi-scale problem detected, CMA-ES recommended")
    else:
        print("=> Standard optimization sufficient")

    print("\n2. Fit with CMA-ES")
    print("-" * 40)

    config = CMAESConfig(
        max_generations=100,
        restart_strategy="bipop",
        max_restarts=5,
        seed=42,
        refine_with_nlsq=True,
    )
    optimizer = CMAESOptimizer(config=config)

    result = optimizer.fit(diffusion_model, x, y, bounds=bounds)

    print("\nTrue parameters:")
    print(f"  D0     = {true_D0:.2e} m^2/s")
    print(f"  gamma0 = {true_gamma0:.2e} 1/s")
    print(f"  n      = {true_n:.2f}")

    print("\nFitted parameters:")
    popt = result["popt"]
    print(
        f"  D0     = {popt[0]:.2e} m^2/s (error: {abs(popt[0] - true_D0) / true_D0 * 100:.1f}%)"
    )
    print(
        f"  gamma0 = {popt[1]:.2e} 1/s (error: {abs(popt[1] - true_gamma0) / true_gamma0 * 100:.1f}%)"
    )
    print(
        f"  n      = {popt[2]:.2f} (error: {abs(popt[2] - true_n) / true_n * 100:.1f}%)"
    )

    print("\n3. Analyze convergence")
    print("-" * 40)

    diag = result["cmaes_diagnostics"]
    print(f"Total generations: {diag['total_generations']}")
    print(f"Total restarts: {diag['total_restarts']}")
    print(f"Final sigma: {diag['final_sigma']:.6e}")
    print(f"Convergence reason: {diag['convergence_reason']}")
    print(f"Wall time: {diag['wall_time']:.3f}s")

    # Check fitness improvement
    fitness_hist = diag["fitness_history"]
    if len(fitness_hist) >= 2:
        improvement = (fitness_hist[-1] - fitness_hist[0]) / abs(fitness_hist[0])
        print(f"Fitness improvement: {improvement * 100:.1f}%")

    print("\n4. Restart history")
    print("-" * 40)

    for i, restart in enumerate(diag["restart_history"]):
        print(
            f"  Restart {i + 1}: popsize={restart['popsize']}, "
            f"generations={restart['generations']}, "
            f"best_fitness={restart['best_fitness']:.2e}"
        )

    print("\n5. Using curve_fit with method='cmaes'")
    print("-" * 40)

    from nlsq import curve_fit

    # Direct curve_fit integration
    result_cf = curve_fit(
        diffusion_model,
        x,
        y,
        bounds=bounds,
        method="cmaes",  # Explicitly request CMA-ES
    )

    print(
        f"curve_fit result: D0={result_cf.x[0]:.2e}, gamma0={result_cf.x[1]:.2e}, n={result_cf.x[2]:.2f}"
    )
    print(f"Message: {result_cf.message}")

    print("\n6. Scale invariance demonstration")
    print("-" * 40)

    # CMA-ES should give similar results regardless of parameter magnitudes
    # because the sigmoid transformation normalizes everything

    # Rescaled model (multiply D0 by 1e10)
    def rescaled_model(x, D0_scaled, gamma0, n):
        D0 = D0_scaled * 1e-10
        return D0 * (1.0 + jnp.power(x / gamma0, n))

    rescaled_bounds = (
        [0.01, 1e-5, 0.1],
        [100.0, 1e-1, 2.0],
    )

    config_scaled = CMAESConfig(max_generations=100, seed=42)
    optimizer_scaled = CMAESOptimizer(config=config_scaled)
    result_scaled = optimizer_scaled.fit(rescaled_model, x, y, bounds=rescaled_bounds)

    print(f"Original D0 fit: {popt[0]:.2e}")
    print(f"Rescaled D0 fit: {result_scaled['popt'][0] * 1e-10:.2e}")

    print("\n" + "=" * 60)
    print("Multi-scale fitting completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
