#!/usr/bin/env python3
"""
Demonstration of Hybrid Streaming Optimizer API Integration

This example shows how to use method='hybrid_streaming' with both
curve_fit() and curve_fit_large() functions.

The hybrid streaming optimizer provides:
- Parameter normalization for better gradient signals
- Adam warmup for robust initial convergence
- Streaming Gauss-Newton for exact covariance computation
- Automatic memory management for large datasets
"""

import numpy as np
import jax.numpy as jnp
from nlsq import curve_fit, curve_fit_large


def exponential_decay(x, a, b, c):
    """Three-parameter exponential decay model."""
    return a * jnp.exp(-b * x) + c


def main():
    print("=" * 70)
    print("Hybrid Streaming Optimizer API Demo")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(0, 10, 2000)
    true_params = np.array([5.0, 0.5, 1.0])
    y_true = exponential_decay(x, *true_params)
    y = y_true + np.random.normal(0, 0.1, len(x))

    # Example 1: Basic usage with curve_fit()
    print("Example 1: Basic usage with curve_fit()")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method='hybrid_streaming',
        verbose=1,
    )

    # Unpack result
    popt, pcov = result

    print(f"\nFitted parameters: {popt}")
    print(f"True parameters:   {true_params}")
    print(f"Parameter errors:  {np.abs(popt - true_params)}")
    print(f"\nCovariance matrix diagonal: {np.diag(pcov)}")
    print(f"Parameter std errors: {np.sqrt(np.diag(pcov))}")
    print()

    # Example 2: With parameter bounds
    print("Example 2: With parameter bounds")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        bounds=([0, 0, 0], [10, 2, 5]),
        method='hybrid_streaming',
        verbose=0,  # Silent mode
    )

    popt, pcov = result
    print(f"Fitted parameters (bounded): {popt}")
    print(f"Within bounds: {np.all(popt >= [0, 0, 0]) and np.all(popt <= [10, 2, 5])}")
    print()

    # Example 3: Large dataset with curve_fit_large()
    print("Example 3: Large dataset with curve_fit_large()")
    print("-" * 70)

    # Generate larger dataset
    x_large = np.linspace(0, 10, 10000)
    y_large = exponential_decay(x_large, *true_params) + np.random.normal(0, 0.1, len(x_large))

    popt, pcov = curve_fit_large(
        exponential_decay,
        x_large,
        y_large,
        p0=np.array([4.0, 0.4, 0.8]),
        method='hybrid_streaming',
        verbose=1,
    )

    print(f"\nFitted parameters (large dataset): {popt}")
    print(f"True parameters:                   {true_params}")
    print(f"Parameter errors:                  {np.abs(popt - true_params)}")
    print()

    # Example 4: Config overrides via kwargs
    print("Example 4: Custom configuration via kwargs")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method='hybrid_streaming',
        verbose=0,
        # HybridStreamingConfig overrides:
        warmup_iterations=300,
        normalization_strategy='p0',  # 'bounds' requires explicit bounds
        phase2_max_iterations=100,
    )

    popt, pcov = result
    print(f"Fitted parameters (custom config): {popt}")
    print(f"Result attributes available: {list(result.keys())[:10]}")
    print()

    # Example 5: Accessing full result details
    print("Example 5: Accessing full result details")
    print("-" * 70)

    result = curve_fit(
        exponential_decay,
        x,
        y,
        p0=np.array([4.0, 0.4, 0.8]),
        method='hybrid_streaming',
        verbose=0,
    )

    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Final cost: {result.get('cost', 'N/A')}")

    if 'streaming_diagnostics' in result:
        diag = result['streaming_diagnostics']
        print(f"\nStreaming diagnostics available:")
        print(f"  Keys: {list(diag.keys())}")

    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
