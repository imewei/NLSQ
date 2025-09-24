#!/usr/bin/env python3
"""
Demonstration of NLSQ Large Dataset Fitting Capabilities

This example shows how to use the LargeDatasetFitter class to efficiently
fit curve parameters to very large datasets with automatic memory management.
"""

import time
import numpy as np
import jax.numpy as jnp
from nlsq import LargeDatasetFitter, fit_large_dataset, estimate_memory_requirements


def exponential_decay(x, a, b, c):
    """Exponential decay model with offset: y = a * exp(-b * x) + c"""
    return a * jnp.exp(-b * x) + c


def polynomial_model(x, a, b, c, d):
    """Polynomial model: y = a*x^3 + b*x^2 + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d


def demo_memory_estimation():
    """Demonstrate memory estimation capabilities."""
    print("="*60)
    print("MEMORY ESTIMATION DEMO")
    print("="*60)

    # Estimate requirements for different dataset sizes
    test_cases = [
        (100_000, 3, "Small dataset"),
        (1_000_000, 3, "Medium dataset"),
        (10_000_000, 3, "Large dataset"),
        (50_000_000, 3, "Very large dataset"),
        (100_000_000, 3, "Extremely large dataset"),
    ]

    for n_points, n_params, description in test_cases:
        stats = estimate_memory_requirements(n_points, n_params)

        print(f"\n{description} ({n_points:,} points, {n_params} parameters):")
        print(f"  Memory estimate: {stats.total_memory_estimate_gb:.2f} GB")
        print(f"  Chunk size: {stats.recommended_chunk_size:,}")
        print(f"  Number of chunks: {stats.n_chunks}")

        if stats.requires_sampling:
            print(f"  Strategy: Sampling recommended")
        elif stats.n_chunks == 1:
            print(f"  Strategy: Single chunk (fits in memory)")
        else:
            print(f"  Strategy: Chunked processing")


def demo_basic_large_dataset_fitting():
    """Demonstrate basic large dataset fitting."""
    print("\n" + "="*60)
    print("BASIC LARGE DATASET FITTING DEMO")
    print("="*60)

    # Generate synthetic large dataset (1M points)
    print("Generating 1M point exponential decay dataset...")
    np.random.seed(42)
    n_points = 1_000_000
    x_data = np.linspace(0, 5, n_points, dtype=np.float64)
    true_params = [5.0, 1.2, 0.5]
    noise_level = 0.05

    y_true = true_params[0] * np.exp(-true_params[1] * x_data) + true_params[2]
    y_data = y_true + np.random.normal(0, noise_level, n_points)

    print(f"Dataset: {n_points:,} points")
    print(f"True parameters: a={true_params[0]}, b={true_params[1]}, c={true_params[2]}")

    # Fit using convenience function
    print("\nFitting with automatic memory management...")
    start_time = time.time()

    result = fit_large_dataset(
        exponential_decay, x_data, y_data,
        p0=[4.0, 1.0, 0.4],
        memory_limit_gb=2.0,  # 2GB limit
        show_progress=True
    )

    fit_time = time.time() - start_time

    if result.success:
        fitted_params = np.array(result.popt)
        errors = np.abs(fitted_params - np.array(true_params))
        rel_errors = errors / np.array(true_params) * 100

        print(f"\n✅ Fit completed in {fit_time:.2f} seconds")
        print(f"Fitted parameters: [{fitted_params[0]:.3f}, {fitted_params[1]:.3f}, {fitted_params[2]:.3f}]")
        print(f"Absolute errors: [{errors[0]:.4f}, {errors[1]:.4f}, {errors[2]:.4f}]")
        print(f"Relative errors: [{rel_errors[0]:.2f}%, {rel_errors[1]:.2f}%, {rel_errors[2]:.2f}%]")
    else:
        print(f"❌ Fit failed: {result.message}")


def demo_chunked_processing():
    """Demonstrate chunked processing with progress reporting."""
    print("\n" + "="*60)
    print("CHUNKED PROCESSING DEMO")
    print("="*60)

    # Generate a dataset that will require chunking
    print("Generating 2M point polynomial dataset...")
    np.random.seed(123)
    n_points = 2_000_000
    x_data = np.linspace(-2, 2, n_points, dtype=np.float64)
    true_params = [0.5, -1.2, 2.0, 1.5]
    noise_level = 0.1

    y_true = (true_params[0] * x_data**3 +
              true_params[1] * x_data**2 +
              true_params[2] * x_data +
              true_params[3])
    y_data = y_true + np.random.normal(0, noise_level, n_points)

    print(f"Dataset: {n_points:,} points")
    print(f"True parameters: {true_params}")

    # Create fitter with limited memory to force chunking
    fitter = LargeDatasetFitter(memory_limit_gb=0.5)  # Small limit to force chunking

    # Get processing recommendations
    recs = fitter.get_memory_recommendations(n_points, 4)
    print(f"\nProcessing strategy: {recs['processing_strategy']}")
    print(f"Chunk size: {recs['recommendations']['chunk_size']:,}")
    print(f"Number of chunks: {recs['recommendations']['n_chunks']}")
    print(f"Memory estimate: {recs['recommendations']['total_memory_estimate_gb']:.2f} GB")

    # Fit with progress reporting
    print("\nFitting with chunked processing...")
    start_time = time.time()

    result = fitter.fit_with_progress(
        polynomial_model, x_data, y_data,
        p0=[0.4, -1.0, 1.8, 1.2]
    )

    fit_time = time.time() - start_time

    if result.success:
        fitted_params = np.array(result.popt)
        errors = np.abs(fitted_params - np.array(true_params))
        rel_errors = errors / np.abs(np.array(true_params)) * 100

        print(f"\n✅ Chunked fit completed in {fit_time:.2f} seconds")
        print(f"Used {result.n_chunks} chunks with {result.success_rate:.1%} success rate")
        print(f"Fitted parameters: {fitted_params}")
        print(f"Absolute errors: {errors}")
        print(f"Relative errors: {rel_errors}%")
    else:
        print(f"❌ Chunked fit failed: {result.message}")


def demo_sampling_strategy():
    """Demonstrate sampling for extremely large datasets."""
    print("\n" + "="*60)
    print("SAMPLING STRATEGY DEMO")
    print("="*60)

    # Simulate a very large dataset scenario
    print("Simulating extremely large dataset (100M points)...")
    n_points_full = 100_000_000  # 100M points
    true_params = [3.0, 0.8, 0.2]

    # For demo purposes, generate a smaller representative sample
    # In practice, you would have this data already or stream it
    np.random.seed(456)
    n_sample = 1_000_000  # 1M sample for demo
    x_sample = np.sort(np.random.uniform(0, 5, n_sample))
    y_sample = (true_params[0] * np.exp(-true_params[1] * x_sample) +
                true_params[2] + np.random.normal(0, 0.05, n_sample))

    print(f"Full dataset size: {n_points_full:,} points (simulated)")
    print(f"Demo sample size: {n_sample:,} points")
    print(f"True parameters: {true_params}")

    # Check memory requirements for full dataset
    stats = estimate_memory_requirements(n_points_full, 3)
    print(f"\nFull dataset memory estimate: {stats.total_memory_estimate_gb:.2f} GB")
    print(f"Sampling recommended: {stats.requires_sampling}")

    # Create fitter with sampling enabled
    from nlsq.large_dataset import LDMemoryConfig
    config = LDMemoryConfig(memory_limit_gb=4.0, enable_sampling=True)
    fitter = LargeDatasetFitter(config=config)

    print("\nFitting with sampling strategy...")
    start_time = time.time()

    # For demo, use our sample as if it were the full dataset
    result = fitter.fit(exponential_decay, x_sample, y_sample, p0=[2.5, 1.0, 0.1])

    fit_time = time.time() - start_time

    if result.success:
        fitted_params = np.array(result.popt)
        errors = np.abs(fitted_params - np.array(true_params))
        rel_errors = errors / np.array(true_params) * 100

        print(f"\n✅ Sampling fit completed in {fit_time:.2f} seconds")
        print(f"Fitted parameters: {fitted_params}")
        print(f"Absolute errors: {errors}")
        print(f"Relative errors: {rel_errors}%")

        if hasattr(result, 'was_sampled') and result.was_sampled:
            print(f"Used sampling: {result.sample_size:,} points from {result.original_size:,}")
    else:
        print(f"❌ Sampling fit failed: {result.message}")


def main():
    """Run all demonstration examples."""
    print("NLSQ Large Dataset Fitting Demonstration")
    print("=========================================")
    print("This demo shows the capabilities of NLSQ for handling very large datasets")
    print("with automatic memory management, chunking, and sampling strategies.\n")

    # Run all demos
    demo_memory_estimation()
    demo_basic_large_dataset_fitting()
    demo_chunked_processing()
    demo_sampling_strategy()

    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)
    print("\nKey takeaways:")
    print("• NLSQ automatically handles memory management for large datasets")
    print("• Chunked processing works for datasets that don't fit in memory")
    print("• Sampling strategies can handle extremely large datasets efficiently")
    print("• Progress reporting helps track long-running fits")
    print("• Memory estimation helps plan processing strategies")


if __name__ == "__main__":
    main()