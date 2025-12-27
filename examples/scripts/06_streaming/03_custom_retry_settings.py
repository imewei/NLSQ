"""
Converted from 03_custom_retry_settings.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Example 3: Custom Retry Settings for Noisy Data
#
# This example demonstrates how to configure retry strategies and success rate
# thresholds for datasets with high failure rates or very noisy data.
#
# Features demonstrated:
# - Configurable success rate thresholds
# - Adaptive retry strategies
# - Error type analysis
# - Retry count tracking
#
# Run this example:
#     python examples/streaming/03_custom_retry_settings.py
#
# ======================================================================

import jax.numpy as jnp
import numpy as np

from nlsq import StreamingConfig, StreamingOptimizer


def noisy_exponential(x, a, b):
    """Exponential model with potential numerical issues"""
    # This model can produce NaN/Inf for certain parameter combinations
    return a * jnp.exp(-b * x)


def inject_noise_into_data(x_data, y_data, noise_rate=0.1):
    """Inject NaN values into data to simulate noisy sensors"""
    n_samples = len(y_data)
    n_corrupted = int(n_samples * noise_rate)
    corrupt_indices = np.random.choice(n_samples, n_corrupted, replace=False)
    # Convert to numpy array to allow in-place modification
    y_corrupted = np.array(y_data, copy=True)
    y_corrupted[corrupt_indices] = np.nan
    return y_corrupted


def main():
    print("=" * 70)
    print("Streaming Optimizer: Custom Retry Settings Example")
    print("=" * 70)
    print()

    # Generate synthetic data with noise
    np.random.seed(42)
    n_samples = 10000
    x_data = np.linspace(0, 10, n_samples)
    true_a, true_b = 2.5, 0.3
    y_true = noisy_exponential(x_data, true_a, true_b)
    y_data = y_true + 0.1 * np.random.randn(n_samples)

    # Inject corrupted data (10% NaN values)
    noise_rate = 0.10
    y_corrupted = inject_noise_into_data(x_data, y_data, noise_rate)
    n_corrupted = np.sum(np.isnan(y_corrupted))

    print(f"Dataset: {n_samples} samples")
    print(f"Corrupted samples: {n_corrupted} ({n_corrupted / n_samples:.1%})")
    print(f"True parameters: a={true_a}, b={true_b}")
    print()

    # Part 1: Standard settings (may fail due to low success rate)
    print("PART 1: Standard Settings (min_success_rate=0.5)")
    print("=" * 70)

    config_standard = StreamingConfig(
        batch_size=100,
        max_epochs=5,
        learning_rate=0.001,
        enable_fault_tolerance=True,
        validate_numerics=True,
        min_success_rate=0.5,  # Require 50% success (standard)
        max_retries_per_batch=2,  # Standard retry limit
    )

    print("Configuration:")
    print(f"  Min success rate: {config_standard.min_success_rate:.0%}")
    print(f"  Max retries per batch: {config_standard.max_retries_per_batch}")
    print()

    optimizer1 = StreamingOptimizer(config_standard)
    p0 = np.array([1.0, 0.1])

    print("Starting optimization with standard settings...")
    result1 = optimizer1.fit(
        (x_data, y_corrupted),
        noisy_exponential,
        p0,
        verbose=0,  # Silent mode
    )

    diag1 = result1["streaming_diagnostics"]
    print(f"Result: {'SUCCESS' if result1['success'] else 'FAILED'}")
    print(f"Message: {result1['message']}")
    print(f"Batch success rate: {diag1['batch_success_rate']:.1%}")
    print(
        f"Failed batches: {len(diag1['failed_batches'])}/{diag1['total_batches_attempted']}"
    )
    print(f"Total retries: {diag1['total_retries']}")
    print()

    # Part 2: Permissive settings (allow more failures)
    print("PART 2: Permissive Settings (min_success_rate=0.3)")
    print("=" * 70)

    config_permissive = StreamingConfig(
        batch_size=100,
        max_epochs=10,
        learning_rate=0.001,
        enable_fault_tolerance=True,
        validate_numerics=True,
        min_success_rate=0.3,  # Allow 70% failures (very permissive)
        max_retries_per_batch=2,  # Standard retry limit
    )

    print("Configuration:")
    print(f"  Min success rate: {config_permissive.min_success_rate:.0%} (permissive)")
    print(f"  Max retries per batch: {config_permissive.max_retries_per_batch}")
    print()

    optimizer2 = StreamingOptimizer(config_permissive)

    print("Starting optimization with permissive settings...")
    result2 = optimizer2.fit(
        (x_data, y_corrupted),
        noisy_exponential,
        p0,
        verbose=1,
    )

    print()
    diag2 = result2["streaming_diagnostics"]
    print(f"Result: {'SUCCESS' if result2['success'] else 'FAILED'}")
    print(f"Message: {result2['message']}")
    print(f"Batch success rate: {diag2['batch_success_rate']:.1%}")
    print(
        f"Failed batches: {len(diag2['failed_batches'])}/{diag2['total_batches_attempted']}"
    )
    print(f"Total retries: {diag2['total_retries']}")
    print()

    # Part 3: Analyze error types and retry patterns
    print("PART 3: Error Analysis")
    print("=" * 70)

    print("Error Type Distribution:")
    for error_type, count in diag2["error_types"].items():
        pct = count / sum(diag2["error_types"].values()) * 100
        print(f"  {error_type}: {count} ({pct:.1f}%)")
    print()

    print("Retry Statistics:")
    if diag2["retry_counts"]:
        retry_counts_list = list(diag2["retry_counts"].values())
        print(f"  Batches with retries: {len(retry_counts_list)}")
        print(f"  Average retries per failed batch: {np.mean(retry_counts_list):.2f}")
        print(f"  Max retries for single batch: {max(retry_counts_list)}")
    else:
        print("  No retries attempted")
    print()

    # Display final results
    print("FINAL RESULTS")
    print("=" * 70)
    if result2["success"]:
        best_params = result2["x"]
        print("Best parameters:")
        print(f"  a = {best_params[0]:.6f} (true: {true_a})")
        print(f"  b = {best_params[1]:.6f} (true: {true_b})")
        print(f"  Best loss = {result2['best_loss']:.6e}")
        print()

        # Parameter errors
        param_errors = np.abs(best_params - np.array([true_a, true_b]))
        rel_errors = param_errors / np.array([true_a, true_b]) * 100
        print("Parameter errors:")
        print(f"  a error: {param_errors[0]:.6f} ({rel_errors[0]:.2f}%)")
        print(f"  b error: {param_errors[1]:.6f} ({rel_errors[1]:.2f}%)")
    else:
        print("Optimization failed - success rate too low")
        print(f"Best parameters found (may be suboptimal): {result2['x']}")
    print()

    # Aggregate statistics
    agg = diag2["aggregate_stats"]
    print("Aggregate Statistics (successful batches only):")
    print(f"  Mean loss: {agg['mean_loss']:.6e}")
    print(f"  Std loss: {agg['std_loss']:.6e}")
    print(f"  Mean gradient norm: {agg['mean_grad_norm']:.6f}")
    print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key takeaways:")
    print("  - min_success_rate controls acceptable failure rate")
    print("  - Permissive settings (0.3-0.5) good for noisy data")
    print("  - Strict settings (0.7-0.9) good for clean data")
    print("  - Retry strategies adapt to error types automatically")
    print("  - Best parameters returned even when success rate fails")
    print("  - Error type distribution helps diagnose data issues")

    # Cleanup checkpoint worker threads to prevent memory corruption on exit
    for opt in [optimizer1, optimizer2]:
        if hasattr(opt, "_shutdown_checkpoint_worker"):
            opt._shutdown_checkpoint_worker()


if __name__ == "__main__":
    main()
