"""
Converted from 01_basic_fault_tolerance.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Example 1: Basic Fault Tolerance with Streaming Optimizer
#
# This example demonstrates the basic usage of the streaming optimizer with
# fault tolerance enabled (default behavior).
#
# Features demonstrated:
# - Automatic best parameter tracking
# - NaN/Inf detection at three validation points
# - Adaptive retry strategies for failed batches
# - Success rate validation
# - Detailed diagnostics
#
# Run this example:
#     python examples/streaming/01_basic_fault_tolerance.py
#
# ======================================================================

import jax.numpy as jnp
import numpy as np

from nlsq import StreamingConfig, StreamingOptimizer


def exponential_decay(x, a, b):
    """Exponential decay model: y = a * exp(-b * x)"""
    return a * jnp.exp(-b * x)


def main():
    print("=" * 70)
    print("Streaming Optimizer: Basic Fault Tolerance Example")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    x_data = np.linspace(0, 10, n_samples)
    true_a, true_b = 2.5, 0.3
    y_true = exponential_decay(x_data, true_a, true_b)
    y_data = y_true + 0.1 * np.random.randn(n_samples)

    print(f"Dataset: {n_samples} samples")
    print(f"True parameters: a={true_a}, b={true_b}")
    print()

    # Configure optimizer with fault tolerance (default)
    config = StreamingConfig(
        batch_size=100,
        max_epochs=10,
        learning_rate=0.001,
        # Fault tolerance settings (all defaults)
        enable_fault_tolerance=True,  # Enable fault tolerance features
        validate_numerics=True,  # Check for NaN/Inf
        min_success_rate=0.5,  # Require 50% batch success
        max_retries_per_batch=2,  # Max 2 retry attempts
        # Checkpoint settings
        checkpoint_dir="checkpoints",
        checkpoint_frequency=100,  # Save every 100 iterations
        enable_checkpoints=True,
    )

    print("Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max epochs: {config.max_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Fault tolerance: {config.enable_fault_tolerance}")
    print(f"  Validate numerics: {config.validate_numerics}")
    print(f"  Min success rate: {config.min_success_rate:.0%}")
    print(f"  Max retries per batch: {config.max_retries_per_batch}")
    print()

    # Create optimizer
    optimizer = StreamingOptimizer(config)

    # Initial guess (deliberately poor to show convergence)
    p0 = np.array([1.0, 0.1])
    print(f"Initial guess: a={p0[0]}, b={p0[1]}")
    print()

    # Fit with automatic error handling
    print("Starting optimization...")
    print("-" * 70)
    result = optimizer.fit(
        (x_data, y_data),  # Data as tuple
        exponential_decay,  # Model function
        p0,  # Initial parameters
        verbose=1,  # Show progress
    )
    print("-" * 70)
    print()

    # Extract results
    best_params = result["x"]
    success = result["success"]
    message = result["message"]
    best_loss = result["best_loss"]
    diagnostics = result["streaming_diagnostics"]

    # Display results
    print("RESULTS")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Message: {message}")
    print()
    print("Best parameters found:")
    print(f"  a = {best_params[0]:.6f} (true: {true_a})")
    print(f"  b = {best_params[1]:.6f} (true: {true_b})")
    print(f"  Best loss = {best_loss:.6e}")
    print()

    # Display diagnostics
    print("DIAGNOSTICS")
    print("=" * 70)
    print(f"Batch success rate: {diagnostics['batch_success_rate']:.1%}")
    print(f"Total batches attempted: {diagnostics['total_batches_attempted']}")
    print(f"Total retries: {diagnostics['total_retries']}")
    print(f"Convergence achieved: {diagnostics['convergence_achieved']}")
    print(f"Final epoch: {diagnostics['final_epoch']}")
    print(f"Elapsed time: {diagnostics['elapsed_time']:.2f}s")
    print()

    # Failed batches (if any)
    if diagnostics["failed_batches"]:
        print(f"Failed batches ({len(diagnostics['failed_batches'])}):")
        print(f"  Indices: {diagnostics['failed_batches']}")
        print(f"  Error types: {diagnostics['error_types']}")
        print()

    # Aggregate statistics
    agg = diagnostics["aggregate_stats"]
    print("Aggregate Statistics (from batch buffer):")
    print(f"  Mean loss: {agg['mean_loss']:.6e}")
    print(f"  Std loss: {agg['std_loss']:.6e}")
    print(f"  Mean gradient norm: {agg['mean_grad_norm']:.6f}")
    print(f"  Mean batch time: {agg['mean_batch_time'] * 1000:.2f}ms")
    print()

    # Recent batch statistics
    recent_stats = diagnostics["recent_batch_stats"]
    if recent_stats:
        print(f"Recent batch statistics (last {len(recent_stats)} batches):")
        # Show last 5 batches (convert deque to list for slicing)
        recent_list = list(recent_stats)[-5:]
        for i, stats in enumerate(recent_list, 1):
            status = "SUCCESS" if stats["success"] else "FAILED"
            retry_info = (
                f" (retries: {stats['retry_count']})"
                if stats["retry_count"] > 0
                else ""
            )
            print(
                f"  Batch {stats['batch_idx']}: {status}, loss={stats['loss']:.6e}{retry_info}"
            )
        print()

    # Checkpoint information
    if diagnostics["checkpoint_info"]:
        cp = diagnostics["checkpoint_info"]
        print("Checkpoint Information:")
        print(f"  Path: {cp['path']}")
        print(f"  Saved at: {cp['saved_at']}")
        print(f"  Batch index: {cp['batch_idx']}")
        print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key takeaways:")
    print("  - Fault tolerance enabled by default (no configuration needed)")
    print("  - Best parameters always returned (never initial p0)")
    print("  - NaN/Inf detection at three validation points")
    print("  - Adaptive retry strategies for failed batches")
    print("  - Comprehensive diagnostics for analysis")
    print("  - Checkpoints saved automatically for recovery")

    # Cleanup checkpoint worker thread to prevent memory corruption on exit
    if hasattr(optimizer, "_shutdown_checkpoint_worker"):
        optimizer._shutdown_checkpoint_worker()


if __name__ == "__main__":
    main()
