"""
Converted from 04_interpreting_diagnostics.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Example 4: Interpreting Detailed Diagnostics
#
# This example demonstrates how to interpret and analyze the comprehensive
# diagnostic information provided by the streaming optimizer.
#
# Features demonstrated:
# - Streaming diagnostics structure
# - Aggregate statistics interpretation
# - Recent batch statistics analysis
# - Checkpoint information access
# - Performance metrics
#
# Run this example:
#     python examples/streaming/04_interpreting_diagnostics.py
#
# ======================================================================

import json

import numpy as np

from nlsq import StreamingConfig, StreamingOptimizer


def polynomial_model(x, a, b, c):
    """Polynomial model: y = a + b*x + c*x^2"""
    return a + b * x + c * x**2


def print_diagnostics_structure(diagnostics):
    """Print the structure and contents of diagnostics dictionary"""
    print("DIAGNOSTICS STRUCTURE")
    print("=" * 70)
    print()

    # Top-level keys
    print("Available diagnostic fields:")
    for key in sorted(diagnostics.keys()):
        value_type = type(diagnostics[key]).__name__
        print(f"  - {key:30s} : {value_type}")
    print()


def analyze_success_metrics(diagnostics):
    """Analyze overall success metrics"""
    print("SUCCESS METRICS")
    print("=" * 70)
    print(f"Batch success rate: {diagnostics['batch_success_rate']:.1%}")
    print(f"Total batches attempted: {diagnostics['total_batches_attempted']}")
    print(f"Failed batches: {len(diagnostics['failed_batches'])}")
    print(f"Total retries: {diagnostics['total_retries']}")
    print(f"Convergence achieved: {diagnostics['convergence_achieved']}")
    print(f"Final epoch: {diagnostics['final_epoch']}")
    print(f"Elapsed time: {diagnostics['elapsed_time']:.2f}s")
    print()


def analyze_failure_patterns(diagnostics):
    """Analyze failure patterns and error types"""
    print("FAILURE ANALYSIS")
    print("=" * 70)

    if not diagnostics["failed_batches"]:
        print("No failed batches!")
        print()
        return

    print(f"Failed batch indices: {diagnostics['failed_batches']}")
    print()

    # Error type distribution
    print("Error Type Distribution:")
    error_types = diagnostics["error_types"]
    total_errors = sum(error_types.values())
    for error_type, count in sorted(
        error_types.items(), key=lambda x: x[1], reverse=True
    ):
        pct = count / total_errors * 100
        print(f"  {error_type:20s}: {count:3d} ({pct:5.1f}%)")
    print()

    # Retry patterns
    print("Retry Patterns:")
    retry_counts = diagnostics["retry_counts"]
    if retry_counts:
        retry_values = list(retry_counts.values())
        print(f"  Batches with retries: {len(retry_values)}")
        print(f"  Min retries: {min(retry_values)}")
        print(f"  Max retries: {max(retry_values)}")
        print(f"  Avg retries: {np.mean(retry_values):.2f}")
        print(f"  Total retries: {sum(retry_values)}")
    else:
        print("  No retries performed")
    print()


def analyze_aggregate_statistics(diagnostics):
    """Analyze aggregate statistics from batch buffer"""
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    agg = diagnostics["aggregate_stats"]
    print(f"Mean loss:          {agg['mean_loss']:.6e}")
    print(f"Std loss:           {agg['std_loss']:.6e}")
    print(f"Mean gradient norm: {agg['mean_grad_norm']:.6f}")
    print(f"Std gradient norm:  {agg['std_grad_norm']:.6f}")
    print(f"Mean batch time:    {agg['mean_batch_time'] * 1000:.2f}ms")
    print(f"Std batch time:     {agg['std_batch_time'] * 1000:.2f}ms")
    print()

    # Interpretation
    print("Interpretation:")
    cv_loss = agg["std_loss"] / max(agg["mean_loss"], 1e-10)
    print(f"  - Coefficient of variation (loss): {cv_loss:.2%}")
    if cv_loss < 0.1:
        print("    => Very stable optimization")
    elif cv_loss < 0.5:
        print("    => Moderately stable optimization")
    else:
        print("    => High variability in loss")
    print()


def analyze_recent_batches(diagnostics, n_recent=10):
    """Analyze recent batch statistics"""
    print(f"RECENT BATCH STATISTICS (last {n_recent} batches)")
    print("=" * 70)

    recent_stats = diagnostics["recent_batch_stats"]
    if not recent_stats:
        print("No batch statistics available")
        print()
        return

    # Show last N batches (convert deque to list for slicing)
    last_n = list(recent_stats)[-n_recent:]
    print(f"Showing {len(last_n)} most recent batches:")
    print()

    # Header
    print(
        f"{'Batch':>8s} {'Status':>10s} {'Loss':>12s} {'GradNorm':>10s} {'Time':>8s} {'Retries':>8s}"
    )
    print("-" * 70)

    # Show each batch
    for stats in last_n:
        batch_idx = stats["batch_idx"]
        status = "SUCCESS" if stats["success"] else "FAILED"
        loss = stats["loss"]
        grad_norm = stats["grad_norm"]
        batch_time = stats["batch_time"] * 1000  # Convert to ms
        retry_count = stats["retry_count"]

        loss_str = f"{loss:.4e}" if np.isfinite(loss) else "inf"
        print(
            f"{batch_idx:8d} {status:>10s} {loss_str:>12s} {grad_norm:10.4f} {batch_time:7.1f}ms {retry_count:8d}"
        )
    print()

    # Statistics on recent batches
    successful_recent = [s for s in last_n if s["success"]]
    if successful_recent:
        recent_losses = [s["loss"] for s in successful_recent]
        print("Recent batch statistics:")
        print(f"  Success rate: {len(successful_recent) / len(last_n):.1%}")
        print(f"  Mean loss: {np.mean(recent_losses):.6e}")
        print(f"  Min loss: {min(recent_losses):.6e}")
        print(f"  Max loss: {max(recent_losses):.6e}")
        print()


def analyze_checkpoint_info(diagnostics):
    """Analyze checkpoint information"""
    print("CHECKPOINT INFORMATION")
    print("=" * 70)

    cp_info = diagnostics.get("checkpoint_info")
    if not cp_info:
        print("No checkpoint information available")
        print("(Checkpoints may be disabled or not saved yet)")
        print()
        return

    print("Latest checkpoint:")
    print(f"  Path: {cp_info['path']}")
    print(f"  Saved at: {cp_info['saved_at']}")
    print(f"  Batch index: {cp_info['batch_idx']}")
    print()
    print("Resume using:")
    print(f"  config = StreamingConfig(resume_from_checkpoint='{cp_info['path']}')")
    print()


def _make_serializable(obj):
    """Recursively convert objects to JSON-serializable types."""
    from collections import deque

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, deque)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    else:
        return str(obj)


def export_diagnostics_json(diagnostics, filename="diagnostics.json"):
    """Export diagnostics to JSON for further analysis"""
    print("EXPORT DIAGNOSTICS")
    print("=" * 70)

    # Create serializable copy (handle non-serializable types like deques)
    diagnostics_copy = _make_serializable(diagnostics)

    # Write to JSON file
    with open(filename, "w") as f:
        json.dump(diagnostics_copy, f, indent=2)

    print(f"Diagnostics exported to: {filename}")
    print(f"File size: {len(json.dumps(diagnostics_copy))} bytes")
    print()


def main():
    print("=" * 70)
    print("Streaming Optimizer: Interpreting Diagnostics Example")
    print("=" * 70)
    print()

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 5000
    x_data = np.linspace(-5, 5, n_samples)
    true_a, true_b, true_c = 1.0, 2.0, -0.5
    y_true = polynomial_model(x_data, true_a, true_b, true_c)
    y_data = y_true + 0.2 * np.random.randn(n_samples)

    print(f"Dataset: {n_samples} samples")
    print(f"True parameters: a={true_a}, b={true_b}, c={true_c}")
    print()

    # Configure optimizer
    config = StreamingConfig(
        batch_size=100,
        max_epochs=5,
        learning_rate=0.001,
        enable_fault_tolerance=True,
        checkpoint_dir="checkpoints_diagnostics",
        checkpoint_frequency=10,
        enable_checkpoints=True,
        batch_stats_buffer_size=100,  # Track last 100 batches
    )

    optimizer = StreamingOptimizer(config)
    p0 = np.array([0.5, 1.0, -0.2])

    # Fit model
    print("Running optimization...")
    result = optimizer.fit(
        (x_data, y_data),
        polynomial_model,
        p0,
        verbose=1,
    )

    print()
    print("Optimization complete!")
    print()

    # Extract diagnostics
    diagnostics = result["streaming_diagnostics"]

    # Analyze diagnostics
    print_diagnostics_structure(diagnostics)
    analyze_success_metrics(diagnostics)
    analyze_failure_patterns(diagnostics)
    analyze_aggregate_statistics(diagnostics)
    analyze_recent_batches(diagnostics, n_recent=10)
    analyze_checkpoint_info(diagnostics)

    # Export diagnostics
    export_diagnostics_json(diagnostics, "streaming_diagnostics_example.json")

    # Final results
    print("FINAL RESULTS")
    print("=" * 70)
    best_params = result["x"]
    print("Best parameters:")
    print(f"  a = {best_params[0]:.6f} (true: {true_a})")
    print(f"  b = {best_params[1]:.6f} (true: {true_b})")
    print(f"  c = {best_params[2]:.6f} (true: {true_c})")
    print(f"  Best loss = {result['best_loss']:.6e}")
    print()

    print("=" * 70)
    print("Example complete!")
    print()
    print("Key takeaways:")
    print("  - streaming_diagnostics contains comprehensive information")
    print("  - Aggregate statistics summarize overall performance")
    print("  - Recent batch statistics show optimization trajectory")
    print("  - Checkpoint information enables recovery")
    print("  - Error analysis helps diagnose issues")
    print("  - Diagnostics can be exported to JSON for further analysis")

    # Cleanup checkpoint worker thread to prevent memory corruption on exit
    if hasattr(optimizer, "_shutdown_checkpoint_worker"):
        optimizer._shutdown_checkpoint_worker()


if __name__ == "__main__":
    main()
