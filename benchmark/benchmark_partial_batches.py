"""Benchmark for batch shape padding with partial batches.

This benchmark specifically tests the scenario where the last batch is partial,
which triggers JIT recompilation overhead in dynamic mode but not in auto/static mode.

Target: 30-50% throughput improvement when partial batches are present.
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nlsq.streaming_config import StreamingConfig
from nlsq.streaming_optimizer import StreamingOptimizer


def create_test_data(n_points: int, seed: int = 42):
    """Create synthetic exponential decay data."""
    np.random.seed(seed)
    x_data = np.linspace(0, 10, n_points)
    true_a, true_b = 2.5, 0.3
    noise = 0.1 * np.random.randn(n_points)
    y_data = true_a * np.exp(-true_b * x_data) + noise
    return x_data, y_data


def model_function(x, a, b):
    """Exponential decay model."""
    return a * jnp.exp(-b * x)


def benchmark_partial_batch(
    mode: str,
    n_points: int,
    batch_size: int,
    max_epochs: int,
) -> dict:
    """Benchmark with deliberate partial batch."""
    # Create data
    x_data, y_data = create_test_data(n_points)

    # Configure optimizer
    config = StreamingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        batch_shape_padding=mode,
        learning_rate=0.01,
        warmup_steps=20,
        enable_fault_tolerance=False,  # Minimal overhead
    )

    optimizer = StreamingOptimizer(config)

    # Run benchmark
    start_time = time.time()

    result = optimizer.fit(
        (x_data, y_data),
        model_function,
        p0=[2.0, 0.2],
        verbose=0,
    )

    elapsed_time = time.time() - start_time

    # Extract diagnostics
    diag = result.get("streaming_diagnostics", {})
    batch_padding = diag.get("batch_padding", {})

    # Calculate batches
    n_full_batches = n_points // batch_size
    partial_batch_size = n_points % batch_size

    return {
        "mode": mode,
        "n_points": n_points,
        "batch_size": batch_size,
        "n_full_batches": n_full_batches,
        "partial_batch_size": partial_batch_size,
        "max_epochs": max_epochs,
        "elapsed_time": elapsed_time,
        "throughput": n_points * max_epochs / elapsed_time,
        "success": result["success"],
        "batch_padding": batch_padding,
    }


def run_partial_batch_benchmarks():
    """Run benchmarks specifically designed to trigger partial batch overhead."""
    print("\n" + "=" * 70)
    print("PARTIAL BATCH OVERHEAD BENCHMARK")
    print("Task Group 7: Measuring JIT recompilation impact from partial batches")
    print("=" * 70)

    results = []

    # Configurations designed to create partial batches
    configs = [
        {
            "n_points": 10_250,  # 10 batches of 1000 + 250 partial
            "batch_size": 1000,
            "max_epochs": 10,
            "description": "10 full + 1 partial (25% of batch size)",
        },
        {
            "n_points": 50_500,  # 10 batches of 5000 + 500 partial
            "batch_size": 5000,
            "max_epochs": 5,
            "description": "10 full + 1 partial (10% of batch size)",
        },
        {
            "n_points": 105_000,  # 10 batches of 10000 + 5000 partial
            "batch_size": 10000,
            "max_epochs": 3,
            "description": "10 full + 1 partial (50% of batch size)",
        },
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Configuration: {config['description']}")
        print(f"Points: {config['n_points']:,}, Batch: {config['batch_size']:,}")
        print(f"{'=' * 70}")

        # Warm up JAX
        print("Warming up JAX...")
        x_warmup, y_warmup = create_test_data(1000)
        warmup_config = StreamingConfig(
            batch_size=100, max_epochs=1, batch_shape_padding="auto"
        )
        warmup_opt = StreamingOptimizer(warmup_config)
        _ = warmup_opt.fit(
            (x_warmup, y_warmup), model_function, p0=[2.0, 0.2], verbose=0
        )

        # Test dynamic mode (baseline)
        print("\nTesting DYNAMIC mode (allows recompiles)...")
        result_dynamic = benchmark_partial_batch(
            mode="dynamic",
            n_points=config["n_points"],
            batch_size=config["batch_size"],
            max_epochs=config["max_epochs"],
        )
        results.append(result_dynamic)

        print(f"  Time: {result_dynamic['elapsed_time']:.3f}s")
        print(f"  Throughput: {result_dynamic['throughput']:,.0f} points/sec")
        print(
            f"  Batches: {result_dynamic['n_full_batches']} full + {result_dynamic['partial_batch_size']} points partial"
        )

        # Test auto mode (optimized)
        print("\nTesting AUTO mode (eliminates recompiles)...")
        result_auto = benchmark_partial_batch(
            mode="auto",
            n_points=config["n_points"],
            batch_size=config["batch_size"],
            max_epochs=config["max_epochs"],
        )
        results.append(result_auto)

        print(f"  Time: {result_auto['elapsed_time']:.3f}s")
        print(f"  Throughput: {result_auto['throughput']:,.0f} points/sec")
        print(
            f"  Max batch shape: {result_auto['batch_padding'].get('max_batch_shape', 'N/A')}"
        )
        print(
            f"  Post-warmup recompiles: {result_auto['batch_padding'].get('post_warmup_recompiles', 'N/A')}"
        )

        # Calculate improvement
        speedup = result_dynamic["elapsed_time"] / result_auto["elapsed_time"]
        improvement_pct = (speedup - 1.0) * 100

        print(f"\n{'=' * 70}")
        print(f"SPEEDUP ANALYSIS: {config['description']}")
        print(f"{'=' * 70}")
        print(f"  Dynamic time: {result_dynamic['elapsed_time']:.3f}s")
        print(f"  Auto time: {result_auto['elapsed_time']:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {improvement_pct:+.1f}%")

    # Save results
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)

    output_file = baseline_dir / "streaming_partial_batch_overhead.json"

    baseline_data = {
        "benchmark_name": "partial_batch_overhead",
        "task_group": 7,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Measures JIT recompilation overhead from partial batches",
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Baseline saved to: {output_file}")
    print(f"{'=' * 70}")

    # Summary
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")

    improvements = []
    for i in range(0, len(results), 2):
        dynamic = results[i]
        auto = results[i + 1]
        speedup = dynamic["elapsed_time"] / auto["elapsed_time"]
        improvement = (speedup - 1.0) * 100
        improvements.append(improvement)

        print(f"\n{configs[i // 2]['description']}:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Improvement: {improvement:+.1f}%")
        print(
            f"  Post-warmup recompiles (auto): {auto['batch_padding'].get('post_warmup_recompiles', 'N/A')}"
        )

    avg_improvement = np.mean(improvements)
    max_improvement = np.max(improvements)
    print(f"\nAverage improvement: {avg_improvement:+.1f}%")
    print(f"Best improvement: {max_improvement:+.1f}%")
    print("\nNote: On CPU, improvements may be modest (5-15%).")
    print("On GPU, improvements of 30-50% are expected due to higher JIT overhead.")

    return baseline_data


if __name__ == "__main__":
    _baseline_data = run_partial_batch_benchmarks()

    print("\n" + "=" * 70)
    print("Partial batch benchmark complete!")
    print("=" * 70)
