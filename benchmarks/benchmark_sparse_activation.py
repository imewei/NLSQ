"""Benchmark sparse activation optimization (Task Group 6).

This script benchmarks the performance improvements from sparse solver activation:
- Target: 3-10x speed improvement on sparse problems
- Target: 5-50x memory reduction on sparse problems

Stores baseline results in benchmarks/baselines/sparse_activation.json
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit
from nlsq.sparse_jacobian import detect_jacobian_sparsity


def sparse_parameter_selection_model(x, *params):
    """Sparse parameter selection model.

    Each parameter only affects a subset of data points.
    This creates a sparse Jacobian pattern.
    """
    x = jnp.asarray(x)
    params = jnp.asarray(params)

    n_groups = len(params) // 10
    result = jnp.zeros_like(x, dtype=jnp.float64)

    for i in range(n_groups):
        mask = (x >= i * 10) & (x < (i + 1) * 10)
        result = jnp.where(mask, params[i * 10], result)

    return result


def benchmark_sparse_detection():
    """Benchmark sparsity detection overhead."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Sparsity Detection Overhead")
    print("=" * 80)

    results = {}

    for n_params in [50, 100, 200]:
        n_data = 10000

        np.random.seed(42)
        x_data = np.linspace(0, n_params, n_data)
        p0 = np.ones(n_params)

        # Benchmark detection time
        start = time.time()
        for _ in range(10):  # Average over 10 runs
            sparsity_ratio, info = detect_jacobian_sparsity(
                sparse_parameter_selection_model,
                p0,
                x_data[:100],  # Sample size
                threshold=0.01,
            )
        detection_time = (time.time() - start) / 10

        results[f"n_params_{n_params}"] = {
            "n_params": n_params,
            "n_data": n_data,
            "sparsity_ratio": float(sparsity_ratio),
            "detection_time_ms": detection_time * 1000,
            "memory_reduction_pct": info["memory_reduction"],
        }

        print(f"\nn_params={n_params}, n_data={n_data}")
        print(f"  Sparsity: {sparsity_ratio:.1%}")
        print(f"  Detection time: {detection_time * 1000:.2f} ms")
        print(f"  Memory reduction: {info['memory_reduction']:.1f}%")

    return results


def benchmark_sparse_vs_dense():
    """Benchmark sparse solver vs dense solver on sparse problems.

    Target: 3-10x speed improvement, 5-50x memory reduction
    """
    print("\n" + "=" * 80)
    print("BENCHMARK: Sparse vs Dense Solver Performance")
    print("=" * 80)

    results = {}

    # Test with different problem sizes
    test_cases = [
        {"n_params": 100, "n_data": 12000, "name": "medium"},
        {"n_params": 200, "n_data": 20000, "name": "large"},
    ]

    for case in test_cases:
        n_params = case["n_params"]
        n_data = case["n_data"]
        name = case["name"]

        print(f"\n{'=' * 60}")
        print(f"Test case: {name} (n_params={n_params}, n_data={n_data})")
        print(f"{'=' * 60}")

        # Generate data
        np.random.seed(42)
        x_data = np.linspace(0, n_params, n_data)
        true_params = np.ones(n_params) * 2.0
        y_data = sparse_parameter_selection_model(x_data, *true_params)
        y_data += np.random.normal(0, 0.05, size=n_data)

        p0 = np.ones(n_params) * 1.5

        # Detect sparsity first
        sparsity_ratio, info = detect_jacobian_sparsity(
            sparse_parameter_selection_model, p0, x_data[:100], threshold=0.01
        )

        print(f"\nSparsity: {sparsity_ratio:.1%}")
        print(f"Memory reduction potential: {info['memory_reduction']:.1f}%")

        # Benchmark with sparse solver activation (if implemented)
        # For now, both use dense solver (sparse path falls back to dense)
        print("\nBenchmark 1: Dense solver (tr_solver='exact')")
        start = time.time()
        result_dense = curve_fit(
            sparse_parameter_selection_model,
            x_data,
            y_data,
            p0=p0,
            tr_solver="exact",  # Force dense solver
            full_output=True,
            maxfev=100,
        )
        time_dense = time.time() - start

        print(f"  Time: {time_dense:.3f}s")
        print(f"  Iterations: {result_dense.get('nit', 'N/A')}")
        print(f"  Final cost: {result_dense.get('cost', 'N/A'):.2e}")

        # Benchmark with auto-selection (currently also uses dense)
        print("\nBenchmark 2: Auto-selection (sparse detection enabled)")
        start = time.time()
        result_auto = curve_fit(
            sparse_parameter_selection_model,
            x_data,
            y_data,
            p0=p0,
            tr_solver=None,  # Let auto-selection decide
            full_output=True,
            maxfev=100,
        )
        time_auto = time.time() - start

        print(f"  Time: {time_auto:.3f}s")
        print(f"  Iterations: {result_auto.get('nit', 'N/A')}")
        print(f"  Final cost: {result_auto.get('cost', 'N/A'):.2e}")

        # Check sparsity diagnostics
        if "sparsity_detected" in result_auto:
            sd = result_auto["sparsity_detected"]
            print(f"  Sparsity detected: {sd['detected']}")
            print(f"  Sparsity ratio: {sd['ratio']:.1%}")
            print(f"  Solver used: {sd['solver']}")

        # Store results
        results[name] = {
            "n_params": n_params,
            "n_data": n_data,
            "sparsity_ratio": float(sparsity_ratio),
            "time_dense_s": time_dense,
            "time_auto_s": time_auto,
            "speedup": time_dense / time_auto if time_auto > 0 else 1.0,
            "solver_used": result_auto.get("sparsity_detected", {}).get(
                "solver", "unknown"
            ),
            "memory_reduction_pct": info["memory_reduction"],
        }

        # Summary
        speedup = time_dense / time_auto if time_auto > 0 else 1.0
        print("\nSummary:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Memory reduction potential: {info['memory_reduction']:.1f}%")
        print("  Note: Sparse SVD not yet implemented (both paths use dense)")

    return results


def run_benchmarks():
    """Run all sparse activation benchmarks."""
    print("\n" + "=" * 80)
    print("SPARSE ACTIVATION BENCHMARKS (Task Group 6)")
    print("=" * 80)
    print("\nTarget Impact:")
    print("  - 3-10x speed improvement on sparse problems")
    print("  - 5-50x memory reduction on sparse problems")
    print("\nStatus:")
    print("  ✓ Sparsity detection implemented")
    print("  ✓ Auto-selection logic implemented")
    print("  ✓ Sparsity diagnostics added to results")
    print("  ⚠ Sparse SVD path: fallback to dense (TODO: implement)")

    # Run benchmarks
    detection_results = benchmark_sparse_detection()
    solver_results = benchmark_sparse_vs_dense()

    # Combine results
    all_results = {
        "task_group": "6_sparse_activation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "partial_implementation",
        "notes": "Sparse detection and auto-selection complete. Sparse SVD to be implemented.",
        "detection": detection_results,
        "solver_comparison": solver_results,
    }

    # Save to baseline file
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file = baseline_dir / "sparse_activation.json"

    with open(baseline_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {baseline_file}")
    print("=" * 80)

    # Print summary
    print("\nSUMMARY:")
    print("-" * 80)
    print("\nSparsity Detection Performance:")
    for key, val in detection_results.items():
        print(
            f"  {key}: {val['detection_time_ms']:.2f} ms (sparsity: {val['sparsity_ratio']:.1%})"
        )

    print("\nSolver Performance:")
    for key, val in solver_results.items():
        print(f"  {key}:")
        print(f"    Speedup: {val['speedup']:.2f}x (target: 3-10x)")
        print(
            f"    Memory reduction: {val['memory_reduction_pct']:.1f}% (target: >80%)"
        )
        print(f"    Solver used: {val['solver_used']}")

    print("\nNEXT STEPS:")
    print("  1. Implement sparse SVD using JAX sparse operations")
    print("  2. Re-benchmark to measure actual speedup (target: 3-10x)")
    print("  3. Validate memory reduction in practice (target: 5-50x)")

    return all_results


if __name__ == "__main__":
    _results = run_benchmarks()
