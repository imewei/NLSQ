"""Benchmark host-device transfers in TRF solver.

This benchmark measures transfer bytes and GPU iteration time to validate
the 80% reduction target and 5-15% performance improvement.

Target Metrics (Task Group 2):
- Transfer bytes: 80% reduction (baseline ~80KB → <16KB per iteration)
- Transfer count: 24+ operations → <5 per iteration
- GPU iteration time: 5-15% reduction

Usage:
    python benchmarks/host_device_transfer_benchmark.py [--gpu] [--save-baseline]

Results are saved to: benchmarks/baselines/host_device_transfers.json
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nlsq.least_squares import least_squares
from nlsq.profiling import HAS_JAX_PROFILER, TransferProfiler


def get_system_info():
    """Get system and JAX configuration information."""
    return {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "has_profiler": HAS_JAX_PROFILER,
    }


def create_benchmark_problem(n_points=10_000, n_params=3):
    """Create exponential decay benchmark problem.

    Parameters
    ----------
    n_points : int
        Number of data points (default: 10,000 for GPU advantage)
    n_params : int
        Number of parameters (default: 3)

    Returns
    -------
    problem : dict
        Dictionary containing model, data, and true parameters
    """

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + c

    np.random.seed(42)  # Reproducible results
    xdata = np.linspace(0, 10, n_points)
    true_params = [2.5, 0.5, 1.0]
    ydata = model(xdata, *true_params) + 0.1 * np.random.randn(n_points)
    p0 = [1.0, 0.1, 0.0]

    return {
        "model": model,
        "xdata": xdata,
        "ydata": ydata,
        "p0": p0,
        "true_params": true_params,
        "n_points": n_points,
        "n_params": n_params,
    }


def benchmark_baseline_transfers(problem, enable_profiling=True):
    """Benchmark baseline transfer behavior (before optimization).

    Parameters
    ----------
    problem : dict
        Benchmark problem from create_benchmark_problem()
    enable_profiling : bool
        Enable JAX profiler (requires jax[profiler])

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - transfer_bytes_per_iteration : int (if profiling enabled)
        - transfer_count_per_iteration : int (if profiling enabled)
        - iteration_time_ms : float
        - total_time_s : float
        - iterations : int
        - success : bool
    """
    model = problem["model"]
    xdata = problem["xdata"]
    ydata = problem["ydata"]
    p0 = problem["p0"]

    # Create profiler
    profiler = TransferProfiler(enable=enable_profiling)

    # Warm-up run (JIT compilation)
    print("Warming up JIT compilation...")
    _ = least_squares(
        lambda p: model(xdata, *p) - ydata,
        p0,
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
    )

    # Benchmark run
    print("Running benchmark...")
    start_time = time.time()

    result = least_squares(
        lambda p: model(xdata, *p) - ydata,
        p0,
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
    )

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    iteration_time_ms = (
        (total_time / result.nit * 1000) if result.nit > 0 else float("inf")
    )

    metrics = {
        "total_time_s": total_time,
        "iterations": result.nit,
        "iteration_time_ms": iteration_time_ms,
        "success": result.success,
        "final_cost": float(result.cost),
        "function_evals": result.nfev,
        "jacobian_evals": result.njev,
    }

    # Add profiling data if available
    if enable_profiling:
        diag = profiler.get_diagnostics()
        metrics.update(
            {
                "transfer_bytes_per_iteration": diag["avg_bytes_per_iteration"],
                "transfer_count_per_iteration": diag["avg_transfers_per_iteration"],
                "profiling_enabled": True,
            }
        )
    else:
        metrics["profiling_enabled"] = False

    return metrics


def estimate_transfer_bytes(n_points, n_params):
    """Estimate transfer bytes based on problem size.

    This provides an estimate when JAX profiler is not available.

    Parameters
    ----------
    n_points : int
        Number of data points (residuals)
    n_params : int
        Number of parameters

    Returns
    -------
    estimate : dict
        Estimated transfer sizes per iteration
    """
    # Based on audit in task-group-2-audit-report.md
    gradient_bytes = 2 * n_params * 8  # 2 conversions
    svd_bytes = (n_params + n_params * n_params + n_points) * 8  # s, V, uf
    cost_bytes = 3 * 8  # 3 scalar conversions
    norm_bytes = 3 * n_params * 8  # 3 norm operations
    logging_bytes = n_params * 8  # params in logging
    misc_bytes = 200  # miscellaneous small arrays

    total_bytes = (
        gradient_bytes
        + svd_bytes
        + cost_bytes
        + norm_bytes
        + logging_bytes
        + misc_bytes
    )

    return {
        "total_estimate_bytes": total_bytes,
        "gradient_bytes": gradient_bytes,
        "svd_bytes": svd_bytes,
        "cost_bytes": cost_bytes,
        "norm_bytes": norm_bytes,
        "logging_bytes": logging_bytes,
        "misc_bytes": misc_bytes,
    }


def save_baseline(metrics, problem_info, system_info, output_path):
    """Save baseline metrics to JSON file.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics
    problem_info : dict
        Problem size information
    system_info : dict
        System configuration
    output_path : Path
        Output JSON file path
    """
    baseline_data = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "problem_size": {
            "n_residuals": problem_info["n_points"],
            "n_params": problem_info["n_params"],
        },
        "baseline_metrics": metrics,
        "estimated_transfers": estimate_transfer_bytes(
            problem_info["n_points"], problem_info["n_params"]
        ),
        "optimization_status": "before_transfer_reduction",
        "target_metrics": {
            "transfer_reduction_percent": 80,
            "iteration_time_reduction_percent": [5, 15],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\nBaseline saved to: {output_path}")


def print_benchmark_results(metrics, problem_info):
    """Print formatted benchmark results.

    Parameters
    ----------
    metrics : dict
        Benchmark metrics
    problem_info : dict
        Problem information
    """
    print("\n" + "=" * 70)
    print("HOST-DEVICE TRANSFER BENCHMARK RESULTS")
    print("=" * 70)

    print("\nProblem Size:")
    print(f"  Data points: {problem_info['n_points']:,}")
    print(f"  Parameters: {problem_info['n_params']}")

    print("\nOptimization Results:")
    print(f"  Success: {metrics['success']}")
    print(f"  Iterations: {metrics['iterations']}")
    print(f"  Final cost: {metrics['final_cost']:.6e}")
    print(f"  Function evaluations: {metrics['function_evals']}")
    print(f"  Jacobian evaluations: {metrics['jacobian_evals']}")

    print("\nPerformance Metrics:")
    print(f"  Total time: {metrics['total_time_s']:.3f} s")
    print(f"  Iteration time: {metrics['iteration_time_ms']:.2f} ms")

    if metrics["profiling_enabled"]:
        print("\nTransfer Metrics (Profiler):")
        print(
            f"  Bytes/iteration: {metrics['transfer_bytes_per_iteration']:.1f} bytes "
            f"({metrics['transfer_bytes_per_iteration'] / 1024:.1f} KB)"
        )
        print(f"  Transfers/iteration: {metrics['transfer_count_per_iteration']:.1f}")
    else:
        # Show estimates
        estimates = estimate_transfer_bytes(
            problem_info["n_points"], problem_info["n_params"]
        )
        print("\nTransfer Estimates (Profiler not available):")
        print(
            f"  Estimated bytes/iteration: {estimates['total_estimate_bytes']:,} bytes "
            f"({estimates['total_estimate_bytes'] / 1024:.1f} KB)"
        )
        print(f"  - Gradient: {estimates['gradient_bytes']} bytes")
        print(f"  - SVD results: {estimates['svd_bytes']:,} bytes")
        print(f"  - Norms: {estimates['norm_bytes']} bytes")

    print("\n" + "=" * 70)


def main():
    """Run host-device transfer benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark host-device transfers in TRF solver"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU backend (error if not available)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save baseline to JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/baselines/host_device_transfers.json",
        help="Output path for baseline JSON",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=10_000,
        help="Number of data points (default: 10,000)",
    )
    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="Disable JAX profiler (use estimates)",
    )

    args = parser.parse_args()

    # Check backend
    system_info = get_system_info()
    print("System Information:")
    print(f"  Platform: {system_info['platform']}")
    print(f"  Python: {system_info['python_version']}")
    print(f"  JAX version: {system_info['jax_version']}")
    print(f"  JAX backend: {system_info['jax_backend']}")
    print(f"  Devices: {', '.join(system_info['jax_devices'])}")
    print(f"  JAX profiler: {'Available' if HAS_JAX_PROFILER else 'Not available'}")

    if args.gpu and system_info["jax_backend"] != "gpu":
        print("\nERROR: GPU requested but not available")
        print("Install GPU JAX with: pip install jax[cuda12]")
        sys.exit(1)

    # Create problem
    print(f"\nCreating benchmark problem ({args.n_points:,} points)...")
    problem = create_benchmark_problem(n_points=args.n_points)

    # Run benchmark
    enable_profiling = HAS_JAX_PROFILER and not args.no_profiler
    metrics = benchmark_baseline_transfers(problem, enable_profiling=enable_profiling)

    # Print results
    print_benchmark_results(metrics, problem)

    # Save baseline if requested
    if args.save_baseline:
        output_path = Path(args.output)
        save_baseline(metrics, problem, system_info, output_path)

        # Print next steps
        print("\nNext Steps:")
        print("1. Implement NumPy→JAX transformations (Task 2.3)")
        print("2. Add logging gating with jax.debug.callback (Task 2.4)")
        print("3. Remove unnecessary .block_until_ready() calls (Task 2.5)")
        print("4. Re-run benchmark to measure improvements")
        print("5. Verify >80% transfer reduction and 5-15% time improvement")


if __name__ == "__main__":
    main()
