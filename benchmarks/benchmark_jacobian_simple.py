"""Simple Jacobian mode benchmark comparing jacfwd vs jacrev directly.

This benchmark directly measures Jacobian computation time without full optimization,
to demonstrate the performance difference between forward and reverse mode AD.
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import jacfwd, jacrev, jit


def create_test_function(n_params: int):
    """Create a test function with many parameters.

    Parameters
    ----------
    n_params : int
        Number of parameters

    Returns
    -------
    func : callable
        Test function that takes parameters and x data
    """

    @jit
    def func(x, params):
        """Sum of exponentials with many parameters."""
        result = jnp.zeros_like(x)
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                a = params[i]
                b = params[i + 1]
                result = result + a * jnp.exp(-b * x)
        return result

    return func


def benchmark_jacobian_direct(n_params: int, n_data: int, n_trials: int = 5):
    """Benchmark jacfwd vs jacrev directly.

    Parameters
    ----------
    n_params : int
        Number of parameters
    n_data : int
        Number of data points
    n_trials : int
        Number of timing trials

    Returns
    -------
    results : dict
        Timing results
    """
    print(f"\nDirect Jacobian Benchmark: {n_params} params, {n_data} data points")
    print("=" * 70)

    func = create_test_function(n_params)
    x = jnp.linspace(0, 5, n_data)
    params = jnp.ones(n_params) * 0.1

    # Wrapper for Jacobian computation
    @jit
    def residual_func(p):
        return func(x, p)

    # Create Jacobian functions
    jac_fwd = jit(jacfwd(residual_func))
    jac_rev = jit(jacrev(residual_func))

    # Warmup (compile)
    print("\nWarming up (JIT compilation)...")
    _ = jac_fwd(params).block_until_ready()
    _ = jac_rev(params).block_until_ready()
    print("Warmup complete.")

    # Benchmark jacfwd
    print(f"\nBenchmarking jacfwd ({n_trials} trials)...")
    times_fwd = []
    for i in range(n_trials):
        start = time.perf_counter()
        _J_fwd = jac_fwd(params).block_until_ready()
        elapsed = time.perf_counter() - start
        times_fwd.append(elapsed * 1000)  # Convert to ms
        print(f"  Trial {i + 1}: {elapsed * 1000:.4f} ms")

    # Benchmark jacrev
    print(f"\nBenchmarking jacrev ({n_trials} trials)...")
    times_rev = []
    for i in range(n_trials):
        start = time.perf_counter()
        _J_rev = jac_rev(params).block_until_ready()
        elapsed = time.perf_counter() - start
        times_rev.append(elapsed * 1000)  # Convert to ms
        print(f"  Trial {i + 1}: {elapsed * 1000:.4f} ms")

    # Verify numerical equivalence
    J_fwd_final = jac_fwd(params)
    J_rev_final = jac_rev(params)
    max_diff = jnp.max(jnp.abs(J_fwd_final - J_rev_final))
    print(f"\nNumerical accuracy: max difference = {max_diff:.2e}")

    # Calculate statistics
    avg_fwd = np.mean(times_fwd)
    avg_rev = np.mean(times_rev)
    std_fwd = np.std(times_fwd)
    std_rev = np.std(times_rev)
    speedup = avg_fwd / avg_rev

    print("\n" + "=" * 70)
    print("Results:")
    print(f"  jacfwd:  {avg_fwd:.4f} ± {std_fwd:.4f} ms")
    print(f"  jacrev:  {avg_rev:.4f} ± {std_rev:.4f} ms")
    print(f"  Speedup: {speedup:.1f}x (jacfwd / jacrev)")

    if speedup >= 10:
        print(f"  ✓ Target achieved: {speedup:.1f}x >= 10x")
    elif speedup >= 5:
        print(f"  ○ Partial success: {speedup:.1f}x >= 5x")
    else:
        print(f"  ✗ Target missed: {speedup:.1f}x < 10x")

    return {
        "n_params": n_params,
        "n_data": n_data,
        "times_fwd_ms": times_fwd,
        "times_rev_ms": times_rev,
        "avg_fwd_ms": avg_fwd,
        "avg_rev_ms": avg_rev,
        "std_fwd_ms": std_fwd,
        "std_rev_ms": std_rev,
        "speedup": speedup,
        "max_diff": float(max_diff),
    }


def run_simple_benchmark():
    """Run simple Jacobian benchmark suite."""
    print("=" * 70)
    print("Jacobian Auto-Switch Simple Benchmark")
    print("=" * 70)

    # Test cases: (n_params, n_data, expected speedup)
    test_cases = [
        (1000, 100, "Expected: 10x+ speedup"),
        (500, 100, "Expected: 5x+ speedup"),
        (200, 100, "Expected: 2x+ speedup"),
        (100, 100, "Expected: ~1x (no advantage)"),
    ]

    all_results = []

    for n_params, n_data, description in test_cases:
        print(f"\n\n{'=' * 70}")
        print(f"Test Case: {n_params} params × {n_data} data")
        print(description)
        print(f"{'=' * 70}")

        results = benchmark_jacobian_direct(n_params, n_data, n_trials=5)
        all_results.append(results)

    # Save results
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file = baseline_dir / "jacobian_autoswitch.json"

    baseline_data = {
        "test_cases": all_results,
        "platform": "linux",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "best_speedup": max(r["speedup"] for r in all_results),
            "target_achieved": any(r["speedup"] >= 10 for r in all_results),
            "partial_success": any(r["speedup"] >= 5 for r in all_results),
        },
    }

    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\n\n{'=' * 70}")
    print("Benchmark Complete")
    print(f"{'=' * 70}")
    print(f"Baseline saved to: {baseline_file}")
    print("\nSummary:")
    print("-" * 70)
    for result in all_results:
        status = (
            "✓" if result["speedup"] >= 10 else ("○" if result["speedup"] >= 5 else "✗")
        )
        print(
            f"{status} {result['n_params']:4d} params / {result['n_data']:3d} data: "
            f"{result['speedup']:5.1f}x speedup ({result['avg_fwd_ms']:.2f}ms → {result['avg_rev_ms']:.2f}ms)"
        )

    print(
        f"\nOverall: Target (10x) {'✓ ACHIEVED' if baseline_data['summary']['target_achieved'] else '✗ MISSED'}"
    )

    return baseline_data


if __name__ == "__main__":
    _baseline = run_simple_benchmark()
