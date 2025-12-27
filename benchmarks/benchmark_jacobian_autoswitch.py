"""Benchmark Jacobian mode auto-switch performance.

This benchmark measures the performance difference between jacfwd and jacrev
on high-parameter problems to validate the 10-100x speedup target.

Task Group 3: Jacobian Auto-Switch
Target Impact: 10-100x Jacobian time reduction on high-parameter problems
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit


def create_high_param_problem(n_params: int, n_data: int):
    """Create a high-parameter curve fitting problem.

    Parameters
    ----------
    n_params : int
        Number of parameters (e.g., 1000)
    n_data : int
        Number of data points (e.g., 100)

    Returns
    -------
    model : callable
        Model function
    xdata : ndarray
        X data
    ydata : ndarray
        Y data with noise
    p0 : ndarray
        Initial parameter guess
    """

    def model(x, *params):
        """Sum of exponentials model with many parameters."""
        # Each pair of parameters defines an exponential term
        result = jnp.zeros_like(x)
        for i in range(0, len(params), 2):
            if i + 1 < len(params):
                a = params[i]
                b = params[i + 1]
                result = result + a * jnp.exp(-b * x)
            else:
                # Odd number of params, last one is a constant
                result = result + params[i]
        return result

    # Generate synthetic data
    np.random.seed(42)
    xdata = np.linspace(0, 5, n_data)

    # True parameters (small random values)
    p_true = np.random.randn(n_params) * 0.1
    ydata = np.array(model(xdata, *p_true))
    ydata += np.random.randn(n_data) * 0.01  # Add noise

    # Initial guess
    p0 = np.ones(n_params) * 0.1

    return model, xdata, ydata, p0


def benchmark_jacobian_modes(n_params: int, n_data: int, n_trials: int = 3):
    """Benchmark jacfwd vs jacrev for given problem dimensions.

    Parameters
    ----------
    n_params : int
        Number of parameters
    n_data : int
        Number of data points
    n_trials : int
        Number of trials to average

    Returns
    -------
    results : dict
        Benchmark results with timing information
    """
    print(f"\nBenchmarking Jacobian modes: {n_params} params, {n_data} data points")
    print("=" * 70)

    model, xdata, ydata, p0 = create_high_param_problem(n_params, n_data)

    # Test 1: jacfwd (expected to be slow for n_params > n_data)
    print("\n1. Testing jacfwd (forward-mode AD)...")
    times_fwd = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _popt_fwd, _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="fwd",
                max_nfev=10,  # Limit iterations for timing
            )
            elapsed = time.perf_counter() - start
            times_fwd.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_fwd.append(float("inf"))

    # Test 2: jacrev (expected to be fast for n_params > n_data)
    print("\n2. Testing jacrev (reverse-mode AD)...")
    times_rev = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _popt_rev, _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="rev",
                max_nfev=10,  # Limit iterations for timing
            )
            elapsed = time.perf_counter() - start
            times_rev.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_rev.append(float("inf"))

    # Test 3: auto mode (should select jacrev)
    print("\n3. Testing auto mode (should select jacrev)...")
    times_auto = []
    for trial in range(n_trials):
        start = time.perf_counter()
        try:
            _popt_auto, _ = curve_fit(
                model,
                xdata,
                ydata,
                p0=p0,
                jacobian_mode="auto",
                max_nfev=10,  # Limit iterations for timing
            )
            elapsed = time.perf_counter() - start
            times_auto.append(elapsed)
            print(f"   Trial {trial + 1}: {elapsed:.4f}s")
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            times_auto.append(float("inf"))

    # Calculate statistics
    avg_fwd = np.mean(times_fwd) if any(np.isfinite(times_fwd)) else float("inf")
    avg_rev = np.mean(times_rev) if any(np.isfinite(times_rev)) else float("inf")
    avg_auto = np.mean(times_auto) if any(np.isfinite(times_auto)) else float("inf")

    speedup = avg_fwd / avg_rev if avg_rev > 0 else 0

    print("\n" + "=" * 70)
    print("Results:")
    print(f"  jacfwd:  {avg_fwd:.4f}s (average)")
    print(f"  jacrev:  {avg_rev:.4f}s (average)")
    print(f"  auto:    {avg_auto:.4f}s (average)")
    print(f"  Speedup: {speedup:.1f}x (jacfwd / jacrev)")

    if speedup >= 10:
        print(f"  ✓ Target achieved: {speedup:.1f}x >= 10x")
    else:
        print(f"  ✗ Target missed: {speedup:.1f}x < 10x")

    return {
        "n_params": n_params,
        "n_data": n_data,
        "times_fwd": times_fwd,
        "times_rev": times_rev,
        "times_auto": times_auto,
        "avg_fwd": avg_fwd,
        "avg_rev": avg_rev,
        "avg_auto": avg_auto,
        "speedup": speedup,
    }


def run_benchmark_suite():
    """Run comprehensive Jacobian mode benchmark suite."""
    print("=" * 70)
    print("Jacobian Auto-Switch Benchmark Suite")
    print("=" * 70)

    # Test configurations: (n_params, n_data, expected speedup range)
    test_cases = [
        (1000, 100, "10-100x"),  # Very tall Jacobian
        (500, 100, "5-50x"),  # Tall Jacobian
        (200, 100, "2-10x"),  # Moderately tall Jacobian
        (100, 100, "~1x"),  # Square Jacobian (no advantage)
    ]

    all_results = []

    for n_params, n_data, expected in test_cases:
        print(f"\n\nTest Case: {n_params} params, {n_data} data (expected: {expected})")
        results = benchmark_jacobian_modes(n_params, n_data, n_trials=3)
        all_results.append(results)

    # Save baseline results
    baseline_dir = Path(__file__).parent / "baselines"
    baseline_dir.mkdir(exist_ok=True)
    baseline_file = baseline_dir / "jacobian_autoswitch.json"

    baseline_data = {
        "test_cases": all_results,
        "platform": "linux",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "best_speedup": max(
                r["speedup"] for r in all_results if np.isfinite(r["speedup"])
            ),
            "target_achieved": any(
                r["speedup"] >= 10 for r in all_results if np.isfinite(r["speedup"])
            ),
        },
    }

    with open(baseline_file, "w") as f:
        json.dump(baseline_data, f, indent=2)

    print(f"\n\n{'=' * 70}")
    print("Benchmark Complete")
    print(f"{'=' * 70}")
    print(f"Baseline saved to: {baseline_file}")
    print(f"Best speedup: {baseline_data['summary']['best_speedup']:.1f}x")
    print(f"Target (10x) achieved: {baseline_data['summary']['target_achieved']}")

    return baseline_data


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    baseline = run_benchmark_suite()

    # Print summary
    print("\n\nSummary:")
    print("-" * 70)
    for result in baseline["test_cases"]:
        status = "✓" if result["speedup"] >= 10 else "○"
        print(
            f"{status} {result['n_params']:4d} params / {result['n_data']:3d} data: "
            f"{result['speedup']:5.1f}x speedup"
        )
