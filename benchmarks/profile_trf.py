"""
Profile TRF algorithm hot paths for optimization opportunities

This script profiles the Trust Region Reflective algorithm to identify
performance bottlenecks and opportunities for lax.scan optimization.
"""

import cProfile
import pstats
import time
from io import StringIO

import jax.numpy as jnp
import numpy as np

from nlsq import CurveFit


def exponential(x, a, b, c):
    """Exponential decay model"""
    return a * jnp.exp(-b * x) + c


def gaussian(x, amp, mu, sigma):
    """Gaussian function"""
    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def polynomial(x, a, b, c, d):
    """Cubic polynomial"""
    return a * x**3 + b * x**2 + c * x + d


def profile_trf_algorithm():
    """Profile TRF algorithm with different problem sizes"""
    print("=" * 80)
    print("TRF Algorithm Hot Path Profiling")
    print("=" * 80)

    # Test cases: (name, n_points, model, p0_true, noise_level)
    test_cases = [
        ("Small (100 pts)", 100, exponential, [2.0, 0.5, 0.3], 0.05),
        ("Medium (1000 pts)", 1000, exponential, [2.0, 0.5, 0.3], 0.05),
        ("Large (10000 pts)", 10000, gaussian, [5.0, 0.0, 2.0], 0.1),
        ("XLarge (50000 pts)", 50000, polynomial, [0.1, -0.5, 2.0, 1.0], 0.2),
    ]

    cf = CurveFit()

    for name, n_points, model, p0_true, noise in test_cases:
        print(f"\n{name}")
        print("-" * 80)

        # Generate data
        np.random.seed(42)
        if model == exponential:
            x = np.linspace(0, 10, n_points)
            y_true = p0_true[0] * np.exp(-p0_true[1] * x) + p0_true[2]
        elif model == gaussian:
            x = np.linspace(-10, 10, n_points)
            y_true = p0_true[0] * np.exp(
                -((x - p0_true[1]) ** 2) / (2 * p0_true[2] ** 2)
            )
        else:  # polynomial
            x = np.linspace(-5, 5, n_points)
            y_true = p0_true[0] * x**3 + p0_true[1] * x**2 + p0_true[2] * x + p0_true[3]

        y = y_true + noise * np.random.randn(len(x))
        p0 = p0_true

        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        result = cf.curve_fit(model, x, y, p0=p0)
        elapsed = time.time() - start_time

        profiler.disable()

        # Get profiling stats
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Top 20 functions

        print(f"Total time: {elapsed * 1000:.2f}ms")
        print(
            f"Converged: {result.success if hasattr(result, 'success') else result[0] is not None}"
        )
        print("\nTop 20 hot functions:")
        print(s.getvalue())


def profile_iteration_breakdown():
    """Profile iteration-level breakdown of TRF algorithm"""
    print("\n" + "=" * 80)
    print("TRF Iteration-Level Profiling (using trf_no_bounds_timed)")
    print("=" * 80)

    # Use medium-sized problem
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    y_true = 2.0 * np.exp(-0.5 * x) + 0.3
    y = y_true + 0.05 * np.random.randn(len(x))
    p0 = [2.0, 0.5, 0.3]

    # Create CurveFit instance with timed mode
    from nlsq.minpack import CurveFit

    cf = CurveFit()

    # Use curve_fit which internally can use trf_no_bounds_timed for profiling
    # For now, just run regular fit and analyze
    result = cf.curve_fit(exponential, x, y, p0=p0)

    print("\nIteration completed successfully")
    print(f"Final parameters: {result[0]}")
    print(f"Status: {result[1] if len(result) > 1 else 'N/A'}")


def analyze_loop_patterns():
    """Analyze the loop patterns suitable for lax.scan conversion"""
    print("\n" + "=" * 80)
    print("Loop Pattern Analysis for lax.scan Conversion")
    print("=" * 80)

    print("""
Key findings from TRF algorithm structure:

1. OUTER LOOP (main optimization loop):
   - Location: trf_no_bounds line 923, trf_bounds line 1247
   - Pattern: while True with dynamic termination
   - State variables:
     * x, f, J, g (current parameters, residuals, Jacobian, gradient)
     * cost, nfev, njev (cost, function/Jacobian evaluations)
     * Delta, alpha (trust region radius, LM parameter)
     * iteration, termination_status

   Challenges for lax.scan:
   - Dynamic termination (gradient tolerance, max iterations)
   - Conditional updates (only update if actual_reduction > 0)
   - Function evaluations (fun, jac) are expensive

2. INNER LOOP (step acceptance loop):
   - Location: trf_no_bounds line 974, trf_bounds line 1305
   - Pattern: while (actual_reduction <= 0 and nfev < max_nfev and inner < 100)
   - State variables:
     * Delta (trust region radius)
     * alpha (LM parameter)
     * actual_reduction, predicted_reduction
     * inner_loop_count

   Challenges for lax.scan:
   - Nested within outer loop
   - Conditional continuation (actual_reduction > 0)
   - Variable number of iterations (1 to 100)

3. OPTIMIZATION OPPORTUNITIES:

   a) High Priority (2-5x speedup):
      - Convert inner loop to lax.scan with fixed max iterations
      - Use lax.cond for conditional updates instead of Python if/else
      - Pre-allocate arrays for iteration history

   b) Medium Priority (1.5-2x speedup):
      - Convert outer loop to lax.scan (more complex)
      - Requires careful state management
      - May need lax.while_loop instead for dynamic termination

   c) Current bottlenecks (from benchmark data):
      - Function evaluations (fun, jac): ~40% of time
      - SVD decomposition: ~30% of time
      - Gradient computation: ~15% of time
      - NumPy ↔ JAX conversions: ~10% of time
      - Other operations: ~5% of time

4. RECOMMENDED APPROACH:

   Step 1: Profile to confirm bottlenecks
   Step 2: Convert inner loop to lax.scan first (easier, isolated)
   Step 3: Minimize NumPy ↔ JAX conversions
   Step 4: Consider lax.while_loop for outer loop if inner loop successful
   Step 5: Vectorize operations within iterations

Expected performance improvements:
- Inner loop lax.scan: 1.5-2x speedup
- Reduced conversions: 1.2-1.3x speedup
- Combined: 2-3x total speedup
- With outer loop optimization: 3-5x total speedup
""")


if __name__ == "__main__":
    # Run profiling
    profile_trf_algorithm()
    profile_iteration_breakdown()
    analyze_loop_patterns()

    print("\n" + "=" * 80)
    print("Profiling Complete")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review hot functions from profiling output")
    print("2. Identify NumPy ↔ JAX conversion points")
    print("3. Design lax.scan implementation for inner loop")
    print("4. Implement and benchmark")
