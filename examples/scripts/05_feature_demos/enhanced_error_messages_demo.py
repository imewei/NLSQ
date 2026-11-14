"""
Converted from enhanced_error_messages_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Demonstration of Enhanced Error Messages in NLSQ
#
# This example shows how NLSQ reports an optimization that stops before it
# converges, and how to detect and handle that programmatically.
#
# Behaviour of the library (see ``CurveFit`` in ``nlsq/core/minpack.py``):
#
# * Exhausting ``max_nfev`` does NOT raise. ``curve_fit`` emits an
#   ``OptimizeWarning`` and returns a best-effort result with
#   ``success=False`` and ``status == 0``. (SciPy raises ``RuntimeError``.)
# * ``OptimizationError`` (with ``reasons``, ``recommendations`` and
#   ``diagnostics``) is raised only for other failure statuses. Simple
#   examples cannot trigger it reliably, so Example 1 builds one explicitly
#   from the failed result to show the diagnostics it carries.
#
# ======================================================================

import warnings

import jax.numpy as jnp
import numpy as np

from nlsq import OptimizeWarning, curve_fit
from nlsq.utils.error_messages import OptimizationError

SEED = 42


def exponential(x, a, b):
    return a * jnp.exp(-b * x)


def fit_with_warnings(*args, **kwargs):
    """Run curve_fit and return (result, list of OptimizeWarning messages)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = curve_fit(*args, **kwargs)
    msgs = [str(w.message) for w in caught if issubclass(w.category, OptimizeWarning)]
    return result, msgs


def example_1_max_iterations():
    """Example 1: Max iterations reached -> warning + success=False."""
    print("=" * 70)
    print("Example 1: Maximum Iterations Reached")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.5 * x) + rng.normal(0, 0.1, 50)

    result, msgs = fit_with_warnings(exponential, x, y, p0=[1, 1], max_nfev=3)
    print("\nFit with max_nfev=3 returned instead of raising:")
    print(f"  success = {result.success}, status = {result.status}")
    print(f"  message = {result.message}")
    print(f"  nfev    = {result.nfev}")
    print(f"  partial parameters: a={result.popt[0]:.3f}, b={result.popt[1]:.3f}")
    print(f"  OptimizeWarning(s) emitted: {len(msgs)}")
    for m in msgs:
        print(f"    - {m}")

    # curve_fit only raises OptimizationError for other failure statuses; here
    # we construct one from the failed result to show the diagnostics it holds.
    err = OptimizationError(result, 1e-8, 1e-8, 1e-8, 3)
    print("\n📊 Diagnostics an OptimizationError carries for this result:")
    for key, value in err.diagnostics.items():
        print(f"  • {key}: {value}")
    print("\n💡 Recommendations it would give:")
    for i, rec in enumerate(err.recommendations, 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 70 + "\n")


def example_2_auto_recovery():
    """Example 2: Re-run with a larger max_nfev."""
    print("=" * 70)
    print("Example 2: Applying the Recommendation - Success!")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.5 * x) + rng.normal(0, 0.1, 50)

    print("\n🔴 First attempt (max_nfev=3):")
    result, msgs = fit_with_warnings(exponential, x, y, p0=[1, 1], max_nfev=3)
    print(f"  success={result.success}; warnings={len(msgs)}")

    print("\n🟢 Second attempt (max_nfev=100):")
    result, msgs = fit_with_warnings(exponential, x, y, p0=[1, 1], max_nfev=100)
    print(f"  success={result.success}; warnings={len(msgs)}; nfev={result.nfev}")
    print(f"  Fitted parameters: a={result.popt[0]:.3f}, b={result.popt[1]:.3f}")
    print("  True parameters:   a=3.000, b=0.500")

    print("\n" + "=" * 70 + "\n")


def example_3_diagnostic_analysis():
    """Example 3: Programmatic handling with an automatic retry loop."""
    print("=" * 70)
    print("Example 3: Programmatic Handling")
    print("=" * 70)

    def gaussian(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    x = np.linspace(-5, 5, 100)
    y = 2 * np.exp(-((x - 1) ** 2) / (2 * 0.5**2))

    max_nfev = 2
    for attempt in range(1, 8):
        result, _ = fit_with_warnings(gaussian, x, y, p0=[1, 0, 1], max_nfev=max_nfev)
        print(
            f"  attempt {attempt}: max_nfev={max_nfev:>3d} -> "
            f"success={result.success}, status={result.status}, nfev={result.nfev}"
        )
        if result.success:
            amp, mu, sigma = result.popt
            print(f"  ✅ Converged: amp={amp:.2f}, mu={mu:.2f}, sigma={sigma:.2f}")
            break
        if result.status != 0:
            print("  ❌ Failed for a reason other than max_nfev; not retrying")
            break
        max_nfev *= 4
    else:
        print("  ❌ Did not converge within the retry budget")

    print("\n" + "=" * 70 + "\n")


def example_4_comparison():
    """Example 4: SciPy vs NLSQ when the evaluation budget is exhausted."""
    print("=" * 70)
    print("Example 4: SciPy vs NLSQ on an Exhausted Budget")
    print("=" * 70)

    def difficult(x, a, b, c):
        return a * jnp.sin(b * x) * jnp.exp(-c * x)

    x = np.linspace(0, 10, 50)
    y = 2 * np.sin(3 * x) * np.exp(-0.5 * x)

    print("\n  scipy.optimize.curve_fit: raises RuntimeError when maxfev is hit.")
    print("  nlsq.curve_fit: warns and returns the best-effort result:")
    result, msgs = fit_with_warnings(difficult, x, y, p0=[1, 1, 1], max_nfev=3)
    print(f"    success = {result.success}, status = {result.status}")
    print(f"    warning = {msgs[0] if msgs else 'none'}")
    print(f"    parameters after {result.nfev} evaluations: {np.round(result.popt, 3)}")
    print("  Always check `result.success` (or catch OptimizeWarning) before")
    print("  trusting the parameters.")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" NLSQ Failure Reporting Demo")
    print("=" * 70 + "\n")

    example_1_max_iterations()
    example_2_auto_recovery()
    example_3_diagnostic_analysis()
    example_4_comparison()

    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Exhausting max_nfev warns (OptimizeWarning) and returns success=False")
    print("  2. Check result.success / result.status instead of relying on exceptions")
    print("  3. OptimizationError carries reasons, recommendations and diagnostics")
    print("  4. Retry with a larger max_nfev, better p0 or different x_scale")
    print("=" * 70 + "\n")
