"""
Converted from enhanced_error_messages_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Demonstration of Enhanced Error Messages in NLSQ
#
# This example shows how NLSQ provides intelligent, actionable error messages
# when optimization fails, helping users debug issues quickly.
#
# ======================================================================

import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit
from nlsq.utils.error_messages import OptimizationError


def example_1_max_iterations():
    """Example 1: Max iterations reached."""
    print("=" * 70)
    print("Example 1: Maximum Iterations Reached")
    print("=" * 70)

    def exponential(x, a, b):
        return a * jnp.exp(-b * x)

    # Generate data
    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.5 * x) + np.random.normal(0, 0.1, 50)

    try:
        # Force failure with very low max_nfev
        _, _ = curve_fit(exponential, x, y, p0=[1, 1], max_nfev=3)
    except OptimizationError as e:
        print("\n❌ Optimization Failed!")
        print("\n" + str(e))
        print("\n" + "-" * 70)
        print("📊 Diagnostic Details:")
        for key, value in e.diagnostics.items():
            print(f"  • {key}: {value}")

        print("\n💡 Actionable Recommendations:")
        for i, rec in enumerate(e.recommendations, 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 70 + "\n")


def example_2_auto_recovery():
    """Example 2: Successful fit after applying recommendations."""
    print("=" * 70)
    print("Example 2: Applying Recommendations - Success!")
    print("=" * 70)

    def exponential(x, a, b):
        return a * jnp.exp(-b * x)

    x = np.linspace(0, 5, 50)
    y = 3 * np.exp(-0.5 * x) + np.random.normal(0, 0.1, 50)

    # First attempt: fails
    print("\n🔴 First attempt (max_nfev=3):")
    try:
        popt, pcov = curve_fit(exponential, x, y, p0=[1, 1], max_nfev=3)
        print("  ✅ Succeeded (unexpected)")
    except OptimizationError as e:
        print(f"  ❌ Failed: {e.reasons[0] if e.reasons else 'Unknown'}")
        print(
            f"  💡 Recommendation: {e.recommendations[0] if e.recommendations else 'Increase max_nfev'}"
        )

    # Second attempt: apply recommendation
    print("\n🟢 Second attempt (max_nfev=100):")
    try:
        popt, pcov = curve_fit(exponential, x, y, p0=[1, 1], max_nfev=100)
        print(f"  ✅ Success! Fitted parameters: a={popt[0]:.3f}, b={popt[1]:.3f}")
        print("  📈 True parameters:           a=3.000, b=0.500")
    except OptimizationError as e:
        print(f"  ❌ Still failed: {e.reasons[0]}")

    print("\n" + "=" * 70 + "\n")


def example_3_diagnostic_analysis():
    """Example 3: Using diagnostic information programmatically."""
    print("=" * 70)
    print("Example 3: Programmatic Error Handling")
    print("=" * 70)

    def gaussian(x, amp, mu, sigma):
        return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))

    x = np.linspace(-5, 5, 100)
    y = 2 * np.exp(-((x - 1) ** 2) / (2 * 0.5**2))

    try:
        _, _ = curve_fit(gaussian, x, y, p0=[1, 0, 1], max_nfev=2)
    except OptimizationError as e:
        print("\n📊 Analyzing Error Diagnostics:")
        print(f"  • Number of reasons: {len(e.reasons)}")
        print(f"  • Number of recommendations: {len(e.recommendations)}")

        # Programmatic decision making
        if any("maximum" in r.lower() for r in e.reasons):
            print("\n🔧 Auto-fix strategy: Increase max_nfev")
            try:
                # Automatically retry with higher max_nfev
                popt, pcov = curve_fit(gaussian, x, y, p0=[1, 0, 1], max_nfev=200)
                print("  ✅ Auto-retry succeeded!")
                print(
                    f"     Fitted: amp={popt[0]:.2f}, mu={popt[1]:.2f}, sigma={popt[2]:.2f}"
                )
            except OptimizationError:
                print("  ❌ Auto-retry failed")

    print("\n" + "=" * 70 + "\n")


def example_4_comparison():
    """Example 4: Compare old vs new error messages."""
    print("=" * 70)
    print("Example 4: Old vs New Error Messages")
    print("=" * 70)

    def difficult(x, a, b, c):
        return a * jnp.sin(b * x) * jnp.exp(-c * x)

    x = np.linspace(0, 10, 50)
    y = 2 * np.sin(3 * x) * np.exp(-0.5 * x)

    print("\n🔴 OLD ERROR (before enhancement):")
    print('  "RuntimeError: Optimal parameters not found: ')
    print('   CONVERGENCE: REL_REDUCTION_OF_F_<=_FACTR*EPSMCH"')
    print("\n  😕 Not helpful! What should I do?")

    print("\n🟢 NEW ERROR (with enhancement):")
    try:
        _, _ = curve_fit(difficult, x, y, p0=[1, 1, 1], max_nfev=3)
    except OptimizationError as e:
        print(f"\n{e}")

    print("\n  ✅ Much better! Clear diagnostics and actionable steps!")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" NLSQ Enhanced Error Messages Demo")
    print("=" * 70 + "\n")

    example_1_max_iterations()
    example_2_auto_recovery()
    example_3_diagnostic_analysis()
    example_4_comparison()

    print("=" * 70)
    print("✅ Demo complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Error messages include detailed diagnostics")
    print("  2. Recommendations are specific and actionable")
    print("  3. Error objects can be used programmatically")
    print("  4. Much easier to debug optimization failures!")
    print("=" * 70 + "\n")
