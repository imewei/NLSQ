"""
Converted from function_library_demo.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""


# ======================================================================
# # Demonstration of NLSQ Common Function Library
#
# This script demonstrates the pre-built functions in NLSQ that make curve fitting
# trivial for common use cases. All functions include:
#
# - Automatic parameter estimation (p0='auto')
# - Reasonable default bounds
# - JAX/GPU acceleration
# - Comprehensive docstrings
#
# No manual parameter guessing needed!
#
# ======================================================================


# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit, functions

# Use NLSQ_EXAMPLES_TMPDIR if set, otherwise system temp
TMPDIR = os.environ.get("NLSQ_EXAMPLES_TMPDIR", tempfile.gettempdir())


def demo_linear():
    """Example 1: Linear function - y = a*x + b"""
    print("=" * 70)
    print("Example 1: Linear Function")
    print("=" * 70)

    # Generate data with noise
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y_true = 2.5 * x + 3.0
    y = y_true + np.random.normal(0, 1.0, len(x))

    # Fit without specifying p0 - automatic estimation!
    popt, pcov = curve_fit(functions.linear, x, y, p0="auto")

    print(f"\n✓ Fitted: slope={popt[0]:.2f}, intercept={popt[1]:.2f}")
    print("  True:   slope=2.50, intercept=3.00")
    print(f"  Error:  {np.abs(popt[0] - 2.5):.2f}, {np.abs(popt[1] - 3.0):.2f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, y_true, "g--", label="True")
    plt.plot(x, functions.linear(x, *popt), "r-", label="Fitted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear: y = ax + b")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\n")


def demo_exponential_decay():
    """Example 2: Exponential decay - y = a*exp(-b*x) + c"""
    print("=" * 70)
    print("Example 2: Exponential Decay")
    print("=" * 70)

    # Generate radioactive decay data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    a_true, b_true, c_true = 100.0, 0.5, 10.0
    y_true = a_true * np.exp(-b_true * x) + c_true
    y = y_true + np.random.normal(0, 2.0, len(x))

    # Fit with automatic p0
    popt, pcov = curve_fit(functions.exponential_decay, x, y, p0="auto")

    # Calculate half-life
    half_life_fitted = np.log(2) / popt[1]
    half_life_true = np.log(2) / b_true

    print(
        f"\n✓ Fitted: amplitude={popt[0]:.1f}, rate={popt[1]:.3f}, offset={popt[2]:.1f}"
    )
    print(f"  True:   amplitude={a_true:.1f}, rate={b_true:.3f}, offset={c_true:.1f}")
    print(f"\n  Half-life (fitted): {half_life_fitted:.2f}")
    print(f"  Half-life (true):   {half_life_true:.2f}")

    # Plot
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, y_true, "g--", label="True")
    plt.plot(x, functions.exponential_decay(x, *popt), "r-", label="Fitted")
    plt.xlabel("Time")
    plt.ylabel("Activity")
    plt.title("Exponential Decay")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(TMPDIR, "nlsq_demo_1.png")
    plt.savefig(outfile, dpi=100)
    plt.close()

    print(f"  Plot saved to {outfile}\n")


def demo_gaussian():
    """Example 3: Gaussian peak - common in spectroscopy"""
    print("=" * 70)
    print("Example 3: Gaussian Peak (Spectroscopy)")
    print("=" * 70)

    # Generate spectral peak data
    np.random.seed(42)
    x = np.linspace(0, 20, 300)
    amp_true, mu_true, sigma_true = 50.0, 12.0, 1.5
    y_true = amp_true * np.exp(-((x - mu_true) ** 2) / (2 * sigma_true**2))
    y = y_true + np.random.normal(0, 1.0, len(x))

    # Fit with automatic p0
    popt, pcov = curve_fit(functions.gaussian, x, y, p0="auto")

    # Calculate FWHM (Full Width at Half Maximum)
    fwhm_fitted = 2.355 * popt[2]
    fwhm_true = 2.355 * sigma_true

    print(
        f"\n✓ Fitted: amplitude={popt[0]:.1f}, center={popt[1]:.2f}, width={popt[2]:.2f}"
    )
    print(
        f"  True:   amplitude={amp_true:.1f}, center={mu_true:.2f}, width={sigma_true:.2f}"
    )
    print(f"\n  FWHM (fitted): {fwhm_fitted:.2f}")
    print(f"  FWHM (true):   {fwhm_true:.2f}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.3, s=10, label="Data")
    plt.plot(x, y_true, "g--", linewidth=2, label="True")
    plt.plot(x, functions.gaussian(x, *popt), "r-", linewidth=2, label="Fitted")
    plt.axvline(
        popt[1], color="r", linestyle=":", alpha=0.5, label=f"Peak at {popt[1]:.1f}"
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.title("Gaussian Peak")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\n")


def demo_sigmoid():
    """Example 4: Sigmoid - dose-response curve"""
    print("=" * 70)
    print("Example 4: Sigmoid (Dose-Response)")
    print("=" * 70)

    # Generate dose-response data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    L_true, x0_true, k_true, b_true = 100.0, 5.0, 1.5, 10.0
    y_true = L_true / (1 + np.exp(-k_true * (x - x0_true))) + b_true
    y = y_true + np.random.normal(0, 3.0, len(x))

    # Fit with automatic p0
    popt, pcov = curve_fit(functions.sigmoid, x, y, p0="auto")

    print(
        f"\n✓ Fitted: max={popt[0]:.1f}, EC50={popt[1]:.2f}, steepness={popt[2]:.2f}, baseline={popt[3]:.1f}"
    )
    print(
        f"  True:   max={L_true:.1f}, EC50={x0_true:.2f}, steepness={k_true:.2f}, baseline={b_true:.1f}"
    )
    print(f"\n  EC50 (half-maximal effective concentration): {popt[1]:.2f}")

    # Plot
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, y_true, "g--", label="True")
    plt.plot(x, functions.sigmoid(x, *popt), "r-", label="Fitted")
    plt.axhline(popt[0] / 2 + popt[3], color="gray", linestyle=":", alpha=0.5)
    plt.axvline(
        popt[1], color="r", linestyle=":", alpha=0.5, label=f"EC50={popt[1]:.1f}"
    )
    plt.xlabel("Dose (concentration)")
    plt.ylabel("Response")
    plt.title("Sigmoid (Dose-Response)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(TMPDIR, "nlsq_demo_2.png")
    plt.savefig(outfile, dpi=100)
    plt.close()

    print(f"  Plot saved to {outfile}\n")


def demo_power_law():
    """Example 5: Power law - allometric scaling"""
    print("=" * 70)
    print("Example 5: Power Law (Allometric Scaling)")
    print("=" * 70)

    # Generate allometric scaling data
    # Example: metabolic rate ∝ mass^0.75
    np.random.seed(42)
    x = np.linspace(1, 100, 50)
    a_true, b_true = 3.0, 0.75
    y_true = a_true * x**b_true
    y = y_true + np.random.normal(0, 0.5 * np.sqrt(y_true), len(x))

    # Fit with automatic p0
    popt, pcov = curve_fit(functions.power_law, x, y, p0="auto")

    print(f"\n✓ Fitted: prefactor={popt[0]:.2f}, exponent={popt[1]:.3f}")
    print(f"  True:   prefactor={a_true:.2f}, exponent={b_true:.3f}")
    print(f"\n  Scaling exponent: {popt[1]:.3f} (Kleiber's law predicts 0.75)")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, y_true, "g--", label="True")
    plt.plot(x, functions.power_law(x, *popt), "r-", label="Fitted")
    plt.xlabel("Body Mass (kg)")
    plt.ylabel("Metabolic Rate")
    plt.title("Power Law: y = ax^b")
    plt.legend()
    plt.grid(True, alpha=0.3)
    print("\n")


def demo_polynomial():
    """Example 6: Polynomial - quadratic fit"""
    print("=" * 70)
    print("Example 6: Polynomial (Quadratic)")
    print("=" * 70)

    # Create quadratic polynomial function
    quadratic = functions.polynomial(2)

    # Generate quadratic data
    np.random.seed(42)
    x = np.linspace(-5, 5, 60)
    # True: y = 0.5x² - 2x + 3
    coeffs_true = [0.5, -2, 3]
    y_true = np.polyval(coeffs_true, x)
    y = y_true + np.random.normal(0, 1.0, len(x))

    # Fit with automatic p0
    popt, pcov = curve_fit(quadratic, x, y, p0="auto")

    print(f"\n✓ Fitted: coeffs = [{popt[0]:.2f}, {popt[1]:.2f}, {popt[2]:.2f}]")
    print("  True:   coeffs = [0.50, -2.00, 3.00]")
    print(f"\n  Polynomial: y = {popt[0]:.2f}x² + {popt[1]:.2f}x + {popt[2]:.2f}")

    # Find vertex
    vertex_x = -popt[1] / (2 * popt[0])
    vertex_y = np.polyval(popt, vertex_x)
    print(f"  Vertex at ({vertex_x:.2f}, {vertex_y:.2f})")

    # Plot
    plt.subplot(1, 2, 2)
    plt.scatter(x, y, alpha=0.5, label="Data")
    plt.plot(x, y_true, "g--", label="True")
    plt.plot(x, quadratic(x, *popt), "r-", label="Fitted")
    plt.plot(vertex_x, vertex_y, "ro", markersize=8, label="Vertex")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Quadratic: y = ax² + bx + c")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    outfile = os.path.join(TMPDIR, "nlsq_demo_3.png")
    plt.savefig(outfile, dpi=100)
    plt.close()

    print(f"  Plot saved to {outfile}\n")


def demo_comparison():
    """Example 7: Comparison - Manual p0 vs Auto p0"""
    print("=" * 70)
    print("Example 7: Manual p0 vs Auto p0 (Time Comparison)")
    print("=" * 70)

    import time

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 5 * np.exp(-0.3 * x) + 2 + np.random.normal(0, 0.2, len(x))

    # Method 1: Manual p0 (old way)
    start = time.time()
    popt_manual, pcov_manual = curve_fit(
        functions.exponential_decay,
        x,
        y,
        p0=[5, 0.3, 2],  # User has to guess these!
    )
    time_manual = time.time() - start

    # Method 2: Auto p0 (new way)
    start = time.time()
    popt_auto, pcov_auto = curve_fit(
        functions.exponential_decay,
        x,
        y,
        p0="auto",  # Automatic estimation!
    )
    time_auto = time.time() - start

    print(f"\n  Manual p0:   {popt_manual}")
    print(f"  Auto p0:     {popt_auto}")
    print(f"\n  Difference:  {np.max(np.abs(popt_manual - popt_auto)):.6f}")
    print(f"\n  Time (manual): {time_manual * 1000:.2f}ms")
    print(f"  Time (auto):   {time_auto * 1000:.2f}ms")
    print("\n  ✓ Auto p0 is just as accurate but saves user effort!\n")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NLSQ COMMON FUNCTION LIBRARY DEMO" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n")

    # Run all demos
    demo_linear()
    demo_exponential_decay()
    demo_gaussian()
    demo_sigmoid()
    demo_power_law()
    demo_polynomial()
    demo_comparison()

    # Summary
    print("=" * 70)
    print("Summary: Available Functions")
    print("=" * 70)
    print("""
Available pre-built functions:

  1. linear(x, a, b)
     → y = a*x + b

  2. exponential_decay(x, a, b, c)
     → y = a*exp(-b*x) + c

  3. exponential_growth(x, a, b, c)
     → y = a*exp(b*x) + c

  4. gaussian(x, amp, mu, sigma)
     → y = amp*exp(-(x-mu)²/(2*sigma²))

  5. sigmoid(x, L, x0, k, b)
     → y = L/(1 + exp(-k*(x-x0))) + b

  6. power_law(x, a, b)
     → y = a*x^b

  7. polynomial(degree)
     → Creates polynomial of any degree

All functions include:
  ✓ Automatic p0 estimation (.estimate_p0() method)
  ✓ Reasonable default bounds (.bounds() method)
  ✓ JAX/GPU acceleration
  ✓ Comprehensive docstrings

Usage:
  >>> from nlsq import curve_fit, functions
  >>> popt, pcov = curve_fit(functions.gaussian, x, y, p0='auto')

No manual parameter guessing needed! 🎉
    """)
    print("=" * 70)
    print("\n")


if __name__ == "__main__":
    main()
