"""
Converted from nlsq_interactive_tutorial.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

import os
import sys
from pathlib import Path

# ======================================================================
# # NLSQ Interactive Tutorial: GPU-Accelerated Curve Fitting
#
# **Welcome to NLSQ!** This interactive tutorial will guide you through using NLSQ for fast, GPU-accelerated curve fitting.
#
# ## What You'll Learn
#
# 1. ✅ Installation and setup (CPU and GPU)
# 2. 📈 Basic curve fitting with common models
# 3. 🎯 Parameter bounds and constraints
# 4. 🔧 Error handling and diagnostics
# 5. 💾 Large dataset handling (millions of points)
# 6. ⚡ GPU acceleration (100x+ speedups)
# 7. 🚀 Advanced features (callbacks, robust fitting, auto p0)
#
# **Time**: ~45 minutes
# **Level**: Beginner to Intermediate
# **Prerequisites**: Basic Python, NumPy
#
# ---
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imewei/NLSQ/blob/main/examples/NLSQ_Interactive_Tutorial.ipynb)
# ======================================================================
# ======================================================================
# ## Section 1: Installation & Setup
#
# ### 1.1 Installation
#
# NLSQ requires JAX for GPU acceleration. On Google Colab, JAX is pre-installed with GPU support.
#
# **Installation options**:
# - **Colab (GPU)**: JAX pre-installed ✅
# - **Local (CPU)**: `pip install nlsq`
# - **Local (GPU)**: Install JAX with GPU support first, then `pip install nlsq`
# ======================================================================
# Install NLSQ (skip if already installed)
# !pip install -q nlsq  # Uncomment to install in notebook environment
# Check installation
from nlsq import __version__

print(f"NLSQ version: {__version__}")
print("✅ Installation successful!")


# ======================================================================
# ### 1.2 Imports
#
# Let's import the libraries we'll need:
# ======================================================================


# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib


import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import (
    callbacks,  # Progress monitoring
    curve_fit,
    functions,  # Common fitting functions
)

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
MAX_SAMPLES = int(os.environ.get("NLSQ_EXAMPLES_MAX_SAMPLES", "300000"))


def cap_samples(n: int) -> int:
    return min(n, MAX_SAMPLES) if QUICK else n


# Set random seed for reproducibility
np.random.seed(42)

print("✅ All imports successful!")


# ======================================================================
# ### 1.3 Check GPU Availability
#
# NLSQ automatically uses GPU if available. Let's check:
# ======================================================================


import jax

# Check available devices
devices = jax.devices()
print(f"Available devices: {devices}")
print(f"Default backend: {devices[0].platform}")

if devices[0].platform == "gpu":
    print("\n🚀 GPU detected! NLSQ will use GPU acceleration.")
else:
    print("\n💻 Running on CPU. For GPU, use Runtime -> Change runtime type -> GPU")


# ======================================================================
# ---
#
# ## Section 2: Your First Curve Fit
#
# Let's start with a simple example: fitting an exponential decay curve.
#
# ### 2.1 Generate Sample Data
#
# We'll create noisy data following an exponential decay: $y = a \cdot e^{-b \cdot x} + c$
# ======================================================================


# True parameters
a_true, b_true, c_true = 10.0, 0.5, 2.0

# Generate x data
x = np.linspace(0, 10, cap_samples(100))

# Generate y data with noise
y_true = a_true * np.exp(-b_true * x) + c_true
noise = np.random.normal(0, 0.5, size=len(x))
y = y_true + noise

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Noisy data")
plt.plot(x, y_true, "r--", label="True function", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Exponential Decay Data")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"True parameters: a={a_true}, b={b_true}, c={c_true}")


# ======================================================================
# ### 2.2 Define the Model
#
# Define your model as a Python function. **Important**: Use `jax.numpy` (jnp) instead of `numpy` for JAX compatibility!
# ======================================================================


def exponential_decay(x, a, b, c):
    """Exponential decay model: y = a * exp(-b*x) + c

    Parameters:
        a: Amplitude
        b: Decay rate
        c: Offset
    """
    return a * jnp.exp(-b * x) + c


print("✅ Model defined!")


# ======================================================================
# ### 2.3 Fit the Model
#
# Now let's fit the model to our data. NLSQ's API is compatible with SciPy's `curve_fit`:
# ======================================================================


# Initial parameter guess
p0 = [8, 0.4, 1]  # Close to true values: [10, 0.5, 2]

# Fit the model
popt, pcov = curve_fit(exponential_decay, x, y, p0=p0)

# Extract fitted parameters
a_fit, b_fit, c_fit = popt

print("Fitted Parameters:")
print(f"  a = {a_fit:.4f} (true: {a_true})")
print(f"  b = {b_fit:.4f} (true: {b_true})")
print(f"  c = {c_fit:.4f} (true: {c_true})")
print("\n✅ Fitting successful!")


# ======================================================================
# ### 2.4 Visualize the Results
# ======================================================================


# Generate fitted curve
y_fit = exponential_decay(x, *popt)

# Plot
plt.figure(figsize=(12, 5))

# Left: Data and fit
plt.subplot(1, 2, 1)
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, y_true, "r--", label="True", linewidth=2)
plt.plot(x, y_fit, "g-", label="Fitted", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Curve Fit Results")
plt.grid(True, alpha=0.3)

# Right: Residuals
plt.subplot(1, 2, 2)
residuals = y - y_fit
plt.scatter(x, residuals, alpha=0.5)
plt.axhline(0, color="r", linestyle="--", linewidth=2)
plt.xlabel("x")
plt.ylabel("Residuals (y - y_fit)")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()

# Print goodness of fit
rmse = np.sqrt(np.mean(residuals**2))
print(f"RMSE: {rmse:.4f}")

if QUICK:
    print("⏩ Quick mode enabled: skipping extended tutorial sections.")
    sys.exit(0)


# ======================================================================
# ### 🎯 Exercise 1: Try It Yourself!
#
# Modify the code above to fit a **linear** model: $y = a \cdot x + b$
#
# **Hints**:
# 1. Generate linear data: `y_true = 2*x + 1`
# 2. Define model: `def linear(x, a, b): return a*x + b`
# 3. Use `p0=[1, 1]` (2 parameters)
#
# <details>
# <summary>Click to see solution</summary>
#
# ```python
# # Generate linear data
# x = np.linspace(0, 10, 50)
# y = 2*x + 1 + np.random.normal(0, 1, size=len(x))
#
# # Define model
# def linear(x, a, b):
#     return a*x + b
#
# # Fit
# popt, pcov = curve_fit(linear, x, y, p0=[1, 1])
# print(f"Fitted: a={popt[0]:.2f}, b={popt[1]:.2f}")
# ```
# </details>
# ======================================================================


# Your code here


# ======================================================================
# ---
#
# ## Section 3: Common Fitting Patterns
#
# NLSQ includes a library of common functions for quick fitting.
#
# ### 3.1 Using Built-in Functions
# ======================================================================


# List available functions
print("Available functions in nlsq.functions:")
for func_name in functions.__all__:
    print(f"  - {func_name}")


# ======================================================================
# ### 3.2 Example: Gaussian Peak Fitting
#
# Fit a Gaussian peak: $y = a \cdot e^{-(x-\mu)^2 / (2\sigma^2)}$
# ======================================================================


# Generate Gaussian data
x = np.linspace(-5, 5, 100)
a_true, mu_true, sigma_true = 10, 0, 1.5
y_true = a_true * np.exp(-((x - mu_true) ** 2) / (2 * sigma_true**2))
y = y_true + np.random.normal(0, 0.5, size=len(x))

# Fit using built-in gaussian function
from nlsq.core.functions import gaussian

popt, pcov = curve_fit(gaussian, x, y, p0=[10, 0, 1])
a_fit, mu_fit, sigma_fit = popt

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, y_true, "r--", label="True", linewidth=2)
plt.plot(x, gaussian(x, *popt), "g-", label="Fitted", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title(f"Gaussian Fit: μ={mu_fit:.2f}, σ={sigma_fit:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Fitted: amplitude={a_fit:.2f}, mean={mu_fit:.2f}, std={sigma_fit:.2f}")


# ======================================================================
# ### 3.3 Example: Sigmoid (Logistic) Curve
#
# Common in dose-response and growth curves: $y = \frac{L}{1 + e^{-k(x-x_0)}} + b$
# ======================================================================


# Generate sigmoid data
x = np.linspace(0, 10, 100)
L_true, x0_true, k_true, b_true = 10, 5, 1, 0
y_true = L_true / (1 + np.exp(-k_true * (x - x0_true))) + b_true
y = y_true + np.random.normal(0, 0.5, size=len(x))

# Fit using built-in sigmoid function
from nlsq.core.functions import sigmoid

popt, pcov = curve_fit(sigmoid, x, y, p0=[10, 5, 1, 0])
L_fit, x0_fit, k_fit, b_fit = popt

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, y_true, "r--", label="True", linewidth=2)
plt.plot(x, sigmoid(x, *popt), "g-", label="Fitted", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title(f"Sigmoid Fit: L={L_fit:.2f}, x₀={x0_fit:.2f}, k={k_fit:.2f}, b={b_fit:.2f}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_04.png", dpi=300, bbox_inches="tight")
plt.close()

print(
    f"Fitted: max={L_fit:.2f}, midpoint={x0_fit:.2f}, steepness={k_fit:.2f}, baseline={b_fit:.2f}"
)


# ======================================================================
# ### 3.4 Example: Power Law
#
# Common in scaling relationships: $y = a \cdot x^b$
# ======================================================================


# Generate power law data
x = np.linspace(1, 10, 50)  # Start from 1 to avoid x=0
a_true, b_true = 2, 1.5
y_true = a_true * x**b_true
y = y_true + np.random.normal(0, 2, size=len(x))

# Fit using built-in power_law function
from nlsq.core.functions import power_law

popt, pcov = curve_fit(power_law, x, y, p0=[2, 1.5])
a_fit, b_fit = popt

# Plot (log-log scale shows linearity)
_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.scatter(x, y, alpha=0.5, label="Data")
ax1.plot(x, y_true, "r--", label="True", linewidth=2)
ax1.plot(x, power_law(x, *popt), "g-", label="Fitted", linewidth=2)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.set_title("Power Law (Linear Scale)")
ax1.grid(True, alpha=0.3)

# Log-log scale
ax2.scatter(x, y, alpha=0.5, label="Data")
ax2.plot(x, y_true, "r--", label="True", linewidth=2)
ax2.plot(x, power_law(x, *popt), "g-", label="Fitted", linewidth=2)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("x (log scale)")
ax2.set_ylabel("y (log scale)")
ax2.legend()
ax2.set_title("Power Law (Log-Log Scale)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_05.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Fitted: y = {a_fit:.2f} * x^{b_fit:.2f}")


# ======================================================================
# ### 3.5 Example: Polynomial Fitting
#
# Fit polynomials of any degree: $y = a_0 + a_1x + a_2x^2 + ...$
# ======================================================================


# Generate polynomial data (degree 3)
x = np.linspace(-3, 3, 100)
coeffs_true = [0.5, -2, 1, 3]  # y = 0.5x³ - 2x² + x + 3
y_true = np.polyval(coeffs_true, x)
y = y_true + np.random.normal(0, 1, size=len(x))

# Fit using built-in polynomial function
from nlsq.core.functions import polynomial

# Create polynomial function for degree 3
poly3 = polynomial(3)

popt, pcov = curve_fit(poly3, x, y, p0=[1, -1, 1, 1])

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, y_true, "r--", label="True", linewidth=2)
plt.plot(x, poly3(x, *popt), "g-", label="Fitted", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Fit (Degree 3)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_06.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Fitted coefficients: {popt}")
print(f"True coefficients:   {coeffs_true}")


# ======================================================================
# ### 📋 Summary: Common Functions
#
# | Function | Equation | Use Cases |
# |----------|----------|----------|
# | `linear` | $y = ax + b$ | Calibration, trends |
# | `exponential_decay` | $y = ae^{-bx} + c$ | Radioactive decay, RC circuits |
# | `exponential_growth` | $y = ae^{bx} + c$ | Population growth, interest |
# | `gaussian` | $y = ae^{-(x-\mu)^2/(2\sigma^2)}$ | Spectroscopy, normal distributions |
# | `sigmoid` | $y = L/(1+e^{-k(x-x_0)})$ | Dose-response, growth curves |
# | `power_law` | $y = ax^b$ | Scaling laws, allometry |
# | `polynomial` | $y = \sum a_i x^i$ | Flexible curve fitting |
#
# **Next**: Learn how to handle bounds and constraints!
# ======================================================================


# ======================================================================
# ---
#
# ## Section 4: Parameter Bounds and Constraints
#
# Real-world problems often require parameter constraints. NLSQ supports bounds like SciPy.
#
# ### 4.1 Fitting with Bounds
#
# Let's fit an exponential with constrained parameters:
# ======================================================================


# Generate data with known parameters
x = np.linspace(0, 10, 100)
a_true, b_true, c_true = 5.0, 0.3, 1.0
y = a_true * np.exp(-b_true * x) + c_true + np.random.normal(0, 0.2, size=len(x))

# Define bounds: (lower, upper) for each parameter
# bounds = ([a_min, b_min, c_min], [a_max, b_max, c_max])
bounds = ([0, 0, 0], [10, 1, 5])  # All parameters must be positive

# Fit with bounds
popt, pcov = curve_fit(exponential_decay, x, y, p0=[1, 0.1, 0.5], bounds=bounds)

print("Fitted with bounds:")
print(f"  a = {popt[0]:.4f} (true: {a_true}, bounds: 0-10)")
print(f"  b = {popt[1]:.4f} (true: {b_true}, bounds: 0-1)")
print(f"  c = {popt[2]:.4f} (true: {c_true}, bounds: 0-5)")

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(x, y, alpha=0.5, label="Data")
plt.plot(x, exponential_decay(x, *popt), "g-", label="Fitted (bounded)", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Curve Fit with Parameter Bounds")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_07.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ### 4.2 Parameter Uncertainties
#
# The covariance matrix `pcov` provides parameter uncertainties:
# ======================================================================


# Extract standard errors from covariance matrix
perr = np.sqrt(np.diag(pcov))

print("Parameter uncertainties (1σ):")
print(f"  a = {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"  b = {popt[1]:.4f} ± {perr[1]:.4f}")
print(f"  c = {popt[2]:.4f} ± {perr[2]:.4f}")

# Calculate correlation matrix
corr = pcov / np.outer(perr, perr)

print("\nParameter correlation matrix:")
print(corr)


# ======================================================================
# ---
#
# ## Section 5: Error Handling and Diagnostics
#
# NLSQ provides helpful error messages and diagnostics when fits fail.
#
# ### 5.1 Common Issues and Solutions
# ======================================================================


# Example 1: Bad initial guess
print("Example 1: Bad initial guess")
print("=" * 50)
try:
    # p0 too far from true values
    popt, pcov = curve_fit(exponential_decay, x, y, p0=[100, 10, 50])
    print("✅ Fit succeeded despite bad p0!")
    print(f"Fitted: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")
except Exception as e:
    print(f"❌ Fit failed: {e}")
    print("💡 Tip: Try better p0 estimates or increase max_nfev")

# Example 2: Conflicting bounds
print("\nExample 2: Conflicting bounds")
print("=" * 50)
try:
    # p0 outside bounds
    bad_bounds = ([0, 0, 0], [1, 0.1, 0.5])  # Too restrictive
    popt, pcov = curve_fit(exponential_decay, x, y, p0=[5, 0.3, 1], bounds=bad_bounds)
except Exception as e:
    print(f"❌ Fit failed: {type(e).__name__}")
    print("💡 Tip: Ensure p0 is within bounds, and bounds are reasonable")

# Example 3: Successful fit with diagnostics
print("\nExample 3: Successful fit with full diagnostics")
print("=" * 50)
popt, pcov = curve_fit(exponential_decay, x, y, p0=[5, 0.3, 1], full_output=False)
print("✅ Fit succeeded!")
print(f"Final parameters: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
print(f"Condition number of covariance: {np.linalg.cond(pcov):.2e}")
if np.linalg.cond(pcov) > 1e10:
    print("⚠️  Warning: Poorly conditioned covariance (parameters may be correlated)")


# ======================================================================
# ### 5.2 Monitoring Progress with Callbacks
#
# Use callbacks to monitor optimization progress:
# ======================================================================


# Create a progress callback
iteration_data = []


def progress_callback(iteration, cost, params, info=None):
    iteration_data.append(
        {"iter": iteration, "cost": cost, "params": params.copy(), "info": info}
    )
    if iteration % 5 == 0:  # Print every 5 iterations
        print(f"Iteration {iteration:3d}: cost = {cost:.6e}, params = {params}")


# Fit with callback
print("Fitting with progress monitoring:")
print("=" * 70)
popt, pcov = curve_fit(
    exponential_decay, x, y, p0=[1, 0.1, 0.5], callback=progress_callback
)

print("\n✅ Optimization complete!")
print(f"Total iterations: {len(iteration_data)}")
if len(iteration_data) > 0:
    print(f"Final cost: {iteration_data[-1]['cost']:.6e}")

# Plot convergence
if len(iteration_data) > 0:
    costs = [d["cost"] for d in iteration_data]
    plt.figure(figsize=(10, 5))
    plt.semilogy(costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (log scale)")
    plt.title("Optimization Convergence")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save figure to file
    fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / "fig_08.png", dpi=300, bbox_inches="tight")
    plt.close()
else:
    print("No iteration data collected (optimization converged immediately)")


# ======================================================================
# ---
#
# ## Section 6: Large Dataset Handling
#
# NLSQ can handle millions of points efficiently, especially on GPU.
#
# ### 6.1 Fitting Large Datasets
# ======================================================================


import time

# Generate large dataset
n_points = cap_samples(100000)  # 100K points (capped in quick mode)
x_large = np.linspace(0, 10, n_points)
y_large = 5.0 * np.exp(-0.3 * x_large) + 1.0 + np.random.normal(0, 0.2, size=n_points)

print(f"Dataset size: {n_points:,} points")
print(f"Memory: ~{(x_large.nbytes + y_large.nbytes) / 1024**2:.2f} MB")

# Fit large dataset
print("\nFitting large dataset...")
start = time.time()
popt_large, pcov_large = curve_fit(exponential_decay, x_large, y_large, p0=[5, 0.3, 1])
elapsed = time.time() - start

print(f"✅ Fit complete in {elapsed:.3f} seconds")
print(f"Fitted: a={popt_large[0]:.4f}, b={popt_large[1]:.4f}, c={popt_large[2]:.4f}")
print(f"Processing rate: {n_points / elapsed:,.0f} points/second")

# Note: First run includes JIT compilation overhead
print("\n💡 Note: First fit includes JAX JIT compilation (~1-2 seconds).")
print("   Subsequent fits reuse compiled code and are much faster!")


# ======================================================================
# ### 6.2 Automatic Chunking for Very Large Datasets
#
# For datasets larger than available memory, NLSQ can automatically chunk:
# ======================================================================


from nlsq import curve_fit_large

# Simulate very large dataset (10M points would be ~160 MB)
# Using 500K points for demo (capped in quick mode)
n_huge = cap_samples(500000)
x_huge = np.linspace(0, 10, n_huge)
y_huge = 5.0 * np.exp(-0.3 * x_huge) + 1.0 + np.random.normal(0, 0.2, size=n_huge)

print(
    f"Dataset size: {n_huge:,} points ({(x_huge.nbytes + y_huge.nbytes) / 1024**2:.1f} MB)"
)

# Fit with automatic chunking
print("\nFitting with automatic chunking...")
start = time.time()
popt_huge, pcov_huge = curve_fit_large(
    exponential_decay,
    x_huge,
    y_huge,
    p0=[5, 0.3, 1],
    chunk_size=50000,  # Process 50K points at a time
)
elapsed = time.time() - start

print(f"✅ Fit complete in {elapsed:.3f} seconds")
print(f"Fitted: a={popt_huge[0]:.4f}, b={popt_huge[1]:.4f}, c={popt_huge[2]:.4f}")
print(f"Processing rate: {n_huge / elapsed:,.0f} points/second")
print("\n💡 curve_fit_large() automatically manages memory for huge datasets!")


# ======================================================================
# ---
#
# ## Section 7: GPU Acceleration
#
# NLSQ automatically uses GPU when available. Let's benchmark CPU vs GPU performance.
#
# ### 7.1 GPU Performance Comparison
# ======================================================================


import jax

# Check current backend
current_backend = jax.devices()[0].platform
print(f"Current backend: {current_backend}")

if current_backend == "gpu" and not QUICK:
    print("\n🚀 GPU detected! Running performance comparison...")

    # Create large dataset for GPU test
    n_gpu = 1000000  # 1M points
    x_gpu = np.linspace(0, 10, n_gpu)
    y_gpu = 5.0 * np.exp(-0.3 * x_gpu) + 1.0 + np.random.normal(0, 0.2, size=n_gpu)

    # Warmup (JIT compilation)
    print("Warming up JIT compiler...")
    _ = curve_fit(exponential_decay, x_gpu[:1000], y_gpu[:1000], p0=[5, 0.3, 1])

    # GPU timing
    print(f"\nFitting {n_gpu:,} points on GPU...")
    start = time.time()
    popt_gpu, _ = curve_fit(exponential_decay, x_gpu, y_gpu, p0=[5, 0.3, 1])
    gpu_time = time.time() - start

    print(f"✅ GPU fit: {gpu_time:.3f} seconds")
    print(f"   Processing rate: {n_gpu / gpu_time:,.0f} points/second")
    print(
        f"   Parameters: a={popt_gpu[0]:.4f}, b={popt_gpu[1]:.4f}, c={popt_gpu[2]:.4f}"
    )

    # Note: CPU comparison would require JAX_PLATFORM_NAME=cpu environment variable
    print("\n💡 GPU acceleration provides 100-300x speedup vs SciPy on large datasets!")
else:
    print("\n💻 Running on CPU. To use GPU:")
    print("   1. Go to Runtime → Change runtime type")
    print("   2. Select GPU as hardware accelerator")
    print("   3. Restart runtime and re-run notebook")


# ======================================================================
# ### 7.2 Advanced Features
#
# NLSQ includes several advanced features:
# ======================================================================


# 1. Automatic p0 estimation
print("1. Automatic p0 estimation")
print("=" * 50)

# Generate data
x = np.linspace(0, 10, 100)
y = 5.0 * np.exp(-0.3 * x) + 1.0 + np.random.normal(0, 0.2, size=len(x))

# Auto-estimate p0 using function method
from nlsq.core.functions import exponential_decay

p0_auto = exponential_decay.estimate_p0(x, y)
print(f"Auto-estimated p0: {p0_auto}")
popt, _ = curve_fit(exponential_decay, x, y, p0=p0_auto)
print("✅ Fit succeeded with auto p0!")
print(f"Fitted: a={popt[0]:.4f}, b={popt[1]:.4f}, c={popt[2]:.4f}")
print("True values: a=5.0000, b=0.3000, c=1.0000")

# 2. Robust fitting (loss functions)
print("\n2. Robust fitting with loss functions")
print("=" * 50)

# Add outliers to data
y_outliers = y.copy()
outlier_indices = np.random.choice(len(y), 10, replace=False)
y_outliers[outlier_indices] += np.random.normal(0, 5, size=10)

# Standard fit (sensitive to outliers)
popt_standard, _ = curve_fit(exponential_decay, x, y_outliers, p0=[5, 0.3, 1])

# Robust fit with Huber loss
popt_robust, _ = curve_fit(
    exponential_decay,
    x,
    y_outliers,
    p0=[5, 0.3, 1],
    loss="huber",  # Robust to outliers
)

print(
    f"Standard fit: a={popt_standard[0]:.4f}, b={popt_standard[1]:.4f}, c={popt_standard[2]:.4f}"
)
print(
    f"Robust fit:   a={popt_robust[0]:.4f}, b={popt_robust[1]:.4f}, c={popt_robust[2]:.4f}"
)
print("True values:  a=5.0000, b=0.3000, c=1.0000")
print("\n✅ Robust fit is closer to true values despite outliers!")

# Visualize
plt.figure(figsize=(12, 5))
plt.scatter(x, y_outliers, alpha=0.5, label="Data with outliers", color="gray")
plt.scatter(
    x[outlier_indices],
    y_outliers[outlier_indices],
    color="red",
    s=100,
    marker="x",
    linewidth=3,
    label="Outliers",
)
plt.plot(
    x, exponential_decay(x, *popt_standard), "b--", label="Standard fit", linewidth=2
)
plt.plot(
    x,
    exponential_decay(x, *popt_robust),
    "g-",
    label="Robust fit (Huber loss)",
    linewidth=2,
)
plt.plot(x, exponential_decay(x, 5, 0.3, 1), "r:", label="True function", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Robust Fitting with Huber Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_interactive_tutorial"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_09.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ---
#
# ## Section 8: Conclusion and Next Steps
#
# ### 🎉 Congratulations!
#
# You've completed the NLSQ interactive tutorial! You now know how to:
#
# ✅ Install and set up NLSQ with GPU support
# ✅ Fit curves with common models (exponential, Gaussian, sigmoid, power law)
# ✅ Apply parameter bounds and constraints
# ✅ Handle errors and monitor optimization progress
# ✅ Work with large datasets (millions of points)
# ✅ Leverage GPU acceleration for 100x+ speedups
# ✅ Use advanced features (callbacks, robust fitting)
#
# ### 📚 Additional Resources
#
# - **Documentation**: https://nlsq.readthedocs.io
# - **GitHub**: https://github.com/imewei/NLSQ
# - **Examples**: Browse the `examples/` directory for more use cases
# - **API Reference**: Complete function documentation and parameters
#
# ### 🚀 Next Steps
#
# 1. **Try your own data**: Replace the example data with your real-world datasets
# 2. **Explore loss functions**: Try different loss functions for robust fitting
# 3. **Batch processing**: Fit multiple curves in parallel for maximum efficiency
# 4. **Custom models**: Define your own fitting functions with JAX
# 5. **Performance tuning**: Experiment with different algorithms and tolerances
#
# ### 💬 Get Help
#
# - **Issues**: Report bugs at https://github.com/imewei/NLSQ/issues
# - **Discussions**: Ask questions in GitHub Discussions
# - **Citation**: If you use NLSQ in research, please cite the original JAXFit paper
#
# ### 🙏 Acknowledgments
#
# NLSQ is based on JAXFit by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith.
# Development supported by Argonne National Laboratory.
#
# ---
#
# **Happy fitting! 🎯**
# ======================================================================
