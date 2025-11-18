"""
Converted from nlsq_challenges.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # NLSQ Challenges: Hands-On Exercises
#
# **Level**: Beginner to Advanced
# **Time**: 60-90 minutes (self-paced)
# **Prerequisites**: NLSQ Quickstart
#
# ## Overview
#
# This notebook contains **7 progressively challenging exercises** to test and improve your NLSQ skills. Each challenge includes:
# - Problem statement with data
# - Learning objectives
# - Hints (if you get stuck)
# - Hidden solution (click to reveal)
# - Self-assessment rubric
#
# ### How to Use This Notebook
#
# 1. **Try each challenge** without looking at the solution
# 2. **Use hints** if stuck (they're ordered by specificity)
# 3. **Check solution** only after attempting
# 4. **Compare** your approach to the provided solution
# 5. **Self-assess** using the rubric
#
# ### Difficulty Levels
#
# - 🟢 **Beginner**: Basic curve fitting, 1-2 parameters
# - 🟡 **Intermediate**: Multi-parameter models, constraints
# - 🔴 **Advanced**: Complex models, troubleshooting, optimization
#
# Good luck!
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import CurveFit

# Reproducible random data
np.random.seed(42)

print("✓ Setup complete - ready for challenges!")


# ======================================================================
# ## Challenge 1: Simple Linear Fit 🟢
#
# **Difficulty**: Beginner
# **Time**: 5-10 minutes
#
# ### Problem
#
# You have experimental data from a spring force experiment: force vs. displacement. Fit a linear model `F = k * x` to determine the spring constant `k`.
#
# **Learning Objectives**:
# - Basic `curve_fit` usage
# - Defining simple models
# - Extracting fitted parameters
#
# **Data**:
# ======================================================================


# Spring displacement (meters)
x_spring = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

# Measured force (Newtons) with experimental noise
F_spring = np.array([0.02, 2.15, 3.98, 6.12, 7.89, 10.05])

# True spring constant: k ≈ 20 N/m

plt.figure(figsize=(8, 5))
plt.plot(x_spring, F_spring, "o", ms=8, label="Experimental data")
plt.xlabel("Displacement (m)")
plt.ylabel("Force (N)")
plt.title("Spring Force vs. Displacement")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_challenges"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()

print("Your task: Fit F = k * x and report k ± uncertainty")


# ======================================================================
# <details>
# <summary><b>💡 Hint 1</b> (Click to expand)</summary>
#
# Define a function `linear_model(x, k)` that returns `k * x`.
# </details>
#
# <details>
# <summary><b>💡 Hint 2</b> (Click to expand)</summary>
#
# Use `CurveFit().curve_fit(model, x, y, p0=[initial_guess])`. Try `p0=[10]` as initial guess.
# </details>
#
# <details>
# <summary><b>💡 Hint 3</b> (Click to expand)</summary>
#
# Uncertainty: `np.sqrt(pcov[0, 0])` gives the standard error on `k`.
# </details>
# ======================================================================


# ═══════════════════════════════════════════════════════════
# YOUR SOLUTION HERE
# ═══════════════════════════════════════════════════════════

# Write your code below:


# ======================================================================
# <details>
# <summary><b>✅ Click to reveal SOLUTION</b></summary>
#
# ```python
# # Solution for Challenge 1
#
# def linear_model(x, k):
#     return k * x
#
# cf = CurveFit()
# popt, pcov = cf.curve_fit(
#     linear_model,
#     jnp.array(x_spring),
#     jnp.array(F_spring),
#     p0=[10.0]
# )
#
# k_fit = popt[0]
# k_err = np.sqrt(pcov[0, 0])
#
# print(f"Spring constant: k = {k_fit:.2f} ± {k_err:.2f} N/m")
#
# # Visualization
# x_fit = np.linspace(0, 0.5, 100)
# plt.plot(x_spring, F_spring, 'o', ms=8, label='Data')
# plt.plot(x_fit, linear_model(x_fit, k_fit), 'r-', lw=2, label=f'Fit: k={k_fit:.2f}')
# plt.xlabel('Displacement (m)')
# plt.ylabel('Force (N)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
# ```
#
# **Expected Result**: `k ≈ 20.0 ± 0.3 N/m`
# </details>
#
# ### Self-Assessment
#
# - [ ] Model function defined correctly
# - [ ] `curve_fit` called with proper arguments
# - [ ] Spring constant extracted from `popt`
# - [ ] Uncertainty calculated from `pcov`
# - [ ] Result is reasonable (k ≈ 20 N/m)
# ======================================================================


# ======================================================================
# ## Challenge 2: Exponential Decay with Offset 🟡
#
# **Difficulty**: Intermediate
# **Time**: 10-15 minutes
#
# ### Problem
#
# Radioactive decay data shows counts vs. time. However, there's background radiation (offset). Fit the model:
#
# $$N(t) = N_0 e^{-\lambda t} + N_{\text{bg}}$$
#
# Find the decay constant λ, initial counts N₀, and background N_bg.
#
# **Learning Objectives**:
# - Multi-parameter fitting
# - Physical interpretation of parameters
# - Plotting residuals
#
# **Data**:
# ======================================================================


# Time (hours)
t_decay = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

# Counts (with background and noise)
N_decay = np.array(
    [1050, 670, 445, 310, 230, 185, 160, 145, 138, 133, 131]
)  # True: N0=1000, λ=0.2/hr, Nbg=130

plt.figure(figsize=(8, 5))
plt.plot(t_decay, N_decay, "o", ms=8)
plt.xlabel("Time (hours)")
plt.ylabel("Counts")
plt.title("Radioactive Decay with Background")
plt.grid(alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_challenges"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()

print("Your task: Fit N(t) = N0 * exp(-λ * t) + Nbg")
print("Report: N0, λ (with units!), Nbg, and half-life = ln(2)/λ")


# ======================================================================
# <details>
# <summary><b>💡 Hint 1</b> (Click to expand)</summary>
#
# Initial guess: N0 ≈ first data point - last data point, λ ≈ 0.1-0.5, Nbg ≈ last data point.
# </details>
#
# <details>
# <summary><b>💡 Hint 2</b> (Click to expand)</summary>
#
# Use `bounds` to ensure physical parameters: N0 > 0, λ > 0, Nbg ≥ 0.
# </details>
# ======================================================================


# ═══════════════════════════════════════════════════════════
# YOUR SOLUTION HERE
# ═══════════════════════════════════════════════════════════


# ======================================================================
# <details>
# <summary><b>✅ Click to reveal SOLUTION</b></summary>
#
# ```python
# def decay_with_background(t, N0, lambda_decay, Nbg):
#     return N0 * jnp.exp(-lambda_decay * t) + Nbg
#
# # Initial guess
# p0 = [1000, 0.2, 130]
#
# # Bounds (all parameters positive)
# bounds = ([0, 0, 0], [np.inf, 1, 500])
#
# cf = CurveFit()
# popt, pcov = cf.curve_fit(
#     decay_with_background,
#     jnp.array(t_decay),
#     jnp.array(N_decay),
#     p0=p0,
#     bounds=bounds
# )
#
# N0, lambda_fit, Nbg = popt
# errors = np.sqrt(np.diag(pcov))
#
# half_life = np.log(2) / lambda_fit
#
# print(f"N₀ = {N0:.1f} ± {errors[0]:.1f} counts")
# print(f"λ = {lambda_fit:.3f} ± {errors[1]:.3f} hr⁻¹")
# print(f"N_bg = {Nbg:.1f} ± {errors[2]:.1f} counts")
# print(f"Half-life = {half_life:.2f} hours")
#
# # Plot
# t_fit = np.linspace(0, 20, 100)
# plt.plot(t_decay, N_decay, 'o', ms=8, label='Data')
# plt.plot(t_fit, decay_with_background(t_fit, *popt), 'r-', lw=2, label='Fit')
# plt.axhline(Nbg, ls='--', color='green', label=f'Background: {Nbg:.0f}')
# plt.xlabel('Time (hours)')
# plt.ylabel('Counts')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
# ```
#
# **Expected**: λ ≈ 0.20 hr⁻¹, N₀ ≈ 1000, N_bg ≈ 130, half-life ≈ 3.5 hours
# </details>
#
# ### Self-Assessment
#
# - [ ] Three-parameter model defined
# - [ ] Bounds used to enforce physical constraints
# - [ ] Decay constant λ has correct units
# - [ ] Half-life calculated correctly
# - [ ] Fit visually matches data
# ======================================================================


# ======================================================================
# ## Challenge 3: Noisy Data with Outliers 🟡
#
# **Difficulty**: Intermediate
# **Time**: 15-20 minutes
#
# ### Problem
#
# Fit a Gaussian peak to spectroscopy data, but 2-3 data points are outliers (bad measurements). Use **robust fitting** to minimize their impact.
#
# **Learning Objectives**:
# - Handling outliers
# - Using weights or loss functions
# - Data cleaning strategies
#
# **Data**:
# ======================================================================


# Wavelength (nm)
x_spec = np.linspace(500, 520, 30)

# Intensity (Gaussian peak + outliers)
true_gaussian = 100 * np.exp(-((x_spec - 510) ** 2) / (2 * 3**2))
y_spec = true_gaussian + np.random.normal(0, 5, len(x_spec))

# Add 3 outliers
y_spec[[5, 12, 22]] += np.array([40, -30, 50])  # Bad measurements

plt.figure(figsize=(10, 5))
plt.plot(x_spec, y_spec, "o", ms=6)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("Spectroscopy Data with Outliers")
plt.grid(alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_challenges"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()

print("Your task: Fit Gaussian I(x) = A * exp(-(x - x0)^2 / (2 * σ^2))")
print("Challenge: Handle outliers! Try 2 approaches and compare.")


# ======================================================================
# <details>
# <summary><b>💡 Hint 1</b> (Click to expand)</summary>
#
# Approach 1: Manually remove outliers (points > 3σ from smooth curve).  # noqa: RUF003
# Approach 2: Use sigma weights to downweight outliers.
# </details>
#
# <details>
# <summary><b>💡 Hint 2</b> (Click to expand)</summary>
#
# For robust fitting with NLSQ, you can:
# 1. Filter data: `mask = np.abs(y - moving_avg) < 3*std`
# 2. Or increase `sigma` for suspicious points to reduce their influence
# </details>
# ======================================================================


# ═══════════════════════════════════════════════════════════
# YOUR SOLUTION HERE
# ═══════════════════════════════════════════════════════════


# ======================================================================
# <details>
# <summary><b>✅ Click to reveal SOLUTION</b></summary>
#
# ```python
# def gaussian(x, A, x0, sigma):
#     return A * jnp.exp(-((x - x0) ** 2) / (2 * sigma**2))
#
# cf = CurveFit()
#
# # Approach 1: Naive fit (outliers included)
# p0 = [100, 510, 3]
# popt_naive, _ = cf.curve_fit(gaussian, jnp.array(x_spec), jnp.array(y_spec), p0=p0)
#
# # Approach 2: Manual outlier removal
# residuals_initial = y_spec - gaussian(x_spec, *p0)
# threshold = 3 * np.std(residuals_initial)
# mask = np.abs(residuals_initial) < threshold
#
# x_clean = x_spec[mask]
# y_clean = y_spec[mask]
#
# popt_clean, _ = cf.curve_fit(gaussian, jnp.array(x_clean), jnp.array(y_clean), p0=p0)
#
# print("Naive fit (with outliers):")
# print(f"  A={popt_naive[0]:.1f}, x0={popt_naive[1]:.2f}, σ={popt_naive[2]:.2f}")  # noqa: RUF003
# print(f"\nRobust fit (outliers removed):")
# print(f"  A={popt_clean[0]:.1f}, x0={popt_clean[1]:.2f}, σ={popt_clean[2]:.2f}")  # noqa: RUF003
# print(f"  Removed {np.sum(~mask)} outliers")
#
# # Plot
# x_fit = np.linspace(500, 520, 200)
# plt.plot(x_spec, y_spec, 'o', ms=6, alpha=0.5, label='Data (with outliers)')
# plt.plot(x_spec[~mask], y_spec[~mask], 'rx', ms=10, mew=2, label='Outliers')
# plt.plot(x_fit, gaussian(x_fit, *popt_naive), 'r--', lw=2, label='Naive fit')
# plt.plot(x_fit, gaussian(x_fit, *popt_clean), 'g-', lw=2, label='Robust fit')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
# ```
#
# **Expected**: Robust fit should give A≈100, x₀≈510, σ≈3 (close to true values)  # noqa: RUF003
# </details>
#
# ### Self-Assessment
#
# - [ ] Identified outliers visually or statistically
# - [ ] Tried at least two approaches
# - [ ] Robust fit parameters closer to true values
# - [ ] Documented which points were removed
# - [ ] Visualized both fits for comparison
# ======================================================================


# ======================================================================
# ## Challenge 4: Convergence Debugging 🔴
#
# **Difficulty**: Advanced
# **Time**: 20-25 minutes
#
# ### Problem
#
# The code below **intentionally fails to converge**. Your task: debug and fix it!
#
# **Learning Objectives**:
# - Troubleshooting convergence failures
# - Improving initial guesses
# - Using bounds and constraints
#
# **Broken Code**:
# ======================================================================


# Data: Logistic growth
t_growth = np.linspace(0, 50, 40)
y_growth = 1000 / (1 + 9 * np.exp(-0.15 * t_growth)) + np.random.normal(0, 10, 40)


# BROKEN MODEL AND FIT (DO NOT CHANGE THIS CELL - FIX IN NEXT CELL)
def logistic(t, L, k, t0):
    return L / (1 + jnp.exp(-k * (t - t0)))


# This will likely fail or give poor results:
p0_bad = [10, 0.001, 100]  # Terrible initial guess!

try:
    cf_broken = CurveFit()
    popt_broken, _ = cf_broken.curve_fit(
        logistic, jnp.array(t_growth), jnp.array(y_growth), p0=p0_bad, maxiter=50
    )
    print(
        f"Fitted (likely wrong): L={popt_broken[0]:.1f}, k={popt_broken[1]:.3f}, t0={popt_broken[2]:.1f}"
    )
except Exception as e:
    print(f"Error: {e}")

plt.figure(figsize=(8, 5))
plt.plot(t_growth, y_growth, "o", label="Data")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Logistic Growth (Broken Fit)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "nlsq_challenges"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_04.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nYour task: Fix the fit! Identify problems and provide working solution below.")


# ======================================================================
# <details>
# <summary><b>💡 Hint 1</b> (Click to expand)</summary>
#
# Problems to check:
# 1. Is p0 reasonable? (L should be near max(y), t0 near inflection point)
# 2. Is maxiter too low?
# 3. Would bounds help constrain the search?
# </details>
#
# <details>
# <summary><b>💡 Hint 2</b> (Click to expand)</summary>
#
# Estimate from data:
# - L ≈ max(y_growth) ≈ 1000
# - t0 ≈ time where y ≈ L/2 ≈ 15-20
# - k ≈ 0.1-0.2 (growth rate)
# </details>
# ======================================================================


# ═══════════════════════════════════════════════════════════
# YOUR FIXED SOLUTION HERE
# ═══════════════════════════════════════════════════════════


# ======================================================================
# <details>
# <summary><b>✅ Click to reveal SOLUTION</b></summary>
#
# ```python
# # Fixed version
#
# # Problem 1: Bad initial guess
# # Fix: Estimate from data
# L_guess = np.max(y_growth)  # Carrying capacity ≈ max value
# half_L = L_guess / 2
# t0_guess = t_growth[np.argmin(np.abs(y_growth - half_L))]  # Inflection point
# k_guess = 0.15  # Reasonable growth rate
#
# p0_fixed = [L_guess, k_guess, t0_guess]
# print(f"Improved p0: L={p0_fixed[0]:.0f}, k={p0_fixed[1]:.2f}, t0={p0_fixed[2]:.1f}")
#
# # Problem 2: maxiter too low
# # Fix: Increase to 1000
#
# # Problem 3: No bounds
# # Fix: Add reasonable bounds
# bounds_fixed = ([0, 0, 0], [2000, 1, 100])
#
# cf_fixed = CurveFit()
# popt_fixed, pcov_fixed = cf_fixed.curve_fit(
#     logistic,
#     jnp.array(t_growth),
#     jnp.array(y_growth),
#     p0=p0_fixed,
#     bounds=bounds_fixed,
#     maxiter=1000
# )
#
# print(f"\nFixed fit: L={popt_fixed[0]:.1f}, k={popt_fixed[1]:.3f}, t0={popt_fixed[2]:.1f}")
# print("(True values: L=1000, k=0.15, t0≈15.4)")
#
# # Plot
# t_fit = np.linspace(0, 50, 200)
# plt.plot(t_growth, y_growth, 'o', label='Data')
# plt.plot(t_fit, logistic(t_fit, *popt_fixed), 'r-', lw=2, label='Fixed fit')
# plt.xlabel('Time')
# plt.ylabel('Population')
# plt.title('Logistic Growth (Fixed!)')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.show()
# ```
#
# **Key Fixes**:
# 1. Better p0 estimation from data
# 2. Increased maxiter
# 3. Added bounds for physical constraints
# </details>
#
# ### Self-Assessment
#
# - [ ] Identified all 3 problems
# - [ ] Improved initial guess systematically
# - [ ] Used appropriate bounds
# - [ ] Fit now converges successfully
# - [ ] Parameters match true values within ~10%
# ======================================================================


# ======================================================================
# ## Bonus Challenges (Optional)
#
# If you've completed all 4 challenges, try these advanced exercises:
#
# ### 🔴 Bonus 1: Multi-Peak Fitting
# Fit 3 overlapping Gaussian peaks to simulated Raman spectrum data. Hint: Initialize each peak separately, then fit all parameters together.
#
# ### 🔴 Bonus 2: Time Series Forecasting
# Fit trend + seasonal model to temperature data, then forecast 30 days ahead with uncertainty bands.
#
# ### 🔴 Bonus 3: Custom Loss Function
# Implement asymmetric loss (penalize overestimation more than underestimation) for safety-critical application.
#
# ---
#
# ## Summary
#
# Congratulations on completing the NLSQ challenges! You've practiced:
#
# - Basic curve fitting (Challenge 1)
# - Multi-parameter models with bounds (Challenge 2)
# - Robust fitting with outliers (Challenge 3)
# - Debugging convergence failures (Challenge 4)
#
# ### Next Steps
#
# - Apply these skills to your own data
# - Explore `research_workflow_case_study.ipynb` for publication-quality analysis
# - Check `troubleshooting_guide.ipynb` when you encounter issues
# - Browse `gallery/` examples for domain-specific applications
#
# ### Feedback
#
# Found these challenges helpful? Have suggestions for more? Open an issue on GitHub!
# ======================================================================
