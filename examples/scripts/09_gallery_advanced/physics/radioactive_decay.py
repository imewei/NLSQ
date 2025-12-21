"""
Advanced Radioactive Decay Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting radioactive decay data to determine
the half-life of an isotope using NLSQ's advanced fit() API and global
optimization for robust parameter estimation.

Compared to 04_gallery/physics/radioactive_decay.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Exponential decay fitting
- Half-life calculation from decay constant
- Uncertainty propagation
- Parameter correlation analysis
- Global optimization for robust parameter estimation
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import GlobalOptimizationConfig, fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK else {}

# Set random seed
np.random.seed(42)


def radioactive_decay(t, N0, lambda_decay):
    """
    Radioactive decay model: N(t) = N0 * exp(-lambda * t)

    Parameters
    ----------
    t : array_like
        Time (years)
    N0 : float
        Initial number of atoms
    lambda_decay : float
        Decay constant (1/years)

    Returns
    -------
    N : array_like
        Number of atoms at time t
    """
    return N0 * jnp.exp(-lambda_decay * t)


def half_life_from_lambda(lambda_decay):
    """Calculate half-life from decay constant: t_half = ln(2) / lambda"""
    return np.log(2) / lambda_decay


def propagate_uncertainty(lambda_val, lambda_err):
    """Propagate uncertainty from decay constant to half-life."""
    t_half = half_life_from_lambda(lambda_val)
    t_half_err = t_half * (lambda_err / lambda_val)
    return t_half, t_half_err


# True parameters (Carbon-14)
N0_true = 1000.0  # Initial count rate (counts per minute)
half_life_true = 5730.0  # years (C-14 half-life)
lambda_true = np.log(2) / half_life_true  # decay constant

# Time points
time = np.linspace(0, 20000, 20 if QUICK else 30)

# True decay curve
N_true = N0_true * np.exp(-lambda_true * time)

# Add realistic measurement noise
noise_level = 0.05 * N_true + np.random.normal(0, 5, size=len(time))
N_measured = N_true + noise_level

# Measurement uncertainties
sigma = 0.05 * N_true + 5


print("=" * 70)
print("RADIOACTIVE DECAY: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# Decay Constant Fitting
# =============================================================================
print("\n" + "-" * 70)
print("DECAY CONSTANT FITTING")
print("-" * 70)

# Initial parameter guess
p0 = [1200, 0.0001]

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt, pcov = fit(
    radioactive_decay,
    time,
    N_measured,
    p0=p0,
    sigma=sigma,
    absolute_sigma=True,
    preset="robust",
    **FIT_KWARGS,
)

N0_fit, lambda_fit = popt
perr = np.sqrt(np.diag(pcov))
N0_err, lambda_err = perr

t_half_fit, t_half_err = propagate_uncertainty(lambda_fit, lambda_err)

print(f"  N0 = {N0_fit:.2f} +/- {N0_err:.2f} counts/min")
print(f"  lambda = {lambda_fit:.6e} +/- {lambda_err:.6e} yr^-1")
print(f"  t_1/2 = {t_half_fit:.0f} +/- {t_half_err:.0f} years")

if QUICK:
    print("\n⏩ Quick mode: skipping global/custom fits and extended analysis.")
    sys.exit(0)

# Method 2: fit() with 'global' preset
print("\nMethod 2: fit() with 'global' preset")
popt_global, pcov_global = fit(
    radioactive_decay,
    time,
    N_measured,
    p0=p0,
    sigma=sigma,
    absolute_sigma=True,
    preset="global",
)

N0_g, lambda_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))
t_half_g, t_half_err_g = propagate_uncertainty(lambda_g, perr_g[1])

print(f"  N0 = {N0_g:.2f} +/- {perr_g[0]:.2f}")
print(f"  lambda = {lambda_g:.6e} +/- {perr_g[1]:.6e}")
print(f"  t_1/2 = {t_half_g:.0f} +/- {t_half_err_g:.0f} years")

# Method 3: GlobalOptimizationConfig with custom settings
print("\nMethod 3: GlobalOptimizationConfig with custom settings")
popt_custom, pcov_custom = fit(
    radioactive_decay,
    time,
    N_measured,
    p0=p0,
    sigma=sigma,
    absolute_sigma=True,
    multistart=True,
    n_starts=15,
    sampler="lhs",
)

N0_c, lambda_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))
t_half_c, t_half_err_c = propagate_uncertainty(lambda_c, perr_c[1])

print(f"  N0 = {N0_c:.2f} +/- {perr_c[0]:.2f}")
print(f"  lambda = {lambda_c:.6e} +/- {perr_c[1]:.6e}")
print(f"  t_1/2 = {t_half_c:.0f} +/- {t_half_err_c:.0f} years")


# Use robust preset results for analysis
N0_fit, lambda_fit = popt
perr = np.sqrt(np.diag(pcov))
N0_err, lambda_err = perr
t_half_fit, t_half_err = propagate_uncertainty(lambda_fit, lambda_err)


print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  N0 = {N0_fit:.2f} +/- {N0_err:.2f} counts/min")
print(f"  lambda = {lambda_fit:.6e} +/- {lambda_err:.6e} yr^-1")

print("\nDerived Half-Life:")
print(f"  t_1/2 = {t_half_fit:.0f} +/- {t_half_err:.0f} years")
print(f"  True value: {half_life_true} years")
print(
    f"  Error: {abs(t_half_fit - half_life_true):.0f} years "
    + f"({100 * abs(t_half_fit - half_life_true) / half_life_true:.1f}%)"
)

within_1sigma = abs(t_half_fit - half_life_true) < t_half_err
print(f"\n  Within 1sigma uncertainty: {within_1sigma}")

# Correlation coefficient
corr = pcov[0, 1] / (perr[0] * perr[1])
print("\nParameter Correlation:")
print(f"  rho(N0, lambda) = {corr:.4f}")

# Goodness of fit
chi_squared = np.sum(((N_measured - radioactive_decay(time, *popt)) / sigma) ** 2)
dof = len(time) - len(popt)
chi_squared_reduced = chi_squared / dof

print("\nGoodness of Fit:")
print(f"  chi^2 = {chi_squared:.2f}")
print(f"  chi^2/dof = {chi_squared_reduced:.2f} (expect ~1.0 for good fit)")


# =============================================================================
# Age Dating Application
# =============================================================================
print("\n" + "-" * 70)
print("AGE DATING APPLICATION")
print("-" * 70)

# Example: Sample with measured N/N0 ratio
ratio_measured = 0.25  # 25% of original C-14 remaining
ratio_uncertainty = 0.02  # +/- 2%

# Calculate age: t = -ln(N/N0) / lambda
age_estimated = -np.log(ratio_measured) / lambda_fit
age_uncertainty = age_estimated * np.sqrt(
    (ratio_uncertainty / ratio_measured) ** 2 + (lambda_err / lambda_fit) ** 2
)

print(f"Sample with N/N0 = {ratio_measured:.2f} +/- {ratio_uncertainty:.2f}:")
print(f"  Estimated age: {age_estimated:.0f} +/- {age_uncertainty:.0f} years")
print(f"  (Using fitted half-life of {t_half_fit:.0f} years)")

# Compare with true half-life
age_true_lambda = -np.log(ratio_measured) / lambda_true
print(f"  (True age would be: {age_true_lambda:.0f} years)")


# =============================================================================
# Visualization
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and fit
ax1 = axes[0, 0]
ax1.errorbar(
    time, N_measured, yerr=sigma, fmt="o", alpha=0.5, label="Measured data", capsize=3
)
ax1.plot(time, N_true, "r--", linewidth=2, label="True decay")
t_fine = np.linspace(0, 20000, 200)
ax1.plot(
    t_fine,
    radioactive_decay(t_fine, *popt),
    "g-",
    linewidth=2,
    label="Fitted decay (robust)",
)

ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Activity (counts/min)")
ax1.set_title("Radioactive Decay - fit() API", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Semi-log plot
ax2 = axes[0, 1]
ax2.semilogy(time, N_measured, "o", alpha=0.5, label="Measured data")
ax2.semilogy(time, N_true, "r--", linewidth=2, label="True decay")
ax2.semilogy(
    t_fine, radioactive_decay(t_fine, *popt), "g-", linewidth=2, label="Fitted decay"
)

ax2.set_xlabel("Time (years)")
ax2.set_ylabel("Activity (counts/min, log scale)")
ax2.set_title("Semi-Log Plot (Linear for Exponential)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
residuals = N_measured - radioactive_decay(time, *popt)
normalized_residuals = residuals / sigma
ax3.errorbar(time, normalized_residuals, yerr=1.0, fmt="o", alpha=0.5, capsize=3)
ax3.axhline(0, color="r", linestyle="--", linewidth=2)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)

ax3.set_xlabel("Time (years)")
ax3.set_ylabel("Normalized Residuals (sigma)")
ax3.set_title("Fit Residuals (+/- 2sigma bounds)")
ax3.grid(True, alpha=0.3)

# Plot 4: Half-life visualization
ax4 = axes[1, 1]
t_plot = np.linspace(0, 15000, 200)
N_plot = radioactive_decay(t_plot, N0_fit, lambda_fit)
ax4.plot(t_plot, N_plot, "g-", linewidth=2, label="Fitted decay")
ax4.axhline(
    N0_fit / 2,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"t_1/2 = {t_half_fit:.0f} yr",
)
ax4.axvline(t_half_fit, color="orange", linestyle="--", linewidth=2)
ax4.fill_between(
    [t_half_fit - t_half_err, t_half_fit + t_half_err],
    0,
    N0_fit,
    alpha=0.2,
    color="orange",
    label="+/- 1sigma uncertainty",
)

ax4.set_xlabel("Time (years)")
ax4.set_ylabel("Activity (counts/min)")
ax4.set_title("Half-Life Determination")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "radioactive_decay"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Radioactive decay parameters determined using fit() API:")
print(f"\n  Fitted half-life: {t_half_fit:.0f} +/- {t_half_err:.0f} years")
print(f"  Literature value: {half_life_true} years (C-14)")
print(
    f"  Agreement: {100 * (1 - abs(t_half_fit - half_life_true) / half_life_true):.1f}%"
)
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print("  - fit() with preset='global' (20 multi-starts)")
print("  - fit() with GlobalOptimizationConfig (custom settings)")
print("\nThis example demonstrates:")
print("  - Exponential decay fitting with fit() API")
print("  - Global optimization for robust parameter estimation")
print("  - Uncertainty propagation from fit parameters to derived quantities")
print("  - Goodness-of-fit analysis with chi^2 statistic")
print("  - Age dating application")
print("=" * 70)
