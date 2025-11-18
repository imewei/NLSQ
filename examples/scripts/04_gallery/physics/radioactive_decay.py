"""
Converted from radioactive_decay.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Radioactive Decay: Half-Life Calculation
#
#
# This example demonstrates fitting radioactive decay data to determine
# the half-life of an isotope. We use real-world inspired data for Carbon-14
# (actual half-life: 5,730 years) and show how to propagate uncertainties
# to the calculated half-life.
#
# Key Concepts:
# - Exponential decay fitting
# - Half-life calculation from decay constant
# - Uncertainty propagation
# - Parameter correlation analysis
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed for reproducibility
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
    """
    Calculate half-life from decay constant.

    t_half = ln(2) / lambda

    Parameters
    ----------
    lambda_decay : float
        Decay constant (1/years)

    Returns
    -------
    t_half : float
        Half-life (years)
    """
    return np.log(2) / lambda_decay


def propagate_uncertainty(lambda_val, lambda_err):
    """
    Propagate uncertainty from decay constant to half-life.

    Using error propagation: σ(t_half) = t_half * (σ(lambda) / lambda)

    Parameters
    ----------
    lambda_val : float
        Decay constant value
    lambda_err : float
        Uncertainty in decay constant

    Returns
    -------
    t_half : float
        Half-life
    t_half_err : float
        Uncertainty in half-life
    """
    t_half = half_life_from_lambda(lambda_val)
    # Derivative: d(ln(2)/λ)/dλ = -ln(2)/λ²
    t_half_err = t_half * (lambda_err / lambda_val)
    return t_half, t_half_err


# True parameters (Carbon-14)
N0_true = 1000.0  # Initial count rate (counts per minute)
half_life_true = 5730.0  # years (C-14 half-life)
lambda_true = np.log(2) / half_life_true  # decay constant

# Time points (0 to 20,000 years, 30 measurements)
time = np.linspace(0, 20000, 30)

# True decay curve
N_true = N0_true * np.exp(-lambda_true * time)

# Add realistic measurement noise (Poisson-like, ~5% relative uncertainty)
# Simulate measurement uncertainty that increases as counts decrease
noise_level = 0.05 * N_true + np.random.normal(0, 5, size=len(time))
N_measured = N_true + noise_level

# Measurement uncertainties (standard deviations)
sigma = 0.05 * N_true + 5  # Poisson + background


print("=" * 70)
print("RADIOACTIVE DECAY: CARBON-14 HALF-LIFE DETERMINATION")
print("=" * 70)

# Initial parameter guess
p0 = [1200, 0.0001]  # Rough estimate

# Fit with weighted least squares (using measurement uncertainties)
popt, pcov = curve_fit(
    radioactive_decay, time, N_measured, p0=p0, sigma=sigma, absolute_sigma=True
)

# Extract fitted parameters
N0_fit, lambda_fit = popt
perr = np.sqrt(np.diag(pcov))  # Parameter uncertainties
N0_err, lambda_err = perr

# Calculate half-life and propagate uncertainty
t_half_fit, t_half_err = propagate_uncertainty(lambda_fit, lambda_err)


print("\nFitted Parameters:")
print(f"  N0 = {N0_fit:.2f} ± {N0_err:.2f} counts/min")
print(f"  λ  = {lambda_fit:.6e} ± {lambda_err:.6e} yr⁻¹")

print("\nDerived Half-Life:")
print(f"  t₁/₂ = {t_half_fit:.0f} ± {t_half_err:.0f} years")
print(f"  True value: {half_life_true} years")
print(
    f"  Error: {abs(t_half_fit - half_life_true):.0f} years "
    + f"({100 * abs(t_half_fit - half_life_true) / half_life_true:.1f}%)"
)

# Check if within 1-sigma
within_1sigma = abs(t_half_fit - half_life_true) < t_half_err
print(f"\n  ✅ Within 1σ uncertainty: {within_1sigma}")

# Correlation coefficient
corr = pcov[0, 1] / (perr[0] * perr[1])
print("\nParameter Correlation:")
print(f"  ρ(N0, λ) = {corr:.4f}")

# Goodness of fit
chi_squared = np.sum(((N_measured - radioactive_decay(time, *popt)) / sigma) ** 2)
dof = len(time) - len(popt)  # degrees of freedom
chi_squared_reduced = chi_squared / dof
print("\nGoodness of Fit:")
print(f"  χ² = {chi_squared:.2f}")
print(f"  χ²/dof = {chi_squared_reduced:.2f} (expect ≈ 1.0 for good fit)")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top left: Data and fit
ax1 = axes[0, 0]
ax1.errorbar(
    time, N_measured, yerr=sigma, fmt="o", alpha=0.5, label="Measured data", capsize=3
)
ax1.plot(time, N_true, "r--", linewidth=2, label="True decay")
t_fine = np.linspace(0, 20000, 200)
ax1.plot(
    t_fine, radioactive_decay(t_fine, *popt), "g-", linewidth=2, label="Fitted decay"
)
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Activity (counts/min)")
ax1.set_title("Radioactive Decay of Carbon-14")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Semi-log plot (shows exponential as straight line)
ax2 = axes[0, 1]
ax2.semilogy(time, N_measured, "o", alpha=0.5, label="Measured data")
ax2.semilogy(time, N_true, "r--", linewidth=2, label="True decay")
ax2.semilogy(
    t_fine, radioactive_decay(t_fine, *popt), "g-", linewidth=2, label="Fitted decay"
)
ax2.set_xlabel("Time (years)")
ax2.set_ylabel("Activity (counts/min, log scale)")
ax2.set_title("Semi-Log Plot (Linear for Exponential Decay)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom left: Residuals
ax3 = axes[1, 0]
residuals = N_measured - radioactive_decay(time, *popt)
normalized_residuals = residuals / sigma
ax3.errorbar(time, normalized_residuals, yerr=1.0, fmt="o", alpha=0.5, capsize=3)
ax3.axhline(0, color="r", linestyle="--", linewidth=2)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Time (years)")
ax3.set_ylabel("Normalized Residuals (σ)")
ax3.set_title("Fit Residuals (±2σ bounds shown)")
ax3.grid(True, alpha=0.3)

# Bottom right: Half-life visualization
ax4 = axes[1, 1]
# Show decay to half activity
t_plot = np.linspace(0, 15000, 200)
N_plot = radioactive_decay(t_plot, N0_fit, lambda_fit)
ax4.plot(t_plot, N_plot, "g-", linewidth=2, label="Fitted decay")
ax4.axhline(
    N0_fit / 2,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"t₁/₂ = {t_half_fit:.0f} yr",
)
ax4.axvline(t_half_fit, color="orange", linestyle="--", linewidth=2)
ax4.fill_between(
    [t_half_fit - t_half_err, t_half_fit + t_half_err],
    0,
    N0_fit,
    alpha=0.2,
    color="orange",
    label="±1σ uncertainty",
)
ax4.set_xlabel("Time (years)")
ax4.set_ylabel("Activity (counts/min)")
ax4.set_title("Half-Life Determination")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("radioactive_decay.png", dpi=150)
print("\n✅ Plot saved as 'radioactive_decay.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "radioactive_decay"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Fitted half-life: {t_half_fit:.0f} ± {t_half_err:.0f} years")
print(f"Literature value: {half_life_true} years (C-14)")
print(
    f"Agreement: {100 * (1 - abs(t_half_fit - half_life_true) / half_life_true):.1f}%"
)
print("\nThis example demonstrates:")
print("  ✓ Fitting exponential decay with weighted least squares")
print("  ✓ Uncertainty propagation from fit parameters to derived quantities")
print("  ✓ Goodness-of-fit analysis with χ² statistic")
print("  ✓ Visualization of residuals and parameter uncertainties")
print("=" * 70)
