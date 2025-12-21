"""
Converted from titration_curves.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

import os
import sys
from pathlib import Path

# ======================================================================
# # Acid-Base Titration Curve Analysis
#
#
# This example demonstrates acid-base titration curve analysis using NLSQ.
# We analyze titration data to determine pKa values, equivalence points,
# and buffer capacity using the Henderson-Hasselbalch equation and related models.
#
# Key Applications:
# - pKa determination from titration curves
# - Equivalence point identification
# - Buffer capacity analysis
# - Polyprotic acid characterization
# - Weak acid/base property quantification
#
# Physical Context:
# The Henderson-Hasselbalch equation relates pH to the acid dissociation constant (pKa)
# and the ratio of conjugate base to acid concentrations:
#     pH = pKa + log10([A⁻]/[HA])
#
# For a titration curve, the relationship between pH and volume of titrant can be
# modeled using the equilibrium expressions for acid-base reactions.
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from nlsq import curve_fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK else {}


def monoprotic_titration(V, pKa, Ve, C_acid, C_base):
    """
    Monoprotic weak acid titration model.

    Parameters
    ----------
    V : array_like
        Volume of base added (mL)
    pKa : float
        Acid dissociation constant (negative log scale)
    Ve : float
        Equivalence point volume (mL)
    C_acid : float
        Initial concentration of acid (M)
    C_base : float
        Concentration of titrant base (M)

    Returns
    -------
    pH : array_like
        pH at each titration point
    """
    # Avoid division by zero
    V = jnp.maximum(V, 1e-6)

    # Initial volume of acid
    _V0 = 25.0  # mL (fixed for this example)

    # Fraction titrated
    f = V / Ve
    f = jnp.clip(f, 1e-6, 1 - 1e-6)  # Avoid log(0)

    # Henderson-Hasselbalch approximation
    # Before equivalence point: pH = pKa + log10(f/(1-f))
    # After equivalence point: excess strong base dominates

    # Calculate pH using Henderson-Hasselbalch
    ratio = f / (1 - f)
    pH = pKa + jnp.log10(jnp.maximum(ratio, 1e-10))

    return pH


def simplified_titration(V, pKa, Ve, pH0):
    """
    Simplified titration model for curve fitting.

    Parameters
    ----------
    V : array_like
        Volume of base added (mL)
    pKa : float
        Acid dissociation constant
    Ve : float
        Equivalence point volume (mL)
    pH0 : float
        Initial pH (at V=0)

    Returns
    -------
    pH : array_like
        pH at each titration point
    """
    # Avoid division by zero
    V_safe = jnp.maximum(V, 0.01)
    Ve_safe = jnp.maximum(Ve, V_safe + 0.1)

    # Fraction titrated
    f = V_safe / Ve_safe
    f = jnp.clip(f, 0.01, 0.99)

    # Henderson-Hasselbalch
    ratio = f / (1 - f)
    pH = pKa + jnp.log10(jnp.maximum(ratio, 1e-10))

    return pH


def diprotic_titration(V, pKa1, pKa2, Ve1, Ve2):
    """
    Diprotic acid titration model (e.g., H2CO3, H2SO3).

    Parameters
    ----------
    V : array_like
        Volume of base added (mL)
    pKa1 : float
        First acid dissociation constant
    pKa2 : float
        Second acid dissociation constant
    Ve1 : float
        First equivalence point volume (mL)
    Ve2 : float
        Second equivalence point volume (mL)

    Returns
    -------
    pH : array_like
        pH at each titration point
    """
    # Avoid division by zero
    V_safe = jnp.maximum(V, 0.01)

    # Determine which region we're in
    # Region 1: 0 < V < Ve1 (first proton)
    # Region 2: Ve1 < V < Ve2 (second proton)

    # Simplified approach: use weighted combination
    f1 = V_safe / jnp.maximum(Ve1, V_safe + 0.1)
    f1 = jnp.clip(f1, 0.01, 0.99)

    f2 = jnp.maximum(0, (V_safe - Ve1)) / jnp.maximum(Ve2 - Ve1, 0.1)
    f2 = jnp.clip(f2, 0.01, 0.99)

    # First equilibrium
    pH1 = pKa1 + jnp.log10(jnp.maximum(f1 / (1 - f1), 1e-10))

    # Second equilibrium
    pH2 = pKa2 + jnp.log10(jnp.maximum(f2 / (1 - f2), 1e-10))

    # Weighted combination based on which region dominates
    weight = jnp.where(V_safe < Ve1, 1.0, 0.0)
    pH = weight * pH1 + (1 - weight) * pH2

    return pH


def buffer_capacity(pH, pKa, C_total):
    """
    Calculate buffer capacity as a function of pH.

    β = 2.303 * C_total * ([H⁺]K_a) / ([H⁺] + K_a)²

    Parameters
    ----------
    pH : array_like
        pH values
    pKa : float
        Acid dissociation constant
    C_total : float
        Total buffer concentration (M)

    Returns
    -------
    beta : array_like
        Buffer capacity (mol/L per pH unit)
    """
    H_plus = jnp.power(10.0, -pH)
    Ka = jnp.power(10.0, -pKa)

    numerator = H_plus * Ka
    denominator = jnp.power(H_plus + Ka, 2)

    beta = 2.303 * C_total * numerator / jnp.maximum(denominator, 1e-20)

    return beta


# ======================================================================
# ## Example 1: Monoprotic Weak Acid Titration (Acetic Acid)
#
# ======================================================================


print("=" * 70)
print("Example 1: Monoprotic Weak Acid Titration (Acetic Acid with NaOH)")
print("=" * 70)

# Experimental parameters
V0_acid = 25.0  # mL of 0.1 M acetic acid
C_acid_true = 0.1  # M
C_base_true = 0.1  # M (NaOH)
pKa_true = 4.76  # Acetic acid
Ve_true = 25.0  # mL (equivalence point)

# Generate synthetic titration data
np.random.seed(42)
V_titrant = np.linspace(0.1, 40, 40 if QUICK else 100)  # Volume of NaOH added (mL)

# Calculate true pH values using more detailed model
pH_true = np.zeros_like(V_titrant)
for i, V in enumerate(V_titrant):
    if Ve_true > V:
        # Before equivalence: buffer region
        f = V / Ve_true
        if f > 0.001 and f < 0.999:
            pH_true[i] = pKa_true + np.log10(f / (1 - f))
        elif f <= 0.001:
            # Very beginning: mostly acid
            pH_true[i] = 2.9
        else:
            # Near equivalence
            pH_true[i] = 8.0
    elif Ve_true == V:
        # At equivalence: pH determined by conjugate base
        pH_true[i] = 8.72
    else:
        # After equivalence: excess strong base
        excess = (V - Ve_true) * C_base_true
        total_volume = V0_acid + V
        pOH = -np.log10(excess / total_volume)
        pH_true[i] = 14 - pOH

# Add realistic noise
noise_level = 0.05
pH_measured = pH_true + np.random.normal(0, noise_level, size=pH_true.shape)
sigma_pH = np.full_like(pH_measured, noise_level)

# Fit simplified titration model to extract pKa and Ve
# Initial guess
p0 = [4.5, 24.0, 3.0]  # pKa, Ve, pH0

# Bounds
bounds_lower = [3.0, 20.0, 2.5]
bounds_upper = [6.0, 30.0, 4.0]

# Fit only the buffer region (5-35 mL) for better pKa estimation
mask_fit = (V_titrant >= 5) & (V_titrant <= 35)
V_fit = V_titrant[mask_fit]
pH_fit = pH_measured[mask_fit]
sigma_fit = sigma_pH[mask_fit]

popt, pcov = curve_fit(
    simplified_titration,
    V_fit,
    pH_fit,
    p0=p0,
    sigma=sigma_fit,
    bounds=(bounds_lower, bounds_upper),
    absolute_sigma=True,
    **FIT_KWARGS,
)

pKa_fitted, Ve_fitted, pH0_fitted = popt
pKa_err, Ve_err, pH0_err = np.sqrt(np.diag(pcov))

print("\nFitted Parameters:")
print(f"  pKa = {pKa_fitted:.3f} ± {pKa_err:.3f} (true: {pKa_true:.2f})")
print(
    f"  Equivalence point = {Ve_fitted:.2f} ± {Ve_err:.2f} mL (true: {Ve_true:.1f} mL)"
)
print(f"  Initial pH = {pH0_fitted:.2f} ± {pH0_err:.2f}")

# Calculate fitted curve
pH_fitted_curve = simplified_titration(V_fit, *popt)

# Statistical validation
residuals = pH_fit - pH_fitted_curve
chi_squared = np.sum((residuals / sigma_fit) ** 2)
dof = len(pH_fit) - len(popt)
reduced_chi_squared = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  χ² = {chi_squared:.2f}")
print(f"  χ²/dof = {reduced_chi_squared:.3f} (should be ≈ 1)")
print(f"  RMSE = {rmse:.4f} pH units")

# Calculate first derivative (titration curve slope) to find inflection point
dV = V_titrant[1] - V_titrant[0]
dpH_dV = np.gradient(pH_measured, dV)
inflection_idx = np.argmax(dpH_dV)
Ve_inflection = V_titrant[inflection_idx]
pH_inflection = pH_measured[inflection_idx]

print("\nInflection Point Analysis:")
print(f"  Volume at inflection = {Ve_inflection:.2f} mL")
print(f"  pH at inflection = {pH_inflection:.2f}")
print(f"  Max slope = {dpH_dV[inflection_idx]:.2f} pH/mL")


# ======================================================================
# ## Example 2: Buffer Capacity Analysis
#
# ======================================================================


print("\n" + "=" * 70)
print("Example 2: Buffer Capacity Analysis")
print("=" * 70)

# Calculate buffer capacity
pH_range = np.linspace(3, 7, 80 if QUICK else 200)
beta_true = buffer_capacity(pH_range, pKa_true, C_acid_true)
beta_fitted = buffer_capacity(pH_range, pKa_fitted, C_acid_true)

# Find maximum buffer capacity (should occur at pH = pKa)
max_beta_idx = np.argmax(beta_fitted)
pH_max_beta = pH_range[max_beta_idx]
max_beta = beta_fitted[max_beta_idx]

print("\nBuffer Capacity Analysis:")
print(
    f"  Maximum capacity at pH = {pH_max_beta:.2f} (should equal pKa = {pKa_fitted:.2f})"
)
print(f"  Maximum β = {max_beta:.4f} mol/(L·pH)")
print(
    f"  Effective buffer range: {pKa_fitted - 1:.2f} - {pKa_fitted + 1:.2f} (pKa ± 1)"
)


# ======================================================================
# ## Example 3: Diprotic Acid Titration (Carbonic Acid)
#
# ======================================================================


if QUICK:
    print("⏩ Quick mode: skipping diprotic analysis and extended plots.")
    sys.exit(0)

print("\n" + "=" * 70)
print("Example 3: Diprotic Acid Titration (Carbonic Acid)")
print("=" * 70)

# True parameters for H2CO3
pKa1_true_di = 6.35  # First dissociation
pKa2_true_di = 10.33  # Second dissociation
Ve1_true = 25.0  # First equivalence point (mL)
Ve2_true = 50.0  # Second equivalence point (mL)

# Generate synthetic diprotic titration data
V_di = np.linspace(0.1, 70, 150)
pH_di_true = np.zeros_like(V_di)

for i, V in enumerate(V_di):
    if Ve1_true > V:
        # First buffer region
        f = V / Ve1_true
        if 0.01 < f < 0.99:
            pH_di_true[i] = pKa1_true_di + np.log10(f / (1 - f))
        else:
            pH_di_true[i] = 4.0
    elif Ve2_true > V:
        # Second buffer region
        f = (V - Ve1_true) / (Ve2_true - Ve1_true)
        if 0.01 < f < 0.99:
            pH_di_true[i] = pKa2_true_di + np.log10(f / (1 - f))
        else:
            pH_di_true[i] = 8.3
    else:
        # After second equivalence
        pH_di_true[i] = 12.0

# Add noise
pH_di_measured = pH_di_true + np.random.normal(0, 0.08, size=pH_di_true.shape)
sigma_pH_di = np.full_like(pH_di_measured, 0.08)

# Fit diprotic model
p0_di = [6.0, 10.0, 24.0, 48.0]  # pKa1, pKa2, Ve1, Ve2
bounds_lower_di = [5.0, 9.0, 20.0, 45.0]
bounds_upper_di = [7.0, 11.0, 30.0, 55.0]

# Fit to data
popt_di, pcov_di = curve_fit(
    diprotic_titration,
    V_di,
    pH_di_measured,
    p0=p0_di,
    sigma=sigma_pH_di,
    bounds=(bounds_lower_di, bounds_upper_di),
    absolute_sigma=True,
    **FIT_KWARGS,
)

pKa1_fitted_di, pKa2_fitted_di, Ve1_fitted, Ve2_fitted = popt_di
pKa1_err_di, pKa2_err_di, Ve1_err, Ve2_err = np.sqrt(np.diag(pcov_di))

print("\nFitted Parameters:")
print(f"  pKa1 = {pKa1_fitted_di:.2f} ± {pKa1_err_di:.2f} (true: {pKa1_true_di:.2f})")
print(f"  pKa2 = {pKa2_fitted_di:.2f} ± {pKa2_err_di:.2f} (true: {pKa2_true_di:.2f})")
print(f"  Ve1 = {Ve1_fitted:.2f} ± {Ve1_err:.2f} mL (true: {Ve1_true:.1f} mL)")
print(f"  Ve2 = {Ve2_fitted:.2f} ± {Ve2_err:.2f} mL (true: {Ve2_true:.1f} mL)")

# Calculate fitted curve
pH_di_fitted = diprotic_titration(V_di, *popt_di)

# Validation
residuals_di = pH_di_measured - pH_di_fitted
rmse_di = np.sqrt(np.mean(residuals_di**2))
chi_squared_di = np.sum((residuals_di / sigma_pH_di) ** 2)
dof_di = len(pH_di_measured) - len(popt_di)

print("\nGoodness of Fit:")
print(f"  RMSE = {rmse_di:.4f} pH units")
print(f"  χ²/dof = {chi_squared_di / dof_di:.3f}")

# Find both inflection points
dpH_dV_di = np.gradient(pH_di_measured, V_di[1] - V_di[0])

# First inflection (around Ve1)
mask_first = V_di < 35
idx_first = np.argmax(dpH_dV_di[mask_first])
Ve1_inflection = V_di[mask_first][idx_first]

# Second inflection (around Ve2)
mask_second = V_di > 35
dpH_dV_second = dpH_dV_di[mask_second]
idx_second = np.argmax(dpH_dV_second)
Ve2_inflection = V_di[mask_second][idx_second]

print("\nInflection Points:")
print(
    f"  First equivalence (inflection): {Ve1_inflection:.2f} mL (fitted: {Ve1_fitted:.2f} mL)"
)
print(
    f"  Second equivalence (inflection): {Ve2_inflection:.2f} mL (fitted: {Ve2_fitted:.2f} mL)"
)


# ======================================================================
# ## Visualization
#
# ======================================================================


fig = plt.figure(figsize=(16, 12))

# Plot 1: Monoprotic titration curve
ax1 = plt.subplot(3, 3, 1)
ax1.errorbar(
    V_titrant,
    pH_measured,
    yerr=sigma_pH,
    fmt="o",
    markersize=4,
    alpha=0.6,
    label="Measured pH",
    capsize=2,
)
ax1.plot(V_fit, pH_fitted_curve, "r-", linewidth=2, label="Fitted curve")
ax1.axvline(Ve_fitted, color="g", linestyle="--", label=f"Ve = {Ve_fitted:.2f} mL")
ax1.axhline(
    pKa_fitted,
    color="orange",
    linestyle="--",
    alpha=0.5,
    label=f"pKa = {pKa_fitted:.2f}",
)
ax1.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax1.set_ylabel("pH", fontsize=11)
ax1.set_title(
    "Monoprotic Weak Acid Titration\n(Acetic Acid + NaOH)",
    fontsize=12,
    fontweight="bold",
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: First derivative (slope)
ax2 = plt.subplot(3, 3, 2)
ax2.plot(V_titrant, dpH_dV, "b-", linewidth=2)
ax2.axvline(
    Ve_inflection,
    color="r",
    linestyle="--",
    label=f"Inflection: {Ve_inflection:.2f} mL",
)
ax2.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax2.set_ylabel("dpH/dV (pH/mL)", fontsize=11)
ax2.set_title(
    "First Derivative\n(Equivalence Point Detection)", fontsize=12, fontweight="bold"
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(V_fit, residuals, alpha=0.6, s=30)
ax3.axhline(0, color="r", linestyle="--", linewidth=1)
ax3.axhline(2 * rmse, color="orange", linestyle=":", label=f"±2σ ({2 * rmse:.3f})")
ax3.axhline(-2 * rmse, color="orange", linestyle=":")
ax3.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax3.set_ylabel("Residuals (pH units)", fontsize=11)
ax3.set_title(f"Residuals (RMSE = {rmse:.4f})", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Buffer capacity
ax4 = plt.subplot(3, 3, 4)
ax4.plot(pH_range, beta_fitted, "b-", linewidth=2, label="Fitted pKa")
ax4.plot(pH_range, beta_true, "r--", linewidth=2, alpha=0.5, label="True pKa")
ax4.axvline(
    pKa_fitted, color="g", linestyle="--", alpha=0.5, label=f"pKa = {pKa_fitted:.2f}"
)
ax4.axvline(pKa_fitted - 1, color="orange", linestyle=":", alpha=0.5)
ax4.axvline(pKa_fitted + 1, color="orange", linestyle=":", alpha=0.5, label="pKa ± 1")
ax4.set_xlabel("pH", fontsize=11)
ax4.set_ylabel("Buffer Capacity β (mol/L·pH)", fontsize=11)
ax4.set_title("Buffer Capacity vs pH", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Parameter comparison
ax5 = plt.subplot(3, 3, 5)
params_names = ["pKa", "Ve (mL)"]
params_true = [pKa_true, Ve_true]
params_fitted = [pKa_fitted, Ve_fitted]
params_err = [pKa_err, Ve_err]

x_pos = np.arange(len(params_names))
width = 0.35

ax5.bar(x_pos - width / 2, params_true, width, label="True", alpha=0.7, color="blue")
ax5.bar(
    x_pos + width / 2,
    params_fitted,
    width,
    yerr=params_err,
    label="Fitted",
    alpha=0.7,
    color="red",
    capsize=5,
)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(params_names)
ax5.set_ylabel("Value", fontsize=11)
ax5.set_title("Parameter Recovery", fontsize=12, fontweight="bold")
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Henderson-Hasselbalch verification
ax6 = plt.subplot(3, 3, 6)
# Calculate [A-]/[HA] ratio from fitted parameters
V_buffer = V_fit[(V_fit > 5) & (V_fit < 45)]
pH_buffer = pH_fitted_curve[(V_fit > 5) & (V_fit < 45)]
ratio_fitted = np.power(10, pH_buffer - pKa_fitted)
log_ratio = np.log10(ratio_fitted)

ax6.scatter(log_ratio, pH_buffer, alpha=0.6, s=30, label="Fitted data")
# Theoretical line: pH = pKa + log10(ratio)
log_ratio_theory = np.linspace(-1.5, 1.5, 100)
pH_theory = pKa_fitted + log_ratio_theory
ax6.plot(log_ratio_theory, pH_theory, "r--", linewidth=2, label="Henderson-Hasselbalch")
ax6.set_xlabel("log₁₀([A⁻]/[HA])", fontsize=11)
ax6.set_ylabel("pH", fontsize=11)
ax6.set_title("Henderson-Hasselbalch Verification", fontsize=12, fontweight="bold")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# Plot 7: Diprotic titration curve
ax7 = plt.subplot(3, 3, 7)
ax7.errorbar(
    V_di,
    pH_di_measured,
    yerr=sigma_pH_di,
    fmt="o",
    markersize=4,
    alpha=0.6,
    label="Measured pH",
    capsize=2,
)
ax7.plot(V_di, pH_di_fitted, "r-", linewidth=2, label="Fitted curve")
ax7.axvline(
    Ve1_fitted, color="g", linestyle="--", alpha=0.7, label=f"Ve1 = {Ve1_fitted:.1f} mL"
)
ax7.axvline(
    Ve2_fitted, color="b", linestyle="--", alpha=0.7, label=f"Ve2 = {Ve2_fitted:.1f} mL"
)
ax7.axhline(pKa1_fitted_di, color="orange", linestyle=":", alpha=0.5)
ax7.axhline(pKa2_fitted_di, color="purple", linestyle=":", alpha=0.5)
ax7.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax7.set_ylabel("pH", fontsize=11)
ax7.set_title(
    "Diprotic Acid Titration\n(Carbonic Acid)", fontsize=12, fontweight="bold"
)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 8: Diprotic first derivative
ax8 = plt.subplot(3, 3, 8)
ax8.plot(V_di, dpH_dV_di, "b-", linewidth=2)
ax8.axvline(
    Ve1_inflection, color="g", linestyle="--", label=f"1st: {Ve1_inflection:.1f} mL"
)
ax8.axvline(
    Ve2_inflection,
    color="purple",
    linestyle="--",
    label=f"2nd: {Ve2_inflection:.1f} mL",
)
ax8.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax8.set_ylabel("dpH/dV (pH/mL)", fontsize=11)
ax8.set_title(
    "Diprotic First Derivative\n(Two Equivalence Points)",
    fontsize=12,
    fontweight="bold",
)
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Plot 9: Diprotic residuals
ax9 = plt.subplot(3, 3, 9)
ax9.scatter(V_di, residuals_di, alpha=0.6, s=30)
ax9.axhline(0, color="r", linestyle="--", linewidth=1)
ax9.axhline(
    2 * rmse_di, color="orange", linestyle=":", label=f"±2σ ({2 * rmse_di:.3f})"
)
ax9.axhline(-2 * rmse_di, color="orange", linestyle=":")
ax9.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax9.set_ylabel("Residuals (pH units)", fontsize=11)
ax9.set_title(
    f"Diprotic Residuals (RMSE = {rmse_di:.4f})", fontsize=12, fontweight="bold"
)
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("titration_curves_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Plot saved as 'titration_curves_analysis.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "titration_curves"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


# ======================================================================
# ## Summary
#
# ======================================================================


print("\n" + "=" * 70)
print("SUMMARY: Acid-Base Titration Analysis")
print("=" * 70)

print("\n1. MONOPROTIC TITRATION (Acetic Acid):")
print(f"   ├─ pKa (fitted) = {pKa_fitted:.3f} ± {pKa_err:.3f}")
print(f"   ├─ pKa (true) = {pKa_true:.2f}")
print(f"   ├─ Equivalence point = {Ve_fitted:.2f} ± {Ve_err:.2f} mL")
print(f"   ├─ Buffer range = {pKa_fitted - 1:.2f} - {pKa_fitted + 1:.2f} (pKa ± 1)")
print(
    f"   ├─ Max buffer capacity = {max_beta:.4f} mol/(L·pH) at pH = {pH_max_beta:.2f}"
)
print(f"   └─ Fit quality: RMSE = {rmse:.4f}, χ²/dof = {reduced_chi_squared:.3f}")

print("\n2. DIPROTIC TITRATION (Carbonic Acid):")
print(
    f"   ├─ pKa1 (fitted) = {pKa1_fitted_di:.2f} ± {pKa1_err_di:.2f} (true: {pKa1_true_di:.2f})"
)
print(
    f"   ├─ pKa2 (fitted) = {pKa2_fitted_di:.2f} ± {pKa2_err_di:.2f} (true: {pKa2_true_di:.2f})"
)
print(f"   ├─ First equivalence = {Ve1_fitted:.2f} ± {Ve1_err:.2f} mL")
print(f"   ├─ Second equivalence = {Ve2_fitted:.2f} ± {Ve2_err:.2f} mL")
print(
    f"   └─ Fit quality: RMSE = {rmse_di:.4f}, χ²/dof = {chi_squared_di / dof_di:.3f}"
)

print("\n3. KEY INSIGHTS:")
print("   ├─ Henderson-Hasselbalch equation accurately models buffer region")
print("   ├─ Equivalence points identified from inflection points (max dpH/dV)")
print("   ├─ Buffer capacity maximized at pH = pKa")
print("   ├─ Effective buffering occurs within pKa ± 1 pH unit")
print("   └─ Diprotic acids show two distinct equivalence points")

print("\n4. APPLICATIONS:")
print("   ├─ Analytical chemistry: pKa determination")
print("   ├─ Buffer preparation: optimal pH range selection")
print("   ├─ Quality control: acid/base concentration verification")
print("   ├─ Environmental monitoring: water quality (alkalinity)")
print("   └─ Biochemistry: protein isoelectric point determination")

print("\n" + "=" * 70)
