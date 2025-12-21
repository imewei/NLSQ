"""
Advanced Acid-Base Titration Curve Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates acid-base titration curve analysis using NLSQ's
advanced fit() API and global optimization capabilities for robust pKa
and equivalence point determination.

Compared to 04_gallery/chemistry/titration_curves.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Henderson-Hasselbalch equation
- pKa determination from titration curves
- Equivalence point identification
- Buffer capacity analysis
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
    V_safe = jnp.maximum(V, 0.01)
    Ve_safe = jnp.maximum(Ve, V_safe + 0.1)

    f = V_safe / Ve_safe
    f = jnp.clip(f, 0.01, 0.99)

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
    V_safe = jnp.maximum(V, 0.01)

    f1 = V_safe / jnp.maximum(Ve1, V_safe + 0.1)
    f1 = jnp.clip(f1, 0.01, 0.99)

    f2 = jnp.maximum(0, (V_safe - Ve1)) / jnp.maximum(Ve2 - Ve1, 0.1)
    f2 = jnp.clip(f2, 0.01, 0.99)

    pH1 = pKa1 + jnp.log10(jnp.maximum(f1 / (1 - f1), 1e-10))
    pH2 = pKa2 + jnp.log10(jnp.maximum(f2 / (1 - f2), 1e-10))

    weight = jnp.where(V_safe < Ve1, 1.0, 0.0)
    pH = weight * pH1 + (1 - weight) * pH2

    return pH


def buffer_capacity(pH, pKa, C_total):
    """
    Calculate buffer capacity as a function of pH.

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


# Experimental parameters
V0_acid = 25.0  # mL of 0.1 M acetic acid
C_acid_true = 0.1  # M
C_base_true = 0.1  # M (NaOH)
pKa_true = 4.76  # Acetic acid
Ve_true = 25.0  # mL (equivalence point)

# Generate synthetic titration data
V_titrant = np.linspace(0.1, 40, 30 if QUICK else 100)

# Calculate true pH values
pH_true = np.zeros_like(V_titrant)
for i, V in enumerate(V_titrant):
    if Ve_true > V:
        f = V / Ve_true
        if 0.001 < f < 0.999:
            pH_true[i] = pKa_true + np.log10(f / (1 - f))
        elif f <= 0.001:
            pH_true[i] = 2.9
        else:
            pH_true[i] = 8.0
    elif Ve_true == V:
        pH_true[i] = 8.72
    else:
        excess = (V - Ve_true) * C_base_true
        total_volume = V0_acid + V
        pOH = -np.log10(excess / total_volume)
        pH_true[i] = 14 - pOH

# Add noise
noise_level = 0.05
pH_measured = pH_true + np.random.normal(0, noise_level, size=pH_true.shape)
sigma_pH = np.full_like(pH_measured, noise_level)


print("=" * 70)
print("ACID-BASE TITRATION: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# Monoprotic Acid Titration Analysis
# =============================================================================
print("\n" + "-" * 70)
print("MONOPROTIC ACID TITRATION (Acetic Acid)")
print("-" * 70)

# Fit only the buffer region (5-35 mL) for better pKa estimation
mask_fit = (V_titrant >= 5) & (V_titrant <= 35)
V_fit = V_titrant[mask_fit]
pH_fit = pH_measured[mask_fit]
sigma_fit = sigma_pH[mask_fit]

# Initial guess and bounds
p0 = [4.5, 24.0, 3.0]  # pKa, Ve, pH0
bounds_lower = [3.0, 20.0, 2.5]
bounds_upper = [6.0, 30.0, 4.0]

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt_robust, pcov_robust = fit(
    simplified_titration,
    V_fit,
    pH_fit,
    p0=p0,
    sigma=sigma_fit,
    bounds=(bounds_lower, bounds_upper),
    absolute_sigma=True,
    preset="robust",
    **FIT_KWARGS,
)

pKa_fit, Ve_fit, pH0_fit = popt_robust
pKa_err, Ve_err, pH0_err = np.sqrt(np.diag(pcov_robust))

print(f"  pKa = {pKa_fit:.3f} +/- {pKa_err:.3f} (true: {pKa_true})")
print(f"  Ve = {Ve_fit:.2f} +/- {Ve_err:.2f} mL (true: {Ve_true})")

if QUICK:
    print("\n⏩ Quick mode: skipping global/custom multi-start fits.")
else:
    # Method 2: fit() with 'global' preset
    global_starts = 20
    print(f"\nMethod 2: fit() with 'global' preset ({global_starts} starts)")
    popt_global, pcov_global = fit(
        simplified_titration,
        V_fit,
        pH_fit,
        p0=p0,
        sigma=sigma_fit,
        bounds=(bounds_lower, bounds_upper),
        absolute_sigma=True,
        preset="global",
        n_starts=global_starts,
    )

    pKa_g, Ve_g, pH0_g = popt_global
    perr_g = np.sqrt(np.diag(pcov_global))

    print(f"  pKa = {pKa_g:.3f} +/- {perr_g[0]:.3f}")
    print(f"  Ve = {Ve_g:.2f} +/- {perr_g[1]:.2f} mL")

    # Method 3: GlobalOptimizationConfig with custom settings
    print("\nMethod 3: GlobalOptimizationConfig with custom settings")

    global_config = GlobalOptimizationConfig(
        n_starts=15,
        sampler="lhs",
        center_on_p0=True,
        scale_factor=1.0,
    )

    popt_custom, pcov_custom = fit(
        simplified_titration,
        V_fit,
        pH_fit,
        p0=p0,
        sigma=sigma_fit,
        bounds=(bounds_lower, bounds_upper),
        absolute_sigma=True,
        multistart=True,
        n_starts=global_config.n_starts,
        sampler="lhs",
    )

    pKa_c, Ve_c, pH0_c = popt_custom
    perr_c = np.sqrt(np.diag(pcov_custom))

    print(f"  pKa = {pKa_c:.3f} +/- {perr_c[0]:.3f}")
    print(f"  Ve = {Ve_c:.2f} +/- {perr_c[1]:.2f} mL")

# Use robust preset results for analysis
pKa_fit, Ve_fit, pH0_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
pKa_err, Ve_err, pH0_err = perr

# Goodness of fit
pH_fitted_curve = simplified_titration(V_fit, *popt_robust)
residuals = pH_fit - pH_fitted_curve
chi_squared = np.sum((residuals / sigma_fit) ** 2)
dof = len(pH_fit) - len(popt_robust)
reduced_chi_squared = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  pKa = {pKa_fit:.3f} +/- {pKa_err:.3f}")
print(f"  Equivalence point = {Ve_fit:.2f} +/- {Ve_err:.2f} mL")
print(f"  Initial pH = {pH0_fit:.2f} +/- {pH0_err:.2f}")

print("\nTrue Values:")
print(f"  pKa = {pKa_true:.2f}")
print(f"  Ve = {Ve_true:.1f} mL")

print("\nGoodness of Fit:")
print(f"  chi^2 = {chi_squared:.2f}")
print(f"  chi^2/dof = {reduced_chi_squared:.3f} (should be ~1)")
print(f"  RMSE = {rmse:.4f} pH units")

# Buffer range
print("\nBuffer Properties:")
print(f"  Effective buffer range: {pKa_fit - 1:.2f} - {pKa_fit + 1:.2f} (pKa +/- 1)")

# Calculate first derivative
dV = V_titrant[1] - V_titrant[0]
dpH_dV = np.gradient(pH_measured, dV)
inflection_idx = np.argmax(dpH_dV)
Ve_inflection = V_titrant[inflection_idx]
pH_inflection = pH_measured[inflection_idx]

print("\nInflection Point Analysis:")
print(f"  Volume at inflection = {Ve_inflection:.2f} mL")
print(f"  pH at inflection = {pH_inflection:.2f}")
print(f"  Max slope = {dpH_dV[inflection_idx]:.2f} pH/mL")


# =============================================================================
# Buffer Capacity Analysis
# =============================================================================
print("\n" + "-" * 70)
print("BUFFER CAPACITY ANALYSIS")
print("-" * 70)

pH_range = np.linspace(3, 7, 80 if QUICK else 200)
beta_fitted = buffer_capacity(pH_range, pKa_fit, C_acid_true)
beta_true = buffer_capacity(pH_range, pKa_true, C_acid_true)

max_beta_idx = np.argmax(beta_fitted)
pH_max_beta = pH_range[max_beta_idx]
max_beta = beta_fitted[max_beta_idx]

print(f"Maximum capacity at pH = {pH_max_beta:.2f} (should equal pKa = {pKa_fit:.2f})")
print(f"Maximum beta = {max_beta:.4f} mol/(L*pH)")
print(f"Effective buffer range: {pKa_fit - 1:.2f} - {pKa_fit + 1:.2f} (pKa +/- 1)")


# =============================================================================
# Diprotic Acid Titration (Carbonic Acid)
# =============================================================================
if QUICK:
    print("⏩ Quick mode: skipping buffer capacity and diprotic sections.")
    sys.exit(0)

print("\n" + "-" * 70)
print("DIPROTIC ACID TITRATION (Carbonic Acid) with fit() API")
print("-" * 70)

# True parameters for H2CO3
pKa1_true_di = 6.35
pKa2_true_di = 10.33
Ve1_true = 25.0
Ve2_true = 50.0

# Generate synthetic diprotic titration data
V_di = np.linspace(0.1, 70, 150)
pH_di_true = np.zeros_like(V_di)

for i, V in enumerate(V_di):
    if Ve1_true > V:
        f = V / Ve1_true
        if 0.01 < f < 0.99:
            pH_di_true[i] = pKa1_true_di + np.log10(f / (1 - f))
        else:
            pH_di_true[i] = 4.0
    elif Ve2_true > V:
        f = (V - Ve1_true) / (Ve2_true - Ve1_true)
        if 0.01 < f < 0.99:
            pH_di_true[i] = pKa2_true_di + np.log10(f / (1 - f))
        else:
            pH_di_true[i] = 8.3
    else:
        pH_di_true[i] = 12.0

pH_di_measured = pH_di_true + np.random.normal(0, 0.08, size=pH_di_true.shape)
sigma_pH_di = np.full_like(pH_di_measured, 0.08)

# Fit diprotic model with robust preset
p0_di = [6.0, 10.0, 24.0, 48.0]
bounds_lower_di = [5.0, 9.0, 20.0, 45.0]
bounds_upper_di = [7.0, 11.0, 30.0, 55.0]

popt_di, pcov_di = fit(
    diprotic_titration,
    V_di,
    pH_di_measured,
    p0=p0_di,
    sigma=sigma_pH_di,
    bounds=(bounds_lower_di, bounds_upper_di),
    absolute_sigma=True,
    preset="robust",
)

pKa1_fit_di, pKa2_fit_di, Ve1_fit, Ve2_fit = popt_di
pKa1_err_di, pKa2_err_di, Ve1_err, Ve2_err = np.sqrt(np.diag(pcov_di))

print("Fitted Parameters (robust preset):")
print(f"  pKa1 = {pKa1_fit_di:.2f} +/- {pKa1_err_di:.2f} (true: {pKa1_true_di:.2f})")
print(f"  pKa2 = {pKa2_fit_di:.2f} +/- {pKa2_err_di:.2f} (true: {pKa2_true_di:.2f})")
print(f"  Ve1 = {Ve1_fit:.2f} +/- {Ve1_err:.2f} mL (true: {Ve1_true:.1f} mL)")
print(f"  Ve2 = {Ve2_fit:.2f} +/- {Ve2_err:.2f} mL (true: {Ve2_true:.1f} mL)")

pH_di_fitted = diprotic_titration(V_di, *popt_di)
residuals_di = pH_di_measured - pH_di_fitted
rmse_di = np.sqrt(np.mean(residuals_di**2))
chi_squared_di = np.sum((residuals_di / sigma_pH_di) ** 2)
dof_di = len(pH_di_measured) - len(popt_di)

print("\nGoodness of Fit:")
print(f"  RMSE = {rmse_di:.4f} pH units")
print(f"  chi^2/dof = {chi_squared_di / dof_di:.3f}")


# =============================================================================
# Visualization
# =============================================================================
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
ax1.plot(V_fit, pH_fitted_curve, "r-", linewidth=2, label="Fitted curve (robust)")
ax1.axvline(Ve_fit, color="g", linestyle="--", label=f"Ve = {Ve_fit:.2f} mL")
ax1.axhline(
    pKa_fit, color="orange", linestyle="--", alpha=0.5, label=f"pKa = {pKa_fit:.2f}"
)

ax1.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax1.set_ylabel("pH", fontsize=11)
ax1.set_title("Monoprotic Titration - fit() API", fontsize=12, fontweight="bold")
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
ax3.axhline(
    2 * rmse, color="orange", linestyle=":", label=f"+/- 2sigma ({2 * rmse:.3f})"
)
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
ax4.axvline(pKa_fit, color="g", linestyle="--", alpha=0.5, label=f"pKa = {pKa_fit:.2f}")
ax4.axvline(pKa_fit - 1, color="orange", linestyle=":", alpha=0.5)
ax4.axvline(pKa_fit + 1, color="orange", linestyle=":", alpha=0.5, label="pKa +/- 1")

ax4.set_xlabel("pH", fontsize=11)
ax4.set_ylabel("Buffer Capacity beta (mol/L*pH)", fontsize=11)
ax4.set_title("Buffer Capacity vs pH", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Parameter comparison
ax5 = plt.subplot(3, 3, 5)
params_names = ["pKa", "Ve (mL)"]
params_true_vals = [pKa_true, Ve_true]
params_fitted = [pKa_fit, Ve_fit]
params_err = [pKa_err, Ve_err]

x_pos = np.arange(len(params_names))
width = 0.35

ax5.bar(
    x_pos - width / 2, params_true_vals, width, label="True", alpha=0.7, color="blue"
)
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
V_buffer = V_fit[(V_fit > 5) & (V_fit < 45)]
pH_buffer = pH_fitted_curve[(V_fit > 5) & (V_fit < 45)]
ratio_fitted = np.power(10.0, np.asarray(pH_buffer) - pKa_fit)
log_ratio = np.log10(ratio_fitted)

ax6.scatter(log_ratio, pH_buffer, alpha=0.6, s=30, label="Fitted data")
log_ratio_theory = np.linspace(-1.5, 1.5, 100)
pH_theory = pKa_fit + log_ratio_theory
ax6.plot(log_ratio_theory, pH_theory, "r--", linewidth=2, label="Henderson-Hasselbalch")

ax6.set_xlabel("log10([A-]/[HA])", fontsize=11)
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
    Ve1_fit, color="g", linestyle="--", alpha=0.7, label=f"Ve1 = {Ve1_fit:.1f} mL"
)
ax7.axvline(
    Ve2_fit, color="b", linestyle="--", alpha=0.7, label=f"Ve2 = {Ve2_fit:.1f} mL"
)
ax7.axhline(pKa1_fit_di, color="orange", linestyle=":", alpha=0.5)
ax7.axhline(pKa2_fit_di, color="purple", linestyle=":", alpha=0.5)

ax7.set_xlabel("Volume of NaOH (mL)", fontsize=11)
ax7.set_ylabel("pH", fontsize=11)
ax7.set_title("Diprotic Acid Titration - fit() API", fontsize=12, fontweight="bold")
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# Plot 8: Diprotic first derivative
ax8 = plt.subplot(3, 3, 8)
dpH_dV_di = np.gradient(pH_di_measured, V_di[1] - V_di[0])
ax8.plot(V_di, dpH_dV_di, "b-", linewidth=2)

mask_first = V_di < 35
idx_first = np.argmax(dpH_dV_di[mask_first])
Ve1_inflection = V_di[mask_first][idx_first]

mask_second = V_di > 35
dpH_dV_second = dpH_dV_di[mask_second]
idx_second = np.argmax(dpH_dV_second)
Ve2_inflection = V_di[mask_second][idx_second]

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
ax8.set_title("Diprotic First Derivative", fontsize=12, fontweight="bold")
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# Plot 9: API comparison
ax9 = plt.subplot(3, 3, 9)
ax9.axis("off")

summary_text = [
    ["Method", "pKa", "Ve (mL)"],
    ["-" * 20, "-" * 8, "-" * 10],
    ["fit() 'robust'", f"{pKa_fit:.3f}", f"{Ve_fit:.2f}"],
    ["fit() 'global'", f"{pKa_g:.3f}", f"{Ve_g:.2f}"],
    ["fit() custom", f"{pKa_c:.3f}", f"{Ve_c:.2f}"],
    ["", "", ""],
    ["True values", f"{pKa_true:.3f}", f"{Ve_true:.2f}"],
    ["", "", ""],
    ["Diprotic (robust):", "", ""],
    [f"  pKa1 = {pKa1_fit_di:.2f}", "", ""],
    [f"  pKa2 = {pKa2_fit_di:.2f}", "", ""],
    ["", "", ""],
    ["Advantages of fit():", "", ""],
    ["  - Multi-start opt.", "", ""],
    ["  - Global search", "", ""],
]

table_text = "\n".join(["  ".join(row) for row in summary_text])
ax9.text(
    0.1,
    0.9,
    table_text,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
    transform=ax9.transAxes,
)
ax9.set_title("fit() API Comparison", fontsize=12, fontweight="bold")

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "titration_curves"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY: Acid-Base Titration Analysis")
print("=" * 70)

print("\n1. MONOPROTIC TITRATION (Acetic Acid):")
print(f"   pKa (fitted) = {pKa_fit:.3f} +/- {pKa_err:.3f}")
print(f"   pKa (true) = {pKa_true:.2f}")
print(f"   Equivalence point = {Ve_fit:.2f} +/- {Ve_err:.2f} mL")
print(f"   Buffer range = {pKa_fit - 1:.2f} - {pKa_fit + 1:.2f} (pKa +/- 1)")
print(f"   Max buffer capacity = {max_beta:.4f} mol/(L*pH) at pH = {pH_max_beta:.2f}")
print(f"   Fit quality: RMSE = {rmse:.4f}, chi^2/dof = {reduced_chi_squared:.3f}")

print("\n2. DIPROTIC TITRATION (Carbonic Acid):")
print(
    f"   pKa1 (fitted) = {pKa1_fit_di:.2f} +/- {pKa1_err_di:.2f} (true: {pKa1_true_di:.2f})"
)
print(
    f"   pKa2 (fitted) = {pKa2_fit_di:.2f} +/- {pKa2_err_di:.2f} (true: {pKa2_true_di:.2f})"
)
print(f"   First equivalence = {Ve1_fit:.2f} +/- {Ve1_err:.2f} mL")
print(f"   Second equivalence = {Ve2_fit:.2f} +/- {Ve2_err:.2f} mL")
print(
    f"   Fit quality: RMSE = {rmse_di:.4f}, chi^2/dof = {chi_squared_di / dof_di:.3f}"
)

print("\n3. API Methods Used:")
print("   - fit() with preset='robust' (5 multi-starts)")
print("   - fit() with preset='global' (20 multi-starts)")
print("   - fit() with GlobalOptimizationConfig (custom settings)")

print("\n4. KEY INSIGHTS:")
print("   - Henderson-Hasselbalch equation accurately models buffer region")
print("   - Equivalence points identified from inflection points (max dpH/dV)")
print("   - Buffer capacity maximized at pH = pKa")
print("   - fit() API provides robust parameter estimation")
print("=" * 70)
