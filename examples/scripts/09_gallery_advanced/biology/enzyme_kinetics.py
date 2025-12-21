"""
Advanced Enzyme Kinetics Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting enzyme kinetics data using the Michaelis-Menten
model with NLSQ's advanced fit() API and global optimization capabilities for
robust K_M and V_max determination.

Compared to 04_gallery/biology/enzyme_kinetics.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Michaelis-Menten kinetics
- K_M and V_max determination
- Competitive inhibition analysis
- Global optimization to avoid local minima
- Multi-start optimization for robust parameter estimation
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


def michaelis_menten(S, Vmax, Km):
    """
    Michaelis-Menten enzyme kinetics model.

    v = V_max * [S] / (K_M + [S])

    Parameters
    ----------
    S : array_like
        Substrate concentration (uM)
    Vmax : float
        Maximum reaction velocity (uM/min)
    Km : float
        Michaelis constant (uM) - substrate concentration at v = Vmax/2

    Returns
    -------
    v : array_like
        Reaction velocity (uM/min)
    """
    return Vmax * S / (Km + S)


def competitive_inhibition(S, Vmax, Km, I, Ki):
    """
    Competitive inhibition model.

    v = V_max * [S] / (K_M(1 + [I]/K_i) + [S])

    K_M appears increased, V_max unchanged.

    Parameters
    ----------
    S : array_like
        Substrate concentration (uM)
    Vmax : float
        Maximum velocity (uM/min)
    Km : float
        Michaelis constant (uM)
    I : float
        Inhibitor concentration (uM)
    Ki : float
        Inhibition constant (uM)

    Returns
    -------
    v : array_like
        Reaction velocity (uM/min)
    """
    Km_app = Km * (1 + I / Ki)
    return Vmax * S / (Km_app + S)


def hill_equation(S, Vmax, K50, n):
    """
    Hill equation for cooperative binding.

    v = V_max * [S]^n / (K_50^n + [S]^n)

    Parameters
    ----------
    S : array_like
        Substrate concentration (uM)
    Vmax : float
        Maximum velocity (uM/min)
    K50 : float
        Half-saturation constant (uM)
    n : float
        Hill coefficient (cooperativity)

    Returns
    -------
    v : array_like
        Reaction velocity (uM/min)
    """
    return Vmax * jnp.power(S, n) / (jnp.power(K50, n) + jnp.power(S, n))


# Substrate concentrations (uM)
S = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

# True enzyme parameters (typical for many enzymes)
Vmax_true = 100.0  # uM/min
Km_true = 50.0  # uM

# Generate velocity data
v_true = michaelis_menten(S, Vmax_true, Km_true)

# Add experimental noise (~5% relative error typical for enzyme assays)
noise = np.random.normal(0, 0.05 * v_true, size=len(S))
v_measured = v_true + noise

# Measurement uncertainties (5% relative)
sigma_v = 0.05 * v_measured + 1.0  # Add small constant for low velocities


print("=" * 70)
print("ENZYME KINETICS: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# Initial parameter guess
p0 = [120, 60]  # Vmax, Km

# =============================================================================
# Method 1: Using fit() with 'robust' preset
# =============================================================================
print("\n" + "-" * 70)
print("Method 1: fit() with 'robust' preset")
print("-" * 70)

popt_robust, pcov_robust = fit(
    michaelis_menten,
    S,
    v_measured,
    p0=p0,
    sigma=sigma_v,
    absolute_sigma=True,
    bounds=([0, 0], [200, 500]),
    preset="robust",
    **FIT_KWARGS,
)

Vmax_fit, Km_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
Vmax_err, Km_err = perr

print(f"  V_max = {Vmax_fit:.2f} +/- {Vmax_err:.2f} uM/min (true: {Vmax_true})")
print(f"  K_M = {Km_fit:.2f} +/- {Km_err:.2f} uM (true: {Km_true})")

if QUICK:
    print("\n⏩ Quick mode: skipping global/custom fits and extended analyses.")
    sys.exit(0)


# =============================================================================
# Method 2: Using fit() with 'global' preset for thorough search
# =============================================================================
print("\n" + "-" * 70)
print("Method 2: fit() with 'global' preset (20 starts)")
print("-" * 70)

popt_global, pcov_global = fit(
    michaelis_menten,
    S,
    v_measured,
    p0=p0,
    sigma=sigma_v,
    absolute_sigma=True,
    bounds=([0, 0], [200, 500]),
    preset="global",
)

Vmax_g, Km_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  V_max = {Vmax_g:.2f} +/- {perr_g[0]:.2f} uM/min")
print(f"  K_M = {Km_g:.2f} +/- {perr_g[1]:.2f} uM")


# =============================================================================
# Method 3: Using GlobalOptimizationConfig with custom settings
# =============================================================================
print("\n" + "-" * 70)
print("Method 3: GlobalOptimizationConfig with custom settings")
print("-" * 70)

# Create custom global optimization configuration
global_config = GlobalOptimizationConfig(
    n_starts=4 if QUICK else 15,
    sampler="lhs",
    center_on_p0=True,
    scale_factor=1.0,
)

# Use explicit multi-start parameters with fit()
popt_custom, pcov_custom = fit(
    michaelis_menten,
    S,
    v_measured,
    p0=p0,
    sigma=sigma_v,
    absolute_sigma=True,
    bounds=([0, 0], [200, 500]),
    multistart=True,
    n_starts=4 if QUICK else 15,
    sampler="lhs",
)

Vmax_c, Km_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))

print(f"  V_max = {Vmax_c:.2f} +/- {perr_c[0]:.2f} uM/min")
print(f"  K_M = {Km_c:.2f} +/- {perr_c[1]:.2f} uM")


# Use robust preset results for analysis
Vmax_fit, Km_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
Vmax_err, Km_err = perr

print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  V_max: {Vmax_fit:.2f} +/- {Vmax_err:.2f} uM/min")
print(f"  K_M:   {Km_fit:.2f} +/- {Km_err:.2f} uM")

print("\nTrue Values:")
print(f"  V_max: {Vmax_true:.2f} uM/min")
print(f"  K_M:   {Km_true:.2f} uM")

print("\nErrors:")
print(
    f"  V_max: {abs(Vmax_fit - Vmax_true):.2f} uM/min "
    + f"({100 * abs(Vmax_fit - Vmax_true) / Vmax_true:.1f}%)"
)
print(
    f"  K_M:   {abs(Km_fit - Km_true):.2f} uM "
    + f"({100 * abs(Km_fit - Km_true) / Km_true:.1f}%)"
)

# Catalytic efficiency
kcat_Km = Vmax_fit / Km_fit  # Assuming [E]_total = 1 uM
print(f"\nCatalytic efficiency (k_cat/K_M): {kcat_Km:.3f} min^-1")

# Velocity at physiological substrate concentration (example: 20 uM)
S_physiol = 20.0
v_physiol = michaelis_menten(S_physiol, Vmax_fit, Km_fit)
saturation = 100 * v_physiol / Vmax_fit
print(f"\nAt [S] = {S_physiol:.1f} uM (physiological):")
print(f"  Velocity: {v_physiol:.2f} uM/min ({saturation:.1f}% of V_max)")

# Goodness of fit
residuals = v_measured - michaelis_menten(S, *popt_robust)
chi_squared = np.sum((residuals / sigma_v) ** 2)
dof = len(S) - len(popt_robust)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.2f} uM/min")
print(f"  chi^2/dof:  {chi_squared_reduced:.2f}")


# =============================================================================
# Competitive Inhibition Analysis
# =============================================================================
print("\n" + "-" * 70)
print("COMPETITIVE INHIBITION ANALYSIS with fit() API")
print("-" * 70)

# Simulate competitive inhibitor data
I_conc = 100.0  # uM inhibitor concentration
Ki_true = 25.0  # uM inhibition constant

v_inhibited_true = competitive_inhibition(S, Vmax_true, Km_true, I_conc, Ki_true)
noise_inh = np.random.normal(0, 0.05 * v_inhibited_true, size=len(S))
v_inhibited = v_inhibited_true + noise_inh


# Fit with inhibition model
def competitive_inh_fit(S, Vmax, Km, Ki):
    return competitive_inhibition(S, Vmax, Km, I_conc, Ki)


popt_inh, pcov_inh = fit(
    competitive_inh_fit,
    S,
    v_inhibited,
    p0=[100, 50, 30],
    sigma=sigma_v,
    bounds=([0, 0, 0], [200, 500, 200]),
    preset="robust",
)

Vmax_inh, Km_inh, Ki_fit = popt_inh
perr_inh = np.sqrt(np.diag(pcov_inh))
Ki_err = perr_inh[2]

print(f"With [{I_conc:.0f} uM] inhibitor:")
print(f"  Apparent K_M: {Km_inh:.2f} uM (increased from {Km_fit:.2f} uM)")
print(f"  V_max:        {Vmax_inh:.2f} uM/min (unchanged)")
print(f"  K_i:          {Ki_fit:.2f} +/- {Ki_err:.2f} uM")
print(f"  True K_i:     {Ki_true:.2f} uM")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Michaelis-Menten curve
ax1 = plt.subplot(3, 2, 1)
ax1.errorbar(
    S,
    v_measured,
    yerr=sigma_v,
    fmt="o",
    capsize=4,
    markersize=8,
    label="Experimental data",
    alpha=0.7,
)

S_fine = np.linspace(0, 1000, 200)
ax1.plot(
    S_fine,
    michaelis_menten(S_fine, Vmax_true, Km_true),
    "r--",
    linewidth=2,
    label="True curve",
    alpha=0.7,
)
ax1.plot(
    S_fine,
    michaelis_menten(S_fine, *popt_robust),
    "g-",
    linewidth=2.5,
    label="Fitted curve (robust)",
)

# Mark K_M and V_max
ax1.axhline(
    Vmax_fit, color="blue", linestyle=":", alpha=0.5, label=f"V_max = {Vmax_fit:.1f}"
)
ax1.axhline(Vmax_fit / 2, color="orange", linestyle=":", alpha=0.5)
ax1.axvline(
    Km_fit, color="orange", linestyle=":", alpha=0.5, label=f"K_M = {Km_fit:.1f} uM"
)
ax1.plot(Km_fit, Vmax_fit / 2, "o", markersize=10, color="orange")

ax1.set_xlabel("Substrate Concentration [S] (uM)", fontsize=12)
ax1.set_ylabel("Velocity v (uM/min)", fontsize=12)
ax1.set_title("Michaelis-Menten Kinetics - fit() API", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: Lineweaver-Burk plot
ax2 = plt.subplot(3, 2, 2)
S_inv = 1 / S
v_inv = 1 / v_measured


def linear_lb(S_inv, slope, intercept):
    return slope * S_inv + intercept


popt_lb, _ = fit(
    linear_lb,
    S_inv,
    v_inv,
    p0=[Km_fit / Vmax_fit, 1 / Vmax_fit],
)

slope_lb, intercept_lb = popt_lb

ax2.plot(S_inv, v_inv, "o", markersize=8, alpha=0.7, label="Data (1/v vs 1/[S])")
S_inv_fine = np.linspace(0, S_inv.max(), 100)
ax2.plot(
    S_inv_fine, linear_lb(S_inv_fine, *popt_lb), "g-", linewidth=2.5, label="Linear fit"
)

ax2.axhline(
    intercept_lb,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label=f"1/V_max = {intercept_lb:.4f}",
)

ax2.set_xlabel("1/[S] (uM^-1)")
ax2.set_ylabel("1/v (min/uM)")
ax2.set_title("Lineweaver-Burk Plot")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = plt.subplot(3, 2, 3)
normalized_residuals = residuals / sigma_v
ax3.plot(S, normalized_residuals, "o", markersize=6, alpha=0.7)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Substrate Concentration [S] (uM)")
ax3.set_ylabel("Normalized Residuals (sigma)")
ax3.set_title("Fit Residuals")
ax3.set_xscale("log")
ax3.grid(True, alpha=0.3)

# Plot 4: Competitive inhibition comparison
ax4 = plt.subplot(3, 2, 4)
ax4.plot(S, v_measured, "o", markersize=8, label="No inhibitor", alpha=0.7)
ax4.plot(S, v_inhibited, "s", markersize=8, label=f"[I] = {I_conc:.0f} uM", alpha=0.7)

ax4.plot(
    S_fine,
    michaelis_menten(S_fine, *popt_robust),
    "g-",
    linewidth=2,
    label="No inhibitor fit",
)
ax4.plot(
    S_fine,
    competitive_inh_fit(S_fine, *popt_inh),
    "b-",
    linewidth=2,
    label="With inhibitor fit",
)

ax4.set_xlabel("Substrate Concentration [S] (uM)")
ax4.set_ylabel("Velocity v (uM/min)")
ax4.set_title(f"Competitive Inhibition (K_i = {Ki_fit:.1f} uM)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Saturation curve (normalized)
ax5 = plt.subplot(3, 2, 5)
v_normalized = v_measured / Vmax_fit
S_normalized = S / Km_fit

ax5.semilogx(S_normalized, v_normalized, "o", markersize=8, alpha=0.7, label="Data")

S_norm_fine = np.logspace(-2, 2, 200)
v_norm_fine = S_norm_fine / (1 + S_norm_fine)
ax5.semilogx(S_norm_fine, v_norm_fine, "g-", linewidth=2.5, label="Universal curve")

ax5.axvline(1, color="orange", linestyle="--", linewidth=1.5, label="[S] = K_M")
ax5.axhline(0.5, color="blue", linestyle=":", alpha=0.5, label="v = 0.5 V_max")

ax5.set_xlabel("[S]/K_M (dimensionless)")
ax5.set_ylabel("v/V_max (dimensionless)")
ax5.set_title("Normalized Saturation Curve")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: API comparison
ax6 = plt.subplot(3, 2, 6)
ax6.axis("off")

api_table = [
    ["Method", "V_max (uM/min)", "K_M (uM)"],
    ["-" * 25, "-" * 15, "-" * 10],
    ["fit() 'robust'", f"{Vmax_fit:.2f}", f"{Km_fit:.2f}"],
    ["fit() 'global'", f"{Vmax_g:.2f}", f"{Km_g:.2f}"],
    ["fit() custom 15-start", f"{Vmax_c:.2f}", f"{Km_c:.2f}"],
    ["", "", ""],
    ["True values", f"{Vmax_true:.2f}", f"{Km_true:.2f}"],
    ["", "", ""],
    ["Key Advantages of fit():", "", ""],
    ["-" * 35, "", ""],
    ["  - Automatic multi-start", "", ""],
    ["  - GlobalOptimizationConfig", "", ""],
    ["  - Preset configurations", "", ""],
    ["  - Robust parameter recovery", "", ""],
]

table_text = "\n".join(["  ".join(row) for row in api_table])
ax6.text(
    0.1,
    0.9,
    table_text,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    transform=ax6.transAxes,
)
ax6.set_title("fit() API Methods Comparison", fontsize=12, fontweight="bold")

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "enzyme_kinetics"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Enzyme kinetics parameters determined using fit() API:")
print(f"\n  Michaelis constant (K_M):  {Km_fit:.2f} +/- {Km_err:.2f} uM")
print(f"  Maximum velocity (V_max):  {Vmax_fit:.2f} +/- {Vmax_err:.2f} uM/min")
print(f"  Catalytic efficiency:      {kcat_Km:.3f} min^-1")
print("\nInhibition analysis:")
print(f"  Inhibition constant (K_i): {Ki_fit:.2f} +/- {Ki_err:.2f} uM")
print("  Type: Competitive (K_M increased, V_max unchanged)")
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print("  - fit() with preset='global' (20 multi-starts)")
print("  - fit() with GlobalOptimizationConfig (custom settings)")
print("\nThis example demonstrates:")
print("  - Michaelis-Menten fitting with fit() API")
print("  - Global optimization for robust parameter estimation")
print("  - Competitive inhibition analysis")
print("  - Lineweaver-Burk linearization")
print("=" * 70)
