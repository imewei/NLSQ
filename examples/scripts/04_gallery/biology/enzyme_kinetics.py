"""
Converted from enzyme_kinetics.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Enzyme Kinetics: Michaelis-Menten Analysis
#
#
# This example demonstrates fitting enzyme kinetics data using the Michaelis-Menten
# model to determine K_M (Michaelis constant) and V_max (maximum velocity). We also
# show Lineweaver-Burk transformation and inhibition analysis.
#
# Key Concepts:
# - Michaelis-Menten kinetics
# - K_M and V_max determination
# - Lineweaver-Burk plot (double reciprocal)
# - Competitive vs non-competitive inhibition
# - Hill equation for cooperative binding
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed
np.random.seed(42)


def michaelis_menten(S, Vmax, Km):
    """
    Michaelis-Menten enzyme kinetics model.

    v = V_max * [S] / (K_M + [S])

    Parameters
    ----------
    S : array_like
        Substrate concentration (μM)
    Vmax : float
        Maximum reaction velocity (μM/min)
    Km : float
        Michaelis constant (μM) - substrate concentration at v = Vmax/2

    Returns
    -------
    v : array_like
        Reaction velocity (μM/min)
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
        Substrate concentration (μM)
    Vmax : float
        Maximum velocity (μM/min)
    Km : float
        Michaelis constant (μM)
    I : float
        Inhibitor concentration (μM)
    Ki : float
        Inhibition constant (μM)

    Returns
    -------
    v : array_like
        Reaction velocity (μM/min)
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
        Substrate concentration (μM)
    Vmax : float
        Maximum velocity (μM/min)
    K50 : float
        Half-saturation constant (μM)
    n : float
        Hill coefficient (cooperativity)

    Returns
    -------
    v : array_like
        Reaction velocity (μM/min)
    """
    return Vmax * S**n / (K50**n + S**n)


# Substrate concentrations (μM)
S = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])

# True enzyme parameters (typical for many enzymes)
Vmax_true = 100.0  # μM/min
Km_true = 50.0  # μM

# Generate velocity data
v_true = michaelis_menten(S, Vmax_true, Km_true)

# Add experimental noise (~5% relative error typical for enzyme assays)
noise = np.random.normal(0, 0.05 * v_true, size=len(S))
v_measured = v_true + noise

# Measurement uncertainties (5% relative)
sigma_v = 0.05 * v_measured + 1.0  # Add small constant for low velocities


print("=" * 70)
print("ENZYME KINETICS: MICHAELIS-MENTEN ANALYSIS")
print("=" * 70)

# Initial parameter guess
p0 = [120, 60]  # Vmax, Km

# Fit the model
popt, pcov = curve_fit(
    michaelis_menten,
    S,
    v_measured,
    p0=p0,
    sigma=sigma_v,
    absolute_sigma=True,
    bounds=([0, 0], [200, 500]),  # Physical bounds
)

Vmax_fit, Km_fit = popt
perr = np.sqrt(np.diag(pcov))
Vmax_err, Km_err = perr


print("\nFitted Parameters:")
print(f"  V_max: {Vmax_fit:.2f} ± {Vmax_err:.2f} μM/min")
print(f"  K_M:   {Km_fit:.2f} ± {Km_err:.2f} μM")

print("\nTrue Values:")
print(f"  V_max: {Vmax_true:.2f} μM/min")
print(f"  K_M:   {Km_true:.2f} μM")

print("\nErrors:")
print(
    f"  V_max: {abs(Vmax_fit - Vmax_true):.2f} μM/min "
    + f"({100 * abs(Vmax_fit - Vmax_true) / Vmax_true:.1f}%)"
)
print(
    f"  K_M:   {abs(Km_fit - Km_true):.2f} μM "
    + f"({100 * abs(Km_fit - Km_true) / Km_true:.1f}%)"
)

# Catalytic efficiency
kcat_Km = Vmax_fit / Km_fit  # Assuming [E]_total = 1 μM
print(f"\nCatalytic efficiency (k_cat/K_M): {kcat_Km:.3f} min⁻¹")

# Velocity at physiological substrate concentration (example: 20 μM)
S_physiol = 20.0
v_physiol = michaelis_menten(S_physiol, Vmax_fit, Km_fit)
saturation = 100 * v_physiol / Vmax_fit
print(f"\nAt [S] = {S_physiol:.1f} μM (physiological):")
print(f"  Velocity: {v_physiol:.2f} μM/min ({saturation:.1f}% of V_max)")

# Goodness of fit
residuals = v_measured - michaelis_menten(S, *popt)
chi_squared = np.sum((residuals / sigma_v) ** 2)
dof = len(S) - len(popt)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.2f} μM/min")
print(f"  χ²/dof:  {chi_squared_reduced:.2f}")


print("\n" + "-" * 70)
print("LINEWEAVER-BURK TRANSFORMATION")
print("-" * 70)

# 1/v vs 1/S
# 1/v = (K_M/V_max) * (1/S) + 1/V_max
# Slope = K_M/V_max, Intercept = 1/V_max

S_inv = 1 / S
v_inv = 1 / v_measured
sigma_v_inv = sigma_v / v_measured**2  # Error propagation


def linear_lb(S_inv, slope, intercept):
    return slope * S_inv + intercept


# Fit linear model
popt_lb, pcov_lb = curve_fit(
    linear_lb,
    S_inv,
    v_inv,
    p0=[Km_fit / Vmax_fit, 1 / Vmax_fit],
    sigma=sigma_v_inv,
    absolute_sigma=True,
)

slope_lb, intercept_lb = popt_lb

# Extract parameters from linear fit
Vmax_lb = 1 / intercept_lb
Km_lb = slope_lb / intercept_lb

print("From Lineweaver-Burk plot:")
print(f"  V_max: {Vmax_lb:.2f} μM/min")
print(f"  K_M:   {Km_lb:.2f} μM")
print("\nNote: Direct Michaelis-Menten fit is preferred (better error weighting)")


print("\n" + "-" * 70)
print("COMPETITIVE INHIBITION ANALYSIS")
print("-" * 70)

# Simulate competitive inhibitor data
I_conc = 100.0  # μM inhibitor concentration
Ki_true = 25.0  # μM inhibition constant

v_inhibited_true = competitive_inhibition(S, Vmax_true, Km_true, I_conc, Ki_true)
noise_inh = np.random.normal(0, 0.05 * v_inhibited_true, size=len(S))
v_inhibited = v_inhibited_true + noise_inh


# Fit with inhibition model


def competitive_inh_fit(S, Vmax, Km, Ki):
    return competitive_inhibition(S, Vmax, Km, I_conc, Ki)


popt_inh, pcov_inh = curve_fit(
    competitive_inh_fit,
    S,
    v_inhibited,
    p0=[100, 50, 30],
    sigma=sigma_v,
    bounds=([0, 0, 0], [200, 500, 200]),
)

Vmax_inh, Km_inh, Ki_fit = popt_inh
perr_inh = np.sqrt(np.diag(pcov_inh))
Ki_err = perr_inh[2]

print(f"With [{I_conc:.0f} μM] inhibitor:")
print(f"  Apparent K_M: {Km_inh:.2f} μM (increased from {Km_fit:.2f} μM)")
print(f"  V_max:        {Vmax_inh:.2f} μM/min (unchanged)")
print(f"  K_i:          {Ki_fit:.2f} ± {Ki_err:.2f} μM")
print(f"  True K_i:     {Ki_true:.2f} μM")


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
    S_fine, michaelis_menten(S_fine, *popt), "g-", linewidth=2.5, label="Fitted curve"
)

# Mark K_M and V_max
ax1.axhline(
    Vmax_fit, color="blue", linestyle=":", alpha=0.5, label=f"V_max = {Vmax_fit:.1f}"
)
ax1.axhline(Vmax_fit / 2, color="orange", linestyle=":", alpha=0.5)
ax1.axvline(
    Km_fit, color="orange", linestyle=":", alpha=0.5, label=f"K_M = {Km_fit:.1f} μM"
)
ax1.plot(Km_fit, Vmax_fit / 2, "o", markersize=10, color="orange")

ax1.set_xlabel("Substrate Concentration [S] (μM)", fontsize=12)
ax1.set_ylabel("Velocity v (μM/min)", fontsize=12)
ax1.set_title("Michaelis-Menten Kinetics", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: Lineweaver-Burk plot
ax2 = plt.subplot(3, 2, 2)
ax2.errorbar(
    S_inv,
    v_inv,
    yerr=sigma_v_inv,
    fmt="o",
    capsize=4,
    markersize=8,
    label="Data (1/v vs 1/[S])",
    alpha=0.7,
)

S_inv_fine = np.linspace(0, S_inv.max(), 100)
ax2.plot(
    S_inv_fine, linear_lb(S_inv_fine, *popt_lb), "g-", linewidth=2.5, label="Linear fit"
)

# Mark intercepts
ax2.axhline(
    intercept_lb,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label=f"1/V_max = {intercept_lb:.4f}",
)
ax2.axvline(0, color="gray", linestyle="-", linewidth=0.8)

ax2.set_xlabel("1/[S] (μM⁻¹)")
ax2.set_ylabel("1/v (min/μM)")
ax2.set_title("Lineweaver-Burk Plot (Double Reciprocal)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = plt.subplot(3, 2, 3)
normalized_residuals = residuals / sigma_v
ax3.plot(S, normalized_residuals, "o", markersize=6, alpha=0.7)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Substrate Concentration [S] (μM)")
ax3.set_ylabel("Normalized Residuals (σ)")
ax3.set_title("Fit Residuals")
ax3.set_xscale("log")
ax3.grid(True, alpha=0.3)

# Plot 4: Competitive inhibition comparison
ax4 = plt.subplot(3, 2, 4)
ax4.plot(S, v_measured, "o", markersize=8, label="No inhibitor", alpha=0.7)
ax4.plot(S, v_inhibited, "s", markersize=8, label=f"[I] = {I_conc:.0f} μM", alpha=0.7)

ax4.plot(
    S_fine, michaelis_menten(S_fine, *popt), "g-", linewidth=2, label="No inhibitor fit"
)
ax4.plot(
    S_fine,
    competitive_inh_fit(S_fine, *popt_inh),
    "b-",
    linewidth=2,
    label="With inhibitor fit",
)

ax4.set_xlabel("Substrate Concentration [S] (μM)")
ax4.set_ylabel("Velocity v (μM/min)")
ax4.set_title(f"Competitive Inhibition (K_i = {Ki_fit:.1f} μM)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Eadie-Hofstee plot (alternative linearization)
ax5 = plt.subplot(3, 2, 5)
# v vs v/S
# v = -K_M * (v/S) + V_max
v_over_S = v_measured / S
ax5.plot(v_over_S, v_measured, "o", markersize=8, alpha=0.7, label="Data")

# Fit linear
v_over_S_fine = np.linspace(0, v_over_S.max(), 100)
v_eh_fit = Vmax_fit - Km_fit * v_over_S_fine
ax5.plot(
    v_over_S_fine, v_eh_fit, "g-", linewidth=2.5, label=f"Slope = -K_M = -{Km_fit:.1f}"
)

ax5.set_xlabel("v/[S] (min⁻¹)")
ax5.set_ylabel("v (μM/min)")
ax5.set_title("Eadie-Hofstee Plot")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Saturation curve (normalized)
ax6 = plt.subplot(3, 2, 6)
v_normalized = v_measured / Vmax_fit
S_normalized = S / Km_fit

ax6.semilogx(S_normalized, v_normalized, "o", markersize=8, alpha=0.7, label="Data")

S_norm_fine = np.logspace(-2, 2, 200)
v_norm_fine = S_norm_fine / (1 + S_norm_fine)
ax6.semilogx(S_norm_fine, v_norm_fine, "g-", linewidth=2.5, label="Universal curve")

ax6.axvline(1, color="orange", linestyle="--", linewidth=1.5, label="[S] = K_M")
ax6.axhline(0.5, color="blue", linestyle=":", alpha=0.5, label="v = 0.5 V_max")

ax6.set_xlabel("[S]/K_M (dimensionless)")
ax6.set_ylabel("v/V_max (dimensionless)")
ax6.set_title("Normalized Saturation Curve")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("enzyme_kinetics.png", dpi=150)
print("\n✅ Plot saved as 'enzyme_kinetics.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "enzyme_kinetics"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Enzyme kinetics parameters successfully determined:")
print(f"\n  Michaelis constant (K_M):  {Km_fit:.2f} ± {Km_err:.2f} μM")
print(f"  Maximum velocity (V_max):  {Vmax_fit:.2f} ± {Vmax_err:.2f} μM/min")
print(f"  Catalytic efficiency:      {kcat_Km:.3f} min⁻¹")
print("\nInhibition analysis:")
print(f"  Inhibition constant (K_i): {Ki_fit:.2f} ± {Ki_err:.2f} μM")
print("  Type: Competitive (K_M increased, V_max unchanged)")
print("\nPhysiological relevance:")
print(f"  At [S] = {S_physiol:.1f} μM: {saturation:.1f}% saturated")
print("\nThis example demonstrates:")
print("  ✓ Michaelis-Menten parameter fitting")
print("  ✓ Lineweaver-Burk linearization")
print("  ✓ Competitive inhibition analysis")
print("  ✓ Alternative plot types (Eadie-Hofstee, normalized)")
print("  ✓ Catalytic efficiency calculation")
print("  ✓ Physiological interpretation")
print("=" * 70)
