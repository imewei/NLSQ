"""
Converted from dose_response.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Dose-Response Curves: EC50 and IC50 Determination
#
#
# This example demonstrates fitting dose-response curves using the Hill equation
# (4-parameter logistic model) to determine EC50/IC50 values. Common in pharmacology,
# toxicology, and drug discovery.
#
# Key Concepts:
# - 4-parameter logistic (4PL) model
# - EC50/IC50 determination (half-maximal effective/inhibitory concentration)
# - Hill slope (cooperativity)
# - Dynamic range and efficacy
# - Comparison of multiple drugs/compounds
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed
np.random.seed(42)


def four_parameter_logistic(dose, bottom, top, EC50, hill_slope):
    """
    4-parameter logistic (4PL) dose-response model.

    Response = Bottom + (Top - Bottom) / (1 + (EC50/dose)^hill_slope)

    Parameters
    ----------
    dose : array_like
        Drug/compound concentration (μM, nM, etc.)
    bottom : float
        Minimum response (baseline, 0% effect)
    top : float
        Maximum response (100% effect)
    EC50 : float
        Half-maximal effective concentration
    hill_slope : float
        Hill slope (steepness of curve, cooperativity)

    Returns
    -------
    response : array_like
        Measured response (% of maximum, fluorescence, etc.)
    """
    return bottom + (top - bottom) / (1 + jnp.power(EC50 / dose, hill_slope))


def inhibition_model(dose, top, IC50, hill_slope):
    """
    Inhibition dose-response model (3-parameter).

    Response = Top / (1 + (dose/IC50)^hill_slope)

    Assumes bottom = 0 (complete inhibition).

    Parameters
    ----------
    dose : array_like
        Inhibitor concentration
    top : float
        Control response (0% inhibition)
    IC50 : float
        Half-maximal inhibitory concentration
    hill_slope : float
        Hill slope

    Returns
    -------
    response : array_like
        Response (% of control)
    """
    return top / (1 + jnp.power(dose / IC50, hill_slope))


# Drug concentrations (log-spaced: 0.01 to 1000 μM)
dose = np.logspace(-2, 3, 15)  # 15 concentration points

# True parameters for agonist (stimulation) curve
bottom_true = 5.0  # Baseline response (% of max)
top_true = 95.0  # Maximum response
EC50_true = 10.0  # μM
hill_slope_true = 1.2  # Slightly cooperative

# Generate true dose-response
response_true = four_parameter_logistic(
    dose, bottom_true, top_true, EC50_true, hill_slope_true
)

# Add measurement noise (realistic for plate reader assays)
noise = np.random.normal(0, 3.0, size=len(dose))  # ±3% noise
response_measured = response_true + noise

# Measurement uncertainties (constant CV)
sigma = 3.0 * np.ones_like(response_measured)


print("=" * 70)
print("DOSE-RESPONSE CURVES: EC50/IC50 DETERMINATION")
print("=" * 70)

# Initial parameter guess
p0 = [0, 100, 15, 1.0]  # bottom, top, EC50, hill_slope

# Parameter bounds
bounds = (
    [-20, 50, 0.01, 0.3],  # Lower bounds
    [30, 120, 1000, 5.0],  # Upper bounds
)

# Fit the model
popt, pcov = curve_fit(
    four_parameter_logistic,
    dose,
    response_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

bottom_fit, top_fit, EC50_fit, hill_slope_fit = popt
perr = np.sqrt(np.diag(pcov))
bottom_err, top_err, EC50_err, hill_slope_err = perr


# Dynamic range
dynamic_range = top_fit - bottom_fit

# EC20 and EC80 (20% and 80% effective concentrations)
# For 4PL: EC_x = EC50 * ((100-x)/x)^(1/hill_slope)
EC20 = EC50_fit * np.power((top_fit - 20) / (20 - bottom_fit), 1 / hill_slope_fit)
EC80 = EC50_fit * np.power((top_fit - 80) / (80 - bottom_fit), 1 / hill_slope_fit)


print("\nFitted Parameters:")
print(f"  Bottom (baseline):  {bottom_fit:.2f} ± {bottom_err:.2f} %")
print(f"  Top (max response): {top_fit:.2f} ± {top_err:.2f} %")
print(f"  EC50:               {EC50_fit:.3f} ± {EC50_err:.3f} μM")
print(f"  Hill slope:         {hill_slope_fit:.3f} ± {hill_slope_err:.3f}")

print("\nTrue Values:")
print(f"  Bottom:    {bottom_true:.2f} %")
print(f"  Top:       {top_true:.2f} %")
print(f"  EC50:      {EC50_true:.3f} μM")
print(f"  Hill slope: {hill_slope_true:.3f}")

print("\nErrors:")
print(
    f"  EC50: {abs(EC50_fit - EC50_true):.3f} μM "
    + f"({100 * abs(EC50_fit - EC50_true) / EC50_true:.1f}%)"
)
print(f"  Hill slope: {abs(hill_slope_fit - hill_slope_true):.3f}")

print("\nDerived Parameters:")
print(
    f"  Dynamic range:  {dynamic_range:.2f} % "
    + f"({bottom_fit:.1f} to {top_fit:.1f}%)"
)
print(f"  EC20:           {EC20:.3f} μM")
print(f"  EC80:           {EC80:.3f} μM")
print(f"  EC20/EC80 ratio: {EC20 / EC80:.3f} (wider = shallower slope)")

# Potency classification
if EC50_fit < 0.1:
    potency = "Very high potency (EC50 < 0.1 μM)"
elif EC50_fit < 1:
    potency = "High potency (0.1 < EC50 < 1 μM)"
elif EC50_fit < 10:
    potency = "Moderate potency (1 < EC50 < 10 μM)"
else:
    potency = "Low potency (EC50 > 10 μM)"

print(f"  Potency:        {potency}")

# Cooperativity
if hill_slope_fit > 1.5:
    cooperativity = "Positive cooperativity (steep, h > 1.5)"
elif hill_slope_fit > 0.7:
    cooperativity = "Non-cooperative (0.7 < h < 1.5)"
else:
    cooperativity = "Negative cooperativity (shallow, h < 0.7)"

print(f"  Cooperativity:  {cooperativity}")

# Goodness of fit
residuals = response_measured - four_parameter_logistic(dose, *popt)
chi_squared = np.sum((residuals / sigma) ** 2)
dof = len(dose) - len(popt)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.2f} %")
print(f"  χ²/dof:  {chi_squared_reduced:.2f}")


print("\n" + "-" * 70)
print("COMPARING MULTIPLE COMPOUNDS")
print("-" * 70)

# Drug B: more potent but lower efficacy
EC50_B = 2.0  # More potent (lower EC50)
top_B = 80.0  # Lower efficacy
bottom_B = 5.0
hill_slope_B = 0.9

response_B_true = four_parameter_logistic(dose, bottom_B, top_B, EC50_B, hill_slope_B)
noise_B = np.random.normal(0, 3.0, size=len(dose))
response_B = response_B_true + noise_B

# Fit Drug B
popt_B, pcov_B = curve_fit(
    four_parameter_logistic,
    dose,
    response_B,
    p0=[0, 100, 5, 1.0],
    sigma=sigma,
    bounds=bounds,
)

bottom_B_fit, top_B_fit, EC50_B_fit, hill_slope_B_fit = popt_B
dynamic_range_B = top_B_fit - bottom_B_fit

print("Drug A (reference):")
print(f"  EC50:     {EC50_fit:.2f} μM")
print(f"  Efficacy: {dynamic_range:.1f} %")
print(f"  Hill:     {hill_slope_fit:.2f}")

print("\nDrug B (comparison):")
print(f"  EC50:     {EC50_B_fit:.2f} μM (more potent)")
print(f"  Efficacy: {dynamic_range_B:.1f} % (lower efficacy)")
print(f"  Hill:     {hill_slope_B_fit:.2f}")

print(f"\nPotency ratio (A/B):  {EC50_fit / EC50_B_fit:.1f}x")
print(f"Efficacy difference:  {dynamic_range - dynamic_range_B:.1f} %")


fig = plt.figure(figsize=(16, 12))

# Plot 1: Dose-response curve (log scale)
ax1 = plt.subplot(3, 2, 1)
ax1.errorbar(
    dose,
    response_measured,
    yerr=sigma,
    fmt="o",
    capsize=4,
    markersize=8,
    alpha=0.7,
    label="Drug A data",
)

dose_fine = np.logspace(-2, 3, 200)
ax1.plot(
    dose_fine,
    four_parameter_logistic(
        dose_fine, bottom_true, top_true, EC50_true, hill_slope_true
    ),
    "r--",
    linewidth=2,
    label="True curve",
    alpha=0.7,
)
ax1.plot(
    dose_fine,
    four_parameter_logistic(dose_fine, *popt),
    "g-",
    linewidth=2.5,
    label="Fitted curve",
)

# Mark key points
ax1.axhline(
    top_fit, color="blue", linestyle=":", alpha=0.5, label=f"Top = {top_fit:.1f}%"
)
ax1.axhline(
    bottom_fit,
    color="gray",
    linestyle=":",
    alpha=0.5,
    label=f"Bottom = {bottom_fit:.1f}%",
)
ax1.axhline((top_fit + bottom_fit) / 2, color="orange", linestyle=":", alpha=0.5)
ax1.axvline(
    EC50_fit,
    color="orange",
    linestyle=":",
    alpha=0.5,
    label=f"EC50 = {EC50_fit:.2f} μM",
)

ax1.set_xscale("log")
ax1.set_xlabel("Dose (μM, log scale)", fontsize=12)
ax1.set_ylabel("Response (%)", fontsize=12)
ax1.set_title("Dose-Response Curve (4PL Model)", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3, which="both")

# Plot 2: Comparison of two drugs
ax2 = plt.subplot(3, 2, 2)
ax2.plot(
    dose,
    response_measured,
    "o",
    markersize=8,
    alpha=0.7,
    label=f"Drug A (EC50={EC50_fit:.2f}μM)",
)
ax2.plot(
    dose,
    response_B,
    "s",
    markersize=8,
    alpha=0.7,
    label=f"Drug B (EC50={EC50_B_fit:.2f}μM)",
)

ax2.plot(
    dose_fine,
    four_parameter_logistic(dose_fine, *popt),
    "g-",
    linewidth=2.5,
    label="Drug A fit",
)
ax2.plot(
    dose_fine,
    four_parameter_logistic(dose_fine, *popt_B),
    "b-",
    linewidth=2.5,
    label="Drug B fit",
)

ax2.set_xscale("log")
ax2.set_xlabel("Dose (μM, log scale)")
ax2.set_ylabel("Response (%)")
ax2.set_title("Comparing Drug Potency and Efficacy")
ax2.legend()
ax2.grid(True, alpha=0.3, which="both")

# Plot 3: Residuals vs dose
ax3 = plt.subplot(3, 2, 3)
normalized_residuals = residuals / sigma
ax3.semilogx(dose, normalized_residuals, "o", markersize=6, alpha=0.7)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Dose (μM, log scale)")
ax3.set_ylabel("Normalized Residuals (σ)")
ax3.set_title("Fit Residuals")
ax3.grid(True, alpha=0.3)

# Plot 4: Hill slope visualization
ax4 = plt.subplot(3, 2, 4)
# Show effect of different Hill slopes
hill_slopes = [0.5, 1.0, 2.0, 4.0]
for h in hill_slopes:
    resp = four_parameter_logistic(dose_fine, bottom_fit, top_fit, EC50_fit, h)
    ax4.plot(dose_fine, resp, linewidth=2, label=f"h = {h:.1f}")

ax4.axvline(EC50_fit, color="orange", linestyle="--", alpha=0.5)
ax4.set_xscale("log")
ax4.set_xlabel("Dose (μM, log scale)")
ax4.set_ylabel("Response (%)")
ax4.set_title("Effect of Hill Slope on Curve Shape")
ax4.legend()
ax4.grid(True, alpha=0.3, which="both")

# Plot 5: Normalized dose-response (universal curve)
ax5 = plt.subplot(3, 2, 5)
# Normalize: x-axis by EC50, y-axis by (response-bottom)/(top-bottom)
dose_normalized = dose / EC50_fit
response_normalized = (response_measured - bottom_fit) / (top_fit - bottom_fit)

ax5.semilogx(
    dose_normalized,
    response_normalized,
    "o",
    markersize=8,
    alpha=0.7,
    label="Normalized data",
)

dose_norm_fine = np.logspace(-3, 3, 200)
# Normalized 4PL: Response = 1 / (1 + (1/x)^h)
response_norm_fine = 1 / (1 + (1 / dose_norm_fine) ** hill_slope_fit)
ax5.semilogx(
    dose_norm_fine,
    response_norm_fine,
    "g-",
    linewidth=2.5,
    label=f"Universal curve (h={hill_slope_fit:.2f})",
)

ax5.axvline(1, color="orange", linestyle="--", linewidth=2, label="Dose = EC50")
ax5.axhline(0.5, color="blue", linestyle=":", alpha=0.5, label="50% response")

ax5.set_xlabel("Normalized Dose (Dose/EC50)")
ax5.set_ylabel("Normalized Response")
ax5.set_title("Normalized Dose-Response Curve")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Concentration-effect table visualization
ax6 = plt.subplot(3, 2, 6)
ax6.axis("off")

# Create table of key concentrations and effects
conc_table = [
    ["Concentration", "Response", "Description"],
    ["─" * 15, "─" * 10, "─" * 25],
    [f"{EC20:.2f} μM", "20%", "EC20 (low effect)"],
    [f"{EC50_fit:.2f} μM", "50%", "EC50 (half-maximal)"],
    [f"{EC80:.2f} μM", "80%", "EC80 (high effect)"],
    ["", "", ""],
    ["Parameter", "Value", "Interpretation"],
    ["─" * 15, "─" * 10, "─" * 25],
    ["Dynamic range", f"{dynamic_range:.1f}%", f"{bottom_fit:.1f} → {top_fit:.1f}%"],
    ["Hill slope", f"{hill_slope_fit:.2f}", cooperativity.split("(")[0].strip()],
    ["Potency", f"{EC50_fit:.2f} μM", potency.split("(")[0].strip()],
    ["", "", ""],
    ["Quality", "Metric", ""],
    ["─" * 15, "─" * 10, "─" * 25],
    ["RMSE", f"{rmse:.2f}%", ""],
    ["χ²/dof", f"{chi_squared_reduced:.2f}", ""],
]

table_text = "\n".join(["  ".join(row) for row in conc_table])
ax6.text(
    0.1,
    0.9,
    table_text,
    fontsize=10,
    verticalalignment="top",
    fontfamily="monospace",
    transform=ax6.transAxes,
)
ax6.set_title("Summary Table", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("dose_response.png", dpi=150)
print("\n✅ Plot saved as 'dose_response.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "dose_response"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Dose-response analysis complete:")
print("\n  Drug A:")
print(f"    EC50:      {EC50_fit:.3f} ± {EC50_err:.3f} μM")
print(f"    Efficacy:  {dynamic_range:.1f}% ({bottom_fit:.1f} to {top_fit:.1f}%)")
print(f"    Hill slope: {hill_slope_fit:.2f} ({cooperativity.split('(')[0].strip()})")
print(f"    Potency:   {potency.split('(')[0].strip()}")
print("\n  Drug B (comparison):")
print(f"    EC50:      {EC50_B_fit:.2f} μM ({EC50_fit / EC50_B_fit:.1f}x less potent)")
print(
    f"    Efficacy:  {dynamic_range_B:.1f}% ({dynamic_range - dynamic_range_B:+.1f}%)"
)
print("\n  Key concentrations:")
print(f"    EC20: {EC20:.2f} μM")
print(f"    EC50: {EC50_fit:.2f} μM")
print(f"    EC80: {EC80:.2f} μM")
print("\nThis example demonstrates:")
print("  ✓ 4-parameter logistic (4PL) dose-response fitting")
print("  ✓ EC50/IC50 determination with uncertainties")
print("  ✓ Hill slope and cooperativity analysis")
print("  ✓ Dynamic range and efficacy quantification")
print("  ✓ Comparison of multiple compounds")
print("  ✓ Normalized dose-response curves")
print("=" * 70)
