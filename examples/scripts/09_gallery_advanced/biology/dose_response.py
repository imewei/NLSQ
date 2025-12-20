"""
Advanced Dose-Response Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting dose-response curves using the Hill equation
(4-parameter logistic model) with NLSQ's advanced fit() API and global optimization
capabilities for robust EC50/IC50 determination.

Compared to 04_gallery/biology/dose_response.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- 4-parameter logistic (4PL) model
- EC50/IC50 determination (half-maximal effective/inhibitory concentration)
- Hill slope (cooperativity)
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

# Set random seed
np.random.seed(42)


def four_parameter_logistic(dose, bottom, top, EC50, hill_slope):
    """
    4-parameter logistic (4PL) dose-response model.

    Response = Bottom + (Top - Bottom) / (1 + (EC50/dose)^hill_slope)

    Parameters
    ----------
    dose : array_like
        Drug/compound concentration
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
    """
    return top / (1 + jnp.power(dose / IC50, hill_slope))


# Drug concentrations (log-spaced: 0.01 to 1000 uM)
dose = np.logspace(-2, 3, 10 if QUICK else 15)  # concentration points

# True parameters for agonist (stimulation) curve
bottom_true = 5.0  # Baseline response (% of max)
top_true = 95.0  # Maximum response
EC50_true = 10.0  # uM
hill_slope_true = 1.2  # Slightly cooperative

# Generate true dose-response
response_true = four_parameter_logistic(
    dose, bottom_true, top_true, EC50_true, hill_slope_true
)

# Add measurement noise (realistic for plate reader assays)
noise = np.random.normal(0, 3.0, size=len(dose))  # +/- 3% noise
response_measured = response_true + noise

# Measurement uncertainties (constant CV)
sigma = 3.0 * np.ones_like(response_measured)


print("=" * 70)
print("DOSE-RESPONSE CURVES: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# Initial parameter guess
p0 = [0, 100, 15, 1.0]  # bottom, top, EC50, hill_slope

# Parameter bounds
bounds = (
    [-20, 50, 0.01, 0.3],  # Lower bounds
    [30, 120, 1000, 5.0],  # Upper bounds
)

# =============================================================================
# Method 1: Using fit() with 'robust' preset (recommended for most cases)
# =============================================================================
print("\n" + "-" * 70)
print("Method 1: fit() with 'robust' preset")
print("-" * 70)

popt_robust, pcov_robust = fit(
    four_parameter_logistic,
    dose,
    response_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="robust",  # Multi-start with 5 starts for robustness
)

bottom_fit, top_fit, EC50_fit, hill_slope_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
bottom_err, top_err, EC50_err, hill_slope_err = perr

print(f"  EC50 = {EC50_fit:.3f} +/- {EC50_err:.3f} uM (true: {EC50_true})")
print(f"  Hill slope = {hill_slope_fit:.3f} +/- {hill_slope_err:.3f}")

# =============================================================================
# Method 2: Using fit() with 'global' preset (thorough global search)
# =============================================================================
global_starts = 6 if QUICK else 20
print("\n" + "-" * 70)
print(f"Method 2: fit() with 'global' preset ({global_starts} starts)")
print("-" * 70)

popt_global, pcov_global = fit(
    four_parameter_logistic,
    dose,
    response_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="global",  # Thorough global search with 20 starts
    n_starts=global_starts,
)

bottom_g, top_g, EC50_g, hill_slope_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  EC50 = {EC50_g:.3f} +/- {perr_g[2]:.3f} uM")
print(f"  Hill slope = {hill_slope_g:.3f} +/- {perr_g[3]:.3f}")

# =============================================================================
# Method 3: Using GlobalOptimizationConfig directly for full control
# =============================================================================
print("\n" + "-" * 70)
print("Method 3: GlobalOptimizationConfig with custom settings")
print("-" * 70)

# Create a custom global optimization configuration
custom_starts = 6 if QUICK else 15
global_config = GlobalOptimizationConfig(
    n_starts=custom_starts,  # starting points
    sampler="lhs",  # Latin Hypercube Sampling for good coverage
    center_on_p0=True,  # Center samples around initial guess
    scale_factor=1.0,  # Exploration range scale
    elimination_rounds=3,  # Progressive elimination rounds
    elimination_fraction=0.5,  # Eliminate 50% worst per round
)

# Use fit() with explicit multi-start parameters
popt_custom, pcov_custom = fit(
    four_parameter_logistic,
    dose,
    response_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    multistart=True,
    n_starts=custom_starts,
    sampler="lhs",
    center_on_p0=True,
    scale_factor=1.0,
)

bottom_c, top_c, EC50_c, hill_slope_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))

print(f"  EC50 = {EC50_c:.3f} +/- {perr_c[2]:.3f} uM")
print(f"  Hill slope = {hill_slope_c:.3f} +/- {perr_c[3]:.3f}")


# Use the robust preset results for analysis
bottom_fit, top_fit, EC50_fit, hill_slope_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
bottom_err, top_err, EC50_err, hill_slope_err = perr

# Dynamic range
dynamic_range = top_fit - bottom_fit

# EC20 and EC80 (20% and 80% effective concentrations)
EC20 = EC50_fit * np.power((top_fit - 20) / (20 - bottom_fit), 1 / hill_slope_fit)
EC80 = EC50_fit * np.power((top_fit - 80) / (80 - bottom_fit), 1 / hill_slope_fit)


print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  Bottom (baseline):  {bottom_fit:.2f} +/- {bottom_err:.2f} %")
print(f"  Top (max response): {top_fit:.2f} +/- {top_err:.2f} %")
print(f"  EC50:               {EC50_fit:.3f} +/- {EC50_err:.3f} uM")
print(f"  Hill slope:         {hill_slope_fit:.3f} +/- {hill_slope_err:.3f}")

print("\nTrue Values:")
print(f"  Bottom:    {bottom_true:.2f} %")
print(f"  Top:       {top_true:.2f} %")
print(f"  EC50:      {EC50_true:.3f} uM")
print(f"  Hill slope: {hill_slope_true:.3f}")

print("\nDerived Parameters:")
print(f"  Dynamic range:  {dynamic_range:.2f} % ({bottom_fit:.1f} to {top_fit:.1f}%)")
print(f"  EC20:           {EC20:.3f} uM")
print(f"  EC80:           {EC80:.3f} uM")

# Potency classification
if EC50_fit < 0.1:
    potency = "Very high potency (EC50 < 0.1 uM)"
elif EC50_fit < 1:
    potency = "High potency (0.1 < EC50 < 1 uM)"
elif EC50_fit < 10:
    potency = "Moderate potency (1 < EC50 < 10 uM)"
else:
    potency = "Low potency (EC50 > 10 uM)"

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
residuals = response_measured - four_parameter_logistic(dose, *popt_robust)
chi_squared = np.sum((residuals / sigma) ** 2)
dof = len(dose) - len(popt_robust)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.2f} %")
print(f"  chi^2/dof:  {chi_squared_reduced:.2f}")


# =============================================================================
# Comparing Multiple Compounds
# =============================================================================
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

if QUICK:
    print("⏩ Quick mode: skipping compound B comparison and heavy plotting.")
    sys.exit(0)

# Fit Drug B with robust preset
popt_B, pcov_B = fit(
    four_parameter_logistic,
    dose,
    response_B,
    p0=[0, 100, 5, 1.0],
    sigma=sigma,
    bounds=bounds,
    preset="robust",
)

bottom_B_fit, top_B_fit, EC50_B_fit, hill_slope_B_fit = popt_B
dynamic_range_B = top_B_fit - bottom_B_fit

print("Drug A (reference):")
print(f"  EC50:     {EC50_fit:.2f} uM")
print(f"  Efficacy: {dynamic_range:.1f} %")
print(f"  Hill:     {hill_slope_fit:.2f}")

print("\nDrug B (comparison):")
print(f"  EC50:     {EC50_B_fit:.2f} uM (more potent)")
print(f"  Efficacy: {dynamic_range_B:.1f} % (lower efficacy)")
print(f"  Hill:     {hill_slope_B_fit:.2f}")

print(f"\nPotency ratio (A/B):  {EC50_fit / EC50_B_fit:.1f}x")
print(f"Efficacy difference:  {dynamic_range - dynamic_range_B:.1f} %")


# =============================================================================
# Visualization
# =============================================================================
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
    four_parameter_logistic(dose_fine, *popt_robust),
    "g-",
    linewidth=2.5,
    label="Fitted curve (robust)",
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
    label=f"EC50 = {EC50_fit:.2f} uM",
)

ax1.set_xscale("log")
ax1.set_xlabel("Dose (uM, log scale)", fontsize=12)
ax1.set_ylabel("Response (%)", fontsize=12)
ax1.set_title(
    "Dose-Response Curve (4PL Model) - fit() API", fontsize=14, fontweight="bold"
)
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
    label=f"Drug A (EC50={EC50_fit:.2f}uM)",
)
ax2.plot(
    dose,
    response_B,
    "s",
    markersize=8,
    alpha=0.7,
    label=f"Drug B (EC50={EC50_B_fit:.2f}uM)",
)

ax2.plot(
    dose_fine,
    four_parameter_logistic(dose_fine, *popt_robust),
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
ax2.set_xlabel("Dose (uM, log scale)")
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
ax3.set_xlabel("Dose (uM, log scale)")
ax3.set_ylabel("Normalized Residuals (sigma)")
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
ax4.set_xlabel("Dose (uM, log scale)")
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

# Plot 6: API comparison summary
ax6 = plt.subplot(3, 2, 6)
ax6.axis("off")

# Create summary table
api_table = [
    ["API Method", "Preset/Config", "EC50 (uM)", "Hill"],
    ["-" * 20, "-" * 15, "-" * 10, "-" * 8],
    ["fit()", "'robust'", f"{EC50_fit:.3f}", f"{hill_slope_fit:.3f}"],
    ["fit()", "'global'", f"{EC50_g:.3f}", f"{hill_slope_g:.3f}"],
    ["fit()", "custom 15-start", f"{EC50_c:.3f}", f"{hill_slope_c:.3f}"],
    ["", "", "", ""],
    ["True values", "-", f"{EC50_true:.3f}", f"{hill_slope_true:.3f}"],
    ["", "", "", ""],
    ["Advantages of fit() API:", "", "", ""],
    ["-" * 40, "", "", ""],
    ["  - Auto workflow selection", "", "", ""],
    ["  - Built-in multi-start", "", "", ""],
    ["  - Preset configurations", "", "", ""],
    ["  - GlobalOptimizationConfig", "", "", ""],
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
fig_dir = Path(__file__).parent / "figures" / "dose_response"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Dose-response analysis complete with fit() API:")
print("\n  Drug A:")
print(f"    EC50:      {EC50_fit:.3f} +/- {EC50_err:.3f} uM")
print(f"    Efficacy:  {dynamic_range:.1f}% ({bottom_fit:.1f} to {top_fit:.1f}%)")
print(f"    Hill slope: {hill_slope_fit:.2f} ({cooperativity.split('(')[0].strip()})")
print(f"    Potency:   {potency.split('(')[0].strip()}")

print("\n  API Methods Used:")
print("    - fit() with preset='robust' (5 multi-starts)")
print("    - fit() with preset='global' (20 multi-starts)")
print("    - fit() with custom multi-start settings")

print("\nAdvantages over curve_fit():")
print("  - Automatic multi-start optimization avoids local minima")
print("  - Preset configurations for common use cases")
print("  - GlobalOptimizationConfig for fine-grained control")
print("  - Better parameter recovery with noisy data")
print("=" * 70)
