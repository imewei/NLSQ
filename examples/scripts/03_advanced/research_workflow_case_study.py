"""
Converted from research_workflow_case_study.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # Research Workflow Case Study: Raman Spectroscopy Peak Analysis
#
# **Level**: Advanced
# **Time**: 40-50 minutes
# **Prerequisites**: NLSQ Quickstart, Advanced Features Demo
#
# ## Overview
#
# This tutorial demonstrates a **complete research workflow** from raw experimental data to publication-ready results. We analyze Raman spectroscopy data from graphene oxide characterization, following best practices for scientific curve fitting.
#
# ### What You'll Learn
#
# 1. **Data Preprocessing**: Baseline subtraction, noise filtering, quality checks
# 2. **Multi-Peak Fitting**: Lorentzian/Voigt profiles for overlapping peaks
# 3. **Uncertainty Quantification**: Confidence intervals, error propagation, bootstrap resampling
# 4. **Publication Plots**: High-quality matplotlib figures with proper styling
# 5. **Statistical Analysis**: Goodness-of-fit metrics, residual analysis
# 6. **Results Reporting**: Tables, uncertainties, physical interpretation
#
# ### Scientific Context
#
# Raman spectroscopy is widely used to characterize carbon materials. Graphene oxide exhibits two characteristic peaks:
# - **D-band** (~1350 cm⁻¹): Disorder-induced peak
# - **G-band** (~1580 cm⁻¹): Graphitic carbon peak
#
# The D/G intensity ratio quantifies the degree of disorder, crucial for materials characterization.
#
# ### Reference
#
# Based on methodology from: Ferrari & Robertson, *Phys. Rev. B* **61**, 14095 (2000)
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import os
import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from nlsq import CurveFit, __version__

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
if QUICK:
    print(
        "Quick mode: skipping full research workflow demo "
        "(unset NLSQ_EXAMPLES_QUICK for full run)."
    )
    sys.exit(0)

# Publication-quality matplotlib settings
rcParams["figure.figsize"] = (10, 6)
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["axes.titlesize"] = 13
rcParams["xtick.labelsize"] = 10
rcParams["ytick.labelsize"] = 10
rcParams["legend.fontsize"] = 10
rcParams["lines.linewidth"] = 1.5
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

warnings.filterwarnings("ignore", category=RuntimeWarning)

print("✓ Imports successful")
print(f"  NLSQ version: {__version__}")


# ======================================================================
# ## Part 1: Data Generation and Preprocessing
#
# We'll simulate realistic Raman spectroscopy data with noise, then apply standard preprocessing steps.
# ======================================================================


# Generate synthetic Raman spectroscopy data

# Experimental parameters (realistic values)
wavenumber = np.linspace(1000, 2000, 500)  # Raman shift in cm^-1

# True parameters for two Lorentzian peaks
# D-band: position, amplitude, width (FWHM)
d_band_true = {"pos": 1350.0, "amp": 800.0, "width": 50.0}

# G-band: position, amplitude, width
g_band_true = {"pos": 1580.0, "amp": 1200.0, "width": 40.0}

# Baseline (polynomial background)
baseline_true = 100.0 + 0.05 * wavenumber


def lorentzian(x, pos, amp, width):
    """Lorentzian (Cauchy) peak profile.

    Parameters
    ----------
    x : array_like
        Independent variable (wavenumber)
    pos : float
        Peak position (center)
    amp : float
        Peak amplitude (height)
    width : float
        Full width at half maximum (FWHM)

    Returns
    -------
    y : array_like
        Lorentzian profile
    """
    gamma = width / 2.0  # Half-width at half-maximum
    return amp * (gamma**2) / ((x - pos) ** 2 + gamma**2)


# Generate clean signal
d_band_signal = lorentzian(
    wavenumber, d_band_true["pos"], d_band_true["amp"], d_band_true["width"]
)
g_band_signal = lorentzian(
    wavenumber, g_band_true["pos"], g_band_true["amp"], g_band_true["width"]
)
clean_signal = d_band_signal + g_band_signal + baseline_true

# Add realistic noise (Poisson + Gaussian)
np.random.seed(42)  # Reproducibility
noise_level = 30.0
noise = np.random.normal(0, noise_level, len(wavenumber))
intensity_measured = clean_signal + noise

# Simulate uncertainty (shot noise scales with sqrt(signal))
sigma_measured = np.sqrt(np.abs(intensity_measured)) + noise_level / 10

print(f"✓ Generated {len(wavenumber)} data points")
print(f"  Wavenumber range: {wavenumber.min():.0f} - {wavenumber.max():.0f} cm⁻¹")
print(f"  Signal-to-noise ratio: {clean_signal.max() / noise_level:.1f}")
print(f"  True D/G ratio: {d_band_true['amp'] / g_band_true['amp']:.3f}")


# Preprocessing: baseline subtraction and quality checks

# Simple linear baseline estimation from edge regions
edge_points = 50
left_baseline = np.mean(intensity_measured[:edge_points])
right_baseline = np.mean(intensity_measured[-edge_points:])
estimated_baseline = np.linspace(left_baseline, right_baseline, len(wavenumber))

# Subtract baseline
intensity_corrected = intensity_measured - estimated_baseline

# Quality checks
print("Data Quality Checks:")
print(f"  Max intensity: {intensity_corrected.max():.1f} counts")
print(f"  Min intensity: {intensity_corrected.min():.1f} counts")
print(
    f"  Negative points: {np.sum(intensity_corrected < 0)} / {len(intensity_corrected)}"
)

# Clip small negative values (common in baseline-corrected spectra)
intensity_corrected = np.maximum(intensity_corrected, 1.0)

print("\n✓ Baseline correction applied")


# Visualize raw and preprocessed data

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Raw data
ax1.plot(wavenumber, intensity_measured, "o", ms=2, alpha=0.5, label="Raw data")
ax1.plot(wavenumber, estimated_baseline, "r--", lw=2, label="Estimated baseline")
ax1.set_xlabel("Raman Shift (cm⁻¹)")
ax1.set_ylabel("Intensity (counts)")
ax1.set_title("(a) Raw Raman Spectrum")
ax1.legend()

# Baseline-corrected data
ax2.plot(wavenumber, intensity_corrected, "o", ms=2, alpha=0.5, label="Corrected data")
ax2.axhline(0, color="k", ls=":", lw=1)
ax2.set_xlabel("Raman Shift (cm⁻¹)")
ax2.set_ylabel("Intensity (counts)")
ax2.set_title("(b) Baseline-Corrected Spectrum")
ax2.legend()

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "research_workflow_case_study"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Data preprocessing complete")


# ======================================================================
# ## Part 2: Multi-Peak Fitting with NLSQ
#
# Fit the D and G bands simultaneously using a two-Lorentzian model.
# ======================================================================


# Define multi-peak model for fitting


def lorentzian_jax(x, pos, amp, width):
    """JAX-compatible Lorentzian profile."""
    gamma = width / 2.0
    return amp * (gamma**2) / ((x - pos) ** 2 + gamma**2)


def two_peak_model(x, d_pos, d_amp, d_width, g_pos, g_amp, g_width):
    """Model for two overlapping Lorentzian peaks.

    Parameters
    ----------
    x : array_like
        Wavenumber values
    d_pos, d_amp, d_width : float
        D-band position, amplitude, and FWHM
    g_pos, g_amp, g_width : float
        G-band position, amplitude, and FWHM

    Returns
    -------
    y : array_like
        Combined spectrum
    """
    d_band = lorentzian_jax(x, d_pos, d_amp, d_width)
    g_band = lorentzian_jax(x, g_pos, g_amp, g_width)
    return d_band + g_band


print("✓ Model defined: 6 parameters (2 peaks × 3 parameters)")


# Perform curve fitting with NLSQ

# Initial parameter guess (from visual inspection)
p0 = [
    1340.0,
    750.0,
    60.0,  # D-band: pos, amp, width
    1590.0,
    1100.0,
    50.0,  # G-band: pos, amp, width
]

# Parameter bounds (physical constraints)
bounds = (
    [1300, 100, 20, 1550, 100, 20],  # Lower bounds
    [1400, 2000, 100, 1650, 2000, 100],  # Upper bounds
)

# Create CurveFit instance with diagnostic output
cf = CurveFit()

# Fit with uncertainty estimation
x_fit = jnp.array(wavenumber)
y_fit = jnp.array(intensity_corrected)
sigma_fit = np.array(sigma_measured)  # sigma must be numpy array

popt, pcov = cf.curve_fit(
    two_peak_model,
    x_fit,
    y_fit,
    p0=p0,
    sigma=sigma_fit,
    bounds=bounds,
    absolute_sigma=True,
    full_output=False,
)

# Extract fitted parameters
d_pos_fit, d_amp_fit, d_width_fit = popt[0], popt[1], popt[2]
g_pos_fit, g_amp_fit, g_width_fit = popt[3], popt[4], popt[5]

# Calculate uncertainties (1-sigma)
perr = np.sqrt(np.diag(pcov))
d_pos_err, d_amp_err, d_width_err = perr[0], perr[1], perr[2]
g_pos_err, g_amp_err, g_width_err = perr[3], perr[4], perr[5]

print("✓ Fitting complete\n")
print("Fitted Parameters:")
print("D-band (Disorder):")
print(f"  Position: {d_pos_fit:.1f} ± {d_pos_err:.1f} cm⁻¹")
print(f"  Amplitude: {d_amp_fit:.1f} ± {d_amp_err:.1f} counts")
print(f"  FWHM: {d_width_fit:.1f} ± {d_width_err:.1f} cm⁻¹")
print("\nG-band (Graphitic):")
print(f"  Position: {g_pos_fit:.1f} ± {g_pos_err:.1f} cm⁻¹")
print(f"  Amplitude: {g_amp_fit:.1f} ± {g_amp_err:.1f} counts")
print(f"  FWHM: {g_width_fit:.1f} ± {g_width_err:.1f} cm⁻¹")


# ======================================================================
# ## Part 3: Uncertainty Quantification and Error Propagation
#
# Calculate derived quantities (D/G ratio) with proper error propagation.
# ======================================================================


# Error propagation for D/G intensity ratio

# D/G ratio (disorder quantification)
dg_ratio = d_amp_fit / g_amp_fit

# Error propagation using partial derivatives
# For R = D/G, δR = R * sqrt((δD/D)^2 + (δG/G)^2)
dg_ratio_err = dg_ratio * np.sqrt(
    (d_amp_err / d_amp_fit) ** 2 + (g_amp_err / g_amp_fit) ** 2
)

print("Derived Quantity:")
print(f"  D/G Intensity Ratio: {dg_ratio:.3f} ± {dg_ratio_err:.3f}")
print(f"  True D/G ratio: {d_band_true['amp'] / g_band_true['amp']:.3f}")
print(
    f"  Relative error: {abs(dg_ratio - d_band_true['amp'] / g_band_true['amp']) / (d_band_true['amp'] / g_band_true['amp']) * 100:.1f}%"
)

# Physical interpretation
print("\nPhysical Interpretation:")
if dg_ratio < 0.5:
    print("  → Low disorder: High-quality graphene")
elif dg_ratio < 1.0:
    print("  → Moderate disorder: Partially reduced graphene oxide")
else:
    print("  → High disorder: Heavily oxidized material")


# Bootstrap resampling for robust uncertainty estimation

n_bootstrap = 100  # Number of bootstrap samples
bootstrap_ratios = []

np.random.seed(123)
for i in range(n_bootstrap):
    # Resample data with replacement
    indices = np.random.choice(len(wavenumber), size=len(wavenumber), replace=True)
    x_boot = x_fit[indices]
    y_boot = y_fit[indices]
    sigma_boot = np.array(sigma_fit[indices])  # sigma must be numpy array

    try:
        # Fit bootstrapped sample
        popt_boot, _ = cf.curve_fit(
            two_peak_model,
            x_boot,
            y_boot,
            p0=p0,
            sigma=sigma_boot,
            bounds=bounds,
            absolute_sigma=True,
        )
        # Calculate D/G ratio for this sample
        ratio_boot = popt_boot[1] / popt_boot[4]
        bootstrap_ratios.append(ratio_boot)
    except Exception:
        continue  # Skip failed fits

bootstrap_ratios = np.array(bootstrap_ratios)

# Bootstrap statistics
dg_ratio_boot_mean = np.mean(bootstrap_ratios)
dg_ratio_boot_std = np.std(bootstrap_ratios)
dg_ratio_boot_ci = np.percentile(bootstrap_ratios, [2.5, 97.5])  # 95% CI

print(f"Bootstrap Results ({len(bootstrap_ratios)} successful samples):")
print(f"  Mean D/G ratio: {dg_ratio_boot_mean:.3f} ± {dg_ratio_boot_std:.3f}")
print(
    f"  95% Confidence Interval: [{dg_ratio_boot_ci[0]:.3f}, {dg_ratio_boot_ci[1]:.3f}]"
)
print(
    f"\n  Agreement with propagated error: {abs(dg_ratio_boot_std - dg_ratio_err) / dg_ratio_err * 100:.1f}%"
)


# ======================================================================
# ## Part 4: Statistical Analysis and Goodness-of-Fit
# ======================================================================


# Calculate goodness-of-fit metrics

# Predicted values
y_pred = two_peak_model(x_fit, *popt)

# Residuals
residuals = y_fit - y_pred
weighted_residuals = residuals / sigma_fit

# Chi-squared statistic
chi_squared = np.sum(weighted_residuals**2)
n_data = len(y_fit)
n_params = len(popt)
dof = n_data - n_params  # Degrees of freedom
reduced_chi_squared = chi_squared / dof

# R-squared
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Root mean square error
rmse = np.sqrt(np.mean(residuals**2))

print("Goodness-of-Fit Statistics:")
print(f"  χ² = {chi_squared:.1f}")
print(f"  Reduced χ² = {reduced_chi_squared:.2f} (expect ~1.0 for good fit)")
print(f"  R² = {r_squared:.4f}")
print(f"  RMSE = {rmse:.2f} counts")
print(f"  Degrees of freedom: {dof}")

# Interpretation
if 0.8 < reduced_chi_squared < 1.2:
    print("\n  ✓ Excellent fit: Model captures data well")
elif reduced_chi_squared > 1.5:
    print("\n  ⚠ Poor fit: Consider more complex model or check uncertainties")
else:
    print("\n  ⚠ Overfit or underestimated uncertainties")


# Residual analysis for model validation

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Residual plot
ax1.axhline(0, color="k", ls="--", lw=1)
ax1.fill_between(
    wavenumber,
    -3 * sigma_measured,
    3 * sigma_measured,
    alpha=0.2,
    color="gray",
    label="±3σ expected",
)
ax1.plot(wavenumber, residuals, "o", ms=3, alpha=0.6, label="Residuals")
ax1.set_xlabel("Raman Shift (cm⁻¹)")
ax1.set_ylabel("Residual (counts)")
ax1.set_title("(a) Residual Plot")
ax1.legend()

# Histogram of weighted residuals
ax2.hist(weighted_residuals, bins=30, alpha=0.7, edgecolor="black")
ax2.axvline(0, color="r", ls="--", lw=2, label="Mean")

# Overlay normal distribution (expected for good fit)
x_norm = np.linspace(-4, 4, 100)
y_norm = (
    len(weighted_residuals)
    * (x_norm[1] - x_norm[0])
    * (1 / np.sqrt(2 * np.pi))
    * np.exp(-(x_norm**2) / 2)
)
ax2.plot(x_norm, y_norm, "r-", lw=2, label="N(0,1)")

ax2.set_xlabel("Weighted Residual (σ)")
ax2.set_ylabel("Frequency")
ax2.set_title("(b) Residual Distribution")
ax2.legend()

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "research_workflow_case_study"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Residual analysis complete")
print(f"  Residual mean: {np.mean(weighted_residuals):.3f} (expect 0 for unbiased fit)")
print(
    f"  Residual std: {np.std(weighted_residuals):.3f} (expect 1 for correct uncertainties)"
)


# ======================================================================
# ## Part 5: Publication-Quality Visualization
# ======================================================================


# Create publication-ready figure with all components

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Main plot: Data + fit + components
ax_main = fig.add_subplot(gs[0, :])

# Plot data with error bars (subsample for clarity)
step = 10
ax_main.errorbar(
    wavenumber[::step],
    intensity_corrected[::step],
    yerr=sigma_measured[::step],
    fmt="o",
    ms=4,
    alpha=0.4,
    elinewidth=1,
    capsize=2,
    label="Experimental data",
    color="steelblue",
)

# Plot total fit
ax_main.plot(wavenumber, y_pred, "r-", lw=2.5, label="Total fit", zorder=10, alpha=0.9)

# Plot individual components
d_component = lorentzian_jax(x_fit, d_pos_fit, d_amp_fit, d_width_fit)
g_component = lorentzian_jax(x_fit, g_pos_fit, g_amp_fit, g_width_fit)

ax_main.fill_between(
    wavenumber, 0, d_component, alpha=0.3, color="orange", label="D-band"
)
ax_main.fill_between(
    wavenumber, 0, g_component, alpha=0.3, color="green", label="G-band"
)

# Annotate peak positions
ax_main.annotate(
    f"D\n{d_pos_fit:.0f} cm⁻¹",
    xy=(d_pos_fit, d_amp_fit),
    xytext=(d_pos_fit - 100, d_amp_fit + 200),
    arrowprops={"arrowstyle": "->", "lw": 1.5},
    fontsize=11,
    ha="center",
)
ax_main.annotate(
    f"G\n{g_pos_fit:.0f} cm⁻¹",
    xy=(g_pos_fit, g_amp_fit),
    xytext=(g_pos_fit + 100, g_amp_fit + 200),
    arrowprops={"arrowstyle": "->", "lw": 1.5},
    fontsize=11,
    ha="center",
)

ax_main.set_xlabel("Raman Shift (cm⁻¹)", fontsize=12)
ax_main.set_ylabel("Intensity (counts)", fontsize=12)
ax_main.set_title(
    "Raman Spectrum of Graphene Oxide: D and G Band Analysis",
    fontsize=14,
    weight="bold",
)
ax_main.legend(loc="upper right", frameon=True, shadow=True)
ax_main.set_xlim(1000, 2000)

# Bottom left: Parameter table
ax_table = fig.add_subplot(gs[1, 0])
ax_table.axis("off")

table_data = [
    ["Parameter", "D-band", "G-band"],
    [
        "Position (cm⁻¹)",
        f"{d_pos_fit:.1f} ± {d_pos_err:.1f}",
        f"{g_pos_fit:.1f} ± {g_pos_err:.1f}",
    ],
    [
        "Amplitude",
        f"{d_amp_fit:.0f} ± {d_amp_err:.0f}",
        f"{g_amp_fit:.0f} ± {g_amp_err:.0f}",
    ],
    [
        "FWHM (cm⁻¹)",
        f"{d_width_fit:.1f} ± {d_width_err:.1f}",
        f"{g_width_fit:.1f} ± {g_width_err:.1f}",
    ],
    ["", "", ""],
    ["D/G Ratio", f"{dg_ratio:.3f} ± {dg_ratio_err:.3f}", ""],
    ["χ²ᵣ", f"{reduced_chi_squared:.2f}", ""],
    ["R²", f"{r_squared:.4f}", ""],
]

table = ax_table.table(
    cellText=table_data,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
    colWidths=[0.4, 0.3, 0.3],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor("#4CAF50")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Bottom right: Bootstrap distribution
ax_boot = fig.add_subplot(gs[1, 1])
ax_boot.hist(bootstrap_ratios, bins=20, alpha=0.7, edgecolor="black", color="steelblue")
ax_boot.axvline(dg_ratio, color="r", ls="--", lw=2, label=f"Fitted: {dg_ratio:.3f}")
ax_boot.axvline(dg_ratio_boot_ci[0], color="orange", ls=":", lw=1.5, label="95% CI")
ax_boot.axvline(dg_ratio_boot_ci[1], color="orange", ls=":", lw=1.5)
ax_boot.set_xlabel("D/G Intensity Ratio")
ax_boot.set_ylabel("Frequency")
ax_boot.set_title("Bootstrap Distribution")
ax_boot.legend()

plt.suptitle(
    "Figure 1: Complete Raman Spectroscopy Analysis",
    fontsize=15,
    weight="bold",
    y=0.98,
)

# Save figure (uncomment to save)
# plt.savefig('raman_analysis_figure1.png', dpi=300, bbox_inches='tight')
# plt.savefig('raman_analysis_figure1.pdf', bbox_inches='tight')

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "research_workflow_case_study"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Publication figure generated")
print("  Recommendation: Save as PDF for LaTeX, PNG (300 dpi) for presentations")


# ======================================================================
# ## Summary and Best Practices
#
# ### Complete Research Workflow
#
# 1. **Data Preprocessing**
#    - Baseline correction (polynomial, edge-fitting, or automated)
#    - Noise filtering (optional: Savitzky-Golay)
#    - Quality checks (range, negative values, outliers)
#
# 2. **Model Selection**
#    - Choose appropriate peak shapes (Lorentzian, Gaussian, Voigt)
#    - Consider physical constraints (bounds)
#    - Start with simple models, add complexity if needed
#
# 3. **Fitting Strategy**
#    - Good initial guesses (visual inspection, peak finding)
#    - Use measurement uncertainties (`sigma` parameter)
#    - Apply bounds for physical validity
#    - Check convergence and diagnostics
#
# 4. **Uncertainty Quantification**
#    - Parameter uncertainties from covariance matrix
#    - Error propagation for derived quantities
#    - Bootstrap resampling for robust estimates
#    - Report 95% confidence intervals
#
# 5. **Validation**
#    - Goodness-of-fit metrics (χ²ᵣ, R², RMSE)
#    - Residual analysis (pattern detection, normality)
#    - Physical interpretation of parameters
#    - Sensitivity analysis (optional)
#
# 6. **Reporting**
#    - Publication-quality figures
#    - Parameter tables with uncertainties
#    - Statistical metrics
#    - Physical interpretation
#
# ### Production Recommendations
#
# ```python
# # For batch processing multiple spectra
# results = []
# for spectrum_file in spectrum_files:
#     wavenumber, intensity = load_spectrum(spectrum_file)
#     # ... preprocessing ...
#     popt, pcov = cf.curve_fit(model, x, y, ...)
#     results.append({'file': spectrum_file, 'params': popt, 'cov': pcov})
#
# # Save results to structured format
# import pandas as pd
# df = pd.DataFrame(results)
# df.to_csv('batch_fitting_results.csv')
# ```
#
# ### Next Steps
#
# - **Extend to 3+ peaks**: Add more Lorentzian components for complex spectra
# - **Voigt profiles**: Mix Gaussian and Lorentzian for realistic broadening
# - **Automated peak finding**: Use `scipy.signal.find_peaks` for initial guesses
# - **Batch processing**: Analyze multiple samples with automated workflows
# - **Advanced models**: Background modeling with splines or polynomials
#
# ### References
#
# 1. Ferrari & Robertson, *Phys. Rev. B* **61**, 14095 (2000)
# 2. NLSQ Documentation: https://nlsq.readthedocs.io/
# 3. Related examples:
#    - `nlsq_quickstart.ipynb` - Basic curve fitting
#    - `advanced_features_demo.ipynb` - Diagnostics and robustness
#    - `gallery/physics/spectroscopy_peaks.py` - Simple peak fitting
# ======================================================================
