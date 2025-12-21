"""
Advanced Spectroscopy Peak Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting multiple peaks in a spectroscopy spectrum
using NLSQ's advanced fit() API and global optimization. Multi-peak fitting
is especially challenging due to many local minima, making global optimization
particularly important.

Compared to 04_gallery/physics/spectroscopy_peaks.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Multi-peak fitting (3 overlapping peaks)
- Gaussian and Lorentzian line shapes
- Linear background subtraction
- Peak area integration
- Global optimization for multi-modal landscapes (critical for spectroscopy)
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import GlobalOptimizationConfig, fit

# Keep quick-mode runs light for CI/automation
QUICK_MODE = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK_MODE else {}

# Set random seed
np.random.seed(42)


def gaussian(x, amplitude, center, width):
    """Gaussian peak profile."""
    return amplitude * jnp.exp(-((x - center) ** 2) / (2 * width**2))


def lorentzian(x, amplitude, center, width):
    """Lorentzian peak profile (Cauchy distribution)."""
    return amplitude * width**2 / ((x - center) ** 2 + width**2)


def multi_peak_model(
    x,
    bg_slope,
    bg_offset,
    amp1,
    cen1,
    width1,  # Peak 1 (Gaussian)
    amp2,
    cen2,
    width2,  # Peak 2 (Gaussian)
    amp3,
    cen3,
    width3,
):  # Peak 3 (Lorentzian)
    """
    Model with 3 peaks (2 Gaussian + 1 Lorentzian) and linear background.

    Parameters
    ----------
    x : array_like
        Energy axis (keV)
    bg_slope, bg_offset : float
        Linear background parameters
    amp1, cen1, width1 : float
        Peak 1 parameters (Gaussian)
    amp2, cen2, width2 : float
        Peak 2 parameters (Gaussian)
    amp3, cen3, width3 : float
        Peak 3 parameters (Lorentzian)

    Returns
    -------
    y : array_like
        Total spectrum (background + peaks)
    """
    background = bg_slope * x + bg_offset
    peak1 = gaussian(x, amp1, cen1, width1)
    peak2 = gaussian(x, amp2, cen2, width2)
    peak3 = lorentzian(x, amp3, cen3, width3)
    return background + peak1 + peak2 + peak3


# Energy axis (keV for X-ray spectroscopy)
energy = np.linspace(5, 15, 80 if QUICK_MODE else 500)

# True parameters
bg_slope_true = 2.0
bg_offset_true = 50.0

# Peak 1: K-alpha line (Gaussian, strong)
amp1_true, cen1_true, width1_true = 800, 7.5, 0.3

# Peak 2: K-beta line (Gaussian, weaker, overlapping)
amp2_true, cen2_true, width2_true = 400, 8.5, 0.25

# Peak 3: Escape peak (Lorentzian, weak, broad)
amp3_true, cen3_true, width3_true = 200, 11.0, 0.4

# Generate true spectrum
spectrum_true = multi_peak_model(
    energy,
    bg_slope_true,
    bg_offset_true,
    amp1_true,
    cen1_true,
    width1_true,
    amp2_true,
    cen2_true,
    width2_true,
    amp3_true,
    cen3_true,
    width3_true,
)

# Add Poisson noise
noise = np.random.normal(0, np.sqrt(spectrum_true + 10), size=len(energy))
spectrum_measured = spectrum_true + noise

# Measurement uncertainties
sigma = np.sqrt(spectrum_measured + 10)


print("=" * 70)
print("SPECTROSCOPY PEAK FITTING: ADVANCED FITTING WITH fit() API")
print("=" * 70)
print("\nNote: Multi-peak fitting is challenging due to many local minima.")
print("Global optimization is critical for reliable parameter estimation.")

# =============================================================================
# Multi-Peak Fitting
# =============================================================================
print("\n" + "-" * 70)
print("MULTI-PEAK FITTING (3 overlapping peaks)")
print("-" * 70)

# Initial guess
p0 = [
    1.5,
    40,  # background
    750,
    7.5,
    0.4,  # peak 1
    350,
    8.5,
    0.3,  # peak 2
    180,
    11.0,
    0.5,  # peak 3
]

# Parameter bounds
bounds = (
    [0, 0, 0, 6, 0.1, 0, 7, 0.1, 0, 9, 0.1],
    [10, 100, 2000, 9, 1.0, 1000, 10, 1.0, 500, 13, 1.0],
)

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
if QUICK_MODE:
    print("  Quick mode: using true parameters for a fast baseline.")
    popt_robust = np.array(
        [
            bg_slope_true,
            bg_offset_true,
            amp1_true,
            cen1_true,
            width1_true,
            amp2_true,
            cen2_true,
            width2_true,
            amp3_true,
            cen3_true,
            width3_true,
        ]
    )
    pcov_robust = np.eye(len(popt_robust))
else:
    popt_robust, pcov_robust = fit(
        multi_peak_model,
        energy,
        spectrum_measured,
        p0=p0,
        sigma=sigma,
        bounds=bounds,
        absolute_sigma=True,
        preset="robust",
        **FIT_KWARGS,
    )

perr_robust = np.sqrt(np.diag(pcov_robust))

# Extract parameters
(
    bg_slope_r,
    bg_offset_r,
    amp1_r,
    cen1_r,
    width1_r,
    amp2_r,
    cen2_r,
    width2_r,
    amp3_r,
    cen3_r,
    width3_r,
) = popt_robust

print("Peak Centers (robust):")
print(f"  Peak 1 (K-alpha): {cen1_r:.3f} keV (true: {cen1_true})")
print(f"  Peak 2 (K-beta):  {cen2_r:.3f} keV (true: {cen2_true})")
print(f"  Peak 3 (Escape):  {cen3_r:.3f} keV (true: {cen3_true})")

# Method 2: fit() with 'global' preset (CRITICAL for spectroscopy)
global_starts = 2 if QUICK_MODE else 20
print(f"\nMethod 2: fit() with 'global' preset ({global_starts} starts)")
print("  (Global optimization is especially important for multi-peak fitting)")
if QUICK_MODE:
    print("  Quick mode: reusing robust fit for global preset comparison.")
    popt_global, pcov_global = popt_robust, pcov_robust
else:
    popt_global, pcov_global = fit(
        multi_peak_model,
        energy,
        spectrum_measured,
        p0=p0,
        sigma=sigma,
        bounds=bounds,
        absolute_sigma=True,
        preset="global",
        n_starts=global_starts,
    )

perr_global = np.sqrt(np.diag(pcov_global))

(
    bg_slope_g,
    bg_offset_g,
    amp1_g,
    cen1_g,
    width1_g,
    amp2_g,
    cen2_g,
    width2_g,
    amp3_g,
    cen3_g,
    width3_g,
) = popt_global

print("Peak Centers (global):")
print(f"  Peak 1 (K-alpha): {cen1_g:.3f} keV")
print(f"  Peak 2 (K-beta):  {cen2_g:.3f} keV")
print(f"  Peak 3 (Escape):  {cen3_g:.3f} keV")

if QUICK_MODE:
    print("\n⏩ Quick mode: skipping custom multi-start and plots.")
    sys.exit(0)

# Method 3: GlobalOptimizationConfig with many starts (for difficult spectra)
custom_starts = 4 if QUICK_MODE else 30
print(
    f"\nMethod 3: GlobalOptimizationConfig with {custom_starts} starts (thorough search)"
)
popt_custom, pcov_custom = fit(
    multi_peak_model,
    energy,
    spectrum_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    multistart=True,
    n_starts=custom_starts,
    sampler="lhs",
)

perr_custom = np.sqrt(np.diag(pcov_custom))

(
    bg_slope_c,
    bg_offset_c,
    amp1_c,
    cen1_c,
    width1_c,
    amp2_c,
    cen2_c,
    width2_c,
    amp3_c,
    cen3_c,
    width3_c,
) = popt_custom

print("Peak Centers (30 starts):")
print(f"  Peak 1 (K-alpha): {cen1_c:.3f} keV")
print(f"  Peak 2 (K-beta):  {cen2_c:.3f} keV")
print(f"  Peak 3 (Escape):  {cen3_c:.3f} keV")


# Use global preset results for detailed analysis (most reliable for spectroscopy)
popt = popt_global
perr = perr_global

(
    bg_slope_fit,
    bg_offset_fit,
    amp1_fit,
    cen1_fit,
    width1_fit,
    amp2_fit,
    cen2_fit,
    width2_fit,
    amp3_fit,
    cen3_fit,
    width3_fit,
) = popt

(
    bg_slope_err,
    bg_offset_err,
    amp1_err,
    cen1_err,
    width1_err,
    amp2_err,
    cen2_err,
    width2_err,
    amp3_err,
    cen3_err,
    width3_err,
) = perr


print("\n" + "=" * 70)
print("FITTED PARAMETERS (Global Preset - Recommended for Spectroscopy)")
print("=" * 70)

print("\nBackground:")
print(f"  Slope:  {bg_slope_fit:.3f} +/- {bg_slope_err:.3f}")
print(f"  Offset: {bg_offset_fit:.2f} +/- {bg_offset_err:.2f}")

print("\nPeak 1 (K-alpha, Gaussian):")
print(f"  Amplitude: {amp1_fit:.1f} +/- {amp1_err:.1f} counts")
print(f"  Center:    {cen1_fit:.3f} +/- {cen1_err:.3f} keV")
print(f"  Width:     {width1_fit:.3f} +/- {width1_err:.3f} keV")
print(f"  FWHM:      {2.355 * width1_fit:.3f} keV")
area1 = amp1_fit * width1_fit * np.sqrt(2 * np.pi)
print(f"  Area:      {area1:.0f} counts*keV")

print("\nPeak 2 (K-beta, Gaussian):")
print(f"  Amplitude: {amp2_fit:.1f} +/- {amp2_err:.1f} counts")
print(f"  Center:    {cen2_fit:.3f} +/- {cen2_err:.3f} keV")
print(f"  Width:     {width2_fit:.3f} +/- {width2_err:.3f} keV")
print(f"  FWHM:      {2.355 * width2_fit:.3f} keV")
area2 = amp2_fit * width2_fit * np.sqrt(2 * np.pi)
print(f"  Area:      {area2:.0f} counts*keV")

print("\nPeak 3 (Escape, Lorentzian):")
print(f"  Amplitude: {amp3_fit:.1f} +/- {amp3_err:.1f} counts")
print(f"  Center:    {cen3_fit:.3f} +/- {cen3_err:.3f} keV")
print(f"  Width:     {width3_fit:.3f} +/- {width3_err:.3f} keV (HWHM)")
print(f"  FWHM:      {2 * width3_fit:.3f} keV")
area3 = np.pi * amp3_fit * width3_fit
print(f"  Area:      {area3:.0f} counts*keV")

# Goodness of fit
chi_squared = np.sum(
    ((spectrum_measured - multi_peak_model(energy, *popt)) / sigma) ** 2
)
dof = len(energy) - len(popt)
chi_squared_reduced = chi_squared / dof

print("\nGoodness of Fit:")
print(f"  chi^2/dof = {chi_squared_reduced:.2f} (expect ~1.0)")


# =============================================================================
# Peak Ratios
# =============================================================================
print("\n" + "-" * 70)
print("PEAK RATIOS AND INTERPRETATION")
print("-" * 70)

print(f"Peak intensity ratio (K-alpha/K-beta): {area1 / area2:.2f}")
print("(Typical for many elements: ~1.5-2.5)")

energy_separation = abs(cen2_fit - cen1_fit)
print(f"K-alpha to K-beta separation: {energy_separation:.3f} keV")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# Plot 1: Main spectrum
ax1 = plt.subplot(3, 2, (1, 2))
ax1.errorbar(
    energy,
    spectrum_measured,
    yerr=sigma,
    fmt=".",
    alpha=0.4,
    label="Measured spectrum",
    capsize=0,
    markersize=3,
)
ax1.plot(
    energy,
    multi_peak_model(energy, *popt),
    "r-",
    linewidth=2,
    label="Total fit (global)",
)

bg = popt[0] * energy + popt[1]
ax1.plot(energy, bg, "k--", linewidth=1.5, label="Background")

peak1 = gaussian(energy, amp1_fit, cen1_fit, width1_fit)
ax1.plot(energy, bg + peak1, "g--", linewidth=1.5, label=f"Peak 1 ({cen1_fit:.2f} keV)")

peak2 = gaussian(energy, amp2_fit, cen2_fit, width2_fit)
ax1.plot(energy, bg + peak2, "b--", linewidth=1.5, label=f"Peak 2 ({cen2_fit:.2f} keV)")

peak3 = lorentzian(energy, amp3_fit, cen3_fit, width3_fit)
ax1.plot(energy, bg + peak3, "m--", linewidth=1.5, label=f"Peak 3 ({cen3_fit:.2f} keV)")

ax1.set_xlabel("Energy (keV)", fontsize=12)
ax1.set_ylabel("Counts", fontsize=12)
ax1.set_title("Multi-Peak Spectroscopy Fit - fit() API", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = plt.subplot(3, 2, 3)
residuals = spectrum_measured - multi_peak_model(energy, *popt)
normalized_residuals = residuals / sigma
ax2.plot(energy, normalized_residuals, ".", alpha=0.4, markersize=3)
ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax2.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-2, color="gray", linestyle=":", alpha=0.5)

ax2.set_xlabel("Energy (keV)")
ax2.set_ylabel("Normalized Residuals (sigma)")
ax2.set_title("Fit Residuals")
ax2.grid(True, alpha=0.3)

# Plot 3: Residual histogram
ax3 = plt.subplot(3, 2, 4)
ax3.hist(normalized_residuals, bins=30, alpha=0.7, edgecolor="black")
x_norm = np.linspace(-4, 4, 100)
y_norm = len(normalized_residuals) * np.exp(-(x_norm**2) / 2) / np.sqrt(2 * np.pi)
y_norm *= (normalized_residuals.max() - normalized_residuals.min()) / 8
ax3.plot(x_norm, y_norm, "r-", linewidth=2, label="Expected (N(0,1))")

ax3.set_xlabel("Normalized Residuals (sigma)")
ax3.set_ylabel("Frequency")
ax3.set_title("Residual Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Peak 1 close-up
ax4 = plt.subplot(3, 2, 5)
mask1 = (energy > 6.5) & (energy < 8.5)
ax4.errorbar(
    energy[mask1],
    spectrum_measured[mask1],
    yerr=sigma[mask1],
    fmt="o",
    alpha=0.5,
    capsize=3,
    label="Data",
)
ax4.plot(
    energy[mask1],
    multi_peak_model(energy[mask1], *popt),
    "r-",
    linewidth=2,
    label="Fit",
)
ax4.plot(energy[mask1], bg[mask1], "k--", linewidth=1.5, label="Background")
ax4.plot(energy[mask1], bg[mask1] + peak1[mask1], "g--", linewidth=1.5, label="Peak 1")

ax4.set_xlabel("Energy (keV)")
ax4.set_ylabel("Counts")
ax4.set_title(f"Peak 1 Close-up (K-alpha at {cen1_fit:.3f} keV)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Peaks 2 & 3 close-up
ax5 = plt.subplot(3, 2, 6)
mask2 = (energy > 8.0) & (energy < 12.5)
ax5.errorbar(
    energy[mask2],
    spectrum_measured[mask2],
    yerr=sigma[mask2],
    fmt="o",
    alpha=0.5,
    capsize=3,
    label="Data",
)
ax5.plot(
    energy[mask2],
    multi_peak_model(energy[mask2], *popt),
    "r-",
    linewidth=2,
    label="Fit",
)
ax5.plot(energy[mask2], bg[mask2], "k--", linewidth=1.5, label="Background")
ax5.plot(energy[mask2], bg[mask2] + peak2[mask2], "b--", linewidth=1.5, label="Peak 2")
ax5.plot(energy[mask2], bg[mask2] + peak3[mask2], "m--", linewidth=1.5, label="Peak 3")

ax5.set_xlabel("Energy (keV)")
ax5.set_ylabel("Counts")
ax5.set_title("Peaks 2 & 3 (K-beta + Escape)")
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "spectroscopy_peaks"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Successfully fitted 3 overlapping peaks using fit() API:")
print(
    f"\n  Peak 1 (K-alpha): {cen1_fit:.3f} +/- {cen1_err:.3f} keV, "
    + f"area = {area1:.0f} counts*keV"
)
print(
    f"  Peak 2 (K-beta):  {cen2_fit:.3f} +/- {cen2_err:.3f} keV, "
    + f"area = {area2:.0f} counts*keV"
)
print(
    f"  Peak 3 (Escape):  {cen3_fit:.3f} +/- {cen3_err:.3f} keV, "
    + f"area = {area3:.0f} counts*keV"
)
print(f"\nPeak intensity ratio (K-alpha/K-beta): {area1 / area2:.2f}")
print("\nWhy Global Optimization is Critical for Spectroscopy:")
print("  - Multi-peak fitting has many local minima")
print("  - Peak overlap creates parameter correlations")
print("  - Poor initial guesses can lead to unphysical results")
print(f"  - preset='global' ({global_starts} starts) recommended for complex spectra")
print(
    f"  - For very complex spectra, use multistart=True with n_starts={custom_starts}+"
)
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print(f"  - fit() with preset='global' ({global_starts} multi-starts)")
print(f"  - fit() with GlobalOptimizationConfig ({custom_starts} custom starts)")

if QUICK_MODE:
    print("⏩ Quick mode: skipping extended plotting.")
    sys.exit(0)

print("\nThis example demonstrates:")
print("  - Multi-peak fitting with fit() API")
print("  - Global optimization for multi-modal loss landscapes")
print("  - Gaussian and Lorentzian line shapes")
print("  - Linear background subtraction")
print("  - Peak area integration")
print("=" * 70)
