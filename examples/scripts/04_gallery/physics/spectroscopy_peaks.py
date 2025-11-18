"""
Converted from spectroscopy_peaks.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Spectroscopy Peak Fitting: Multi-Peak Deconvolution
#
#
# This example demonstrates fitting multiple peaks in a spectroscopy spectrum,
# a common task in X-ray, Raman, and optical spectroscopy. We fit overlapping
# Gaussian and Lorentzian peaks with background subtraction.
#
# Key Concepts:
# - Multi-peak fitting (3 overlapping peaks)
# - Gaussian and Lorentzian line shapes
# - Linear background subtraction
# - Peak area integration
# - Parameter constraints (positive widths, amplitudes)
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


def gaussian(x, amplitude, center, width):
    """
    Gaussian peak profile.

    Parameters
    ----------
    x : array_like
        Energy/wavelength axis
    amplitude : float
        Peak amplitude (max height)
    center : float
        Peak center position
    width : float
        Peak width (standard deviation)

    Returns
    -------
    y : array_like
        Gaussian profile
    """
    return amplitude * jnp.exp(-((x - center) ** 2) / (2 * width**2))


def lorentzian(x, amplitude, center, width):
    """
    Lorentzian peak profile (Cauchy distribution).

    Common in spectroscopy for broadened lines.

    Parameters
    ----------
    x : array_like
        Energy/wavelength axis
    amplitude : float
        Peak amplitude (max height)
    center : float
        Peak center position
    width : float
        Peak width (half-width at half-maximum, HWHM)

    Returns
    -------
    y : array_like
        Lorentzian profile
    """
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
    # Linear background
    background = bg_slope * x + bg_offset

    # Three peaks
    peak1 = gaussian(x, amp1, cen1, width1)
    peak2 = gaussian(x, amp2, cen2, width2)
    peak3 = lorentzian(x, amp3, cen3, width3)

    return background + peak1 + peak2 + peak3


# Energy axis (keV for X-ray spectroscopy example)
energy = np.linspace(5, 15, 500)  # 500 channels

# True parameters
bg_slope_true = 2.0
bg_offset_true = 50.0

# Peak 1: Kα line (Gaussian, strong)  # noqa: RUF003
amp1_true, cen1_true, width1_true = 800, 7.5, 0.3

# Peak 2: Kβ line (Gaussian, weaker, overlapping)
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

# Add Poisson noise (realistic for counting statistics)
# Use square root noise (Poisson approximation)
noise = np.random.normal(0, np.sqrt(spectrum_true + 10), size=len(energy))
spectrum_measured = spectrum_true + noise

# Measurement uncertainties (Poisson)
sigma = np.sqrt(spectrum_measured + 10)


print("=" * 70)
print("SPECTROSCOPY PEAK FITTING: MULTI-PEAK DECONVOLUTION")
print("=" * 70)

# Initial guess (roughly from visual inspection)
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

# Parameter bounds (all amplitudes > 0, widths > 0, centers within range)
bounds = (
    [
        0,
        0,  # background
        0,
        6,
        0.1,  # peak 1: amp > 0, 6 < cen < 9, width > 0.1
        0,
        7,
        0.1,  # peak 2: amp > 0, 7 < cen < 10, width > 0.1
        0,
        9,
        0.1,
    ],  # peak 3: amp > 0, 9 < cen < 13, width > 0.1
    [
        10,
        100,  # background
        2000,
        9,
        1.0,  # peak 1
        1000,
        10,
        1.0,  # peak 2
        500,
        13,
        1.0,
    ],  # peak 3
)

# Fit with weighted least squares
popt, pcov = curve_fit(
    multi_peak_model,
    energy,
    spectrum_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

# Extract fitted parameters
perr = np.sqrt(np.diag(pcov))
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

# Parameter uncertainties
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


print("\nFitted Parameters:")
print("\nBackground:")
print(f"  Slope:  {bg_slope_fit:.3f} ± {bg_slope_err:.3f}")
print(f"  Offset: {bg_offset_fit:.2f} ± {bg_offset_err:.2f}")

print("\nPeak 1 (Kα, Gaussian):")
print(f"  Amplitude: {amp1_fit:.1f} ± {amp1_err:.1f} counts")
print(f"  Center:    {cen1_fit:.3f} ± {cen1_err:.3f} keV")
print(f"  Width:     {width1_fit:.3f} ± {width1_err:.3f} keV")
print(f"  FWHM:      {2.355 * width1_fit:.3f} keV")  # FWHM = 2.355 * σ  # noqa: RUF003
# Peak area (Gaussian)
area1 = amp1_fit * width1_fit * np.sqrt(2 * np.pi)
print(f"  Area:      {area1:.0f} counts·keV")

print("\nPeak 2 (Kβ, Gaussian):")
print(f"  Amplitude: {amp2_fit:.1f} ± {amp2_err:.1f} counts")
print(f"  Center:    {cen2_fit:.3f} ± {cen2_err:.3f} keV")
print(f"  Width:     {width2_fit:.3f} ± {width2_err:.3f} keV")
print(f"  FWHM:      {2.355 * width2_fit:.3f} keV")
area2 = amp2_fit * width2_fit * np.sqrt(2 * np.pi)
print(f"  Area:      {area2:.0f} counts·keV")

print("\nPeak 3 (Escape, Lorentzian):")
print(f"  Amplitude: {amp3_fit:.1f} ± {amp3_err:.1f} counts")
print(f"  Center:    {cen3_fit:.3f} ± {cen3_err:.3f} keV")
print(f"  Width:     {width3_fit:.3f} ± {width3_err:.3f} keV (HWHM)")
print(f"  FWHM:      {2 * width3_fit:.3f} keV")
# Peak area (Lorentzian)
area3 = np.pi * amp3_fit * width3_fit
print(f"  Area:      {area3:.0f} counts·keV")

# Goodness of fit
chi_squared = np.sum(
    ((spectrum_measured - multi_peak_model(energy, *popt)) / sigma) ** 2
)
dof = len(energy) - len(popt)
chi_squared_reduced = chi_squared / dof
print("\nGoodness of Fit:")
print(f"  χ²/dof = {chi_squared_reduced:.2f} (expect ≈ 1.0)")


fig = plt.figure(figsize=(16, 10))

# Main spectrum plot
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
ax1.plot(energy, multi_peak_model(energy, *popt), "r-", linewidth=2, label="Total fit")

# Plot individual components
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
ax1.set_title("Multi-Peak Spectroscopy Fit", fontsize=14, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# Residuals plot
ax2 = plt.subplot(3, 2, 3)
residuals = spectrum_measured - multi_peak_model(energy, *popt)
normalized_residuals = residuals / sigma
ax2.plot(energy, normalized_residuals, ".", alpha=0.4, markersize=3)
ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax2.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Energy (keV)")
ax2.set_ylabel("Normalized Residuals (σ)")
ax2.set_title("Fit Residuals")
ax2.grid(True, alpha=0.3)

# Histogram of normalized residuals
ax3 = plt.subplot(3, 2, 4)
ax3.hist(normalized_residuals, bins=30, alpha=0.7, edgecolor="black")
# Overlay expected normal distribution
x_norm = np.linspace(-4, 4, 100)
y_norm = len(normalized_residuals) * np.exp(-(x_norm**2) / 2) / np.sqrt(2 * np.pi)
y_norm *= (normalized_residuals.max() - normalized_residuals.min()) / 8
ax3.plot(x_norm, y_norm, "r-", linewidth=2, label="Expected (N(0,1))")
ax3.set_xlabel("Normalized Residuals (σ)")
ax3.set_ylabel("Frequency")
ax3.set_title("Residual Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Zoomed view of Peak 1
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
ax4.set_title(f"Peak 1 Close-up (Kα at {cen1_fit:.3f} keV)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Zoomed view of Peaks 2 & 3
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
ax5.set_title("Peaks 2 & 3 (Kβ + Escape)")
ax5.legend()
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("spectroscopy_peaks.png", dpi=150)
print("\n✅ Plot saved as 'spectroscopy_peaks.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "spectroscopy_peaks"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Successfully fitted 3 overlapping peaks:")
print(
    f"  Peak 1 (Kα):     {cen1_fit:.3f} ± {cen1_err:.3f} keV, "
    + f"area = {area1:.0f} counts·keV"
)
print(
    f"  Peak 2 (Kβ):     {cen2_fit:.3f} ± {cen2_err:.3f} keV, "
    + f"area = {area2:.0f} counts·keV"
)
print(
    f"  Peak 3 (Escape): {cen3_fit:.3f} ± {cen3_err:.3f} keV, "
    + f"area = {area3:.0f} counts·keV"
)
print(f"\nPeak intensity ratio (Kα/Kβ): {area1 / area2:.2f}")
print("(Typical for many elements: ~1.5-2.5)")
print("\nThis example demonstrates:")
print("  ✓ Multi-peak fitting with overlapping peaks")
print("  ✓ Gaussian and Lorentzian line shapes")
print("  ✓ Linear background subtraction")
print("  ✓ Parameter bounds for physical constraints")
print("  ✓ Peak area integration")
print("  ✓ Weighted least squares with Poisson noise")
print("=" * 70)
