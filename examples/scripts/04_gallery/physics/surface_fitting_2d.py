"""
2D Surface Fitting: Gaussian Beam Profile Analysis

This example demonstrates fitting 2D surface data, a common task in:
- Laser beam profiling
- Microscopy point spread functions
- AFM/STM topography analysis
- Image analysis and blob detection

NLSQ supports 2D fitting by passing xdata as shape (2, n_points) where:
- xdata[0] = x coordinates (flattened)
- xdata[1] = y coordinates (flattened)
- ydata = z values at each (x, y) point

Key Concepts:
- 2D Gaussian surface fitting
- Multi-dimensional independent variables
- Elliptical vs circular Gaussian profiles
- Surface parameter extraction (center, widths, amplitude)
- Weighted fitting with 2D uncertainty arrays
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# 2D Model Functions
# =============================================================================


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian surface model (elliptical).

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) where xy[0] = x coordinates, xy[1] = y coordinates.
    amplitude : float
        Peak amplitude (max height above offset).
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations in x and y directions.
    offset : float
        Background offset (baseline).

    Returns
    -------
    z : ndarray
        Shape (n_points,) with z values at each (x, y) point.
    """
    x = xy[0]
    y = xy[1]
    return (
        amplitude
        * jnp.exp(
            -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
        )
        + offset
    )


def gaussian_2d_rotated(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian surface with rotation (tilted ellipse).

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) with x, y coordinates.
    amplitude : float
        Peak amplitude.
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations along principal axes.
    theta : float
        Rotation angle in radians.
    offset : float
        Background offset.

    Returns
    -------
    z : ndarray
        Surface values.
    """
    x = xy[0]
    y = xy[1]

    # Rotate coordinates
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t

    return (
        amplitude * jnp.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
        + offset
    )


# =============================================================================
# Generate Synthetic 2D Data
# =============================================================================

print("=" * 70)
print("2D SURFACE FITTING: GAUSSIAN BEAM PROFILE ANALYSIS")
print("=" * 70)

# Create a 2D grid (simulating a camera sensor)
nx, ny = 50, 50  # 50x50 pixel sensor
x_1d = np.linspace(-5, 5, nx)
y_1d = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x_1d, y_1d)

# Flatten for curve_fit (NLSQ expects flattened 2D data)
x_flat = X.flatten()
y_flat = Y.flatten()
n_points = len(x_flat)

# Stack into shape (2, n_points) for xdata
xdata = np.vstack([x_flat, y_flat])

print(f"\nGrid: {nx} x {ny} = {n_points} points")
print(f"xdata shape: {xdata.shape}")

# True parameters (simulating a laser beam profile)
amplitude_true = 1000.0  # Peak intensity (counts)
x0_true = 0.5  # Beam center x (slightly off-center)
y0_true = -0.3  # Beam center y
sigma_x_true = 1.5  # Beam width x (mm)
sigma_y_true = 1.2  # Beam width y (elliptical)
offset_true = 50.0  # Background (dark counts)

print("\nTrue Parameters (Laser Beam Profile):")
print(f"  Amplitude: {amplitude_true:.1f} counts")
print(f"  Center:    ({x0_true:.2f}, {y0_true:.2f}) mm")
print(f"  Widths:    sigma_x = {sigma_x_true:.2f}, sigma_y = {sigma_y_true:.2f} mm")
print(f"  Offset:    {offset_true:.1f} counts")

# Generate true surface
z_true = gaussian_2d(
    xdata, amplitude_true, x0_true, y0_true, sigma_x_true, sigma_y_true, offset_true
)

# Add Poisson-like noise (realistic for photon counting)
noise_scale = np.sqrt(z_true + 10)  # Poisson approximation
noise = np.random.normal(0, noise_scale)
z_measured = z_true + noise

# Uncertainties
sigma = noise_scale

print(
    f"\nData generated with Poisson noise (SNR ~ {amplitude_true / np.mean(noise_scale):.1f})"
)


# =============================================================================
# Fit 2D Gaussian Surface
# =============================================================================

print("\n" + "-" * 70)
print("FITTING 2D GAUSSIAN SURFACE")
print("-" * 70)

# Initial guess (from data inspection)
p0 = [
    z_measured.max() - z_measured.min(),  # amplitude
    0.0,  # x0 (center guess)
    0.0,  # y0
    2.0,  # sigma_x
    2.0,  # sigma_y
    z_measured.min(),  # offset
]

# Parameter bounds (physical constraints)
bounds = (
    [0, -5, -5, 0.1, 0.1, 0],  # Lower: positive amplitude, widths > 0
    [2000, 5, 5, 10, 10, 200],  # Upper: reasonable maxima
)

print("\nInitial guess:")
print(f"  p0 = {p0}")

# Perform the fit
popt, pcov = curve_fit(
    gaussian_2d,
    xdata,
    z_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

# Extract fitted parameters
amplitude_fit, x0_fit, y0_fit, sigma_x_fit, sigma_y_fit, offset_fit = popt
perr = np.sqrt(np.diag(pcov))
amp_err, x0_err, y0_err, sx_err, sy_err, off_err = perr

print("\nFitted Parameters:")
print(
    f"  Amplitude: {amplitude_fit:.1f} +/- {amp_err:.1f} counts (true: {amplitude_true:.1f})"
)
print(f"  Center x:  {x0_fit:.3f} +/- {x0_err:.3f} mm (true: {x0_true:.2f})")
print(f"  Center y:  {y0_fit:.3f} +/- {y0_err:.3f} mm (true: {y0_true:.2f})")
print(f"  Sigma x:   {sigma_x_fit:.3f} +/- {sx_err:.3f} mm (true: {sigma_x_true:.2f})")
print(f"  Sigma y:   {sigma_y_fit:.3f} +/- {sy_err:.3f} mm (true: {sigma_y_true:.2f})")
print(
    f"  Offset:    {offset_fit:.1f} +/- {off_err:.1f} counts (true: {offset_true:.1f})"
)

# Derived quantities
fwhm_x = 2.355 * sigma_x_fit  # FWHM = 2.355 * sigma
fwhm_y = 2.355 * sigma_y_fit
beam_area = 2 * np.pi * sigma_x_fit * sigma_y_fit  # Beam area
total_intensity = amplitude_fit * beam_area  # Integrated intensity
ellipticity = sigma_x_fit / sigma_y_fit

print("\nDerived Beam Properties:")
print(f"  FWHM x:           {fwhm_x:.3f} mm")
print(f"  FWHM y:           {fwhm_y:.3f} mm")
print(f"  Ellipticity:      {ellipticity:.3f} (1.0 = circular)")
print(f"  Beam area:        {beam_area:.2f} mm^2")
print(f"  Total intensity:  {total_intensity:.0f} counts*mm^2")

# Goodness of fit
z_fit = gaussian_2d(xdata, *popt)
residuals = z_measured - z_fit
chi_squared = np.sum((residuals / sigma) ** 2)
dof = n_points - len(popt)
chi_squared_reduced = chi_squared / dof

print("\nGoodness of Fit:")
print(f"  chi^2/dof = {chi_squared_reduced:.3f} (expect ~1.0)")


# =============================================================================
# Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 12))

# Reshape for 2D plotting
Z_measured = z_measured.reshape(ny, nx)
Z_fit = z_fit.reshape(ny, nx)
Z_residuals = residuals.reshape(ny, nx)

# Plot 1: Measured data (2D image)
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.imshow(
    Z_measured,
    extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
ax1.contour(X, Y, Z_measured, levels=5, colors="white", alpha=0.5, linewidths=0.5)
ax1.plot(x0_fit, y0_fit, "r+", markersize=15, markeredgewidth=2, label="Fitted center")
ax1.set_xlabel("x (mm)")
ax1.set_ylabel("y (mm)")
ax1.set_title("Measured Data", fontweight="bold")
ax1.legend(loc="upper right")
plt.colorbar(im1, ax=ax1, label="Intensity (counts)")

# Plot 2: Fitted surface
ax2 = fig.add_subplot(2, 3, 2)
im2 = ax2.imshow(
    Z_fit,
    extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
ax2.contour(X, Y, Z_fit, levels=5, colors="white", alpha=0.5, linewidths=0.5)
ax2.plot(x0_fit, y0_fit, "r+", markersize=15, markeredgewidth=2)
ax2.set_xlabel("x (mm)")
ax2.set_ylabel("y (mm)")
ax2.set_title("Fitted 2D Gaussian", fontweight="bold")
plt.colorbar(im2, ax=ax2, label="Intensity (counts)")

# Plot 3: Residuals
ax3 = fig.add_subplot(2, 3, 3)
im3 = ax3.imshow(
    Z_residuals,
    extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
    origin="lower",
    cmap="RdBu_r",
    aspect="equal",
    vmin=-3 * np.std(residuals),
    vmax=3 * np.std(residuals),
)
ax3.set_xlabel("x (mm)")
ax3.set_ylabel("y (mm)")
ax3.set_title("Residuals (Data - Fit)", fontweight="bold")
plt.colorbar(im3, ax=ax3, label="Residual (counts)")

# Plot 4: X cross-section through center
ax4 = fig.add_subplot(2, 3, 4)
y_idx = np.argmin(np.abs(y_1d - y0_fit))
ax4.errorbar(
    x_1d,
    Z_measured[y_idx, :],
    yerr=sigma.reshape(ny, nx)[y_idx, :],
    fmt="o",
    alpha=0.6,
    label="Data",
    capsize=2,
    markersize=4,
)
ax4.plot(x_1d, Z_fit[y_idx, :], "r-", linewidth=2, label="Fit")
ax4.axvline(x0_fit, color="gray", linestyle="--", alpha=0.5, label=f"x0 = {x0_fit:.2f}")
ax4.set_xlabel("x (mm)")
ax4.set_ylabel("Intensity (counts)")
ax4.set_title(f"X Cross-section (y = {y_1d[y_idx]:.2f} mm)", fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Y cross-section through center
ax5 = fig.add_subplot(2, 3, 5)
x_idx = np.argmin(np.abs(x_1d - x0_fit))
ax5.errorbar(
    y_1d,
    Z_measured[:, x_idx],
    yerr=sigma.reshape(ny, nx)[:, x_idx],
    fmt="o",
    alpha=0.6,
    label="Data",
    capsize=2,
    markersize=4,
)
ax5.plot(y_1d, Z_fit[:, x_idx], "r-", linewidth=2, label="Fit")
ax5.axvline(y0_fit, color="gray", linestyle="--", alpha=0.5, label=f"y0 = {y0_fit:.2f}")
ax5.set_xlabel("y (mm)")
ax5.set_ylabel("Intensity (counts)")
ax5.set_title(f"Y Cross-section (x = {x_1d[x_idx]:.2f} mm)", fontweight="bold")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Residual histogram
ax6 = fig.add_subplot(2, 3, 6)
norm_residuals = residuals / sigma
ax6.hist(norm_residuals, bins=40, alpha=0.7, edgecolor="black", density=True)
# Overlay expected normal distribution
x_norm = np.linspace(-4, 4, 100)
y_norm = np.exp(-(x_norm**2) / 2) / np.sqrt(2 * np.pi)
ax6.plot(x_norm, y_norm, "r-", linewidth=2, label="N(0,1)")
ax6.set_xlabel("Normalized Residuals")
ax6.set_ylabel("Probability Density")
ax6.set_title("Residual Distribution", fontweight="bold")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_dir = Path(__file__).parent / "figures" / "surface_fitting_2d"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Successfully fitted 2D Gaussian surface to beam profile data!")
print(f"\nBeam Center: ({x0_fit:.3f}, {y0_fit:.3f}) mm")
print(f"Beam Widths: {fwhm_x:.3f} x {fwhm_y:.3f} mm (FWHM)")
print(f"Ellipticity: {ellipticity:.3f}")
print(f"Peak Intensity: {amplitude_fit:.1f} counts")
print(f"\nFit Quality: chi^2/dof = {chi_squared_reduced:.3f}")

print("\nThis example demonstrates:")
print("  - 2D surface fitting with NLSQ")
print("  - Multi-dimensional xdata (shape (2, n_points))")
print("  - Elliptical Gaussian beam profile analysis")
print("  - Cross-sectional visualization of 2D fits")
print("  - Weighted fitting with 2D uncertainty arrays")

print("\nApplications:")
print("  - Laser beam profiling and M^2 measurements")
print("  - Microscopy PSF characterization")
print("  - AFM/STM topography fitting")
print("  - Astronomical source extraction")
print("  - Particle/blob detection in images")
print("=" * 70)
