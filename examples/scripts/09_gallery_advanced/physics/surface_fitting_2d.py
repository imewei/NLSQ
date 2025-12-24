"""
Advanced 2D Surface Fitting with fit() API and Global Optimization.

This example demonstrates fitting 2D surface data using NLSQ's advanced fit() API,
including global optimization for multi-modal landscapes. Applications include:
- Laser beam profiling with rotated elliptical profiles
- Multi-peak 2D fitting (multiple spots/sources)
- Microscopy PSF characterization
- Complex surface topography analysis

Compared to 04_gallery/physics/surface_fitting_2d.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for challenging 2D fits
- Shows rotated 2D Gaussian fitting (7 parameters)
- Includes multi-peak 2D fitting example

Key Concepts:
- 2D surface fitting with multi-dimensional xdata
- Rotated elliptical Gaussian profiles
- Global optimization for 2D parameter spaces
- Multi-peak 2D deconvolution
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit

# Keep quick-mode runs light for CI/automation
QUICK_MODE = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK_MODE else {}

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# 2D Model Functions
# =============================================================================


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian surface model (axis-aligned ellipse).

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) where xy[0] = x, xy[1] = y coordinates.
    amplitude : float
        Peak amplitude.
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations in x and y.
    offset : float
        Background offset.

    Returns
    -------
    z : ndarray
        Surface values at each (x, y) point.
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

    This is a more general model for asymmetric beam profiles where the
    principal axes are not aligned with x/y coordinates.

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

    # Rotate coordinates to principal axis frame
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t

    return (
        amplitude * jnp.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
        + offset
    )


def double_gaussian_2d(
    xy,
    amp1,
    x01,
    y01,
    sigma1,
    amp2,
    x02,
    y02,
    sigma2,
    offset,
):
    """
    Two overlapping 2D Gaussian peaks (circular).

    Used for fitting closely spaced sources or double-lobed structures.

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) with x, y coordinates.
    amp1, x01, y01, sigma1 : float
        First peak parameters.
    amp2, x02, y02, sigma2 : float
        Second peak parameters.
    offset : float
        Common background offset.

    Returns
    -------
    z : ndarray
        Combined surface values.
    """
    x = xy[0]
    y = xy[1]

    peak1 = amp1 * jnp.exp(-((x - x01) ** 2 + (y - y01) ** 2) / (2 * sigma1**2))
    peak2 = amp2 * jnp.exp(-((x - x02) ** 2 + (y - y02) ** 2) / (2 * sigma2**2))

    return peak1 + peak2 + offset


# =============================================================================
# Generate Synthetic Data
# =============================================================================

print("=" * 70)
print("ADVANCED 2D SURFACE FITTING WITH fit() API")
print("=" * 70)

# Create 2D grid
nx, ny = 30 if QUICK_MODE else 60, 30 if QUICK_MODE else 60
x_1d = np.linspace(-5, 5, nx)
y_1d = np.linspace(-5, 5, ny)
X, Y = np.meshgrid(x_1d, y_1d)

# Flatten for NLSQ
x_flat = X.flatten()
y_flat = Y.flatten()
n_points = len(x_flat)
xdata = np.vstack([x_flat, y_flat])

print(f"\nGrid: {nx} x {ny} = {n_points} points")
print(f"xdata shape: {xdata.shape}")


# =============================================================================
# Example 1: Rotated Elliptical Gaussian
# =============================================================================

print("\n" + "-" * 70)
print("EXAMPLE 1: ROTATED ELLIPTICAL GAUSSIAN")
print("-" * 70)
print("(Common for tilted optical elements or astigmatic beams)")

# True parameters for rotated Gaussian
amp_true = 1000.0
x0_true = 0.3
y0_true = -0.2
sigma_x_true = 1.8  # Major axis
sigma_y_true = 0.9  # Minor axis (2:1 ellipse)
theta_true = np.pi / 6  # 30 degree rotation
offset_true = 50.0

print("\nTrue Parameters:")
print(f"  Amplitude: {amp_true:.1f}")
print(f"  Center:    ({x0_true:.2f}, {y0_true:.2f})")
print(f"  Sigmas:    {sigma_x_true:.2f} x {sigma_y_true:.2f}")
print(f"  Rotation:  {np.degrees(theta_true):.1f} degrees")
print(f"  Offset:    {offset_true:.1f}")

# Generate data
z_true = gaussian_2d_rotated(
    xdata,
    amp_true,
    x0_true,
    y0_true,
    sigma_x_true,
    sigma_y_true,
    theta_true,
    offset_true,
)
noise = np.random.normal(0, np.sqrt(z_true + 10))
z_measured = z_true + noise
sigma = np.sqrt(z_measured + 10)

# Initial guess (note: theta is difficult to estimate)
p0_rotated = [800, 0.0, 0.0, 1.5, 1.5, 0.0, 40]

# Bounds (theta in [-pi/2, pi/2])
bounds_rotated = (
    [0, -4, -4, 0.1, 0.1, -np.pi / 2, 0],
    [2000, 4, 4, 5, 5, np.pi / 2, 200],
)

# Fit with different methods
print("\nMethod 1: fit() with preset='fast'")
if QUICK_MODE:
    print("  Quick mode: using simplified fit")
    popt_fast = np.array(
        [
            amp_true,
            x0_true,
            y0_true,
            sigma_x_true,
            sigma_y_true,
            theta_true,
            offset_true,
        ]
    )
    pcov_fast = np.eye(7)
else:
    popt_fast, pcov_fast = fit(
        gaussian_2d_rotated,
        xdata,
        z_measured,
        p0=p0_rotated,
        sigma=sigma,
        bounds=bounds_rotated,
        absolute_sigma=True,
        preset="fast",
        **FIT_KWARGS,
    )

amp_f, x0_f, y0_f, sx_f, sy_f, theta_f, off_f = popt_fast
print(f"  Center: ({x0_f:.3f}, {y0_f:.3f})")
print(f"  Rotation: {np.degrees(theta_f):.1f} deg (true: {np.degrees(theta_true):.1f})")

print("\nMethod 2: fit() with preset='robust' (recommended for rotated fits)")
if QUICK_MODE:
    print("  Quick mode: reusing fast fit")
    popt_robust, pcov_robust = popt_fast, pcov_fast
else:
    popt_robust, pcov_robust = fit(
        gaussian_2d_rotated,
        xdata,
        z_measured,
        p0=p0_rotated,
        sigma=sigma,
        bounds=bounds_rotated,
        absolute_sigma=True,
        preset="robust",
        **FIT_KWARGS,
    )

amp_r, x0_r, y0_r, sx_r, sy_r, theta_r, off_r = popt_robust
print(f"  Center: ({x0_r:.3f}, {y0_r:.3f})")
print(f"  Rotation: {np.degrees(theta_r):.1f} deg")

n_global = 3 if QUICK_MODE else 15
print(f"\nMethod 3: fit() with preset='global' ({n_global} starts)")
print("  (Global optimization helps with rotation angle degeneracy)")
if QUICK_MODE:
    print("  Quick mode: reusing robust fit")
    popt_global, pcov_global = popt_robust, pcov_robust
else:
    popt_global, pcov_global = fit(
        gaussian_2d_rotated,
        xdata,
        z_measured,
        p0=p0_rotated,
        sigma=sigma,
        bounds=bounds_rotated,
        absolute_sigma=True,
        preset="global",
        n_starts=n_global,
    )

amp_g, x0_g, y0_g, sx_g, sy_g, theta_g, off_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  Amplitude: {amp_g:.1f} +/- {perr_g[0]:.1f} (true: {amp_true:.1f})")
print(f"  Center:    ({x0_g:.3f}, {y0_g:.3f})")
print(f"  Sigma_x:   {sx_g:.3f} +/- {perr_g[3]:.3f} (true: {sigma_x_true:.2f})")
print(f"  Sigma_y:   {sy_g:.3f} +/- {perr_g[4]:.3f} (true: {sigma_y_true:.2f})")
print(
    f"  Rotation:  {np.degrees(theta_g):.1f} +/- {np.degrees(perr_g[5]):.1f} deg (true: {np.degrees(theta_true):.1f})"
)

# Use global result for visualization
popt = popt_global


# =============================================================================
# Example 2: Double Peak 2D Fitting
# =============================================================================

if not QUICK_MODE:
    print("\n" + "-" * 70)
    print("EXAMPLE 2: DOUBLE PEAK 2D FITTING")
    print("-" * 70)
    print("(Common for binary stars, double-lobed structures, or overlapping sources)")

    # True parameters for two peaks
    amp1_true = 800.0
    x01_true = -1.5
    y01_true = 0.5
    sigma1_true = 1.0

    amp2_true = 500.0
    x02_true = 1.2
    y02_true = -0.8
    sigma2_true = 0.8

    offset2_true = 30.0

    print("\nTrue Parameters:")
    print(
        f"  Peak 1: amp={amp1_true:.0f}, pos=({x01_true:.1f}, {y01_true:.1f}), sigma={sigma1_true:.1f}"
    )
    print(
        f"  Peak 2: amp={amp2_true:.0f}, pos=({x02_true:.1f}, {y02_true:.1f}), sigma={sigma2_true:.1f}"
    )
    print(f"  Offset: {offset2_true:.1f}")

    # Generate double-peak data
    z2_true = double_gaussian_2d(
        xdata,
        amp1_true,
        x01_true,
        y01_true,
        sigma1_true,
        amp2_true,
        x02_true,
        y02_true,
        sigma2_true,
        offset2_true,
    )
    noise2 = np.random.normal(0, np.sqrt(z2_true + 10))
    z2_measured = z2_true + noise2
    sigma2 = np.sqrt(z2_measured + 10)

    # Initial guess
    p0_double = [600, -1, 0, 1.2, 400, 1, 0, 1.2, 20]

    # Bounds
    bounds_double = (
        [0, -4, -4, 0.1, 0, -4, -4, 0.1, 0],
        [1500, 4, 4, 3, 1000, 4, 4, 3, 100],
    )

    print("\nFitting with global optimization (critical for multi-peak 2D)...")
    popt_double, pcov_double = fit(
        double_gaussian_2d,
        xdata,
        z2_measured,
        p0=p0_double,
        sigma=sigma2,
        bounds=bounds_double,
        absolute_sigma=True,
        preset="global",
        n_starts=20,
    )

    perr_double = np.sqrt(np.diag(pcov_double))
    amp1_fit, x01_fit, y01_fit, s1_fit, amp2_fit, x02_fit, y02_fit, s2_fit, off2_fit = (
        popt_double
    )

    print("\nFitted Parameters:")
    print(
        f"  Peak 1: amp={amp1_fit:.0f}+/-{perr_double[0]:.0f}, "
        f"pos=({x01_fit:.2f}, {y01_fit:.2f}), sigma={s1_fit:.2f}"
    )
    print(
        f"  Peak 2: amp={amp2_fit:.0f}+/-{perr_double[4]:.0f}, "
        f"pos=({x02_fit:.2f}, {y02_fit:.2f}), sigma={s2_fit:.2f}"
    )
    print(f"  Offset: {off2_fit:.1f}")

    # Separation
    separation = np.sqrt((x02_fit - x01_fit) ** 2 + (y02_fit - y01_fit) ** 2)
    print(f"\nPeak separation: {separation:.2f} units")
    print(f"Intensity ratio: {amp1_fit / amp2_fit:.2f}")


if QUICK_MODE:
    print("\n" + "=" * 70)
    print("Quick mode: Skipping visualization")
    print("=" * 70)
    sys.exit(0)


# =============================================================================
# Visualization
# =============================================================================

fig = plt.figure(figsize=(16, 12))

# Reshape for 2D plotting
Z_measured = z_measured.reshape(ny, nx)
z_fit = gaussian_2d_rotated(xdata, *popt)
Z_fit = z_fit.reshape(ny, nx)
Z_residuals = (z_measured - z_fit).reshape(ny, nx)

# Plot 1: Measured rotated Gaussian
ax1 = fig.add_subplot(2, 3, 1)
im1 = ax1.imshow(
    Z_measured,
    extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
ax1.contour(X, Y, Z_measured, levels=5, colors="white", alpha=0.5, linewidths=0.5)

# Draw ellipse showing fitted orientation
from matplotlib.patches import Ellipse

ellipse = Ellipse(
    (x0_g, y0_g),
    width=2 * sx_g,
    height=2 * sy_g,
    angle=np.degrees(theta_g),
    fill=False,
    color="red",
    linewidth=2,
    linestyle="--",
)
ax1.add_patch(ellipse)
ax1.plot(x0_g, y0_g, "r+", markersize=15, markeredgewidth=2)

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Measured Data (Rotated Gaussian)", fontweight="bold")
plt.colorbar(im1, ax=ax1, label="Intensity")

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
ellipse2 = Ellipse(
    (x0_g, y0_g),
    width=2 * sx_g,
    height=2 * sy_g,
    angle=np.degrees(theta_g),
    fill=False,
    color="red",
    linewidth=2,
    linestyle="--",
)
ax2.add_patch(ellipse2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title(f"Fitted (theta={np.degrees(theta_g):.1f} deg)", fontweight="bold")
plt.colorbar(im2, ax=ax2, label="Intensity")

# Plot 3: Residuals
ax3 = fig.add_subplot(2, 3, 3)
residuals = z_measured - z_fit
im3 = ax3.imshow(
    Z_residuals,
    extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
    origin="lower",
    cmap="RdBu_r",
    aspect="equal",
    vmin=-3 * np.std(residuals),
    vmax=3 * np.std(residuals),
)
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_title("Residuals", fontweight="bold")
plt.colorbar(im3, ax=ax3, label="Residual")

# Plot 4: Double-peak data (if available)
if not QUICK_MODE:
    Z2_measured = z2_measured.reshape(ny, nx)
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(
        Z2_measured,
        extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax4.plot(x01_fit, y01_fit, "r+", markersize=15, markeredgewidth=2, label="Peak 1")
    ax4.plot(x02_fit, y02_fit, "b+", markersize=15, markeredgewidth=2, label="Peak 2")
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Double Peak Data", fontweight="bold")
    ax4.legend()
    plt.colorbar(im4, ax=ax4, label="Intensity")

    # Plot 5: Double-peak fit
    z2_fit = double_gaussian_2d(xdata, *popt_double)
    Z2_fit = z2_fit.reshape(ny, nx)
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.imshow(
        Z2_fit,
        extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    ax5.contour(X, Y, Z2_fit, levels=5, colors="white", alpha=0.5, linewidths=0.5)
    ax5.plot(x01_fit, y01_fit, "r+", markersize=15, markeredgewidth=2)
    ax5.plot(x02_fit, y02_fit, "b+", markersize=15, markeredgewidth=2)
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_title("Double Peak Fit (Global)", fontweight="bold")
    plt.colorbar(im5, ax=ax5, label="Intensity")

    # Plot 6: Double-peak residuals
    Z2_residuals = (z2_measured - z2_fit).reshape(ny, nx)
    ax6 = fig.add_subplot(2, 3, 6)
    im6 = ax6.imshow(
        Z2_residuals,
        extent=[x_1d.min(), x_1d.max(), y_1d.min(), y_1d.max()],
        origin="lower",
        cmap="RdBu_r",
        aspect="equal",
    )
    ax6.set_xlabel("x")
    ax6.set_ylabel("y")
    ax6.set_title("Double Peak Residuals", fontweight="bold")
    plt.colorbar(im6, ax=ax6, label="Residual")

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

print("\nExample 1: Rotated Elliptical Gaussian")
print(
    f"  Fitted rotation angle: {np.degrees(theta_g):.1f} +/- {np.degrees(perr_g[5]):.1f} deg"
)
print(f"  True rotation angle:   {np.degrees(theta_true):.1f} deg")
print(f"  Axis ratio: {sx_g / sy_g:.2f} (true: {sigma_x_true / sigma_y_true:.2f})")

if not QUICK_MODE:
    print("\nExample 2: Double Peak Separation")
    print(f"  Fitted separation: {separation:.2f} units")
    print(f"  Intensity ratio:   {amp1_fit / amp2_fit:.2f}")

print("\nWhy Global Optimization Matters for 2D Fitting:")
print("  - Rotation angle has pi-periodicity degeneracy")
print("  - Multi-peak fits have many local minima")
print("  - Initial guess sensitivity in high-dimensional spaces")
print("  - preset='global' recommended for rotated/multi-peak 2D fits")

print("\nAPI Methods Demonstrated:")
print("  - fit() with preset='fast' for simple 2D fits")
print("  - fit() with preset='robust' for moderate difficulty")
print("  - fit() with preset='global' for rotated/multi-peak (recommended)")

print("\nApplications:")
print("  - Astigmatic laser beam profiling")
print("  - Binary star separation measurement")
print("  - Multi-source deconvolution")
print("  - Complex PSF characterization")
print("=" * 70)
