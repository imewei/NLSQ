"""
Advanced Damped Oscillation Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting damped harmonic oscillator data to extract
the damping coefficient and natural frequency using NLSQ's advanced fit() API
and global optimization for robust parameter estimation.

Compared to 04_gallery/physics/damped_oscillation.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Damped harmonic oscillator model
- Exponential envelope extraction
- Quality factor (Q) calculation
- Frequency and damping time constants
- Global optimization for robust parameter estimation
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from nlsq import GlobalOptimizationConfig, fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK else {}

# Set random seed
np.random.seed(42)


def damped_oscillator(t, A0, gamma, omega, phi):
    """
    Damped harmonic oscillator model.

    x(t) = A0 * exp(-gamma*t) * cos(omega*t + phi)

    Parameters
    ----------
    t : array_like
        Time (seconds)
    A0 : float
        Initial amplitude (meters or degrees)
    gamma : float
        Damping coefficient (1/seconds)
    omega : float
        Angular frequency (rad/s)
    phi : float
        Phase offset (radians)

    Returns
    -------
    x : array_like
        Displacement at time t
    """
    return A0 * jnp.exp(-gamma * t) * jnp.cos(omega * t + phi)


def quality_factor(gamma, omega):
    """Calculate quality factor Q = omega / (2*gamma)"""
    return omega / (2 * gamma)


def damping_time(gamma):
    """Calculate damping time constant tau = 1/gamma"""
    return 1 / gamma


# True parameters for a lightly damped pendulum
A0_true = 15.0  # Initial amplitude (degrees)
gamma_true = 0.05  # Damping coefficient (1/s)
omega0_true = 2 * np.pi / 2.0  # Natural frequency (rad/s)
phi_true = 0.0  # Phase offset

# Time points
time = np.linspace(0, 60, 120 if QUICK else 300)

# True oscillation
displacement_true = damped_oscillator(time, A0_true, gamma_true, omega0_true, phi_true)

# Add measurement noise
noise = np.random.normal(0, 0.2, size=len(time))
displacement_measured = displacement_true + noise

# Measurement uncertainties
sigma = 0.2 * np.ones_like(time)


print("=" * 70)
print("DAMPED OSCILLATION: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# Damped Oscillator Fitting
# =============================================================================
print("\n" + "-" * 70)
print("DAMPED OSCILLATOR FITTING")
print("-" * 70)

# Initial parameter guess
p0 = [14, 0.04, 3.0, 0.0]  # A0, gamma, omega, phi

# Parameter bounds
bounds = ([0, 0, 0, -np.pi], [20, 0.5, 10, np.pi])

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt, pcov = fit(
    damped_oscillator,
    time,
    displacement_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="robust",
    **FIT_KWARGS,
)

A0_fit, gamma_fit, omega_fit, phi_fit = popt
perr = np.sqrt(np.diag(pcov))
A0_err, gamma_err, omega_err, phi_err = perr

print(f"  A0 (amplitude): {A0_fit:.3f} +/- {A0_err:.3f} degrees")
print(f"  gamma (damping): {gamma_fit:.5f} +/- {gamma_err:.5f} s^-1")
print(f"  omega (frequency): {omega_fit:.4f} +/- {omega_err:.4f} rad/s")
print(f"  phi (phase): {phi_fit:.4f} +/- {phi_err:.4f} rad")

if QUICK:
    print("\n⏩ Quick mode: skipping global/custom fits and extended plots.")
    sys.exit(0)

# Method 2: fit() with 'global' preset
global_starts = 20
print(f"\nMethod 2: fit() with 'global' preset ({global_starts} starts)")
popt_global, pcov_global = fit(
    damped_oscillator,
    time,
    displacement_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="global",
    n_starts=global_starts,
)

A0_g, gamma_g, omega_g, phi_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  A0: {A0_g:.3f} +/- {perr_g[0]:.3f}")
print(f"  gamma: {gamma_g:.5f} +/- {perr_g[1]:.5f}")
print(f"  omega: {omega_g:.4f} +/- {perr_g[2]:.4f}")
print(f"  phi: {phi_g:.4f} +/- {perr_g[3]:.4f}")

# Method 3: GlobalOptimizationConfig with custom settings
print("\nMethod 3: GlobalOptimizationConfig with custom settings")
popt_custom, pcov_custom = fit(
    damped_oscillator,
    time,
    displacement_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    multistart=True,
    n_starts=15,
    sampler="lhs",
)

A0_c, gamma_c, omega_c, phi_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))

print(f"  A0: {A0_c:.3f} +/- {perr_c[0]:.3f}")
print(f"  gamma: {gamma_c:.5f} +/- {perr_c[1]:.5f}")
print(f"  omega: {omega_c:.4f} +/- {perr_c[2]:.4f}")
print(f"  phi: {phi_c:.4f} +/- {perr_c[3]:.4f}")


# Use robust preset results for analysis
A0_fit, gamma_fit, omega_fit, phi_fit = popt
perr = np.sqrt(np.diag(pcov))
A0_err, gamma_err, omega_err, phi_err = perr

# Derived quantities
Q_fit = quality_factor(gamma_fit, omega_fit)
tau_fit = damping_time(gamma_fit)
period_fit = 2 * np.pi / omega_fit
frequency_fit = omega_fit / (2 * np.pi)

# Uncertainties in derived quantities
Q_err = Q_fit * np.sqrt((gamma_err / gamma_fit) ** 2 + (omega_err / omega_fit) ** 2)
tau_err = tau_fit * (gamma_err / gamma_fit)
period_err = period_fit * (omega_err / omega_fit)


print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  A0 (initial amplitude): {A0_fit:.3f} +/- {A0_err:.3f} degrees")
print(f"  gamma (damping coeff): {gamma_fit:.5f} +/- {gamma_err:.5f} s^-1")
print(f"  omega (angular freq):   {omega_fit:.4f} +/- {omega_err:.4f} rad/s")
print(f"  phi (phase offset):     {phi_fit:.4f} +/- {phi_err:.4f} rad")

print("\nDerived Quantities:")
print(f"  Frequency (f):    {frequency_fit:.4f} Hz")
print(f"  Period (T):       {period_fit:.3f} +/- {period_err:.3f} seconds")
print(f"  Damping time (tau): {tau_fit:.2f} +/- {tau_err:.2f} seconds")
print(f"  Quality factor (Q): {Q_fit:.1f} +/- {Q_err:.1f}")

print("\nComparison with True Values:")
print(f"  A0:    {A0_fit:.3f} vs {A0_true:.3f} (true)")
print(f"  gamma: {gamma_fit:.5f} vs {gamma_true:.5f} (true)")
print(f"  omega: {omega_fit:.4f} vs {omega0_true:.4f} (true)")
print(f"  phi:   {phi_fit:.4f} vs {phi_true:.4f} (true)")

gamma_agreement = abs(gamma_fit - gamma_true) < gamma_err
omega_agreement = abs(omega_fit - omega0_true) < omega_err
print(f"\n  gamma within 1sigma: {gamma_agreement}")
print(f"  omega within 1sigma: {omega_agreement}")

# Goodness of fit
chi_squared = np.sum(
    ((displacement_measured - damped_oscillator(time, *popt)) / sigma) ** 2
)
dof = len(time) - len(popt)
chi_squared_reduced = chi_squared / dof

print("\nGoodness of Fit:")
print(f"  chi^2/dof = {chi_squared_reduced:.2f} (expect ~1.0)")


# =============================================================================
# Physical Interpretation
# =============================================================================
print("\n" + "=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

n_oscillations_decay = tau_fit * frequency_fit
print(f"Number of oscillations before 1/e decay: {n_oscillations_decay:.1f}")

amp_30s = A0_fit * np.exp(-gamma_fit * 30)
print(
    f"Amplitude after 30 seconds: {amp_30s:.2f} degrees "
    + f"({100 * amp_30s / A0_fit:.1f}% of initial)"
)

g = 9.81  # m/s^2
length_estimated = g / omega_fit**2
print(f"\nEstimated pendulum length: {length_estimated:.3f} meters")
print("(Assuming simple pendulum: T = 2*pi*sqrt(L/g))")

critical_damping = 2 * omega_fit
damping_ratio = gamma_fit / critical_damping
print("\nDamping classification:")
print(f"  Damping ratio (zeta): {damping_ratio:.4f}")
if damping_ratio < 0.1:
    print("  -> Lightly damped (zeta < 0.1)")
elif damping_ratio < 1:
    print("  -> Underdamped (zeta < 1)")
elif damping_ratio == 1:
    print("  -> Critically damped (zeta = 1)")
else:
    print("  -> Overdamped (zeta > 1)")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Data and fit
ax1 = plt.subplot(3, 2, 1)
ax1.plot(
    time, displacement_measured, "o", alpha=0.4, markersize=3, label="Measured data"
)
t_fine = np.linspace(0, 60, 1000)
ax1.plot(
    t_fine,
    damped_oscillator(t_fine, *popt),
    "r-",
    linewidth=2,
    label="Fitted model (robust)",
)

envelope_upper = A0_fit * np.exp(-gamma_fit * t_fine)
envelope_lower = -envelope_upper
ax1.plot(
    t_fine,
    envelope_upper,
    "g--",
    linewidth=1.5,
    label=f"Envelope (tau = {tau_fit:.1f}s)",
)
ax1.plot(t_fine, envelope_lower, "g--", linewidth=1.5)

ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Displacement (degrees)", fontsize=12)
ax1.set_title("Damped Oscillation - fit() API", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Semi-log plot
ax2 = plt.subplot(3, 2, 2)
peaks_time = []
peaks_amp = []
for i in range(1, len(time) - 1):
    if (
        displacement_measured[i] > displacement_measured[i - 1]
        and displacement_measured[i] > displacement_measured[i + 1]
        and displacement_measured[i] > 0
    ):
        peaks_time.append(time[i])
        peaks_amp.append(displacement_measured[i])

if peaks_time:
    ax2.semilogy(peaks_time, peaks_amp, "o", markersize=6, label="Peak amplitudes")
ax2.semilogy(
    t_fine,
    A0_fit * np.exp(-gamma_fit * t_fine),
    "r-",
    linewidth=2,
    label="Fitted envelope",
)
ax2.axhline(
    A0_fit / np.e,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"1/e decay (t = {tau_fit:.1f}s)",
)
ax2.axvline(tau_fit, color="orange", linestyle="--", linewidth=2)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (degrees, log scale)")
ax2.set_title("Exponential Decay of Amplitude")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = plt.subplot(3, 2, 3)
residuals = displacement_measured - damped_oscillator(time, *popt)
normalized_residuals = residuals / sigma
ax3.plot(time, normalized_residuals, ".", alpha=0.4, markersize=3)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)

ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Normalized Residuals (sigma)")
ax3.set_title("Fit Residuals")
ax3.grid(True, alpha=0.3)

# Plot 4: Phase space plot
ax4 = plt.subplot(3, 2, 4)
velocity_measured = np.gradient(displacement_measured, time)
velocity_fit = np.gradient(damped_oscillator(time, *popt), time)
ax4.plot(
    displacement_measured,
    velocity_measured,
    ".",
    alpha=0.3,
    markersize=3,
    label="Measured",
)
ax4.plot(
    damped_oscillator(time, *popt), velocity_fit, "r-", linewidth=1.5, label="Fitted"
)

ax4.set_xlabel("Displacement (degrees)")
ax4.set_ylabel("Velocity (degrees/s)")
ax4.set_title("Phase Space Portrait")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Frequency spectrum
ax5 = plt.subplot(3, 2, 5)
frequencies, power = signal.periodogram(
    displacement_measured, fs=1 / (time[1] - time[0])
)
mask = frequencies > 0
ax5.semilogy(frequencies[mask], power[mask], "b-", linewidth=1.5)
ax5.axvline(
    frequency_fit,
    color="r",
    linestyle="--",
    linewidth=2,
    label=f"Fitted frequency: {frequency_fit:.3f} Hz",
)

ax5.set_xlabel("Frequency (Hz)")
ax5.set_ylabel("Power Spectral Density")
ax5.set_title("Frequency Spectrum (FFT)")
ax5.set_xlim([0, 2])
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Zoomed view
ax6 = plt.subplot(3, 2, 6)
mask_zoom = time < 10
ax6.plot(
    time[mask_zoom],
    displacement_measured[mask_zoom],
    "o",
    alpha=0.6,
    markersize=4,
    label="Data",
)
t_zoom = np.linspace(0, 10, 500)
ax6.plot(t_zoom, damped_oscillator(t_zoom, *popt), "r-", linewidth=2, label="Fit")

ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Displacement (degrees)")
ax6.set_title("First 10 Seconds (Detail)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "damped_oscillation"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Damped oscillation parameters determined using fit() API:")
print(
    f"\n  Fitted damping coefficient: gamma = {gamma_fit:.5f} +/- {gamma_err:.5f} s^-1"
)
print(
    f"  Fitted angular frequency:   omega = {omega_fit:.4f} +/- {omega_err:.4f} rad/s"
)
print(f"  Quality factor:             Q = {Q_fit:.1f} +/- {Q_err:.1f}")
print(f"  Damping time:               tau = {tau_fit:.2f} +/- {tau_err:.2f} s")
print(f"  Period:                     T = {period_fit:.3f} +/- {period_err:.3f} s")
print(f"\n  Estimated pendulum length: {length_estimated:.3f} m")
print(f"  Damping regime: Lightly damped (zeta = {damping_ratio:.4f})")
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print(f"  - fit() with preset='global' ({global_starts} multi-starts)")
print(f"  - fit() with GlobalOptimizationConfig ({6 if QUICK else 15} custom starts)")
if QUICK:
    print("⏩ Quick mode: skipping extended plotting.")
    sys.exit(0)
print("\nThis example demonstrates:")
print("  - Damped harmonic oscillator fitting with fit() API")
print("  - Global optimization for robust parameter estimation")
print("  - Quality factor calculation")
print("  - Phase space analysis")
print("  - Frequency domain analysis (FFT)")
print("=" * 70)
