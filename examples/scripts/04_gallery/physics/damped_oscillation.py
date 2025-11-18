"""
Converted from damped_oscillation.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Damped Oscillation: Pendulum Damping Analysis
#
#
# This example demonstrates fitting damped harmonic oscillator data to extract
# the damping coefficient and natural frequency. We use realistic pendulum data
# and compare fitted values with theoretical predictions.
#
# Key Concepts:
# - Damped harmonic oscillator model
# - Exponential envelope extraction
# - Quality factor (Q) calculation
# - Frequency and damping time constants
# - Comparison with theoretical models
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


def damped_oscillator(t, A0, gamma, omega, phi):
    """
    Damped harmonic oscillator model.

    x(t) = A0 * exp(-γt) * cos(ωt + φ)

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
    """
    Calculate quality factor Q.

    Q = ω / (2γ)

    Higher Q means less damping (more oscillations before decay).

    Parameters
    ----------
    gamma : float
        Damping coefficient (1/s)
    omega : float
        Angular frequency (rad/s)

    Returns
    -------
    Q : float
        Quality factor (dimensionless)
    """
    return omega / (2 * gamma)


def damping_time(gamma):
    """
    Calculate damping time constant (1/e decay time).

    τ = 1/γ

    Parameters
    ----------
    gamma : float
        Damping coefficient (1/s)

    Returns
    -------
    tau : float
        Damping time (seconds)
    """
    return 1 / gamma


# True parameters for a lightly damped pendulum
# (e.g., 1m length pendulum with small air resistance)
A0_true = 15.0  # Initial amplitude (degrees)
gamma_true = 0.05  # Damping coefficient (1/s) - light damping
omega0_true = 2 * np.pi / 2.0  # Natural frequency (rad/s) - period ~2 seconds
phi_true = 0.0  # Phase offset (starts at max displacement)

# Time points (0 to 60 seconds, 300 measurements)
time = np.linspace(0, 60, 300)

# True oscillation
displacement_true = damped_oscillator(time, A0_true, gamma_true, omega0_true, phi_true)

# Add measurement noise (realistic for optical tracking: ±0.2 degrees)
noise = np.random.normal(0, 0.2, size=len(time))
displacement_measured = displacement_true + noise

# Measurement uncertainties
sigma = 0.2 * np.ones_like(time)  # Constant uncertainty


print("=" * 70)
print("DAMPED OSCILLATION: PENDULUM DAMPING ANALYSIS")
print("=" * 70)

# Initial parameter guess
# (from visual inspection of the data)
p0 = [14, 0.04, 3.0, 0.0]  # A0, gamma, omega, phi

# Bounds (physical constraints)
# A0 > 0, gamma > 0, omega > 0, -pi < phi < pi
bounds = ([0, 0, 0, -np.pi], [20, 0.5, 10, np.pi])

# Fit the model
popt, pcov = curve_fit(
    damped_oscillator,
    time,
    displacement_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

# Extract fitted parameters
A0_fit, gamma_fit, omega_fit, phi_fit = popt
perr = np.sqrt(np.diag(pcov))
A0_err, gamma_err, omega_err, phi_err = perr

# Calculate derived quantities
Q_fit = quality_factor(gamma_fit, omega_fit)
tau_fit = damping_time(gamma_fit)
period_fit = 2 * np.pi / omega_fit
frequency_fit = omega_fit / (2 * np.pi)

# Uncertainties in derived quantities
Q_err = Q_fit * np.sqrt((gamma_err / gamma_fit) ** 2 + (omega_err / omega_fit) ** 2)
tau_err = tau_fit * (gamma_err / gamma_fit)
period_err = period_fit * (omega_err / omega_fit)


print("\nFitted Parameters:")
print(f"  A0 (initial amplitude): {A0_fit:.3f} ± {A0_err:.3f} degrees")
print(f"  γ (damping coefficient): {gamma_fit:.5f} ± {gamma_err:.5f} s⁻¹")
print(f"  ω (angular frequency):   {omega_fit:.4f} ± {omega_err:.4f} rad/s")
print(f"  φ (phase offset):        {phi_fit:.4f} ± {phi_err:.4f} rad")

print("\nDerived Quantities:")
print(f"  Frequency (f):    {frequency_fit:.4f} Hz")
print(f"  Period (T):       {period_fit:.3f} ± {period_err:.3f} seconds")
print(f"  Damping time (τ): {tau_fit:.2f} ± {tau_err:.2f} seconds")
print(f"  Quality factor (Q): {Q_fit:.1f} ± {Q_err:.1f}")

print("\nComparison with True Values:")
print(f"  A0:    {A0_fit:.3f} vs {A0_true:.3f} (true)")
print(f"  γ:     {gamma_fit:.5f} vs {gamma_true:.5f} (true)")
print(f"  ω:     {omega_fit:.4f} vs {omega0_true:.4f} (true)")
print(f"  φ:     {phi_fit:.4f} vs {phi_true:.4f} (true)")

# Check agreement
gamma_agreement = abs(gamma_fit - gamma_true) < gamma_err
omega_agreement = abs(omega_fit - omega0_true) < omega_err
print(
    f"\n  γ within 1σ: {gamma_agreement} ✓"
    if gamma_agreement
    else f"\n  γ within 1σ: {gamma_agreement}"
)
print(
    f"  ω within 1σ: {omega_agreement} ✓"
    if omega_agreement
    else f"  ω within 1σ: {omega_agreement}"
)

# Goodness of fit
chi_squared = np.sum(
    ((displacement_measured - damped_oscillator(time, *popt)) / sigma) ** 2
)
dof = len(time) - len(popt)
chi_squared_reduced = chi_squared / dof
print("\nGoodness of Fit:")
print(f"  χ²/dof = {chi_squared_reduced:.2f} (expect ≈ 1.0)")


print("\n" + "=" * 70)
print("PHYSICAL INTERPRETATION")
print("=" * 70)

# Number of oscillations before 1/e decay
n_oscillations_decay = tau_fit * frequency_fit
print(f"Number of oscillations before 1/e decay: {n_oscillations_decay:.1f}")

# Amplitude after 30 seconds
amp_30s = A0_fit * np.exp(-gamma_fit * 30)
print(
    f"Amplitude after 30 seconds: {amp_30s:.2f} degrees "
    + f"({100 * amp_30s / A0_fit:.1f}% of initial)"
)

# Pendulum length (from period, assuming simple pendulum)
g = 9.81  # m/s² (gravitational acceleration)
length_estimated = g / omega_fit**2
print(f"\nEstimated pendulum length: {length_estimated:.3f} meters")
print("(Assuming simple pendulum: T = 2π√(L/g))")

# Damping regime classification
critical_damping = 2 * omega_fit
damping_ratio = gamma_fit / critical_damping
print("\nDamping classification:")
print(f"  Damping ratio (ζ): {damping_ratio:.4f}")
if damping_ratio < 0.1:
    print("  → Lightly damped (ζ < 0.1) ✓")
elif damping_ratio < 1:
    print("  → Underdamped (ζ < 1)")
elif damping_ratio == 1:
    print("  → Critically damped (ζ = 1)")
else:
    print("  → Overdamped (ζ > 1)")


fig = plt.figure(figsize=(16, 12))

# Main plot: data and fit
ax1 = plt.subplot(3, 2, 1)
ax1.plot(
    time, displacement_measured, "o", alpha=0.4, markersize=3, label="Measured data"
)
t_fine = np.linspace(0, 60, 1000)
ax1.plot(
    t_fine, damped_oscillator(t_fine, *popt), "r-", linewidth=2, label="Fitted model"
)
# Plot envelope
envelope_upper = A0_fit * np.exp(-gamma_fit * t_fine)
envelope_lower = -envelope_upper
ax1.plot(
    t_fine, envelope_upper, "g--", linewidth=1.5, label=f"Envelope (τ = {tau_fit:.1f}s)"
)
ax1.plot(t_fine, envelope_lower, "g--", linewidth=1.5)
ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Displacement (degrees)", fontsize=12)
ax1.set_title("Damped Oscillation", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Semi-log plot (shows exponential decay)
ax2 = plt.subplot(3, 2, 2)
# Plot amplitude envelope
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

# Residuals
ax3 = plt.subplot(3, 2, 3)
residuals = displacement_measured - damped_oscillator(time, *popt)
normalized_residuals = residuals / sigma
ax3.plot(time, normalized_residuals, ".", alpha=0.4, markersize=3)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Normalized Residuals (σ)")
ax3.set_title("Fit Residuals")
ax3.grid(True, alpha=0.3)

# Phase space plot (velocity vs displacement)
ax4 = plt.subplot(3, 2, 4)
# Compute numerical derivative for velocity
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

# Frequency spectrum (FFT)
ax5 = plt.subplot(3, 2, 5)
from scipy import signal

frequencies, power = signal.periodogram(
    displacement_measured, fs=1 / (time[1] - time[0])
)
# Only plot positive frequencies
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

# Zoomed view of first few oscillations
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
plt.savefig("damped_oscillation.png", dpi=150)
print("\n✅ Plot saved as 'damped_oscillation.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "damped_oscillation"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Fitted damping coefficient: γ = {gamma_fit:.5f} ± {gamma_err:.5f} s⁻¹")
print(f"Fitted angular frequency:   ω = {omega_fit:.4f} ± {omega_err:.4f} rad/s")
print(f"Quality factor:             Q = {Q_fit:.1f} ± {Q_err:.1f}")
print(f"Damping time:               τ = {tau_fit:.2f} ± {tau_err:.2f} s")
print(f"Period:                     T = {period_fit:.3f} ± {period_err:.3f} s")
print(f"\nEstimated pendulum length: {length_estimated:.3f} m")
print(f"Damping regime: Lightly damped (ζ = {damping_ratio:.4f})")
print("\nThis example demonstrates:")
print("  ✓ Damped harmonic oscillator fitting")
print("  ✓ Extraction of damping coefficient and natural frequency")
print("  ✓ Quality factor calculation")
print("  ✓ Phase space analysis")
print("  ✓ Frequency domain analysis (FFT)")
print("  ✓ Physical parameter estimation from fitted values")
print("=" * 70)
