"""
Converted from system_identification.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # System Identification: First-Order System Step Response
#
#
# This example demonstrates system identification by fitting a first-order
# transfer function to step response data. This is common in control systems,
# chemical processes, and thermal systems.
#
# Key Concepts:
# - First-order system dynamics
# - Step response fitting
# - Time constant and gain extraction
# - Model validation with residuals
# - Rise time and settling time calculation
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


def first_order_step_response(t, K, tau, t_delay):
    """
    First-order system step response with time delay.

    y(t) = K * (1 - exp(-(t - t_delay)/tau))  for t >= t_delay
    y(t) = 0  for t < t_delay

    Parameters
    ----------
    t : array_like
        Time (seconds)
    K : float
        System gain (steady-state output / input)
    tau : float
        Time constant (seconds)
    t_delay : float
        Time delay (seconds)

    Returns
    -------
    y : array_like
        System output
    """
    # Use jnp.where for conditional (JIT-compatible)
    t_eff = t - t_delay
    response = K * (1 - jnp.exp(-t_eff / tau))
    return jnp.where(t >= t_delay, response, 0.0)


def second_order_step_response(t, K, zeta, omega_n, t_delay):
    """
    Second-order system step response (underdamped).

    For ζ < 1 (underdamped):
    y(t) = K * (1 - exp(-ζωₙt) * cos(ωₐt - φ) / cos(φ))

    where ωₐ = ωₙ√(1-ζ²) and φ = arctan(ζ/√(1-ζ²))

    Parameters
    ----------
    t : array_like
        Time (seconds)
    K : float
        System gain
    zeta : float
        Damping ratio (0 < ζ < 1 for underdamped)
    omega_n : float
        Natural frequency (rad/s)
    t_delay : float
        Time delay (seconds)

    Returns
    -------
    y : array_like
        System output
    """
    t_eff = t - t_delay
    omega_d = omega_n * jnp.sqrt(1 - zeta**2)  # Damped frequency
    phi = jnp.arctan(zeta / jnp.sqrt(1 - zeta**2))

    response = K * (
        1
        - jnp.exp(-zeta * omega_n * t_eff)
        * jnp.cos(omega_d * t_eff - phi)
        / jnp.cos(phi)
    )
    return jnp.where(t >= t_delay, response, 0.0)


# Simulate a thermal system (e.g., heating chamber)
# Input: step from 0 to 100% power at t=0
# Output: temperature rise

# True system parameters
K_true = 80.0  # °C (final temperature rise for 100% power)
tau_true = 15.0  # seconds (time constant)
t_delay_true = 2.0  # seconds (transport delay)

# Time vector (0 to 100 seconds, 200 samples)
time = np.linspace(0, 100, 200)

# True step response
output_true = first_order_step_response(time, K_true, tau_true, t_delay_true)

# Add measurement noise (±1.5°C sensor noise)
noise = np.random.normal(0, 1.5, size=len(time))
output_measured = output_true + noise

# Measurement uncertainties
sigma = 1.5 * np.ones_like(output_measured)


print("=" * 70)
print("SYSTEM IDENTIFICATION: FIRST-ORDER STEP RESPONSE")
print("=" * 70)

# Initial parameter guess
p0 = [75, 12, 1.5]  # K, tau, t_delay

# Parameter bounds (physical constraints)
# K > 0, tau > 0, t_delay >= 0
bounds = ([0, 0.1, 0], [150, 50, 10])

# Fit the model
popt, pcov = curve_fit(
    first_order_step_response,
    time,
    output_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

# Extract fitted parameters
K_fit, tau_fit, t_delay_fit = popt
perr = np.sqrt(np.diag(pcov))
K_err, tau_err, t_delay_err = perr


# Time to reach 63.2% of final value (1 time constant)
t_63 = t_delay_fit + tau_fit

# Rise time (10% to 90% of final value)
t_10 = t_delay_fit + tau_fit * np.log(1 / 0.9)
t_90 = t_delay_fit + tau_fit * np.log(1 / 0.1)
t_rise = t_90 - t_10

# Settling time (2% criterion: within 2% of final value)
# For first-order: t_settle ≈ 4*tau
t_settle_2pct = t_delay_fit + 4 * tau_fit


print("\nFitted Parameters:")
print(f"  K (gain):        {K_fit:.2f} ± {K_err:.2f} °C")
print(f"  τ (time const):  {tau_fit:.2f} ± {tau_err:.2f} s")
print(f"  t_d (delay):     {t_delay_fit:.2f} ± {t_delay_err:.2f} s")

print("\nComparison with True Values:")
print(f"  K:   {K_fit:.2f} vs {K_true:.2f} (true)")
print(f"  τ:   {tau_fit:.2f} vs {tau_true:.2f} (true)")
print(f"  t_d: {t_delay_fit:.2f} vs {t_delay_true:.2f} (true)")

# Check agreement
K_agreement = abs(K_fit - K_true) < K_err
tau_agreement = abs(tau_fit - tau_true) < tau_err
print(f"\n  K within 1σ:  {K_agreement} {'✓' if K_agreement else ''}")
print(f"  τ within 1σ:  {tau_agreement} {'✓' if tau_agreement else ''}")

print("\nDerived System Characteristics:")
print(f"  63.2% rise time:     {t_63:.2f} s")
print(f"  10-90% rise time:    {t_rise:.2f} s")
print(f"  Settling time (2%):  {t_settle_2pct:.2f} s")
print(f"  Bandwidth (-3dB):    {1 / (2 * np.pi * tau_fit):.4f} Hz")

# Goodness of fit
residuals = output_measured - first_order_step_response(time, *popt)
chi_squared = np.sum((residuals / sigma) ** 2)
dof = len(time) - len(popt)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.2f} °C")
print(f"  χ²/dof:  {chi_squared_reduced:.2f} (expect ≈ 1.0)")


print("\n" + "=" * 70)
print("TRANSFER FUNCTION (Laplace Domain)")
print("=" * 70)
print(f"\n  G(s) = {K_fit:.2f} / ({tau_fit:.2f}s + 1) * exp(-{t_delay_fit:.2f}s)")
print(f"\n  Pole location:  s = -{1 / tau_fit:.4f} rad/s")
print(f"  DC gain:        K = {K_fit:.2f}")
print(f"  Time delay:     t_d = {t_delay_fit:.2f} s")


print("\n" + "=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

# Calculate R² (coefficient of determination)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((output_measured - np.mean(output_measured)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"  R²: {r_squared:.4f} (closer to 1 is better)")
print(f"  RMSE/Range: {rmse / (output_measured.max() - output_measured.min()):.2%}")

# Check for systematic residuals
from scipy import stats

# Durbin-Watson statistic for autocorrelation
dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
print(f"  Durbin-Watson: {dw:.2f} (2.0 = no autocorrelation)")

# Check residual normality
_, p_value_normality = stats.normaltest(residuals)
print(
    f"  Residuals normal? p = {p_value_normality:.3f} "
    + f"({'Yes' if p_value_normality > 0.05 else 'No'} at α=0.05)"
)


fig = plt.figure(figsize=(16, 12))

# Plot 1: Step response fit
ax1 = plt.subplot(3, 2, 1)
ax1.errorbar(
    time,
    output_measured,
    yerr=sigma,
    fmt="o",
    alpha=0.4,
    markersize=3,
    capsize=0,
    label="Measured data",
)
t_fine = np.linspace(0, 100, 500)
ax1.plot(
    t_fine,
    first_order_step_response(t_fine, K_true, tau_true, t_delay_true),
    "r--",
    linewidth=2,
    label="True system",
    alpha=0.7,
)
ax1.plot(
    t_fine,
    first_order_step_response(t_fine, *popt),
    "g-",
    linewidth=2.5,
    label="Fitted model",
)

# Mark key points
ax1.axhline(K_fit * 0.632, color="blue", linestyle=":", alpha=0.5)
ax1.axvline(
    t_63, color="blue", linestyle=":", alpha=0.5, label=f"63.2% at t={t_63:.1f}s"
)
ax1.axhline(
    K_fit, color="gray", linestyle="--", alpha=0.5, label=f"Steady-state: {K_fit:.1f}°C"
)

ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Temperature Rise (°C)", fontsize=12)
ax1.set_title("Step Response: Model Fit", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs time
ax2 = plt.subplot(3, 2, 2)
ax2.plot(time, residuals, "o", alpha=0.5, markersize=4)
ax2.axhline(0, color="r", linestyle="--", linewidth=2)
ax2.axhline(2 * sigma[0], color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-2 * sigma[0], color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Residual (°C)")
ax2.set_title("Fit Residuals vs Time")
ax2.grid(True, alpha=0.3)

# Plot 3: Normalized residuals histogram
ax3 = plt.subplot(3, 2, 3)
normalized_res = residuals / sigma
ax3.hist(normalized_res, bins=20, alpha=0.7, edgecolor="black", density=True)
# Overlay expected normal distribution
x_norm = np.linspace(-4, 4, 100)
ax3.plot(
    x_norm,
    np.exp(-(x_norm**2) / 2) / np.sqrt(2 * np.pi),
    "r-",
    linewidth=2,
    label="N(0,1)",
)
ax3.set_xlabel("Normalized Residual (σ)")
ax3.set_ylabel("Probability Density")
ax3.set_title("Residual Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Q-Q plot (quantile-quantile)
ax4 = plt.subplot(3, 2, 4)
stats.probplot(normalized_res, dist="norm", plot=ax4)
ax4.set_title("Q-Q Plot (Normality Check)")
ax4.grid(True, alpha=0.3)

# Plot 5: Rise time analysis
ax5 = plt.subplot(3, 2, 5)
mask_rise = (time >= 0) & (time <= 50)
ax5.plot(
    time[mask_rise],
    output_measured[mask_rise],
    "o",
    alpha=0.5,
    markersize=4,
    label="Data",
)
ax5.plot(
    t_fine[:250],
    first_order_step_response(t_fine[:250], *popt),
    "g-",
    linewidth=2,
    label="Fitted model",
)

# Mark rise time points
ax5.axhline(0.1 * K_fit, color="blue", linestyle=":", linewidth=1.5)
ax5.axhline(0.9 * K_fit, color="blue", linestyle=":", linewidth=1.5)
ax5.axvline(t_10, color="blue", linestyle=":", linewidth=1.5)
ax5.axvline(t_90, color="blue", linestyle=":", linewidth=1.5)
ax5.axhspan(0.1 * K_fit, 0.9 * K_fit, alpha=0.1, color="blue")
ax5.annotate(
    f"Rise time\n{t_rise:.2f}s",
    xy=((t_10 + t_90) / 2, 0.5 * K_fit),
    ha="center",
    fontsize=11,
    fontweight="bold",
)

ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Temperature Rise (°C)")
ax5.set_title("Rise Time Analysis (10-90%)")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Autocorrelation of residuals
ax6 = plt.subplot(3, 2, 6)
# Compute autocorrelation
max_lag = min(50, len(residuals) // 4)
autocorr = np.correlate(
    residuals - np.mean(residuals), residuals - np.mean(residuals), mode="full"
)
autocorr = autocorr[len(autocorr) // 2 :]
autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize

lags = np.arange(max_lag)
ax6.stem(lags, autocorr, basefmt=" ")
ax6.axhline(0, color="black", linewidth=0.8)
# 95% confidence interval
conf_interval = 1.96 / np.sqrt(len(residuals))
ax6.axhline(conf_interval, color="r", linestyle="--", alpha=0.5, label="95% CI")
ax6.axhline(-conf_interval, color="r", linestyle="--", alpha=0.5)
ax6.set_xlabel("Lag")
ax6.set_ylabel("Autocorrelation")
ax6.set_title("Residual Autocorrelation")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("system_identification.png", dpi=150)
print("\n✅ Plot saved as 'system_identification.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "system_identification"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("System successfully identified as first-order with delay:")
print(
    f"\n  Transfer function:  G(s) = {K_fit:.2f}/({tau_fit:.2f}s + 1) * e^(-{t_delay_fit:.2f}s)"
)
print(f"\n  Time constant:      τ = {tau_fit:.2f} ± {tau_err:.2f} s")
print(f"  Steady-state gain:  K = {K_fit:.2f} ± {K_err:.2f} °C")
print(f"  Time delay:         t_d = {t_delay_fit:.2f} ± {t_delay_err:.2f} s")
print(f"\n  Rise time (10-90%): {t_rise:.2f} s")
print(f"  Settling time (2%): {t_settle_2pct:.2f} s")
print(f"  Bandwidth:          {1 / (2 * np.pi * tau_fit):.4f} Hz")
print(f"\n  Model quality:      R² = {r_squared:.4f}, RMSE = {rmse:.2f}°C")
print("\nThis example demonstrates:")
print("  ✓ First-order system identification from step response")
print("  ✓ Transfer function parameter extraction")
print("  ✓ Rise time and settling time calculation")
print("  ✓ Model validation with statistical tests")
print("  ✓ Residual autocorrelation analysis")
print("  ✓ System dynamics characterization")
print("=" * 70)
