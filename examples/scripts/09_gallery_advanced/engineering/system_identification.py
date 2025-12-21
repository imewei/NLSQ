"""
Advanced System Identification with fit() API and GlobalOptimizationConfig.

This example demonstrates system identification by fitting a first-order
transfer function to step response data using NLSQ's advanced fit() API
and global optimization for robust parameter extraction.

Compared to 04_gallery/engineering/system_identification.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- First-order system dynamics
- Step response fitting
- Time constant and gain extraction
- Rise time and settling time calculation
- Global optimization for robust parameter estimation
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from nlsq import GlobalOptimizationConfig, fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"
FIT_KWARGS = {"max_nfev": 200} if QUICK else {}

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
    t_eff = t - t_delay
    response = K * (1 - jnp.exp(-t_eff / tau))
    return jnp.where(t >= t_delay, response, 0.0)


def second_order_step_response(t, K, zeta, omega_n, t_delay):
    """
    Second-order system step response (underdamped).

    Parameters
    ----------
    t : array_like
        Time (seconds)
    K : float
        System gain
    zeta : float
        Damping ratio (0 < zeta < 1 for underdamped)
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
    omega_d = omega_n * jnp.sqrt(1 - zeta**2)
    phi = jnp.arctan(zeta / jnp.sqrt(1 - zeta**2))

    response = K * (
        1
        - jnp.exp(-zeta * omega_n * t_eff)
        * jnp.cos(omega_d * t_eff - phi)
        / jnp.cos(phi)
    )
    return jnp.where(t >= t_delay, response, 0.0)


# Simulate a thermal system (e.g., heating chamber)
# True system parameters
K_true = 80.0  # C (final temperature rise for 100% power)
tau_true = 15.0  # seconds (time constant)
t_delay_true = 2.0  # seconds (transport delay)

# Time vector
time = np.linspace(0, 100, 120 if QUICK else 200)

# True step response
output_true = first_order_step_response(time, K_true, tau_true, t_delay_true)

# Add measurement noise
noise = np.random.normal(0, 1.5, size=len(time))
output_measured = output_true + noise

# Measurement uncertainties
sigma = 1.5 * np.ones_like(output_measured)


print("=" * 70)
print("SYSTEM IDENTIFICATION: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# First-Order System Fitting
# =============================================================================
print("\n" + "-" * 70)
print("FIRST-ORDER SYSTEM FITTING")
print("-" * 70)

# Initial parameter guess
p0 = [75, 12, 1.5]  # K, tau, t_delay

# Parameter bounds
bounds = ([0, 0.1, 0], [150, 50, 10])

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt, pcov = fit(
    first_order_step_response,
    time,
    output_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="robust",
    **FIT_KWARGS,
)

K_fit, tau_fit, t_delay_fit = popt
perr = np.sqrt(np.diag(pcov))
K_err, tau_err, t_delay_err = perr

print(f"  K (gain):        {K_fit:.2f} +/- {K_err:.2f} C")
print(f"  tau (time const): {tau_fit:.2f} +/- {tau_err:.2f} s")
print(f"  t_d (delay):     {t_delay_fit:.2f} +/- {t_delay_err:.2f} s")

if QUICK:
    print("\n⏩ Quick mode: skipping global/custom fits and extended analysis.")
    sys.exit(0)

# Method 2: fit() with 'global' preset
print("\nMethod 2: fit() with 'global' preset")
popt_global, pcov_global = fit(
    first_order_step_response,
    time,
    output_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="global",
)

K_g, tau_g, t_delay_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  K (gain):        {K_g:.2f} +/- {perr_g[0]:.2f} C")
print(f"  tau (time const): {tau_g:.2f} +/- {perr_g[1]:.2f} s")
print(f"  t_d (delay):     {t_delay_g:.2f} +/- {perr_g[2]:.2f} s")

# Method 3: GlobalOptimizationConfig with custom settings
print("\nMethod 3: GlobalOptimizationConfig with custom settings")
popt_custom, pcov_custom = fit(
    first_order_step_response,
    time,
    output_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    multistart=True,
    n_starts=15,
    sampler="lhs",
)

K_c, tau_c, t_delay_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))

print(f"  K (gain):        {K_c:.2f} +/- {perr_c[0]:.2f} C")
print(f"  tau (time const): {tau_c:.2f} +/- {perr_c[1]:.2f} s")
print(f"  t_d (delay):     {t_delay_c:.2f} +/- {perr_c[2]:.2f} s")


# Use robust preset results for analysis
K_fit, tau_fit, t_delay_fit = popt
perr = np.sqrt(np.diag(pcov))
K_err, tau_err, t_delay_err = perr

# Derived quantities
t_63 = t_delay_fit + tau_fit
t_10 = t_delay_fit + tau_fit * np.log(1 / 0.9)
t_90 = t_delay_fit + tau_fit * np.log(1 / 0.1)
t_rise = t_90 - t_10
t_settle_2pct = t_delay_fit + 4 * tau_fit


print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  K (gain):        {K_fit:.2f} +/- {K_err:.2f} C")
print(f"  tau (time const): {tau_fit:.2f} +/- {tau_err:.2f} s")
print(f"  t_d (delay):     {t_delay_fit:.2f} +/- {t_delay_err:.2f} s")

print("\nComparison with True Values:")
print(f"  K:   {K_fit:.2f} vs {K_true:.2f} (true)")
print(f"  tau:  {tau_fit:.2f} vs {tau_true:.2f} (true)")
print(f"  t_d: {t_delay_fit:.2f} vs {t_delay_true:.2f} (true)")

K_agreement = abs(K_fit - K_true) < K_err
tau_agreement = abs(tau_fit - tau_true) < tau_err
print(f"\n  K within 1sigma: {K_agreement}")
print(f"  tau within 1sigma: {tau_agreement}")

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
print(f"  RMSE:    {rmse:.2f} C")
print(f"  chi^2/dof:  {chi_squared_reduced:.2f} (expect ~1.0)")


# =============================================================================
# Transfer Function
# =============================================================================
print("\n" + "=" * 70)
print("TRANSFER FUNCTION (Laplace Domain)")
print("=" * 70)
print(f"\n  G(s) = {K_fit:.2f} / ({tau_fit:.2f}s + 1) * exp(-{t_delay_fit:.2f}s)")
print(f"\n  Pole location:  s = -{1 / tau_fit:.4f} rad/s")
print(f"  DC gain:        K = {K_fit:.2f}")
print(f"  Time delay:     t_d = {t_delay_fit:.2f} s")


# =============================================================================
# Model Validation
# =============================================================================
print("\n" + "=" * 70)
print("MODEL VALIDATION")
print("=" * 70)

ss_res = np.sum(residuals**2)
ss_tot = np.sum((output_measured - np.mean(output_measured)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"  R^2: {r_squared:.4f} (closer to 1 is better)")
print(f"  RMSE/Range: {rmse / (output_measured.max() - output_measured.min()):.2%}")

# Durbin-Watson statistic
dw = np.sum(np.diff(residuals) ** 2) / np.sum(residuals**2)
print(f"  Durbin-Watson: {dw:.2f} (2.0 = no autocorrelation)")

# Check residual normality
_, p_value_normality = stats.normaltest(residuals)
print(
    f"  Residuals normal? p = {p_value_normality:.3f} "
    + f"({'Yes' if p_value_normality > 0.05 else 'No'} at alpha=0.05)"
)


# =============================================================================
# Visualization
# =============================================================================
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
    label="Fitted model (robust)",
)

ax1.axhline(K_fit * 0.632, color="blue", linestyle=":", alpha=0.5)
ax1.axvline(
    t_63, color="blue", linestyle=":", alpha=0.5, label=f"63.2% at t={t_63:.1f}s"
)
ax1.axhline(
    K_fit, color="gray", linestyle="--", alpha=0.5, label=f"Steady-state: {K_fit:.1f}C"
)

ax1.set_xlabel("Time (s)", fontsize=12)
ax1.set_ylabel("Temperature Rise (C)", fontsize=12)
ax1.set_title("Step Response - fit() API", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals vs time
ax2 = plt.subplot(3, 2, 2)
ax2.plot(time, residuals, "o", alpha=0.5, markersize=4)
ax2.axhline(0, color="r", linestyle="--", linewidth=2)
ax2.axhline(2 * sigma[0], color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-2 * sigma[0], color="gray", linestyle=":", alpha=0.5)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Residual (C)")
ax2.set_title("Fit Residuals vs Time")
ax2.grid(True, alpha=0.3)

# Plot 3: Normalized residuals histogram
ax3 = plt.subplot(3, 2, 3)
normalized_res = residuals / sigma
ax3.hist(normalized_res, bins=20, alpha=0.7, edgecolor="black", density=True)
x_norm = np.linspace(-4, 4, 100)
ax3.plot(
    x_norm,
    np.exp(-(x_norm**2) / 2) / np.sqrt(2 * np.pi),
    "r-",
    linewidth=2,
    label="N(0,1)",
)

ax3.set_xlabel("Normalized Residual (sigma)")
ax3.set_ylabel("Probability Density")
ax3.set_title("Residual Distribution")
ax3.legend()
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Q-Q plot
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
ax5.set_ylabel("Temperature Rise (C)")
ax5.set_title("Rise Time Analysis (10-90%)")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Autocorrelation of residuals
ax6 = plt.subplot(3, 2, 6)
max_lag = min(50, len(residuals) // 4)
autocorr = np.correlate(
    residuals - np.mean(residuals), residuals - np.mean(residuals), mode="full"
)
autocorr = autocorr[len(autocorr) // 2 :]
autocorr = autocorr[:max_lag] / autocorr[0]

lags = np.arange(max_lag)
ax6.stem(lags, autocorr, basefmt=" ")
ax6.axhline(0, color="black", linewidth=0.8)
conf_interval = 1.96 / np.sqrt(len(residuals))
ax6.axhline(conf_interval, color="r", linestyle="--", alpha=0.5, label="95% CI")
ax6.axhline(-conf_interval, color="r", linestyle="--", alpha=0.5)

ax6.set_xlabel("Lag")
ax6.set_ylabel("Autocorrelation")
ax6.set_title("Residual Autocorrelation")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "system_identification"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("System successfully identified using fit() API:")
print(
    f"\n  Transfer function:  G(s) = {K_fit:.2f}/({tau_fit:.2f}s + 1) * e^(-{t_delay_fit:.2f}s)"
)
print(f"\n  Time constant:      tau = {tau_fit:.2f} +/- {tau_err:.2f} s")
print(f"  Steady-state gain:  K = {K_fit:.2f} +/- {K_err:.2f} C")
print(f"  Time delay:         t_d = {t_delay_fit:.2f} +/- {t_delay_err:.2f} s")
print(f"\n  Rise time (10-90%): {t_rise:.2f} s")
print(f"  Settling time (2%): {t_settle_2pct:.2f} s")
print(f"  Bandwidth:          {1 / (2 * np.pi * tau_fit):.4f} Hz")
print(f"\n  Model quality:      R^2 = {r_squared:.4f}, RMSE = {rmse:.2f}C")
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print("  - fit() with preset='global' (20 multi-starts)")
print("  - fit() with GlobalOptimizationConfig (custom settings)")
print("\nThis example demonstrates:")
print("  - First-order system identification with fit() API")
print("  - Global optimization for robust parameter estimation")
print("  - Transfer function parameter extraction")
print("  - Rise time and settling time calculation")
print("  - Model validation with statistical tests")
print("=" * 70)
