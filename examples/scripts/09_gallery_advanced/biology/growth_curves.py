"""
Advanced Bacterial Growth Curve Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting bacterial growth curves using the logistic growth
model with NLSQ's advanced fit() API and global optimization capabilities for
robust growth rate and carrying capacity determination.

Compared to 04_gallery/biology/growth_curves.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Logistic growth model (Verhulst equation)
- Growth rate (r) determination
- Lag phase, exponential phase, stationary phase
- Doubling time calculation
- Global optimization for robust parameter estimation
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


def logistic_growth(t, N0, K, r):
    """
    Logistic growth model (Verhulst equation).

    N(t) = K / (1 + ((K - N0)/N0) * exp(-r*t))

    Parameters
    ----------
    t : array_like
        Time (hours)
    N0 : float
        Initial population (OD600)
    K : float
        Carrying capacity (maximum OD600)
    r : float
        Intrinsic growth rate (per hour)

    Returns
    -------
    N : array_like
        Population (OD600) at time t
    """
    A = (K - N0) / N0
    return K / (1 + A * jnp.exp(-r * t))


def gompertz_model(t, A, mu, lambda_lag):
    """
    Modified Gompertz model for bacterial growth with lag phase.

    N(t) = A * exp(-exp(mu*e/A * (lambda - t) + 1))

    Parameters
    ----------
    t : array_like
        Time (hours)
    A : float
        Asymptotic maximum (OD600)
    mu : float
        Maximum specific growth rate (per hour)
    lambda_lag : float
        Lag time (hours)

    Returns
    -------
    N : array_like
        Population (OD600) at time t
    """
    e = np.e
    exponent = mu * e / A * (lambda_lag - t) + 1
    return A * jnp.exp(-jnp.exp(exponent))


def exponential_phase(t, N0, mu):
    """
    Exponential growth (no lag, no saturation).

    N(t) = N0 * exp(mu*t)

    Parameters
    ----------
    t : array_like
        Time (hours)
    N0 : float
        Initial population (OD600)
    mu : float
        Specific growth rate (per hour)

    Returns
    -------
    N : array_like
        Population (OD600)
    """
    return N0 * jnp.exp(mu * t)


# Time points (0 to 24 hours, every 30 minutes)
time = np.linspace(0, 24, 25 if QUICK else 49)

# True growth parameters
N0_true = 0.01  # Initial OD600
K_true = 1.2  # Carrying capacity (max OD600)
r_true = 0.8  # Growth rate (per hour)

# Generate true growth curve
OD_true = logistic_growth(time, N0_true, K_true, r_true)

# Add measurement noise (realistic for plate reader)
noise = np.random.normal(0, 0.02 + 0.03 * OD_true, size=len(time))
OD_measured = np.maximum(OD_true + noise, 0.001)  # OD can't be negative

# Measurement uncertainties
sigma = 0.02 + 0.03 * OD_measured


print("=" * 70)
print("BACTERIAL GROWTH CURVES: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# Initial parameter guess
p0 = [0.015, 1.0, 0.7]  # N0, K, r

# Parameter bounds
bounds = (
    [0, 0, 0],  # All positive
    [0.1, 3.0, 2.0],  # Reasonable upper limits
)

# =============================================================================
# Method 1: Using fit() with 'robust' preset
# =============================================================================
print("\n" + "-" * 70)
print("Method 1: fit() with 'robust' preset")
print("-" * 70)

popt_robust, pcov_robust = fit(
    logistic_growth,
    time,
    OD_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="robust",
)

N0_fit, K_fit, r_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
N0_err, K_err, r_err = perr

print(f"  N0 = {N0_fit:.4f} +/- {N0_err:.4f} (true: {N0_true})")
print(f"  K = {K_fit:.3f} +/- {K_err:.3f} (true: {K_true})")
print(f"  r = {r_fit:.3f} +/- {r_err:.3f} hr^-1 (true: {r_true})")

if QUICK:
    print("⏩ Quick mode: skipping extended comparisons and plots.")
    sys.exit(0)


# =============================================================================
# Method 2: Using fit() with 'global' preset for thorough search
# =============================================================================
print("\n" + "-" * 70)
print("Method 2: fit() with 'global' preset (20 starts)")
print("-" * 70)

popt_global, pcov_global = fit(
    logistic_growth,
    time,
    OD_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    preset="global",
    n_starts=6 if QUICK else 20,
)

N0_g, K_g, r_g = popt_global
perr_g = np.sqrt(np.diag(pcov_global))

print(f"  N0 = {N0_g:.4f} +/- {perr_g[0]:.4f}")
print(f"  K = {K_g:.3f} +/- {perr_g[1]:.3f}")
print(f"  r = {r_g:.3f} +/- {perr_g[2]:.3f} hr^-1")


# =============================================================================
# Method 3: Using GlobalOptimizationConfig with custom settings
# =============================================================================
print("\n" + "-" * 70)
print("Method 3: GlobalOptimizationConfig with custom settings")
print("-" * 70)

# Create custom global optimization configuration
global_config = GlobalOptimizationConfig(
    n_starts=6 if QUICK else 15,
    sampler="lhs",
    center_on_p0=True,
    scale_factor=1.0,
)

# Use explicit multi-start parameters with fit()
popt_custom, pcov_custom = fit(
    logistic_growth,
    time,
    OD_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
    multistart=True,
    n_starts=6 if QUICK else 15,
    sampler="lhs",
)

N0_c, K_c, r_c = popt_custom
perr_c = np.sqrt(np.diag(pcov_custom))

print(f"  N0 = {N0_c:.4f} +/- {perr_c[0]:.4f}")
print(f"  K = {K_c:.3f} +/- {perr_c[1]:.3f}")
print(f"  r = {r_c:.3f} +/- {perr_c[2]:.3f} hr^-1")


# Use robust preset results for analysis
N0_fit, K_fit, r_fit = popt_robust
perr = np.sqrt(np.diag(pcov_robust))
N0_err, K_err, r_err = perr

# Derived quantities
doubling_time = np.log(2) / r_fit
t_mid = np.log((K_fit - N0_fit) / N0_fit) / r_fit
max_growth_rate = r_fit * K_fit / 4

print("\n" + "=" * 70)
print("FITTED PARAMETERS (Robust Preset)")
print("=" * 70)
print(f"  N0 (initial OD):    {N0_fit:.4f} +/- {N0_err:.4f}")
print(f"  K (carrying cap.):  {K_fit:.3f} +/- {K_err:.3f}")
print(f"  r (growth rate):    {r_fit:.3f} +/- {r_err:.3f} hr^-1")

print("\nTrue Values:")
print(f"  N0:  {N0_true:.4f}")
print(f"  K:   {K_true:.3f}")
print(f"  r:   {r_true:.3f} hr^-1")

print("\nDerived Growth Characteristics:")
print(f"  Doubling time (t_d):      {doubling_time:.2f} hours")
print(f"  Time to mid-exp (K/2):    {t_mid:.2f} hours")
print(f"  Max growth rate:          {max_growth_rate:.4f} OD/hr")
print(f"  Generation time:          {60 * doubling_time:.1f} minutes")

# Goodness of fit
residuals = OD_measured - logistic_growth(time, *popt_robust)
chi_squared = np.sum((residuals / sigma) ** 2)
dof = len(time) - len(popt_robust)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.4f} OD")
print(f"  chi^2/dof:  {chi_squared_reduced:.2f}")


# =============================================================================
# Exponential Phase Analysis
# =============================================================================
print("\n" + "-" * 70)
print("EXPONENTIAL PHASE ANALYSIS")
print("-" * 70)

# Select exponential phase (typically OD 0.1 to 0.6)
mask_exp = (OD_measured > 0.1) & (OD_measured < 0.6)

if np.sum(mask_exp) > 5:

    def linear_log(t, ln_N0, mu):
        return ln_N0 + mu * t

    ln_OD = np.log(OD_measured[mask_exp])
    t_exp = time[mask_exp]

    popt_exp, pcov_exp = fit(linear_log, t_exp, ln_OD, p0=[np.log(0.1), 0.8])

    ln_N0_exp, mu_exp = popt_exp
    N0_exp = np.exp(ln_N0_exp)
    mu_err = np.sqrt(pcov_exp[1, 1])

    doubling_time_exp = np.log(2) / mu_exp

    print("Exponential phase parameters (from log fit):")
    print(f"  mu (specific growth rate): {mu_exp:.3f} +/- {mu_err:.3f} hr^-1")
    print(f"  Doubling time:            {doubling_time_exp:.2f} hours")
    print(f"  N0 (extrapolated):        {N0_exp:.4f}")
    print(f"\nCompare with logistic r:    {r_fit:.3f} hr^-1")


# =============================================================================
# Growth Phase Classification
# =============================================================================
print("\n" + "-" * 70)
print("GROWTH PHASE CLASSIFICATION")
print("-" * 70)

phases = []
for t, od in zip(time, OD_measured, strict=False):
    if od < 0.05:
        phases.append("Lag")
    elif od < 0.9 * K_fit:
        phases.append("Exponential")
    else:
        phases.append("Stationary")

lag_end = np.where(np.array(phases) != "Lag")[0]
if len(lag_end) > 0:
    lag_duration = time[lag_end[0]]
else:
    lag_duration = 0

exp_end = np.where(np.array(phases) == "Stationary")[0]
if len(exp_end) > 0:
    exp_duration = time[exp_end[0]] - lag_duration
    t_stationary = time[exp_end[0]]
else:
    exp_duration = time[-1] - lag_duration
    t_stationary = time[-1]

print("Phase durations:")
print(f"  Lag phase:         ~{lag_duration:.1f} hours")
print(f"  Exponential phase: ~{exp_duration:.1f} hours")
print(f"  Stationary phase:  starts at ~{t_stationary:.1f} hours")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Growth curve (linear scale)
ax1 = plt.subplot(3, 2, 1)
ax1.errorbar(
    time,
    OD_measured,
    yerr=sigma,
    fmt="o",
    capsize=3,
    markersize=6,
    alpha=0.6,
    label="Measured OD",
)

t_fine = np.linspace(0, 24, 200)
ax1.plot(
    t_fine,
    logistic_growth(t_fine, N0_true, K_true, r_true),
    "r--",
    linewidth=2,
    label="True curve",
    alpha=0.7,
)
ax1.plot(
    t_fine,
    logistic_growth(t_fine, *popt_robust),
    "g-",
    linewidth=2.5,
    label="Fitted (robust)",
)

ax1.axhline(
    K_fit,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label=f"Carrying capacity K = {K_fit:.2f}",
)
ax1.axhline(K_fit / 2, color="orange", linestyle=":", alpha=0.5)
ax1.axvline(
    t_mid, color="orange", linestyle=":", alpha=0.5, label=f"Mid-exp (t = {t_mid:.1f}h)"
)

ax1.set_xlabel("Time (hours)", fontsize=12)
ax1.set_ylabel("OD600", fontsize=12)
ax1.set_title("Bacterial Growth Curve - fit() API", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: Semi-log plot
ax2 = plt.subplot(3, 2, 2)
ax2.semilogy(time, OD_measured, "o", markersize=6, alpha=0.6, label="Measured OD")
ax2.semilogy(
    t_fine,
    logistic_growth(t_fine, *popt_robust),
    "g-",
    linewidth=2.5,
    label="Fitted logistic",
)

if np.sum(mask_exp) > 5:
    ax2.semilogy(
        t_fine,
        exponential_phase(t_fine, N0_exp, mu_exp),
        "b--",
        linewidth=2,
        label=f"Exponential (mu={mu_exp:.2f})",
    )

ax2.axvspan(0, lag_duration, alpha=0.1, color="red", label="Lag phase")
ax2.axvspan(lag_duration, t_stationary, alpha=0.1, color="green")
ax2.axvspan(t_stationary, 24, alpha=0.1, color="blue")

ax2.set_xlabel("Time (hours)")
ax2.set_ylabel("OD600 (log scale)")
ax2.set_title("Semi-Log Plot (Shows Exponential as Linear)")
ax2.legend()
ax2.grid(True, alpha=0.3, which="both")

# Plot 3: Growth rate (dN/dt)
ax3 = plt.subplot(3, 2, 3)
N_vals = logistic_growth(t_fine, *popt_robust)
growth_rate_analytical = r_fit * N_vals * (1 - N_vals / K_fit)

ax3.plot(
    t_fine, growth_rate_analytical, "g-", linewidth=2.5, label="Growth rate (dN/dt)"
)

max_gr_idx = np.argmax(growth_rate_analytical)
ax3.plot(
    t_fine[max_gr_idx],
    growth_rate_analytical[max_gr_idx],
    "ro",
    markersize=10,
    label=f"Max at t={t_fine[max_gr_idx]:.1f}h",
)

ax3.axvline(t_mid, color="orange", linestyle="--", alpha=0.5)
ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Growth Rate dN/dt (OD/hr)")
ax3.set_title("Instantaneous Growth Rate")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Specific growth rate
ax4 = plt.subplot(3, 2, 4)
specific_growth_rate = growth_rate_analytical / N_vals

ax4.plot(t_fine, specific_growth_rate, "g-", linewidth=2.5)
ax4.axhline(
    r_fit,
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Intrinsic rate r = {r_fit:.3f} hr^-1",
)
ax4.axhline(r_fit / 2, color="orange", linestyle=":", alpha=0.5)

ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Specific Growth Rate mu (hr^-1)")
ax4.set_title("Specific Growth Rate (1/N * dN/dt)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Residuals
ax5 = plt.subplot(3, 2, 5)
normalized_residuals = residuals / sigma
ax5.plot(time, normalized_residuals, "o", markersize=6, alpha=0.7)
ax5.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax5.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax5.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax5.set_xlabel("Time (hours)")
ax5.set_ylabel("Normalized Residuals (sigma)")
ax5.set_title("Fit Residuals")
ax5.grid(True, alpha=0.3)

# Plot 6: API comparison
ax6 = plt.subplot(3, 2, 6)
ax6.axis("off")

api_table = [
    ["Method", "N0", "K", "r (hr^-1)"],
    ["-" * 20, "-" * 8, "-" * 6, "-" * 10],
    ["fit() 'robust'", f"{N0_fit:.4f}", f"{K_fit:.3f}", f"{r_fit:.3f}"],
    ["fit() 'global'", f"{N0_g:.4f}", f"{K_g:.3f}", f"{r_g:.3f}"],
    ["fit() custom", f"{N0_c:.4f}", f"{K_c:.3f}", f"{r_c:.3f}"],
    ["", "", "", ""],
    ["True values", f"{N0_true:.4f}", f"{K_true:.3f}", f"{r_true:.3f}"],
    ["", "", "", ""],
    ["Derived values:", "", "", ""],
    ["-" * 35, "", "", ""],
    [f"  Doubling time: {doubling_time:.2f} h", "", "", ""],
    [f"  Generation: {60 * doubling_time:.1f} min", "", "", ""],
    [f"  Max rate: {max_growth_rate:.4f} OD/h", "", "", ""],
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
fig_dir = Path(__file__).parent / "figures" / "growth_curves"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Bacterial growth successfully characterized using fit() API:")
print("\n  Growth model: Logistic (Verhulst equation)")
print(f"  Intrinsic growth rate (r): {r_fit:.3f} +/- {r_err:.3f} hr^-1")
print(
    f"  Doubling time:             {doubling_time:.2f} hours ({60 * doubling_time:.0f} min)"
)
print(f"  Carrying capacity (K):     {K_fit:.3f} +/- {K_err:.3f} OD600")
print(f"  Initial density (N0):      {N0_fit:.4f} +/- {N0_err:.4f} OD600")
print("\nGrowth phases:")
print(f"  Lag phase:         {lag_duration:.1f} hours")
print(f"  Exponential phase: {exp_duration:.1f} hours")
print(f"  Stationary phase:  after {t_stationary:.1f} hours")
print(f"\nModel quality: chi^2/dof = {chi_squared_reduced:.2f}, RMSE = {rmse:.4f}")
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print("  - fit() with preset='global' (20 multi-starts)")
print("  - fit() with GlobalOptimizationConfig (custom settings)")
print("\nThis example demonstrates:")
print("  - Logistic growth model fitting with fit() API")
print("  - Global optimization for robust parameter estimation")
print("  - Growth phase identification and analysis")
print("  - Specific growth rate analysis")
print("=" * 70)
