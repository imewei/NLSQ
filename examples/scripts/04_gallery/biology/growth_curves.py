"""
Converted from growth_curves.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Bacterial Growth Curves: Logistic Growth Model
#
#
# This example demonstrates fitting bacterial growth curves using the logistic
# growth model. We extract growth rate, lag time, and carrying capacity from
# optical density (OD) measurements.
#
# Key Concepts:
# - Logistic growth model (Verhulst equation)
# - Growth rate (μ) determination
# - Lag phase, exponential phase, stationary phase
# - Doubling time calculation
# - Modified Gompertz model for lag phase
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


def logistic_growth(t, N0, K, r):
    """
    Logistic growth model (Verhulst equation).

    N(t) = K / (1 + ((K - N0)/N0) * exp(-r*t))

    Or equivalently:
    N(t) = K / (1 + A * exp(-r*t))  where A = (K - N0)/N0

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

    N(t) = A * exp(-exp(μ*e/A * (λ - t) + 1))

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

    N(t) = N0 * exp(μ*t)

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
time = np.linspace(0, 24, 49)

# True growth parameters
N0_true = 0.01  # Initial OD600
K_true = 1.2  # Carrying capacity (max OD600)
r_true = 0.8  # Growth rate (per hour)

# Generate true growth curve
OD_true = logistic_growth(time, N0_true, K_true, r_true)

# Add measurement noise (realistic for plate reader)
# Low OD: higher relative noise, High OD: constant absolute noise
noise = np.random.normal(0, 0.02 + 0.03 * OD_true, size=len(time))
OD_measured = np.maximum(OD_true + noise, 0.001)  # OD can't be negative

# Measurement uncertainties
sigma = 0.02 + 0.03 * OD_measured


print("=" * 70)
print("BACTERIAL GROWTH CURVES: LOGISTIC MODEL FITTING")
print("=" * 70)

# Initial parameter guess
p0 = [0.015, 1.0, 0.7]  # N0, K, r

# Parameter bounds
bounds = (
    [0, 0, 0],  # All positive
    [0.1, 3.0, 2.0],  # Reasonable upper limits
)

# Fit the model
popt, pcov = curve_fit(
    logistic_growth,
    time,
    OD_measured,
    p0=p0,
    sigma=sigma,
    bounds=bounds,
    absolute_sigma=True,
)

N0_fit, K_fit, r_fit = popt
perr = np.sqrt(np.diag(pcov))
N0_err, K_err, r_err = perr


# Doubling time
doubling_time = np.log(2) / r_fit

# Time to reach mid-exponential phase (N = K/2)
t_mid = np.log((K_fit - N0_fit) / N0_fit) / r_fit

# Maximum growth rate (at inflection point, N = K/2)
max_growth_rate = r_fit * K_fit / 4  # dN/dt at N = K/2


print("\nFitted Parameters:")
print(f"  N0 (initial OD):    {N0_fit:.4f} ± {N0_err:.4f}")
print(f"  K (carrying cap.):  {K_fit:.3f} ± {K_err:.3f}")
print(f"  r (growth rate):    {r_fit:.3f} ± {r_err:.3f} hr⁻¹")

print("\nTrue Values:")
print(f"  N0:  {N0_true:.4f}")
print(f"  K:   {K_true:.3f}")
print(f"  r:   {r_true:.3f} hr⁻¹")

print("\nDerived Growth Characteristics:")
print(f"  Doubling time (t_d):      {doubling_time:.2f} hours")
print(f"  Time to mid-exp (K/2):    {t_mid:.2f} hours")
print(f"  Max growth rate:          {max_growth_rate:.4f} OD/hr")
print(f"  Generation time:          {60 * doubling_time:.1f} minutes")

# Goodness of fit
residuals = OD_measured - logistic_growth(time, *popt)
chi_squared = np.sum((residuals / sigma) ** 2)
dof = len(time) - len(popt)
chi_squared_reduced = chi_squared / dof
rmse = np.sqrt(np.mean(residuals**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse:.4f} OD")
print(f"  χ²/dof:  {chi_squared_reduced:.2f}")


print("\n" + "-" * 70)
print("EXPONENTIAL PHASE ANALYSIS")
print("-" * 70)

# Select exponential phase (typically OD 0.1 to 0.6)
mask_exp = (OD_measured > 0.1) & (OD_measured < 0.6)

if np.sum(mask_exp) > 5:
    # Fit exponential model to log-transformed data
    # ln(N) = ln(N0) + μ*t
    def linear_log(t, ln_N0, mu):
        return ln_N0 + mu * t

    ln_OD = np.log(OD_measured[mask_exp])
    t_exp = time[mask_exp]

    popt_exp, pcov_exp = curve_fit(linear_log, t_exp, ln_OD, p0=[np.log(0.1), 0.8])

    ln_N0_exp, mu_exp = popt_exp
    N0_exp = np.exp(ln_N0_exp)
    mu_err = np.sqrt(pcov_exp[1, 1])

    doubling_time_exp = np.log(2) / mu_exp

    print("Exponential phase parameters (from log fit):")
    print(f"  μ (specific growth rate): {mu_exp:.3f} ± {mu_err:.3f} hr⁻¹")
    print(f"  Doubling time:            {doubling_time_exp:.2f} hours")
    print(f"  N0 (extrapolated):        {N0_exp:.4f}")
    print(f"\nCompare with logistic r:    {r_fit:.3f} hr⁻¹")
    print("(Should be similar in exponential phase)")


print("\n" + "-" * 70)
print("GROWTH PHASE CLASSIFICATION")
print("-" * 70)

# Classify each time point
phases = []
for t, od in zip(time, OD_measured, strict=False):
    if od < 0.05:
        phases.append("Lag")
    elif od < 0.9 * K_fit:
        phases.append("Exponential")
    else:
        phases.append("Stationary")

# Find phase transitions
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
    t_fine, logistic_growth(t_fine, *popt), "g-", linewidth=2.5, label="Fitted logistic"
)

# Mark key points
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
ax1.set_title("Bacterial Growth Curve", fontsize=14, fontweight="bold")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Plot 2: Semi-log plot
ax2 = plt.subplot(3, 2, 2)
ax2.semilogy(time, OD_measured, "o", markersize=6, alpha=0.6, label="Measured OD")
ax2.semilogy(
    t_fine, logistic_growth(t_fine, *popt), "g-", linewidth=2.5, label="Fitted logistic"
)

# Show exponential fit
if np.sum(mask_exp) > 5:
    ax2.semilogy(
        t_fine,
        exponential_phase(t_fine, N0_exp, mu_exp),
        "b--",
        linewidth=2,
        label=f"Exponential (μ={mu_exp:.2f})",
    )

# Shade growth phases
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
# Analytical derivative of logistic equation
# dN/dt = r*N*(1 - N/K)
N_vals = logistic_growth(t_fine, *popt)
growth_rate_analytical = r_fit * N_vals * (1 - N_vals / K_fit)

ax3.plot(
    t_fine, growth_rate_analytical, "g-", linewidth=2.5, label="Growth rate (dN/dt)"
)

# Mark maximum
max_gr_idx = np.argmax(growth_rate_analytical)
ax3.plot(
    t_fine[max_gr_idx],
    growth_rate_analytical[max_gr_idx],
    "ro",
    markersize=10,
    label=f"Max at t={t_fine[max_gr_idx]:.1f}h, N={N_vals[max_gr_idx]:.2f}",
)

ax3.axvline(t_mid, color="orange", linestyle="--", alpha=0.5)
ax3.set_xlabel("Time (hours)")
ax3.set_ylabel("Growth Rate dN/dt (OD/hr)")
ax3.set_title("Instantaneous Growth Rate")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Specific growth rate
ax4 = plt.subplot(3, 2, 4)
# μ(t) = (1/N) * dN/dt = r*(1 - N/K)
specific_growth_rate = growth_rate_analytical / N_vals

ax4.plot(t_fine, specific_growth_rate, "g-", linewidth=2.5)
ax4.axhline(
    r_fit,
    color="blue",
    linestyle="--",
    linewidth=2,
    label=f"Intrinsic rate r = {r_fit:.3f} hr⁻¹",
)
ax4.axhline(r_fit / 2, color="orange", linestyle=":", alpha=0.5)

ax4.set_xlabel("Time (hours)")
ax4.set_ylabel("Specific Growth Rate μ (hr⁻¹)")
ax4.set_title("Specific Growth Rate (1/N × dN/dt)")
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
ax5.set_ylabel("Normalized Residuals (σ)")
ax5.set_title("Fit Residuals")
ax5.grid(True, alpha=0.3)

# Plot 6: Phase diagram (N vs dN/dt)
ax6 = plt.subplot(3, 2, 6)
ax6.plot(
    N_vals, growth_rate_analytical, "g-", linewidth=2.5, label="Logistic trajectory"
)

# Theoretical maximum (parabola)
N_theory = np.linspace(0, K_fit, 100)
dNdt_theory = r_fit * N_theory * (1 - N_theory / K_fit)
ax6.plot(N_theory, dNdt_theory, "b--", linewidth=2, label="Theoretical (r*N*(1-N/K))")

# Mark current measurements
OD_measured_sorted_idx = np.argsort(OD_measured)
dNdt_measured = np.gradient(
    OD_measured[OD_measured_sorted_idx], time[OD_measured_sorted_idx]
)
ax6.plot(
    OD_measured[OD_measured_sorted_idx],
    dNdt_measured,
    "o",
    alpha=0.3,
    markersize=4,
    label="Measured (numerical)",
)

ax6.set_xlabel("Population N (OD600)")
ax6.set_ylabel("Growth Rate dN/dt (OD/hr)")
ax6.set_title("Phase Diagram")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("growth_curves.png", dpi=150)
print("\n✅ Plot saved as 'growth_curves.png'")
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "growth_curves"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Bacterial growth successfully characterized:")
print("\n  Growth model: Logistic (Verhulst equation)")
print(f"  Intrinsic growth rate (r): {r_fit:.3f} ± {r_err:.3f} hr⁻¹")
print(
    f"  Doubling time:             {doubling_time:.2f} hours ({60 * doubling_time:.0f} min)"
)
print(f"  Carrying capacity (K):     {K_fit:.3f} ± {K_err:.3f} OD600")
print(f"  Initial density (N0):      {N0_fit:.4f} ± {N0_err:.4f} OD600")
print("\nGrowth phases:")
print(f"  Lag phase:         {lag_duration:.1f} hours")
print(f"  Exponential phase: {exp_duration:.1f} hours")
print(f"  Stationary phase:  after {t_stationary:.1f} hours")
print(f"\nModel quality: χ²/dof = {chi_squared_reduced:.2f}, RMSE = {rmse:.4f}")
print("\nThis example demonstrates:")
print("  ✓ Logistic growth model fitting")
print("  ✓ Growth rate and doubling time extraction")
print("  ✓ Growth phase identification (lag, exponential, stationary)")
print("  ✓ Specific growth rate analysis")
print("  ✓ Phase diagram visualization")
print("  ✓ Semi-log transformation for exponential phase")
print("=" * 70)
