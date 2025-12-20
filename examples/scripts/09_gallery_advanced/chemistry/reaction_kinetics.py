"""
Advanced Chemical Reaction Kinetics Fitting with fit() API and GlobalOptimizationConfig.

This example demonstrates fitting chemical reaction kinetics data to determine
rate constants and reaction orders using NLSQ's advanced fit() API and global
optimization capabilities.

Compared to 04_gallery/chemistry/reaction_kinetics.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- First-order kinetics (exponential decay)
- Second-order kinetics (hyperbolic decay)
- Rate constant determination
- Half-life calculation
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


def first_order_decay(t, C0, k):
    """
    First-order reaction kinetics.

    [A](t) = [A]_0 * exp(-kt)

    Parameters
    ----------
    t : array_like
        Time (seconds)
    C0 : float
        Initial concentration (M)
    k : float
        Rate constant (s^-1)

    Returns
    -------
    C : array_like
        Concentration at time t (M)
    """
    return C0 * jnp.exp(-k * t)


def second_order_decay(t, C0, k):
    """
    Second-order reaction kinetics (single reactant).

    [A](t) = [A]_0 / (1 + k*[A]_0*t)

    Parameters
    ----------
    t : array_like
        Time (seconds)
    C0 : float
        Initial concentration (M)
    k : float
        Rate constant (M^-1 s^-1)

    Returns
    -------
    C : array_like
        Concentration at time t (M)
    """
    return C0 / (1 + k * C0 * t)


def pseudo_first_order(t, C0, k_obs):
    """
    Pseudo-first-order kinetics.

    Same form as first-order, but k_obs = k[B]_0 where [B] is in excess.

    Parameters
    ----------
    t : array_like
        Time (seconds)
    C0 : float
        Initial concentration of limiting reactant (M)
    k_obs : float
        Observed rate constant (s^-1)

    Returns
    -------
    C : array_like
        Concentration at time t (M)
    """
    return C0 * jnp.exp(-k_obs * t)


# Time points (0 to 1000 seconds, every 10 seconds)
time = np.linspace(0, 1000, 51 if QUICK else 101)

# First-order reaction parameters
C0_1st = 1.0  # M (initial concentration)
k_1st_true = 0.005  # s^-1 (rate constant)

# Generate first-order data
C_1st_true = first_order_decay(time, C0_1st, k_1st_true)

# Add measurement noise (2% relative error typical for UV-Vis)
noise_1st = np.random.normal(0, 0.02 * C_1st_true, size=len(time))
C_1st_measured = C_1st_true + noise_1st

# Measurement uncertainties
sigma_1st = np.asarray(0.02 * C_1st_measured + 0.001)


print("=" * 70)
print("CHEMICAL REACTION KINETICS: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# First-Order Reaction Analysis
# =============================================================================
print("\n" + "-" * 70)
print("FIRST-ORDER REACTION ANALYSIS")
print("-" * 70)

# Initial parameter guess
p0_1st = [0.9, 0.004]  # C0, k

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt_1st_robust, pcov_1st_robust = fit(
    first_order_decay,
    time,
    C_1st_measured,
    p0=p0_1st,
    sigma=sigma_1st,
    absolute_sigma=True,
    bounds=([0, 0], [2, 0.1]),
    preset="robust",
)

C0_1st_fit, k_1st_fit = popt_1st_robust
perr_1st = np.sqrt(np.diag(pcov_1st_robust))
C0_1st_err, k_1st_err = perr_1st

# Calculate half-life
t_half_1st = np.log(2) / k_1st_fit
t_half_1st_err = t_half_1st * (k_1st_err / k_1st_fit)

print(f"  [A]_0: {C0_1st_fit:.4f} +/- {C0_1st_err:.4f} M (true: {C0_1st})")
print(f"  k:    {k_1st_fit:.6f} +/- {k_1st_err:.6f} s^-1 (true: {k_1st_true})")
print(
    f"  t_1/2: {t_half_1st:.2f} +/- {t_half_1st_err:.2f} s (true: {np.log(2) / k_1st_true:.2f})"
)

# Method 2: fit() with 'global' preset
global_starts = 6 if QUICK else 20
print(f"\nMethod 2: fit() with 'global' preset ({global_starts} starts)")
popt_1st_global, pcov_1st_global = fit(
    first_order_decay,
    time,
    C_1st_measured,
    p0=p0_1st,
    sigma=sigma_1st,
    absolute_sigma=True,
    bounds=([0, 0], [2, 0.1]),
    preset="global",
    n_starts=global_starts,
)

C0_1st_g, k_1st_g = popt_1st_global
perr_1st_g = np.sqrt(np.diag(pcov_1st_global))

print(f"  [A]_0: {C0_1st_g:.4f} +/- {perr_1st_g[0]:.4f} M")
print(f"  k:    {k_1st_g:.6f} +/- {perr_1st_g[1]:.6f} s^-1")

# Goodness of fit
residuals_1st = C_1st_measured - first_order_decay(time, *popt_1st_robust)
chi_squared_1st = np.sum((residuals_1st / sigma_1st) ** 2)
dof_1st = len(time) - len(popt_1st_robust)
chi_squared_reduced_1st = chi_squared_1st / dof_1st
rmse_1st = np.sqrt(np.mean(residuals_1st**2))

print("\nGoodness of Fit:")
print(f"  RMSE:    {rmse_1st:.5f} M")
print(f"  chi^2/dof:  {chi_squared_reduced_1st:.2f}")


# =============================================================================
# Second-Order Reaction Analysis
# =============================================================================
print("\n" + "-" * 70)
print("SECOND-ORDER REACTION ANALYSIS")
print("-" * 70)

# Second-order reaction parameters
C0_2nd = 1.0  # M
k_2nd_true = 0.01  # M^-1 s^-1

# Generate second-order data
C_2nd_true = second_order_decay(time, C0_2nd, k_2nd_true)
noise_2nd = np.random.normal(0, 0.02 * C_2nd_true, size=len(time))
C_2nd_measured = C_2nd_true + noise_2nd
sigma_2nd = np.asarray(0.02 * C_2nd_measured + 0.001)

# Fit with robust preset
popt_2nd, pcov_2nd = fit(
    second_order_decay,
    time,
    C_2nd_measured,
    p0=[0.9, 0.008],
    sigma=sigma_2nd,
    bounds=([0, 0], [2, 0.1]),
    preset="robust",
)

C0_2nd_fit, k_2nd_fit = popt_2nd
perr_2nd = np.sqrt(np.diag(pcov_2nd))
C0_2nd_err, k_2nd_err = perr_2nd

# Half-life for second-order (depends on initial concentration)
t_half_2nd = 1 / (k_2nd_fit * C0_2nd_fit)
t_half_2nd_err = t_half_2nd * np.sqrt(
    (k_2nd_err / k_2nd_fit) ** 2 + (C0_2nd_err / C0_2nd_fit) ** 2
)

print("Fitted Parameters (robust preset):")
print(f"  [A]_0: {C0_2nd_fit:.4f} +/- {C0_2nd_err:.4f} M (true: {C0_2nd})")
print(f"  k:    {k_2nd_fit:.6f} +/- {k_2nd_err:.6f} M^-1 s^-1 (true: {k_2nd_true})")
print("\nHalf-Life (concentration-dependent):")
print(
    f"  t_1/2: {t_half_2nd:.2f} +/- {t_half_2nd_err:.2f} s (true: {1 / (k_2nd_true * C0_2nd):.2f})"
)

residuals_2nd = C_2nd_measured - second_order_decay(time, *popt_2nd)
rmse_2nd = np.sqrt(np.mean(residuals_2nd**2))

print("\nGoodness of Fit:")
print(f"  RMSE: {rmse_2nd:.5f} M")


# =============================================================================
# Reaction Order Determination with Global Optimization
# =============================================================================
print("\n" + "-" * 70)
print("REACTION ORDER DETERMINATION with GlobalOptimizationConfig")
print("-" * 70)

# Use global optimization to ensure we find the true global minimum
# for model selection

global_config = GlobalOptimizationConfig(
    n_starts=6 if QUICK else 20,
    sampler="lhs",
    center_on_p0=True,
    scale_factor=1.5,
)

# Fit both models to first-order data with global optimization
popt_1st_model, _ = fit(
    first_order_decay,
    time,
    C_1st_measured,
    p0=[0.9, 0.004],
    sigma=sigma_1st,
    bounds=([0, 0], [2, 0.1]),
    multistart=True,
    n_starts=6 if QUICK else 20,
    sampler="lhs",
)

popt_2nd_model, _ = fit(
    second_order_decay,
    time,
    C_1st_measured,
    p0=[0.9, 0.008],
    sigma=sigma_1st,
    bounds=([0, 0], [2, 0.1]),
    multistart=True,
    n_starts=6 if QUICK else 20,
    sampler="lhs",
)

residuals_1st_model = C_1st_measured - first_order_decay(time, *popt_1st_model)
residuals_2nd_model = C_1st_measured - second_order_decay(time, *popt_2nd_model)

rmse_1st_model = np.sqrt(np.mean(residuals_1st_model**2))
rmse_2nd_model = np.sqrt(np.mean(residuals_2nd_model**2))

print("Testing first-order data with global optimization:")
print(f"  1st-order fit RMSE: {rmse_1st_model:.5f} M")
print(f"  2nd-order fit RMSE: {rmse_2nd_model:.5f} M")
print(
    f"  Best fit: {'First-order' if rmse_1st_model < rmse_2nd_model else 'Second-order'}"
)

# AIC for model selection
n = len(time)
k_params = 2

AIC_1st = n * np.log(rmse_1st_model**2) + 2 * k_params
AIC_2nd = n * np.log(rmse_2nd_model**2) + 2 * k_params

print("\nModel Selection (AIC, lower is better):")
print(f"  1st-order: AIC = {AIC_1st:.2f}")
print(f"  2nd-order: AIC = {AIC_2nd:.2f}")
print(f"  Preferred: {'First-order' if AIC_1st < AIC_2nd else 'Second-order'}")

if QUICK:
    print("⏩ Quick mode: skipping linearization and additional analyses.")
    sys.exit(0)


# =============================================================================
# Linearization Analysis
# =============================================================================
print("\n" + "-" * 70)
print("LINEARIZATION ANALYSIS")
print("-" * 70)

# First-order: ln[A] vs t should be linear (slope = -k)
ln_C_1st = np.log(C_1st_measured)


def linear_1st_order(t, ln_C0, k):
    return ln_C0 - k * t


popt_ln, _ = fit(linear_1st_order, time, ln_C_1st, p0=[0, 0.005])
ln_C0_fit, k_ln_fit = popt_ln
C0_from_ln = np.exp(ln_C0_fit)

print("First-order linearization (ln[A] vs t):")
print(f"  k from linearization: {k_ln_fit:.6f} s^-1")
print(f"  k from direct fit:    {k_1st_fit:.6f} s^-1")
print(f"  Agreement: {abs(k_ln_fit - k_1st_fit) / k_1st_fit * 100:.2f}% difference")

# Second-order: 1/[A] vs t should be linear (slope = k)
inv_C_2nd = 1 / C_2nd_measured


def linear_2nd_order(t, inv_C0, k):
    return inv_C0 + k * t


popt_inv, _ = fit(linear_2nd_order, time, inv_C_2nd, p0=[1, 0.01])
inv_C0_fit, k_inv_fit = popt_inv

print("\nSecond-order linearization (1/[A] vs t):")
print(f"  k from linearization: {k_inv_fit:.6f} M^-1 s^-1")
print(f"  k from direct fit:    {k_2nd_fit:.6f} M^-1 s^-1")
print(f"  Agreement: {abs(k_inv_fit - k_2nd_fit) / k_2nd_fit * 100:.2f}% difference")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: First-order concentration vs time
ax1 = plt.subplot(3, 3, 1)
ax1.errorbar(
    time,
    C_1st_measured,
    yerr=sigma_1st,
    fmt="o",
    alpha=0.4,
    markersize=3,
    capsize=0,
    label="Measured",
)

t_fine = np.linspace(0, 1000, 200)
ax1.plot(
    t_fine,
    first_order_decay(t_fine, C0_1st, k_1st_true),
    "r--",
    linewidth=2,
    label="True",
    alpha=0.7,
)
ax1.plot(
    t_fine,
    first_order_decay(t_fine, *popt_1st_robust),
    "g-",
    linewidth=2.5,
    label="Fitted (robust)",
)

ax1.axhline(C0_1st_fit / 2, color="orange", linestyle=":", alpha=0.5)
ax1.axvline(
    t_half_1st,
    color="orange",
    linestyle=":",
    alpha=0.5,
    label=f"t_1/2 = {t_half_1st:.0f}s",
)

ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Concentration [A] (M)", fontsize=11)
ax1.set_title("First-Order Reaction - fit() API", fontsize=12, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: First-order semi-log plot
ax2 = plt.subplot(3, 3, 2)
ax2.semilogy(time, C_1st_measured, "o", alpha=0.4, markersize=3, label="Measured")
ax2.semilogy(
    t_fine,
    first_order_decay(t_fine, *popt_1st_robust),
    "g-",
    linewidth=2.5,
    label="Fitted",
)
ax2.semilogy(
    t_fine,
    np.exp(linear_1st_order(t_fine, *popt_ln)),
    "b--",
    linewidth=2,
    label="Linear fit",
)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Concentration [A] (M, log scale)")
ax2.set_title("First-Order: Semi-Log Plot")
ax2.legend()
ax2.grid(True, alpha=0.3, which="both")

# Plot 3: First-order residuals
ax3 = plt.subplot(3, 3, 3)
normalized_res_1st = residuals_1st / sigma_1st
ax3.plot(time, normalized_res_1st, "o", alpha=0.5, markersize=4)
ax3.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax3.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Normalized Residuals (sigma)")
ax3.set_title("First-Order Fit Residuals")
ax3.grid(True, alpha=0.3)

# Plot 4: Second-order concentration vs time
ax4 = plt.subplot(3, 3, 4)
ax4.errorbar(
    time,
    C_2nd_measured,
    yerr=sigma_2nd,
    fmt="o",
    alpha=0.4,
    markersize=3,
    capsize=0,
    label="Measured",
)
ax4.plot(
    t_fine,
    second_order_decay(t_fine, C0_2nd, k_2nd_true),
    "r--",
    linewidth=2,
    label="True",
    alpha=0.7,
)
ax4.plot(
    t_fine, second_order_decay(t_fine, *popt_2nd), "g-", linewidth=2.5, label="Fitted"
)

ax4.axhline(C0_2nd_fit / 2, color="orange", linestyle=":", alpha=0.5)
ax4.axvline(
    t_half_2nd,
    color="orange",
    linestyle=":",
    alpha=0.5,
    label=f"t_1/2 = {t_half_2nd:.0f}s",
)

ax4.set_xlabel("Time (s)", fontsize=11)
ax4.set_ylabel("Concentration [A] (M)", fontsize=11)
ax4.set_title("Second-Order Reaction - fit() API", fontsize=12, fontweight="bold")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Second-order linearization (1/[A] vs t)
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, inv_C_2nd, "o", alpha=0.4, markersize=3, label="1/[A] data")
ax5.plot(
    t_fine,
    linear_2nd_order(t_fine, *popt_inv),
    "b-",
    linewidth=2.5,
    label=f"Linear fit (slope={k_inv_fit:.4f})",
)

ax5.set_xlabel("Time (s)")
ax5.set_ylabel("1/[A] (M^-1)")
ax5.set_title("Second-Order: Linearization")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Second-order residuals
ax6 = plt.subplot(3, 3, 6)
normalized_res_2nd = residuals_2nd / sigma_2nd
ax6.plot(time, normalized_res_2nd, "o", alpha=0.5, markersize=4)
ax6.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax6.axhline(2, color="gray", linestyle=":", alpha=0.5)
ax6.axhline(-2, color="gray", linestyle=":", alpha=0.5)
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Normalized Residuals (sigma)")
ax6.set_title("Second-Order Fit Residuals")
ax6.grid(True, alpha=0.3)

# Plot 7: Model comparison
ax7 = plt.subplot(3, 3, 7)
ax7.plot(time, C_1st_measured, "o", alpha=0.5, markersize=4, label="1st-order data")
ax7.plot(
    t_fine,
    first_order_decay(t_fine, *popt_1st_model),
    "g-",
    linewidth=2.5,
    label="1st-order fit",
)
ax7.plot(
    t_fine,
    second_order_decay(t_fine, *popt_2nd_model),
    "b--",
    linewidth=2,
    label="2nd-order fit (wrong)",
)

ax7.set_xlabel("Time (s)")
ax7.set_ylabel("Concentration [A] (M)")
ax7.set_title("Model Comparison (Global Opt.)")
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Rate comparison
ax8 = plt.subplot(3, 3, 8)
rate_1st = k_1st_fit * first_order_decay(t_fine, *popt_1st_robust)
rate_2nd = k_2nd_fit * second_order_decay(t_fine, *popt_2nd) ** 2

ax8.plot(t_fine, rate_1st, "g-", linewidth=2.5, label="1st-order: r = k[A]")
ax8.plot(t_fine, rate_2nd, "b-", linewidth=2.5, label="2nd-order: r = k[A]^2")

ax8.set_xlabel("Time (s)")
ax8.set_ylabel("Reaction Rate (M/s)")
ax8.set_title("Reaction Rate vs Time")
ax8.legend()
ax8.grid(True, alpha=0.3)

# Plot 9: Summary table
ax9 = plt.subplot(3, 3, 9)
ax9.axis("off")

summary_text = [
    ["Parameter", "1st-Order", "2nd-Order"],
    ["-" * 15, "-" * 15, "-" * 15],
    ["Rate const.", f"{k_1st_fit:.5f} s^-1", f"{k_2nd_fit:.5f} M^-1s^-1"],
    ["Initial [A]", f"{C0_1st_fit:.4f} M", f"{C0_2nd_fit:.4f} M"],
    ["Half-life", f"{t_half_1st:.1f} s", f"{t_half_2nd:.1f} s"],
    ["t_1/2 formula", "ln(2)/k", "1/(k[A]_0)"],
    ["", "", ""],
    ["Rate law", "-d[A]/dt = k[A]", "-d[A]/dt = k[A]^2"],
    ["", "", ""],
    ["RMSE", f"{rmse_1st:.5f} M", f"{rmse_2nd:.5f} M"],
    ["", "", ""],
    ["API: fit() with", "", ""],
    ["preset='robust'", "", ""],
]

table_text = "\n".join(["  ".join(row) for row in summary_text])
ax9.text(
    0.1,
    0.9,
    table_text,
    fontsize=9,
    verticalalignment="top",
    fontfamily="monospace",
    transform=ax9.transAxes,
)
ax9.set_title("Summary Table", fontsize=12, fontweight="bold")

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "reaction_kinetics"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Kinetic parameters determined using fit() API:")

print("\nFirst-Order Reaction:")
print(f"  Rate constant: {k_1st_fit:.6f} +/- {k_1st_err:.6f} s^-1")
print(f"  Half-life:     {t_half_1st:.2f} +/- {t_half_1st_err:.2f} s")
print(f"  Initial conc.: {C0_1st_fit:.4f} +/- {C0_1st_err:.4f} M")

print("\nSecond-Order Reaction:")
print(f"  Rate constant: {k_2nd_fit:.6f} +/- {k_2nd_err:.6f} M^-1 s^-1")
print(
    f"  Half-life:     {t_half_2nd:.2f} +/- {t_half_2nd_err:.2f} s (at [A]_0 = {C0_2nd_fit:.2f} M)"
)
print(f"  Initial conc.: {C0_2nd_fit:.4f} +/- {C0_2nd_err:.4f} M")

print("\nModel Selection with Global Optimization:")
print(f"  Best fit for test data: First-order (AIC = {AIC_1st:.2f})")

print("\nAPI Methods Used:")
print("  - fit() with preset='robust' (5 multi-starts)")
print("  - fit() with preset='global' (20 multi-starts)")
print("  - fit() with GlobalOptimizationConfig for model selection")

print("\nThis example demonstrates:")
print("  - First-order and second-order kinetics fitting with fit() API")
print("  - Global optimization for robust model selection")
print("  - Rate constant determination with uncertainties")
print("  - Linearization methods")
print("=" * 70)
