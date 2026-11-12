"""
Advanced Sensor Calibration with fit() API and workflow presets.

This example demonstrates calibrating a non-linear temperature sensor
(e.g., thermistor, RTD) using NLSQ's advanced fit() API and global optimization
for robust calibration curve fitting.

Compared to 04_gallery/engineering/sensor_calibration.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates workflow="auto_global" for multi-start optimization
- Shows how workflows ('auto', 'auto_global') improve fitting reliability

Key Concepts:
- Non-linear sensor response modeling
- Polynomial calibration curves vs. the physics-based Steinhart-Hart model (linear in ln V)
- Model comparison (linear, quadratic, cubic)
- Residual analysis for calibration quality
- Global optimization for robust parameter estimation
"""

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import fit

QUICK = os.environ.get("NLSQ_EXAMPLES_QUICK") == "1"

# Set random seed
np.random.seed(42)


def linear_model(V, a, b):
    """Linear calibration: T = a*V + b"""
    return a * V + b


def quadratic_model(V, c0, c1, c2):
    """Quadratic calibration: T = c0 + c1*V + c2*V^2"""
    return c0 + c1 * V + c2 * V**2


def cubic_model(V, d0, d1, d2, d3):
    """Cubic calibration: T = d0 + d1*V + d2*V^2 + d3*V^3"""
    return d0 + d1 * V + d2 * V**2 + d3 * V**3


def thermistor_model(V, A, B, C):
    """Steinhart-Hart calibration in Celsius: T = 1/(A + B*ln(V) + C*ln(V)^3) - 273.15."""
    return steinhart_hart(V, A, B, C) - 273.15


def steinhart_hart(R, A, B, C):
    """
    Steinhart-Hart equation for thermistors.

    1/T = A + B*ln(R) + C*(ln(R))^3

    Parameters
    ----------
    R : array_like
        Resistance (Ohms)
    A, B, C : float
        Steinhart-Hart coefficients

    Returns
    -------
    T : array_like
        Temperature (Kelvin)
    """
    ln_R = jnp.log(R)
    T_inv = A + B * ln_R + C * ln_R**3
    return 1.0 / T_inv


# Simulate thermistor-like sensor with non-linear response
# True reference temperatures (C)
temp_reference = np.array(
    [-20, -10, 0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
)

# Thermistor parameters
beta = 3500  # K
T_ref = 25 + 273.15  # K
V_ref = 2500  # mV

# True voltage response
T_kelvin_true = temp_reference + 273.15
voltage_true = V_ref * np.exp(beta * (1 / T_kelvin_true - 1 / T_ref))

# Add measurement noise
voltage_measured = voltage_true + np.random.normal(0, 5, size=len(temp_reference))

# Measurement uncertainties: the 5 mV voltage noise maps to temperature through the
# thermistor sensitivity, sigma_T = (T^2 / beta) * sigma_V / V (nominal beta)
sigma_voltage = 5.0 * np.ones_like(voltage_measured)
sigma_temp = (T_kelvin_true**2 / beta) * sigma_voltage / voltage_measured


if QUICK:
    print("=" * 70)
    print("SENSOR CALIBRATION: Quick Mode Summary")
    print("=" * 70)
    print("\n⏩ Quick mode: skipping full sensor calibration demo.")
    print("\nThis example demonstrates:")
    print("  - Linear, quadratic, cubic calibration models")
    print("  - Steinhart-Hart thermistor model")
    print("  - fit() with workflow='auto' and workflow='auto_global'")
    print("  - Multi-start optimization for non-linear models")
    print("=" * 70)
    sys.exit(0)

print("=" * 70)
print("SENSOR CALIBRATION: ADVANCED FITTING WITH fit() API")
print("=" * 70)

# =============================================================================
# Model 1: Linear Calibration
# =============================================================================
print("\n" + "-" * 70)
print("Model 1: Linear Calibration")
print("-" * 70)

popt_linear, pcov_linear = fit(
    linear_model,
    voltage_measured,
    temp_reference,
    p0=[0.01, -20],
    sigma=sigma_temp,
    absolute_sigma=True,
    workflow="auto",
)

a_lin, b_lin = popt_linear
perr_linear = np.sqrt(np.diag(pcov_linear))

temp_pred_linear = linear_model(voltage_measured, *popt_linear)
residuals_linear = temp_reference - temp_pred_linear
rmse_linear = np.sqrt(np.mean(residuals_linear**2))

print(f"  y = {a_lin:.6f}*V + {b_lin:.3f}")
print(f"  RMSE: {rmse_linear:.2f} C")


# =============================================================================
# Model 2: Quadratic Calibration
# =============================================================================
print("\n" + "-" * 70)
print("Model 2: Quadratic Calibration")
print("-" * 70)

popt_quad, pcov_quad = fit(
    quadratic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6],
    sigma=sigma_temp,
    absolute_sigma=True,
    workflow="auto",
)

c0, c1, c2 = popt_quad
perr_quad = np.sqrt(np.diag(pcov_quad))

temp_pred_quad = quadratic_model(voltage_measured, *popt_quad)
residuals_quad = temp_reference - temp_pred_quad
rmse_quad = np.sqrt(np.mean(residuals_quad**2))

print(f"  y = {c0:.4f} + {c1:.6f}*V + {c2:.3e}*V^2")
print(f"  RMSE: {rmse_quad:.2f} C")


# =============================================================================
# Model 3: Cubic Polynomial Calibration with Global Optimization
# =============================================================================
print("\n" + "-" * 70)
print("Model 3: Cubic Calibration with fit() API")
print("-" * 70)

# Method 1: fit() with 'auto' workflow
print("\nMethod 1: fit() with 'auto' workflow")
popt_cubic, pcov_cubic = fit(
    cubic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6, 1e-10],
    sigma=sigma_temp,
    absolute_sigma=True,
    workflow="auto",
)

d0, d1, d2, d3 = popt_cubic
perr_cubic = np.sqrt(np.diag(pcov_cubic))

temp_pred_cubic = cubic_model(voltage_measured, *popt_cubic)
residuals_cubic = temp_reference - temp_pred_cubic
rmse_cubic = np.sqrt(np.mean(residuals_cubic**2))

print(f"  y = {d0:.4f} + {d1:.6f}*V + {d2:.3e}*V^2 + {d3:.3e}*V^3")
print(f"  RMSE: {rmse_cubic:.3f} C")
print(f"  Max residual: {np.max(np.abs(residuals_cubic)):.2f} C")

if QUICK:
    # Use auto workflow results as fallback for visualization
    d0_g, d1_g, d2_g, d3_g = d0, d1, d2, d3
    rmse_cubic_g = rmse_cubic
    residuals_cubic_g = residuals_cubic
    d0_c, d1_c, d2_c, d3_c = d0, d1, d2, d3
    rmse_cubic_c = rmse_cubic
    residuals_cubic_c = residuals_cubic
    print("\n[Quick mode] Skipping auto_global methods - using auto results")
else:
    # Bounds for global optimization (must bracket the optimum: negative slope,
    # since the voltage falls as the temperature rises)
    bounds_cubic_lower = [-200, -0.2, -1e-4, -1e-8]
    bounds_cubic_upper = [200, 0.2, 1e-4, 1e-8]

    # Method 2: fit() with 'auto_global' workflow
    print("\nMethod 2: fit() with 'auto_global' workflow")
    popt_cubic_g, pcov_cubic_g = fit(
        cubic_model,
        voltage_measured,
        temp_reference,
        p0=[-20, 0.01, -1e-6, 1e-10],
        sigma=sigma_temp,
        absolute_sigma=True,
        workflow="auto_global",
        bounds=(bounds_cubic_lower, bounds_cubic_upper),
    )

    d0_g, d1_g, d2_g, d3_g = popt_cubic_g
    perr_g = np.sqrt(np.diag(pcov_cubic_g))
    temp_pred_cubic_g = cubic_model(voltage_measured, *popt_cubic_g)
    residuals_cubic_g = temp_reference - temp_pred_cubic_g
    rmse_cubic_g = np.sqrt(np.mean(residuals_cubic_g**2))

    print(f"  RMSE: {rmse_cubic_g:.3f} C")

    # Method 3: workflow="auto_global" with custom settings
    print("\nMethod 3: workflow='auto_global' with custom settings")
    popt_cubic_c, pcov_cubic_c = fit(
        cubic_model,
        voltage_measured,
        temp_reference,
        p0=[-20, 0.01, -1e-6, 1e-10],
        sigma=sigma_temp,
        absolute_sigma=True,
        workflow="auto_global",
        bounds=(bounds_cubic_lower, bounds_cubic_upper),
        n_starts=15,
        sampler="lhs",
    )

    d0_c, d1_c, d2_c, d3_c = popt_cubic_c
    perr_c = np.sqrt(np.diag(pcov_cubic_c))
    temp_pred_cubic_c = cubic_model(voltage_measured, *popt_cubic_c)
    residuals_cubic_c = temp_reference - temp_pred_cubic_c
    rmse_cubic_c = np.sqrt(np.mean(residuals_cubic_c**2))

    print(f"  RMSE: {rmse_cubic_c:.3f} C")


# =============================================================================
# Model 4: Steinhart-Hart Calibration (Recommended)
# =============================================================================
print("\n" + "-" * 70)
print("Model 4: Steinhart-Hart Calibration (Recommended)")
print("-" * 70)

# A thermistor obeys 1/T = A + B*ln(R) + C*ln(R)^3, so the calibration is linear in
# ln(V), not a polynomial in V. Initial guess: linear fit of 1/T against ln(V).
B0, A0 = np.polyfit(np.log(voltage_measured), 1.0 / (temp_reference + 273.15), 1)
p0_sh = [A0, B0, 0.0]

print("\nMethod 1: fit() with 'auto' workflow")
popt_sh, pcov_sh = fit(
    thermistor_model,
    voltage_measured,
    temp_reference,
    p0=p0_sh,
    sigma=sigma_temp,
    absolute_sigma=True,
    workflow="auto",
)

A_sh, B_sh, C_sh = popt_sh
perr_sh = np.sqrt(np.diag(pcov_sh))
temp_pred_sh = np.asarray(thermistor_model(voltage_measured, *popt_sh))
residuals_sh = temp_reference - temp_pred_sh
rmse_sh = np.sqrt(np.mean(residuals_sh**2))

print(f"  1/T[K] = {A_sh:.5e} + {B_sh:.5e}*ln(V) + {C_sh:.3e}*ln(V)^3")
print(
    f"  True (beta model): A = {1 / T_ref - np.log(V_ref) / beta:.5e}, B = {1 / beta:.5e}, C = 0"
)
print(f"  RMSE: {rmse_sh:.3f} C")
print(f"  Max residual: {np.max(np.abs(residuals_sh)):.2f} C")

print("\nMethod 2: fit() with 'auto_global' workflow")
popt_sh_g, pcov_sh_g = fit(
    thermistor_model,
    voltage_measured,
    temp_reference,
    p0=p0_sh,
    sigma=sigma_temp,
    absolute_sigma=True,
    workflow="auto_global",
    bounds=([5e-4, 1e-4, -1e-7], [2e-3, 1e-3, 1e-7]),
)
residuals_sh_g = temp_reference - np.asarray(
    thermistor_model(voltage_measured, *popt_sh_g)
)
rmse_sh_g = np.sqrt(np.mean(residuals_sh_g**2))
print(f"  RMSE: {rmse_sh_g:.3f} C")


# =============================================================================
# Model Comparison
# =============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


def reduced_chi2(residuals, n_par):
    return np.sum((residuals / sigma_temp) ** 2) / (len(residuals) - n_par)


rows = [
    ("Linear", residuals_linear, 2),
    ("Quadratic", residuals_quad, 3),
    ("Cubic (robust)", residuals_cubic, 4),
    ("Cubic (global)", residuals_cubic_g, 4),
    ("Steinhart-Hart", residuals_sh, 3),
    ("Steinhart-Hart (glob)", residuals_sh_g, 3),
]
print(
    f"{'Model':<22} {'RMSE (C)':<11} {'Max Error (C)':<15} {'chi2/dof':<11} {'Parameters':<10}"
)
print("-" * 70)
for label, res, n_par in rows:
    print(
        f"{label:<22} {np.sqrt(np.mean(res**2)):<11.3f} {np.max(np.abs(res)):<15.3f} "
        f"{reduced_chi2(res, n_par):<11.2f} {n_par:<10}"
    )
print("-" * 70)
print(
    f"Recommended: Steinhart-Hart model (chi2/dof = {reduced_chi2(residuals_sh, 3):.2f}); "
    "polynomials in V cannot follow the exponential thermistor response"
)


# =============================================================================
# Uncertainty Analysis
# =============================================================================
print("\n" + "=" * 70)
print("UNCERTAINTY ANALYSIS (Steinhart-Hart Model)")
print("=" * 70)

V_unknown = 3500.0  # mV
V_unknown_err = 5.0  # mV

T_predicted = float(thermistor_model(V_unknown, *popt_sh))
dT_dV = float(jax.grad(thermistor_model)(V_unknown, *popt_sh))
# Calibration (parameter) uncertainty propagated through the model Jacobian
J_p = np.asarray(
    jax.jacfwd(lambda p: thermistor_model(V_unknown, *p))(jnp.asarray(popt_sh))
)
T_cal_err = float(np.sqrt(J_p @ pcov_sh @ J_p))
T_read_err = abs(dT_dV) * V_unknown_err
T_uncertainty = np.sqrt(T_read_err**2 + T_cal_err**2)

print("\nExample measurement:")
print(f"  Sensor voltage: {V_unknown:.1f} +/- {V_unknown_err:.1f} mV")
print(f"  Predicted temperature: {T_predicted:.2f} +/- {T_uncertainty:.2f} C")
print(f"  (reading noise {T_read_err:.3f} C, calibration fit {T_cal_err:.3f} C)")
print(f"  Sensitivity: dT/dV = {dT_dV:.4f} C/mV")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Calibration curve
ax1 = plt.subplot(3, 2, 1)
V_fine = np.geomspace(voltage_measured.min(), voltage_measured.max(), 200)
ax1.errorbar(
    voltage_measured,
    temp_reference,
    xerr=sigma_voltage,
    yerr=sigma_temp,
    fmt="o",
    capsize=4,
    markersize=6,
    label="Calibration data",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    linear_model(V_fine, *popt_linear),
    "--",
    linewidth=2,
    label=f"Linear (RMSE={rmse_linear:.2f}C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    quadratic_model(V_fine, *popt_quad),
    "--",
    linewidth=2,
    label=f"Quadratic (RMSE={rmse_quad:.2f}C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    cubic_model(V_fine, *popt_cubic),
    "--",
    linewidth=2,
    label=f"Cubic (RMSE={rmse_cubic:.2f}C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    thermistor_model(V_fine, *popt_sh),
    "-",
    linewidth=2.5,
    label=f"Steinhart-Hart (RMSE={rmse_sh:.3f}C)",
    color="green",
)
ax1.set_xscale("log")
ax1.set_ylim(temp_reference.min() - 30, temp_reference.max() + 30)

ax1.set_xlabel("Sensor Voltage (mV)", fontsize=12)
ax1.set_ylabel("Temperature (C)", fontsize=12)
ax1.set_title("Calibration Curves - fit() API", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
ax2 = plt.subplot(3, 2, 2)
width = 0.2
x_pos = np.arange(len(temp_reference))
ax2.bar(x_pos - 1.5 * width, residuals_linear, width, label="Linear", alpha=0.7)
ax2.bar(x_pos - 0.5 * width, residuals_quad, width, label="Quadratic", alpha=0.7)
ax2.bar(x_pos + 0.5 * width, residuals_cubic, width, label="Cubic", alpha=0.7)
ax2.bar(
    x_pos + 1.5 * width,
    residuals_sh,
    width,
    label="Steinhart-Hart",
    alpha=0.7,
    color="green",
)
ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-0.5, color="gray", linestyle=":", alpha=0.5)

ax2.set_xlabel("Calibration Point Index")
ax2.set_ylabel("Residual (C)")
ax2.set_title("Residuals: Model Comparison")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Residuals vs temperature (cubic model)
ax3 = plt.subplot(3, 2, 3)
ax3.errorbar(
    temp_reference,
    residuals_sh,
    yerr=sigma_temp,
    fmt="o",
    capsize=4,
    markersize=6,
    alpha=0.7,
)
ax3.axhline(0, color="r", linestyle="--", linewidth=2)
ax3.axhline(0.2, color="gray", linestyle=":", alpha=0.5)
ax3.axhline(-0.2, color="gray", linestyle=":", alpha=0.5)

ax3.set_xlabel("Reference Temperature (C)")
ax3.set_ylabel("Residual (C)")
ax3.set_title("Steinhart-Hart Model Residuals vs Temperature")
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals vs voltage (cubic model)
ax4 = plt.subplot(3, 2, 4)
ax4.errorbar(
    voltage_measured,
    residuals_sh,
    xerr=sigma_voltage,
    fmt="o",
    capsize=4,
    markersize=6,
    alpha=0.7,
)
ax4.axhline(0, color="r", linestyle="--", linewidth=2)
ax4.axhline(0.2, color="gray", linestyle=":", alpha=0.5)
ax4.axhline(-0.2, color="gray", linestyle=":", alpha=0.5)

ax4.set_xlabel("Sensor Voltage (mV)")
ax4.set_xscale("log")
ax4.set_ylabel("Residual (C)")
ax4.set_title("Steinhart-Hart Model Residuals vs Voltage")
ax4.grid(True, alpha=0.3)

# Plot 5: Histogram of residuals
ax5 = plt.subplot(3, 2, 5)
ax5.hist(residuals_sh, bins=8, alpha=0.7, edgecolor="black", color="green")
ax5.axvline(0, color="r", linestyle="--", linewidth=2)
ax5.axvline(
    np.mean(residuals_sh),
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Mean: {np.mean(residuals_sh):.3f}C",
)

ax5.set_xlabel("Residual (C)")
ax5.set_ylabel("Frequency")
ax5.set_title("Residual Distribution (Steinhart-Hart Model)")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Sensitivity analysis
ax6 = plt.subplot(3, 2, 6)
V_sens = np.geomspace(voltage_measured.min(), voltage_measured.max(), 100)
dV_small = 1.0
sensitivity = (
    thermistor_model(V_sens + dV_small, *popt_sh)
    - thermistor_model(V_sens - dV_small, *popt_sh)
) / (2 * dV_small)
ax6.plot(V_sens, sensitivity, "g-", linewidth=2)

ax6.set_xlabel("Sensor Voltage (mV)")
ax6.set_xscale("log")
ax6.set_ylabel("Sensitivity (C/mV)")
ax6.set_title("Calibration Sensitivity (dT/dV)")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "sensor_calibration"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Calibration complete using fit() API:")
print(f"  1/T[K] = {A_sh:.5e} + {B_sh:.5e}*ln(V) + {C_sh:.3e}*ln(V)^3")
print("\nCalibration quality:")
print(f"  RMSE: {rmse_sh:.3f} C")
print(f"  chi^2/dof: {reduced_chi2(residuals_sh, 3):.2f}")
print(f"  Max error: {np.max(np.abs(residuals_sh)):.3f} C")
print(f"  Mean residual: {np.mean(residuals_sh):.3f} C")
print(f"  Std residual: {np.std(residuals_sh):.3f} C")
print(f"\nValid range: {temp_reference.min():.0f} to {temp_reference.max():.0f} C")
print(f"             ({voltage_measured.min():.0f} to {voltage_measured.max():.0f} mV)")
print("\nAPI Methods Used:")
print("  - fit() with workflow='auto' for all models")
print("  - fit() with workflow='auto_global' for thorough search")
print("  - fit() with workflow='auto_global' and custom n_starts/sampler")
print("\nThis example demonstrates:")
print("  - Non-linear sensor calibration with fit() API")
print("  - Model comparison (linear, quadratic, cubic)")
print("  - Global optimization for robust parameter estimation")
print("  - Residual analysis for calibration quality")
print("  - Uncertainty propagation to measurements")
print("=" * 70)
