"""
Advanced Sensor Calibration with fit() API and GlobalOptimizationConfig.

This example demonstrates calibrating a non-linear temperature sensor
(e.g., thermistor, RTD) using NLSQ's advanced fit() API and global optimization
for robust calibration curve fitting.

Compared to 04_gallery/engineering/sensor_calibration.py:
- Uses fit() instead of curve_fit() for automatic workflow selection
- Demonstrates GlobalOptimizationConfig for multi-start optimization
- Shows how presets ('robust', 'global') improve fitting reliability

Key Concepts:
- Non-linear sensor response modeling
- Polynomial calibration curves
- Model comparison (linear, quadratic, cubic)
- Residual analysis for calibration quality
- Global optimization for robust parameter estimation
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import GlobalOptimizationConfig, fit

if os.environ.get("NLSQ_EXAMPLES_QUICK"):
    print("Quick mode: skipping advanced sensor calibration demo.")
    sys.exit(0)

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

# Measurement uncertainties
sigma_voltage = 5.0 * np.ones_like(voltage_measured)
sigma_temp = 0.1 * np.ones_like(temp_reference)


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
    preset="robust",
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
    preset="robust",
)

c0, c1, c2 = popt_quad
perr_quad = np.sqrt(np.diag(pcov_quad))

temp_pred_quad = quadratic_model(voltage_measured, *popt_quad)
residuals_quad = temp_reference - temp_pred_quad
rmse_quad = np.sqrt(np.mean(residuals_quad**2))

print(f"  y = {c0:.4f} + {c1:.6f}*V + {c2:.3e}*V^2")
print(f"  RMSE: {rmse_quad:.2f} C")


# =============================================================================
# Model 3: Cubic Calibration (Recommended) with Global Optimization
# =============================================================================
print("\n" + "-" * 70)
print("Model 3: Cubic Calibration with fit() API")
print("-" * 70)

# Method 1: fit() with 'robust' preset
print("\nMethod 1: fit() with 'robust' preset")
popt_cubic, pcov_cubic = fit(
    cubic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6, 1e-10],
    sigma=sigma_temp,
    absolute_sigma=True,
    preset="robust",
)

d0, d1, d2, d3 = popt_cubic
perr_cubic = np.sqrt(np.diag(pcov_cubic))

temp_pred_cubic = cubic_model(voltage_measured, *popt_cubic)
residuals_cubic = temp_reference - temp_pred_cubic
rmse_cubic = np.sqrt(np.mean(residuals_cubic**2))

print(f"  y = {d0:.4f} + {d1:.6f}*V + {d2:.3e}*V^2 + {d3:.3e}*V^3")
print(f"  RMSE: {rmse_cubic:.3f} C")
print(f"  Max residual: {np.max(np.abs(residuals_cubic)):.2f} C")

# Method 2: fit() with 'global' preset
print("\nMethod 2: fit() with 'global' preset")
popt_cubic_g, pcov_cubic_g = fit(
    cubic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6, 1e-10],
    sigma=sigma_temp,
    absolute_sigma=True,
    preset="global",
)

d0_g, d1_g, d2_g, d3_g = popt_cubic_g
perr_g = np.sqrt(np.diag(pcov_cubic_g))
temp_pred_cubic_g = cubic_model(voltage_measured, *popt_cubic_g)
residuals_cubic_g = temp_reference - temp_pred_cubic_g
rmse_cubic_g = np.sqrt(np.mean(residuals_cubic_g**2))

print(f"  RMSE: {rmse_cubic_g:.3f} C")

# Method 3: GlobalOptimizationConfig with custom settings
print("\nMethod 3: GlobalOptimizationConfig with custom settings")
popt_cubic_c, pcov_cubic_c = fit(
    cubic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6, 1e-10],
    sigma=sigma_temp,
    absolute_sigma=True,
    multistart=True,
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
# Model Comparison
# =============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(f"{'Model':<20} {'RMSE (C)':<15} {'Max Error (C)':<15} {'Parameters':<10}")
print("-" * 70)
print(
    f"{'Linear':<20} {rmse_linear:<15.2f} {np.max(np.abs(residuals_linear)):<15.2f} {2:<10}"
)
print(
    f"{'Quadratic':<20} {rmse_quad:<15.2f} {np.max(np.abs(residuals_quad)):<15.2f} {3:<10}"
)
print(
    f"{'Cubic (robust)':<20} {rmse_cubic:<15.3f} {np.max(np.abs(residuals_cubic)):<15.3f} {4:<10}"
)
print(
    f"{'Cubic (global)':<20} {rmse_cubic_g:<15.3f} {np.max(np.abs(residuals_cubic_g)):<15.3f} {4:<10}"
)
print("-" * 70)
print(f"Recommended: Cubic model (best accuracy for {len(temp_reference)} points)")


# =============================================================================
# Uncertainty Analysis
# =============================================================================
print("\n" + "=" * 70)
print("UNCERTAINTY ANALYSIS (Cubic Model)")
print("=" * 70)

V_unknown = 3500.0  # mV
V_unknown_err = 5.0  # mV

T_predicted = cubic_model(V_unknown, *popt_cubic)

dV = 1.0
dT_dV = (
    cubic_model(V_unknown + dV, *popt_cubic) - cubic_model(V_unknown - dV, *popt_cubic)
) / (2 * dV)

T_uncertainty = np.sqrt((dT_dV * V_unknown_err) ** 2 + rmse_cubic**2)

print("\nExample measurement:")
print(f"  Sensor voltage: {V_unknown:.1f} +/- {V_unknown_err:.1f} mV")
print(f"  Predicted temperature: {T_predicted:.2f} +/- {T_uncertainty:.2f} C")
print(f"  Sensitivity: dT/dV = {dT_dV:.4f} C/mV")


# =============================================================================
# Visualization
# =============================================================================
fig = plt.figure(figsize=(16, 12))

# Plot 1: Calibration curve
ax1 = plt.subplot(3, 2, 1)
V_fine = np.linspace(voltage_measured.min(), voltage_measured.max(), 200)
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
    "-",
    linewidth=2.5,
    label=f"Cubic (RMSE={rmse_cubic:.3f}C)",
    color="green",
)

ax1.set_xlabel("Sensor Voltage (mV)", fontsize=12)
ax1.set_ylabel("Temperature (C)", fontsize=12)
ax1.set_title("Calibration Curves - fit() API", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
ax2 = plt.subplot(3, 2, 2)
width = 0.25
x_pos = np.arange(len(temp_reference))
ax2.bar(x_pos - width, residuals_linear, width, label="Linear", alpha=0.7)
ax2.bar(x_pos, residuals_quad, width, label="Quadratic", alpha=0.7)
ax2.bar(x_pos + width, residuals_cubic, width, label="Cubic", alpha=0.7, color="green")
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
    residuals_cubic,
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
ax3.set_title("Cubic Model Residuals vs Temperature")
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals vs voltage (cubic model)
ax4 = plt.subplot(3, 2, 4)
ax4.errorbar(
    voltage_measured,
    residuals_cubic,
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
ax4.set_ylabel("Residual (C)")
ax4.set_title("Cubic Model Residuals vs Voltage")
ax4.grid(True, alpha=0.3)

# Plot 5: Histogram of residuals
ax5 = plt.subplot(3, 2, 5)
ax5.hist(residuals_cubic, bins=8, alpha=0.7, edgecolor="black", color="green")
ax5.axvline(0, color="r", linestyle="--", linewidth=2)
ax5.axvline(
    np.mean(residuals_cubic),
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Mean: {np.mean(residuals_cubic):.3f}C",
)

ax5.set_xlabel("Residual (C)")
ax5.set_ylabel("Frequency")
ax5.set_title("Residual Distribution (Cubic Model)")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Sensitivity analysis
ax6 = plt.subplot(3, 2, 6)
V_sens = np.linspace(voltage_measured.min(), voltage_measured.max(), 100)
dV_small = 1.0
sensitivity = (
    cubic_model(V_sens + dV_small, *popt_cubic)
    - cubic_model(V_sens - dV_small, *popt_cubic)
) / (2 * dV_small)
ax6.plot(V_sens, sensitivity, "g-", linewidth=2)

ax6.set_xlabel("Sensor Voltage (mV)")
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
print(f"  T(C) = {d0:.4f} + {d1:.6f}*V + {d2:.3e}*V^2 + {d3:.3e}*V^3")
print("\nCalibration quality:")
print(f"  RMSE: {rmse_cubic:.3f} C")
print(f"  Max error: {np.max(np.abs(residuals_cubic)):.3f} C")
print(f"  Mean residual: {np.mean(residuals_cubic):.3f} C")
print(f"  Std residual: {np.std(residuals_cubic):.3f} C")
print(f"\nValid range: {temp_reference.min():.0f} to {temp_reference.max():.0f} C")
print(f"             ({voltage_measured.min():.0f} to {voltage_measured.max():.0f} mV)")
print("\nAPI Methods Used:")
print("  - fit() with preset='robust' for all models")
print("  - fit() with preset='global' for thorough search")
print("  - fit() with GlobalOptimizationConfig for custom settings")
print("\nThis example demonstrates:")
print("  - Non-linear sensor calibration with fit() API")
print("  - Model comparison (linear, quadratic, cubic)")
print("  - Global optimization for robust parameter estimation")
print("  - Residual analysis for calibration quality")
print("  - Uncertainty propagation to measurements")
print("=" * 70)
