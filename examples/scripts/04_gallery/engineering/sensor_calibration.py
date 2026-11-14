"""
Converted from sensor_calibration.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

from pathlib import Path

# ======================================================================
# # Sensor Calibration: Non-Linear Temperature Sensor
#
#
# This example demonstrates calibrating a non-linear temperature sensor
# (e.g., thermistor, RTD) by fitting a calibration curve to reference
# measurements. We use polynomial fitting and analyze residuals to assess
# calibration quality.
#
# Key Concepts:
# - Non-linear sensor response modeling
# - Polynomial calibration curves
# - Residual analysis for calibration quality
# - Uncertainty propagation to measured values
# - Comparison of different calibration models
#
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import curve_fit

# Set random seed
np.random.seed(42)


def steinhart_hart(R, A, B, C):
    """
    Steinhart-Hart equation for thermistors.

    1/T = A + B*ln(R) + C*(ln(R))³

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


def thermistor_model(V, A, B, C):
    """Steinhart-Hart temperature in degrees Celsius; the sensor voltage plays the role of R."""
    return steinhart_hart(V, A, B, C) - 273.15


def polynomial_calibration(x, *coeffs):
    """
    Polynomial calibration model.

    y = c0 + c1*x + c2*x² + c3*x³ + ...

    Parameters
    ----------
    x : array_like
        Input variable (e.g., sensor voltage)
    coeffs : tuple
        Polynomial coefficients (c0, c1, c2, ...)

    Returns
    -------
    y : array_like
        Output variable (e.g., temperature)
    """
    result = jnp.zeros_like(x)
    for i, c in enumerate(coeffs):
        result += c * x**i
    return result


# Simulate thermistor-like sensor with non-linear response
# Sensor voltage vs true temperature (°C)

# True reference temperatures (°C)
temp_reference = np.array(
    [-20, -10, 0, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
)

# True non-linear sensor response (thermistor-like)
# Voltage output (mV) = f(temperature)
# Based on simplified exponential model: V ∝ exp(-β/T)
beta = 3500  # K (thermistor parameter)
T_ref = 25 + 273.15  # K (reference temperature)
V_ref = 2500  # mV (voltage at reference temperature)

# True voltage response
T_kelvin_true = temp_reference + 273.15
voltage_true = V_ref * np.exp(beta * (1 / T_kelvin_true - 1 / T_ref))

# Add measurement noise (±5 mV instrument precision)
voltage_measured = voltage_true + np.random.normal(0, 5, size=len(temp_reference))

# Measurement uncertainties
sigma_voltage = 5.0 * np.ones_like(voltage_measured)  # mV
# Voltage noise referred to temperature: V = V_ref exp(beta (1/T - 1/T_ref)) gives
# dT/dV = T^2 / (beta V), so sigma_T = T^2 sigma_V / (beta V) (nominal beta)
sigma_temp = (T_kelvin_true**2 / beta) * sigma_voltage / voltage_measured


print("=" * 70)
print("SENSOR CALIBRATION: NON-LINEAR TEMPERATURE SENSOR")
print("=" * 70)

# Model 1: Linear calibration (simplest, likely poor fit)
print("\n" + "-" * 70)
print("Model 1: Linear Calibration")
print("-" * 70)


def linear_model(V, a, b):
    return a * V + b


popt_linear, pcov_linear = curve_fit(
    linear_model,
    voltage_measured,
    temp_reference,
    p0=[0.01, -20],
    sigma=sigma_temp,
    absolute_sigma=True,
)

a_lin, b_lin = popt_linear
perr_linear = np.sqrt(np.diag(pcov_linear))

# Calculate residuals
temp_pred_linear = linear_model(voltage_measured, *popt_linear)
residuals_linear = temp_reference - temp_pred_linear
rmse_linear = np.sqrt(np.mean(residuals_linear**2))

print(f"  y = {a_lin:.6f}*V + {b_lin:.3f}")
print(f"  RMSE: {rmse_linear:.2f} °C")

# Model 2: Quadratic calibration (better for moderate non-linearity)
print("\n" + "-" * 70)
print("Model 2: Quadratic Calibration")
print("-" * 70)


def quadratic_model(V, c0, c1, c2):
    return c0 + c1 * V + c2 * V**2


popt_quad, pcov_quad = curve_fit(
    quadratic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6],
    sigma=sigma_temp,
    absolute_sigma=True,
)

c0, c1, c2 = popt_quad
perr_quad = np.sqrt(np.diag(pcov_quad))

temp_pred_quad = quadratic_model(voltage_measured, *popt_quad)
residuals_quad = temp_reference - temp_pred_quad
rmse_quad = np.sqrt(np.mean(residuals_quad**2))

print(f"  y = {c0:.4f} + {c1:.6f}*V + {c2:.3e}*V²")
print(f"  RMSE: {rmse_quad:.2f} °C")

# Model 3: Cubic calibration (polynomial in V, kept as a contrast to the physical model)
print("\n" + "-" * 70)
print("Model 3: Cubic Calibration")
print("-" * 70)


def cubic_model(V, d0, d1, d2, d3):
    return d0 + d1 * V + d2 * V**2 + d3 * V**3


popt_cubic, pcov_cubic = curve_fit(
    cubic_model,
    voltage_measured,
    temp_reference,
    p0=[-20, 0.01, -1e-6, 1e-10],
    sigma=sigma_temp,
    absolute_sigma=True,
)

d0, d1, d2, d3 = popt_cubic
perr_cubic = np.sqrt(np.diag(pcov_cubic))

temp_pred_cubic = cubic_model(voltage_measured, *popt_cubic)
residuals_cubic = temp_reference - temp_pred_cubic
rmse_cubic = np.sqrt(np.mean(residuals_cubic**2))

print(f"  y = {d0:.4f} + {d1:.6f}*V + {d2:.3e}*V² + {d3:.3e}*V³")
print(f"  RMSE: {rmse_cubic:.2f} °C")
print(f"  Max residual: {np.max(np.abs(residuals_cubic)):.2f} °C")

# Model 4: Steinhart-Hart (physical model)
print("\n" + "-" * 70)
print("Model 4: Steinhart-Hart Calibration (Recommended)")
print("-" * 70)

# A thermistor obeys 1/T = A + B*ln(R) + C*ln(R)^3, so the calibration is linear in
# ln(V), not a polynomial in V. Initial guess: linear fit of 1/T against ln(V).
B0, A0 = np.polyfit(np.log(voltage_measured), 1.0 / (temp_reference + 273.15), 1)

popt_sh, pcov_sh = curve_fit(
    thermistor_model,
    voltage_measured,
    temp_reference,
    p0=[A0, B0, 0.0],
    sigma=sigma_temp,
    absolute_sigma=True,
)

A_sh, B_sh, C_sh = popt_sh
perr_sh = np.sqrt(np.diag(pcov_sh))
temp_pred_sh = np.asarray(thermistor_model(voltage_measured, *popt_sh))
residuals_sh = temp_reference - temp_pred_sh
rmse_sh = np.sqrt(np.mean(residuals_sh**2))

print(f"  1/T[K] = {A_sh:.5e} + {B_sh:.5e}*ln(V) + {C_sh:.3e}*ln(V)³")
# beta model: ln V = ln V_ref + beta (1/T - 1/T_ref)  =>  A = 1/T_ref - ln(V_ref)/beta, B = 1/beta, C = 0
print(
    f"  True (beta model): A = {1 / T_ref - np.log(V_ref) / beta:.5e}, B = {1 / beta:.5e}, C = 0"
)
print(f"  Fitted B +/- sigma: {B_sh:.5e} +/- {perr_sh[1]:.1e} (true {1 / beta:.5e})")
print(f"  RMSE: {rmse_sh:.3f} °C")
print(f"  Max residual: {np.max(np.abs(residuals_sh)):.3f} °C")


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


def reduced_chi2(residuals, n_par):
    return np.sum((residuals / sigma_temp) ** 2) / (len(residuals) - n_par)


rows = [
    ("Linear", residuals_linear, 2),
    ("Quadratic", residuals_quad, 3),
    ("Cubic", residuals_cubic, 4),
    ("Steinhart-Hart", residuals_sh, 3),
]
print(
    f"{'Model':<20} {'RMSE (°C)':<12} {'Max Error (°C)':<16} {'chi2/dof':<11} {'Parameters':<10}"
)
print("-" * 70)
for label, res, n_par in rows:
    print(
        f"{label:<20} {np.sqrt(np.mean(res**2)):<12.3f} {np.max(np.abs(res)):<16.3f} "
        f"{reduced_chi2(res, n_par):<11.2f} {n_par:<10}"
    )
print("-" * 70)
print(
    f"✓ Recommended: Steinhart-Hart model (chi2/dof = {reduced_chi2(residuals_sh, 3):.2f}); "
    "polynomials in V cannot follow the exponential thermistor response"
)


print("\n" + "=" * 70)
print("UNCERTAINTY ANALYSIS (Steinhart-Hart Model)")
print("=" * 70)

# Example: Measure unknown temperature
V_unknown = 3500.0  # mV (sensor reading)
V_unknown_err = 5.0  # mV (measurement uncertainty)

# Predict temperature
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
print(f"  Sensor voltage: {V_unknown:.1f} ± {V_unknown_err:.1f} mV")
print(f"  Predicted temperature: {T_predicted:.2f} ± {T_uncertainty:.2f} °C")
print(f"  (reading noise {T_read_err:.3f} °C, calibration fit {T_cal_err:.3f} °C)")
print(f"  Sensitivity: dT/dV = {dT_dV:.4f} °C/mV")


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
    label=f"Linear (RMSE={rmse_linear:.2f}°C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    quadratic_model(V_fine, *popt_quad),
    "--",
    linewidth=2,
    label=f"Quadratic (RMSE={rmse_quad:.2f}°C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    cubic_model(V_fine, *popt_cubic),
    "--",
    linewidth=2,
    label=f"Cubic (RMSE={rmse_cubic:.2f}°C)",
    alpha=0.7,
)
ax1.plot(
    V_fine,
    thermistor_model(V_fine, *popt_sh),
    "-",
    linewidth=2.5,
    label=f"Steinhart-Hart (RMSE={rmse_sh:.3f}°C)",
    color="green",
)
ax1.set_xlabel("Sensor Voltage (mV)", fontsize=12)
ax1.set_ylabel("Temperature (°C)", fontsize=12)
ax1.set_title("Calibration Curves", fontsize=14, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals comparison
ax2 = plt.subplot(3, 2, 2)
width = 0.2
x_pos = np.arange(len(temp_reference))
ax2.bar(x_pos - 1.5 * width, residuals_linear, width, label="Linear", alpha=0.7)
ax2.bar(x_pos - 0.5 * width, residuals_quad, width, label="Quadratic", alpha=0.7)
ax2.bar(x_pos + 0.5 * width, residuals_cubic, width, label="Cubic", alpha=0.7)
ax2.bar(x_pos + 1.5 * width, residuals_sh, width, label="Steinhart-Hart", color="green")
ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
ax2.axhline(-0.5, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Calibration Point Index")
ax2.set_ylabel("Residual (°C)")
ax2.set_title("Residuals: Model Comparison")
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Residuals vs temperature (Steinhart-Hart model)
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
ax3.set_xlabel("Reference Temperature (°C)")
ax3.set_ylabel("Residual (°C)")
ax3.set_title("Steinhart-Hart Model Residuals vs Temperature")
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals vs voltage (Steinhart-Hart model)
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
ax4.set_ylabel("Residual (°C)")
ax4.set_title("Steinhart-Hart Model Residuals vs Voltage")
ax4.grid(True, alpha=0.3)

# Plot 5: Histogram of residuals (Steinhart-Hart model)
ax5 = plt.subplot(3, 2, 5)
ax5.hist(residuals_sh, bins=8, alpha=0.7, edgecolor="black", color="green")
ax5.axvline(0, color="r", linestyle="--", linewidth=2)
ax5.axvline(
    np.mean(residuals_sh),
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Mean: {np.mean(residuals_sh):.3f}°C",
)
ax5.set_xlabel("Residual (°C)")
ax5.set_ylabel("Frequency")
ax5.set_title("Residual Distribution (Steinhart-Hart Model)")
ax5.legend()
ax5.grid(True, alpha=0.3, axis="y")

# Plot 6: Sensitivity analysis
ax6 = plt.subplot(3, 2, 6)
# Calculate sensitivity dT/dV across voltage range
V_sens = np.linspace(voltage_measured.min(), voltage_measured.max(), 100)
dV_small = 1.0  # mV
sensitivity = np.asarray(
    thermistor_model(V_sens + dV_small, *popt_sh)
    - thermistor_model(V_sens - dV_small, *popt_sh)
) / (2 * dV_small)
ax6.plot(V_sens, sensitivity, "g-", linewidth=2)
ax6.set_xlabel("Sensor Voltage (mV)")
ax6.set_ylabel("Sensitivity (°C/mV)")
ax6.set_title("Calibration Sensitivity (dT/dV)")
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "sensor_calibration"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
print(f"\n✅ Plot saved as {fig_dir / 'fig_01.png'}")
plt.close()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Calibration complete using the Steinhart-Hart model:")
print(f"  1/T(K) = {A_sh:.5e} + {B_sh:.5e}*ln(V) + {C_sh:.3e}*ln(V)³")
print("\nCalibration quality:")
print(f"  RMSE: {rmse_sh:.3f} °C")
print(f"  Max error: {np.max(np.abs(residuals_sh)):.3f} °C")
print(f"  Mean residual: {np.mean(residuals_sh):.3f} °C")
print(f"  Std residual: {np.std(residuals_sh):.3f} °C")
print(f"  chi2/dof: {reduced_chi2(residuals_sh, 3):.2f}")
print(f"\nValid range: {temp_reference.min():.0f} to {temp_reference.max():.0f} °C")
print(
    f"                ({voltage_measured.min():.0f} to {voltage_measured.max():.0f} mV)"
)
print("\nThis example demonstrates:")
print("  ✓ Non-linear sensor calibration with a physical (Steinhart-Hart) model")
print("  ✓ Model comparison (linear, quadratic, cubic, Steinhart-Hart)")
print("  ✓ Residual analysis for calibration quality")
print("  ✓ Uncertainty propagation to measurements")
print("  ✓ Sensitivity analysis")
print("=" * 70)
