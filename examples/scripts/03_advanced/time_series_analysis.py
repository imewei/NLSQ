"""
Converted from time_series_analysis.ipynb

This script was automatically generated from a Jupyter notebook.
Plots are saved to the figures/ directory instead of displayed inline.
"""

# ======================================================================
# # Time Series Analysis with NLSQ
#
# **Level**: Intermediate to Advanced
# **Time**: 35-45 minutes
# **Prerequisites**: NLSQ Quickstart
#
# ## Overview
#
# Time series analysis involves fitting models to sequential data with temporal dependencies. While traditional approaches use ARIMA or state-space models, **NLSQ excels at fitting parametric trend and seasonal components** with physical interpretations.
#
# ### What You'll Learn
#
# 1. **Trend Fitting**: Polynomial, exponential, and logistic growth models
# 2. **Seasonal Decomposition**: Fourier series for periodic components
# 3. **Combined Models**: Trend + seasonality + noise
# 4. **Forecasting**: Extrapolation with uncertainty quantification
# 5. **Autocorrelation**: Checking residual independence
#
# ### When to Use NLSQ for Time Series
#
# **NLSQ is ideal when:**
# - You have a **physical or mechanistic model** for the process (e.g., exponential growth, damped oscillations)
# - You need **interpretable parameters** (e.g., growth rate, decay constant, period)
# - Data exhibits **strong deterministic trends** or **periodic patterns**
# - You want to **leverage JAX's speed** for large datasets or batch fitting
#
# **Use traditional time series methods (ARIMA, Prophet) when:**
# - Data is dominated by **stochastic processes** without clear parametric form
# - You need **complex autoregressive structures**
# - **Missing data** or **irregular sampling** is common
#
# ### Applications
#
# - **Scientific**: Radioactive decay, population dynamics, chemical kinetics
# - **Engineering**: Sensor drift, system response, degradation models
# - **Environmental**: Temperature cycles, tidal patterns, climate trends
# - **Business**: Product lifecycle, seasonal sales (with physical constraints)
# ======================================================================
# Configure matplotlib for inline plotting in VS Code/Jupyter
# MUST come before importing matplotlib
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nlsq import CurveFit

# Plotting configuration
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

warnings.filterwarnings("ignore")

print("✓ Imports successful")


# ======================================================================
# ## Part 1: Trend Fitting
#
# We'll explore different growth models commonly found in time series data.
# ======================================================================


# Generate synthetic growth data (logistic curve)

# Time points (days)
t_data = np.linspace(0, 100, 150)

# True logistic growth parameters
L_true = 1000.0  # Carrying capacity
k_true = 0.08  # Growth rate
t0_true = 40.0  # Inflection point


def logistic_growth(t, L, k, t0):
    """Logistic growth model (S-curve).

    Common in population dynamics, product adoption, epidemic spread.

    Parameters
    ----------
    t : array_like
        Time
    L : float
        Carrying capacity (asymptotic maximum)
    k : float
        Growth rate
    t0 : float
        Inflection point (time of maximum growth rate)

    Returns
    -------
    y : array_like
        Population/quantity at time t
    """
    return L / (1.0 + jnp.exp(-k * (t - t0)))


# Generate clean signal
y_true = logistic_growth(t_data, L_true, k_true, t0_true)

# Add noise (proportional to signal level)
np.random.seed(42)
noise = np.random.normal(0, 30, len(t_data))
y_observed = y_true + noise

print("✓ Generated logistic growth data")
print(f"  Time range: {t_data.min():.0f} - {t_data.max():.0f} days")
print(f"  True parameters: L={L_true}, k={k_true}, t0={t0_true}")


# Fit logistic growth model

cf = CurveFit()

# Initial guess (from visual inspection)
p0 = [900.0, 0.1, 45.0]  # L, k, t0

# Bounds (physical constraints)
bounds = ([0, 0, 0], [2000, 1.0, 100])  # L, k, t0 must be positive

# Fit
popt, pcov = cf.curve_fit(
    logistic_growth, jnp.array(t_data), jnp.array(y_observed), p0=p0, bounds=bounds
)

L_fit, k_fit, t0_fit = popt
L_err, k_err, t0_err = np.sqrt(np.diag(pcov))

print("Fitted Parameters:")
print(f"  Carrying capacity (L): {L_fit:.1f} ± {L_err:.1f} (true: {L_true})")
print(f"  Growth rate (k): {k_fit:.3f} ± {k_err:.3f} (true: {k_true})")
print(f"  Inflection point (t0): {t0_fit:.1f} ± {t0_err:.1f} days (true: {t0_true})")

# Calculate derived quantities
max_growth_rate = k_fit * L_fit / 4  # dN/dt at t0
print("\nDerived:")
print(f"  Maximum growth rate: {max_growth_rate:.1f} units/day")
print(f"  Doubling time (early phase): {np.log(2) / k_fit:.1f} days")


# Visualize growth fit with forecast

# Extended time for forecasting
t_extended = np.linspace(0, 150, 300)
y_fit = logistic_growth(jnp.array(t_extended), L_fit, k_fit, t0_fit)

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Main plot: data + fit + forecast
ax1.plot(t_data, y_observed, "o", alpha=0.5, label="Observed data", ms=4)
ax1.plot(t_extended, y_fit, "r-", lw=2, label="Fitted logistic model")
ax1.axvline(t_data.max(), color="gray", ls="--", lw=1, label="Forecast boundary")
ax1.axhline(
    L_fit, color="green", ls=":", lw=1.5, label=f"Carrying capacity: {L_fit:.0f}"
)
ax1.axvline(t0_fit, color="orange", ls=":", lw=1.5, label=f"Inflection: {t0_fit:.0f} d")

ax1.set_xlabel("Time (days)")
ax1.set_ylabel("Population / Quantity")
ax1.set_title("Logistic Growth Fitting and Forecasting")
ax1.legend()
ax1.grid(alpha=0.3)

# Growth rate plot
growth_rate = k_fit * y_fit * (1 - y_fit / L_fit)  # dN/dt
ax2.plot(t_extended, growth_rate, "b-", lw=2)
ax2.axvline(t0_fit, color="orange", ls=":", lw=1.5, label="Maximum growth rate")
ax2.axvline(t_data.max(), color="gray", ls="--", lw=1)
ax2.set_xlabel("Time (days)")
ax2.set_ylabel("Growth Rate (dN/dt)")
ax2.set_title("Instantaneous Growth Rate")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "time_series_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_01.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Growth trend analysis complete")


# ======================================================================
# ## Part 2: Seasonal Decomposition with Fourier Series
#
# Many time series exhibit periodic patterns (daily, weekly, annual cycles). We can model these using Fourier series.
# ======================================================================


# Generate seasonal data (temperature with annual cycle)

# Daily temperature over 3 years
days = np.linspace(0, 3 * 365, 3 * 365)

# True components
annual_mean = 15.0  # °C
annual_amplitude = 10.0  # °C
annual_period = 365.25  # days
trend_slope = 0.01  # °C/day (climate warming)

# Generate signal: trend + annual cycle + noise
trend_component = annual_mean + trend_slope * days
seasonal_component = annual_amplitude * np.sin(2 * np.pi * days / annual_period)
temp_true = trend_component + seasonal_component

# Add weather noise
np.random.seed(123)
temp_observed = temp_true + np.random.normal(0, 2.0, len(days))

print("✓ Generated seasonal temperature data")
print(f"  Duration: {len(days)} days ({len(days) / 365:.1f} years)")
print(f"  True parameters: mean={annual_mean}°C, amplitude={annual_amplitude}°C")
print(f"  Warming trend: {trend_slope * 365:.2f}°C/year")


# Fit trend + seasonal model


def trend_seasonal_model(t, mean, trend, amplitude, period, phase):
    """Combined linear trend and sinusoidal seasonal component.

    Parameters
    ----------
    t : array_like
        Time (days)
    mean : float
        Baseline level
    trend : float
        Linear trend (units per day)
    amplitude : float
        Seasonal amplitude
    period : float
        Seasonal period (days)
    phase : float
        Phase shift (radians)

    Returns
    -------
    y : array_like
        Modeled values
    """
    trend_part = mean + trend * t
    seasonal_part = amplitude * jnp.sin(2 * jnp.pi * t / period + phase)
    return trend_part + seasonal_part


# Initial guess
p0_seasonal = [15.0, 0.0, 8.0, 365.0, 0.0]  # mean, trend, amplitude, period, phase

# Bounds (constrain period to be near annual)
bounds_seasonal = (
    [-50, -0.1, 0, 300, -2 * np.pi],  # Lower
    [50, 0.1, 20, 400, 2 * np.pi],  # Upper
)

# Fit
popt_seasonal, pcov_seasonal = cf.curve_fit(
    trend_seasonal_model,
    jnp.array(days),
    jnp.array(temp_observed),
    p0=p0_seasonal,
    bounds=bounds_seasonal,
)

mean_fit, trend_fit, amp_fit, period_fit, phase_fit = popt_seasonal
errors = np.sqrt(np.diag(pcov_seasonal))

print("Fitted Seasonal Parameters:")
print(f"  Baseline: {mean_fit:.2f} ± {errors[0]:.2f} °C")
print(
    f"  Trend: {trend_fit:.4f} ± {errors[1]:.4f} °C/day = {trend_fit * 365:.2f} °C/year"
)
print(f"  Amplitude: {amp_fit:.2f} ± {errors[2]:.2f} °C")
print(f"  Period: {period_fit:.1f} ± {errors[3]:.1f} days")
print(f"  Phase: {phase_fit:.3f} ± {errors[4]:.3f} rad")


# Decompose time series into components

# Fitted components
trend_fitted = mean_fit + trend_fit * days
seasonal_fitted = amp_fit * np.sin(2 * np.pi * days / period_fit + phase_fit)
total_fitted = trend_fitted + seasonal_fitted
residuals = temp_observed - total_fitted

# Plot decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# Original data
axes[0].plot(days / 365, temp_observed, "o", alpha=0.3, ms=2, label="Observed")
axes[0].plot(days / 365, total_fitted, "r-", lw=1.5, label="Fitted model")
axes[0].set_ylabel("Temperature (°C)")
axes[0].set_title("Original Time Series")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Trend component
axes[1].plot(days / 365, trend_fitted, "b-", lw=2)
axes[1].set_ylabel("Trend (°C)")
axes[1].set_title(f"Trend Component (slope: {trend_fit * 365:.3f} °C/year)")
axes[1].grid(alpha=0.3)

# Seasonal component
axes[2].plot(days / 365, seasonal_fitted, "g-", lw=1.5)
axes[2].set_ylabel("Seasonal (°C)")
axes[2].set_title(
    f"Seasonal Component (period: {period_fit:.1f} days, amplitude: {amp_fit:.1f} °C)"
)
axes[2].grid(alpha=0.3)

# Residuals
axes[3].plot(days / 365, residuals, "o", alpha=0.4, ms=2, color="gray")
axes[3].axhline(0, color="k", ls="--", lw=1)
axes[3].set_ylabel("Residual (°C)")
axes[3].set_xlabel("Time (years)")
axes[3].set_title(f"Residuals (std: {np.std(residuals):.2f} °C, should be white noise)")
axes[3].grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "time_series_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_02.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Seasonal decomposition complete")


# ======================================================================
# ## Part 3: Forecasting with Uncertainty
#
# Extrapolate the fitted model into the future with prediction intervals.
# ======================================================================


# Generate forecast with uncertainty bands

# Forecast horizon: 1 additional year
days_forecast = np.linspace(0, 4 * 365, 4 * 365)
forecast_boundary = 3 * 365

# Point forecast
temp_forecast = trend_seasonal_model(jnp.array(days_forecast), *popt_seasonal)

# Uncertainty from parameter covariance (simplified)
# Full approach: Monte Carlo sampling from parameter distribution
n_samples = 200
param_samples = np.random.multivariate_normal(
    popt_seasonal, pcov_seasonal, size=n_samples
)

forecast_samples = np.array(
    [
        trend_seasonal_model(jnp.array(days_forecast), *params)
        for params in param_samples
    ]
)

# Calculate prediction intervals
forecast_mean = np.mean(forecast_samples, axis=0)
forecast_std = np.std(forecast_samples, axis=0)
forecast_lower = np.percentile(forecast_samples, 2.5, axis=0)  # 95% PI
forecast_upper = np.percentile(forecast_samples, 97.5, axis=0)

# Add residual uncertainty
residual_std = np.std(residuals)
forecast_lower_total = forecast_lower - 2 * residual_std
forecast_upper_total = forecast_upper + 2 * residual_std

print(f"✓ Generated {len(days_forecast) - len(days)} day forecast")
print(
    f"  Forecast period: {len(days) / 365:.1f} - {len(days_forecast) / 365:.1f} years"
)
print(f"  Prediction interval width: {2 * residual_std:.1f} °C (±1σ residuals)")


# Visualize forecast with prediction intervals

fig, ax = plt.subplots(figsize=(14, 6))

# Historical data
ax.plot(
    days / 365, temp_observed, "o", alpha=0.3, ms=2, color="steelblue", label="Observed"
)

# Fitted model (historical)
ax.plot(
    days / 365,
    total_fitted,
    "r-",
    lw=2,
    label="Fitted model",
    alpha=0.8,
)

# Forecast (future)
forecast_mask = days_forecast > forecast_boundary
ax.plot(
    days_forecast[forecast_mask] / 365,
    forecast_mean[forecast_mask],
    "r--",
    lw=2,
    label="Forecast",
)

# Prediction intervals (parameter uncertainty only)
ax.fill_between(
    days_forecast[forecast_mask] / 365,
    forecast_lower[forecast_mask],
    forecast_upper[forecast_mask],
    alpha=0.3,
    color="red",
    label="95% PI (parameter)",
)

# Total prediction intervals (parameter + residual uncertainty)
ax.fill_between(
    days_forecast[forecast_mask] / 365,
    forecast_lower_total[forecast_mask],
    forecast_upper_total[forecast_mask],
    alpha=0.15,
    color="orange",
    label="95% PI (total)",
)

# Forecast boundary
ax.axvline(forecast_boundary / 365, color="gray", ls="--", lw=2, label="Forecast start")

ax.set_xlabel("Time (years)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.set_title("Time Series Forecast with Uncertainty Quantification", fontsize=14)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "time_series_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_03.png", dpi=300, bbox_inches="tight")
plt.close()

print("✓ Forecast visualization complete")


# ======================================================================
# ## Part 4: Residual Diagnostics (Autocorrelation)
#
# For valid inference, residuals should be uncorrelated (white noise). We check this with autocorrelation analysis.
# ======================================================================


# Calculate and plot autocorrelation of residuals


def autocorrelation(x, max_lag=50):
    """Calculate autocorrelation function (ACF).

    Parameters
    ----------
    x : array_like
        Time series (residuals)
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    lags : array
        Lag values
    acf : array
        Autocorrelation values
    """
    x_centered = x - np.mean(x)
    c0 = np.dot(x_centered, x_centered) / len(x)

    lags = np.arange(0, max_lag + 1)
    acf = np.zeros(len(lags))

    for i, lag in enumerate(lags):
        if lag == 0:
            acf[i] = 1.0
        else:
            c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / len(x)
            acf[i] = c_lag / c0

    return lags, acf


# Calculate ACF
lags, acf_values = autocorrelation(residuals, max_lag=60)

# Significance bounds (95% confidence for white noise)
conf_bound = 1.96 / np.sqrt(len(residuals))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.stem(lags, acf_values, basefmt=" ", linefmt="C0-", markerfmt="C0o")
ax.axhline(0, color="k", lw=1)
ax.axhline(conf_bound, color="r", ls="--", lw=1.5, label="95% confidence")
ax.axhline(-conf_bound, color="r", ls="--", lw=1.5)
ax.fill_between(
    lags, -conf_bound, conf_bound, alpha=0.2, color="red", label="White noise region"
)

ax.set_xlabel("Lag (days)")
ax.set_ylabel("Autocorrelation")
ax.set_title("Autocorrelation Function (ACF) of Residuals")
ax.set_xlim(-1, max(lags) + 1)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
# Save figure to file
fig_dir = Path(__file__).parent / "figures" / "time_series_analysis"
fig_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_dir / "fig_04.png", dpi=300, bbox_inches="tight")
plt.close()

# Check for significant autocorrelation
significant_lags = np.sum(np.abs(acf_values[1:]) > conf_bound)  # Exclude lag 0
print("✓ Autocorrelation analysis complete")
print(
    f"  Significant lags (95% level): {significant_lags} / {len(lags) - 1} ({significant_lags / (len(lags) - 1) * 100:.1f}%)"
)
if significant_lags / (len(lags) - 1) < 0.05:
    print("  ✓ Residuals consistent with white noise (good fit)")
else:
    print(
        "  ⚠ Significant autocorrelation detected: consider more complex model or autoregressive errors"
    )


# ======================================================================
# ## Summary and Best Practices
#
# ### When to Use NLSQ for Time Series
#
# | **Use Case** | **NLSQ Strength** | **Alternative** |
# |--------------|-------------------|------------------|
# | Physical growth models (exponential, logistic) | ✅ Excellent (interpretable parameters) | ARIMA (less interpretable) |
# | Periodic data with known/unknown period | ✅ Good (Fourier series) | Seasonal decomposition (STL, Prophet) |
# | Trend + seasonality | ✅ Good (combined parametric model) | Prophet, TBATS |
# | Autoregressive processes (ARMA) | ❌ Poor (not designed for this) | ARIMA, SARIMA |
# | Irregular sampling | ✅ Excellent (handles any time grid) | Interpolation + ARIMA |
# | Large datasets (millions of points) | ✅ Excellent (JAX GPU acceleration) | Dask + statsmodels |
#
# ### Key Takeaways
#
# 1. **Model Selection**: Choose parametric forms based on domain knowledge
#    - Growth: Exponential, logistic, Gompertz
#    - Decay: Exponential, power-law
#    - Periodic: Fourier series (sum of sines/cosines)
#
# 2. **Forecasting Uncertainty**
#    - **Parameter uncertainty**: From covariance matrix (Monte Carlo sampling)
#    - **Model uncertainty**: Residual standard deviation
#    - **Total**: Combine both sources for realistic prediction intervals
#
# 3. **Diagnostics**
#    - **Residuals**: Should be centered at zero, no trend
#    - **Autocorrelation**: Should be within confidence bounds (white noise)
#    - **Heteroscedasticity**: Check if residual variance changes over time
#
# 4. **Multi-Seasonal Data**
#    - Use multiple sinusoids: `amp1 * sin(2π t / P1) + amp2 * sin(2π t / P2)`
#    - Example: Daily + weekly cycles in energy consumption
#
# ### Production Code Template
#
# ```python
# from nlsq import CurveFit
# import jax.numpy as jnp
#
# def forecast_time_series(t, y, forecast_days=30):
#     """Fit trend+seasonal model and forecast."""
#
#     # Model
#     def model(t, mean, trend, amp, period, phase):
#         return mean + trend * t + amp * jnp.sin(2 * jnp.pi * t / period + phase)
#
#     # Fit
#     cf = CurveFit()
#     p0 = [jnp.mean(y), 0.0, jnp.std(y) / 2, 365.0, 0.0]
#     popt, pcov = cf.curve_fit(model, jnp.array(t), jnp.array(y), p0=p0)
#
#     # Forecast
#     t_future = jnp.arange(t[-1] + 1, t[-1] + 1 + forecast_days)
#     y_forecast = model(t_future, *popt)
#
#     # Uncertainty (simplified)
#     residual_std = jnp.std(y - model(jnp.array(t), *popt))
#     forecast_uncertainty = residual_std
#
#     return t_future, y_forecast, forecast_uncertainty
# ```
#
# ### Next Steps
#
# - **Advanced Seasonality**: Multi-frequency Fourier series for complex cycles
# - **State-Space Models**: Kalman filtering with NLSQ parameter estimation
# - **Batch Processing**: Fit thousands of time series in parallel with `jax.vmap`
# - **Hybrid Models**: Combine NLSQ (trend/seasonal) with ARIMA (residual modeling)
#
# ### References
#
# 1. **Time Series Analysis**: Chatfield, *The Analysis of Time Series* (2004)
# 2. **Forecasting**: Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (2021)
# 3. **Related Examples**:
#    - `gallery/biology/growth_curves.py` - Bacterial growth fitting
#    - `gallery/physics/damped_oscillation.py` - Oscillatory time series
#    - `advanced_features_demo.ipynb` - Robustness for outliers in time series
# ======================================================================
