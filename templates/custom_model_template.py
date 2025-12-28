"""Custom Model Template for NLSQ CLI Workflows.

This template demonstrates how to create custom model functions for use
with NLSQ curve fitting workflows. Copy this file and modify it to create
your own custom models.

IMPORTANT: Model functions should use JAX numpy (jax.numpy) instead of
regular numpy for GPU/TPU acceleration and automatic differentiation.

Structure
---------
A custom model file can contain:

1. **Model Function** (REQUIRED):
   The main fitting function with signature: f(x, param1, param2, ...)
   - First parameter must be x (independent variable)
   - Remaining parameters are fitting parameters

2. **estimate_p0 Function** (OPTIONAL):
   Estimates initial parameter values from data.
   Signature: estimate_p0(xdata, ydata) -> list[float]

3. **bounds Function** (OPTIONAL):
   Returns default parameter bounds.
   Signature: bounds() -> tuple[list[float], list[float]]

Usage
-----
1. Copy this file to your project directory
2. Modify the model function to match your physics/mathematics
3. Update estimate_p0 and bounds if needed
4. Reference in your workflow YAML:

   model:
     type: custom
     path: /path/to/your_model.py
     function: your_model_name

Example YAML Configuration
--------------------------
model:
  type: custom
  path: ./my_custom_model.py
  function: damped_oscillator
  # Optional: Override estimate_p0 with explicit p0
  p0: [1.0, 0.5, 1.0, 0.0]

Notes
-----
- Always import jax.numpy as jnp for model functions
- Use numpy (np) for estimate_p0 since it operates on input data
- Model functions are JIT-compiled, so avoid Python control flow
- Keep models simple and mathematically well-defined
"""

# =============================================================================
# Imports - Use JAX numpy for the model function
# =============================================================================

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Model Function (REQUIRED)
# =============================================================================


def damped_oscillator(
    x: np.ndarray, amplitude: float, decay: float, frequency: float, phase: float
) -> np.ndarray:
    """Damped sinusoidal oscillator model.

    Mathematical form:
        y = amplitude * exp(-decay * x) * cos(frequency * x + phase)

    This model describes systems like:
    - Mechanical vibrations with damping
    - RLC circuit transient response
    - Damped pendulum motion

    Parameters
    ----------
    x : array_like
        Independent variable (e.g., time)
    amplitude : float
        Initial amplitude of oscillation (amplitude > 0)
    decay : float
        Exponential decay rate (decay > 0)
    frequency : float
        Angular frequency of oscillation (rad/unit of x)
    phase : float
        Phase offset (radians)

    Returns
    -------
    y : array_like
        Dependent variable (displacement, voltage, etc.)

    Notes
    -----
    - Period: T = 2 * pi / frequency
    - Half-life of amplitude: t_half = ln(2) / decay
    - At x=0: y = amplitude * cos(phase)
    """
    # JAX arrays are duck-type compatible with numpy arrays
    return amplitude * jnp.exp(-decay * x) * jnp.cos(frequency * x + phase)  # type: ignore[return-value]


# =============================================================================
# Parameter Estimation (OPTIONAL)
# =============================================================================


def estimate_p0(xdata: np.ndarray, ydata: np.ndarray) -> list[float]:
    """Estimate initial parameters for the damped oscillator model.

    This function is called automatically when p0='auto' is specified
    in the workflow configuration.

    Strategy:
    - amplitude: Maximum absolute value of y
    - decay: Estimated from envelope decay
    - frequency: Estimated from zero crossings
    - phase: Estimated from initial value

    Parameters
    ----------
    xdata : ndarray
        Independent variable data (x values)
    ydata : ndarray
        Dependent variable data (y values)

    Returns
    -------
    p0 : list[float]
        Initial parameter estimates [amplitude, decay, frequency, phase]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Amplitude: maximum absolute value
    amplitude = np.max(np.abs(ydata))
    if amplitude == 0:
        amplitude = 1.0

    # Decay rate: estimate from envelope
    # Find peaks and fit exponential decay
    abs_y = np.abs(ydata)
    peak_indices = [
        i
        for i in range(1, len(abs_y) - 1)
        if abs_y[i] > abs_y[i - 1] and abs_y[i] > abs_y[i + 1]
    ]

    if len(peak_indices) >= 2:
        # Estimate decay from peak envelope
        x_peaks = xdata[peak_indices]
        y_peaks = abs_y[peak_indices]

        # Simple linear regression on log(y) vs x
        valid_mask = y_peaks > 0
        if np.sum(valid_mask) >= 2:
            log_y = np.log(y_peaks[valid_mask])
            x_valid = x_peaks[valid_mask]
            # Fit: log(y) = log(A) - decay * x
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            try:
                slope, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
                decay = max(-slope, 0.01)
            except np.linalg.LinAlgError:
                decay = 0.1
        else:
            decay = 0.1
    else:
        # Fallback: estimate from x range
        x_range = np.max(xdata) - np.min(xdata)
        decay = 1.0 / x_range if x_range > 0 else 0.1

    # Frequency: estimate from zero crossings
    zero_crossings = [
        xdata[i] - ydata[i] * (xdata[i + 1] - xdata[i]) / (ydata[i + 1] - ydata[i])
        for i in range(len(ydata) - 1)
        if ydata[i] * ydata[i + 1] < 0
    ]

    if len(zero_crossings) >= 2:
        # Period is approximately 2 * average crossing interval
        crossings = np.array(zero_crossings)
        periods = np.diff(crossings) * 2
        avg_period = np.mean(periods)
        frequency = 2 * np.pi / avg_period if avg_period > 0 else 1.0
    else:
        # Fallback: one complete oscillation over x range
        x_range = np.max(xdata) - np.min(xdata)
        frequency = 2 * np.pi / x_range if x_range > 0 else 1.0

    # Phase: estimate from initial value
    # At x=0: y = amplitude * cos(phase)
    # So: phase = arccos(y0 / amplitude)
    y0 = ydata[0]
    if amplitude > 0 and abs(y0 / amplitude) <= 1:
        phase = np.arccos(y0 / amplitude)
        # Determine sign from slope
        if len(ydata) > 1 and ydata[1] < ydata[0]:
            phase = -phase
    else:
        phase = 0.0

    return [float(amplitude), float(decay), float(frequency), float(phase)]


# =============================================================================
# Parameter Bounds (OPTIONAL)
# =============================================================================


def bounds() -> tuple[list[float], list[float]]:
    """Return default parameter bounds for the damped oscillator.

    These bounds constrain the optimizer to physically meaningful
    parameter ranges.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [amplitude, decay, frequency, phase]
    """
    lower = [0.0, 0.0, 0.0, -2 * np.pi]  # amplitude >= 0, decay >= 0, freq >= 0
    upper = [np.inf, np.inf, np.inf, 2 * np.pi]

    return (lower, upper)


# =============================================================================
# Additional Examples (Commented Out)
# =============================================================================

# Example: Simple power law with offset
# def power_law_offset(x, a, b, c):
#     """Power law with offset: y = a * x^b + c"""
#     return a * jnp.power(x, b) + c
#
# def estimate_p0(xdata, ydata):
#     # Estimate using log-log linear regression on (y - c) vs x
#     c = np.min(ydata)
#     y_adj = ydata - c
#     mask = (xdata > 0) & (y_adj > 0)
#     if np.sum(mask) < 2:
#         return [1.0, 1.0, float(c)]
#     log_x = np.log(xdata[mask])
#     log_y = np.log(y_adj[mask])
#     A = np.vstack([log_x, np.ones(len(log_x))]).T
#     b, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
#     return [float(np.exp(log_a)), float(b), float(c)]
#
# def bounds():
#     return ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf])

# Example: Lorentzian peak (spectroscopy)
# def lorentzian(x, amplitude, x0, gamma):
#     """Lorentzian peak: y = amplitude * gamma^2 / ((x - x0)^2 + gamma^2)"""
#     return amplitude * gamma**2 / ((x - x0)**2 + gamma**2)
