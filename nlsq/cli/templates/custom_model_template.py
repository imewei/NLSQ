"""Custom Model Template for NLSQ CLI Workflows (JAX-First).

This template demonstrates how to create JAX-optimized custom model functions
for use with NLSQ curve fitting workflows. All model functions use JAX for
GPU/TPU acceleration and automatic differentiation.

JAX-First Design Principles
----------------------------
1. Use jax.numpy (jnp) exclusively in model functions
2. Avoid Python control flow (if/else, for loops) in JIT-compiled code
3. Use jax.lax.cond, jax.lax.fori_loop, or jnp.where for conditionals
4. Keep functions pure (no side effects, no global state mutation)
5. Use vectorized operations instead of explicit loops

Structure
---------
A custom model file can contain:

1. **Model Function** (REQUIRED):
   The main fitting function with signature: f(x, param1, param2, ...)
   - First parameter must be x (independent variable as jax.Array)
   - Remaining parameters are fitting parameters (floats)
   - Returns jax.Array

2. **estimate_p0 Function** (OPTIONAL):
   Estimates initial parameter values from data.
   Signature: estimate_p0(xdata, ydata) -> list[float]
   Note: Can use numpy here since it runs once at initialization.

3. **bounds Function** (OPTIONAL):
   Returns default parameter bounds.
   Signature: bounds() -> tuple[list[float], list[float]]

4. **parameter_names Function** (OPTIONAL):
   Returns human-readable parameter names for reporting.
   Signature: parameter_names() -> list[str]

Usage
-----
1. Copy this file to your project directory
2. Modify the model function to match your physics/mathematics
3. Update estimate_p0 and bounds if needed
4. Reference in your workflow YAML:

   model:
     type: custom
     custom:
       file: /path/to/your_model.py
       function: your_model_name

Example YAML Configuration
--------------------------
model:
  type: custom
  custom:
    file: ./my_custom_model.py
    function: damped_oscillator
  auto_p0: true      # Uses estimate_p0() if defined
  auto_bounds: true  # Uses bounds() if defined

JIT Compilation Notes
---------------------
- Model functions are automatically JIT-compiled by NLSQ
- Avoid: Python if/else, for/while loops, list comprehensions
- Use instead: jnp.where(), jax.lax.cond(), jax.lax.fori_loop()
- All array operations must use jax.numpy, not numpy

Common Pitfalls
---------------
1. Using numpy instead of jax.numpy in model functions
2. Using Python if/else instead of jnp.where()
3. Creating side effects (printing, file I/O) in model functions
4. Using mutable default arguments
5. Forgetting that division by zero returns inf, not error
"""

# =============================================================================
# Imports - JAX-First
# =============================================================================

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import jax

# =============================================================================
# MAIN MODEL: Damped Oscillator (REQUIRED)
# =============================================================================
# This is the primary model function that NLSQ will fit to your data.
# Rename this function and modify as needed for your application.


def damped_oscillator(
    x: "jax.Array",
    amplitude: float,
    decay: float,
    frequency: float,
    phase: float,
) -> "jax.Array":
    """Damped sinusoidal oscillator model (JAX-optimized).

    Mathematical form:
        y = amplitude * exp(-decay * x) * cos(frequency * x + phase)

    This model describes systems like:
    - Mechanical vibrations with damping
    - RLC circuit transient response
    - Damped pendulum motion

    Parameters
    ----------
    x : jax.Array
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
    y : jax.Array
        Dependent variable (displacement, voltage, etc.)

    Notes
    -----
    - Period: T = 2π / frequency
    - Half-life of amplitude: t_half = ln(2) / decay
    - At x=0: y = amplitude * cos(phase)
    - This function is JIT-compiled automatically by NLSQ
    """
    return amplitude * jnp.exp(-decay * x) * jnp.cos(frequency * x + phase)


# =============================================================================
# PARAMETER ESTIMATION (OPTIONAL)
# =============================================================================
# This function estimates initial parameters from data when auto_p0=true.
# Uses NumPy for compatibility with input data formats.


def estimate_p0(xdata: np.ndarray, ydata: np.ndarray) -> list[float]:
    """Estimate initial parameters for the damped oscillator model.

    This function is called once at initialization when auto_p0=true.
    Uses NumPy for compatibility with input data formats.

    Strategy:
    - amplitude: Maximum absolute value of y
    - decay: Estimated from envelope decay via linear regression
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
    xdata = np.asarray(xdata, dtype=np.float64)
    ydata = np.asarray(ydata, dtype=np.float64)

    # Amplitude: maximum absolute value
    amplitude = float(np.max(np.abs(ydata)))
    if amplitude == 0:
        amplitude = 1.0

    # Decay rate: estimate from envelope using vectorized peak detection
    abs_y = np.abs(ydata)
    # Vectorized local maxima detection
    is_peak = np.zeros(len(abs_y), dtype=bool)
    if len(abs_y) > 2:
        is_peak[1:-1] = (abs_y[1:-1] > abs_y[:-2]) & (abs_y[1:-1] > abs_y[2:])
    peak_indices = np.where(is_peak)[0]

    if len(peak_indices) >= 2:
        x_peaks = xdata[peak_indices]
        y_peaks = abs_y[peak_indices]
        valid_mask = y_peaks > 0

        if np.sum(valid_mask) >= 2:
            log_y = np.log(y_peaks[valid_mask])
            x_valid = x_peaks[valid_mask]
            # Linear regression: log(y) = log(A) - decay * x
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            result = np.linalg.lstsq(A, log_y, rcond=None)
            slope = result[0][0]
            decay = float(max(-slope, 0.01))
        else:
            decay = 0.1
    else:
        x_range = float(np.ptp(xdata))  # ptp = max - min
        decay = 1.0 / x_range if x_range > 0 else 0.1

    # Frequency: vectorized zero crossing detection
    sign_changes = ydata[:-1] * ydata[1:] < 0
    if np.sum(sign_changes) >= 2:
        # Interpolate zero crossing positions
        idx = np.where(sign_changes)[0]
        # Linear interpolation for crossing positions
        x0, x1 = xdata[idx], xdata[idx + 1]
        y0, y1 = ydata[idx], ydata[idx + 1]
        crossings = x0 - y0 * (x1 - x0) / (y1 - y0)
        periods = np.diff(crossings) * 2
        avg_period = float(np.mean(periods))
        frequency = 2 * np.pi / avg_period if avg_period > 0 else 1.0
    else:
        x_range = float(np.ptp(xdata))
        frequency = 2 * np.pi / x_range if x_range > 0 else 1.0

    # Phase: estimate from initial value
    y0 = ydata[0]
    ratio = y0 / amplitude if amplitude > 0 else 0.0
    if abs(ratio) <= 1:
        phase = float(np.arccos(np.clip(ratio, -1, 1)))
        # Determine sign from slope
        if len(ydata) > 1 and ydata[1] < ydata[0]:
            phase = -phase
    else:
        phase = 0.0

    return [amplitude, decay, frequency, phase]


# =============================================================================
# PARAMETER BOUNDS (OPTIONAL)
# =============================================================================
# These bounds constrain the optimizer to physically meaningful ranges.


def bounds() -> tuple[list[float], list[float]]:
    """Return default parameter bounds for the damped oscillator.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [amplitude, decay, frequency, phase]
    """
    lower = [0.0, 0.0, 0.0, -2 * np.pi]
    upper = [float("inf"), float("inf"), float("inf"), 2 * np.pi]
    return (lower, upper)


# =============================================================================
# PARAMETER NAMES (OPTIONAL)
# =============================================================================
# Human-readable names for parameter reporting.


def parameter_names() -> list[str]:
    """Return parameter names for reporting.

    Returns
    -------
    names : list[str]
        Human-readable parameter names
    """
    return ["amplitude", "decay", "frequency", "phase"]


# =============================================================================
# ADDITIONAL MODEL EXAMPLES
# =============================================================================
# A few more patterns worth knowing before you write your own model:
# a peaked curve (gaussian_peak), a decay curve (exponential_decay), an
# S-shaped curve (sigmoid), and a scaling-law curve (power_law). Copy
# whichever is closest to your data's shape.


def gaussian_peak(
    x: "jax.Array",
    amplitude: float,
    center: float,
    sigma: float,
    baseline: float,
) -> "jax.Array":
    """Gaussian peak model (spectroscopy, chromatography).

    Mathematical form:
        y = amplitude * exp(-(x - center)² / (2σ²)) + baseline

    Parameters
    ----------
    x : jax.Array
        Independent variable (wavelength, time, etc.)
    amplitude : float
        Peak height above baseline
    center : float
        Peak center position
    sigma : float
        Standard deviation (width parameter)
    baseline : float
        Constant background offset

    Returns
    -------
    y : jax.Array
        Model values

    Notes
    -----
    FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    """
    return amplitude * jnp.exp(-((x - center) ** 2) / (2 * sigma**2)) + baseline


def exponential_decay(
    x: "jax.Array",
    amplitude: float,
    decay_rate: float,
    offset: float,
) -> "jax.Array":
    """Single exponential decay model.

    Mathematical form:
        y = amplitude * exp(-decay_rate * x) + offset

    Parameters
    ----------
    x : jax.Array
        Time variable
    amplitude : float
        Initial amplitude (A0)
    decay_rate : float
        Decay rate constant (k)
    offset : float
        Asymptotic value (baseline)

    Returns
    -------
    y : jax.Array
        Model values

    Notes
    -----
    Half-life: t_half = ln(2) / decay_rate
    """
    return amplitude * jnp.exp(-decay_rate * x) + offset


def sigmoid(
    x: "jax.Array",
    amplitude: float,
    center: float,
    rate: float,
    baseline: float,
) -> "jax.Array":
    """Logistic sigmoid model (dose-response, growth curves).

    Mathematical form:
        y = amplitude / (1 + exp(-rate * (x - center))) + baseline

    Parameters
    ----------
    x : jax.Array
        Independent variable (dose, time)
    amplitude : float
        Maximum response (saturation level)
    center : float
        Inflection point (EC50 for dose-response)
    rate : float
        Steepness of the transition (Hill slope)
    baseline : float
        Minimum response

    Returns
    -------
    y : jax.Array
        Model values

    Notes
    -----
    For dose-response: center = EC50, rate = Hill coefficient
    """
    return amplitude / (1 + jnp.exp(-rate * (x - center))) + baseline


def power_law(
    x: "jax.Array",
    coefficient: float,
    exponent: float,
    offset: float,
) -> "jax.Array":
    """Power law model (scaling phenomena, fractal analysis).

    Mathematical form:
        y = coefficient * x^exponent + offset

    Parameters
    ----------
    x : jax.Array
        Independent variable (must be positive for non-integer exponents)
    coefficient : float
        Scaling coefficient
    exponent : float
        Power law exponent
    offset : float
        Baseline offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    return coefficient * jnp.power(x, exponent) + offset


# =============================================================================
# HELPER FUNCTIONS FOR PARAMETER ESTIMATION
# =============================================================================
# These utilities can help you write estimate_p0 functions for your models.


def estimate_gaussian_p0(xdata: np.ndarray, ydata: np.ndarray) -> list[float]:
    """Estimate initial parameters for a Gaussian peak.

    Returns [amplitude, center, sigma, baseline]
    """
    baseline = float(np.min(ydata))
    y_corrected = ydata - baseline
    amplitude = float(np.max(y_corrected))

    # Center: weighted average
    if amplitude > 0:
        center = float(np.average(xdata, weights=np.maximum(y_corrected, 0)))
    else:
        center = float(np.mean(xdata))

    # Sigma: estimate from FWHM
    half_max = amplitude / 2
    above_half = y_corrected > half_max
    if np.sum(above_half) >= 2:
        x_above = xdata[above_half]
        fwhm = float(np.max(x_above) - np.min(x_above))
        sigma = fwhm / 2.355  # FWHM = 2.355 * sigma
    else:
        sigma = float(np.ptp(xdata)) / 6  # Rough estimate

    return [amplitude, center, max(sigma, 1e-6), baseline]


def estimate_exponential_p0(xdata: np.ndarray, ydata: np.ndarray) -> list[float]:
    """Estimate initial parameters for exponential decay.

    Returns [amplitude, decay_rate, offset]
    """
    offset = float(np.min(ydata))
    y_corrected = ydata - offset
    amplitude = float(np.max(y_corrected))

    # Decay rate: linear regression on log(y)
    valid = y_corrected > 0.1 * amplitude
    if np.sum(valid) >= 2:
        log_y = np.log(y_corrected[valid])
        x_valid = xdata[valid]
        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
        result = np.linalg.lstsq(A, log_y, rcond=None)
        decay_rate = float(max(-result[0][0], 0.01))
    else:
        x_range = float(np.ptp(xdata))
        decay_rate = 1.0 / x_range if x_range > 0 else 0.1

    return [amplitude, decay_rate, offset]
