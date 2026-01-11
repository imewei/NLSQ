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
  auto_p0: true  # Uses estimate_p0() if defined

JIT Compilation Notes
---------------------
- Model functions are automatically JIT-compiled by NLSQ
- Avoid: Python if/else, for/while loops, list comprehensions
- Use instead: jnp.where(), jax.lax.cond(), jax.lax.fori_loop()
- All array operations must use jax.numpy, not numpy
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
# Model Function (REQUIRED) - Pure JAX Implementation
# =============================================================================


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
# Parameter Estimation (OPTIONAL) - NumPy for initialization
# =============================================================================


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
# Parameter Bounds (OPTIONAL)
# =============================================================================


def bounds() -> tuple[list[float], list[float]]:
    """Return default parameter bounds for the damped oscillator.

    These bounds constrain the optimizer to physically meaningful
    parameter ranges. Using jnp.inf for JAX compatibility.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [amplitude, decay, frequency, phase]
    """
    lower = [0.0, 0.0, 0.0, -2 * np.pi]
    upper = [float("inf"), float("inf"), float("inf"), 2 * np.pi]
    return (lower, upper)


# =============================================================================
# Additional JAX-First Model Examples
# =============================================================================


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
    """
    return amplitude * jnp.exp(-((x - center) ** 2) / (2 * sigma**2)) + baseline


def double_exponential(
    x: "jax.Array",
    a1: float,
    tau1: float,
    a2: float,
    tau2: float,
    offset: float,
) -> "jax.Array":
    """Bi-exponential decay model (fluorescence lifetime, kinetics).

    Mathematical form:
        y = a1 * exp(-x/τ1) + a2 * exp(-x/τ2) + offset

    Common applications:
    - Fluorescence decay with two lifetime components
    - Chemical kinetics with parallel reactions
    - Heat transfer with multiple time constants

    Parameters
    ----------
    x : jax.Array
        Independent variable (time)
    a1 : float
        Amplitude of first exponential
    tau1 : float
        Time constant of first exponential
    a2 : float
        Amplitude of second exponential
    tau2 : float
        Time constant of second exponential
    offset : float
        Baseline offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    return a1 * jnp.exp(-x / tau1) + a2 * jnp.exp(-x / tau2) + offset


def lorentzian_peak(
    x: "jax.Array",
    amplitude: float,
    center: float,
    gamma: float,
    baseline: float,
) -> "jax.Array":
    """Lorentzian (Cauchy) peak model (NMR, optical spectroscopy).

    Mathematical form:
        y = amplitude * γ² / ((x - center)² + γ²) + baseline

    Parameters
    ----------
    x : jax.Array
        Independent variable (frequency, wavelength)
    amplitude : float
        Peak height at center
    center : float
        Peak center position
    gamma : float
        Half-width at half-maximum (HWHM)
    baseline : float
        Background offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2) + baseline


def power_law(
    x: "jax.Array",
    coefficient: float,
    exponent: float,
    offset: float,
) -> "jax.Array":
    """Power law model (scaling phenomena, fractal analysis).

    Mathematical form:
        y = coefficient * x^exponent + offset

    Common applications:
    - Allometric scaling in biology
    - Turbulence energy spectra
    - Fractal dimension analysis

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

    Common applications:
    - Dose-response curves (pharmacology)
    - Population growth models
    - Neural network activation

    Parameters
    ----------
    x : jax.Array
        Independent variable (dose, time)
    amplitude : float
        Maximum response (saturation level)
    center : float
        Inflection point (EC50 for dose-response)
    rate : float
        Steepness of the transition
    baseline : float
        Minimum response

    Returns
    -------
    y : jax.Array
        Model values
    """
    return amplitude / (1 + jnp.exp(-rate * (x - center))) + baseline


# =============================================================================
# 2D Surface Model Example (for 2D/image fitting)
# =============================================================================


def gaussian_2d(
    xy: "jax.Array",
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    offset: float,
) -> "jax.Array":
    """2D Gaussian surface model (image fitting, beam profiling).

    Mathematical form:
        z = amplitude * exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²))) + offset

    Common applications:
    - Laser beam profiling
    - Point spread function fitting
    - 2D peak fitting in images

    Parameters
    ----------
    xy : jax.Array, shape (2, n)
        Coordinates: xy[0] = x values, xy[1] = y values
    amplitude : float
        Peak amplitude
    x0 : float
        Center x-coordinate
    y0 : float
        Center y-coordinate
    sigma_x : float
        Standard deviation in x direction
    sigma_y : float
        Standard deviation in y direction
    offset : float
        Background offset

    Returns
    -------
    z : jax.Array
        Surface values at each (x, y) coordinate

    Notes
    -----
    For 2D fitting, configure data.columns.z in your workflow YAML.
    """
    x, y = xy[0], xy[1]
    exponent = (x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2)
    return amplitude * jnp.exp(-exponent) + offset


# =============================================================================
# JAX Control Flow Examples (for complex models)
# =============================================================================


def piecewise_linear(
    x: "jax.Array",
    slope1: float,
    slope2: float,
    breakpoint: float,
    intercept: float,
) -> "jax.Array":
    """Piecewise linear model with one breakpoint.

    Uses jnp.where() for JIT-compatible conditional logic.

    Mathematical form:
        y = slope1 * x + intercept                     for x < breakpoint
        y = slope1 * bp + intercept + slope2 * (x-bp)  for x >= breakpoint

    Parameters
    ----------
    x : jax.Array
        Independent variable
    slope1 : float
        Slope before breakpoint
    slope2 : float
        Slope after breakpoint
    breakpoint : float
        x-value where slope changes
    intercept : float
        y-intercept (value at x=0)

    Returns
    -------
    y : jax.Array
        Model values
    """
    y_at_break = slope1 * breakpoint + intercept
    return jnp.where(
        x < breakpoint,
        slope1 * x + intercept,
        y_at_break + slope2 * (x - breakpoint),
    )


def safe_exponential_decay(
    x: "jax.Array",
    amplitude: float,
    decay_rate: float,
    offset: float,
) -> "jax.Array":
    """Exponential decay with numerical safety.

    Uses jnp.clip() to prevent overflow for large decay_rate * x values.

    Parameters
    ----------
    x : jax.Array
        Independent variable (time)
    amplitude : float
        Initial amplitude
    decay_rate : float
        Decay rate constant
    offset : float
        Asymptotic value

    Returns
    -------
    y : jax.Array
        Model values
    """
    # Clip exponent to prevent overflow (exp(-700) ≈ 0, exp(700) overflows)
    exponent = jnp.clip(-decay_rate * x, -500.0, 500.0)
    return amplitude * jnp.exp(exponent) + offset
