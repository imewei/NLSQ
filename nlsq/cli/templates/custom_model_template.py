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
# Below are additional model examples for common scientific applications.
# Copy and modify these for your specific use case.


# -----------------------------------------------------------------------------
# SPECTROSCOPY MODELS
# -----------------------------------------------------------------------------


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

    Notes
    -----
    FWHM = 2 * gamma
    """
    return amplitude * gamma**2 / ((x - center) ** 2 + gamma**2) + baseline


def voigt_peak(
    x: "jax.Array",
    amplitude: float,
    center: float,
    sigma: float,
    gamma: float,
    baseline: float,
) -> "jax.Array":
    """Pseudo-Voigt peak model (X-ray diffraction, Raman spectroscopy).

    Approximation of Voigt profile using weighted sum of Gaussian and Lorentzian.
    Uses mixing parameter eta computed from sigma and gamma.

    Mathematical form:
        y = amplitude * (eta * L(x) + (1-eta) * G(x)) + baseline

    Parameters
    ----------
    x : jax.Array
        Independent variable
    amplitude : float
        Peak amplitude
    center : float
        Peak center
    sigma : float
        Gaussian width parameter
    gamma : float
        Lorentzian width parameter
    baseline : float
        Background offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    # Compute total FWHM and mixing parameter
    fwhm_g = 2.355 * sigma
    fwhm_l = 2.0 * gamma
    fwhm = (
        fwhm_g**5
        + 2.69269 * fwhm_g**4 * fwhm_l
        + 2.42843 * fwhm_g**3 * fwhm_l**2
        + 4.47163 * fwhm_g**2 * fwhm_l**3
        + 0.07842 * fwhm_g * fwhm_l**4
        + fwhm_l**5
    ) ** 0.2

    # Mixing parameter (0 = pure Gaussian, 1 = pure Lorentzian)
    eta = (
        1.36603 * (fwhm_l / fwhm)
        - 0.47719 * (fwhm_l / fwhm) ** 2
        + 0.11116 * (fwhm_l / fwhm) ** 3
    )

    # Gaussian and Lorentzian components
    gaussian = jnp.exp(-((x - center) ** 2) / (2 * sigma**2))
    lorentzian = gamma**2 / ((x - center) ** 2 + gamma**2)

    return amplitude * (eta * lorentzian + (1 - eta) * gaussian) + baseline


def double_gaussian(
    x: "jax.Array",
    amp1: float,
    center1: float,
    sigma1: float,
    amp2: float,
    center2: float,
    sigma2: float,
    baseline: float,
) -> "jax.Array":
    """Sum of two Gaussian peaks (overlapping peaks, doublets).

    Parameters
    ----------
    x : jax.Array
        Independent variable
    amp1, center1, sigma1 : float
        First peak parameters
    amp2, center2, sigma2 : float
        Second peak parameters
    baseline : float
        Common baseline

    Returns
    -------
    y : jax.Array
        Model values
    """
    peak1 = amp1 * jnp.exp(-((x - center1) ** 2) / (2 * sigma1**2))
    peak2 = amp2 * jnp.exp(-((x - center2) ** 2) / (2 * sigma2**2))
    return peak1 + peak2 + baseline


# -----------------------------------------------------------------------------
# KINETICS / DECAY MODELS
# -----------------------------------------------------------------------------


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
    a1, tau1 : float
        Amplitude and time constant of first exponential
    a2, tau2 : float
        Amplitude and time constant of second exponential
    offset : float
        Baseline offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    return a1 * jnp.exp(-x / tau1) + a2 * jnp.exp(-x / tau2) + offset


def stretched_exponential(
    x: "jax.Array",
    amplitude: float,
    tau: float,
    beta: float,
    offset: float,
) -> "jax.Array":
    """Stretched exponential (Kohlrausch-Williams-Watts) decay.

    Mathematical form:
        y = amplitude * exp(-(x/τ)^β) + offset

    Common applications:
    - Dielectric relaxation in glasses
    - Luminescence decay in disordered systems
    - Polymer dynamics

    Parameters
    ----------
    x : jax.Array
        Time variable
    amplitude : float
        Initial amplitude
    tau : float
        Characteristic time
    beta : float
        Stretching exponent (0 < β ≤ 1)
    offset : float
        Baseline

    Returns
    -------
    y : jax.Array
        Model values

    Notes
    -----
    β = 1: standard exponential
    β < 1: stretched (broader distribution of relaxation times)
    """
    return amplitude * jnp.exp(-jnp.power(x / tau, beta)) + offset


# -----------------------------------------------------------------------------
# BIOLOGICAL / PHARMACOLOGY MODELS
# -----------------------------------------------------------------------------


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


def hill_equation(
    x: "jax.Array",
    vmax: float,
    km: float,
    n: float,
    baseline: float,
) -> "jax.Array":
    """Hill equation for cooperative binding (enzyme kinetics, pharmacology).

    Mathematical form:
        y = vmax * x^n / (km^n + x^n) + baseline

    Parameters
    ----------
    x : jax.Array
        Ligand/substrate concentration
    vmax : float
        Maximum velocity/response
    km : float
        Half-maximal concentration (K_m or EC50)
    n : float
        Hill coefficient (cooperativity)
    baseline : float
        Baseline response

    Returns
    -------
    y : jax.Array
        Model values

    Notes
    -----
    n = 1: Michaelis-Menten kinetics (no cooperativity)
    n > 1: positive cooperativity
    n < 1: negative cooperativity
    """
    return vmax * jnp.power(x, n) / (jnp.power(km, n) + jnp.power(x, n)) + baseline


def michaelis_menten(
    x: "jax.Array",
    vmax: float,
    km: float,
) -> "jax.Array":
    """Michaelis-Menten enzyme kinetics.

    Mathematical form:
        v = Vmax * [S] / (Km + [S])

    Parameters
    ----------
    x : jax.Array
        Substrate concentration [S]
    vmax : float
        Maximum reaction velocity
    km : float
        Michaelis constant

    Returns
    -------
    v : jax.Array
        Reaction velocity
    """
    return vmax * x / (km + x)


def four_parameter_logistic(
    x: "jax.Array",
    bottom: float,
    top: float,
    ec50: float,
    hill: float,
) -> "jax.Array":
    """4-Parameter Logistic (4PL) curve for immunoassays (ELISA).

    Mathematical form:
        y = bottom + (top - bottom) / (1 + (x/EC50)^(-hill))

    Parameters
    ----------
    x : jax.Array
        Concentration
    bottom : float
        Minimum asymptote
    top : float
        Maximum asymptote
    ec50 : float
        Inflection point (concentration at 50% response)
    hill : float
        Hill slope (steepness)

    Returns
    -------
    y : jax.Array
        Response values
    """
    return bottom + (top - bottom) / (1 + jnp.power(x / ec50, -hill))


# -----------------------------------------------------------------------------
# PHYSICS MODELS
# -----------------------------------------------------------------------------


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


def planck_radiation(
    wavelength: "jax.Array",
    temperature: float,
    amplitude: float,
) -> "jax.Array":
    """Planck's blackbody radiation law.

    Mathematical form:
        B(λ,T) = amplitude * (2hc²/λ⁵) * 1/(exp(hc/λkT) - 1)

    Parameters
    ----------
    wavelength : jax.Array
        Wavelength in meters
    temperature : float
        Temperature in Kelvin
    amplitude : float
        Scaling factor (accounts for solid angle, area, etc.)

    Returns
    -------
    B : jax.Array
        Spectral radiance
    """
    # Physical constants
    h = 6.62607015e-34  # Planck constant (J·s)
    c = 299792458.0  # Speed of light (m/s)
    k = 1.380649e-23  # Boltzmann constant (J/K)

    # Avoid overflow by limiting exponent
    exponent = jnp.clip(h * c / (wavelength * k * temperature), 0, 700)
    return amplitude * (2 * h * c**2 / wavelength**5) / (jnp.exp(exponent) - 1)


# -----------------------------------------------------------------------------
# 2D SURFACE MODELS
# -----------------------------------------------------------------------------


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

    Parameters
    ----------
    xy : jax.Array, shape (2, n)
        Coordinates: xy[0] = x values, xy[1] = y values
    amplitude : float
        Peak amplitude
    x0, y0 : float
        Center coordinates
    sigma_x, sigma_y : float
        Standard deviations in x and y directions
    offset : float
        Background offset

    Returns
    -------
    z : jax.Array
        Surface values

    Notes
    -----
    For 2D fitting, configure data.columns.z in your workflow YAML.
    """
    x, y = xy[0], xy[1]
    exponent = (x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2)
    return amplitude * jnp.exp(-exponent) + offset


def gaussian_2d_rotated(
    xy: "jax.Array",
    amplitude: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> "jax.Array":
    """Rotated 2D Gaussian (elliptical beam at arbitrary angle).

    Parameters
    ----------
    xy : jax.Array, shape (2, n)
        Coordinates
    amplitude : float
        Peak amplitude
    x0, y0 : float
        Center coordinates
    sigma_x, sigma_y : float
        Widths along principal axes
    theta : float
        Rotation angle (radians)
    offset : float
        Background

    Returns
    -------
    z : jax.Array
        Surface values
    """
    x, y = xy[0], xy[1]
    cos_t, sin_t = jnp.cos(theta), jnp.sin(theta)

    # Rotate coordinates
    x_rot = (x - x0) * cos_t + (y - y0) * sin_t
    y_rot = -(x - x0) * sin_t + (y - y0) * cos_t

    exponent = x_rot**2 / (2 * sigma_x**2) + y_rot**2 / (2 * sigma_y**2)
    return amplitude * jnp.exp(-exponent) + offset


# -----------------------------------------------------------------------------
# JAX CONTROL FLOW EXAMPLES
# -----------------------------------------------------------------------------


def piecewise_linear(
    x: "jax.Array",
    slope1: float,
    slope2: float,
    breakpoint: float,
    intercept: float,
) -> "jax.Array":
    """Piecewise linear model with one breakpoint.

    Uses jnp.where() for JIT-compatible conditional logic.

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


def safe_division(
    x: "jax.Array",
    numerator: float,
    denominator_scale: float,
    offset: float,
) -> "jax.Array":
    """Model with safe division to avoid divide-by-zero.

    Uses jnp.where() to handle near-zero denominators.

    Parameters
    ----------
    x : jax.Array
        Independent variable
    numerator : float
        Numerator coefficient
    denominator_scale : float
        Denominator scale factor
    offset : float
        Baseline offset

    Returns
    -------
    y : jax.Array
        Model values
    """
    denom = denominator_scale * x
    # Use a small epsilon to avoid division by zero
    eps = 1e-10
    safe_denom = jnp.where(jnp.abs(denom) < eps, eps, denom)
    return numerator / safe_denom + offset


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
