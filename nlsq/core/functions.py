"""Common curve fitting functions with automatic parameter estimation.

This module provides pre-built functions for common curve fitting tasks.
Each function includes:

- JAX-compatible implementation for GPU/TPU acceleration
- Automatic p0 estimation via `.estimate_p0(xdata, ydata)` method
- Reasonable default bounds via `.bounds()` method
- Comprehensive docstrings with mathematical equations

Examples
--------
Basic usage with automatic parameter estimation:

>>> from nlsq import curve_fit
>>> from nlsq.core.functions import exponential_decay
>>> import numpy as np
>>>
>>> # Generate data
>>> x = np.linspace(0, 5, 50)
>>> y = 3 * np.exp(-0.5 * x) + 1 + np.random.normal(0, 0.1, 50)
>>>
>>> # Fit with automatic p0 estimation
>>> popt, pcov = curve_fit(exponential_decay, x, y, p0='auto')
>>> print(f"Fitted: amplitude={popt[0]:.2f}, rate={popt[1]:.2f}, offset={popt[2]:.2f}")

All functions work seamlessly with NLSQ's auto p0 estimation.

See Also
--------
nlsq.parameter_estimation : Automatic parameter estimation
nlsq.minpack.curve_fit : Main curve fitting function
"""

import warnings
from collections.abc import Callable
from typing import Union

import jax.numpy as jnp
import numpy as np

# Type aliases for clarity
ArrayLike = Union[np.ndarray, jnp.ndarray, list[float], float]
ParameterList = list[float]
BoundsTuple = tuple[list[float], list[float]]


# ============================================================================
# Linear Functions
# ============================================================================


def linear(x: ArrayLike, a: float, b: float) -> ArrayLike:
    """Linear function: y = a*x + b

    Parameters
    ----------
    x : array_like
        Independent variable
    a : float
        Slope
    b : float
        Intercept

    Returns
    -------
    y : array_like
        Dependent variable

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import linear
    >>> import numpy as np
    >>>
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = 2 * x + 3 + np.random.normal(0, 0.1, 5)
    >>> popt, pcov = curve_fit(linear, x, y, p0='auto')
    >>> print(f"Slope: {popt[0]:.2f}, Intercept: {popt[1]:.2f}")
    """
    return a * x + b  # type: ignore[operator]


def estimate_p0_linear(xdata: np.ndarray, ydata: np.ndarray) -> ParameterList:
    """Estimate initial parameters for linear function.

    Uses ordinary least squares to estimate slope and intercept.

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list
        [slope, intercept]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Ordinary least squares: solve Ax = b where A = [x, 1]
    A = np.vstack([xdata, np.ones(len(xdata))]).T
    try:
        a, b = np.linalg.lstsq(A, ydata, rcond=None)[0]
        return [float(a), float(b)]
    except np.linalg.LinAlgError:
        # Fallback if singular
        return [1.0, float(np.mean(ydata))]


def bounds_linear() -> BoundsTuple:
    """Return default bounds for linear function.

    Returns
    -------
    bounds : tuple
        ([lower_a, lower_b], [upper_a, upper_b])
    """
    return ([-np.inf, -np.inf], [np.inf, np.inf])


# Attach methods to function
linear.estimate_p0 = estimate_p0_linear  # type: ignore[attr-defined]
linear.bounds = bounds_linear  # type: ignore[attr-defined]


# ============================================================================
# Exponential Functions
# ============================================================================


def exponential_decay(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    """Exponential decay: y = a * exp(-b*x) + c

    Common for radioactive decay, cooling curves, discharge curves.

    Parameters
    ----------
    x : array_like
        Independent variable (time, distance, etc.)
    a : float
        Amplitude (initial value minus asymptote)
    b : float
        Decay rate (positive, units: 1/x)
    c : float
        Offset (asymptotic value as x → ∞)

    Returns
    -------
    y : array_like
        Dependent variable

    Notes
    -----
    - Half-life: t_half = ln(2) / b
    - Time constant: τ = 1 / b
    - At x=0: y = a + c
    - As x→∞: y → c

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import exponential_decay
    >>> import numpy as np
    >>>
    >>> # Radioactive decay with half-life = ln(2)/0.5 ≈ 1.4
    >>> x = np.linspace(0, 10, 100)
    >>> y = 100 * np.exp(-0.5 * x) + 10 + np.random.normal(0, 2, 100)
    >>> popt, pcov = curve_fit(exponential_decay, x, y, p0='auto')
    >>> print(f"Half-life: {np.log(2)/popt[1]:.2f}")
    """
    return a * jnp.exp(-b * x) + c  # type: ignore[operator]


def estimate_p0_exponential_decay(
    xdata: np.ndarray, ydata: np.ndarray
) -> ParameterList:
    """Estimate initial parameters for exponential decay.

    Strategy:
    - Amplitude (a): range of y values
    - Offset (c): minimum y value (asymptote)
    - Rate (b): estimated from half-life

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list
        [amplitude, rate, offset]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    y_max = np.max(ydata)
    y_min = np.min(ydata)

    a = y_max - y_min
    c = y_min

    # Estimate decay rate from half-life
    # Find where y ≈ (y_max + y_min) / 2
    half_val = (y_max + y_min) / 2

    try:
        # Find closest point to half value
        half_idx = np.argmin(np.abs(ydata - half_val))
        x_half = xdata[half_idx]

        if x_half > 0:
            b = np.log(2) / x_half
        else:
            # Fallback: estimate from x range
            x_range = np.max(xdata) - np.min(xdata)
            b = 1.0 / x_range if x_range > 0 else 0.1
    except (ValueError, IndexError):
        b = 0.1  # Safe default

    # Ensure b is positive
    b = max(b, 0.01)

    return [float(a), float(b), float(c)]


def bounds_exponential_decay() -> BoundsTuple:
    """Return default bounds for exponential decay.

    Returns
    -------
    bounds : tuple
        ([lower_a, lower_b, lower_c], [upper_a, upper_b, upper_c])
    """
    return ([0, 0, -np.inf], [np.inf, np.inf, np.inf])


exponential_decay.estimate_p0 = estimate_p0_exponential_decay  # type: ignore[attr-defined]
exponential_decay.bounds = bounds_exponential_decay  # type: ignore[attr-defined]


def exponential_growth(x: ArrayLike, a: float, b: float, c: float) -> ArrayLike:
    """Exponential growth: y = a * exp(b*x) + c

    Common for population growth, compound interest, bacterial growth.

    Parameters
    ----------
    x : array_like
        Independent variable (time, distance, etc.)
    a : float
        Initial amplitude
    b : float
        Growth rate (positive, units: 1/x)
    c : float
        Offset (baseline)

    Returns
    -------
    y : array_like
        Dependent variable

    Notes
    -----
    - Doubling time: t_double = ln(2) / b
    - At x=0: y = a + c
    - As x→∞: y → ∞ (unbounded growth)

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import exponential_growth
    >>> import numpy as np
    >>>
    >>> # Bacterial growth with doubling time = ln(2)/0.3 ≈ 2.3
    >>> x = np.linspace(0, 5, 50)
    >>> y = 10 * np.exp(0.3 * x) + np.random.normal(0, 1, 50)
    >>> popt, pcov = curve_fit(exponential_growth, x, y, p0='auto')
    >>> print(f"Doubling time: {np.log(2)/popt[1]:.2f}")
    """
    return a * jnp.exp(b * x) + c  # type: ignore[operator]


def estimate_p0_exponential_growth(
    xdata: np.ndarray, ydata: np.ndarray
) -> ParameterList:
    """Estimate initial parameters for exponential growth.

    Similar to decay but inverted.

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list
        [amplitude, rate, offset]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    y_min = np.min(ydata)
    y_max = np.max(ydata)

    a = y_min  # Initial value
    c = 0.0  # Offset

    # Estimate growth rate
    y_range = y_max - y_min
    x_range = np.max(xdata) - np.min(xdata)

    if x_range > 0 and y_range > 0:
        # Rough estimate: b ≈ ln(y_max/y_min) / x_range
        if y_min > 0:
            b = np.log(y_max / y_min) / x_range
        else:
            b = 0.1  # Default
    else:
        b = 0.1

    return [float(a), float(b), float(c)]


exponential_growth.estimate_p0 = estimate_p0_exponential_growth  # type: ignore[attr-defined]
exponential_growth.bounds = lambda: ([0, 0, -np.inf], [np.inf, np.inf, np.inf])  # type: ignore[attr-defined]


# ============================================================================
# Gaussian Functions
# ============================================================================


def gaussian(x: ArrayLike, amp: float, mu: float, sigma: float) -> ArrayLike:
    """Gaussian (normal distribution) function: y = amp * exp(-(x-mu)² / (2*sigma²))

    Common for spectral peaks, chromatography peaks, probability distributions.

    Parameters
    ----------
    x : array_like
        Independent variable
    amp : float
        Amplitude (peak height)
    mu : float
        Mean (center position, peak location)
    sigma : float
        Standard deviation (width parameter, positive)

    Returns
    -------
    y : array_like
        Dependent variable

    Notes
    -----
    - Peak position: x = mu
    - Peak height: y = amp
    - FWHM (Full Width at Half Maximum): FWHM = 2.355 * sigma
    - Integral: ∫ gaussian dx = amp * sigma * sqrt(2π)

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import gaussian
    >>> import numpy as np
    >>>
    >>> # Spectral peak at x=5 with FWHM ≈ 2.355
    >>> x = np.linspace(0, 10, 200)
    >>> y = 10 * np.exp(-(x-5)**2 / (2*1**2)) + np.random.normal(0, 0.2, 200)
    >>> popt, pcov = curve_fit(gaussian, x, y, p0='auto')
    >>> print(f"Peak at {popt[1]:.2f}, FWHM = {2.355*popt[2]:.2f}")
    """
    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))  # type: ignore[operator]


def estimate_p0_gaussian(xdata: np.ndarray, ydata: np.ndarray) -> ParameterList:
    """Estimate initial parameters for Gaussian function.

    Strategy:
    - Amplitude: peak height (max - min)
    - Mean: location of maximum
    - Sigma: estimated from FWHM

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list
        [amplitude, mean, sigma]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    y_min = np.min(ydata)
    y_max = np.max(ydata)

    amp = y_max - y_min
    mu = xdata[np.argmax(ydata)]

    # Estimate sigma from FWHM (Full Width at Half Maximum)
    half_max = (y_max + y_min) / 2
    indices = np.where(ydata > half_max)[0]

    if len(indices) > 1:
        fwhm = xdata[indices[-1]] - xdata[indices[0]]
        # FWHM = 2.355 * sigma → sigma = FWHM / 2.355
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    else:
        # Fallback: use data range / 4 as rough estimate
        sigma = (np.max(xdata) - np.min(xdata)) / 4

    # Ensure sigma is positive
    sigma = max(sigma, 0.01)

    return [float(amp), float(mu), float(sigma)]


gaussian.estimate_p0 = estimate_p0_gaussian  # type: ignore[attr-defined]
gaussian.bounds = lambda: ([0, -np.inf, 0], [np.inf, np.inf, np.inf])  # type: ignore[attr-defined]


# ============================================================================
# Sigmoid Functions
# ============================================================================


def sigmoid(x: ArrayLike, L: float, x0: float, k: float, b: float) -> ArrayLike:
    """Sigmoid (logistic) function: y = L / (1 + exp(-k*(x-x0))) + b

    Common for dose-response curves, growth saturation, S-curves.

    Parameters
    ----------
    x : array_like
        Independent variable
    L : float
        Maximum value (saturation level)
    x0 : float
        Midpoint (inflection point, x value at half-maximum)
    k : float
        Steepness (growth rate, positive)
    b : float
        Baseline offset (minimum asymptote)

    Returns
    -------
    y : array_like
        Dependent variable

    Notes
    -----
    - At x=x0: y = L/2 + b (midpoint)
    - As x→-∞: y → b (lower asymptote)
    - As x→+∞: y → L + b (upper asymptote)
    - Steeper curve: larger k

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import sigmoid
    >>> import numpy as np
    >>>
    >>> # Dose-response curve
    >>> x = np.linspace(0, 10, 100)
    >>> y = 5 / (1 + np.exp(-2*(x-5))) + 1 + np.random.normal(0, 0.1, 100)
    >>> popt, pcov = curve_fit(sigmoid, x, y, p0='auto')
    >>> print(f"EC50 (midpoint): {popt[1]:.2f}")
    """
    return L / (1 + jnp.exp(-k * (x - x0))) + b  # type: ignore[operator]


def estimate_p0_sigmoid(xdata: np.ndarray, ydata: np.ndarray) -> ParameterList:
    """Estimate initial parameters for sigmoid function.

    Strategy:
    - L: range of y values
    - b: minimum y value (baseline)
    - x0: x value at midpoint
    - k: estimated from steepness

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list
        [L, x0, k, b]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    y_min = np.min(ydata)
    y_max = np.max(ydata)

    L = y_max - y_min
    b = y_min

    # Midpoint: find x where y ≈ (y_max + y_min) / 2
    y_mid = (y_max + y_min) / 2
    try:
        x0_idx = np.argmin(np.abs(ydata - y_mid))
        x0 = xdata[x0_idx]
    except (ValueError, IndexError):
        x0 = np.mean(xdata)

    # Steepness: rough estimate from x range
    x_range = np.max(xdata) - np.min(xdata)
    k = 1.0 / x_range if x_range > 0 else 1.0

    return [float(L), float(x0), float(k), float(b)]


sigmoid.estimate_p0 = estimate_p0_sigmoid  # type: ignore[attr-defined]
sigmoid.bounds = lambda: ([0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])  # type: ignore[attr-defined]


# ============================================================================
# Power Law
# ============================================================================


def power_law(x: ArrayLike, a: float, b: float) -> ArrayLike:
    """Power law function: y = a * x^b

    Common for scaling relationships, fractals, allometry.

    Parameters
    ----------
    x : array_like
        Independent variable (must be positive)
    a : float
        Prefactor (amplitude)
    b : float
        Exponent (power)

    Returns
    -------
    y : array_like
        Dependent variable

    Notes
    -----
    - b > 0: increasing function (growth)
    - b < 0: decreasing function (decay)
    - b = 1: linear relationship
    - For x=1: y = a

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import power_law
    >>> import numpy as np
    >>>
    >>> # Allometric scaling: metabolic rate ∝ mass^0.75
    >>> x = np.linspace(1, 100, 50)
    >>> y = 3 * x**0.75 + np.random.normal(0, 0.5, 50)
    >>> popt, pcov = curve_fit(power_law, x, y, p0='auto')
    >>> print(f"Scaling exponent: {popt[1]:.2f}")
    """
    return a * jnp.power(x, b)  # type: ignore[arg-type]


def estimate_p0_power_law(xdata: np.ndarray, ydata: np.ndarray) -> ParameterList:
    """Estimate initial parameters for power law.

    Uses log-log linear regression: log(y) = log(a) + b*log(x)

    Parameters
    ----------
    xdata : ndarray
        Independent variable data (must be positive)
    ydata : ndarray
        Dependent variable data (must be positive)

    Returns
    -------
    p0 : list
        [prefactor, exponent]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Filter out non-positive values
    mask = (xdata > 0) & (ydata > 0)
    if np.sum(mask) < 2:
        return [1.0, 1.0]

    x_pos = xdata[mask]
    y_pos = ydata[mask]

    try:
        # Log-log linear regression
        log_x = np.log(x_pos)
        log_y = np.log(y_pos)

        A = np.vstack([log_x, np.ones(len(log_x))]).T
        b, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
        a = np.exp(log_a)

        return [float(a), float(b)]
    except (np.linalg.LinAlgError, ValueError):
        return [1.0, 1.0]


power_law.estimate_p0 = estimate_p0_power_law  # type: ignore[attr-defined]
power_law.bounds = lambda: ([0, -np.inf], [np.inf, np.inf])  # type: ignore[attr-defined]


# ============================================================================
# Polynomial Factory
# ============================================================================


def polynomial(degree: int) -> Callable:
    """Create polynomial function of given degree.

    Returns a function that computes: y = c0\\*x^n + c1\\*x^(n-1) + ... + cn
    where n is the degree.

    Parameters
    ----------
    degree : int
        Polynomial degree (0, 1, 2, 3, ...)

    Returns
    -------
    poly_func : callable
        Polynomial function with signature poly(x, \\*coeffs)

    Examples
    --------
    >>> from nlsq import curve_fit
    >>> from nlsq.core.functions import polynomial
    >>> import numpy as np
    >>>
    >>> # Fit quadratic: y = ax² + bx + c
    >>> quadratic = polynomial(2)
    >>> x = np.linspace(-5, 5, 50)
    >>> y = 2*x**2 + 3*x + 1 + np.random.normal(0, 0.5, 50)
    >>> popt, pcov = curve_fit(quadratic, x, y, p0='auto')
    >>> print(f"Coefficients: {popt}")
    """

    def poly(x, *coeffs):
        """Polynomial function with coefficients from highest to lowest degree."""
        if len(coeffs) != degree + 1:
            raise ValueError(f"Expected {degree + 1} coefficients, got {len(coeffs)}")
        return jnp.polyval(jnp.array(coeffs), x)

    def estimate_p0_poly(xdata: np.ndarray, ydata: np.ndarray) -> list[float]:
        """Estimate polynomial coefficients using least squares."""
        xdata = np.asarray(xdata)
        ydata = np.asarray(ydata)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", np.exceptions.RankWarning)
                coeffs = np.polyfit(xdata, ydata, degree)
            return [float(c) for c in coeffs]
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to zeros except constant term
            p0 = [0.0] * degree + [float(np.mean(ydata))]
            return p0

    def bounds_poly() -> tuple[list[float], list[float]]:
        """Return unbounded limits for polynomial coefficients."""
        return ([-np.inf] * (degree + 1), [np.inf] * (degree + 1))

    # Attach methods and metadata
    poly.estimate_p0 = estimate_p0_poly  # type: ignore[attr-defined]
    poly.bounds = bounds_poly  # type: ignore[attr-defined]
    poly.__name__ = f"polynomial_degree_{degree}"
    poly.__doc__ = f"""Polynomial of degree {degree}: y = c0\\*x^{degree} + c1\\*x^{degree - 1} + ... + c{degree}

    Parameters
    ----------
    x : array_like
        Independent variable
    \\*coeffs : float
        Coefficients from highest to lowest degree ({degree + 1} coefficients)

    Returns
    -------
    y : array_like
        Dependent variable
    """

    return poly


# ============================================================================
# Module exports
# ============================================================================


__all__ = [
    "exponential_decay",
    "exponential_growth",
    "gaussian",
    "linear",
    "polynomial",
    "power_law",
    "sigmoid",
]
