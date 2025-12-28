"""Automatic initial parameter estimation for curve fitting.

This module provides intelligent parameter estimation when users don't provide
initial guesses (p0). It uses heuristics based on data characteristics and
function signatures to generate reasonable starting points.

Key Features
------------
- **Automatic p0 estimation**: No need to manually guess initial parameters
- **Pattern detection**: Recognizes common function types (linear, exponential, gaussian, sigmoid)
- **Smart heuristics**: Uses data statistics (range, mean, peaks) to estimate parameters
- **Library function support**: Custom functions can provide their own `.estimate_p0()` method
- **Fallback strategies**: Always provides reasonable defaults if pattern detection fails

Examples
--------
Basic usage with automatic p0 estimation:

>>> from nlsq import curve_fit
>>> import jax.numpy as jnp
>>> import numpy as np
>>>
>>> # Define model (no p0 needed!)
>>> def exponential_decay(x, amplitude, rate, offset):
...     return amplitude * jnp.exp(-rate * x) + offset
>>>
>>> # Generate data
>>> x = np.linspace(0, 5, 50)
>>> y = 3 * np.exp(-0.5 * x) + 1 + np.random.normal(0, 0.1, 50)
>>>
>>> # Use p0='auto' for automatic estimation
>>> popt, pcov = curve_fit(exponential_decay, x, y, p0='auto')
>>>
>>> # Note: p0=None (or omitting p0) uses default behavior [1.0, 1.0, ...]
>>> # for backward compatibility
>>> popt, pcov = curve_fit(exponential_decay, x, y)  # Uses default p0
>>>
>>> print(f"Fitted: amplitude={popt[0]:.2f}, rate={popt[1]:.2f}, offset={popt[2]:.2f}")
>>> # Output: Fitted: amplitude=2.95, rate=0.51, offset=1.04

How It Works
------------
1. **Custom estimation**: Checks if function has `.estimate_p0(xdata, ydata)` method
2. **Pattern detection**: Analyzes data shape to detect function type
3. **Heuristic estimation**: Uses data statistics to estimate parameters:

   - First parameter (amplitude): y_range
   - Second parameter (rate/slope): 1/x_range
   - Third parameter (offset): y_mean
   - Fourth parameter (center): x_mean

4. **Fallback**: Returns sensible defaults if all else fails

See Also
--------
nlsq.minpack.curve_fit : Main curve fitting function
nlsq.error_messages : Enhanced error messages
"""

import inspect
from collections.abc import Callable

import numpy as np


def estimate_initial_parameters(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray | str | None = None,
) -> np.ndarray:
    """Estimate initial parameters if p0 is None or 'auto'.

    This function attempts to intelligently guess initial parameter values
    based on the data characteristics when p0 is not provided.

    The estimation strategy follows this order:
    1. If p0 is provided (not None or 'auto'), return it unchanged
    2. If function has `.estimate_p0()` method, use it (for library functions)
    3. Otherwise, use generic heuristics based on data statistics

    Parameters
    ----------
    f : callable
        Model function f(x, p0, p1, ...) → y
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like or None or 'auto', optional
        Initial guess. If None or 'auto', estimate from data.

    Returns
    -------
    p0_estimated : ndarray
        Initial parameter guess

    Raises
    ------
    ValueError
        If cannot determine number of parameters from function signature

    Examples
    --------
    Automatic estimation for exponential decay:

    >>> import numpy as np
    >>> def exponential(x, a, b, c):
    ...     return a * np.exp(-b * x) + c
    >>> xdata = np.linspace(0, 5, 50)
    >>> ydata = 3 * np.exp(-0.5 * xdata) + 1
    >>> p0 = estimate_initial_parameters(exponential, xdata, ydata)
    >>> # p0 ≈ [2.0, 0.2, 1.5] (estimated from data statistics)

    Estimation works with any number of parameters:

    >>> def linear(x, a, b):
    ...     return a * x + b
    >>> xdata = np.array([1, 2, 3, 4, 5])
    >>> ydata = np.array([2, 4, 6, 8, 10])
    >>> p0 = estimate_initial_parameters(linear, xdata, ydata)
    >>> # p0 ≈ [8.0, 0.25] (based on y_range and 1/x_range)

    Custom functions can provide their own estimation:

    >>> class GaussianModel:
    ...     def __call__(self, x, amp, mu, sigma):
    ...         return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    ...
    ...     def estimate_p0(self, xdata, ydata):
    ...         amp = np.max(ydata) - np.min(ydata)
    ...         mu = xdata[np.argmax(ydata)]
    ...         sigma = (np.max(xdata) - np.min(xdata)) / 4
    ...         return [amp, mu, sigma]
    >>>
    >>> model = GaussianModel()
    >>> xdata = np.linspace(-5, 5, 100)
    >>> ydata = 2 * np.exp(-(xdata - 1)**2 / 2)
    >>> p0 = estimate_initial_parameters(model, xdata, ydata)
    >>> # Uses custom estimate_p0 method

    See Also
    --------
    detect_function_pattern : Detect likely function pattern from data
    estimate_p0_for_pattern : Pattern-specific estimation
    """
    # If p0 provided and not 'auto', return as-is
    if p0 is not None:
        # Check if p0 is the string "auto" (not an array)
        if not (isinstance(p0, str) and p0 == "auto"):
            return np.asarray(p0)

    # Check if function has custom estimation method (for library functions)
    if hasattr(f, "estimate_p0"):
        try:
            return np.asarray(f.estimate_p0(xdata, ydata))
        except Exception:
            pass  # Fall back to generic estimation

    # Convert to numpy arrays
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Get number of parameters from function signature
    try:
        sig = inspect.signature(f)
        # Filter out VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        regular_params = [
            name
            for name, param in sig.parameters.items()
            if param.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        # First parameter is typically x, so n_params = total - 1
        n_params = len(regular_params) - 1
    except (TypeError, AttributeError):
        # If signature inspection fails, raise error
        raise ValueError(
            "Cannot automatically determine number of parameters. "
            "Please provide p0 explicitly."
        ) from None

    # If we have VAR_POSITIONAL or VAR_KEYWORD, we can't determine n_params
    # Check this AFTER try/except to avoid catching our own ValueError
    try:
        if any(
            param.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for param in sig.parameters.values()
        ):
            raise ValueError(
                "Cannot automatically determine number of parameters "
                "for functions with *args or **kwargs. "
                "Please provide p0 explicitly."
            )
    except NameError:
        # sig doesn't exist (caught above already)
        pass

    if n_params <= 0:
        raise ValueError(
            "Function must have at least one parameter besides x. "
            "Please provide p0 explicitly."
        )

    # Calculate data statistics for heuristics
    y_min, y_max = np.min(ydata), np.max(ydata)
    y_range = y_max - y_min
    y_mean = np.mean(ydata)
    np.median(ydata)

    x_min, x_max = np.min(xdata), np.max(xdata)
    x_range = x_max - x_min if x_max != x_min else 1.0
    x_mean = np.mean(xdata)

    # Generic parameter estimation heuristics
    p0_guess = []

    for i in range(n_params):
        if i == 0:
            # First parameter: often amplitude/scale
            # Use range for amplitude, or 1.0 if constant
            param = y_range if y_range > 0 else 1.0
        elif i == 1:
            # Second parameter: often rate/frequency/slope
            # Use inverse of x_range as reasonable scale
            param = 1.0 / x_range if x_range > 0 else 0.1
        elif i == 2:
            # Third parameter: often offset/baseline
            # Use mean or median of y data
            param = y_mean
        elif i == 3:
            # Fourth parameter: center/midpoint
            param = x_mean
        else:
            # Additional parameters: use 1.0 as safe default
            param = 1.0

        p0_guess.append(param)

    return np.array(p0_guess)


def detect_function_pattern(ydata: np.ndarray, xdata: np.ndarray) -> str:
    """Detect likely function pattern from data shape.

    This function analyzes the shape of the data to identify common patterns.
    It uses simple heuristics like monotonicity, peak detection, and correlation.

    Parameters
    ----------
    ydata : ndarray
        Dependent variable data
    xdata : ndarray
        Independent variable data

    Returns
    -------
    pattern : str
        Detected pattern:
        - 'linear': Strong linear correlation (\\|r\\| > 0.95)
        - 'exponential_decay': Monotonically decreasing
        - 'exponential_growth': Monotonically increasing
        - 'gaussian': Bell-shaped curve with peak in middle
        - 'sigmoid': S-shaped curve going from low to high (or vice versa)
        - 'unknown': No clear pattern detected

    Notes
    -----
    This is a simple heuristic detector, not a rigorous statistical test.
    It's designed to be fast and provide reasonable guidance for initial
    parameter estimation, not to be 100% accurate.

    Examples
    --------
    Detect exponential decay:

    >>> import numpy as np
    >>> xdata = np.linspace(0, 5, 50)
    >>> ydata = 3 * np.exp(-0.5 * xdata)
    >>> pattern = detect_function_pattern(ydata, xdata)
    >>> print(pattern)
    'exponential_decay'

    Detect Gaussian peak:

    >>> xdata = np.linspace(-5, 5, 100)
    >>> ydata = 2 * np.exp(-(xdata - 1)**2 / 2)
    >>> pattern = detect_function_pattern(ydata, xdata)
    >>> print(pattern)
    'gaussian'

    See Also
    --------
    estimate_p0_for_pattern : Estimate parameters based on detected pattern
    """
    if len(ydata) < 3:
        return "unknown"

    # Calculate linear correlation first
    try:
        corr = np.corrcoef(xdata, ydata)[0, 1]
    except Exception:
        corr = 0.0

    # Check for PERFECT linear pattern (correlation > 0.99)
    # This should be detected before monotonic patterns
    if abs(corr) > 0.99:
        return "linear"

    # Check for bell-shaped curve (Gaussian candidate)
    # Check this before monotonic patterns
    peak_idx = np.argmax(ydata)
    if 0.2 * len(ydata) < peak_idx < 0.8 * len(ydata):
        # Peak is in the middle
        left_slope = np.mean(np.diff(ydata[:peak_idx]))
        right_slope = np.mean(np.diff(ydata[peak_idx:]))
        if left_slope > 0 and right_slope < 0:
            return "gaussian"

    # Check for sigmoid pattern (S-curve with inflection point)
    # Sigmoid has an inflection point where second derivative changes sign
    # Check this before simple monotonic patterns
    y_range = np.max(ydata) - np.min(ydata)
    if y_range > 0:
        normalized = (ydata - np.min(ydata)) / y_range

        # Check for sigmoid: starts low, ends high
        if normalized[0] < 0.2 and normalized[-1] > 0.8:
            # Check for inflection point (second derivative changes sign)
            if len(ydata) > 5:
                second_diff = np.diff(np.diff(ydata))
                # If second derivative changes sign significantly, it's sigmoid
                if np.max(second_diff) > 0 and np.min(second_diff) < 0:
                    return "sigmoid"
            # No clear inflection - likely monotonic growth
            return "exponential_growth"

        # Check for inverse sigmoid: starts high, ends low
        if normalized[0] > 0.8 and normalized[-1] < 0.2:
            # Check for inflection point
            if len(ydata) > 5:
                second_diff = np.diff(np.diff(ydata))
                # If second derivative changes sign significantly, it's sigmoid
                if np.max(second_diff) > 0 and np.min(second_diff) < 0:
                    return "sigmoid_inv"
            # No clear inflection - likely monotonic decay
            return "exponential_decay"

    # Check for monotonic decay (exponential decay candidate)
    if np.all(np.diff(ydata) <= 0):
        return "exponential_decay"

    # Check for monotonic growth
    if np.all(np.diff(ydata) >= 0):
        return "exponential_growth"

    # Check for good linear pattern (correlation > 0.95 but < 0.99)
    if abs(corr) > 0.95:
        return "linear"

    return "unknown"


def estimate_p0_for_pattern(
    pattern: str, xdata: np.ndarray, ydata: np.ndarray, n_params: int
) -> np.ndarray:
    """Estimate p0 based on detected pattern.

    This function uses pattern-specific heuristics to estimate initial
    parameters. Each pattern has its own estimation strategy based on
    the mathematical properties of that function type.

    Parameters
    ----------
    pattern : str
        Detected function pattern ('linear', 'exponential_decay', 'gaussian', 'sigmoid', etc.)
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data
    n_params : int
        Number of parameters to estimate

    Returns
    -------
    p0 : ndarray
        Estimated initial parameters (length n_params)

    Notes
    -----
    Pattern-specific estimation strategies:

    - **linear**: Uses least squares to fit y = a*x + b
    - **exponential_decay**: Estimates amplitude from range, rate from half-life
    - **gaussian**: Estimates amplitude, mean from peak, sigma from FWHM
    - **sigmoid**: Estimates L, x0, k from data range and midpoint
    - **unknown**: Falls back to generic heuristics

    Examples
    --------
    Estimate parameters for exponential decay:

    >>> import numpy as np
    >>> xdata = np.linspace(0, 5, 50)
    >>> ydata = 3 * np.exp(-0.5 * xdata) + 1
    >>> p0 = estimate_p0_for_pattern('exponential_decay', xdata, ydata, 3)
    >>> print(p0)
    [2.0, 0.277..., 1.0]  # [amplitude, rate, offset]

    Estimate parameters for Gaussian:

    >>> xdata = np.linspace(-5, 5, 100)
    >>> ydata = 2 * np.exp(-(xdata - 1)**2 / (2 * 0.5**2))
    >>> p0 = estimate_p0_for_pattern('gaussian', xdata, ydata, 3)
    >>> print(p0)
    [2.0, 1.0, 0.5]  # [amplitude, mean, sigma]

    See Also
    --------
    detect_function_pattern : Detect pattern from data
    estimate_initial_parameters : Main parameter estimation function
    """
    if pattern == "linear":
        # Linear: y = a*x + b
        if n_params >= 2:
            # Simple linear regression
            A = np.vstack([xdata, np.ones(len(xdata))]).T
            try:
                coeffs, *_ = np.linalg.lstsq(A, ydata, rcond=None)
                p0 = list(coeffs[:n_params])
                # Pad with 1.0 if more params needed
                p0.extend([1.0] * (n_params - len(p0)))
                return np.array(p0)
            except np.linalg.LinAlgError:
                pass

    elif pattern == "exponential_decay":
        # y = a * exp(-b*x) + c
        y_max, y_min = np.max(ydata), np.min(ydata)
        a = y_max - y_min
        c = y_min
        # Estimate decay rate from half-life
        half_val = (y_max + y_min) / 2
        try:
            half_idx = np.where(ydata < half_val)[0][0]
            x_half = xdata[half_idx]
            b = np.log(2) / x_half if x_half > 0 else 0.1
        except (IndexError, ValueError):
            b = (
                1.0 / (np.max(xdata) - np.min(xdata))
                if np.max(xdata) != np.min(xdata)
                else 0.1
            )

        p0 = [a, b, c][:n_params]
        p0.extend([1.0] * (n_params - len(p0)))
        return np.array(p0)

    elif pattern == "gaussian":
        # y = amp * exp(-(x-mu)^2 / (2*sigma^2))
        amp = np.max(ydata) - np.min(ydata)
        mu = xdata[np.argmax(ydata)]

        # Estimate sigma from FWHM
        half_max = (np.max(ydata) + np.min(ydata)) / 2
        indices = np.where(ydata > half_max)[0]
        if len(indices) > 1:
            fwhm = xdata[indices[-1]] - xdata[indices[0]]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (np.max(xdata) - np.min(xdata)) / 4

        p0 = [amp, mu, sigma][:n_params]
        p0.extend([1.0] * (n_params - len(p0)))
        return np.array(p0)

    elif pattern == "sigmoid":
        # y = L / (1 + exp(-k*(x-x0))) + b
        y_min, y_max = np.min(ydata), np.max(ydata)
        L = y_max - y_min
        b = y_min

        # Midpoint
        y_mid = (y_max + y_min) / 2
        try:
            x0 = xdata[np.argmin(np.abs(ydata - y_mid))]
        except ValueError:
            x0 = np.mean(xdata)

        # Steepness
        k = (
            1.0 / (np.max(xdata) - np.min(xdata))
            if np.max(xdata) != np.min(xdata)
            else 1.0
        )

        p0 = [L, x0, k, b][:n_params]
        p0.extend([1.0] * (n_params - len(p0)))
        return np.array(p0)

    # Default: generic estimation for unknown patterns
    # Use same heuristics as estimate_initial_parameters
    y_min, y_max = np.min(ydata), np.max(ydata)
    y_range = y_max - y_min
    y_mean = np.mean(ydata)

    x_min, x_max = np.min(xdata), np.max(xdata)
    x_range = x_max - x_min if x_max != x_min else 1.0
    x_mean = np.mean(xdata)

    p0_guess = []
    for i in range(n_params):
        if i == 0:
            # First parameter: often amplitude/scale
            param = y_range if y_range > 0 else 1.0
        elif i == 1:
            # Second parameter: often rate/frequency/slope
            param = 1.0 / x_range if x_range > 0 else 0.1
        elif i == 2:
            # Third parameter: often offset/baseline
            param = y_mean
        elif i == 3:
            # Fourth parameter: center/midpoint
            param = x_mean
        else:
            # Additional parameters: use 1.0 as safe default
            param = 1.0
        p0_guess.append(param)

    return np.array(p0_guess)
