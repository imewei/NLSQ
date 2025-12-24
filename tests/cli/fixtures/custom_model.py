"""Custom model file for CLI testing.

This module provides a custom model function for testing the CLI's
ability to load models from external Python files.
"""

import jax.numpy as jnp
import numpy as np


def custom_linear(x, a, b):
    """Custom linear model: y = a * x + b

    Parameters
    ----------
    x : array-like
        Independent variable
    a : float
        Slope
    b : float
        Intercept

    Returns
    -------
    array
        Model values
    """
    return a * x + b


def estimate_p0(xdata, ydata):
    """Estimate initial parameters for the custom linear model.

    Uses simple linear regression estimates.

    Parameters
    ----------
    xdata : array-like
        Independent variable data
    ydata : array-like
        Dependent variable data

    Returns
    -------
    list
        Initial parameter estimates [a, b]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Simple linear regression
    n = len(xdata)
    sum_x = np.sum(xdata)
    sum_y = np.sum(ydata)
    sum_xx = np.sum(xdata * xdata)
    sum_xy = np.sum(xdata * ydata)

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-10:
        return [1.0, 0.0]

    a = (n * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - a * sum_x) / n

    return [float(a), float(b)]


def bounds():
    """Return parameter bounds for the custom linear model.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) as lists
    """
    return ([-np.inf, -np.inf], [np.inf, np.inf])
