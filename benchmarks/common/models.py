"""Shared benchmark model functions.

These models are used across multiple benchmark scripts to ensure
consistency and reduce code duplication.
"""

from __future__ import annotations

import jax.numpy as jnp


def exponential_model(x, a, b, c):
    """Exponential decay model: y = a * exp(-b * x) + c

    Parameters
    ----------
    x : array_like
        Independent variable
    a : float
        Amplitude
    b : float
        Decay rate
    c : float
        Offset

    Returns
    -------
    array_like
        Model values
    """
    return a * jnp.exp(-b * x) + c


def gaussian_model(x, amp, mu, sigma):
    """Gaussian function: y = amp * exp(-((x - mu)^2) / (2 * sigma^2))

    Parameters
    ----------
    x : array_like
        Independent variable
    amp : float
        Amplitude
    mu : float
        Mean (center)
    sigma : float
        Standard deviation

    Returns
    -------
    array_like
        Model values
    """
    return amp * jnp.exp(-((x - mu) ** 2) / (2 * sigma**2))


def polynomial_model(x, a, b, c, d):
    """Cubic polynomial: y = a*x^3 + b*x^2 + c*x + d

    Parameters
    ----------
    x : array_like
        Independent variable
    a, b, c, d : float
        Polynomial coefficients

    Returns
    -------
    array_like
        Model values
    """
    return a * x**3 + b * x**2 + c * x + d


def sinusoidal_model(x, amp, freq, phase, offset):
    """Sinusoidal function: y = amp * sin(freq * x + phase) + offset

    Parameters
    ----------
    x : array_like
        Independent variable
    amp : float
        Amplitude
    freq : float
        Frequency
    phase : float
        Phase offset
    offset : float
        Vertical offset

    Returns
    -------
    array_like
        Model values
    """
    return amp * jnp.sin(freq * x + phase) + offset
