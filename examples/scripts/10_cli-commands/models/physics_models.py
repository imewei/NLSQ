"""Physics Models for NLSQ CLI Demonstrations.

This module contains custom model functions for physics curve fitting,
extracted from the advanced gallery examples.

Models:
    - damped_oscillator: Damped harmonic oscillation
    - gaussian_2d: 2D Gaussian surface (beam profile)
    - gaussian_2d_rotated: Rotated 2D Gaussian surface

Usage in workflow YAML:
    model:
      type: custom
      path: models/physics_models.py
      function: damped_oscillator
"""

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Damped Oscillator Model
# =============================================================================


def damped_oscillator(t, A0, gamma, omega, phi):
    """Damped harmonic oscillator model.

    Mathematical form:
        x(t) = A0 * exp(-gamma*t) * cos(omega*t + phi)

    Parameters
    ----------
    t : array_like
        Time (independent variable)
    A0 : float
        Initial amplitude
    gamma : float
        Damping coefficient (1/s)
    omega : float
        Angular frequency (rad/s)
    phi : float
        Phase offset (radians)

    Returns
    -------
    x : array_like
        Displacement at time t
    """
    return A0 * jnp.exp(-gamma * t) * jnp.cos(omega * t + phi)


def estimate_p0(xdata, ydata):
    """Estimate initial parameters for damped oscillator.

    Parameters
    ----------
    xdata : ndarray
        Time data
    ydata : ndarray
        Displacement data

    Returns
    -------
    p0 : list[float]
        Initial estimates [A0, gamma, omega, phi]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # A0: maximum absolute value
    A0 = float(np.max(np.abs(ydata)))
    if A0 == 0:
        A0 = 1.0

    # Gamma: estimate from envelope decay
    abs_y = np.abs(ydata)
    peak_indices = [
        i
        for i in range(1, len(abs_y) - 1)
        if abs_y[i] > abs_y[i - 1] and abs_y[i] > abs_y[i + 1]
    ]

    if len(peak_indices) >= 2:
        x_peaks = xdata[peak_indices]
        y_peaks = abs_y[peak_indices]
        valid_mask = y_peaks > 0
        if np.sum(valid_mask) >= 2:
            log_y = np.log(y_peaks[valid_mask])
            x_valid = x_peaks[valid_mask]
            A = np.vstack([x_valid, np.ones(len(x_valid))]).T
            try:
                slope, _ = np.linalg.lstsq(A, log_y, rcond=None)[0]
                gamma = max(-slope, 0.01)
            except np.linalg.LinAlgError:
                gamma = 0.1
        else:
            gamma = 0.1
    else:
        x_range = np.max(xdata) - np.min(xdata)
        gamma = 1.0 / x_range if x_range > 0 else 0.1

    # Omega: estimate from zero crossings
    zero_crossings = []
    for i in range(len(ydata) - 1):
        if ydata[i] * ydata[i + 1] < 0:
            # Linear interpolation to find crossing
            x_cross = xdata[i] - ydata[i] * (xdata[i + 1] - xdata[i]) / (
                ydata[i + 1] - ydata[i]
            )
            zero_crossings.append(x_cross)

    if len(zero_crossings) >= 2:
        crossings = np.array(zero_crossings)
        periods = np.diff(crossings) * 2
        avg_period = np.mean(periods)
        omega = 2 * np.pi / avg_period if avg_period > 0 else 1.0
    else:
        x_range = np.max(xdata) - np.min(xdata)
        omega = 2 * np.pi / x_range if x_range > 0 else 1.0

    # Phi: estimate from initial value
    y0 = ydata[0]
    if A0 > 0 and abs(y0 / A0) <= 1:
        phi = float(np.arccos(y0 / A0))
        if len(ydata) > 1 and ydata[1] < ydata[0]:
            phi = -phi
    else:
        phi = 0.0

    return [A0, gamma, omega, phi]


def bounds():
    """Return default parameter bounds for damped oscillator.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [A0, gamma, omega, phi]
    """
    lower = [0.0, 0.0, 0.0, -np.pi]
    upper = [np.inf, np.inf, np.inf, np.pi]
    return (lower, upper)


# =============================================================================
# 2D Gaussian Surface Model
# =============================================================================


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian surface model (elliptical, axis-aligned).

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) where xy[0] = x coordinates, xy[1] = y coordinates.
    amplitude : float
        Peak amplitude (max height above offset).
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations in x and y directions.
    offset : float
        Background offset (baseline).

    Returns
    -------
    z : ndarray
        Shape (n_points,) with z values at each (x, y) point.
    """
    x = xy[0]
    y = xy[1]
    return (
        amplitude
        * jnp.exp(
            -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
        )
        + offset
    )


def gaussian_2d_rotated(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """2D Gaussian surface with rotation (tilted ellipse).

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) with x, y coordinates.
    amplitude : float
        Peak amplitude.
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations along principal axes.
    theta : float
        Rotation angle in radians.
    offset : float
        Background offset.

    Returns
    -------
    z : ndarray
        Surface values.
    """
    x = xy[0]
    y = xy[1]

    # Rotate coordinates
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)
    xr = (x - x0) * cos_t + (y - y0) * sin_t
    yr = -(x - x0) * sin_t + (y - y0) * cos_t

    return (
        amplitude * jnp.exp(-(xr**2 / (2 * sigma_x**2) + yr**2 / (2 * sigma_y**2)))
        + offset
    )


# =============================================================================
# Radioactive Decay Model
# =============================================================================


def radioactive_decay(t, N0, decay_constant):
    """Radioactive decay model.

    Mathematical form:
        N(t) = N0 * exp(-lambda * t)

    Parameters
    ----------
    t : array_like
        Time
    N0 : float
        Initial activity/count
    decay_constant : float
        Decay constant (lambda)

    Returns
    -------
    N : array_like
        Activity at time t
    """
    return N0 * jnp.exp(-decay_constant * t)
