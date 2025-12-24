"""2D Gaussian surface model for CLI testing."""

import jax.numpy as jnp


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """2D Gaussian surface model.

    Parameters
    ----------
    xy : ndarray
        Shape (2, n_points) where xy[0] = x coordinates, xy[1] = y coordinates.
    amplitude : float
        Peak amplitude.
    x0, y0 : float
        Center coordinates.
    sigma_x, sigma_y : float
        Standard deviations in x and y directions.
    offset : float
        Background offset.

    Returns
    -------
    ndarray
        Shape (n_points,) with z values.
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


def estimate_p0(xdata, ydata):
    """Estimate initial parameters from data.

    Parameters
    ----------
    xdata : ndarray
        Shape (2, n_points) for 2D data.
    ydata : ndarray
        Shape (n_points,) z values.

    Returns
    -------
    list
        Initial parameter guess [amplitude, x0, y0, sigma_x, sigma_y, offset].
    """
    x = xdata[0]
    y = xdata[1]

    # Estimate offset from minimum
    offset = float(ydata.min())

    # Estimate amplitude from range
    amplitude = float(ydata.max() - offset)

    # Estimate center from weighted average
    weights = ydata - offset
    weights = jnp.maximum(weights, 0)
    total_weight = jnp.sum(weights)
    if total_weight > 0:
        x0 = float(jnp.sum(x * weights) / total_weight)
        y0 = float(jnp.sum(y * weights) / total_weight)
    else:
        x0 = float(jnp.mean(x))
        y0 = float(jnp.mean(y))

    # Estimate sigma from data range
    sigma_x = float((x.max() - x.min()) / 4)
    sigma_y = float((y.max() - y.min()) / 4)

    return [amplitude, x0, y0, sigma_x, sigma_y, offset]


def bounds():
    """Return parameter bounds.

    Returns
    -------
    tuple
        (lower_bounds, upper_bounds) for [amplitude, x0, y0, sigma_x, sigma_y, offset].
    """
    import numpy as np

    return (
        [0.0, -np.inf, -np.inf, 1e-6, 1e-6, -np.inf],  # Lower bounds
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],  # Upper bounds
    )


# Attach methods to the model function for CLI auto-detection
gaussian_2d.estimate_p0 = estimate_p0
gaussian_2d.bounds = bounds
