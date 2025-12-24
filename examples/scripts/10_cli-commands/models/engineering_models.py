"""Engineering Models for NLSQ CLI Demonstrations.

This module contains custom model functions for engineering curve fitting,
extracted from the advanced gallery examples.

Models:
    - sensor_calibration: Linear with offset
    - transfer_function: First-order system response
    - power_law: Power law relationship
    - arrhenius: Temperature-dependent rate constant

Usage in workflow YAML:
    model:
      type: custom
      path: models/engineering_models.py
      function: sensor_calibration
"""

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Sensor Calibration (Linear with Offset)
# =============================================================================


def sensor_calibration(x, sensitivity, offset):
    """Linear sensor calibration model.

    Mathematical form:
        y = sensitivity * x + offset

    Parameters
    ----------
    x : array_like
        True physical value (input)
    sensitivity : float
        Sensor sensitivity (slope, output per unit input)
    offset : float
        Zero offset (output when input is zero)

    Returns
    -------
    y : array_like
        Sensor reading (output)
    """
    return sensitivity * x + offset


def estimate_p0(xdata, ydata):
    """Estimate initial parameters for sensor calibration.

    Parameters
    ----------
    xdata : ndarray
        Input data
    ydata : ndarray
        Output data

    Returns
    -------
    p0 : list[float]
        Initial estimates [sensitivity, offset]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Simple linear regression
    n = len(xdata)
    x_mean = np.mean(xdata)
    y_mean = np.mean(ydata)

    sensitivity = float(
        np.sum((xdata - x_mean) * (ydata - y_mean)) / np.sum((xdata - x_mean) ** 2)
    )
    offset = float(y_mean - sensitivity * x_mean)

    return [sensitivity, offset]


def bounds():
    """Return default parameter bounds for sensor calibration.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [sensitivity, offset]
    """
    lower = [-np.inf, -np.inf]
    upper = [np.inf, np.inf]
    return (lower, upper)


# =============================================================================
# First-Order System Response
# =============================================================================


def first_order_response(t, K, tau, y0):
    """First-order system step response.

    Mathematical form:
        y(t) = y0 + K * (1 - exp(-t / tau))

    Parameters
    ----------
    t : array_like
        Time
    K : float
        Steady-state gain
    tau : float
        Time constant
    y0 : float
        Initial value

    Returns
    -------
    y : array_like
        System response
    """
    return y0 + K * (1 - jnp.exp(-t / tau))


# =============================================================================
# Power Law Model
# =============================================================================


def power_law(x, a, b):
    """Power law model.

    Mathematical form:
        y = a * x^b

    Parameters
    ----------
    x : array_like
        Independent variable
    a : float
        Coefficient (amplitude)
    b : float
        Exponent

    Returns
    -------
    y : array_like
        Dependent variable
    """
    return a * jnp.power(x, b)


# =============================================================================
# Power Law with Offset
# =============================================================================


def power_law_offset(x, a, b, c):
    """Power law model with offset.

    Mathematical form:
        y = a * x^b + c

    Parameters
    ----------
    x : array_like
        Independent variable
    a : float
        Coefficient (amplitude)
    b : float
        Exponent
    c : float
        Offset

    Returns
    -------
    y : array_like
        Dependent variable
    """
    return a * jnp.power(x, b) + c


# =============================================================================
# Arrhenius Equation
# =============================================================================


def arrhenius(T, A, Ea):
    """Arrhenius equation for temperature-dependent rate constants.

    Mathematical form:
        k = A * exp(-Ea / (R * T))

    where R = 8.314 J/(mol·K) is the gas constant.

    Parameters
    ----------
    T : array_like
        Absolute temperature (K)
    A : float
        Pre-exponential factor (frequency factor)
    Ea : float
        Activation energy (J/mol)

    Returns
    -------
    k : array_like
        Rate constant
    """
    R = 8.314  # Gas constant J/(mol·K)
    return A * jnp.exp(-Ea / (R * T))


# =============================================================================
# Stress-Strain (Linear Elastic)
# =============================================================================


def linear_elastic(strain, E, sigma0):
    """Linear elastic stress-strain model.

    Mathematical form:
        stress = E * strain + sigma0

    Parameters
    ----------
    strain : array_like
        Engineering strain
    E : float
        Young's modulus (elastic modulus)
    sigma0 : float
        Initial stress (prestress/offset)

    Returns
    -------
    stress : array_like
        Engineering stress
    """
    return E * strain + sigma0


# =============================================================================
# RLC Circuit Resonance
# =============================================================================


def rlc_resonance(omega, A, omega0, Q):
    """RLC circuit frequency response (amplitude).

    Mathematical form:
        |H(omega)| = A / sqrt((1 - (omega/omega0)^2)^2 + (omega/(Q*omega0))^2)

    Parameters
    ----------
    omega : array_like
        Angular frequency (rad/s)
    A : float
        Peak amplitude
    omega0 : float
        Resonant frequency (rad/s)
    Q : float
        Quality factor

    Returns
    -------
    H : array_like
        Response amplitude
    """
    omega_ratio = omega / omega0
    denominator = jnp.sqrt((1 - omega_ratio**2) ** 2 + (omega_ratio / Q) ** 2)
    return A / denominator
