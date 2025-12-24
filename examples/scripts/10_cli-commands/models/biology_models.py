"""Biology Models for NLSQ CLI Demonstrations.

This module contains custom model functions for biological curve fitting,
extracted from the advanced gallery examples.

Models:
    - michaelis_menten: Enzyme kinetics (Michaelis-Menten)
    - hill_equation: Cooperative binding (Hill equation)
    - competitive_inhibition: Enzyme inhibition model
    - logistic_growth: Population growth model

Usage in workflow YAML:
    model:
      type: custom
      path: models/biology_models.py
      function: michaelis_menten
"""

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Michaelis-Menten Enzyme Kinetics
# =============================================================================


def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten enzyme kinetics model.

    Mathematical form:
        v = Vmax * [S] / (Km + [S])

    Parameters
    ----------
    S : array_like
        Substrate concentration
    Vmax : float
        Maximum reaction velocity
    Km : float
        Michaelis constant (substrate concentration at v = Vmax/2)

    Returns
    -------
    v : array_like
        Reaction velocity
    """
    return Vmax * S / (Km + S)


def estimate_p0(xdata, ydata):
    """Estimate initial parameters for Michaelis-Menten.

    Parameters
    ----------
    xdata : ndarray
        Substrate concentration data
    ydata : ndarray
        Velocity data

    Returns
    -------
    p0 : list[float]
        Initial estimates [Vmax, Km]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Vmax: maximum observed velocity (plateau estimate)
    Vmax = float(np.max(ydata)) * 1.2  # Add 20% buffer

    # Km: substrate concentration at half Vmax
    half_Vmax = Vmax / 2
    # Find the x value closest to half Vmax
    idx = np.argmin(np.abs(ydata - half_Vmax))
    Km = float(xdata[idx])

    if Km <= 0:
        Km = float(np.median(xdata))

    return [Vmax, Km]


def bounds():
    """Return default parameter bounds for Michaelis-Menten.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [Vmax, Km]
    """
    lower = [0.0, 0.0]
    upper = [np.inf, np.inf]
    return (lower, upper)


# =============================================================================
# Hill Equation (Cooperative Binding)
# =============================================================================


def hill_equation(S, Vmax, K50, n):
    """Hill equation for cooperative binding.

    Mathematical form:
        v = Vmax * [S]^n / (K50^n + [S]^n)

    Parameters
    ----------
    S : array_like
        Substrate/ligand concentration
    Vmax : float
        Maximum velocity/response
    K50 : float
        Half-saturation constant
    n : float
        Hill coefficient (cooperativity)
        - n > 1: positive cooperativity
        - n = 1: no cooperativity (Michaelis-Menten)
        - n < 1: negative cooperativity

    Returns
    -------
    v : array_like
        Reaction velocity/response
    """
    return Vmax * jnp.power(S, n) / (jnp.power(K50, n) + jnp.power(S, n))


# =============================================================================
# Competitive Inhibition
# =============================================================================


def competitive_inhibition(S, Vmax, Km, Ki, I=100.0):
    """Competitive inhibition model.

    Mathematical form:
        v = Vmax * [S] / (Km * (1 + [I]/Ki) + [S])

    Note: Inhibitor concentration I is a constant (default 100.0).
    For variable I, create a wrapper function.

    Parameters
    ----------
    S : array_like
        Substrate concentration
    Vmax : float
        Maximum velocity
    Km : float
        Michaelis constant (without inhibitor)
    Ki : float
        Inhibition constant
    I : float, optional
        Inhibitor concentration (default: 100.0)

    Returns
    -------
    v : array_like
        Reaction velocity
    """
    Km_app = Km * (1 + I / Ki)
    return Vmax * S / (Km_app + S)


# =============================================================================
# Logistic Growth Model
# =============================================================================


def logistic_growth(t, N0, K, r):
    """Logistic growth model.

    Mathematical form:
        N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))

    Parameters
    ----------
    t : array_like
        Time
    N0 : float
        Initial population size
    K : float
        Carrying capacity
    r : float
        Growth rate

    Returns
    -------
    N : array_like
        Population size at time t
    """
    return K / (1 + ((K - N0) / N0) * jnp.exp(-r * t))


# =============================================================================
# Four-Parameter Logistic (4PL) - Dose-Response
# =============================================================================


def four_parameter_logistic(x, A, B, C, D):
    """Four-parameter logistic model for dose-response curves.

    Mathematical form:
        y = D + (A - D) / (1 + (x / C)^B)

    Parameters
    ----------
    x : array_like
        Dose/concentration
    A : float
        Minimum asymptote (lower plateau)
    B : float
        Hill slope (steepness)
    C : float
        EC50/IC50 (inflection point)
    D : float
        Maximum asymptote (upper plateau)

    Returns
    -------
    y : array_like
        Response
    """
    return D + (A - D) / (1 + jnp.power(x / C, B))
