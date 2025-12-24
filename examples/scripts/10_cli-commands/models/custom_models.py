"""Custom models for NLSQ CLI workflow demonstrations.

This module provides custom model functions for use with the NLSQ CLI.
Each model can optionally include:
- estimate_p0(): Automatic initial parameter estimation
- bounds(): Default parameter bounds

IMPORTANT: Use jax.numpy (jnp) for model functions to enable
GPU/TPU acceleration and automatic differentiation.
"""

import jax.numpy as jnp
import numpy as np

# =============================================================================
# Michaelis-Menten Enzyme Kinetics
# =============================================================================


def michaelis_menten(S, Vmax, Km):
    """Michaelis-Menten enzyme kinetics model.

    Model: v = Vmax * [S] / (Km + [S])

    Parameters
    ----------
    S : array_like
        Substrate concentration (uM)
    Vmax : float
        Maximum reaction velocity (uM/min)
    Km : float
        Michaelis constant (uM) - substrate concentration at v = Vmax/2

    Returns
    -------
    v : array_like
        Reaction velocity (uM/min)

    Notes
    -----
    - At [S] = Km, v = Vmax/2
    - At [S] >> Km, v approaches Vmax (zero-order kinetics)
    - At [S] << Km, v = (Vmax/Km)*[S] (first-order kinetics)
    """
    return Vmax * S / (Km + S)


def michaelis_menten_estimate_p0(xdata, ydata):
    """Estimate initial parameters for Michaelis-Menten model.

    Strategy:
    - Vmax: Maximum observed velocity * 1.2 (account for incomplete saturation)
    - Km: Substrate concentration at half-maximum velocity

    Parameters
    ----------
    xdata : ndarray
        Substrate concentration data
    ydata : ndarray
        Velocity data

    Returns
    -------
    p0 : list[float]
        Initial parameter estimates [Vmax, Km]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Estimate Vmax from maximum velocity
    Vmax = np.max(ydata) * 1.2

    # Estimate Km from half-maximum velocity
    v_half = Vmax / 2
    idx_half = np.argmin(np.abs(ydata - v_half))
    Km = xdata[idx_half] if idx_half > 0 else np.median(xdata)

    return [float(Vmax), float(Km)]


def michaelis_menten_bounds():
    """Default parameter bounds for Michaelis-Menten model.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [Vmax, Km]
    """
    lower = [0.0, 0.0]  # Both must be positive
    upper = [np.inf, np.inf]

    return (lower, upper)


# =============================================================================
# Damped Oscillator
# =============================================================================


def damped_oscillator(x, amplitude, decay, frequency, phase):
    """Damped sinusoidal oscillator model.

    Model: y = amplitude * exp(-decay * x) * cos(frequency * x + phase)

    This model describes systems like:
    - Mechanical vibrations with damping
    - RLC circuit transient response
    - Damped pendulum motion

    Parameters
    ----------
    x : array_like
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
    y : array_like
        Dependent variable (displacement, voltage, etc.)

    Notes
    -----
    - Period: T = 2 * pi / frequency
    - Half-life of amplitude: t_half = ln(2) / decay
    - At x=0: y = amplitude * cos(phase)
    """
    return amplitude * jnp.exp(-decay * x) * jnp.cos(frequency * x + phase)


def damped_oscillator_estimate_p0(xdata, ydata):
    """Estimate initial parameters for damped oscillator model.

    Strategy:
    - amplitude: Maximum absolute value of y
    - decay: Estimated from envelope decay
    - frequency: Estimated from zero crossings
    - phase: Estimated from initial value

    Parameters
    ----------
    xdata : ndarray
        Independent variable data
    ydata : ndarray
        Dependent variable data

    Returns
    -------
    p0 : list[float]
        Initial parameter estimates [amplitude, decay, frequency, phase]
    """
    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Amplitude: maximum absolute value
    amplitude = np.max(np.abs(ydata))
    if amplitude == 0:
        amplitude = 1.0

    # Decay rate: estimate from envelope
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
                decay = max(-slope, 0.01)
            except np.linalg.LinAlgError:
                decay = 0.1
        else:
            decay = 0.1
    else:
        x_range = np.max(xdata) - np.min(xdata)
        decay = 1.0 / x_range if x_range > 0 else 0.1

    # Frequency: estimate from zero crossings
    zero_crossings = []
    for i in range(len(ydata) - 1):
        if ydata[i] * ydata[i + 1] < 0:
            # Linear interpolation to find crossing point
            x_cross = xdata[i] - ydata[i] * (xdata[i + 1] - xdata[i]) / (
                ydata[i + 1] - ydata[i]
            )
            zero_crossings.append(x_cross)

    if len(zero_crossings) >= 2:
        crossings = np.array(zero_crossings)
        periods = np.diff(crossings) * 2
        avg_period = np.mean(periods)
        frequency = 2 * np.pi / avg_period if avg_period > 0 else 1.0
    else:
        x_range = np.max(xdata) - np.min(xdata)
        frequency = 2 * np.pi / x_range if x_range > 0 else 1.0

    # Phase: estimate from initial value
    y0 = ydata[0]
    if amplitude > 0 and abs(y0 / amplitude) <= 1:
        phase = np.arccos(y0 / amplitude)
        if len(ydata) > 1 and ydata[1] < ydata[0]:
            phase = -phase
    else:
        phase = 0.0

    return [float(amplitude), float(decay), float(frequency), float(phase)]


def damped_oscillator_bounds():
    """Default parameter bounds for damped oscillator model.

    Returns
    -------
    bounds : tuple[list[float], list[float]]
        (lower_bounds, upper_bounds) for [amplitude, decay, frequency, phase]
    """
    lower = [0.0, 0.0, 0.0, -2 * np.pi]
    upper = [np.inf, np.inf, np.inf, 2 * np.pi]

    return (lower, upper)


# =============================================================================
# Helper functions for automatic discovery
# =============================================================================

# Map model names to their estimate_p0 and bounds functions
_MODEL_HELPERS = {
    "michaelis_menten": {
        "estimate_p0": michaelis_menten_estimate_p0,
        "bounds": michaelis_menten_bounds,
    },
    "damped_oscillator": {
        "estimate_p0": damped_oscillator_estimate_p0,
        "bounds": damped_oscillator_bounds,
    },
}


def estimate_p0(xdata, ydata, model_name="damped_oscillator"):
    """Generic estimate_p0 dispatcher.

    This is the function called by NLSQ CLI when auto_p0=true.
    Defaults to damped_oscillator for backward compatibility.
    """
    if model_name in _MODEL_HELPERS:
        return _MODEL_HELPERS[model_name]["estimate_p0"](xdata, ydata)
    return damped_oscillator_estimate_p0(xdata, ydata)


def bounds(model_name="damped_oscillator"):
    """Generic bounds dispatcher.

    This is the function called by NLSQ CLI when auto_bounds=true.
    Defaults to damped_oscillator for backward compatibility.
    """
    if model_name in _MODEL_HELPERS:
        return _MODEL_HELPERS[model_name]["bounds"]()
    return damped_oscillator_bounds()
