"""
Smart Parameter Bounds Inference
=================================

This module provides automatic inference of reasonable parameter bounds based on
data characteristics and model structure.

Key Features:
- Data-driven bound estimation from x/y ranges
- Model-aware heuristics for common function types
- Conservative bounds that avoid over-constraining
- Support for user-provided partial bounds

Example:
    >>> from nlsq.precision.bound_inference import infer_bounds
    >>>
    >>> bounds = infer_bounds(xdata, ydata, p0=[1, 0.5, 0.1])
    >>> print(f"Lower bounds: {bounds[0]}")
    >>> print(f"Upper bounds: {bounds[1]}")
"""

from typing import Any

import numpy as np

__all__ = [
    "BoundsInference",
    "infer_bounds",
    "infer_bounds_for_multistart",
    "merge_bounds",
]


class BoundsInference:
    """
    Infer reasonable parameter bounds from data characteristics.

    This class implements heuristics to estimate parameter bounds that help
    constrain optimization without being overly restrictive. The bounds are
    based on data ranges, parameter scales, and common patterns.

    Parameters
    ----------
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like
        Initial parameter guess
    safety_factor : float, optional
        Multiplier for bound ranges (larger = more conservative). Default: 10.0
    enforce_positivity : bool, optional
        Force all bounds to be non-negative if p0 is positive. Default: True

    Attributes
    ----------
    x_min, x_max : float
        Range of independent variable
    y_min, y_max : float
        Range of dependent variable
    x_range, y_range : float
        Span of data

    Examples
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2.5 * np.exp(-0.5 * x) + 1.0
    >>> p0 = [2.0, 0.5, 1.0]
    >>>
    >>> inference = BoundsInference(x, y, p0)
    >>> bounds = inference.infer()
    >>> print(bounds)
    ([0.0, 0.0, 0.0], [25.0, 5.0, 10.0])
    """

    def __init__(
        self,
        xdata,
        ydata,
        p0,
        safety_factor: float = 10.0,
        enforce_positivity: bool = True,
    ):
        """Initialize bounds inference."""
        self.xdata = np.asarray(xdata)
        self.ydata = np.asarray(ydata)
        self.p0 = np.asarray(p0)
        self.safety_factor = safety_factor
        self.enforce_positivity = enforce_positivity

        # Data statistics
        self.x_min = np.min(self.xdata)
        self.x_max = np.max(self.xdata)
        self.y_min = np.min(self.ydata)
        self.y_max = np.max(self.ydata)

        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min

        # Handle edge cases
        if self.x_range == 0:
            self.x_range = 1.0
        if self.y_range == 0:
            self.y_range = 1.0

    def infer(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Infer bounds for all parameters.

        Returns
        -------
        lower_bounds : ndarray
            Lower bounds for each parameter
        upper_bounds : ndarray
            Upper bounds for each parameter
        """
        n_params = len(self.p0)
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)

        for i in range(n_params):
            lower[i], upper[i] = self._infer_parameter_bounds(i)

        return lower, upper

    def _infer_parameter_bounds(self, param_idx: int) -> tuple[float, float]:
        """
        Infer bounds for a single parameter.

        Parameters
        ----------
        param_idx : int
            Index of parameter in p0

        Returns
        -------
        lower : float
            Lower bound
        upper : float
            Upper bound
        """
        p0_val = self.p0[param_idx]

        # Strategy 1: Bounds based on parameter magnitude
        if p0_val != 0:
            # Use parameter magnitude as scale
            scale = abs(p0_val)
            lower_mag = p0_val / self.safety_factor
            upper_mag = p0_val * self.safety_factor
        else:
            # For zero p0, use data range as scale
            scale = max(self.y_range, 1.0)
            lower_mag = -scale * self.safety_factor
            upper_mag = scale * self.safety_factor

        # Strategy 2: Bounds based on data characteristics
        # First parameter often relates to amplitude/scale
        if param_idx == 0:
            # Amplitude usually related to y-range
            lower_data = self.y_min - self.y_range
            upper_data = self.y_max + self.y_range
        # Second parameter often relates to rate/frequency
        elif param_idx == 1:
            # Rate parameters often in range [1/x_range, x_range]
            if self.x_range > 0:
                lower_data = 0.1 / self.x_range
                upper_data = 10.0 / self.x_range
            else:
                lower_data = 0.01
                upper_data = 100.0
        # Third+ parameters often offsets or secondary effects
        else:
            lower_data = self.y_min - self.y_range
            upper_data = self.y_max + self.y_range

        # Combine strategies (take union for safety)
        lower = min(lower_mag, lower_data)
        upper = max(upper_mag, upper_data)

        # Enforce positivity if requested and p0 is positive
        if self.enforce_positivity and p0_val > 0:
            lower = max(0, lower)
            # Also enforce positivity for rate-like parameters (param_idx == 1)
            if param_idx == 1:
                lower = max(1e-10, lower)

        # Ensure lower < upper
        if lower >= upper:
            # Fallback to symmetric bounds around p0
            if p0_val != 0:
                lower = p0_val / self.safety_factor
                upper = p0_val * self.safety_factor
            else:
                lower = -self.safety_factor
                upper = self.safety_factor

        return lower, upper


def infer_bounds(
    xdata,
    ydata,
    p0,
    safety_factor: float = 10.0,
    enforce_positivity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer reasonable parameter bounds from data and initial guess.

    This is a convenience function that creates a BoundsInference instance
    and returns the inferred bounds.

    Parameters
    ----------
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like
        Initial parameter guess
    safety_factor : float, optional
        Multiplier for bound ranges (larger = more conservative). Default: 10.0
    enforce_positivity : bool, optional
        Force all bounds to be non-negative if p0 is positive. Default: True

    Returns
    -------
    lower_bounds : ndarray
        Lower bounds for each parameter
    upper_bounds : ndarray
        Upper bounds for each parameter

    Examples
    --------
    Basic usage:

    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2.5 * np.exp(-0.5 * x) + 1.0
    >>> p0 = [2.0, 0.5, 1.0]
    >>> bounds = infer_bounds(x, y, p0)
    >>> print(f"Lower: {bounds[0]}")
    >>> print(f"Upper: {bounds[1]}")

    With custom safety factor:

    >>> bounds = infer_bounds(x, y, p0, safety_factor=20.0)  # More conservative

    Allow negative parameters:

    >>> bounds = infer_bounds(x, y, p0, enforce_positivity=False)
    """
    inference = BoundsInference(
        xdata,
        ydata,
        p0,
        safety_factor=safety_factor,
        enforce_positivity=enforce_positivity,
    )
    return inference.infer()


def infer_bounds_for_multistart(
    xdata,
    ydata,
    p0,
    user_bounds: tuple | None = None,
    safety_factor: float = 20.0,
    enforce_positivity: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Infer bounds suitable for multi-start LHS sampling.

    This is a wrapper around infer_bounds() with defaults appropriate for
    global exploration in multi-start optimization. Uses a larger safety_factor
    to allow broader exploration of the parameter space.

    Parameters
    ----------
    xdata : array_like
        Independent variable data
    ydata : array_like
        Dependent variable data
    p0 : array_like
        Initial parameter guess (used for centering and scale inference)
    user_bounds : tuple, optional
        User-provided (lower, upper) bounds. If provided and finite, these
        take precedence over inferred bounds.
    safety_factor : float, optional
        Multiplier for bound ranges. Default: 20.0 (larger than standard
        infer_bounds to allow broader exploration)
    enforce_positivity : bool, optional
        Force all bounds to be non-negative if p0 is positive. Default: True

    Returns
    -------
    lower_bounds : ndarray
        Lower bounds suitable for LHS sampling
    upper_bounds : ndarray
        Upper bounds suitable for LHS sampling

    Notes
    -----
    This function ensures that returned bounds are always finite, which is
    required for Latin Hypercube Sampling. If user provides infinite bounds,
    they are replaced with inferred finite bounds.

    The default safety_factor of 20.0 is larger than the standard 10.0 used
    in infer_bounds() to allow for broader exploration during multi-start
    optimization.

    Examples
    --------
    Basic usage for multi-start:

    >>> import numpy as np
    >>> x = np.linspace(0, 10, 100)
    >>> y = 2.5 * np.exp(-0.5 * x) + 1.0
    >>> p0 = [2.0, 0.5, 1.0]
    >>> bounds = infer_bounds_for_multistart(x, y, p0)
    >>> # bounds will be wider than standard infer_bounds

    With user-provided bounds (finite bounds are preserved):

    >>> user_bounds = ([0, 0, 0], [10, np.inf, 5])
    >>> bounds = infer_bounds_for_multistart(x, y, p0, user_bounds=user_bounds)
    >>> # First and third params use user bounds, second is inferred

    See Also
    --------
    infer_bounds : Standard bound inference with smaller safety_factor
    merge_bounds : Merge user-provided and inferred bounds
    """
    p0 = np.asarray(p0)

    # Infer bounds with larger safety factor for exploration
    inferred_bounds = infer_bounds(
        xdata,
        ydata,
        p0,
        safety_factor=safety_factor,
        enforce_positivity=enforce_positivity,
    )

    # If user provides bounds, merge them
    if user_bounds is not None:
        merged_bounds = merge_bounds(inferred_bounds, user_bounds)
    else:
        merged_bounds = inferred_bounds

    # Ensure all bounds are finite (required for LHS)
    lower, upper = merged_bounds

    # Replace any remaining infinities with inferred values
    inferred_lower, inferred_upper = inferred_bounds

    lower = np.where(np.isfinite(lower), lower, inferred_lower)
    upper = np.where(np.isfinite(upper), upper, inferred_upper)

    # Final check: if bounds are still not finite, use fallback
    if not np.all(np.isfinite(lower)):
        # Fallback for lower bounds: use p0 / safety_factor or -safety_factor
        for i in range(len(lower)):
            if not np.isfinite(lower[i]):
                if p0[i] > 0:
                    lower[i] = p0[i] / safety_factor
                elif p0[i] < 0:
                    lower[i] = p0[i] * safety_factor
                else:
                    lower[i] = -safety_factor

    if not np.all(np.isfinite(upper)):
        # Fallback for upper bounds: use p0 * safety_factor or safety_factor
        for i in range(len(upper)):
            if not np.isfinite(upper[i]):
                if p0[i] > 0:
                    upper[i] = p0[i] * safety_factor
                elif p0[i] < 0:
                    upper[i] = p0[i] / safety_factor
                else:
                    upper[i] = safety_factor

    return lower, upper


def merge_bounds(
    inferred_bounds: tuple[np.ndarray, np.ndarray],
    user_bounds: tuple | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge user-provided bounds with inferred bounds.

    User-provided bounds take precedence. If user provides partial bounds
    (e.g., only lower or only for some parameters), the remaining bounds
    are filled from inferred bounds.

    Parameters
    ----------
    inferred_bounds : tuple of ndarray
        Inferred (lower, upper) bounds
    user_bounds : tuple of array_like or None, optional
        User-provided (lower, upper) bounds. Can contain -np.inf or np.inf
        for unbounded parameters.

    Returns
    -------
    lower_bounds : ndarray
        Merged lower bounds
    upper_bounds : ndarray
        Merged upper bounds

    Examples
    --------
    >>> inferred = (np.array([0, 0, 0]), np.array([10, 5, 10]))
    >>> user = (np.array([1, -np.inf, 0]), np.array([5, np.inf, 10]))
    >>> merged = merge_bounds(inferred, user)
    >>> print(merged)
    (array([1., 0., 0.]), array([5., 5., 10.]))

    Notes
    -----
    - If user provides -np.inf for lower bound, use inferred lower bound
    - If user provides np.inf for upper bound, use inferred upper bound
    - Scalar user bounds are broadcast to all parameters
    """
    lower_inferred, upper_inferred = inferred_bounds

    if user_bounds is None:
        return lower_inferred, upper_inferred

    lower_user, upper_user = user_bounds

    # Check if user bounds are effectively infinite (all -inf, all +inf)
    lower_user_arr = np.atleast_1d(lower_user)
    upper_user_arr = np.atleast_1d(upper_user)
    if np.all(np.isneginf(lower_user_arr)) and np.all(np.isposinf(upper_user_arr)):
        return lower_inferred, upper_inferred

    # Convert to arrays
    lower_user = np.atleast_1d(lower_user)
    upper_user = np.atleast_1d(upper_user)

    # Broadcast scalar bounds to all parameters
    n_params = len(lower_inferred)
    if len(lower_user) == 1:
        lower_user = np.full(n_params, lower_user[0])
    if len(upper_user) == 1:
        upper_user = np.full(n_params, upper_user[0])

    # Merge bounds: user takes precedence, but use inferred for inf values
    lower_merged = np.where(np.isfinite(lower_user), lower_user, lower_inferred)
    upper_merged = np.where(np.isfinite(upper_user), upper_user, upper_inferred)

    return lower_merged, upper_merged


def analyze_bounds_quality(
    bounds: tuple[np.ndarray, np.ndarray],
    p0: np.ndarray,
) -> dict[str, Any]:
    """
    Analyze quality and characteristics of parameter bounds.

    Parameters
    ----------
    bounds : tuple of ndarray
        (lower, upper) bounds
    p0 : ndarray
        Initial parameter guess

    Returns
    -------
    analysis : dict
        Dictionary with quality metrics:
        - is_feasible: bool, whether p0 is within bounds
        - bound_ratios: ndarray, upper/lower ratios for each parameter
        - avg_bound_ratio: float, geometric mean of bound ratios
        - tight_parameters: list, indices of tightly constrained parameters
        - loose_parameters: list, indices of loosely constrained parameters

    Examples
    --------
    >>> bounds = (np.array([0, 0]), np.array([10, 100]))
    >>> p0 = np.array([5, 50])
    >>> analysis = analyze_bounds_quality(bounds, p0)
    >>> print(f"Feasible: {analysis['is_feasible']}")
    >>> print(f"Average bound ratio: {analysis['avg_bound_ratio']:.1f}")
    """
    lower, upper = bounds
    p0 = np.asarray(p0)

    # Check feasibility
    is_feasible = np.all((p0 >= lower) & (p0 <= upper))

    # Compute bound ratios (handling zero lower bounds)
    bound_ratios = np.zeros_like(lower, dtype=float)
    for i in range(len(lower)):
        if lower[i] > 0:
            bound_ratios[i] = upper[i] / lower[i]
        elif lower[i] == 0 and upper[i] > 0:
            bound_ratios[i] = np.inf
        else:
            # Negative lower bound
            bound_ratios[i] = (upper[i] - lower[i]) / max(abs(lower[i]), abs(upper[i]))

    # Geometric mean of finite ratios
    finite_ratios = bound_ratios[np.isfinite(bound_ratios)]
    if len(finite_ratios) > 0:
        avg_bound_ratio = np.exp(np.mean(np.log(finite_ratios)))
    else:
        avg_bound_ratio = np.inf

    # Identify tight and loose parameters
    tight_parameters = list(np.where(bound_ratios < 5.0)[0])
    loose_parameters = list(np.where(bound_ratios > 50.0)[0])

    return {
        "is_feasible": bool(is_feasible),
        "bound_ratios": bound_ratios,
        "avg_bound_ratio": float(avg_bound_ratio),
        "tight_parameters": tight_parameters,
        "loose_parameters": loose_parameters,
    }
