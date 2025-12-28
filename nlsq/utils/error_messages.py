"""Enhanced error messages with diagnostics and recommendations.

This module provides informative error messages that help users debug
optimization failures by providing diagnostics and actionable recommendations.

The main component is the :class:`OptimizationError` exception, which is raised
when curve fitting fails to converge. Unlike generic errors, it provides:

- **Diagnostics**: Final cost, gradient norm, iterations, function evaluations
- **Reasons**: Why the optimization failed (e.g., max iterations, gradient too large)
- **Recommendations**: Actionable suggestions to fix the issue

Examples
--------
The enhanced error messages are raised automatically when optimization fails:

>>> from nlsq import curve_fit
>>> import jax.numpy as jnp
>>> import numpy as np
>>>
>>> def difficult_func(x, a, b):
...     return a * jnp.exp(b * x**2)
>>>
>>> xdata = np.linspace(0, 1, 10)
>>> ydata = difficult_func(xdata, 1, -5)
>>>
>>> try:
...     popt, pcov = curve_fit(difficult_func, xdata, ydata, p0=[0.1, 0.1], max_nfev=5)
... except OptimizationError as e:
...     print(e)
...     # Prints detailed diagnostics and recommendations
...     print(f"Reasons: {e.reasons}")
...     print(f"Recommendations: {e.recommendations}")

See Also
--------
nlsq.minpack.curve_fit : Main curve fitting function
nlsq.parameter_estimation : Automatic parameter estimation
"""

from typing import Any

import numpy as np


class OptimizationDiagnostics:
    """Collect and analyze optimization diagnostics.

    Parameters
    ----------
    result : OptimizeResult
        Optimization result object

    Attributes
    ----------
    cost : float or None
        Final cost function value
    gradient_norm : float or None
        Norm of final gradient
    nfev : int
        Number of function evaluations
    nit : int
        Number of iterations
    """

    def __init__(self, result):
        self.result = result
        self.cost = getattr(result, "cost", None)
        self.gradient_norm = getattr(result, "grad", None)
        self.nfev = getattr(result, "nfev", 0)
        self.nit = getattr(result, "nit", 0)


def analyze_failure(
    result, gtol: float, ftol: float, xtol: float, max_nfev: int
) -> tuple[list[str], list[str]]:
    """Analyze why optimization failed and generate recommendations.

    Parameters
    ----------
    result : OptimizeResult
        Optimization result object
    gtol : float
        Gradient tolerance
    ftol : float
        Function tolerance
    xtol : float
        Parameter tolerance
    max_nfev : int
        Maximum number of function evaluations

    Returns
    -------
    reasons : list of str
        Why the optimization failed
    recommendations : list of str
        What the user should try
    """
    reasons = []
    recommendations = []

    # Check gradient convergence
    if hasattr(result, "grad") and result.grad is not None:
        grad = np.asarray(result.grad)
        grad_norm = np.linalg.norm(grad, ord=np.inf)
        if grad_norm > gtol:
            reasons.append(
                f"Gradient norm {grad_norm:.2e} exceeds tolerance {gtol:.2e}"
            )
            recommendations.append(
                f"✓ Try looser gradient tolerance: gtol={gtol * 10:.1e}"
            )
            recommendations.append("✓ Check if initial guess p0 is reasonable")
            recommendations.append("✓ Consider parameter scaling with x_scale")

    # Check max iterations
    if hasattr(result, "nfev") and result.nfev >= max_nfev:
        reasons.append(f"Reached maximum function evaluations ({max_nfev})")
        recommendations.append(f"✓ Increase iteration limit: max_nfev={max_nfev * 2}")
        recommendations.append("✓ Provide better initial guess p0")
        recommendations.append("✓ Try different optimization method (trf/dogbox/lm)")

    # Check for numerical issues
    if hasattr(result, "x") and result.x is not None:
        x = np.asarray(result.x)
        if not np.all(np.isfinite(x)):
            reasons.append("NaN or Inf in solution parameters")
            recommendations.append("⚠ Numerical instability detected")
            recommendations.append("✓ Add parameter bounds to constrain search")
            recommendations.append("✓ Scale parameters to similar magnitudes")
            recommendations.append("✓ Check if model function is well-defined")

    # Check cost function value
    if hasattr(result, "cost") and result.cost is not None:
        if not np.isfinite(result.cost):
            reasons.append("Cost function is NaN or Inf")
            recommendations.append("⚠ Model evaluation failed")
            recommendations.append("✓ Check model function for domain errors")
            recommendations.append("✓ Verify data doesn't contain NaN/Inf")

    # Generic recommendations if unclear
    if not recommendations:
        recommendations.append("✓ Run with verbose=2 to see iteration details")
        recommendations.append("✓ Check residual plot for systematic errors")
        recommendations.append("✓ Verify model function matches data pattern")
        recommendations.append(
            "✓ Try robust loss function if outliers present (loss='soft_l1')"
        )

    return reasons, recommendations


def format_error_message(
    reasons: list[str], recommendations: list[str], diagnostics: dict[str, Any]
) -> str:
    """Format comprehensive error message.

    Parameters
    ----------
    reasons : list of str
        Reasons for optimization failure
    recommendations : list of str
        Recommended actions
    diagnostics : dict
        Diagnostic information

    Returns
    -------
    message : str
        Formatted error message
    """
    msg = "Optimization failed to converge.\n\n"

    # Diagnostics section
    if diagnostics:
        msg += "Diagnostics:\n"
        for key, value in diagnostics.items():
            msg += f"  - {key}: {value}\n"
        msg += "\n"

    # Reasons section
    if reasons:
        msg += "Reasons:\n"
        for reason in reasons:
            msg += f"  - {reason}\n"
        msg += "\n"

    # Recommendations section
    msg += "Recommendations:\n"
    for rec in recommendations:
        msg += f"  {rec}\n"

    msg += "\nFor more help, see: https://nlsq.readthedocs.io/troubleshooting"

    return msg


class OptimizationError(RuntimeError):
    """Enhanced optimization error with diagnostics and recommendations.

    This exception provides detailed information about why an optimization
    failed and actionable recommendations for fixing the issue.

    Parameters
    ----------
    result : OptimizeResult
        Optimization result object
    gtol : float
        Gradient tolerance used
    ftol : float
        Function tolerance used
    xtol : float
        Parameter tolerance used
    max_nfev : int
        Maximum function evaluations used

    Attributes
    ----------
    result : OptimizeResult
        The optimization result
    reasons : list of str
        Reasons for failure
    recommendations : list of str
        Recommended actions
    diagnostics : dict
        Diagnostic information

    Examples
    --------
    The error is raised automatically by curve_fit when optimization fails:

    >>> from nlsq import curve_fit
    >>> from nlsq.utils.error_messages import OptimizationError
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>>
    >>> def exponential(x, a, b, c):
    ...     return a * jnp.exp(-b * x) + c
    >>>
    >>> x = np.linspace(0, 5, 50)
    >>> y = 3 * np.exp(-0.5 * x) + 1
    >>>
    >>> try:
    ...     # Force failure with very low max_nfev
    ...     popt, pcov = curve_fit(exponential, x, y, p0=[1, 1, 1], max_nfev=3)
    ... except OptimizationError as e:
    ...     # Access error attributes programmatically
    ...     if any("maximum" in r.lower() for r in e.reasons):
    ...         # Auto-retry with higher max_nfev
    ...         popt, pcov = curve_fit(exponential, x, y, p0=[1, 1, 1], max_nfev=200)
    ...         print("Auto-retry succeeded!")

    See Also
    --------
    analyze_failure : Function that analyzes why optimization failed
    format_error_message : Function that formats the error message
    """

    def __init__(self, result, gtol: float, ftol: float, xtol: float, max_nfev: int):
        self.result = result

        # Analyze failure
        reasons, recommendations = analyze_failure(result, gtol, ftol, xtol, max_nfev)

        # Collect diagnostics
        diagnostics = {}
        if hasattr(result, "cost") and result.cost is not None:
            diagnostics["Final cost"] = f"{result.cost:.6e}"

        if hasattr(result, "grad") and result.grad is not None:
            grad = np.asarray(result.grad)
            grad_norm = np.linalg.norm(grad, ord=np.inf)
            diagnostics["Gradient norm"] = f"{grad_norm:.6e}"
            diagnostics["Gradient tolerance"] = f"{gtol:.6e}"

        if hasattr(result, "nfev"):
            diagnostics["Function evaluations"] = f"{result.nfev} / {max_nfev}"

        if hasattr(result, "nit"):
            diagnostics["Iterations"] = result.nit

        if hasattr(result, "message") and result.message:
            diagnostics["Status"] = result.message

        # Format message
        msg = format_error_message(reasons, recommendations, diagnostics)

        super().__init__(msg)
        self.reasons = reasons
        self.recommendations = recommendations
        self.diagnostics = diagnostics


class ConvergenceWarning(UserWarning):
    """Warning for optimization that converged but may have issues.

    This warning is raised when optimization technically converged but
    there are potential quality issues (e.g., poor fit, covariance issues).
    """


def check_convergence_quality(result, pcov) -> list[str]:
    """Check quality of converged solution and generate warnings.

    Parameters
    ----------
    result : OptimizeResult
        Optimization result
    pcov : ndarray or None
        Parameter covariance matrix

    Returns
    -------
    warnings : list of str
        Warning messages about solution quality
    """
    warnings = []

    # Check covariance matrix
    if pcov is None or np.any(np.isnan(pcov)) or np.any(np.isinf(pcov)):
        warnings.append(
            "⚠ Parameter covariance could not be estimated. "
            "This may indicate:\n"
            "  - Parameters are at bounds\n"
            "  - Singular or ill-conditioned Jacobian\n"
            "  - Optimization converged to local minimum\n"
            "  Try: check bounds, improve p0, or use different method"
        )

    # Check for parameters at bounds
    if hasattr(result, "x") and hasattr(result, "active_mask"):
        active_mask = getattr(result, "active_mask", None)
        if active_mask is not None:
            at_bounds = np.any(active_mask != 0)
            if at_bounds:
                warnings.append(
                    "⚠ One or more parameters are at bounds. "
                    "Consider relaxing bounds or checking if model is appropriate."
                )

    # Check residuals if available
    if hasattr(result, "fun") and result.fun is not None:
        residuals = np.asarray(result.fun)
        if np.any(np.abs(residuals) > 1e3):
            warnings.append(
                "⚠ Large residuals detected. "
                "Model may not fit data well. Check:\n"
                "  - Is the model appropriate for this data?\n"
                "  - Are there outliers? (try robust loss function)\n"
                "  - Is data properly scaled?"
            )

    return warnings
