"""Parameter results display component for NLSQ GUI.

This module provides components for displaying fitted parameter values
with their uncertainties and confidence intervals in a tabular format.

Functions
---------
format_parameter_table
    Format parameters with uncertainties as a DataFrame.
compute_confidence_intervals
    Compute confidence intervals for parameters.
render_parameter_results
    Render parameter results in Streamlit.
"""

import contextlib
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# Parameter Table Formatting
# =============================================================================


def format_parameter_table(
    popt: np.ndarray,
    pcov: np.ndarray | None,
    param_names: list[str] | None = None,
    precision: int = 6,
) -> pd.DataFrame:
    """Format parameters with uncertainties as a DataFrame.

    Parameters
    ----------
    popt : np.ndarray
        Optimal parameter values from curve fitting.
    pcov : np.ndarray | None
        Parameter covariance matrix. If None, uncertainties will be NaN.
    param_names : list[str] | None
        Parameter names for row labels. If None, uses p0, p1, p2, etc.
    precision : int
        Number of significant digits for formatting.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Parameter, Value, Std Error, Rel Error (%).

    Examples
    --------
    >>> popt = np.array([2.5, 0.1, 3.0])
    >>> pcov = np.diag([0.01, 0.001, 0.05])
    >>> df = format_parameter_table(popt, pcov, ['a', 'b', 'c'])
    >>> list(df.columns)
    ['Parameter', 'Value', 'Std Error', 'Rel Error (%)']
    """
    popt = np.asarray(popt)
    n_params = len(popt)

    # Generate parameter names if not provided
    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    elif len(param_names) < n_params:
        # Extend with generic names if not enough provided
        param_names = list(param_names) + [
            f"p{i}" for i in range(len(param_names), n_params)
        ]

    # Compute standard errors
    if pcov is not None:
        pcov = np.asarray(pcov)
        diag = np.diag(pcov)
        # Handle negative diagonals (numerical artifacts) by setting to NaN
        perr = np.where(diag >= 0, np.sqrt(np.maximum(diag, 0)), np.nan)
    else:
        perr = np.full(n_params, np.nan)

    # Compute relative errors
    rel_err = np.where(
        (popt != 0) & np.isfinite(perr),
        np.abs(perr / popt) * 100,
        np.nan,
    )

    # Build DataFrame
    data = {
        "Parameter": param_names[:n_params],
        "Value": [f"{v:.{precision}g}" for v in popt],
        "Std Error": [f"{e:.{precision}g}" if np.isfinite(e) else "N/A" for e in perr],
        "Rel Error (%)": [f"{r:.2f}" if np.isfinite(r) else "N/A" for r in rel_err],
    }

    return pd.DataFrame(data)


# =============================================================================
# Confidence Interval Computation
# =============================================================================


def compute_confidence_intervals(
    popt: np.ndarray,
    pcov: np.ndarray | None,
    n_data: int | None = None,
    alpha: float = 0.05,
) -> list[tuple[float, float]]:
    """Compute confidence intervals for parameters.

    Parameters
    ----------
    popt : np.ndarray
        Optimal parameter values.
    pcov : np.ndarray | None
        Parameter covariance matrix.
    n_data : int | None
        Number of data points. Used to compute degrees of freedom.
        If None, uses normal distribution instead of t-distribution.
    alpha : float
        Significance level (default: 0.05 for 95% CI).

    Returns
    -------
    list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.

    Examples
    --------
    >>> popt = np.array([2.5, 0.1])
    >>> pcov = np.diag([0.01, 0.001])
    >>> ci = compute_confidence_intervals(popt, pcov, n_data=100, alpha=0.05)
    >>> len(ci)
    2
    """
    popt = np.asarray(popt)
    n_params = len(popt)

    # Handle missing covariance
    if pcov is None:
        return [(-np.inf, np.inf) for _ in range(n_params)]

    pcov = np.asarray(pcov)

    # Check for invalid covariance (e.g., all inf)
    if not np.isfinite(pcov).all():
        return [(-np.inf, np.inf) for _ in range(n_params)]

    # Standard errors
    try:
        perr = np.sqrt(np.diag(pcov))
    except (ValueError, FloatingPointError):
        return [(-np.inf, np.inf) for _ in range(n_params)]

    # Compute critical value
    if n_data is not None and n_data > n_params:
        dof = n_data - n_params
        t_val = stats.t.ppf(1 - alpha / 2, dof)
    else:
        # Use normal distribution as fallback
        t_val = stats.norm.ppf(1 - alpha / 2)

    # Compute intervals
    intervals = []
    for i in range(n_params):
        if np.isfinite(perr[i]):
            margin = t_val * perr[i]
            intervals.append((float(popt[i] - margin), float(popt[i] + margin)))
        else:
            intervals.append((-np.inf, np.inf))

    return intervals


def format_confidence_intervals(
    intervals: list[tuple[float, float]],
    param_names: list[str] | None = None,
    alpha: float = 0.05,
    precision: int = 4,
) -> pd.DataFrame:
    """Format confidence intervals as a DataFrame.

    Parameters
    ----------
    intervals : list[tuple[float, float]]
        List of (lower, upper) bounds for each parameter.
    param_names : list[str] | None
        Parameter names. If None, uses p0, p1, etc.
    alpha : float
        Significance level used to compute intervals.
    precision : int
        Number of significant digits.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Parameter, Lower, Upper.
    """
    n_params = len(intervals)

    if param_names is None:
        param_names = [f"p{i}" for i in range(n_params)]
    elif len(param_names) < n_params:
        param_names = list(param_names) + [
            f"p{i}" for i in range(len(param_names), n_params)
        ]

    ci_level = int((1 - alpha) * 100)

    data = {
        "Parameter": param_names[:n_params],
        f"{ci_level}% CI Lower": [
            f"{low:.{precision}g}" if np.isfinite(low) else "-inf"
            for low, _ in intervals
        ],
        f"{ci_level}% CI Upper": [
            f"{high:.{precision}g}" if np.isfinite(high) else "+inf"
            for _, high in intervals
        ],
    }

    return pd.DataFrame(data)


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_parameter_results(
    result: Any,
    param_names: list[str] | None = None,
    show_ci: bool = True,
    alpha: float = 0.05,
    container: Any = None,
) -> None:
    """Render parameter results in Streamlit.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    param_names : list[str] | None
        Parameter names for display.
    show_ci : bool
        Whether to show confidence intervals.
    alpha : float
        Significance level for confidence intervals.
    container : Any, optional
        Streamlit container to render in.

    Examples
    --------
    >>> import streamlit as st
    >>> result = curve_fit(model, x, y)
    >>> render_parameter_results(result, param_names=['a', 'b', 'c'])
    """
    import streamlit as st

    target = container if container is not None else st

    # Extract popt and pcov from result
    popt = getattr(result, "popt", None)
    pcov = getattr(result, "pcov", None)

    if popt is None:
        target.warning("No fitted parameters available")
        return

    # Get n_data for degrees of freedom
    n_data = None
    if hasattr(result, "ydata"):
        with contextlib.suppress(TypeError, AttributeError):
            n_data = len(result.ydata)

    # Format parameter table
    df = format_parameter_table(popt, pcov, param_names)

    target.markdown("**Fitted Parameters**")
    target.dataframe(
        df,
        width="stretch",
        hide_index=True,
    )

    # Display confidence intervals if requested
    if show_ci:
        ci = compute_confidence_intervals(popt, pcov, n_data=n_data, alpha=alpha)
        ci_df = format_confidence_intervals(ci, param_names, alpha=alpha)

        ci_level = int((1 - alpha) * 100)
        target.markdown(f"**{ci_level}% Confidence Intervals**")
        target.dataframe(
            ci_df,
            width="stretch",
            hide_index=True,
        )

    # Display parameter metrics in columns for quick view
    if len(popt) <= 6:  # Only show metrics for reasonable number of params
        target.markdown("**Quick View**")
        cols = target.columns(len(popt))

        names = param_names or [f"p{i}" for i in range(len(popt))]
        perr = np.sqrt(np.diag(pcov)) if pcov is not None else [None] * len(popt)

        for i, (col, name) in enumerate(zip(cols, names, strict=False)):
            with col:
                value = popt[i]
                err = perr[i] if perr[i] is not None and np.isfinite(perr[i]) else None

                if err is not None:
                    st.metric(
                        label=name,
                        value=f"{value:.4g}",
                        delta=f"+/- {err:.2g}",
                        delta_color="off",
                    )
                else:
                    st.metric(
                        label=name,
                        value=f"{value:.4g}",
                    )
