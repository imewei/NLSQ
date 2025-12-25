"""Fit statistics display component for NLSQ GUI.

This module provides components for displaying fit quality statistics
and convergence information in the Streamlit GUI.

Functions
---------
format_statistics
    Format R^2, RMSE, MAE, AIC, BIC statistics.
format_convergence_info
    Format success, nfev, cost information.
render_fit_statistics
    Render fit statistics in Streamlit.
"""

from typing import Any

import numpy as np

# =============================================================================
# Statistics Formatting
# =============================================================================


def format_statistics(result: Any) -> dict[str, str]:
    """Format fit quality statistics as display strings.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object with statistical properties.

    Returns
    -------
    dict[str, str]
        Dictionary with formatted strings for:
        - r_squared: R^2 value
        - adj_r_squared: Adjusted R^2
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion

    Examples
    --------
    >>> result = curve_fit(model, x, y)
    >>> stats = format_statistics(result)
    >>> stats['r_squared']
    '0.9876'
    """
    stats: dict[str, str] = {}

    # R-squared
    try:
        r2 = float(result.r_squared)
        stats["r_squared"] = f"{r2:.6f}" if np.isfinite(r2) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["r_squared"] = "N/A"

    # Adjusted R-squared
    try:
        adj_r2 = float(result.adj_r_squared)
        stats["adj_r_squared"] = f"{adj_r2:.6f}" if np.isfinite(adj_r2) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["adj_r_squared"] = "N/A"

    # RMSE
    try:
        rmse = float(result.rmse)
        stats["rmse"] = f"{rmse:.6g}" if np.isfinite(rmse) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["rmse"] = "N/A"

    # MAE
    try:
        mae = float(result.mae)
        stats["mae"] = f"{mae:.6g}" if np.isfinite(mae) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["mae"] = "N/A"

    # AIC
    try:
        aic = float(result.aic)
        stats["aic"] = f"{aic:.2f}" if np.isfinite(aic) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["aic"] = "N/A"

    # BIC
    try:
        bic = float(result.bic)
        stats["bic"] = f"{bic:.2f}" if np.isfinite(bic) else "N/A"
    except (AttributeError, TypeError, ValueError):
        stats["bic"] = "N/A"

    return stats


def format_convergence_info(result: Any) -> dict[str, Any]:
    """Format convergence information as display values.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - success: bool
        - success_str: "Yes" or "No"
        - message: str
        - nfev: int or "N/A"
        - nfev_str: formatted string
        - cost: float or "N/A"
        - cost_str: formatted string
        - optimality: float or "N/A"
        - optimality_str: formatted string

    Examples
    --------
    >>> result = curve_fit(model, x, y)
    >>> info = format_convergence_info(result)
    >>> info['success']
    True
    """
    info: dict[str, Any] = {}

    # Success
    try:
        success = getattr(result, "success", False)
        if hasattr(success, "item"):
            success = success.item()
        info["success"] = bool(success)
        info["success_str"] = "Yes" if success else "No"
    except (AttributeError, TypeError):
        info["success"] = False
        info["success_str"] = "Unknown"

    # Message
    try:
        message = getattr(result, "message", "")
        if message is None:
            message = ""
        info["message"] = str(message)
    except (AttributeError, TypeError):
        info["message"] = ""

    # Number of function evaluations
    try:
        nfev = getattr(result, "nfev", None)
        if nfev is not None:
            if hasattr(nfev, "item"):
                nfev = nfev.item()
            info["nfev"] = int(nfev)
            info["nfev_str"] = f"{int(nfev):,}"
        else:
            info["nfev"] = None
            info["nfev_str"] = "N/A"
    except (AttributeError, TypeError, ValueError):
        info["nfev"] = None
        info["nfev_str"] = "N/A"

    # Final cost
    try:
        cost = getattr(result, "cost", None)
        if cost is not None:
            if hasattr(cost, "item"):
                cost = cost.item()
            cost = float(cost)
            info["cost"] = cost
            info["cost_str"] = f"{cost:.6e}" if np.isfinite(cost) else "N/A"
        else:
            info["cost"] = None
            info["cost_str"] = "N/A"
    except (AttributeError, TypeError, ValueError):
        info["cost"] = None
        info["cost_str"] = "N/A"

    # Optimality
    try:
        opt = getattr(result, "optimality", None)
        if opt is not None:
            if hasattr(opt, "item"):
                opt = opt.item()
            opt = float(opt)
            info["optimality"] = opt
            info["optimality_str"] = f"{opt:.6e}" if np.isfinite(opt) else "N/A"
        else:
            info["optimality"] = None
            info["optimality_str"] = "N/A"
    except (AttributeError, TypeError, ValueError):
        info["optimality"] = None
        info["optimality_str"] = "N/A"

    return info


def get_fit_quality_label(r_squared: float) -> tuple[str, str]:
    """Get a quality label based on R-squared value.

    Parameters
    ----------
    r_squared : float
        The R-squared value.

    Returns
    -------
    tuple[str, str]
        (label, color) where color is for Streamlit display.

    Examples
    --------
    >>> label, color = get_fit_quality_label(0.98)
    >>> label
    'Excellent'
    """
    if not np.isfinite(r_squared):
        return "Unknown", "gray"
    elif r_squared >= 0.99:
        return "Excellent", "green"
    elif r_squared >= 0.95:
        return "Very Good", "green"
    elif r_squared >= 0.90:
        return "Good", "blue"
    elif r_squared >= 0.80:
        return "Moderate", "orange"
    elif r_squared >= 0.50:
        return "Weak", "orange"
    else:
        return "Poor", "red"


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_fit_statistics(
    result: Any,
    container: Any = None,
    show_convergence: bool = True,
) -> None:
    """Render fit statistics in Streamlit.

    Displays fit quality metrics (R^2, RMSE, MAE, AIC, BIC) and optionally
    convergence information (success, nfev, cost) using Streamlit metrics.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    container : Any, optional
        Streamlit container to render in.
    show_convergence : bool
        Whether to show convergence information.

    Examples
    --------
    >>> import streamlit as st
    >>> result = curve_fit(model, x, y)
    >>> render_fit_statistics(result)
    """
    import streamlit as st

    target = container if container is not None else st

    # Get formatted statistics
    stats = format_statistics(result)
    info = format_convergence_info(result)

    # Fit Quality Section
    target.markdown("**Fit Quality**")

    # Get R-squared value for quality assessment
    try:
        r2_val = float(result.r_squared)
        quality_label, quality_color = get_fit_quality_label(r2_val)
    except (AttributeError, TypeError, ValueError):
        quality_label, quality_color = "Unknown", "gray"

    # Display primary metrics in columns
    col1, col2, col3 = target.columns(3)

    with col1:
        st.metric(
            label="R-squared",
            value=stats["r_squared"],
            help="Coefficient of determination. 1.0 is perfect fit.",
        )

    with col2:
        st.metric(
            label="Adj. R-squared",
            value=stats["adj_r_squared"],
            help="R-squared adjusted for number of parameters.",
        )

    with col3:
        st.metric(
            label="RMSE",
            value=stats["rmse"],
            help="Root mean squared error. Lower is better.",
        )

    # Display secondary metrics
    col4, col5, col6 = target.columns(3)

    with col4:
        st.metric(
            label="MAE",
            value=stats["mae"],
            help="Mean absolute error. Robust to outliers.",
        )

    with col5:
        st.metric(
            label="AIC",
            value=stats["aic"],
            help="Akaike Information Criterion. Lower is better.",
        )

    with col6:
        st.metric(
            label="BIC",
            value=stats["bic"],
            help="Bayesian Information Criterion. Lower is better.",
        )

    # Quality assessment badge
    if quality_color == "green":
        target.success(f"Fit Quality: {quality_label}")
    elif quality_color == "blue":
        target.info(f"Fit Quality: {quality_label}")
    elif quality_color == "orange":
        target.warning(f"Fit Quality: {quality_label}")
    elif quality_color == "red":
        target.error(f"Fit Quality: {quality_label}")
    else:
        target.caption(f"Fit Quality: {quality_label}")

    # Convergence Information Section
    if show_convergence:
        target.divider()
        target.markdown("**Convergence Information**")

        col1, col2, col3 = target.columns(3)

        with col1:
            if info["success"]:
                st.success(f"Converged: {info['success_str']}")
            else:
                st.warning(f"Converged: {info['success_str']}")

        with col2:
            st.metric(
                label="Function Evaluations",
                value=info["nfev_str"],
                help="Number of model evaluations during optimization.",
            )

        with col3:
            st.metric(
                label="Final Cost",
                value=info["cost_str"],
                help="Final value of the objective function.",
            )

        # Display message if available
        if info["message"]:
            target.caption(f"Message: {info['message']}")

        # Display optimality if available
        if info["optimality"] is not None and np.isfinite(info["optimality"]):
            target.caption(f"Optimality: {info['optimality_str']}")


def render_statistics_summary(
    result: Any,
    container: Any = None,
) -> None:
    """Render a compact statistics summary.

    A more compact version for use in sidebars or summaries.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    container : Any, optional
        Streamlit container to render in.
    """
    import streamlit as st

    target = container if container is not None else st

    stats = format_statistics(result)
    info = format_convergence_info(result)

    # Compact display
    target.markdown(f"**R^2:** {stats['r_squared']}")
    target.markdown(f"**RMSE:** {stats['rmse']}")

    if info["success"]:
        target.success("Converged")
    else:
        target.warning("Did not converge")
