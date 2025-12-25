"""Plotly fit plot component for NLSQ GUI.

This module provides Plotly-based visualization of curve fitting results,
including data points, fitted curve, and optional confidence bands.

Functions
---------
create_fit_plot
    Create a Plotly figure with data and fitted curve.
render_fit_plot
    Render the fit plot in Streamlit.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from nlsq.gui.utils.theme import (
    get_annotation_style,
    get_current_theme,
    get_data_marker_color,
    get_fit_line_color,
    get_plotly_colors,
    get_plotly_layout_updates,
    get_plotly_template,
)

# =============================================================================
# Plot Creation
# =============================================================================


def create_fit_plot(
    xdata: np.ndarray,
    ydata: np.ndarray,
    xfit: np.ndarray | None = None,
    yfit: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    confidence_band: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Curve Fit Results",
    x_label: str = "x",
    y_label: str = "y",
    show_legend: bool = True,
    theme: str | None = None,
) -> go.Figure:
    """Create a Plotly figure with data and fitted curve.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data points.
    ydata : np.ndarray
        Dependent variable data points (observations).
    xfit : np.ndarray | None
        x values for the fitted curve. If None, uses sorted xdata.
    yfit : np.ndarray | None
        y values for the fitted curve (predictions).
    sigma : np.ndarray | None
        Uncertainties in ydata (for error bars).
    confidence_band : tuple[np.ndarray, np.ndarray] | None
        Lower and upper bounds for confidence band as (lower, upper).
    title : str
        Plot title.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    show_legend : bool
        Whether to show the legend.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = 2 * x + 3 + np.random.normal(0, 0.5, 50)
    >>> y_fit = 2 * x + 3
    >>> fig = create_fit_plot(x, y, xfit=x, yfit=y_fit)
    """
    # Get theme colors
    if theme is None:
        theme = get_current_theme()

    colors = get_plotly_colors(theme)
    data_color = get_data_marker_color(theme)
    fit_color = get_fit_line_color(theme)
    annotation_style = get_annotation_style(theme)

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Sort data for consistent display
    sort_idx = np.argsort(xdata)
    x_sorted = xdata[sort_idx]
    y_sorted = ydata[sort_idx]

    # Create figure
    fig = go.Figure()

    # Add confidence band if provided
    if confidence_band is not None and xfit is not None:
        lower, upper = confidence_band
        xfit_arr = np.asarray(xfit)
        sort_fit_idx = np.argsort(xfit_arr)
        x_band = xfit_arr[sort_fit_idx]
        lower_band = np.asarray(lower)[sort_fit_idx]
        upper_band = np.asarray(upper)[sort_fit_idx]

        # Create filled region for confidence band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_band, x_band[::-1]]),
                y=np.concatenate([upper_band, lower_band[::-1]]),
                fill="toself",
                fillcolor=colors["confidence"],
                line={"color": "rgba(255, 255, 255, 0)"},
                name="95% CI",
                showlegend=show_legend,
                hoverinfo="skip",
            )
        )

    # Add data points with error bars
    if sigma is not None:
        sigma = np.asarray(sigma)
        sigma_sorted = sigma[sort_idx]
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_sorted,
                mode="markers",
                name="Data",
                marker={
                    "color": data_color,
                    "size": 8,
                    "opacity": 0.7,
                },
                error_y={
                    "type": "data",
                    "array": sigma_sorted,
                    "visible": True,
                    "color": colors["grid"],
                    "thickness": 1,
                    "width": 3,
                },
                hovertemplate="x: %{x:.4g}<br>y: %{y:.4g}<extra></extra>",
                showlegend=show_legend,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_sorted,
                y=y_sorted,
                mode="markers",
                name="Data",
                marker={
                    "color": data_color,
                    "size": 8,
                    "opacity": 0.7,
                },
                hovertemplate="x: %{x:.4g}<br>y: %{y:.4g}<extra></extra>",
                showlegend=show_legend,
            )
        )

    # Add fitted curve if provided
    if xfit is not None and yfit is not None:
        xfit_arr = np.asarray(xfit)
        yfit_arr = np.asarray(yfit)

        # Sort for smooth line
        sort_fit_idx = np.argsort(xfit_arr)
        x_fit_sorted = xfit_arr[sort_fit_idx]
        y_fit_sorted = yfit_arr[sort_fit_idx]

        fig.add_trace(
            go.Scatter(
                x=x_fit_sorted,
                y=y_fit_sorted,
                mode="lines",
                name="Fit",
                line={"color": fit_color, "width": 2},
                hovertemplate="x: %{x:.4g}<br>y_fit: %{y:.4g}<extra></extra>",
                showlegend=show_legend,
            )
        )

    # Configure layout with theme
    layout_updates = get_plotly_layout_updates(theme)

    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16},
        },
        xaxis={
            "title": x_label,
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        yaxis={
            "title": y_label,
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        template=get_plotly_template(theme),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        height=400,
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
            "bgcolor": colors["legend_bg"],
        },
        hovermode="closest",
        font={"color": colors["text"]},
    )

    return fig


def create_fit_plot_from_result(
    result: Any,
    x_label: str = "x",
    y_label: str = "y",
    show_confidence: bool = True,
    n_fit_points: int = 200,
    theme: str | None = None,
) -> go.Figure:
    """Create a fit plot directly from a CurveFitResult.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    show_confidence : bool
        Whether to show confidence band.
    n_fit_points : int
        Number of points for the fitted curve.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Extract data from result
    xdata = np.asarray(result.xdata) if hasattr(result, "xdata") else None
    ydata = np.asarray(result.ydata) if hasattr(result, "ydata") else None

    if xdata is None or ydata is None:
        raise ValueError("Result must contain xdata and ydata")

    # Generate smooth fit curve
    x_min, x_max = xdata.min(), xdata.max()
    xfit = np.linspace(x_min, x_max, n_fit_points)

    # Compute predictions
    if hasattr(result, "model") and hasattr(result, "popt"):
        yfit = result.model(xfit, *result.popt)
    elif hasattr(result, "predictions"):
        # Use existing predictions
        xfit = xdata
        yfit = result.predictions
    else:
        yfit = None

    # Get confidence band if available and requested
    confidence_band = None
    if show_confidence and hasattr(result, "prediction_interval"):
        try:
            pi = result.prediction_interval(xfit, alpha=0.95)
            confidence_band = (pi[:, 0], pi[:, 1])
        except (AttributeError, ValueError, TypeError):
            pass

    # Get R-squared for title
    try:
        r2 = float(result.r_squared)
        title = f"Curve Fit (R^2 = {r2:.4f})"
    except (AttributeError, TypeError, ValueError):
        title = "Curve Fit Results"

    return create_fit_plot(
        xdata=xdata,
        ydata=ydata,
        xfit=xfit,
        yfit=yfit,
        sigma=None,  # Could extract from result if stored
        confidence_band=confidence_band,
        title=title,
        x_label=x_label,
        y_label=y_label,
        theme=theme,
    )


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_fit_plot(
    xdata: np.ndarray | None = None,
    ydata: np.ndarray | None = None,
    xfit: np.ndarray | None = None,
    yfit: np.ndarray | None = None,
    result: Any = None,
    sigma: np.ndarray | None = None,
    confidence_band: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Curve Fit Results",
    x_label: str = "x",
    y_label: str = "y",
    container: Any = None,
    key: str = "fit_plot",
) -> None:
    """Render the fit plot in Streamlit.

    Can be called either with explicit arrays or with a result object.
    Uses the current theme from session state.

    Parameters
    ----------
    xdata : np.ndarray | None
        Independent variable data points.
    ydata : np.ndarray | None
        Dependent variable data points.
    xfit : np.ndarray | None
        x values for fitted curve.
    yfit : np.ndarray | None
        y values for fitted curve.
    result : CurveFitResult | None
        Alternatively, provide a result object.
    sigma : np.ndarray | None
        Uncertainties in ydata.
    confidence_band : tuple[np.ndarray, np.ndarray] | None
        Confidence band bounds.
    title : str
        Plot title.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    container : Any, optional
        Streamlit container to render in.
    key : str
        Unique key for the Streamlit component.

    Examples
    --------
    >>> import streamlit as st
    >>> result = curve_fit(model, x, y)
    >>> render_fit_plot(result=result)
    """
    import streamlit as st

    target = container if container is not None else st

    # Get current theme
    theme = get_current_theme()

    # Create figure from result or explicit arrays
    if result is not None:
        try:
            fig = create_fit_plot_from_result(
                result,
                x_label=x_label,
                y_label=y_label,
                theme=theme,
            )
        except (ValueError, AttributeError) as e:
            target.warning(f"Cannot create fit plot: {e}")
            return
    elif xdata is not None and ydata is not None:
        fig = create_fit_plot(
            xdata=xdata,
            ydata=ydata,
            xfit=xfit,
            yfit=yfit,
            sigma=sigma,
            confidence_band=confidence_band,
            title=title,
            x_label=x_label,
            y_label=y_label,
            theme=theme,
        )
    else:
        target.warning("No data available for fit plot")
        return

    # Render plot
    target.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )
