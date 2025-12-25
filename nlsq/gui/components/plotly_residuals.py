"""Plotly residuals plot component for NLSQ GUI.

This module provides Plotly-based visualization of residuals
(observed - predicted) with zero line and standard deviation bands.

Functions
---------
create_residuals_plot
    Create a Plotly figure with residuals.
render_residuals_plot
    Render the residuals plot in Streamlit.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go

from nlsq.gui.utils.theme import (
    get_annotation_style,
    get_current_theme,
    get_plotly_colors,
    get_plotly_template,
    get_sigma_band_colors,
)

# =============================================================================
# Plot Creation
# =============================================================================


def create_residuals_plot(
    x: np.ndarray,
    residuals: np.ndarray,
    show_std_bands: bool = True,
    show_zero_line: bool = True,
    title: str = "Residuals Plot",
    x_label: str = "x",
    y_label: str = "Residuals",
    theme: str | None = None,
) -> go.Figure:
    """Create a Plotly figure with residuals.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    residuals : np.ndarray
        Residual values (observed - predicted).
    show_std_bands : bool
        Whether to show +/- 1, 2, 3 standard deviation bands.
    show_zero_line : bool
        Whether to show the zero reference line.
    title : str
        Plot title.
    x_label : str
        Label for x-axis.
    y_label : str
        Label for y-axis.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> residuals = np.random.normal(0, 0.5, 50)
    >>> fig = create_residuals_plot(x, residuals)
    """
    # Get theme colors
    if theme is None:
        theme = get_current_theme()

    colors = get_plotly_colors(theme)
    sigma_colors = get_sigma_band_colors(theme)
    annotation_style = get_annotation_style(theme)

    x = np.asarray(x)
    residuals = np.asarray(residuals)

    # Sort by x for consistent display
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    res_sorted = residuals[sort_idx]

    # Calculate statistics
    std_res = np.std(residuals)
    mean_res = np.mean(residuals)

    # Create figure
    fig = go.Figure()

    # Get x range for bands
    x_min, x_max = x_sorted.min(), x_sorted.max()
    x_range = np.array([x_min, x_max])

    # Add standard deviation bands if requested
    if show_std_bands and std_res > 0:
        # 3 sigma band (outer)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate(
                    [
                        np.full(2, mean_res + 3 * std_res),
                        np.full(2, mean_res - 3 * std_res),
                    ]
                ),
                fill="toself",
                fillcolor=sigma_colors["sigma_3"],
                line={"color": "rgba(255, 0, 0, 0)"},
                name="3 sigma",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # 2 sigma band (middle)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate(
                    [
                        np.full(2, mean_res + 2 * std_res),
                        np.full(2, mean_res - 2 * std_res),
                    ]
                ),
                fill="toself",
                fillcolor=sigma_colors["sigma_2"],
                line={"color": "rgba(255, 165, 0, 0)"},
                name="2 sigma",
                showlegend=True,
                hoverinfo="skip",
            )
        )

        # 1 sigma band (inner)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate(
                    [
                        np.full(2, mean_res + std_res),
                        np.full(2, mean_res - std_res),
                    ]
                ),
                fill="toself",
                fillcolor=sigma_colors["sigma_1"],
                line={"color": "rgba(0, 128, 0, 0)"},
                name="1 sigma",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    # Add zero line if requested
    if show_zero_line:
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=colors["zero_line"],
            line_width=1,
            annotation_text="",
        )

    # Add residual points
    # Color points by distance from zero
    point_colors = np.abs(res_sorted)

    fig.add_trace(
        go.Scatter(
            x=x_sorted,
            y=res_sorted,
            mode="markers",
            name="Residuals",
            marker={
                "color": point_colors,
                "colorscale": "RdYlGn_r",
                "size": 8,
                "opacity": 0.7,
                "showscale": False,
            },
            hovertemplate="x: %{x:.4g}<br>residual: %{y:.4g}<extra></extra>",
            showlegend=True,
        )
    )

    # Configure layout with theme
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
            "zeroline": True,
            "zerolinecolor": colors["zero_line"],
            "zerolinewidth": 1,
        },
        template=get_plotly_template(theme),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
        height=300,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": colors["legend_bg"],
        },
        hovermode="closest",
        font={"color": colors["text"]},
    )

    # Add statistics annotation
    fig.add_annotation(
        x=1.0,
        y=1.0,
        xref="paper",
        yref="paper",
        text=f"Mean: {mean_res:.4g}<br>Std: {std_res:.4g}",
        showarrow=False,
        font={"size": 10, "color": colors["text"]},
        align="right",
        xanchor="right",
        yanchor="top",
        bgcolor=annotation_style["bgcolor"],
        bordercolor=annotation_style["bordercolor"],
        borderwidth=1,
        borderpad=4,
    )

    return fig


def create_residuals_plot_from_result(
    result: Any,
    x_label: str = "x",
    theme: str | None = None,
) -> go.Figure:
    """Create a residuals plot directly from a CurveFitResult.

    Parameters
    ----------
    result : CurveFitResult
        The curve fitting result object.
    x_label : str
        Label for x-axis.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    # Extract data from result
    xdata = np.asarray(result.xdata) if hasattr(result, "xdata") else None
    residuals = np.asarray(result.residuals) if hasattr(result, "residuals") else None

    if xdata is None:
        raise ValueError("Result must contain xdata")
    if residuals is None:
        raise ValueError("Result must contain residuals")

    return create_residuals_plot(
        x=xdata,
        residuals=residuals,
        x_label=x_label,
        theme=theme,
    )


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_residuals_plot(
    x: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    result: Any = None,
    show_std_bands: bool = True,
    show_zero_line: bool = True,
    title: str = "Residuals Plot",
    x_label: str = "x",
    y_label: str = "Residuals",
    container: Any = None,
    key: str = "residuals_plot",
) -> None:
    """Render the residuals plot in Streamlit.

    Can be called either with explicit arrays or with a result object.
    Uses the current theme from session state.

    Parameters
    ----------
    x : np.ndarray | None
        Independent variable values.
    residuals : np.ndarray | None
        Residual values.
    result : CurveFitResult | None
        Alternatively, provide a result object.
    show_std_bands : bool
        Whether to show standard deviation bands.
    show_zero_line : bool
        Whether to show the zero line.
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
    >>> render_residuals_plot(result=result)
    """
    import streamlit as st

    target = container if container is not None else st

    # Get current theme
    theme = get_current_theme()

    # Create figure from result or explicit arrays
    if result is not None:
        try:
            fig = create_residuals_plot_from_result(
                result,
                x_label=x_label,
                theme=theme,
            )
        except (ValueError, AttributeError) as e:
            target.warning(f"Cannot create residuals plot: {e}")
            return
    elif x is not None and residuals is not None:
        fig = create_residuals_plot(
            x=x,
            residuals=residuals,
            show_std_bands=show_std_bands,
            show_zero_line=show_zero_line,
            title=title,
            x_label=x_label,
            y_label=y_label,
            theme=theme,
        )
    else:
        target.warning("No data available for residuals plot")
        return

    # Render plot
    target.plotly_chart(
        fig,
        width="stretch",
        key=key,
    )
