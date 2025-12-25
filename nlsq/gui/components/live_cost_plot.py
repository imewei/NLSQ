"""Live cost function plot component for NLSQ GUI.

This module provides a Plotly-based live updating chart that displays
the cost (objective function) value versus iteration number during
curve fitting optimization.

The plot updates incrementally during optimization to show the
convergence trajectory in real-time.

Functions
---------
create_cost_history
    Create an empty cost history dictionary.
update_cost_history
    Add a new iteration/cost pair to the history.
create_cost_plot_figure
    Create a Plotly figure from cost history.
get_plot_config
    Get configuration for Plotly chart display.
render_live_cost_plot
    Render the live cost plot in Streamlit.
"""

from typing import Any

import plotly.graph_objects as go

from nlsq.gui.utils.theme import (
    get_annotation_style,
    get_current_theme,
    get_plotly_colors,
    get_plotly_template,
)

# =============================================================================
# Cost History Management
# =============================================================================


def create_cost_history() -> dict[str, list]:
    """Create an empty cost history dictionary.

    Returns
    -------
    dict[str, list]
        Dictionary with 'iterations' and 'costs' lists.

    Examples
    --------
    >>> history = create_cost_history()
    >>> history['iterations']
    []
    >>> history['costs']
    []
    """
    return {
        "iterations": [],
        "costs": [],
    }


def update_cost_history(
    history: dict[str, list],
    iteration: int,
    cost: float,
) -> dict[str, list]:
    """Add a new iteration/cost pair to the history.

    Parameters
    ----------
    history : dict[str, list]
        The cost history dictionary.
    iteration : int
        The iteration number.
    cost : float
        The cost (objective) value.

    Returns
    -------
    dict[str, list]
        Updated history dictionary.

    Examples
    --------
    >>> history = create_cost_history()
    >>> history = update_cost_history(history, 1, 10.5)
    >>> history = update_cost_history(history, 2, 5.2)
    >>> history['iterations']
    [1, 2]
    """
    history["iterations"].append(iteration)
    history["costs"].append(cost)
    return history


def clear_cost_history(history: dict[str, list]) -> dict[str, list]:
    """Clear the cost history.

    Parameters
    ----------
    history : dict[str, list]
        The cost history dictionary.

    Returns
    -------
    dict[str, list]
        Empty history dictionary.
    """
    history["iterations"].clear()
    history["costs"].clear()
    return history


# =============================================================================
# Plotly Figure Creation
# =============================================================================


def create_cost_plot_figure(
    history: dict[str, list],
    title: str = "Cost vs Iteration",
    log_scale: bool = True,
    theme: str | None = None,
) -> go.Figure:
    """Create a Plotly figure from cost history.

    Parameters
    ----------
    history : dict[str, list]
        Cost history with 'iterations' and 'costs' lists.
    title : str
        Plot title.
    log_scale : bool
        Whether to use log scale for y-axis.
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Plotly figure object.

    Examples
    --------
    >>> history = {'iterations': [1, 2, 3], 'costs': [10, 5, 2]}
    >>> fig = create_cost_plot_figure(history)
    """
    # Get theme colors
    if theme is None:
        theme = get_current_theme()

    colors = get_plotly_colors(theme)
    annotation_style = get_annotation_style(theme)

    iterations = history.get("iterations", [])
    costs = history.get("costs", [])

    # Create figure
    fig = go.Figure()

    # Add cost trace
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=costs,
            mode="lines+markers",
            name="Cost",
            line={"color": colors["cost_line"], "width": 2},
            marker={"size": 4},
            hovertemplate="Iteration: %{x}<br>Cost: %{y:.6e}<extra></extra>",
        )
    )

    # Configure layout with theme
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16},
        },
        xaxis={
            "title": "Iteration",
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        yaxis={
            "title": "Cost",
            "type": "log"
            if log_scale and len(costs) > 0 and min(costs) > 0
            else "linear",
            "showgrid": True,
            "gridcolor": colors["grid"],
        },
        template=get_plotly_template(theme),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        height=300,
        showlegend=False,
        font={"color": colors["text"]},
    )

    # Add annotation for current cost if data exists
    if len(costs) > 0:
        current_cost = costs[-1]
        fig.add_annotation(
            x=1.0,
            y=1.0,
            xref="paper",
            yref="paper",
            text=f"Current: {current_cost:.6e}",
            showarrow=False,
            font={"size": 12, "color": colors["text"]},
            align="right",
            xanchor="right",
            yanchor="top",
            bgcolor=annotation_style["bgcolor"],
            bordercolor=annotation_style["bordercolor"],
            borderwidth=1,
            borderpad=4,
        )

    return fig


def create_empty_cost_plot(theme: str | None = None) -> go.Figure:
    """Create an empty cost plot placeholder.

    Parameters
    ----------
    theme : str | None
        Theme to use ('light' or 'dark'). If None, uses current theme.

    Returns
    -------
    go.Figure
        Empty Plotly figure with placeholder text.
    """
    # Get theme colors
    if theme is None:
        theme = get_current_theme()

    colors = get_plotly_colors(theme)

    fig = go.Figure()

    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Cost plot will appear during fitting",
        showarrow=False,
        font={"size": 14, "color": colors["text"]},
    )

    fig.update_layout(
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        template=get_plotly_template(theme),
        paper_bgcolor=colors["background"],
        plot_bgcolor=colors["background"],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=300,
    )

    return fig


# =============================================================================
# Configuration
# =============================================================================


def get_plot_config() -> dict[str, Any]:
    """Get configuration for Plotly chart display.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary for st.plotly_chart.

    Notes
    -----
    The configuration disables the mode bar for cleaner display
    during live updates.
    """
    return {
        "displayModeBar": False,
        "staticPlot": False,
        "scrollZoom": False,
    }


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_live_cost_plot(
    history: dict[str, list],
    container: Any = None,
    key: str = "live_cost_plot",
) -> None:
    """Render the live cost plot in Streamlit.

    Uses the current theme from session state.

    Parameters
    ----------
    history : dict[str, list]
        Cost history dictionary.
    container : Any, optional
        Streamlit container to render in. If None, renders in main area.
    key : str
        Unique key for the Streamlit component.

    Notes
    -----
    This function should be called with a placeholder container for
    live updates during fitting.

    Examples
    --------
    >>> import streamlit as st
    >>> placeholder = st.empty()
    >>> history = create_cost_history()
    >>> # During fitting loop:
    >>> history = update_cost_history(history, iteration, cost)
    >>> with placeholder:
    ...     render_live_cost_plot(history)
    """
    import streamlit as st

    target = container if container is not None else st

    # Get current theme
    theme = get_current_theme()

    if len(history.get("iterations", [])) == 0:
        fig = create_empty_cost_plot(theme=theme)
    else:
        fig = create_cost_plot_figure(history, theme=theme)

    target.plotly_chart(
        fig,
        width="stretch",
        config=get_plot_config(),
        key=key,
    )


def get_cost_plot_summary(history: dict[str, list]) -> dict[str, float]:
    """Get summary statistics from cost history.

    Parameters
    ----------
    history : dict[str, list]
        Cost history dictionary.

    Returns
    -------
    dict[str, float]
        Summary with initial_cost, final_cost, reduction, n_iterations.
    """
    costs = history.get("costs", [])
    iterations = history.get("iterations", [])

    if len(costs) == 0:
        return {
            "initial_cost": float("nan"),
            "final_cost": float("nan"),
            "reduction": float("nan"),
            "n_iterations": 0,
        }

    initial = costs[0]
    final = costs[-1]
    reduction = (initial - final) / initial if initial != 0 else 0

    return {
        "initial_cost": initial,
        "final_cost": final,
        "reduction": reduction,
        "n_iterations": len(iterations),
    }
