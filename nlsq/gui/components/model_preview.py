"""Model preview component for NLSQ GUI.

This module provides functions for displaying model information including
LaTeX equations, parameter names, and model summaries.

Functions
---------
format_parameter_list
    Format parameter names for display.
get_model_summary
    Get comprehensive summary of a model.
render_model_preview
    Render model preview UI with Streamlit.
render_equation_display
    Render LaTeX equation with Streamlit.
"""

from collections.abc import Callable
from typing import Any

from nlsq.gui.adapters.model_adapter import (
    get_latex_equation,
    get_model_info,
    get_polynomial_latex,
)


def format_parameter_list(param_names: list[str]) -> str:
    """Format parameter names for display.

    Joins parameter names into a comma-separated string suitable
    for UI display.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names.

    Returns
    -------
    str
        Formatted string of parameter names.
        Returns "(none)" if list is empty.

    Examples
    --------
    >>> format_parameter_list(["a", "b", "c"])
    'a, b, c'

    >>> format_parameter_list([])
    '(none)'
    """
    if not param_names:
        return "(none)"

    return ", ".join(param_names)


def get_model_summary(model: Callable) -> dict[str, Any]:
    """Get comprehensive summary of a model.

    Extracts model information including parameter count, names,
    and capability flags.

    Parameters
    ----------
    model : Callable
        A model function (e.g., from get_model).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - param_count: int - Number of model parameters
        - param_names: list[str] - Names of parameters
        - param_display: str - Formatted parameter names
        - has_auto_p0: bool - Whether model has estimate_p0 method
        - has_auto_bounds: bool - Whether model has bounds method
        - equation: str | None - LaTeX equation if available

    Examples
    --------
    >>> from nlsq.gui.adapters.model_adapter import get_model
    >>> model = get_model("builtin", {"name": "linear"})
    >>> summary = get_model_summary(model)
    >>> summary["param_count"]
    2
    """
    info = get_model_info(model)

    return {
        "param_count": info["param_count"],
        "param_names": info["param_names"],
        "param_display": format_parameter_list(info["param_names"]),
        "has_auto_p0": info["has_estimate_p0"],
        "has_auto_bounds": info["has_bounds"],
        "equation": info.get("equation"),
    }


def get_equation_for_model(
    model_type: str,
    model_name: str = "",
    polynomial_degree: int = 0,
) -> str:
    """Get LaTeX equation string for a model.

    Parameters
    ----------
    model_type : str
        Type of model: "builtin", "polynomial", or "custom".
    model_name : str, optional
        Name of built-in model. Required for "builtin" type.
    polynomial_degree : int, optional
        Degree for polynomial models. Required for "polynomial" type.

    Returns
    -------
    str
        LaTeX equation string for the model.

    Examples
    --------
    >>> get_equation_for_model("builtin", "linear")
    'y = ax + b'

    >>> get_equation_for_model("polynomial", polynomial_degree=2)
    'y = c_0 x^2 + c_1 x + c_2'
    """
    if model_type == "builtin":
        return get_latex_equation(model_name)
    elif model_type == "polynomial":
        return get_polynomial_latex(polynomial_degree)
    else:
        # Custom model - generic equation
        return r"y = f(x; \theta)"


def render_model_preview(
    model: Callable | None,
    model_type: str,
    model_name: str = "",
    polynomial_degree: int = 0,
) -> None:
    """Render model preview UI with Streamlit.

    Displays model equation, parameter information, and capability
    indicators in a structured format.

    Parameters
    ----------
    model : Callable | None
        The model function, or None if not yet selected.
    model_type : str
        Type of model: "builtin", "polynomial", or "custom".
    model_name : str, optional
        Name of built-in model.
    polynomial_degree : int, optional
        Degree for polynomial models.

    Note
    ----
    This function requires Streamlit to be running.
    """
    import streamlit as st

    st.subheader("Model Preview")

    if model is None:
        st.info("Select a model to see its preview")
        return

    # Get model summary
    summary = get_model_summary(model)

    # Display equation
    render_equation_display(model_type, model_name, polynomial_degree)

    st.divider()

    # Display parameter info in columns
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Parameter Count", summary["param_count"])

        if summary["has_auto_p0"]:
            st.success("Auto initial guess available")
        else:
            st.info("Manual initial guess required")

    with col2:
        st.markdown("**Parameters:**")
        st.code(summary["param_display"], language=None)

        if summary["has_auto_bounds"]:
            st.success("Auto bounds available")


def render_equation_display(
    model_type: str,
    model_name: str = "",
    polynomial_degree: int = 0,
) -> None:
    """Render LaTeX equation with Streamlit.

    Displays the model equation using st.latex for proper
    mathematical rendering.

    Parameters
    ----------
    model_type : str
        Type of model: "builtin", "polynomial", or "custom".
    model_name : str, optional
        Name of built-in model.
    polynomial_degree : int, optional
        Degree for polynomial models.

    Note
    ----
    This function requires Streamlit to be running.
    """
    import streamlit as st

    equation = get_equation_for_model(model_type, model_name, polynomial_degree)

    st.markdown("**Equation:**")
    st.latex(equation)


def render_parameter_table(param_names: list[str]) -> None:
    """Render a table of parameter names with indices.

    Displays parameters in a tabular format with index numbers
    for easy reference.

    Parameters
    ----------
    param_names : list[str]
        List of parameter names.

    Note
    ----
    This function requires Streamlit to be running.
    """
    import pandas as pd
    import streamlit as st

    if not param_names:
        st.info("No parameters to display")
        return

    # Create DataFrame for display
    df = pd.DataFrame(
        {
            "Index": range(len(param_names)),
            "Parameter": param_names,
        }
    )

    st.dataframe(df, hide_index=True, width="stretch")


def render_model_capabilities(model: Callable) -> None:
    """Render model capability indicators.

    Shows icons/badges for model features like auto p0 estimation
    and auto bounds.

    Parameters
    ----------
    model : Callable
        A model function.

    Note
    ----
    This function requires Streamlit to be running.
    """
    import streamlit as st

    summary = get_model_summary(model)

    col1, col2 = st.columns(2)

    with col1:
        if summary["has_auto_p0"]:
            st.markdown(":white_check_mark: **Auto p0**")
        else:
            st.markdown(":x: **Auto p0**")

    with col2:
        if summary["has_auto_bounds"]:
            st.markdown(":white_check_mark: **Auto Bounds**")
        else:
            st.markdown(":x: **Auto Bounds**")
