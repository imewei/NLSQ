"""Iteration parameter table component for NLSQ GUI.

This module provides a table component that displays parameter values
at each iteration during curve fitting optimization, allowing users
to observe convergence patterns.

The table is limited to the last N iterations for performance and
displays parameters with appropriate formatting.

Functions
---------
create_iteration_history
    Create an empty iteration history dictionary.
update_iteration_history
    Add a new iteration's parameters to the history.
format_iteration_table
    Format history as a pandas DataFrame for display.
get_table_display_config
    Get configuration for table display.
limit_history_size
    Limit history to last N entries.
render_iteration_table
    Render the iteration table in Streamlit.
"""

from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# Iteration History Management
# =============================================================================


def create_iteration_history(param_names: list[str] | None = None) -> dict[str, Any]:
    """Create an empty iteration history dictionary.

    Parameters
    ----------
    param_names : list[str] | None
        Names of the parameters. Used for display formatting.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'iterations', 'params', and 'param_names'.

    Examples
    --------
    >>> history = create_iteration_history(['a', 'b', 'c'])
    >>> history['param_names']
    ['a', 'b', 'c']
    """
    return {
        "iterations": [],
        "params": [],
        "param_names": param_names or [],
        "costs": [],
    }


def update_iteration_history(
    history: dict[str, Any],
    iteration: int,
    params: np.ndarray,
    cost: float | None = None,
) -> dict[str, Any]:
    """Add a new iteration's parameters to the history.

    Parameters
    ----------
    history : dict[str, Any]
        The iteration history dictionary.
    iteration : int
        The iteration number.
    params : np.ndarray
        Parameter values at this iteration.
    cost : float | None
        Optional cost value at this iteration.

    Returns
    -------
    dict[str, Any]
        Updated history dictionary.

    Examples
    --------
    >>> history = create_iteration_history(['a', 'b'])
    >>> history = update_iteration_history(history, 1, np.array([1.0, 2.0]))
    >>> len(history['iterations'])
    1
    """
    history["iterations"].append(iteration)
    history["params"].append(np.array(params).copy())
    if cost is not None:
        history["costs"].append(cost)
    return history


def clear_iteration_history(history: dict[str, Any]) -> dict[str, Any]:
    """Clear the iteration history while preserving param_names.

    Parameters
    ----------
    history : dict[str, Any]
        The iteration history dictionary.

    Returns
    -------
    dict[str, Any]
        Cleared history dictionary.
    """
    param_names = history.get("param_names", [])
    history["iterations"].clear()
    history["params"].clear()
    history["costs"].clear()
    history["param_names"] = param_names
    return history


def limit_history_size(
    history: dict[str, Any],
    max_entries: int = 50,
) -> dict[str, Any]:
    """Limit history to last N entries.

    Parameters
    ----------
    history : dict[str, Any]
        The iteration history dictionary.
    max_entries : int
        Maximum number of entries to keep.

    Returns
    -------
    dict[str, Any]
        Limited history dictionary.

    Examples
    --------
    >>> history = {'iterations': list(range(100)), 'params': [...], ...}
    >>> limited = limit_history_size(history, max_entries=10)
    >>> len(limited['iterations'])
    10
    """
    if len(history["iterations"]) <= max_entries:
        return history

    # Keep only the last max_entries
    return {
        "iterations": history["iterations"][-max_entries:],
        "params": history["params"][-max_entries:],
        "costs": history["costs"][-max_entries:] if history.get("costs") else [],
        "param_names": history.get("param_names", []),
    }


# =============================================================================
# DataFrame Formatting
# =============================================================================


def format_iteration_table(
    history: dict[str, Any],
    param_names: list[str] | None = None,
    precision: int = 6,
) -> pd.DataFrame:
    """Format history as a pandas DataFrame for display.

    Parameters
    ----------
    history : dict[str, Any]
        The iteration history dictionary.
    param_names : list[str] | None
        Override parameter names. If None, uses history['param_names'].
    precision : int
        Number of decimal places for parameter values.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for display.

    Examples
    --------
    >>> history = {
    ...     'iterations': [1, 2, 3],
    ...     'params': [np.array([1.0, 2.0]), np.array([1.5, 2.5]), np.array([2.0, 5.0])],
    ...     'param_names': ['a', 'b'],
    ...     'costs': [10.0, 5.0, 1.0],
    ... }
    >>> df = format_iteration_table(history)
    >>> list(df.columns)
    ['Iteration', 'a', 'b', 'Cost']
    """
    if len(history["iterations"]) == 0:
        return pd.DataFrame()

    # Get parameter names
    names = param_names or history.get("param_names", [])
    if not names:
        # Generate default names
        n_params = len(history["params"][0]) if history["params"] else 0
        names = [f"p{i}" for i in range(n_params)]

    # Build data dictionary
    data: dict[str, list] = {"Iteration": history["iterations"]}

    # Add parameter columns
    for i, name in enumerate(names):
        data[name] = [
            round(p[i], precision) if i < len(p) else None for p in history["params"]
        ]

    # Add cost column if available
    if history.get("costs") and len(history["costs"]) == len(history["iterations"]):
        data["Cost"] = [f"{c:.6e}" for c in history["costs"]]

    return pd.DataFrame(data)


def format_parameter_change(
    history: dict[str, Any],
    param_index: int,
) -> dict[str, Any]:
    """Compute change statistics for a parameter.

    Parameters
    ----------
    history : dict[str, Any]
        The iteration history dictionary.
    param_index : int
        Index of the parameter to analyze.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'initial', 'final', 'change', 'percent_change'.
    """
    if len(history["params"]) < 2:
        return {
            "initial": float("nan"),
            "final": float("nan"),
            "change": float("nan"),
            "percent_change": float("nan"),
        }

    initial = history["params"][0][param_index]
    final = history["params"][-1][param_index]
    change = final - initial
    percent = (change / initial * 100) if initial != 0 else float("inf")

    return {
        "initial": initial,
        "final": final,
        "change": change,
        "percent_change": percent,
    }


# =============================================================================
# Configuration
# =============================================================================


def get_table_display_config() -> dict[str, Any]:
    """Get configuration for table display.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with display settings.
    """
    return {
        "max_rows": 50,
        "precision": 6,
        "show_cost": True,
        "reverse_order": True,  # Show most recent first
    }


# =============================================================================
# Streamlit Rendering
# =============================================================================


def render_iteration_table(
    history: dict[str, Any],
    param_names: list[str] | None = None,
    container: Any = None,
    key: str = "iteration_table",
    max_rows: int = 20,
) -> None:
    """Render the iteration table in Streamlit.

    Parameters
    ----------
    history : dict[str, Any]
        Iteration history dictionary.
    param_names : list[str] | None
        Parameter names for column headers.
    container : Any, optional
        Streamlit container to render in.
    key : str
        Unique key for the component.
    max_rows : int
        Maximum rows to display.

    Examples
    --------
    >>> import streamlit as st
    >>> history = create_iteration_history(['a', 'b'])
    >>> # After fitting:
    >>> render_iteration_table(history)
    """
    import streamlit as st

    target = container if container is not None else st

    if len(history.get("iterations", [])) == 0:
        target.caption("Parameter values will appear during fitting")
        return

    # Limit and format history
    limited = limit_history_size(history, max_entries=max_rows)
    df = format_iteration_table(limited, param_names)

    if df.empty:
        target.caption("No iteration data available")
        return

    # Reverse to show most recent first
    df = df.iloc[::-1]

    # Display table
    target.dataframe(
        df,
        width="stretch",
        hide_index=True,
        key=key,
    )

    # Show convergence indicator
    if len(history["params"]) >= 2:
        last_params = history["params"][-1]
        prev_params = history["params"][-2]
        max_change = np.max(np.abs(last_params - prev_params))

        if max_change < 1e-6:
            target.success("Parameters converging (change < 1e-6)")
        elif max_change < 1e-4:
            target.info("Parameters stabilizing")


def render_convergence_summary(
    history: dict[str, Any],
    param_names: list[str] | None = None,
) -> None:
    """Render a convergence summary panel.

    Parameters
    ----------
    history : dict[str, Any]
        Iteration history dictionary.
    param_names : list[str] | None
        Parameter names for display.
    """
    import streamlit as st

    if len(history.get("params", [])) < 2:
        return

    names = param_names or history.get("param_names", [])
    if not names:
        n_params = len(history["params"][0])
        names = [f"p{i}" for i in range(n_params)]

    st.markdown("**Parameter Changes**")

    cols = st.columns(len(names))
    for i, (col, name) in enumerate(zip(cols, names, strict=False)):
        stats = format_parameter_change(history, i)
        with col:
            st.metric(
                label=name,
                value=f"{stats['final']:.4g}",
                delta=f"{stats['change']:+.4g}",
            )
