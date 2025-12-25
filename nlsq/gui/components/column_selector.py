"""Column selector component for the NLSQ GUI data loading page.

This module provides functions for column selection UI including color-coded
column assignment, role display names, validation, and data preview statistics.

The column selector allows users to assign roles (x, y, z, sigma) to data columns
with visual feedback through distinct colors and validation messages.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Column Role Colors
# =============================================================================

# Color scheme for column roles (colorblind-friendly)
ROLE_COLORS: dict[str, str] = {
    "x": "#1f77b4",  # Blue - X axis / independent variable
    "y": "#2ca02c",  # Green - Y axis / dependent variable (1D)
    "z": "#ff7f0e",  # Orange - Z axis / dependent variable (2D)
    "sigma": "#9467bd",  # Purple - uncertainty/error
}

ROLE_DISPLAY_NAMES: dict[str, str] = {
    "x": "X (Independent)",
    "y": "Y (Dependent)",
    "z": "Z (Dependent - 2D)",
    "sigma": "Sigma (Uncertainty)",
    "unassigned": "Unassigned",
}


def get_column_color(role: str) -> str | None:
    """Get the color for a column role.

    Parameters
    ----------
    role : str
        The role name: "x", "y", "z", "sigma", or "unassigned".

    Returns
    -------
    str or None
        Hex color code for the role, or None if unassigned/unknown.

    Examples
    --------
    >>> get_column_color("x")
    '#1f77b4'

    >>> get_column_color("unassigned")
    None
    """
    return ROLE_COLORS.get(role)


def get_role_display_name(role: str) -> str:
    """Get the display name for a column role.

    Parameters
    ----------
    role : str
        The role name: "x", "y", "z", "sigma", or "unassigned".

    Returns
    -------
    str
        Human-readable display name for the role.

    Examples
    --------
    >>> get_role_display_name("x")
    'X (Independent)'

    >>> get_role_display_name("sigma")
    'Sigma (Uncertainty)'
    """
    return ROLE_DISPLAY_NAMES.get(role, role.capitalize())


# =============================================================================
# Column Assignment Management
# =============================================================================


def create_column_assignments(
    columns: dict[str, int | None],
    mode: str = "1d",
) -> dict[str, int | None]:
    """Create a column assignments dictionary with defaults.

    Parameters
    ----------
    columns : dict
        Dictionary with column assignments. Keys are role names ("x", "y", "z", "sigma")
        and values are column indices or None.
    mode : str
        Data mode: "1d" or "2d". In 2D mode, z column is available.

    Returns
    -------
    dict
        Column assignments dictionary with all expected keys populated.

    Examples
    --------
    >>> create_column_assignments({"x": 0, "y": 1}, mode="1d")
    {'x': 0, 'y': 1, 'z': None, 'sigma': None}
    """
    # Start with defaults
    assignments: dict[str, int | None] = {
        "x": columns.get("x", 0),
        "y": columns.get("y", 1),
        "z": None,
        "sigma": columns.get("sigma"),
    }

    # In 2D mode, include z column
    if mode == "2d":
        assignments["z"] = columns.get("z")

    return assignments


def get_available_roles(mode: str = "1d") -> list[str]:
    """Get the list of available column roles for a given mode.

    Parameters
    ----------
    mode : str
        Data mode: "1d" or "2d".

    Returns
    -------
    list[str]
        List of available role names.

    Examples
    --------
    >>> get_available_roles("1d")
    ['x', 'y', 'sigma']

    >>> get_available_roles("2d")
    ['x', 'y', 'z', 'sigma']
    """
    if mode == "2d":
        return ["x", "y", "z", "sigma"]
    return ["x", "y", "sigma"]


def get_required_roles(mode: str = "1d") -> list[str]:
    """Get the list of required column roles for a given mode.

    Parameters
    ----------
    mode : str
        Data mode: "1d" or "2d".

    Returns
    -------
    list[str]
        List of required role names.

    Examples
    --------
    >>> get_required_roles("1d")
    ['x', 'y']

    >>> get_required_roles("2d")
    ['x', 'y', 'z']
    """
    if mode == "2d":
        return ["x", "y", "z"]
    return ["x", "y"]


# =============================================================================
# Data Preview and Statistics
# =============================================================================


def compute_data_preview_stats(data: NDArray[np.float64]) -> dict[str, Any]:
    """Compute statistics for data preview display.

    Parameters
    ----------
    data : ndarray
        2D numpy array with shape (n_rows, n_columns).

    Returns
    -------
    dict
        Dictionary with preview statistics:
        - "num_rows": int - Number of data rows
        - "num_columns": int - Number of columns
        - "nan_count": int - Total NaN values
        - "inf_count": int - Total Inf values
        - "column_stats": list[dict] - Per-column statistics (min, max, mean, std)

    Examples
    --------
    >>> data = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> stats = compute_data_preview_stats(data)
    >>> stats["num_rows"]
    2
    >>> stats["num_columns"]
    2
    """
    if data.size == 0:
        return {
            "num_rows": 0 if data.ndim < 2 else data.shape[0],
            "num_columns": data.shape[1] if data.ndim == 2 else 0,
            "nan_count": 0,
            "inf_count": 0,
            "column_stats": [],
        }

    num_rows = data.shape[0]
    num_columns = data.shape[1] if data.ndim == 2 else 1

    # Count non-finite values
    nan_count = int(np.sum(np.isnan(data)))
    inf_count = int(np.sum(np.isinf(data)))

    # Compute per-column statistics
    column_stats = []
    for col_idx in range(num_columns):
        col_data = data[:, col_idx] if data.ndim == 2 else data
        finite_data = col_data[np.isfinite(col_data)]

        if len(finite_data) > 0:
            col_stat = {
                "index": col_idx,
                "min": float(np.min(finite_data)),
                "max": float(np.max(finite_data)),
                "mean": float(np.mean(finite_data)),
                "std": float(np.std(finite_data)),
                "nan_count": int(np.sum(np.isnan(col_data))),
            }
        else:
            col_stat = {
                "index": col_idx,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "nan_count": len(col_data),
            }
        column_stats.append(col_stat)

    return {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "column_stats": column_stats,
    }


# =============================================================================
# Validation
# =============================================================================


def validate_column_selections(
    columns: dict[str, int | None],
    num_columns: int,
    mode: str = "1d",
) -> dict[str, Any]:
    """Validate column selections for completeness and correctness.

    Checks that:
    - All required columns are assigned
    - Column indices are within valid range
    - No duplicate assignments (same column assigned to multiple roles)

    Parameters
    ----------
    columns : dict
        Column assignments with role names as keys and indices as values.
    num_columns : int
        Total number of columns in the data.
    mode : str
        Data mode: "1d" or "2d".

    Returns
    -------
    dict
        Validation result with:
        - "is_valid": bool - Whether selections are valid
        - "message": str - Error message if invalid, empty string if valid
        - "warnings": list[str] - Non-fatal warnings

    Examples
    --------
    >>> validate_column_selections({"x": 0, "y": 1}, num_columns=3, mode="1d")
    {'is_valid': True, 'message': '', 'warnings': []}

    >>> validate_column_selections({"x": 0, "y": None}, num_columns=3, mode="1d")
    {'is_valid': False, 'message': 'Required column Y is not assigned', 'warnings': []}
    """
    result: dict[str, Any] = {
        "is_valid": True,
        "message": "",
        "warnings": [],
    }

    required_roles = get_required_roles(mode)

    # Check that all required columns are assigned
    for role in required_roles:
        col_idx = columns.get(role)
        if col_idx is None:
            result["is_valid"] = False
            result["message"] = f"Required column {role.upper()} is not assigned"
            return result

    # Check that column indices are within valid range
    for role, col_idx in columns.items():
        if col_idx is not None:
            if col_idx < 0 or col_idx >= num_columns:
                result["is_valid"] = False
                result["message"] = (
                    f"Column index {col_idx} for {role.upper()} is out of range (0-{num_columns - 1})"
                )
                return result

    # Check for duplicate assignments
    assigned_indices: dict[int, list[str]] = {}
    for role, col_idx in columns.items():
        if col_idx is not None:
            if col_idx not in assigned_indices:
                assigned_indices[col_idx] = []
            assigned_indices[col_idx].append(role)

    for col_idx, roles in assigned_indices.items():
        if len(roles) > 1:
            result["is_valid"] = False
            result["message"] = (
                f"Column {col_idx} is assigned to multiple roles: {', '.join(r.upper() for r in roles)}. Each column should have a unique role (duplicate assignment)."
            )
            return result

    return result


# =============================================================================
# Streamlit Rendering (optional - only used when Streamlit is available)
# =============================================================================


def render_column_selector(
    data: NDArray[np.float64],
    columns: dict[str, int | None],
    mode: str = "1d",
) -> dict[str, int | None]:
    """Render the column selector UI and return updated selections.

    This function is intended for use within a Streamlit app context.
    It renders dropdown selectors for each column role and returns
    the updated column assignments.

    Parameters
    ----------
    data : ndarray
        2D numpy array with data to display.
    columns : dict
        Current column assignments.
    mode : str
        Data mode: "1d" or "2d".

    Returns
    -------
    dict
        Updated column assignments after user interaction.

    Note
    ----
    This function requires Streamlit to be running. It will import
    streamlit when called.
    """
    import streamlit as st

    num_columns = data.shape[1] if data.ndim == 2 else 1
    column_options = list(range(num_columns))

    st.subheader("Column Assignment")

    available_roles = get_available_roles(mode)
    required_roles = get_required_roles(mode)

    updated_columns: dict[str, int | None] = {}

    # Create columns for the selectors
    cols = st.columns(len(available_roles))

    for i, role in enumerate(available_roles):
        with cols[i]:
            is_required = role in required_roles
            label = get_role_display_name(role)
            if is_required:
                label += " *"

            current_value = columns.get(role)
            color = get_column_color(role)

            # Add color indicator
            if color:
                st.markdown(
                    f'<div style="width: 20px; height: 20px; background-color: {color}; '
                    f'border-radius: 3px; display: inline-block; margin-right: 5px;"></div>',
                    unsafe_allow_html=True,
                )

            if is_required:
                # Required columns use selectbox (no None option)
                selected = st.selectbox(
                    label,
                    options=column_options,
                    index=current_value if current_value is not None else 0,
                    key=f"col_select_{role}",
                )
                updated_columns[role] = selected
            else:
                # Optional columns can be None
                options_with_none = [None, *column_options]
                current_index = 0
                if current_value is not None:
                    current_index = column_options.index(current_value) + 1

                selected = st.selectbox(
                    label,
                    options=options_with_none,
                    index=current_index,
                    format_func=lambda x: "Not assigned"
                    if x is None
                    else f"Column {x}",
                    key=f"col_select_{role}",
                )
                updated_columns[role] = selected

    # Validate and show status
    validation = validate_column_selections(updated_columns, num_columns, mode)

    if not validation["is_valid"]:
        st.error(validation["message"])
    else:
        st.success("Column assignments are valid")

    for warning in validation.get("warnings", []):
        st.warning(warning)

    return updated_columns
