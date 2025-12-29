"""Data Loading Page for NLSQ GUI.

This page provides the interface for loading data from files or clipboard,
selecting columns, and previewing data with statistics.

Supported formats:
- CSV files (.csv)
- ASCII text files (.txt, .dat, .asc)
- NumPy archives (.npz)
- HDF5 files (.h5, .hdf5)
- Clipboard paste from Excel/Google Sheets
"""

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from nlsq.gui.adapters.data_adapter import (
    compute_statistics,
    load_from_clipboard,
    load_from_file,
    validate_data,
)
from nlsq.gui.components.column_selector import (
    get_available_roles,
    get_column_color,
    get_required_roles,
    get_role_display_name,
    validate_column_selections,
)
from nlsq.gui.state import SessionState, initialize_state
from nlsq.gui.utils.theme import apply_dark_theme_css

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Data Loading - NLSQ",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# =============================================================================
# Session State Initialization
# =============================================================================


def get_session_state() -> SessionState:
    """Get or initialize the session state."""
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()
    return st.session_state.nlsq_state


def init_data_page_state() -> None:
    """Initialize data page specific state."""
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None
    if "column_assignments" not in st.session_state:
        st.session_state.column_assignments = {"x": 0, "y": 1, "z": None, "sigma": None}
    if "data_source" not in st.session_state:
        st.session_state.data_source = None


# =============================================================================
# Data Loading Functions
# =============================================================================


def handle_file_upload(uploaded_file: Any, format_override: str) -> None:
    """Handle file upload and parse data.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The Streamlit uploaded file object.
    format_override : str
        Format override: "auto", "csv", "ascii", "npz", or "hdf5".
    """
    state = get_session_state()

    try:
        # Build config for data loader
        config: dict[str, Any] = {
            "format": format_override if format_override != "auto" else "auto",
            "columns": st.session_state.column_assignments,
            "csv": {"header": True, "delimiter": ","},
            "ascii": {"comment_char": "#"},
        }

        xdata, ydata, sigma = load_from_file(uploaded_file, config)

        # Store in session state
        state.xdata = xdata
        state.ydata = ydata
        state.sigma = sigma
        state.data_file_name = uploaded_file.name

        # Also store raw data for column selection
        # Re-read file for raw data display
        uploaded_file.seek(0)
        try:
            # Try to read as DataFrame for display
            content = uploaded_file.read().decode("utf-8")
            lines = content.strip().split("\n")

            # Detect delimiter
            if "\t" in lines[0]:
                delimiter = "\t"
            elif "," in lines[0]:
                delimiter = ","
            else:
                delimiter = None

            # Parse to DataFrame
            if delimiter:
                st.session_state.raw_data = pd.read_csv(
                    io.StringIO(content),
                    delimiter=delimiter,
                    header=0 if config["csv"].get("header") else None,
                )
            else:
                # Whitespace delimited
                st.session_state.raw_data = pd.read_csv(
                    io.StringIO(content),
                    delim_whitespace=True,
                    header=None,
                    comment="#",
                )
        except Exception:
            # Fallback: create DataFrame from loaded arrays
            if xdata.ndim == 2:
                # 2D data
                df_data = {
                    "x": xdata[0],
                    "y": xdata[1],
                    "z": ydata,
                }
            else:
                df_data = {"x": xdata, "y": ydata}
            if sigma is not None:
                df_data["sigma"] = sigma
            st.session_state.raw_data = pd.DataFrame(df_data)

        st.session_state.data_source = "file"
        st.success(f"Loaded {len(ydata)} data points from {uploaded_file.name}")

    except Exception as e:
        st.error(f"Error loading file: {e}")


def handle_clipboard_paste(clipboard_text: str, has_header: bool) -> None:
    """Handle clipboard data paste.

    Parameters
    ----------
    clipboard_text : str
        Text pasted from clipboard.
    has_header : bool
        Whether the first row is a header.
    """
    state = get_session_state()

    if not clipboard_text.strip():
        st.warning("Please paste some data first")
        return

    try:
        config: dict[str, Any] = {
            "columns": st.session_state.column_assignments,
            "has_header": has_header,
        }

        xdata, ydata, sigma = load_from_clipboard(clipboard_text, config)

        # Store in session state
        state.xdata = xdata
        state.ydata = ydata
        state.sigma = sigma
        state.data_file_name = "clipboard"

        # Parse for display
        lines = clipboard_text.strip().split("\n")
        if "\t" in lines[0]:
            delimiter = "\t"
        elif "," in lines[0]:
            delimiter = ","
        else:
            delimiter = None

        if delimiter:
            st.session_state.raw_data = pd.read_csv(
                io.StringIO(clipboard_text),
                delimiter=delimiter,
                header=0 if has_header else None,
            )
        else:
            st.session_state.raw_data = pd.read_csv(
                io.StringIO(clipboard_text),
                delim_whitespace=True,
                header=0 if has_header else None,
            )

        st.session_state.data_source = "clipboard"
        st.success(f"Parsed {len(ydata)} data points from clipboard")

    except Exception as e:
        st.error(f"Error parsing clipboard data: {e}")


# =============================================================================
# UI Rendering Functions
# =============================================================================


def render_data_input_section() -> None:
    """Render the data input section with file upload and clipboard paste."""
    st.header("Load Data")

    tab1, tab2 = st.tabs(["File Upload", "Paste from Clipboard"])

    with tab1:
        st.subheader("Upload Data File")

        # Format override dropdown
        format_options = ["auto", "csv", "ascii", "npz", "hdf5"]
        format_override = st.selectbox(
            "Format",
            options=format_options,
            index=0,
            help="Select file format or use 'auto' for automatic detection",
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["csv", "txt", "dat", "asc", "npz", "h5", "hdf5"],
            help="Supported formats: CSV, ASCII text, NumPy NPZ, HDF5",
        )

        if uploaded_file is not None:
            if st.button("Load File", type="primary"):
                handle_file_upload(uploaded_file, format_override)

    with tab2:
        st.subheader("Paste Data from Clipboard")
        st.caption("Copy data from Excel, Google Sheets, or any tabular source")

        has_header = st.checkbox(
            "First row is header",
            value=True,
            help="Check if the first row contains column names",
        )

        clipboard_text = st.text_area(
            "Paste data here",
            height=200,
            placeholder="Paste tab or comma separated data...\nExample:\nx\ty\n1.0\t2.0\n2.0\t4.0",
        )

        if st.button("Parse Data", type="primary"):
            handle_clipboard_paste(clipboard_text, has_header)


def render_mode_selector() -> None:
    """Render the 1D/2D mode selector."""
    state = get_session_state()

    st.subheader("Data Mode")

    mode = st.radio(
        "Select data dimensionality",
        options=["1d", "2d"],
        format_func=lambda x: "1D Curve (x, y)"
        if x == "1d"
        else "2D Surface (x, y, z)",
        horizontal=True,
        index=0 if state.data_mode == "1d" else 1,
        help="1D: Standard curve fitting. 2D: Surface fitting with two independent variables.",
    )

    state.data_mode = mode


def render_column_selector() -> None:
    """Render the column selector UI."""
    state = get_session_state()

    if st.session_state.raw_data is None:
        st.info("Load data to configure column assignments")
        return

    st.subheader("Column Assignment")

    df = st.session_state.raw_data
    num_columns = len(df.columns)
    column_names = list(df.columns)

    # Build options (column index and name)
    options = [(i, str(name)) for i, name in enumerate(column_names)]

    mode = state.data_mode
    available_roles = get_available_roles(mode)
    required_roles = get_required_roles(mode)

    # Create columns for selectors
    cols = st.columns(len(available_roles))

    updated_assignments: dict[str, int | None] = {}

    for i, role in enumerate(available_roles):
        with cols[i]:
            is_required = role in required_roles
            label = get_role_display_name(role)
            if is_required:
                label += " *"

            color = get_column_color(role)
            if color:
                st.markdown(
                    f'<span style="display: inline-block; width: 12px; height: 12px; '
                    f'background-color: {color}; border-radius: 2px; margin-right: 5px;"></span>'
                    f"<strong>{label}</strong>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{label}**")

            current_value = st.session_state.column_assignments.get(role)

            if is_required:
                # Required column - no None option
                selected_idx = st.selectbox(
                    "Column",
                    options=list(range(num_columns)),
                    index=current_value
                    if current_value is not None and current_value < num_columns
                    else i,
                    format_func=lambda x: f"{x}: {column_names[x]}"
                    if x < len(column_names)
                    else str(x),
                    key=f"col_{role}",
                    label_visibility="collapsed",
                )
                updated_assignments[role] = selected_idx
            else:
                # Optional column - allow None
                options_list: list[int | None] = [None, *list(range(num_columns))]
                current_idx = 0
                if current_value is not None and current_value < num_columns:
                    current_idx = current_value + 1

                selected = st.selectbox(
                    "Column",
                    options=options_list,
                    index=current_idx,
                    format_func=lambda x: "Not assigned"
                    if x is None
                    else f"{x}: {column_names[x]}"
                    if x < len(column_names)
                    else str(x),
                    key=f"col_{role}",
                    label_visibility="collapsed",
                )
                updated_assignments[role] = selected

    # Update session state
    st.session_state.column_assignments = updated_assignments

    # Validate and show status
    validation = validate_column_selections(updated_assignments, num_columns, mode)

    if not validation["is_valid"]:
        st.error(validation["message"])
    else:
        st.success("Column assignments are valid")


def render_data_preview() -> None:
    """Render the data preview table."""
    if st.session_state.raw_data is None:
        return

    st.subheader("Data Preview")

    df = st.session_state.raw_data

    # Show first N rows
    max_preview_rows = st.slider(
        "Preview rows",
        min_value=5,
        max_value=min(100, len(df)),
        value=min(10, len(df)),
    )

    # Style the DataFrame with column colors
    mode = get_session_state().data_mode
    assignments = st.session_state.column_assignments

    # Create a styled DataFrame
    preview_df = df.head(max_preview_rows).copy()

    # Build column styles
    def style_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Apply column styling based on assignments."""
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for role, col_idx in assignments.items():
            if col_idx is not None and col_idx < len(df.columns):
                color = get_column_color(role)
                if color:
                    styles.iloc[:, col_idx] = f"background-color: {color}20"
        return styles

    styled_df = preview_df.style.apply(
        lambda _: style_columns(preview_df),
        axis=None,
    )

    st.dataframe(styled_df, width="stretch")


def render_statistics() -> None:
    """Render data statistics after loading."""
    state = get_session_state()

    if state.xdata is None or state.ydata is None:
        return

    st.subheader("Data Statistics")

    # Compute validation
    validation = validate_data(state.xdata, state.ydata, state.sigma)

    # Show validation status
    if validation.is_valid:
        st.success(f"Data validation passed - {validation.point_count} points")
    else:
        st.error(f"Data validation failed: {validation.message}")

    # Compute and display statistics
    stats = compute_statistics(state.xdata, state.ydata, state.sigma)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Points", stats["point_count"])
        st.metric("Data Mode", "2D Surface" if stats["is_2d"] else "1D Curve")

    with col2:
        st.metric("X Range", f"[{stats['x_min']:.4g}, {stats['x_max']:.4g}]")
        st.metric("Y Range", f"[{stats['y_min']:.4g}, {stats['y_max']:.4g}]")

    with col3:
        st.metric("X Mean", f"{stats['x_mean']:.4g}")
        st.metric("Y Mean", f"{stats['y_mean']:.4g}")

    # Show NaN/Inf counts if any
    if validation.nan_count > 0 or validation.inf_count > 0:
        st.warning(
            f"Non-finite values detected: {validation.nan_count} NaN, {validation.inf_count} Inf"
        )

    # Show sigma statistics if present
    if stats.get("has_sigma"):
        with st.expander("Uncertainty (Sigma) Statistics"):
            st.write(
                f"Sigma Range: [{stats['sigma_min']:.4g}, {stats['sigma_max']:.4g}]"
            )
            st.write(f"Sigma Mean: {stats['sigma_mean']:.4g}")


def render_apply_button() -> None:
    """Render the apply button to confirm column assignments."""
    state = get_session_state()

    if st.session_state.raw_data is None:
        return

    st.divider()

    mode = state.data_mode
    assignments = st.session_state.column_assignments
    num_columns = len(st.session_state.raw_data.columns)

    validation = validate_column_selections(assignments, num_columns, mode)

    if st.button(
        "Apply Column Selection",
        type="primary",
        disabled=not validation["is_valid"],
    ):
        try:
            # Re-load data with new column assignments
            df = st.session_state.raw_data
            data_array = df.values.astype(np.float64)

            x_col = assignments["x"]
            y_col = assignments["y"]
            z_col = assignments.get("z")
            sigma_col = assignments.get("sigma")

            if mode == "2d" and z_col is not None:
                # 2D surface data
                x_coords = data_array[:, x_col]
                y_coords = data_array[:, y_col]
                state.xdata = np.vstack([x_coords, y_coords])
                state.ydata = data_array[:, z_col]
            else:
                # 1D curve data
                state.xdata = data_array[:, x_col]
                state.ydata = data_array[:, y_col]

            if sigma_col is not None:
                state.sigma = data_array[:, sigma_col]
            else:
                state.sigma = None

            # Update column assignments in state
            state.x_column = x_col
            state.y_column = y_col
            state.z_column = z_col
            state.sigma_column = sigma_col

            st.success("Column assignments applied successfully")
            st.rerun()

        except Exception as e:
            st.error(f"Error applying column selection: {e}")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Main entry point for the data loading page."""
    apply_dark_theme_css()
    init_data_page_state()

    st.title("Data Loading")
    st.caption("Load your data from files or clipboard")

    # Data input section
    render_data_input_section()

    st.divider()

    # Mode selector
    render_mode_selector()

    st.divider()

    # Column selector (if data loaded)
    render_column_selector()

    # Data preview
    render_data_preview()

    # Apply button
    render_apply_button()

    # Statistics (if data loaded and processed)
    state = get_session_state()
    if state.xdata is not None:
        st.divider()
        render_statistics()


if __name__ == "__main__":
    main()
