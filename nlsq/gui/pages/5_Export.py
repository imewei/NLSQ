"""Export Page for NLSQ GUI.

This page provides export functionality for fitting results including:
- Session bundle (ZIP) containing all artifacts
- JSON results file
- CSV parameters file
- Python code to reproduce the fit
- Interactive Plotly HTML exports

The page is only available after a successful curve fit.
"""

from typing import Any

import streamlit as st

from nlsq.gui.adapters.export_adapter import (
    create_session_bundle,
    export_csv,
    export_json,
)
from nlsq.gui.components.param_config import get_param_names_from_model
from nlsq.gui.state import SessionState, initialize_state
from nlsq.gui.utils.code_generator import generate_fit_script
from nlsq.gui.utils.theme import apply_dark_theme_css

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Export - NLSQ",
    page_icon="arrow_down",
    layout="wide",
)


# =============================================================================
# Session State
# =============================================================================


def get_session_state() -> SessionState:
    """Get or initialize the session state."""
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()
    return st.session_state.nlsq_state


def get_fit_result() -> Any:
    """Get the fit result from session state."""
    state = get_session_state()
    return state.fit_result


def get_current_model() -> Any:
    """Get the current model from session state."""
    return st.session_state.get("current_model", None)


def get_figures() -> dict[str, Any]:
    """Get stored figures from session state."""
    return st.session_state.get("export_figures", {})


# =============================================================================
# Results Check
# =============================================================================


def has_valid_result() -> bool:
    """Check if there is a valid fit result to export."""
    result = get_fit_result()
    if result is None:
        return False

    # Check for required attributes
    return hasattr(result, "popt")


# =============================================================================
# Export Section: Session Bundle
# =============================================================================


def render_session_bundle_section() -> None:
    """Render the session bundle export section."""
    st.markdown("### Session Bundle")
    st.markdown(
        "Download a complete session bundle containing data, configuration, "
        "results, and visualizations in a single ZIP file."
    )

    state = get_session_state()
    result = get_fit_result()
    figures = get_figures()

    # Show bundle contents preview
    with st.expander("Bundle Contents Preview"):
        st.markdown("""
        The session bundle includes:
        - **data.csv** - Data snapshot (x, y, sigma values)
        - **config.yaml** - Workflow configuration
        - **results.json** - Full fit results with statistics
        """)

        if figures:
            st.markdown("- **Visualizations:**")
            for name in figures:
                st.markdown(f"  - {name}.html")

    # Generate and provide download
    try:
        zip_bytes = create_session_bundle(state, result, figures)

        st.download_button(
            label="Download Session Bundle (ZIP)",
            data=zip_bytes,
            file_name="nlsq_session.zip",
            mime="application/zip",
            key="download_session_bundle",
            width="stretch",
        )
    except Exception as e:
        st.error(f"Failed to create session bundle: {e}")


# =============================================================================
# Export Section: Individual Files
# =============================================================================


def render_individual_exports_section() -> None:
    """Render individual file export options."""
    st.markdown("### Individual Exports")

    result = get_fit_result()
    model = get_current_model()

    # Get parameter names
    param_names = None
    if model is not None:
        param_names = get_param_names_from_model(model)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**JSON Results**")
        st.caption("Full results with parameters, covariance, and statistics")

        try:
            json_str = export_json(result, param_names=param_names)

            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="nlsq_results.json",
                mime="application/json",
                key="download_json",
                width="stretch",
            )
        except Exception as e:
            st.error(f"Failed to generate JSON: {e}")

    with col2:
        st.markdown("**CSV Parameters**")
        st.caption("Parameter values and uncertainties in CSV format")

        try:
            csv_str = export_csv(result, param_names=param_names)

            st.download_button(
                label="Download CSV",
                data=csv_str,
                file_name="nlsq_parameters.csv",
                mime="text/csv",
                key="download_csv",
                width="stretch",
            )
        except Exception as e:
            st.error(f"Failed to generate CSV: {e}")


# =============================================================================
# Export Section: Python Code
# =============================================================================


def render_python_code_section() -> None:
    """Render the Python code export section."""
    st.markdown("### Python Script")
    st.markdown(
        "Generate a standalone Python script that reproduces this fit. "
        "The script includes data, model definition, and curve_fit call."
    )

    state = get_session_state()
    result = get_fit_result()

    try:
        code = generate_fit_script(state, result)

        # Display code with syntax highlighting
        st.code(code, language="python", line_numbers=True)

        # Download button for the script
        col1, _col2 = st.columns([1, 3])

        with col1:
            st.download_button(
                label="Download .py File",
                data=code,
                file_name="nlsq_fit_script.py",
                mime="text/x-python",
                key="download_python",
                width="stretch",
            )

    except Exception as e:
        st.error(f"Failed to generate Python script: {e}")


# =============================================================================
# Sidebar
# =============================================================================


def render_sidebar() -> None:
    """Render the sidebar with export summary."""
    result = get_fit_result()

    st.sidebar.subheader("Export Summary")

    if result is not None:
        # Show what's available to export
        st.sidebar.markdown("**Available Data:**")

        if hasattr(result, "popt"):
            st.sidebar.markdown(f"- {len(result.popt)} fitted parameters")

        if hasattr(result, "pcov") and result.pcov is not None:
            st.sidebar.markdown("- Covariance matrix")

        if hasattr(result, "r_squared"):
            st.sidebar.markdown("- Fit statistics")

        figures = get_figures()
        if figures:
            st.sidebar.markdown(f"- {len(figures)} visualization(s)")

    st.sidebar.divider()

    st.sidebar.caption(
        "The session bundle includes all data in a single download. "
        "Use individual exports for specific file formats."
    )


# =============================================================================
# No Results Page
# =============================================================================


def render_no_results() -> None:
    """Render the page when no results are available."""
    st.title("Export")
    st.warning("No fitting results available to export.")

    st.markdown("""
    To export results:

    1. **Load Data** - Import your dataset on the Data Loading page
    2. **Select Model** - Choose a model on the Model Selection page
    3. **Run Fit** - Configure and run the fit on the Fitting Options page
    4. **Export** - Return here to download results

    Once a fit completes successfully, you can export:
    - Session bundle (ZIP) with all artifacts
    - JSON results file
    - CSV parameters file
    - Python script to reproduce the fit
    """)

    # Quick status
    state = get_session_state()

    st.markdown("### Current Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if state.xdata is not None:
            st.success("Data loaded")
        else:
            st.error("No data loaded")

    with col2:
        if get_current_model() is not None:
            st.success("Model selected")
        else:
            st.error("No model selected")

    with col3:
        if state.fit_result is not None:
            st.success("Fit complete")
        else:
            st.error("No fit results")


# =============================================================================
# Main Page
# =============================================================================


def main() -> None:
    """Main entry point for the Export page."""
    apply_dark_theme_css()

    st.title("Export Results")

    # Check if we have results to export
    if not has_valid_result():
        render_no_results()
        return

    st.caption("Download fitting results in various formats")

    # Render sidebar
    render_sidebar()

    # Session bundle section
    render_session_bundle_section()

    st.divider()

    # Individual exports section
    render_individual_exports_section()

    st.divider()

    # Python code section
    render_python_code_section()


if __name__ == "__main__":
    main()
