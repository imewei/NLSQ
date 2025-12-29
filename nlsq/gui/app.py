"""NLSQ GUI - Main Streamlit application entry point.

This module provides the main entry point for the NLSQ Streamlit GUI application.
It sets up the multi-page app structure, initializes session state on startup,
implements sidebar with navigation and status indicators, and provides navigation
guards to enforce workflow order.

Usage:
    streamlit run nlsq/gui/app.py
"""

from typing import Any

import streamlit as st

from nlsq.gui.state import SessionState, initialize_state
from nlsq.gui.utils.theme import (
    apply_dark_theme_css,
    get_current_theme,
    set_theme,
)


def configure_page() -> None:
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="NLSQ Curve Fitting",
        page_icon="chart_with_upwards_trend",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/imewei/NLSQ",
            "Report a bug": "https://github.com/imewei/NLSQ/issues",
            "About": "# NLSQ Curve Fitting\nGPU/TPU-accelerated nonlinear least squares fitting.",
        },
    )


def initialize_session_state() -> None:
    """Initialize session state on startup.

    This function ensures that the session state contains a valid SessionState
    instance. If one doesn't exist, it creates a new one with default values.
    Also initializes theme state if not present.
    """
    if "nlsq_state" not in st.session_state:
        st.session_state.nlsq_state = initialize_state()

    # Initialize theme state
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def get_session_state() -> SessionState:
    """Get the current session state.

    Returns
    -------
    SessionState
        The current session state instance.
    """
    initialize_session_state()
    return st.session_state.nlsq_state


def get_page_status(state: SessionState) -> dict[str, Any]:
    """Get the status of each workflow page.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    dict
        A dictionary containing status information for each page:
        - data: loaded, point_count, file_name
        - model: selected, name, type
        - fit: running, complete
    """
    # Check data status
    data_loaded = state.xdata is not None and state.ydata is not None
    point_count = 0
    if data_loaded:
        try:
            point_count = len(state.xdata)
        except (TypeError, AttributeError):
            point_count = 0

    data_status = {
        "loaded": data_loaded,
        "point_count": point_count,
        "file_name": state.data_file_name or "",
    }

    # Check model status
    model_selected = bool(state.model_type) and (
        state.model_type in {"builtin", "polynomial"}
        or (state.model_type == "custom" and state.custom_code)
    )
    model_status = {
        "selected": model_selected,
        "name": state.model_name if state.model_type == "builtin" else state.model_type,
        "type": state.model_type,
    }

    # Check fit status
    fit_complete = state.fit_result is not None
    fit_running = state.fit_running

    fit_status = {
        "running": fit_running,
        "complete": fit_complete,
    }

    return {
        "data": data_status,
        "model": model_status,
        "fit": fit_status,
    }


def can_access_page(page: str, state: SessionState) -> bool:
    """Check if a page can be accessed based on workflow state.

    This implements navigation guards to enforce workflow order:
    - Data Loading: Always accessible
    - Model Selection: Requires data loaded
    - Fitting Options: Requires data loaded and model selected
    - Results: Requires fit complete
    - Export: Requires fit complete

    Parameters
    ----------
    page : str
        The page name: "data", "model", "fitting", "results", or "export".
    state : SessionState
        The current session state.

    Returns
    -------
    bool
        True if the page can be accessed, False otherwise.
    """
    status = get_page_status(state)

    if page == "data":
        return True
    elif page == "model":
        return status["data"]["loaded"]
    elif page == "fitting":
        return status["data"]["loaded"] and status["model"]["selected"]
    elif page in {"results", "export"}:
        return status["fit"]["complete"]
    else:
        return True


def get_navigation_message(page: str, state: SessionState) -> str:
    """Get a helpful message explaining why a page is disabled.

    Parameters
    ----------
    page : str
        The page name.
    state : SessionState
        The current session state.

    Returns
    -------
    str
        A helpful message explaining what needs to be done to access the page.
    """
    status = get_page_status(state)

    if page == "model" and not status["data"]["loaded"]:
        return "Load data first to select a model."
    elif page == "fitting":
        if not status["data"]["loaded"]:
            return "Load data first to configure fitting options."
        elif not status["model"]["selected"]:
            return "Select a model first to configure fitting options."
    elif page == "results" and not status["fit"]["complete"]:
        return "Run a fit first to view results."
    elif page == "export" and not status["fit"]["complete"]:
        return "Run a fit first to export results."

    return ""


def get_workflow_step_message(state: SessionState) -> str:
    """Get a message describing the current workflow step.

    Parameters
    ----------
    state : SessionState
        The current session state.

    Returns
    -------
    str
        A message describing what to do next.
    """
    status = get_page_status(state)

    if not status["data"]["loaded"]:
        return "Load data to begin your curve fitting workflow."
    elif not status["model"]["selected"]:
        return "Select a model for fitting your data."
    elif not status["fit"]["complete"]:
        if status["fit"]["running"]:
            return "Fit is running..."
        return "Run the fit to optimize parameters."
    else:
        return "Fit complete! View Results or Export your analysis."


def get_version_info() -> dict[str, str]:
    """Get NLSQ version information.

    Returns
    -------
    dict
        A dictionary with version info and URLs.
    """
    try:
        from nlsq import __version__

        version = __version__
    except ImportError:
        version = "unknown"

    return {
        "version": version,
        "github_url": "https://github.com/imewei/NLSQ",
        "docs_url": "https://nlsq.readthedocs.io",
    }


def get_help_tips() -> list[str]:
    """Get help tips for the user.

    Returns
    -------
    list[str]
        A list of help tip strings.
    """
    return [
        "Load your data from CSV, ASCII, NPZ, or HDF5 files.",
        "Use clipboard paste to quickly import data from Excel or Google Sheets.",
        "Select from 7 built-in models or define your own custom Python function.",
        "Use Guided Mode with presets (Fast/Robust/Quality) for quick setup.",
        "Switch to Advanced Mode for full control over all fitting parameters.",
        "Monitor fit progress in real-time with live cost function plots.",
        "Export your results as a session bundle containing data, config, and plots.",
        "Generate reproducible Python code to share your analysis.",
    ]


def render_theme_toggle() -> None:
    """Render the theme toggle switch in the sidebar.

    Displays a toggle to switch between light and dark modes.
    The theme preference is stored in session state.
    """
    current_theme = get_current_theme()
    is_dark = current_theme == "dark"

    # Theme toggle
    dark_mode = st.toggle(
        "Dark Mode",
        value=is_dark,
        key="theme_toggle",
        help="Switch between light and dark themes",
    )

    # Update theme in session state if changed
    new_theme = "dark" if dark_mode else "light"
    if new_theme != current_theme:
        set_theme(new_theme)
        st.rerun()


def render_sidebar_status(state: SessionState) -> None:
    """Render the workflow status panel in the sidebar.

    Parameters
    ----------
    state : SessionState
        The current session state.
    """
    st.subheader("Workflow Status")

    status = get_page_status(state)

    # Data status
    if status["data"]["loaded"]:
        st.success(f"Data: {status['data']['point_count']:,} points")
        if status["data"]["file_name"]:
            st.caption(f"File: {status['data']['file_name']}")
    else:
        st.info("Data: Not loaded")

    # Model status
    if status["model"]["selected"]:
        if status["model"]["type"] == "builtin":
            st.success(f"Model: {status['model']['name']}")
        elif status["model"]["type"] == "polynomial":
            st.success("Model: Polynomial")
        else:
            st.success("Model: Custom")
    else:
        st.info("Model: Not selected")

    # Fit status
    if status["fit"]["running"]:
        st.warning("Fit: Running...")
    elif status["fit"]["complete"]:
        st.success("Fit: Complete")
    else:
        st.info("Fit: Not run")

    st.divider()

    # Next step message
    message = get_workflow_step_message(state)
    st.caption(message)


def render_help_section() -> None:
    """Render the help section in the sidebar."""
    with st.expander("Help"):
        st.markdown("**Quick Start:**")
        tips = get_help_tips()
        # Show first 5 tips in quick start
        for i, tip in enumerate(tips[:5], 1):
            st.markdown(f"{i}. {tip}")

        st.markdown("---")
        st.markdown("**Keyboard Shortcuts:**")
        st.markdown("- `R` - Rerun app")
        st.markdown("- `C` - Clear cache")


def render_about_section() -> None:
    """Render the about section in the sidebar."""
    with st.expander("About NLSQ"):
        version_info = get_version_info()
        st.markdown(f"**Version:** {version_info['version']}")
        st.markdown("""
GPU/TPU-accelerated nonlinear least squares
curve fitting library built on JAX.

**Features:**
- Drop-in SciPy compatibility
- Automatic differentiation
- Streaming for 100M+ points
- Multi-start global optimization
""")
        st.markdown(
            f"[GitHub]({version_info['github_url']}) | "
            f"[Documentation]({version_info['docs_url']})"
        )


def render_sidebar() -> None:
    """Render the sidebar with navigation and status information."""
    with st.sidebar:
        st.title("NLSQ")
        st.caption("Curve Fitting GUI")

        # Theme toggle at the top of sidebar
        render_theme_toggle()

        st.divider()

        # Get current state
        state = get_session_state()

        # Workflow status section
        render_sidebar_status(state)

        st.divider()

        # Help section
        render_help_section()

        # About section
        render_about_section()


def render_main_content() -> None:
    """Render the main content area for the home page."""
    st.title("NLSQ Curve Fitting")
    st.markdown("""
Welcome to the NLSQ Curve Fitting GUI. This application provides
an interactive interface for performing nonlinear least squares curve fitting
with GPU/TPU acceleration.

**Getting Started:**
- Use the sidebar pages to navigate through the workflow
- Start by loading your data on the Data Loading page
- Select a model and configure fitting parameters
- Run the fit and visualize results
- Export your results in various formats
""")

    # Feature overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Data Loading")
        st.markdown("""
- CSV, ASCII, NPZ, HDF5 formats
- Clipboard paste support
- Column selection with preview
- 1D and 2D data modes
""")

    with col2:
        st.subheader("Model Selection")
        st.markdown("""
- 7 built-in models
- Polynomial (degree 0-10)
- Custom Python functions
- LaTeX equation preview
""")

    with col3:
        st.subheader("Fitting & Export")
        st.markdown("""
- Real-time progress feedback
- Interactive Plotly visualizations
- Session bundle export
- Python code generation
""")


def check_page_access_and_show_message(page: str) -> bool:
    """Check page access and display message if not accessible.

    Parameters
    ----------
    page : str
        The page name.

    Returns
    -------
    bool
        True if the page can be accessed, False otherwise.
    """
    state = get_session_state()

    if not can_access_page(page, state):
        message = get_navigation_message(page, state)
        st.warning(message)
        return False

    return True


def main() -> None:
    """Main entry point for the NLSQ GUI application."""
    configure_page()
    initialize_session_state()

    # Apply dark theme CSS if dark mode is enabled
    apply_dark_theme_css()

    render_sidebar()
    render_main_content()


if __name__ == "__main__":
    main()
