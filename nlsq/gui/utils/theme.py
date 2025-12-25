"""Theme utility module for NLSQ GUI.

This module provides theme-related utilities for consistent styling across
the application, including Plotly chart theming and code editor themes.

Functions
---------
get_current_theme
    Get the current theme from session state.
get_plotly_template
    Get the Plotly template name for a theme.
get_plotly_colors
    Get the color scheme dictionary for a theme.
get_code_editor_theme
    Get the Monaco editor theme for a theme.
toggle_theme
    Toggle between light and dark themes.
apply_theme_to_figure
    Apply theme styling to a Plotly figure.
get_plotly_layout_updates
    Get layout updates for Plotly figures.

Constants
---------
LIGHT_COLORS
    Color scheme dictionary for light theme.
DARK_COLORS
    Color scheme dictionary for dark theme.
"""

from typing import Any

import plotly.graph_objects as go

# =============================================================================
# Theme Color Definitions
# =============================================================================

LIGHT_COLORS: dict[str, str] = {
    # Primary UI colors
    "primary": "#1f77b4",  # Blue - primary accent
    "background": "#ffffff",  # White background
    "secondary_background": "#f0f2f6",  # Light gray secondary bg
    "text": "#262730",  # Dark text
    # Chart colors
    "data": "#1f77b4",  # Blue for data points
    "fit": "#d62728",  # Red for fit line
    "confidence": "rgba(31, 119, 180, 0.2)",  # Semi-transparent blue
    "residuals": "#2ca02c",  # Green for residuals
    "grid": "#e0e0e0",  # Light gray grid
    # Status colors
    "success": "#28a745",
    "warning": "#ffc107",
    "error": "#dc3545",
    # Histogram colors
    "histogram": "rgba(31, 119, 180, 0.7)",
    "histogram_line": "#1f77b4",
    "normal_overlay": "#d62728",
    "kde_overlay": "#2ca02c",
    # Residual bands
    "sigma_1": "rgba(0, 128, 0, 0.15)",  # Green, 1-sigma
    "sigma_2": "rgba(255, 165, 0, 0.15)",  # Orange, 2-sigma
    "sigma_3": "rgba(255, 0, 0, 0.1)",  # Red, 3-sigma
    # Annotation colors
    "annotation_bg": "white",
    "annotation_border": "lightgray",
    "zero_line": "black",
    # Cost plot
    "cost_line": "#1f77b4",
    # Legend
    "legend_bg": "rgba(255, 255, 255, 0.8)",
}

DARK_COLORS: dict[str, str] = {
    # Primary UI colors
    "primary": "#4dabf7",  # Lighter blue for dark mode
    "background": "#0e1117",  # Dark background
    "secondary_background": "#262730",  # Dark secondary bg
    "text": "#fafafa",  # Light text
    # Chart colors
    "data": "#4dabf7",  # Lighter blue for visibility
    "fit": "#ff6b6b",  # Lighter red for visibility
    "confidence": "rgba(77, 171, 247, 0.25)",  # Semi-transparent light blue
    "residuals": "#51cf66",  # Lighter green
    "grid": "#3d3d3d",  # Dark gray grid
    # Status colors
    "success": "#51cf66",
    "warning": "#fcc419",
    "error": "#ff6b6b",
    # Histogram colors
    "histogram": "rgba(77, 171, 247, 0.7)",
    "histogram_line": "#4dabf7",
    "normal_overlay": "#ff6b6b",
    "kde_overlay": "#51cf66",
    # Residual bands
    "sigma_1": "rgba(81, 207, 102, 0.2)",  # Green, 1-sigma
    "sigma_2": "rgba(252, 196, 25, 0.2)",  # Yellow, 2-sigma
    "sigma_3": "rgba(255, 107, 107, 0.15)",  # Red, 3-sigma
    # Annotation colors
    "annotation_bg": "#262730",
    "annotation_border": "#4d4d4d",
    "zero_line": "#fafafa",
    # Cost plot
    "cost_line": "#4dabf7",
    # Legend
    "legend_bg": "rgba(38, 39, 48, 0.8)",
}


# =============================================================================
# Theme Detection and Management
# =============================================================================


def get_current_theme() -> str:
    """Get the current theme from session state.

    Returns
    -------
    str
        Either 'light' or 'dark'.

    Notes
    -----
    If no theme is set in session state, defaults to 'light'.
    This function attempts to import streamlit and access session state,
    but will return 'light' if streamlit is not available or session
    state is not initialized.
    """
    try:
        import streamlit as st

        if "theme" in st.session_state:
            return st.session_state.theme
    except Exception:
        pass

    return "light"


def toggle_theme(current_theme: str) -> str:
    """Toggle between light and dark themes.

    Parameters
    ----------
    current_theme : str
        The current theme ('light' or 'dark').

    Returns
    -------
    str
        The new theme after toggling.
    """
    return "dark" if current_theme == "light" else "light"


def set_theme(theme: str) -> None:
    """Set the theme in session state.

    Parameters
    ----------
    theme : str
        The theme to set ('light' or 'dark').
    """
    try:
        import streamlit as st

        st.session_state.theme = theme
    except Exception:
        pass


# =============================================================================
# Plotly Theme Support
# =============================================================================


def get_plotly_template(theme: str) -> str:
    """Get the Plotly template name for a theme.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    str
        The Plotly template name ('plotly_white' or 'plotly_dark').

    Examples
    --------
    >>> get_plotly_template("light")
    'plotly_white'
    >>> get_plotly_template("dark")
    'plotly_dark'
    """
    if theme == "dark":
        return "plotly_dark"
    return "plotly_white"


def get_plotly_colors(theme: str) -> dict[str, str]:
    """Get the color scheme dictionary for a theme.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    dict[str, str]
        Dictionary of color names to color values.

    Examples
    --------
    >>> colors = get_plotly_colors("light")
    >>> colors["data"]
    '#1f77b4'
    """
    if theme == "dark":
        return DARK_COLORS.copy()
    return LIGHT_COLORS.copy()


def get_plotly_layout_updates(theme: str) -> dict[str, Any]:
    """Get layout updates for Plotly figures based on theme.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    dict[str, Any]
        Dictionary of layout properties to update.
    """
    colors = get_plotly_colors(theme)

    return {
        "template": get_plotly_template(theme),
        "paper_bgcolor": colors["background"],
        "plot_bgcolor": colors["background"],
        "font": {"color": colors["text"]},
        "xaxis": {
            "gridcolor": colors["grid"],
            "zerolinecolor": colors["grid"],
        },
        "yaxis": {
            "gridcolor": colors["grid"],
            "zerolinecolor": colors["grid"],
        },
        "legend": {
            "bgcolor": colors["legend_bg"],
        },
    }


def apply_theme_to_figure(fig: go.Figure, theme: str) -> None:
    """Apply theme styling to a Plotly figure.

    This function modifies the figure in place.

    Parameters
    ----------
    fig : go.Figure
        The Plotly figure to update.
    theme : str
        Either 'light' or 'dark'.
    """
    layout_updates = get_plotly_layout_updates(theme)
    fig.update_layout(**layout_updates)


# =============================================================================
# Code Editor Theme Support
# =============================================================================


def get_code_editor_theme(theme: str) -> str:
    """Get the Monaco editor theme for a theme.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    str
        The Monaco editor theme name ('vs' or 'vs-dark').

    Examples
    --------
    >>> get_code_editor_theme("light")
    'vs'
    >>> get_code_editor_theme("dark")
    'vs-dark'
    """
    if theme == "dark":
        return "vs-dark"
    return "vs"


# =============================================================================
# Theme-Aware Color Getters for Specific Components
# =============================================================================


def get_data_marker_color(theme: str) -> str:
    """Get the color for data point markers.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    str
        Hex color string.
    """
    colors = get_plotly_colors(theme)
    return colors["data"]


def get_fit_line_color(theme: str) -> str:
    """Get the color for the fit curve line.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    str
        Hex color string.
    """
    colors = get_plotly_colors(theme)
    return colors["fit"]


def get_confidence_band_color(theme: str) -> str:
    """Get the color for confidence bands.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    str
        RGBA color string.
    """
    colors = get_plotly_colors(theme)
    return colors["confidence"]


def get_histogram_colors(theme: str) -> dict[str, str]:
    """Get colors for histogram components.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    dict[str, str]
        Dictionary with 'bar', 'line', 'normal', and 'kde' colors.
    """
    colors = get_plotly_colors(theme)
    return {
        "bar": colors["histogram"],
        "line": colors["histogram_line"],
        "normal": colors["normal_overlay"],
        "kde": colors["kde_overlay"],
    }


def get_sigma_band_colors(theme: str) -> dict[str, str]:
    """Get colors for sigma bands in residual plots.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    dict[str, str]
        Dictionary with 'sigma_1', 'sigma_2', and 'sigma_3' colors.
    """
    colors = get_plotly_colors(theme)
    return {
        "sigma_1": colors["sigma_1"],
        "sigma_2": colors["sigma_2"],
        "sigma_3": colors["sigma_3"],
    }


def get_annotation_style(theme: str) -> dict[str, str]:
    """Get styling for plot annotations.

    Parameters
    ----------
    theme : str
        Either 'light' or 'dark'.

    Returns
    -------
    dict[str, str]
        Dictionary with 'bgcolor' and 'bordercolor'.
    """
    colors = get_plotly_colors(theme)
    return {
        "bgcolor": colors["annotation_bg"],
        "bordercolor": colors["annotation_border"],
    }


# =============================================================================
# Dark Theme CSS Application
# =============================================================================


def apply_dark_theme_css() -> None:
    """Apply dark theme CSS overrides when dark mode is enabled.

    This injects custom CSS to override Streamlit's default light theme
    with dark mode colors. Should be called near the top of each page's
    main function, after set_page_config.

    Only applies CSS if the current theme is 'dark'.
    """
    import streamlit as st

    if get_current_theme() != "dark":
        return

    colors = get_plotly_colors("dark")

    dark_css = f"""
    <style>
    /* Dark mode CSS overrides */
    .stApp {{
        background-color: {colors["background"]};
        color: {colors["text"]};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {colors["secondary_background"]};
    }}

    /* Main content area */
    .main .block-container {{
        background-color: {colors["background"]};
    }}

    /* Text elements */
    .stMarkdown, .stText, p, span, label {{
        color: {colors["text"]} !important;
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {colors["text"]} !important;
    }}

    /* Widgets and inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {{
        background-color: {colors["secondary_background"]} !important;
        color: {colors["text"]} !important;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {colors["primary"]} !important;
        color: {colors["text"]} !important;
    }}

    /* Metrics and info boxes */
    [data-testid="stMetricValue"] {{
        color: {colors["text"]} !important;
    }}

    /* Expanders */
    .streamlit-expanderHeader {{
        background-color: {colors["secondary_background"]} !important;
        color: {colors["text"]} !important;
    }}

    /* Tables */
    .stDataFrame {{
        background-color: {colors["secondary_background"]} !important;
    }}

    /* Code blocks */
    .stCodeBlock {{
        background-color: {colors["secondary_background"]} !important;
    }}

    /* Dividers */
    hr {{
        border-color: {colors["grid"]} !important;
    }}

    /* Status messages */
    .stSuccess, .stInfo, .stWarning, .stError {{
        background-color: {colors["secondary_background"]} !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background-color: {colors["secondary_background"]};
    }}

    .stTabs [data-baseweb="tab"] {{
        color: {colors["text"]} !important;
    }}

    /* Radio buttons and checkboxes */
    .stRadio > div, .stCheckbox > div {{
        color: {colors["text"]} !important;
    }}

    /* Sliders */
    .stSlider > div {{
        color: {colors["text"]} !important;
    }}
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
