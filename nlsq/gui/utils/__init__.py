"""NLSQ GUI Utilities - Helper functions and utilities."""

from nlsq.gui.utils.code_generator import generate_fit_script
from nlsq.gui.utils.theme import (
    DARK_COLORS,
    LIGHT_COLORS,
    apply_theme_to_figure,
    get_annotation_style,
    get_code_editor_theme,
    get_confidence_band_color,
    get_current_theme,
    get_data_marker_color,
    get_fit_line_color,
    get_histogram_colors,
    get_plotly_colors,
    get_plotly_layout_updates,
    get_plotly_template,
    get_sigma_band_colors,
    set_theme,
    toggle_theme,
)

__all__ = [
    # Theme constants
    "DARK_COLORS",
    "LIGHT_COLORS",
    # Theme functions
    "apply_theme_to_figure",
    # Code generator
    "generate_fit_script",
    "get_annotation_style",
    "get_code_editor_theme",
    "get_confidence_band_color",
    "get_current_theme",
    "get_data_marker_color",
    "get_fit_line_color",
    "get_histogram_colors",
    "get_plotly_colors",
    "get_plotly_layout_updates",
    "get_plotly_template",
    "get_sigma_band_colors",
    "set_theme",
    "toggle_theme",
]
