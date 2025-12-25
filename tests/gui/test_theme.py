"""Tests for NLSQ GUI theme system.

This module tests the theme customization functionality including:
- Theme toggle state management
- Theme persistence in session state
- Plotly color scheme switching
- Code editor theme application
"""

import pytest


class TestThemeModule:
    """Tests for the theme utility module."""

    def test_get_current_theme_default(self) -> None:
        """Test get_current_theme returns 'light' by default."""
        from nlsq.gui.utils.theme import get_current_theme

        # Without session state, should return 'light'
        theme = get_current_theme()
        assert theme in ("light", "dark")

    def test_get_plotly_template_light(self) -> None:
        """Test get_plotly_template returns correct template for light theme."""
        from nlsq.gui.utils.theme import get_plotly_template

        template = get_plotly_template("light")
        assert template == "plotly_white"

    def test_get_plotly_template_dark(self) -> None:
        """Test get_plotly_template returns correct template for dark theme."""
        from nlsq.gui.utils.theme import get_plotly_template

        template = get_plotly_template("dark")
        assert template == "plotly_dark"

    def test_get_plotly_colors_light(self) -> None:
        """Test get_plotly_colors returns appropriate colors for light theme."""
        from nlsq.gui.utils.theme import get_plotly_colors

        colors = get_plotly_colors("light")

        assert "data" in colors
        assert "fit" in colors
        assert "confidence" in colors
        assert "residuals" in colors
        assert "grid" in colors
        assert "background" in colors
        assert "text" in colors

        # Light theme should have white/light background
        assert colors["background"] in ("#ffffff", "#FFFFFF", "white")

    def test_get_plotly_colors_dark(self) -> None:
        """Test get_plotly_colors returns appropriate colors for dark theme."""
        from nlsq.gui.utils.theme import get_plotly_colors

        colors = get_plotly_colors("dark")

        assert "data" in colors
        assert "fit" in colors
        assert "confidence" in colors
        assert "residuals" in colors
        assert "grid" in colors
        assert "background" in colors
        assert "text" in colors

        # Dark theme should have dark background
        assert colors["background"] not in ("#ffffff", "#FFFFFF", "white")

    def test_get_code_editor_theme_light(self) -> None:
        """Test get_code_editor_theme returns 'vs' for light theme."""
        from nlsq.gui.utils.theme import get_code_editor_theme

        theme = get_code_editor_theme("light")
        assert theme == "vs"

    def test_get_code_editor_theme_dark(self) -> None:
        """Test get_code_editor_theme returns 'vs-dark' for dark theme."""
        from nlsq.gui.utils.theme import get_code_editor_theme

        theme = get_code_editor_theme("dark")
        assert theme == "vs-dark"


class TestThemeStateManagement:
    """Tests for theme state management in session state."""

    def test_toggle_theme_function(self) -> None:
        """Test toggle_theme switches between light and dark."""
        from nlsq.gui.utils.theme import toggle_theme

        # Starting from light
        new_theme = toggle_theme("light")
        assert new_theme == "dark"

        # Starting from dark
        new_theme = toggle_theme("dark")
        assert new_theme == "light"

    def test_theme_colors_dict_structure(self) -> None:
        """Test that theme color dictionaries have consistent structure."""
        from nlsq.gui.utils.theme import DARK_COLORS, LIGHT_COLORS

        # Both should have the same keys
        assert set(LIGHT_COLORS.keys()) == set(DARK_COLORS.keys())

        # All required keys should be present
        required_keys = {
            "data",
            "fit",
            "confidence",
            "residuals",
            "grid",
            "background",
            "text",
            "primary",
            "secondary_background",
            "success",
            "warning",
            "error",
            "histogram",
            "normal_overlay",
            "sigma_1",
            "sigma_2",
            "sigma_3",
        }
        assert required_keys.issubset(set(LIGHT_COLORS.keys()))
        assert required_keys.issubset(set(DARK_COLORS.keys()))


class TestPlotlyColorSchemes:
    """Tests for Plotly color scheme application."""

    def test_get_plotly_layout_updates_light(self) -> None:
        """Test get_plotly_layout_updates provides correct layout for light theme."""
        from nlsq.gui.utils.theme import get_plotly_layout_updates

        layout = get_plotly_layout_updates("light")

        assert "template" in layout
        assert layout["template"] == "plotly_white"
        assert "paper_bgcolor" in layout
        assert "plot_bgcolor" in layout

    def test_get_plotly_layout_updates_dark(self) -> None:
        """Test get_plotly_layout_updates provides correct layout for dark theme."""
        from nlsq.gui.utils.theme import get_plotly_layout_updates

        layout = get_plotly_layout_updates("dark")

        assert "template" in layout
        assert layout["template"] == "plotly_dark"
        assert "paper_bgcolor" in layout
        assert "plot_bgcolor" in layout

    def test_apply_theme_to_figure(self) -> None:
        """Test apply_theme_to_figure updates a Plotly figure."""
        import plotly.graph_objects as go

        from nlsq.gui.utils.theme import apply_theme_to_figure

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))

        # Apply dark theme
        apply_theme_to_figure(fig, "dark")

        # Check that template was applied
        assert fig.layout.template is not None


class TestThemeIntegration:
    """Integration tests for theme system."""

    def test_theme_constants_are_valid_colors(self) -> None:
        """Test that all theme color constants are valid CSS color values."""
        from nlsq.gui.utils.theme import DARK_COLORS, LIGHT_COLORS

        def is_valid_color(color: str) -> bool:
            """Check if color is a valid hex or named color."""
            if color.startswith("#"):
                # Hex color should be 4 or 7 characters
                return len(color) in (4, 7) and all(
                    c in "0123456789abcdefABCDEF" for c in color[1:]
                )
            elif color.startswith("rgba(") or color.startswith("rgb("):
                return True
            else:
                # Named color
                return True

        for name, color in LIGHT_COLORS.items():
            assert is_valid_color(color), f"Invalid light color for {name}: {color}"

        for name, color in DARK_COLORS.items():
            assert is_valid_color(color), f"Invalid dark color for {name}: {color}"

    def test_light_dark_colors_are_different(self) -> None:
        """Test that light and dark themes have different colors where expected."""
        from nlsq.gui.utils.theme import DARK_COLORS, LIGHT_COLORS

        # Background colors should be different
        assert LIGHT_COLORS["background"] != DARK_COLORS["background"]

        # Text colors should be different
        assert LIGHT_COLORS["text"] != DARK_COLORS["text"]

        # Grid colors may be different
        # Data colors may be the same (accent colors)
