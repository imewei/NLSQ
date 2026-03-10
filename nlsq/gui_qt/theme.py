"""
NLSQ Qt GUI Theme Management

This module provides theme configuration and management for the Qt application.
Supports light and dark themes with consistent styling across widgets and plots.

Uses Qt 6.5+ built-in color scheme API for widget theming. pyqtgraph plots
are themed separately via ThemeConfig since they don't follow Qt's color scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import Qt

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication

__all__ = ["DARK_THEME", "LIGHT_THEME", "ThemeConfig", "ThemeManager"]


@dataclass
class ThemeConfig:
    """Theme colors and styles for Qt widgets and pyqtgraph plots."""

    name: str  # "light" or "dark"

    # Widget colors
    background: str  # Main background
    surface: str  # Card/panel background
    surface_variant: (
        str  # Elevated card / panel background (slightly lighter than surface)
    )
    text_primary: str  # Primary text
    text_secondary: str  # Secondary/muted text
    border: str  # Borders and dividers
    accent: str  # Primary accent color
    accent_hover: str  # Accent hover state

    # Status colors
    success: str
    warning: str
    error: str
    info: str

    # Semantic stat-card colors (for R², convergence indicators)
    stat_good: str  # Excellent fit / converged
    stat_warning: str  # Acceptable fit / marginal
    stat_bad: str  # Poor fit / failed

    # Plot colors (pyqtgraph)
    plot_background: str
    plot_foreground: str  # Axes, text
    plot_grid: str
    data_marker: str  # Data points
    fit_line: str  # Fitted curve
    residual_color: str
    confidence_band: str

    # Multi-series plot palette — 6 colorblind-safe colors (Wong 2011)
    plot_colors: list  # list[str]  — kept as `list` for dataclass compatibility

    # Syntax-highlight colors (code editor)
    syntax_keyword: str  # Language keywords (def, return, …)
    syntax_builtin: str  # Built-in functions (len, print, …)
    syntax_string: str  # String literals
    syntax_comment: str  # Comments
    syntax_number: str  # Numeric literals
    syntax_function: str  # Function-definition names & scientific identifiers
    syntax_decorator: str  # Decorators (@jit, …)

    @property
    def is_dark(self) -> bool:
        """Check if this is a dark theme."""
        return self.name == "dark"


# Predefined dark theme
DARK_THEME = ThemeConfig(
    name="dark",
    background="#1e1e1e",
    surface="#2d2d2d",
    surface_variant="#383838",  # slightly elevated panels / stat cards
    text_primary="#ffffff",
    text_secondary="#b0b0b0",
    border="#3d3d3d",
    accent="#2196F3",
    accent_hover="#1976D2",
    success="#4CAF50",
    warning="#FF9800",
    error="#f44336",
    info="#2196F3",
    # Stat-card semantic colors (WCAG AA on surface_variant #383838)
    stat_good="#66BB6A",  # green — R² ≥ 0.99
    stat_warning="#FFA726",  # amber — R² ≥ 0.90
    stat_bad="#EF5350",  # red   — R² < 0.90
    plot_background="#1e1e1e",
    plot_foreground="#e0e0e0",
    plot_grid="#3d3d3d",
    data_marker="#2196F3",
    fit_line="#FF5722",
    residual_color="#4CAF50",
    confidence_band="rgba(33, 150, 243, 0.2)",
    # Wong (2011) colorblind-safe palette, adapted for dark backgrounds
    plot_colors=[
        "#56B4E9",  # sky blue   — series 0 / data
        "#E69F00",  # orange     — series 1
        "#009E73",  # green      — series 2
        "#F0E442",  # yellow     — series 3
        "#CC79A7",  # pink       — series 4
        "#D55E00",  # vermilion — series 5
    ],
    # Syntax colors — VS Code Dark+ palette (readable on #2d2d2d)
    syntax_keyword="#569CD6",  # blue
    syntax_builtin="#4EC9B0",  # cyan
    syntax_string="#CE9178",  # orange-brown
    syntax_comment="#6A9955",  # green (italic applied separately)
    syntax_number="#B5CEA8",  # light green
    syntax_function="#DCDCAA",  # yellow
    syntax_decorator="#C586C0",  # purple
)

# Predefined light theme
LIGHT_THEME = ThemeConfig(
    name="light",
    background="#ffffff",
    surface="#f5f5f5",
    surface_variant="#ebebeb",  # slightly elevated panels / stat cards
    text_primary="#212121",
    text_secondary="#757575",
    border="#e0e0e0",
    accent="#1976D2",
    accent_hover="#1565C0",
    success="#388E3C",
    warning="#F57C00",
    error="#D32F2F",
    info="#1976D2",
    # Stat-card semantic colors (WCAG AA on surface_variant #ebebeb)
    stat_good="#2E7D32",  # dark green — R² ≥ 0.99
    stat_warning="#E65100",  # deep orange — R² ≥ 0.90
    stat_bad="#C62828",  # dark red   — R² < 0.90
    plot_background="#ffffff",
    plot_foreground="#212121",
    plot_grid="#e0e0e0",
    data_marker="#1976D2",
    fit_line="#D84315",
    residual_color="#388E3C",
    confidence_band="rgba(25, 118, 210, 0.2)",
    # Wong (2011) colorblind-safe palette, adapted for light backgrounds
    plot_colors=[
        "#0072B2",  # blue       — series 0 / data
        "#E69F00",  # orange     — series 1
        "#009E73",  # green      — series 2
        "#CC79A7",  # pink       — series 3
        "#56B4E9",  # sky blue   — series 4
        "#D55E00",  # vermilion — series 5
    ],
    # Syntax colors — VS Code Light+ palette (readable on #f5f5f5)
    syntax_keyword="#0000FF",  # blue
    syntax_builtin="#267F99",  # teal
    syntax_string="#A31515",  # dark red
    syntax_comment="#008000",  # green (italic applied separately)
    syntax_number="#098658",  # dark green
    syntax_function="#795E26",  # brown-gold
    syntax_decorator="#AF00DB",  # purple
)


class ThemeManager(QObject):
    """Manages application theme switching.

    This class handles theme changes for the entire application using Qt 6.5+
    built-in color scheme API and emitting signals for widgets to update their styling.
    """

    # Signal emitted when theme changes
    theme_changed = Signal(ThemeConfig)

    def __init__(self, app: QApplication) -> None:
        """Initialize the theme manager.

        Args:
            app: The QApplication instance
        """
        super().__init__()
        self._app = app
        self._current_theme = DARK_THEME

    @property
    def current_theme(self) -> ThemeConfig:
        """Get the current theme configuration."""
        return self._current_theme

    def set_theme(self, name: str) -> None:
        """Set the application theme.

        Args:
            name: Theme name ("light" or "dark")
        """
        if name == "light":
            self._current_theme = LIGHT_THEME
            self._app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        else:
            self._current_theme = DARK_THEME
            self._app.styleHints().setColorScheme(Qt.ColorScheme.Dark)

        self.theme_changed.emit(self._current_theme)

    def get_theme(self) -> ThemeConfig:
        """Get the current theme configuration.

        Returns:
            The current ThemeConfig
        """
        return self._current_theme

    def toggle(self) -> None:
        """Toggle between light and dark themes."""
        if self._current_theme.name == "dark":
            self.set_theme("light")
        else:
            self.set_theme("dark")
