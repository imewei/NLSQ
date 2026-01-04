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

    # Plot colors (pyqtgraph)
    plot_background: str
    plot_foreground: str  # Axes, text
    plot_grid: str
    data_marker: str  # Data points
    fit_line: str  # Fitted curve
    residual_color: str
    confidence_band: str

    @property
    def is_dark(self) -> bool:
        """Check if this is a dark theme."""
        return self.name == "dark"


# Predefined dark theme
DARK_THEME = ThemeConfig(
    name="dark",
    background="#1e1e1e",
    surface="#2d2d2d",
    text_primary="#ffffff",
    text_secondary="#b0b0b0",
    border="#3d3d3d",
    accent="#2196F3",
    accent_hover="#1976D2",
    success="#4CAF50",
    warning="#FF9800",
    error="#f44336",
    info="#2196F3",
    plot_background="#1e1e1e",
    plot_foreground="#ffffff",
    plot_grid="#3d3d3d",
    data_marker="#2196F3",
    fit_line="#FF5722",
    residual_color="#4CAF50",
    confidence_band="rgba(33, 150, 243, 0.2)",
)

# Predefined light theme
LIGHT_THEME = ThemeConfig(
    name="light",
    background="#ffffff",
    surface="#f5f5f5",
    text_primary="#212121",
    text_secondary="#757575",
    border="#e0e0e0",
    accent="#1976D2",
    accent_hover="#1565C0",
    success="#388E3C",
    warning="#F57C00",
    error="#D32F2F",
    info="#1976D2",
    plot_background="#ffffff",
    plot_foreground="#212121",
    plot_grid="#e0e0e0",
    data_marker="#1976D2",
    fit_line="#D84315",
    residual_color="#388E3C",
    confidence_band="rgba(25, 118, 210, 0.2)",
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
