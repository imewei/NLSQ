"""
NLSQ Qt GUI Theme Management

This module provides theme configuration and management for the Qt application.
Supports light and dark themes with consistent styling across widgets and plots.

Uses the Qt 6.5+ color scheme API plus a complete application QPalette (see
``_palette_for``) for widget theming. pyqtgraph plots are themed separately via
ThemeConfig since they don't follow Qt's color scheme.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette, Qt

if TYPE_CHECKING:
    from PySide6.QtWidgets import QApplication, QLabel

__all__ = [
    "DARK_THEME",
    "LIGHT_THEME",
    "StatusPainter",
    "StatusRole",
    "ThemeConfig",
    "ThemeManager",
]

StatusRole = Literal["ok", "bad", "warn", "info"]


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
        """Check if this is a dark theme.

        Returns
        -------
        bool
            True if ``name`` is ``"dark"``.
        """
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


def _palette_for(theme: ThemeConfig) -> QPalette:
    """Complete application palette for ``theme``.

    ``setColorScheme`` alone leaves a Fusion palette half-switched under some
    platform themes (light window, light-on-light labels, dark item views), so
    every role the widgets read is set from the ThemeConfig.
    """
    pal = QPalette(QColor(theme.surface_variant), QColor(theme.background))
    roles = {
        QPalette.ColorRole.Window: theme.background,
        QPalette.ColorRole.WindowText: theme.text_primary,
        QPalette.ColorRole.Base: theme.surface,
        QPalette.ColorRole.AlternateBase: theme.surface_variant,
        QPalette.ColorRole.Text: theme.text_primary,
        QPalette.ColorRole.Button: theme.surface_variant,
        QPalette.ColorRole.ButtonText: theme.text_primary,
        QPalette.ColorRole.BrightText: theme.error,
        QPalette.ColorRole.ToolTipBase: theme.surface_variant,
        QPalette.ColorRole.ToolTipText: theme.text_primary,
        QPalette.ColorRole.PlaceholderText: theme.text_secondary,
        QPalette.ColorRole.Link: theme.accent,
        QPalette.ColorRole.Highlight: theme.accent,
        QPalette.ColorRole.HighlightedText: "#ffffff",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(theme.text_secondary))
    return pal


class StatusPainter:
    """Color status labels by semantic role and repaint them on a theme change.

    Fixed hex colors set at event time stay in the old theme's shade after a
    switch; the role is remembered so ``set_theme`` can re-resolve it.
    """

    _ROLES = ("ok", "bad", "warn", "info")

    def __init__(self) -> None:
        self._theme = DARK_THEME
        self._roles: dict[QLabel, StatusRole] = {}

    def paint(self, label: QLabel, role: StatusRole) -> None:
        """Color ``label`` with ``role`` of the active theme.

        Parameters
        ----------
        label : QLabel
            Label to color.
        role : {"ok", "bad", "warn", "info"}
            Semantic role: the theme's success, error, warning or info color.

        Raises
        ------
        ValueError
            If ``role`` is not one of the listed roles.
        """
        if role not in self._ROLES:
            raise ValueError(
                f"unknown status role {role!r}; expected one of {self._ROLES}"
            )
        self._roles[label] = role
        self._apply(label, role)

    def clear(self, label: QLabel, style: str = "") -> None:
        """Forget ``label``'s role and set a theme-independent ``style``."""
        self._roles.pop(label, None)
        label.setStyleSheet(style)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Adopt ``theme`` and repaint every labelled role."""
        self._theme = theme
        for label, role in self._roles.items():
            self._apply(label, role)

    def _apply(self, label: QLabel, role: StatusRole) -> None:
        colors = {
            "ok": self._theme.success,
            "bad": self._theme.error,
            "warn": self._theme.warning,
            "info": self._theme.info,
        }
        label.setStyleSheet(f"color: {colors[role]};")


def _theme_for_scheme(scheme: Qt.ColorScheme, app: QApplication) -> ThemeConfig:
    """Theme for an OS color scheme.

    When the platform reports none (``Unknown``) the theme follows the lightness
    of the window palette the user actually sees, so plots and widgets agree.
    """
    if scheme == Qt.ColorScheme.Light:
        return LIGHT_THEME
    if scheme == Qt.ColorScheme.Dark:
        return DARK_THEME
    lightness = app.palette().window().color().lightness()
    return DARK_THEME if lightness < 128 else LIGHT_THEME


class ThemeManager(QObject):
    """Manages application theme switching.

    Sets the Qt color scheme and the matching application palette, and emits
    ``theme_changed`` for widgets and plots to update their styling.
    """

    # Signal emitted when theme changes
    theme_changed = Signal(ThemeConfig)

    def __init__(self, app: QApplication) -> None:
        """Initialize the theme manager.

        Parameters
        ----------
        app : QApplication
            The QApplication instance
        """
        super().__init__()
        self._app = app
        # Follow the OS light/dark scheme until the user picks one (Ctrl+T,
        # the toggle, --theme or a saved choice); no scheme is forced here.
        self._explicit = False
        self._current_theme = _theme_for_scheme(app.styleHints().colorScheme(), app)
        self._app.setPalette(_palette_for(self._current_theme))
        app.styleHints().colorSchemeChanged.connect(self._on_system_scheme_changed)

    @property
    def current_theme(self) -> ThemeConfig:
        """Get the current theme configuration.

        Returns
        -------
        ThemeConfig
            The active theme.
        """
        return self._current_theme

    @property
    def explicit(self) -> bool:
        """True once the user (or --theme / saved settings) chose a theme.

        Returns
        -------
        bool
            True if a theme was set explicitly, False while following the OS
            color scheme.
        """
        return self._explicit

    def _on_system_scheme_changed(self, scheme: Qt.ColorScheme) -> None:
        """Follow an OS light/dark switch unless a theme was chosen explicitly."""
        # Qt rebuilds the platform palette on a scheme change (also one we
        # triggered in set_theme), so re-assert ours before anything else.
        if self._explicit:
            self._app.setPalette(_palette_for(self._current_theme))
            return
        theme = _theme_for_scheme(scheme, self._app)
        if theme is not self._current_theme:
            self._current_theme = theme
            self._app.setPalette(_palette_for(theme))
            self.theme_changed.emit(theme)

    def set_theme(self, name: str) -> None:
        """Set the application theme.

        Parameters
        ----------
        name : {"light", "dark"}
            Theme name.

        Raises
        ------
        ValueError
            If ``name`` is neither ``"light"`` nor ``"dark"``.
        """
        if name not in ("light", "dark"):
            raise ValueError(f"unknown theme {name!r}; expected 'light' or 'dark'")
        self._explicit = True
        if name == "light":
            self._current_theme = LIGHT_THEME
            self._app.styleHints().setColorScheme(Qt.ColorScheme.Light)
        else:
            self._current_theme = DARK_THEME
            self._app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
        self._app.setPalette(_palette_for(self._current_theme))

        self.theme_changed.emit(self._current_theme)

    def get_theme(self) -> ThemeConfig:
        """Get the current theme configuration.

        Returns
        -------
        ThemeConfig
            The current theme configuration.
        """
        return self._current_theme

    def toggle(self) -> None:
        """Toggle between light and dark themes."""
        if self._current_theme.name == "dark":
            self.set_theme("light")
        else:
            self.set_theme("dark")
