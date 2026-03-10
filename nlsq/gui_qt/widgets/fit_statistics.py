"""
NLSQ Qt GUI Fit Statistics Widget

This widget displays fit quality statistics like R², RMSE, MAE, AIC, BIC.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["FitStatisticsWidget"]


class StatCard(QFrame):
    """A card widget for displaying a single statistic."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        """Initialize the stat card.

        Args:
            title: Card title
            parent: Parent widget
        """
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)

        self._title_label = QLabel(title)
        layout.addWidget(self._title_label)

        self._value_label = QLabel("-")
        layout.addWidget(self._value_label)

        # Apply default (dark) styling before any theme is propagated
        self._apply_card_style(
            surface_variant="#383838",
            border="#4a4a4a",
            text_secondary="#b0b0b0",
            text_primary="#ffffff",
        )

    def _apply_card_style(
        self,
        surface_variant: str,
        border: str,
        text_secondary: str,
        text_primary: str,
    ) -> None:
        """Apply base card styling from explicit color values.

        Args:
            surface_variant: Card background color
            border: Border color
            text_secondary: Title label color
            text_primary: Value label base color (overridden by set_value_color)
        """
        self.setStyleSheet(
            f"StatCard {{ background-color: {surface_variant}; "
            f"border: 1px solid {border}; border-radius: 6px; }}"
        )
        self._title_label.setStyleSheet(
            f"font-size: 10px; font-weight: 600; letter-spacing: 0.5px; "
            f"color: {text_secondary}; border: none; background: transparent;"
        )
        self._value_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {text_primary}; "
            "border: none; background: transparent;"
        )

    def set_value(self, value: str) -> None:
        """Set the displayed value.

        Args:
            value: Value string to display
        """
        self._value_label.setText(value)

    def set_value_color(self, color: str) -> None:
        """Set the value text color.

        Args:
            color: CSS color string
        """
        self._value_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; color: {color}; "
            "border: none; background: transparent;"
        )

    def apply_theme(self, theme: ThemeConfig) -> None:
        """Re-apply base card styling from a ThemeConfig.

        Args:
            theme: Theme configuration
        """
        self._apply_card_style(
            surface_variant=theme.surface_variant,
            border=theme.border,
            text_secondary=theme.text_secondary,
            text_primary=theme.text_primary,
        )


class FitStatisticsWidget(QWidget):
    """Widget for displaying fit quality statistics.

    Provides:
    - R² (coefficient of determination)
    - RMSE (root mean square error)
    - MAE (mean absolute error)
    - AIC (Akaike information criterion)
    - BIC (Bayesian information criterion)
    - Convergence status
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the fit statistics widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)

        from nlsq.gui_qt.theme import DARK_THEME

        self._theme: ThemeConfig = DARK_THEME
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QGridLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)

        # Uniform column stretch so cards share width equally
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        # Create stat cards
        self._r2_card = StatCard("R² (Coefficient of Determination)")
        self._rmse_card = StatCard("RMSE (Root Mean Square Error)")
        self._mae_card = StatCard("MAE (Mean Absolute Error)")
        self._aic_card = StatCard("AIC (Akaike Information Criterion)")
        self._bic_card = StatCard("BIC (Bayesian Information Criterion)")
        self._iter_card = StatCard("Iterations")
        self._status_card = StatCard("Convergence")

        # Arrange in grid
        layout.addWidget(self._r2_card, 0, 0)
        layout.addWidget(self._rmse_card, 0, 1)
        layout.addWidget(self._mae_card, 0, 2)
        layout.addWidget(self._aic_card, 1, 0)
        layout.addWidget(self._bic_card, 1, 1)
        layout.addWidget(self._iter_card, 1, 2)
        layout.addWidget(self._status_card, 2, 0, 1, 3)

    def set_statistics(
        self,
        r_squared: float,
        rmse: float,
        mae: float,
        aic: float,
        bic: float,
        n_iterations: int,
        converged: bool,
    ) -> None:
        """Set the statistics values.

        Args:
            r_squared: Coefficient of determination
            rmse: Root mean square error
            mae: Mean absolute error
            aic: Akaike information criterion
            bic: Bayesian information criterion
            n_iterations: Number of iterations
            converged: Whether the fit converged
        """
        # R² with theme-aware semantic color coding
        self._r2_card.set_value(f"{r_squared:.6f}")
        if r_squared >= 0.99:
            self._r2_card.set_value_color(self._theme.stat_good)
        elif r_squared >= 0.95:
            # Interpolate between stat_good and stat_warning at 95%
            self._r2_card.set_value_color(self._theme.stat_good)
        elif r_squared >= 0.9:
            self._r2_card.set_value_color(self._theme.stat_warning)
        else:
            self._r2_card.set_value_color(self._theme.stat_bad)

        # RMSE
        self._rmse_card.set_value(f"{rmse:.6g}")

        # MAE
        self._mae_card.set_value(f"{mae:.6g}")

        # AIC
        self._aic_card.set_value(f"{aic:.2f}")

        # BIC
        self._bic_card.set_value(f"{bic:.2f}")

        # Iterations
        self._iter_card.set_value(str(n_iterations))

        # Convergence status
        if converged:
            self._status_card.set_value("Converged")
            self._status_card.set_value_color(self._theme.stat_good)
        else:
            self._status_card.set_value("Did not converge")
            self._status_card.set_value_color(self._theme.stat_bad)

    def clear(self) -> None:
        """Clear all statistics."""
        self._r2_card.set_value("-")
        self._rmse_card.set_value("-")
        self._mae_card.set_value("-")
        self._aic_card.set_value("-")
        self._bic_card.set_value("-")
        self._iter_card.set_value("-")
        self._status_card.set_value("-")

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        self._theme = theme

        # Propagate base card styling (background, border, title color)
        for card in (
            self._r2_card,
            self._rmse_card,
            self._mae_card,
            self._aic_card,
            self._bic_card,
            self._iter_card,
            self._status_card,
        ):
            card.apply_theme(theme)
