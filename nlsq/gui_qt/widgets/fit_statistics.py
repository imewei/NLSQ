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
        self.setStyleSheet("padding: 8px; border-radius: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet("font-size: 11px; color: gray;")
        layout.addWidget(self._title_label)

        self._value_label = QLabel("-")
        self._value_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self._value_label)

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
            f"font-size: 18px; font-weight: bold; color: {color};"
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
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QGridLayout(self)
        layout.setSpacing(8)

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
        # R² with color coding
        self._r2_card.set_value(f"{r_squared:.6f}")
        if r_squared >= 0.99:
            self._r2_card.set_value_color("#4CAF50")  # Green
        elif r_squared >= 0.95:
            self._r2_card.set_value_color("#8BC34A")  # Light green
        elif r_squared >= 0.9:
            self._r2_card.set_value_color("#FF9800")  # Orange
        else:
            self._r2_card.set_value_color("#f44336")  # Red

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
            self._status_card.set_value_color("#4CAF50")
        else:
            self._status_card.set_value("Did not converge")
            self._status_card.set_value_color("#f44336")

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
        pass  # Theme is applied globally via Qt color scheme
