"""
NLSQ Qt GUI Parameter Results Widget

This widget displays the fitted parameter results with uncertainties
and confidence intervals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ParamResultsWidget"]


class ParamResultsWidget(QWidget):
    """Widget for displaying fitted parameter results.

    Provides:
    - Table with parameter names, values, uncertainties
    - Confidence intervals
    - Value formatting
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the parameter results widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Parameter", "Value", "Uncertainty", "95% CI"]
        )

        # Configure header
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        # Configure table
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(self._table)

    def set_results(
        self,
        names: list[str],
        values: list[float],
        uncertainties: list[float],
        confidence_intervals: list[tuple[float, float]] | None = None,
    ) -> None:
        """Set the parameter results.

        Args:
            names: Parameter names
            values: Fitted values
            uncertainties: Standard errors
            confidence_intervals: Optional 95% CI as (lower, upper) tuples
        """
        self._table.setRowCount(len(names))

        for row, name in enumerate(names):
            # Name
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Value
            value = values[row] if row < len(values) else 0.0
            value_item = QTableWidgetItem(f"{value:.6g}")
            value_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 1, value_item)

            # Uncertainty
            uncert = uncertainties[row] if row < len(uncertainties) else 0.0
            uncert_item = QTableWidgetItem(f"\u00b1 {uncert:.6g}")  # ± symbol
            uncert_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 2, uncert_item)

            # Confidence interval
            if confidence_intervals and row < len(confidence_intervals):
                ci = confidence_intervals[row]
                ci_text = f"[{ci[0]:.6g}, {ci[1]:.6g}]"
            else:
                ci_text = "-"
            ci_item = QTableWidgetItem(ci_text)
            ci_item.setTextAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            self._table.setItem(row, 3, ci_item)

    def clear(self) -> None:
        """Clear the table."""
        self._table.setRowCount(0)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        pass  # Theme is applied globally via Qt color scheme
