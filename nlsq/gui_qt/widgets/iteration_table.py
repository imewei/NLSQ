"""
NLSQ Qt GUI Iteration Table Widget

This widget displays iteration history during fitting.
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

__all__ = ["IterationTableWidget"]


class IterationTableWidget(QWidget):
    """Widget for displaying iteration history during fitting.

    Provides:
    - Table with iteration number, cost, step norm, gradient norm
    - Auto-scroll to latest iteration
    - Maximum row limit for performance
    """

    MAX_ROWS = 500  # Limit rows for performance

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the iteration table widget.

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
            ["Iteration", "Cost", "Step Norm", "Gradient Norm"]
        )

        # Configure header
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        # Configure table
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(self._table)

    def add_iteration(
        self,
        iteration: int,
        cost: float,
        step_norm: float | None = None,
        gradient_norm: float | None = None,
    ) -> None:
        """Add an iteration to the table.

        Args:
            iteration: Iteration number
            cost: Cost function value
            step_norm: Optional step norm
            gradient_norm: Optional gradient norm
        """
        # Remove oldest rows if at limit
        if self._table.rowCount() >= self.MAX_ROWS:
            self._table.removeRow(0)

        # Add new row
        row = self._table.rowCount()
        self._table.insertRow(row)

        # Iteration
        iter_item = QTableWidgetItem(str(iteration))
        iter_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, 0, iter_item)

        # Cost
        cost_item = QTableWidgetItem(f"{cost:.6g}")
        cost_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._table.setItem(row, 1, cost_item)

        # Step norm
        step_text = f"{step_norm:.6g}" if step_norm is not None else "-"
        step_item = QTableWidgetItem(step_text)
        step_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._table.setItem(row, 2, step_item)

        # Gradient norm
        grad_text = f"{gradient_norm:.6g}" if gradient_norm is not None else "-"
        grad_item = QTableWidgetItem(grad_text)
        grad_item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self._table.setItem(row, 3, grad_item)

        # Auto-scroll to bottom
        self._table.scrollToBottom()

    def clear(self) -> None:
        """Clear the table."""
        self._table.setRowCount(0)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        pass  # Theme is applied globally via Qt color scheme
