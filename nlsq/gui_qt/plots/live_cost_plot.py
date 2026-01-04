"""
NLSQ Qt GUI Live Cost Plot Widget

This widget displays a real-time plot of the cost function during fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["LiveCostPlotWidget"]


class LiveCostPlotWidget(QWidget):
    """Real-time cost function plot during fitting.

    Provides:
    - Line plot of cost vs iteration
    - Auto-scaling axes
    - Log scale option for y-axis
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the live cost plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._iterations: list[int] = []
        self._costs: list[float] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("left", "Cost")
        self._plot_widget.setLabel("bottom", "Iteration")
        self._plot_widget.setTitle("Cost Function Progress")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # Enable downsampling for performance
        self._plot_widget.setDownsampling(auto=True, mode="peak")
        self._plot_widget.setClipToView(True)

        # Create the line plot item
        self._line = self._plot_widget.plot(
            [],
            [],
            pen=pg.mkPen(color="#2196F3", width=2),
            symbol="o",
            symbolSize=5,
            symbolBrush="#2196F3",
        )

        layout.addWidget(self._plot_widget)

    def add_point(self, iteration: int, cost: float) -> None:
        """Add a data point to the plot.

        Args:
            iteration: Iteration number
            cost: Cost function value
        """
        self._iterations.append(iteration)
        self._costs.append(cost)

        # Update plot
        self._line.setData(self._iterations, self._costs)

        # Auto-range on new data
        self._plot_widget.enableAutoRange()

    def reset(self) -> None:
        """Reset the plot to empty state."""
        self._iterations = []
        self._costs = []
        self._line.setData([], [])

    def set_log_scale(self, enabled: bool) -> None:
        """Set log scale for y-axis.

        Args:
            enabled: Whether to use log scale
        """
        self._plot_widget.setLogMode(x=False, y=enabled)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        # Update plot colors based on theme
        self._plot_widget.setBackground(theme.plot_background)

        # Update line color
        self._line.setPen(pg.mkPen(color=theme.data_marker, width=2))
        self._line.setSymbolBrush(theme.data_marker)

        # Update axis colors
        axis_pen = pg.mkPen(color=theme.plot_foreground)
        for axis_name in ["left", "bottom"]:
            axis = self._plot_widget.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)

        # Update grid color
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
