"""
NLSQ Qt GUI Base Plot Widget

This module provides a base class for pyqtgraph-based plot widgets with
theme support and common configuration for large dataset handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["BasePlotWidget"]


class BasePlotWidget(QWidget):
    """Base class for pyqtgraph-based plot widgets.

    Provides common functionality for theme support, downsampling configuration,
    and layout management. All plot widgets should inherit from this class.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the base plot widget.

        Args:
            parent: Optional parent widget
        """
        super().__init__(parent)
        self._setup_ui()
        self._configure_plot()

    def _setup_ui(self) -> None:
        """Set up the widget layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create the plot widget
        self._plot_widget = pg.PlotWidget()
        layout.addWidget(self._plot_widget)

        # Get plot item for configuration
        self._plot_item = self._plot_widget.getPlotItem()

    def _configure_plot(self) -> None:
        """Configure common plot settings for performance."""
        # Enable automatic downsampling for large datasets
        # Uses 'peak' mode to preserve visual features (min/max per pixel)
        self._plot_item.setDownsampling(auto=True, mode="peak")

        # Only render data that's visible in the current view
        self._plot_item.setClipToView(True)

        # Enable mouse interaction
        self._plot_item.setMouseEnabled(x=True, y=True)

        # Show grid by default
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)

    @property
    def plot_widget(self) -> pg.PlotWidget:
        """Get the underlying pyqtgraph PlotWidget."""
        return self._plot_widget

    @property
    def plot_item(self) -> pg.PlotItem:
        """Get the PlotItem for direct configuration."""
        return self._plot_item

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme colors to the plot.

        Args:
            theme: ThemeConfig with color definitions
        """
        # Set background color
        self._plot_widget.setBackground(theme.plot_background)

        # Configure axis colors
        axis_pen = pg.mkPen(color=theme.plot_foreground)
        for axis_name in ["bottom", "left", "top", "right"]:
            axis = self._plot_item.getAxis(axis_name)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)

        # Update grid color
        self._plot_item.getAxis("bottom").setGrid(128)  # Grid alpha
        self._plot_item.getAxis("left").setGrid(128)

    def clear(self) -> None:
        """Clear all plot data."""
        self._plot_item.clear()

    def set_labels(
        self,
        title: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
    ) -> None:
        """Set plot labels.

        Args:
            title: Optional plot title
            x_label: Optional x-axis label
            y_label: Optional y-axis label
        """
        if title is not None:
            self._plot_item.setTitle(title)
        if x_label is not None:
            self._plot_item.setLabel("bottom", x_label)
        if y_label is not None:
            self._plot_item.setLabel("left", y_label)

    def auto_range(self) -> None:
        """Auto-scale the plot to fit all data."""
        self._plot_item.autoRange()
