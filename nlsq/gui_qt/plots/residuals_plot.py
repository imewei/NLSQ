"""
NLSQ Qt GUI Residuals Plot Widget

This widget displays residuals analysis plots including scatter plot
and residuals vs fitted values using pyqtgraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ResidualsPlotWidget"]


class ResidualsPlotWidget(QWidget):
    """Widget for displaying residuals analysis plots.

    Provides:
    - Residuals vs X scatter plot
    - Residuals vs Fitted values
    - Standardized residuals
    - Zero reference line
    - GPU-accelerated rendering via pyqtgraph
    """

    # Downsampling threshold
    DOWNSAMPLE_THRESHOLD = 50000

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the residuals plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._x: NDArray | None = None
        self._residuals: NDArray | None = None
        self._fitted: NDArray | None = None
        self._std_residuals: NDArray | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Plot type selector
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Plot Type:"))

        self._plot_type_combo = QComboBox()
        self._plot_type_combo.addItems(
            [
                "Residuals vs X",
                "Residuals vs Fitted",
                "Standardized Residuals vs X",
                "Standardized Residuals vs Fitted",
            ]
        )
        self._plot_type_combo.currentIndexChanged.connect(self._update_plot)
        selector_row.addWidget(self._plot_type_combo)
        selector_row.addStretch()

        layout.addLayout(selector_row)

        # Create plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left", "Residuals")
        self._plot_widget.setLabel("bottom", "X")

        # Enable mouse interactions
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.enableAutoRange()

        layout.addWidget(self._plot_widget)

        # Initialize plot items
        self._scatter: pg.ScatterPlotItem | None = None
        self._zero_line: pg.InfiniteLine | None = None

    def set_residuals(
        self,
        x: NDArray,
        residuals: NDArray,
        fitted: NDArray | None = None,
    ) -> None:
        """Set the residuals data.

        Args:
            x: X data array
            residuals: Residuals (y - y_fit)
            fitted: Fitted values (optional, for residuals vs fitted)
        """
        self._x = np.asarray(x)
        self._residuals = np.asarray(residuals)
        self._fitted = np.asarray(fitted) if fitted is not None else None

        # Compute standardized residuals
        std = np.std(self._residuals)
        if std > 0:
            self._std_residuals = self._residuals / std
        else:
            self._std_residuals = self._residuals

        self._update_plot()

    def _update_plot(self) -> None:
        """Update the plot based on selected type."""
        if self._residuals is None or self._x is None:
            return

        # Clear existing items
        self._plot_widget.clear()

        # Get plot type
        plot_type = self._plot_type_combo.currentIndex()

        # Determine x and y data based on plot type
        if plot_type == 0:  # Residuals vs X
            plot_x = self._x
            plot_y = self._residuals
            x_label = "X"
            y_label = "Residuals"
        elif plot_type == 1:  # Residuals vs Fitted
            plot_x = self._fitted if self._fitted is not None else self._x
            plot_y = self._residuals
            x_label = "Fitted Values"
            y_label = "Residuals"
        elif plot_type == 2:  # Standardized Residuals vs X
            plot_x = self._x
            plot_y = self._std_residuals
            x_label = "X"
            y_label = "Standardized Residuals"
        else:  # Standardized Residuals vs Fitted
            plot_x = self._fitted if self._fitted is not None else self._x
            plot_y = self._std_residuals
            x_label = "Fitted Values"
            y_label = "Standardized Residuals"

        # Downsample if needed
        plot_x, plot_y = self._downsample(plot_x, plot_y)

        # Add zero reference line
        self._zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=(128, 128, 128), width=1, style=pg.QtCore.Qt.DashLine),
        )
        self._plot_widget.addItem(self._zero_line)

        # Create scatter plot with color coding
        # Green for small residuals, red for large
        colors = self._get_residual_colors(plot_y)

        self._scatter = pg.ScatterPlotItem(
            x=plot_x,
            y=plot_y,
            pen=pg.mkPen(None),
            brush=colors,
            size=6,
            symbol="o",
        )
        self._plot_widget.addItem(self._scatter)

        # Update labels
        self._plot_widget.setLabel("bottom", x_label)
        self._plot_widget.setLabel("left", y_label)

        # Auto-range
        self._plot_widget.autoRange()

    def _get_residual_colors(self, residuals: NDArray) -> list:
        """Get colors for residuals based on magnitude.

        Args:
            residuals: Residual values

        Returns:
            List of QBrush objects
        """
        # Normalize by standard deviation or max
        std = np.std(residuals)
        if std > 0:
            normalized = np.abs(residuals) / (2 * std)
        else:
            max_val = np.max(np.abs(residuals))
            normalized = (
                np.abs(residuals) / max_val if max_val > 0 else np.zeros_like(residuals)
            )

        # Clip to [0, 1]
        normalized = np.clip(normalized, 0, 1)

        # Create color gradient: green (small) -> yellow -> red (large)
        colors = []
        for n in normalized:
            if n < 0.5:
                # Green to yellow
                r = int(255 * (2 * n))
                g = 200
                b = 50
            else:
                # Yellow to red
                r = 255
                g = int(200 * (2 * (1 - n)))
                b = 50

            colors.append(pg.mkBrush(r, g, b, 150))

        return colors

    def _downsample(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Downsample data if it exceeds the threshold.

        Args:
            x: X data array
            y: Y data array

        Returns:
            Tuple of (downsampled_x, downsampled_y)
        """
        n = len(x)
        if n <= self.DOWNSAMPLE_THRESHOLD:
            return x, y

        # Calculate stride
        stride = n // self.DOWNSAMPLE_THRESHOLD

        return x[::stride], y[::stride]

    def get_statistics(self) -> dict:
        """Get residuals statistics.

        Returns:
            Dictionary with residual statistics
        """
        if self._residuals is None:
            return {}

        return {
            "mean": float(np.mean(self._residuals)),
            "std": float(np.std(self._residuals)),
            "min": float(np.min(self._residuals)),
            "max": float(np.max(self._residuals)),
            "median": float(np.median(self._residuals)),
        }

    def clear(self) -> None:
        """Clear the plot."""
        self._plot_widget.clear()
        self._x = None
        self._residuals = None
        self._fitted = None
        self._std_residuals = None
        self._scatter = None
        self._zero_line = None

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        if theme.is_dark:
            self._plot_widget.setBackground("#1e1e1e")
            for axis in ["left", "bottom"]:
                axis_item = self._plot_widget.getAxis(axis)
                if axis_item is not None:
                    axis_item.setTextPen(pg.mkPen(color="w"))
        else:
            self._plot_widget.setBackground("w")
            for axis in ["left", "bottom"]:
                axis_item = self._plot_widget.getAxis(axis)
                if axis_item is not None:
                    axis_item.setTextPen(pg.mkPen(color="k"))

    def export_image(self, path: str) -> None:
        """Export the plot as an image.

        Args:
            path: Output file path
        """
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self._plot_widget.plotItem)
        exporter.export(path)
