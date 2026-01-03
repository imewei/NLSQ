"""
NLSQ Qt GUI Histogram Plot Widget

This widget displays histogram of residuals with optional
normal distribution overlay using pyqtgraph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["HistogramPlotWidget"]


class HistogramPlotWidget(QWidget):
    """Widget for displaying histogram of residuals.

    Provides:
    - Histogram of residuals distribution
    - Configurable number of bins
    - Optional normal distribution overlay
    - GPU-accelerated rendering via pyqtgraph
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the histogram plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._data: NDArray | None = None
        self._n_bins = 30

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Options row
        options_row = QHBoxLayout()

        options_row.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 100)
        self._bins_spin.setValue(30)
        self._bins_spin.valueChanged.connect(self._on_bins_changed)
        options_row.addWidget(self._bins_spin)

        self._show_normal_cb = QCheckBox("Show Normal Distribution")
        self._show_normal_cb.setChecked(True)
        self._show_normal_cb.stateChanged.connect(self._update_plot)
        options_row.addWidget(self._show_normal_cb)

        options_row.addStretch()
        layout.addLayout(options_row)

        # Create plot widget
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left", "Frequency")
        self._plot_widget.setLabel("bottom", "Residuals")

        # Enable mouse interactions
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.enableAutoRange()

        layout.addWidget(self._plot_widget)

        # Initialize plot items
        self._bar_graph: pg.BarGraphItem | None = None
        self._normal_curve: pg.PlotDataItem | None = None

    def set_data(self, data: NDArray) -> None:
        """Set the data for histogram.

        Args:
            data: Data array (typically residuals)
        """
        self._data = np.asarray(data)
        self._update_plot()

    def _on_bins_changed(self, value: int) -> None:
        """Handle bins value change.

        Args:
            value: New number of bins
        """
        self._n_bins = value
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the histogram plot."""
        if self._data is None:
            return

        # Clear existing items
        self._plot_widget.clear()

        # Compute histogram
        counts, bin_edges = np.histogram(self._data, bins=self._n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Create bar graph
        self._bar_graph = pg.BarGraphItem(
            x=bin_centers,
            height=counts,
            width=bin_width * 0.9,
            brush=pg.mkBrush(33, 150, 243, 150),  # Blue with alpha
            pen=pg.mkPen(33, 150, 243, 200),
        )
        self._plot_widget.addItem(self._bar_graph)

        # Add normal distribution overlay if enabled
        if self._show_normal_cb.isChecked():
            self._add_normal_overlay(bin_edges, counts)

        # Auto-range
        self._plot_widget.autoRange()

    def _add_normal_overlay(
        self,
        bin_edges: NDArray,
        counts: NDArray,
    ) -> None:
        """Add normal distribution overlay to the plot.

        Args:
            bin_edges: Histogram bin edges
            counts: Histogram counts
        """
        if self._data is None:
            return

        # Compute mean and std
        mean = np.mean(self._data)
        std = np.std(self._data)

        if std <= 0:
            return

        # Create x values for curve
        x_min = np.min(self._data)
        x_max = np.max(self._data)
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 200)

        # Normal distribution PDF
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        # Scale to match histogram
        bin_width = bin_edges[1] - bin_edges[0]
        n_samples = len(self._data)
        y_scaled = y * n_samples * bin_width

        # Create curve
        self._normal_curve = pg.PlotDataItem(
            x=x,
            y=y_scaled,
            pen=pg.mkPen(color=(244, 67, 54), width=2),  # Red line
        )
        self._plot_widget.addItem(self._normal_curve)

    def get_statistics(self) -> dict:
        """Get histogram statistics.

        Returns:
            Dictionary with statistics
        """
        if self._data is None:
            return {}

        from scipy import stats

        # Normality test
        try:
            _, p_value = stats.shapiro(
                self._data[:5000] if len(self._data) > 5000 else self._data
            )
        except Exception:
            p_value = None

        return {
            "mean": float(np.mean(self._data)),
            "std": float(np.std(self._data)),
            "skewness": float(stats.skew(self._data)),
            "kurtosis": float(stats.kurtosis(self._data)),
            "normality_p": float(p_value) if p_value is not None else None,
        }

    def clear(self) -> None:
        """Clear the plot."""
        self._plot_widget.clear()
        self._data = None
        self._bar_graph = None
        self._normal_curve = None

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
