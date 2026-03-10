"""
NLSQ Qt GUI Fit Plot Widget

This widget displays the data points, fitted curve, and optional
confidence bands using pyqtgraph for GPU-accelerated rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["FitPlotWidget"]


class FitPlotWidget(QWidget):
    """Widget for displaying fit results with data and curve.

    Provides:
    - Scatter plot of original data points
    - Fitted curve line
    - Optional confidence bands (±1σ, ±2σ)
    - GPU-accelerated rendering via pyqtgraph
    - Automatic downsampling for large datasets
    """

    # Downsampling threshold
    DOWNSAMPLE_THRESHOLD = 50000

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the fit plot widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._data_x: NDArray | None = None
        self._data_y: NDArray | None = None
        self._fit_x: NDArray | None = None
        self._fit_y: NDArray | None = None
        self._conf_lower: NDArray | None = None
        self._conf_upper: NDArray | None = None

        # Import here to avoid a circular import at module level
        from nlsq.gui_qt.theme import DARK_THEME

        self._theme: ThemeConfig = DARK_THEME

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create plot widget — background set properly in set_theme()
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._theme.plot_background)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setLabel("left", "Y")
        self._plot_widget.setLabel("bottom", "X")

        # Enable mouse interactions
        self._plot_widget.setMouseEnabled(x=True, y=True)
        self._plot_widget.enableAutoRange()

        layout.addWidget(self._plot_widget)

        # Options row
        options_row = QHBoxLayout()

        self._show_data_cb = QCheckBox("Show Data")
        self._show_data_cb.setChecked(True)
        self._show_data_cb.stateChanged.connect(self._update_visibility)
        options_row.addWidget(self._show_data_cb)

        self._show_fit_cb = QCheckBox("Show Fit")
        self._show_fit_cb.setChecked(True)
        self._show_fit_cb.stateChanged.connect(self._update_visibility)
        options_row.addWidget(self._show_fit_cb)

        self._show_conf_cb = QCheckBox("Show Confidence Band")
        self._show_conf_cb.setChecked(True)
        self._show_conf_cb.stateChanged.connect(self._update_visibility)
        options_row.addWidget(self._show_conf_cb)

        options_row.addStretch()
        layout.addLayout(options_row)

        # Initialize plot items (will be populated when data is set)
        self._data_scatter: pg.ScatterPlotItem | None = None
        self._fit_curve: pg.PlotDataItem | None = None
        self._conf_band_upper: pg.PlotDataItem | None = None
        self._conf_band_lower: pg.PlotDataItem | None = None
        self._conf_fill: pg.FillBetweenItem | None = None

    def set_data(
        self,
        x: NDArray,
        y: NDArray,
        sigma: NDArray | None = None,
    ) -> None:
        """Set the original data points.

        Args:
            x: X data array
            y: Y data array
            sigma: Optional uncertainty array for error bars
        """
        self._data_x = np.asarray(x)
        self._data_y = np.asarray(y)

        # Clear existing data scatter
        if self._data_scatter is not None:
            self._plot_widget.removeItem(self._data_scatter)
            self._data_scatter = None

        # Downsample if needed
        plot_x, plot_y = self._downsample(self._data_x, self._data_y)

        # Create scatter plot — color from theme
        marker_color = pg.mkColor(self._theme.data_marker)
        marker_color.setAlpha(160)
        self._data_scatter = pg.ScatterPlotItem(
            x=plot_x,
            y=plot_y,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(marker_color),
            size=6,
            symbol="o",
        )
        self._plot_widget.addItem(self._data_scatter)
        self._data_scatter.setVisible(self._show_data_cb.isChecked())

        # Auto-range to show all data
        self._plot_widget.autoRange()

    def set_fit(
        self,
        x: NDArray,
        y: NDArray,
        conf_lower: NDArray | None = None,
        conf_upper: NDArray | None = None,
    ) -> None:
        """Set the fitted curve and optional confidence bands.

        Args:
            x: X values for fit curve (dense grid)
            y: Y values for fit curve
            conf_lower: Lower confidence band (y - delta)
            conf_upper: Upper confidence band (y + delta)
        """
        self._fit_x = np.asarray(x)
        self._fit_y = np.asarray(y)

        # Clear existing fit items
        if self._fit_curve is not None:
            self._plot_widget.removeItem(self._fit_curve)
            self._fit_curve = None

        if self._conf_fill is not None:
            self._plot_widget.removeItem(self._conf_fill)
            self._conf_fill = None

        if self._conf_band_upper is not None:
            self._plot_widget.removeItem(self._conf_band_upper)
            self._conf_band_upper = None

        if self._conf_band_lower is not None:
            self._plot_widget.removeItem(self._conf_band_lower)
            self._conf_band_lower = None

        # Create fit curve — color from theme
        self._fit_curve = pg.PlotDataItem(
            x=self._fit_x,
            y=self._fit_y,
            pen=pg.mkPen(color=self._theme.fit_line, width=2),
        )
        self._plot_widget.addItem(self._fit_curve)
        self._fit_curve.setVisible(self._show_fit_cb.isChecked())

        # Create confidence band if provided
        if conf_lower is not None and conf_upper is not None:
            self._conf_lower = np.asarray(conf_lower)
            self._conf_upper = np.asarray(conf_upper)

            # Create band lines (invisible, used for fill)
            self._conf_band_lower = pg.PlotDataItem(
                x=self._fit_x,
                y=self._conf_lower,
                pen=pg.mkPen(None),  # Invisible line
            )
            self._conf_band_upper = pg.PlotDataItem(
                x=self._fit_x,
                y=self._conf_upper,
                pen=pg.mkPen(None),  # Invisible line
            )
            self._plot_widget.addItem(self._conf_band_lower)
            self._plot_widget.addItem(self._conf_band_upper)

            # Create fill between bands — use fit_line color at low opacity
            band_color = pg.mkColor(self._theme.fit_line)
            band_color.setAlpha(50)
            self._conf_fill = pg.FillBetweenItem(
                self._conf_band_lower,
                self._conf_band_upper,
                brush=pg.mkBrush(band_color),
            )
            self._plot_widget.addItem(self._conf_fill)

            # Set visibility
            show_conf = self._show_conf_cb.isChecked()
            self._conf_band_lower.setVisible(show_conf)
            self._conf_band_upper.setVisible(show_conf)
            self._conf_fill.setVisible(show_conf)

    def _downsample(
        self,
        x: NDArray,
        y: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Downsample data if it exceeds the threshold.

        Uses a simple stride-based downsampling for performance.

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

    def _update_visibility(self) -> None:
        """Update visibility of plot elements based on checkboxes."""
        if self._data_scatter is not None:
            self._data_scatter.setVisible(self._show_data_cb.isChecked())

        if self._fit_curve is not None:
            self._fit_curve.setVisible(self._show_fit_cb.isChecked())

        show_conf = self._show_conf_cb.isChecked()
        if self._conf_fill is not None:
            self._conf_fill.setVisible(show_conf)
        if self._conf_band_lower is not None:
            self._conf_band_lower.setVisible(show_conf)
        if self._conf_band_upper is not None:
            self._conf_band_upper.setVisible(show_conf)

    def clear(self) -> None:
        """Clear all plot data."""
        self._plot_widget.clear()
        self._data_scatter = None
        self._fit_curve = None
        self._conf_band_upper = None
        self._conf_band_lower = None
        self._conf_fill = None
        self._data_x = None
        self._data_y = None
        self._fit_x = None
        self._fit_y = None
        self._conf_lower = None
        self._conf_upper = None

    def set_labels(self, x_label: str, y_label: str) -> None:
        """Set axis labels.

        Args:
            x_label: X axis label
            y_label: Y axis label
        """
        self._plot_widget.setLabel("bottom", x_label)
        self._plot_widget.setLabel("left", y_label)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        self._theme = theme

        # Plot background and axes
        self._plot_widget.setBackground(theme.plot_background)
        axis_pen = pg.mkPen(color=theme.plot_foreground)
        for axis in ["left", "bottom"]:
            axis_item = self._plot_widget.getAxis(axis)
            if axis_item is not None:
                axis_item.setPen(axis_pen)
                axis_item.setTextPen(axis_pen)

        # Update existing scatter if present
        if self._data_scatter is not None:
            marker_color = pg.mkColor(theme.data_marker)
            marker_color.setAlpha(160)
            self._data_scatter.setBrush(pg.mkBrush(marker_color))

        # Update existing fit curve if present
        if self._fit_curve is not None:
            self._fit_curve.setPen(pg.mkPen(color=theme.fit_line, width=2))

        # Update confidence band fill if present
        if self._conf_fill is not None:
            band_color = pg.mkColor(theme.fit_line)
            band_color.setAlpha(50)
            self._conf_fill.setBrush(pg.mkBrush(band_color))

    def export_image(self, path: str) -> None:
        """Export the plot as an image.

        Args:
            path: Output file path (PNG, JPG, SVG)
        """
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self._plot_widget.plotItem)
        exporter.export(path)
