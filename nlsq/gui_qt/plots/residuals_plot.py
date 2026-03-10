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
        self._confidence_interval: NDArray | None = None

        from nlsq.gui_qt.theme import DARK_THEME

        self._theme: ThemeConfig = DARK_THEME

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

        # Create plot widget — background from default theme
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground(self._theme.plot_background)
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
        confidence_interval: NDArray | None = None,
    ) -> None:
        """Set the residuals data.

        Args:
            x: X data array
            residuals: Residuals (y - y_fit)
            fitted: Fitted values (optional, for residuals vs fitted)
            confidence_interval: Half-width of confidence bands (at x points)
        """
        self._x = np.asarray(x)
        self._residuals = np.asarray(residuals)
        self._fitted = np.asarray(fitted) if fitted is not None else None
        self._confidence_interval = (
            np.asarray(confidence_interval) if confidence_interval is not None else None
        )

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

        # Draw confidence bands (only for Residuals vs X)
        if plot_type == 0 and self._confidence_interval is not None:
            # We assume confidence interval is symmetric around zero for residuals
            band_upper = self._confidence_interval
            band_lower = -self._confidence_interval

            # Downsample bands to match x
            # Note: We reuse plot_x from above which is downsampled self._x
            # So we must downsample the CI array using the same logic/slice
            # But self._downsample calculates stride internally.
            # To be safe and consistent, we should downsample CI using the same internal logic
            # OR pass them together. Simpler to just re-downsample here.
            _, band_upper_ds = self._downsample(self._x, band_upper)
            _, band_lower_ds = self._downsample(self._x, band_lower)

            curve_lower = pg.PlotDataItem(plot_x, band_lower_ds, pen=pg.mkPen(None))
            curve_upper = pg.PlotDataItem(plot_x, band_upper_ds, pen=pg.mkPen(None))

            # Confidence band fill — use fit_line color at very low alpha
            band_color = pg.mkColor(self._theme.fit_line)
            band_color.setAlpha(30)
            fill = pg.FillBetweenItem(
                curve_lower,
                curve_upper,
                brush=pg.mkBrush(band_color),
            )
            self._plot_widget.addItem(curve_lower)
            self._plot_widget.addItem(curve_upper)
            self._plot_widget.addItem(fill)

        # Draw warning lines for standardized residuals — use theme warning color
        if plot_type >= 2:
            for y_val in [-2, 2]:
                line = pg.InfiniteLine(
                    pos=y_val,
                    angle=0,
                    pen=pg.mkPen(
                        color=self._theme.warning,
                        width=1,
                        style=pg.QtCore.Qt.DashLine,
                    ),
                )
                self._plot_widget.addItem(line)

        # Add zero reference line — use muted foreground
        zero_color = pg.mkColor(self._theme.plot_foreground)
        zero_color.setAlpha(100)
        self._zero_line = pg.InfiniteLine(
            pos=0,
            angle=0,
            pen=pg.mkPen(color=zero_color, width=1, style=pg.QtCore.Qt.DashLine),
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

        Colors transition from the theme's 'stat_good' (small residuals) through
        'stat_warning' to 'stat_bad' (large residuals).

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

        # Parse theme semantic colors for the gradient endpoints
        c_good = pg.mkColor(self._theme.stat_good)
        c_warn = pg.mkColor(self._theme.stat_warning)
        c_bad = pg.mkColor(self._theme.stat_bad)

        def _lerp(a: int, b: int, t: float) -> int:
            return int(a + (b - a) * t)

        colors = []
        for n in normalized:
            if n < 0.5:
                t = n * 2.0
                r = _lerp(c_good.red(), c_warn.red(), t)
                g = _lerp(c_good.green(), c_warn.green(), t)
                b = _lerp(c_good.blue(), c_warn.blue(), t)
            else:
                t = (n - 0.5) * 2.0
                r = _lerp(c_warn.red(), c_bad.red(), t)
                g = _lerp(c_warn.green(), c_bad.green(), t)
                b = _lerp(c_warn.blue(), c_bad.blue(), t)
            colors.append(pg.mkBrush(r, g, b, 160))

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
        self._theme = theme

        self._plot_widget.setBackground(theme.plot_background)
        axis_pen = pg.mkPen(color=theme.plot_foreground)
        for axis in ["left", "bottom"]:
            axis_item = self._plot_widget.getAxis(axis)
            if axis_item is not None:
                axis_item.setPen(axis_pen)
                axis_item.setTextPen(axis_pen)

        # Re-draw with new theme colors if data is present
        if self._residuals is not None:
            self._update_plot()

    def export_image(self, path: str) -> None:
        """Export the plot as an image.

        Args:
            path: Output file path
        """
        from pyqtgraph.exporters import ImageExporter

        exporter = ImageExporter(self._plot_widget.plotItem)
        exporter.export(path)
