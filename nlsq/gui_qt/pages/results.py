"""
NLSQ Qt GUI Results Page

This page displays the fitted parameters, statistical metrics,
and interactive visualizations of the fit quality.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy import stats

from nlsq.gui_qt.plots import decimation_note
from nlsq.gui_qt.plots.fit_plot import FitPlotWidget
from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget
from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget
from nlsq.gui_qt.session_state import n_points
from nlsq.gui_qt.widgets.fit_statistics import FitStatisticsWidget
from nlsq.gui_qt.widgets.param_results import ParamResultsWidget

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ResultsPage"]


def _dense_grid(xdata: np.ndarray, n: int = 500) -> np.ndarray:
    """Evaluation grid for the fit curve, confined to the data domain.

    No padding: models such as power laws or logs are undefined just outside
    ``[min(x), max(x)]`` and would produce NaN gaps / extrapolation.
    """
    return np.linspace(np.min(xdata), np.max(xdata), n)


def _information_criteria(
    result: Any, ss_res: float, n: int, k: int
) -> tuple[float, float]:
    """AIC / BIC, using ``CurveFitResult`` as the single source of truth.

    ``CurveFitResult`` uses the sigma-weighted chi-square when the fit had
    ``sigma``. The fallback (no result statistics available) is unweighted:
    ``AIC = 2k + n ln(RSS/n)``, ``BIC = k ln(n) + n ln(RSS/n)``; NaN when
    RSS is non-finite or <= 0 (undefined), matching ``CurveFitResult``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(result.aic), float(result.bic)
    except (AttributeError, TypeError, ValueError):
        pass
    if not np.isfinite(ss_res) or ss_res <= 0 or n <= 0:
        return float("nan"), float("nan")
    base = n * np.log(ss_res / n)
    return float(2 * k + base), float(k * np.log(n) + base)


class ResultsPage(QWidget):
    """Page for viewing fit results.

    Provides:
    - Parameter results table with values, uncertainties, confidence intervals
    - Fit statistics (R², RMSE, MAE, AIC, BIC)
    - Interactive fit plot with data, fit curve, and confidence band
    - Residuals scatter plot
    - Residuals histogram
    """

    def __init__(self, app_state: AppState) -> None:
        """Initialize the results page.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._has_results = False
        self._last_result = None
        self._last_statistics: dict[str, float] = {}  # finite R^2 / RMSE for export
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header row with title and export button — 24px bold, consistent across pages
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        title = QLabel("Results")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        self._export_btn = QPushButton("Export Figure...")
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._export_btn.setEnabled(False)
        header_layout.addWidget(self._export_btn)

        layout.addLayout(header_layout)

        # Main content - splitter with left/right panels
        splitter = QSplitter()

        # Left panel - Parameters and statistics (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

        # Parameter results table
        param_group = QGroupBox("Fitted Parameters")
        param_layout = QVBoxLayout(param_group)
        self._param_results = ParamResultsWidget()
        param_layout.addWidget(self._param_results)
        left_layout.addWidget(param_group)

        # Fit statistics
        stats_group = QGroupBox("Fit Statistics")
        stats_layout = QVBoxLayout(stats_group)
        self._fit_statistics = FitStatisticsWidget()
        stats_layout.addWidget(self._fit_statistics)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # Right panel - Plots in tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        self._plot_tabs = QTabWidget()

        # Fit plot tab
        fit_tab = QWidget()
        fit_layout = QVBoxLayout(fit_tab)
        fit_layout.setContentsMargins(4, 4, 4, 4)
        self._fit_plot = FitPlotWidget()
        fit_layout.addWidget(self._fit_plot)
        self._plot_tabs.addTab(fit_tab, "Fit")

        # Residuals plot tab
        residuals_tab = QWidget()
        residuals_layout = QVBoxLayout(residuals_tab)
        residuals_layout.setContentsMargins(4, 4, 4, 4)
        self._residuals_plot = ResidualsPlotWidget()
        residuals_layout.addWidget(self._residuals_plot)
        self._plot_tabs.addTab(residuals_tab, "Residuals")

        # Histogram tab
        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)
        histogram_layout.setContentsMargins(4, 4, 4, 4)
        self._histogram_plot = HistogramPlotWidget()
        histogram_layout.addWidget(self._histogram_plot)
        self._plot_tabs.addTab(histogram_tab, "Histogram")

        right_layout.addWidget(self._plot_tabs)
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter, 1)

        # Status row
        status_row = QHBoxLayout()
        self._status_label = QLabel("No fit results available")
        self._status_label.setStyleSheet("color: gray;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

    def _connect_signals(self) -> None:
        """Connect to app state signals."""
        self._app_state.fit_completed.connect(self.update_results)

    def set_app_state(self, state: AppState) -> None:
        """Set the application state.

        Args:
            state: Application state manager
        """
        self._app_state = state

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this page.

        Args:
            theme: Theme configuration
        """
        self._param_results.set_theme(theme)
        self._fit_statistics.set_theme(theme)
        self._fit_plot.set_theme(theme)
        self._residuals_plot.set_theme(theme)
        self._histogram_plot.set_theme(theme)

    def update_results(self, result: Any) -> None:
        """Update the display with new fit results.

        Args:
            result: The fit result object (OptimizeResult)
        """
        if result is None:
            self.clear()
            self._last_result = None
            self._export_btn.setEnabled(False)
            return

        try:
            self._update_results(result)
        except Exception as e:
            # Never leave stale stats/plots (or an enabled Export) behind
            logger.exception("Failed to display fit results")
            self._has_results = False
            self._fit_statistics.clear()
            self._fit_plot.clear()
            self._residuals_plot.clear()
            self._histogram_plot.clear()
            self._export_btn.setEnabled(False)
            self._status_label.setText(f"Could not display fit results: {e}")
            self._status_label.setStyleSheet("color: #f44336;")

    def _model_values(self, xdata: np.ndarray, popt: Any) -> np.ndarray | None:
        """Model evaluated at the data, or None if it cannot be evaluated."""
        model = self._app_state.state.model_func
        if model is None:
            return None
        try:
            return np.asarray(model(xdata, *popt))
        except Exception:
            logger.warning("Post-fit model evaluation failed", exc_info=True)
            return None

    def _update_results(self, result: Any) -> None:
        """Populate all widgets from ``result`` (may raise; see update_results)."""
        self._has_results = True
        self._last_result = result
        self._last_statistics = {}

        # Extract result data
        state = self._app_state.state
        xdata = state.xdata
        ydata = state.ydata

        # Get fitted parameters
        popt = getattr(result, "x", None)
        if popt is None:
            popt = getattr(result, "popt", None)

        # Get covariance/uncertainties
        pcov = getattr(result, "pcov", None)
        if pcov is not None:
            try:
                perr = np.sqrt(np.diag(pcov))
            except Exception:
                perr = None
        else:
            perr = None

        # Get parameter names
        param_names = self._get_param_names()

        # Update parameter results
        if popt is not None and param_names:
            values = list(popt)
            uncertainties = list(perr) if perr is not None else []

            # Compute confidence intervals using t-distribution (correct for small n)
            ci = None
            if perr is not None:
                n_data = len(state.ydata) if state.ydata is not None else 0
                n_params = len(values)
                dof = max(n_data - n_params, 1)
                t_val = stats.t.ppf(0.975, dof)
                ci = [
                    (v - t_val * e, v + t_val * e)
                    for v, e in zip(values, uncertainties, strict=False)
                ]

            self._param_results.set_results(
                names=param_names,
                values=values,
                uncertainties=uncertainties,
                confidence_intervals=ci,
            )

        # Compute fit statistics
        n_nonfinite_residuals = 0
        model_failed = False
        if xdata is not None and ydata is not None and popt is not None:
            y_fit = self._model_values(xdata, popt)
            model_failed = y_fit is None
            self._update_statistics(ydata, popt, result, y_fit)
            n_nonfinite_residuals = self._update_plots(
                xdata, ydata, popt, result, y_fit
            )

        # Update status
        converged = getattr(result, "success", True)
        n_iter = getattr(result, "nfev", 0)
        if converged:
            status_text = f"Fit completed successfully ({n_iter} evaluations)"
            status_color = "#4CAF50"
        else:
            status_text = f"Fit completed with warnings ({n_iter} evaluations)"
            status_color = "#FF9800"

        if n_nonfinite_residuals:
            # The residuals/histogram plots silently drop non-finite points
            # to avoid crashing (np.histogram can't autorange on NaN/Inf) —
            # surface that so a partially-diverged fit isn't mistaken for a
            # clean one just because the plot looks self-consistent.
            status_text += (
                f" — {n_nonfinite_residuals} non-finite residual(s) excluded from plots"
            )
            status_color = "#FF9800"

        if xdata is not None and (
            note := decimation_note(
                n_points(xdata),
                min(
                    self._fit_plot.DOWNSAMPLE_THRESHOLD,
                    self._residuals_plot.DOWNSAMPLE_THRESHOLD,
                ),
            )
        ):
            status_text += f" — {note}"

        if model_failed:
            status_text += " — model could not be evaluated; statistics unavailable"
            status_color = "#FF9800"

        # Notes recorded by the fit adapter (settings that could not be honoured)
        notes = getattr(result, "gui_notes", None)
        if notes:
            status_text += " — " + "; ".join(notes)
            status_color = "#FF9800"

        if state.sigma is not None:
            status_text += (
                " — sigma treated as absolute (uncertainties not rescaled by "
                "reduced chi-square)"
                if state.absolute_sigma
                else " — sigma treated as relative (uncertainties rescaled by "
                "reduced chi-square)"
            )

        self._status_label.setText(status_text)
        self._status_label.setStyleSheet(f"color: {status_color};")

        self._export_btn.setEnabled(True)

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        from pathlib import Path

        from nlsq.cli.visualization import STYLE_PRESETS, FitVisualizer

        # Select style preset
        presets = list(STYLE_PRESETS.keys())
        style, ok = QInputDialog.getItem(
            self,
            "Select Style",
            "Visualization Preset:",
            presets,
            0,
            False,
        )
        if not ok or not style:
            return

        # Select output file (base name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "",
            "PDF Files (*.pdf);;PNG Files (*.png);;All Files (*)",
        )
        if not file_path:
            return

        # Parse output path
        path_obj = Path(file_path)
        output_dir = path_obj.parent
        filename_prefix = path_obj.stem
        # Determining desired format from extension if explicitly set?
        # FitVisualizer configuration 'formats' list.
        # If user selected .png, use png. If .pdf, use pdf.
        ext = path_obj.suffix.lower().lstrip(".")
        formats = [ext] if ext in ["png", "pdf", "svg", "eps"] else ["pdf", "png"]

        # Prepare visualizer arguments
        state = self._app_state.state
        xdata = state.xdata
        ydata = state.ydata
        # Check if we have result data stored from update_results()
        if not hasattr(self, "_last_result") or self._last_result is None:
            QMessageBox.warning(self, "Export Failed", "No result data available.")
            return

        data_dict = {"xdata": xdata, "ydata": ydata, "sigma": state.sigma}

        # The visualizer reads popt/pcov/fun/statistics by key; the R^2 / RMSE
        # annotation needs `statistics`, which a fit result does not carry.
        last = self._last_result
        popt = getattr(last, "popt", None)
        vis_result = {
            "popt": getattr(last, "x", None) if popt is None else popt,
            "pcov": getattr(last, "pcov", None),
            "fun": getattr(last, "fun", None),
            "statistics": dict(self._last_statistics),
        }

        # Config dict
        config = {
            "visualization": {
                "enabled": True,
                "output_dir": str(output_dir),
                "filename_prefix": filename_prefix,
                "style": style,
                "formats": formats,
                "main_plot": {
                    "confidence_band": {"enabled": True},
                    "annotation": {"enabled": True},
                    "legend": {"enabled": True},
                },
                "residuals_plot": {"enabled": True},
                "histogram": {"enabled": True},
            },
        }

        try:
            visualizer = FitVisualizer()
            paths = visualizer.generate(
                result=vis_result,
                data=data_dict,
                model=state.model_func,
                config=config,
            )

            QMessageBox.information(
                self,
                "Export Successful",
                f"Figures exported to:\n{output_dir}\n\nGenerated {len(paths)} files.",
            )
        except Exception as e:
            import logging as _logging

            _logging.getLogger(__name__).warning("Export failed", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"An error occurred:\n{e!s}")

    def _get_param_names(self) -> list[str]:
        """Get parameter names from the model.

        Returns:
            List of parameter names
        """
        from nlsq.gui_qt.adapters.model_adapter import get_model_info

        state = self._app_state.state
        if state.model_func is not None:
            info = get_model_info(state.model_func)
            return info.get("param_names", [])
        return []

    def _update_statistics(
        self,
        ydata: np.ndarray,
        popt: np.ndarray,
        result: Any,
        y_fit: np.ndarray | None,
    ) -> None:
        """Update fit statistics display.

        Args:
            ydata: Y data
            popt: Fitted parameters
            result: Fit result object
            y_fit: Model values at the data, or None if unavailable (all
                quality statistics are then shown as NaN, never faked)
        """
        if y_fit is None:
            nan = float("nan")
            self._fit_statistics.set_statistics(
                r_squared=nan,
                rmse=nan,
                mae=nan,
                aic=nan,
                bic=nan,
                n_iterations=getattr(result, "nfev", 0),
                converged=getattr(result, "success", True),
            )
            return

        # Compute residuals
        residuals = ydata - y_fit

        # R² (coefficient of determination)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = (
            1.0 - (ss_res / ss_tot)
            if (np.isfinite(ss_tot) and ss_tot >= np.finfo(float).tiny)
            else float("nan")  # undefined for constant y; never faked
        )

        # RMSE (root mean square error)
        rmse = np.sqrt(np.mean(residuals**2))

        # MAE (mean absolute error)
        mae = np.mean(np.abs(residuals))

        # Number of data points and parameters
        n = len(ydata)
        k = len(popt)

        aic, bic = _information_criteria(result, ss_res, n, k)

        self._last_statistics = {
            name: float(v)
            for name, v in (("r_squared", r_squared), ("rmse", rmse))
            if np.isfinite(v)
        }

        # Get iteration count
        n_iterations = getattr(result, "nfev", 0)
        converged = getattr(result, "success", True)

        self._fit_statistics.set_statistics(
            r_squared=r_squared,
            rmse=rmse,
            mae=mae,
            aic=aic,
            bic=bic,
            n_iterations=n_iterations,
            converged=converged,
        )

    def _update_plots(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        popt: np.ndarray,
        result: Any,
        y_fit: np.ndarray | None,
    ) -> int:
        """Update all plots with fit results.

        Args:
            xdata: X data
            ydata: Y data
            popt: Fitted parameters
            result: Fit result object
            y_fit: Model values at the data, or None if unavailable

        Returns:
            Count of non-finite (NaN/Inf) residuals excluded from the
            histogram/residuals plots — 0 if none. The plot widgets filter
            these out to avoid crashing, so this is the caller's only signal
            that some data points were silently dropped from the plots.
        """
        state = self._app_state.state

        if y_fit is None:
            # No valid model values: show the data only, no fake fit/residuals
            self._fit_plot.set_data(
                np.arange(len(ydata)) if xdata.ndim == 2 else xdata, ydata
            )
            self._residuals_plot.clear()
            self._histogram_plot.clear()
            return 0

        if xdata.ndim == 2:
            # Multi-dimensional (surface) data: no single x axis, so plot
            # against the point index and skip dense curve / bands.
            idx = np.arange(len(ydata), dtype=float)
            self._fit_plot.set_data(idx, ydata)
            self._fit_plot.set_fit(idx, y_fit)
            residuals = ydata - y_fit
            self._residuals_plot.set_residuals(idx, residuals, y_fit)
            self._histogram_plot.set_data(residuals)
            return int(np.count_nonzero(~np.isfinite(residuals)))

        # Compute fitted curve on dense grid
        x_dense = _dense_grid(xdata)
        y_dense = state.model_func(x_dense, *popt)

        # Compute residuals
        residuals = ydata - y_fit

        # Compute confidence bands if covariance available
        pcov = getattr(result, "pcov", None)
        conf_lower = None
        conf_upper = None

        if pcov is not None and state.model_func is not None:
            try:
                # Use delta method for confidence bands
                conf_lower, conf_upper = self._compute_confidence_bands(
                    x_dense,
                    popt,
                    pcov,
                    state.model_func,
                    n_data=n_points(xdata),
                )
            except Exception as e:
                # Log error but don't crash UI
                logger.error(f"Failed to compute confidence bands: {e}", exc_info=True)
                conf_lower = None
                conf_upper = None

        # Update fit plot
        self._fit_plot.set_data(xdata, ydata)
        self._fit_plot.set_fit(x_dense, y_dense, conf_lower, conf_upper)

        # Compute confidence interval for residuals (at data points)
        conf_interval_resid = None
        if pcov is not None and state.model_func is not None:
            try:
                # Calculate CI at original data points
                c_low, c_high = self._compute_confidence_bands(
                    xdata,
                    popt,
                    pcov,
                    state.model_func,
                )
                conf_interval_resid = (c_high - c_low) / 2
            except Exception:
                # Logging handled in _compute_confidence_bands, just ignore here
                pass

        # Update residuals plot
        self._residuals_plot.set_residuals(
            xdata,
            residuals,
            y_fit,
            confidence_interval=conf_interval_resid,
        )

        # Update histogram
        self._histogram_plot.set_data(residuals)

        return int(np.count_nonzero(~np.isfinite(residuals)))

    def _compute_confidence_bands(
        self,
        x: np.ndarray,
        popt: np.ndarray,
        pcov: np.ndarray,
        model_func: Any,
        n_data: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence bands using the delta method.

        Args:
            x: X values for band computation
            popt: Fitted parameters
            pcov: Covariance matrix
            model_func: Model function
            n_data: Actual number of data points for DOF calculation.
                If None, falls back to len(x).

        Returns:
            Tuple of (lower_band, upper_band)
        """
        # Compute numerical gradient of model w.r.t. parameters
        # Step scaled to |p| so p + h != p for large parameters (an absolute
        # 1e-8 step gives J = 0 once |p| >~ 1e8).
        eps = float(np.sqrt(np.finfo(float).eps))
        # Ensure popt is a mutable numpy array (handle JAX arrays)
        popt = np.array(popt)
        n_params = len(popt)
        n_points = len(x)

        # Jacobian: df/dp at each x point
        J = np.zeros((n_points, n_params))

        y_base = np.asarray(model_func(x, *popt), dtype=float)

        for i in range(n_params):
            h = eps * max(1.0, abs(popt[i]))
            popt_plus = popt.copy()
            popt_plus[i] += h
            popt_minus = popt.copy()
            popt_minus[i] -= h
            y_plus = np.asarray(model_func(x, *popt_plus), dtype=float)
            y_minus = np.asarray(model_func(x, *popt_minus), dtype=float)
            J[:, i] = (y_plus - y_minus) / (2.0 * h)

        # Variance at each point: diag(J @ pcov @ J.T)
        # More efficient: sum((J @ pcov) * J, axis=1)
        var = np.sum((J @ pcov) * J, axis=1)
        std = np.sqrt(np.maximum(var, 0))

        # 95% confidence using t-distribution (consistent with parameter CIs)
        dof = max((n_data if n_data is not None else n_points) - n_params, 1)
        t_val = stats.t.ppf(0.975, dof)
        conf_lower = y_base - t_val * std
        conf_upper = y_base + t_val * std

        return conf_lower, conf_upper

    def clear(self) -> None:
        """Clear all results."""
        self._has_results = False
        self._param_results.clear()
        self._fit_statistics.clear()
        self._fit_plot.clear()
        self._residuals_plot.clear()
        self._histogram_plot.clear()
        self._status_label.setText("No fit results available")
        self._status_label.setStyleSheet("color: gray;")

    def has_results(self) -> bool:
        """Check if results are available.

        Returns:
            True if results are displayed
        """
        return self._has_results
