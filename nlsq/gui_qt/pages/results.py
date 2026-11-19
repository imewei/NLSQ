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

from nlsq.gui_qt.adapters.fit_adapter import (
    confidence_band,
    parameter_confidence_intervals,
)
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


# Status text colours per theme. The light amber/green/red are >= 4.5:1 on every
# light-theme surface (the stock #FF9800 is ~2.2:1 on white).
_STATUS_COLORS = {
    "dark": {
        "ok": "#4CAF50",
        "warn": "#FF9800",
        "bad": "#f44336",
        "muted": "gray",
    },
    "light": {
        "ok": "#2E7D32",
        "warn": "#A65300",
        "bad": "#C62828",
        "muted": "#616161",
    },
}


def _degeneracy_note(
    popt: Any, pcov: Any, ydata: Any, intervals: np.ndarray | None
) -> str | None:
    """Why a "successful" fit is statistically meaningless, or None.

    ``intervals`` are the parameter intervals actually shown: with a known
    absolute sigma ``n <= p`` still gives finite ones, so only undefined
    (non-finite) intervals are called undefined.
    """
    n_data = len(ydata) if ydata is not None else 0
    undefined = intervals is None or not np.all(np.isfinite(intervals))
    if popt is not None and 0 < n_data <= len(popt) and undefined:
        return (
            f"underdetermined fit ({n_data} data points, {len(popt)} parameters): "
            "uncertainties and confidence intervals are undefined"
        )
    if pcov is not None and not np.all(np.isfinite(pcov)):
        return "covariance could not be estimated: uncertainties are not finite"
    return None


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

        Parameters
        ----------
        app_state : AppState
            Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._has_results = False
        self._last_result = None
        self._theme_name = "dark"
        self._status_role = "muted"
        self._last_statistics: dict[str, float] = {}  # finite R^2 / RMSE for export
        self._theme_name = "dark"
        self._status_role = "muted"
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
        self._set_status("No fit results available", "muted")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

    def _connect_signals(self) -> None:
        """Connect to app state signals."""
        self._app_state.fit_completed.connect(self.update_results)

    def set_app_state(self, state: AppState) -> None:
        """Set the application state.

        Parameters
        ----------
        state : AppState
            Application state manager.
        """
        self._app_state = state

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this page.

        Parameters
        ----------
        theme : ThemeConfig
            Theme configuration.
        """
        self._theme_name = theme.name
        self._set_status(self._status_label.text(), self._status_role)
        self._param_results.set_theme(theme)
        self._fit_statistics.set_theme(theme)
        self._fit_plot.set_theme(theme)
        self._residuals_plot.set_theme(theme)
        self._histogram_plot.set_theme(theme)

    def _set_status(self, text: str, role: str) -> None:
        """Show ``text`` in the status colour of ``role`` for the current theme."""
        self._status_role = role
        self._status_label.setText(text)
        color = _STATUS_COLORS.get(self._theme_name, _STATUS_COLORS["dark"])[role]
        self._status_label.setStyleSheet(f"color: {color};")

    def update_results(self, result: Any) -> None:
        """Update the display with new fit results.

        Parameters
        ----------
        result : OptimizeResult or None
            The fit result object. None clears the page.
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
            self._param_results.clear()
            self._fit_statistics.clear()
            self._fit_plot.clear()
            self._residuals_plot.clear()
            self._histogram_plot.clear()
            self._export_btn.setEnabled(False)
            self._set_status(f"Could not display fit results: {e}", "bad")

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

        # Update parameter results (never keep the previous fit's table)
        self._param_results.clear()
        shown = None  # the intervals displayed, for the degeneracy note
        if popt is not None and param_names:
            values = list(popt)
            uncertainties = list(perr) if perr is not None else []

            # Intervals come from the result (known-sigma / dof rules; NaN when
            # undefined), the same numbers the CSV/JSON exports write.
            ci = None
            if perr is not None:
                n_data = len(state.ydata) if state.ydata is not None else 0
                shown = parameter_confidence_intervals(result, n_data)
                if shown is not None:
                    ci = [(float(lo), float(hi)) for lo, hi in shown]

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
        # chunked fits report no function-evaluation count: do not show "0"
        evaluations = f" ({n_iter} evaluations)" if n_iter else ""
        if converged:
            status_text = f"Fit completed successfully{evaluations}"
            status_color = "ok"
        else:
            status_text = f"Fit completed with warnings{evaluations}"
            status_color = "warn"

        if degenerate := _degeneracy_note(popt, pcov, ydata, shown):
            status_text += f" — {degenerate}"
            status_color = "warn"

        if n_nonfinite_residuals:
            # The residuals/histogram plots silently drop non-finite points
            # to avoid crashing (np.histogram can't autorange on NaN/Inf) —
            # surface that so a partially-diverged fit isn't mistaken for a
            # clean one just because the plot looks self-consistent.
            status_text += (
                f" — {n_nonfinite_residuals} non-finite residual(s) excluded from plots"
            )
            status_color = "warn"

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
            status_color = "warn"

        # Notes recorded by the fit adapter (settings that could not be honoured)
        notes = getattr(result, "gui_notes", None)
        if notes:
            status_text += " — " + "; ".join(notes)
            status_color = "warn"

        if state.sigma is not None:
            status_text += (
                " — sigma treated as absolute (uncertainties not rescaled by "
                "reduced chi-square)"
                if self._app_state.fit_settings.absolute_sigma
                else " — sigma treated as relative (uncertainties rescaled by "
                "reduced chi-square)"
            )

        self._set_status(status_text, status_color)

        # The visualizer only draws a 1-D independent variable
        surface = xdata is not None and np.ndim(xdata) > 1
        self._export_btn.setEnabled(not surface)
        self._export_btn.setToolTip(
            "Figure export supports 1-D fits only (this fit has multiple "
            "independent variables)"
            if surface
            else ""
        )

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

        # The visualizer writes <prefix>_combined.<fmt> and <prefix>_histogram.<fmt>,
        # not the path picked in the dialog: confirm those before replacing them.
        targets = [
            output_dir / f"{filename_prefix}_{kind}.{fmt}"
            for kind in ("combined", "histogram")
            for fmt in formats
        ]
        existing = [t.name for t in targets if t.exists()]
        if existing and (
            QMessageBox.question(
                self,
                "Overwrite Files?",
                "These files already exist and will be replaced:\n\n"
                + "\n".join(existing),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return

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
            # the fit's own quantile rule (normal vs Student-t) for the band
            "absolute_sigma": bool(self._app_state.fit_settings.absolute_sigma),
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

        Returns
        -------
        list[str]
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

        Parameters
        ----------
        ydata : np.ndarray
            Y data
        popt : np.ndarray
            Fitted parameters
        result : Any
            Fit result object
        y_fit : np.ndarray | None
            Model values at the data, or None if unavailable (all
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

        Parameters
        ----------
        xdata : np.ndarray
            X data
        ydata : np.ndarray
            Y data
        popt : np.ndarray
            Fitted parameters
        result : Any
            Fit result object
        y_fit : np.ndarray | None
            Model values at the data, or None if unavailable

        Returns
        -------
        int
            Count of non-finite (NaN/Inf) residuals excluded from the
            histogram/residuals plots — 0 if none. The plot widgets filter
            these out to avoid crashing, so this is the caller's only signal
            that some data points were silently dropped from the plots.
        """
        state = self._app_state.state

        if y_fit is None:
            # No valid model values: show the data only, no fake fit/residuals
            # (clear first: set_data keeps the previous fit's curve and band)
            self._fit_plot.clear()
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
        order = None  # set when the curve is drawn at the observed points
        try:
            y_dense = state.model_func(x_dense, *popt)
        except Exception:
            # e.g. a model tied to the data length: draw the curve through the
            # data points instead of discarding the whole results page
            logger.warning("Model evaluation on the dense grid failed", exc_info=True)
            order = np.argsort(xdata, kind="stable")
            x_dense, y_dense = xdata[order], y_fit[order]

        # Compute residuals
        residuals = ydata - y_fit

        # Confidence band from the result (the library's domain-aware delta
        # method), evaluated on the dense grid only: a band at every data point
        # of a 500K-point fit would block the GUI thread for seconds.
        conf_lower = None
        conf_upper = None
        # At the observed points the model is evaluated in observation order (it
        # may be tied to it, like the curve above) and sorted with the same
        # permutation as the curve.
        band = confidence_band(result, xdata if order is not None else x_dense)
        if band is not None:
            conf_lower, conf_upper = band
            if order is not None:
                # row-wise half-widths at the observations: duplicate x must not
                # share one interpolated value on the Residuals tab
                rowwise_half_width = (conf_upper - conf_lower) / 2
                conf_lower, conf_upper = conf_lower[order], conf_upper[order]

        # Update fit plot
        self._fit_plot.set_data(xdata, ydata)
        self._fit_plot.set_fit(x_dense, y_dense, conf_lower, conf_upper)

        # Residual interval at the data points: the band half-width is smooth,
        # so interpolate it from the dense grid instead of re-evaluating there.
        # Display-only limit: a feature narrower than the 500-point grid spacing
        # is smoothed out (the dense curve has the same resolution).
        conf_interval_resid = None
        if band is not None:
            conf_interval_resid = (
                rowwise_half_width
                if order is not None
                else np.interp(xdata, x_dense, (conf_upper - conf_lower) / 2)
            )

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

    def clear(self) -> None:
        """Clear all results."""
        self._has_results = False
        self._param_results.clear()
        self._fit_statistics.clear()
        self._fit_plot.clear()
        self._residuals_plot.clear()
        self._histogram_plot.clear()
        self._set_status("No fit results available", "muted")

    def has_results(self) -> bool:
        """Check if results are available.

        Returns
        -------
        bool
            True if results are displayed.
        """
        return self._has_results
