"""
NLSQ Qt GUI Results Page

This page displays the fitted parameters, statistical metrics,
and interactive visualizations of the fit quality.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from nlsq.gui_qt.plots.fit_plot import FitPlotWidget
from nlsq.gui_qt.plots.histogram_plot import HistogramPlotWidget
from nlsq.gui_qt.plots.residuals_plot import ResidualsPlotWidget
from nlsq.gui_qt.widgets.fit_statistics import FitStatisticsWidget
from nlsq.gui_qt.widgets.param_results import ParamResultsWidget

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ResultsPage"]


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
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Results")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Main content - splitter with left/right panels
        splitter = QSplitter()

        # Left panel - Parameters and statistics (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)

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
            self._status_label.setText("No fit results available")
            self._status_label.setStyleSheet("color: gray;")
            return

        self._has_results = True

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
            uncertainties = list(perr) if perr is not None else [0.0] * len(values)

            # Compute confidence intervals (95% = ±1.96*std)
            ci = None
            if perr is not None:
                ci = [
                    (v - 1.96 * e, v + 1.96 * e)
                    for v, e in zip(values, uncertainties, strict=False)
                ]

            self._param_results.set_results(
                names=param_names,
                values=values,
                uncertainties=uncertainties,
                confidence_intervals=ci,
            )

        # Compute fit statistics
        if xdata is not None and ydata is not None and popt is not None:
            self._update_statistics(xdata, ydata, popt, result)
            self._update_plots(xdata, ydata, popt, result)

        # Update status
        converged = getattr(result, "success", True)
        n_iter = getattr(result, "nfev", 0)
        if converged:
            self._status_label.setText(
                f"Fit completed successfully ({n_iter} evaluations)"
            )
            self._status_label.setStyleSheet("color: #4CAF50;")
        else:
            self._status_label.setText(
                f"Fit completed with warnings ({n_iter} evaluations)"
            )
            self._status_label.setStyleSheet("color: #FF9800;")

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
        xdata: np.ndarray,
        ydata: np.ndarray,
        popt: np.ndarray,
        result: Any,
    ) -> None:
        """Update fit statistics display.

        Args:
            xdata: X data
            ydata: Y data
            popt: Fitted parameters
            result: Fit result object
        """
        state = self._app_state.state

        # Compute fitted values
        try:
            if state.model_func is not None:
                y_fit = state.model_func(xdata, *popt)
            else:
                y_fit = ydata  # Fallback
        except Exception:
            y_fit = ydata

        # Compute residuals
        residuals = ydata - y_fit

        # R² (coefficient of determination)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE (root mean square error)
        rmse = np.sqrt(np.mean(residuals**2))

        # MAE (mean absolute error)
        mae = np.mean(np.abs(residuals))

        # Number of data points and parameters
        n = len(ydata)
        k = len(popt)

        # AIC (Akaike Information Criterion)
        # AIC = n * log(RSS/n) + 2k
        if ss_res > 0:
            aic = n * np.log(ss_res / n) + 2 * k
        else:
            aic = -np.inf

        # BIC (Bayesian Information Criterion)
        # BIC = n * log(RSS/n) + k * log(n)
        if ss_res > 0:
            bic = n * np.log(ss_res / n) + k * np.log(n)
        else:
            bic = -np.inf

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
    ) -> None:
        """Update all plots with fit results.

        Args:
            xdata: X data
            ydata: Y data
            popt: Fitted parameters
            result: Fit result object
        """
        state = self._app_state.state

        # Compute fitted curve on dense grid
        x_min, x_max = np.min(xdata), np.max(xdata)
        x_range = x_max - x_min
        x_dense = np.linspace(x_min - 0.05 * x_range, x_max + 0.05 * x_range, 500)

        try:
            if state.model_func is not None:
                y_dense = state.model_func(x_dense, *popt)
                y_fit = state.model_func(xdata, *popt)
            else:
                y_dense = np.zeros_like(x_dense)
                y_fit = ydata
        except Exception:
            y_dense = np.zeros_like(x_dense)
            y_fit = ydata

        # Compute residuals
        residuals = ydata - y_fit

        # Compute confidence bands if covariance available
        pcov = getattr(result, "pcov", None)
        conf_lower = None
        conf_upper = None

        if pcov is not None and state.model_func is not None:
            with contextlib.suppress(Exception):
                # Use delta method for confidence bands
                conf_lower, conf_upper = self._compute_confidence_bands(
                    x_dense, popt, pcov, state.model_func
                )

        # Update fit plot
        self._fit_plot.set_data(xdata, ydata)
        self._fit_plot.set_fit(x_dense, y_dense, conf_lower, conf_upper)

        # Update residuals plot
        self._residuals_plot.set_residuals(xdata, residuals, y_fit)

        # Update histogram
        self._histogram_plot.set_data(residuals)

    def _compute_confidence_bands(
        self,
        x: np.ndarray,
        popt: np.ndarray,
        pcov: np.ndarray,
        model_func: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute confidence bands using the delta method.

        Args:
            x: X values for band computation
            popt: Fitted parameters
            pcov: Covariance matrix
            model_func: Model function

        Returns:
            Tuple of (lower_band, upper_band)
        """
        # Compute numerical gradient of model w.r.t. parameters
        eps = 1e-8
        n_params = len(popt)
        n_points = len(x)

        # Jacobian: df/dp at each x point
        J = np.zeros((n_points, n_params))

        y_base = model_func(x, *popt)

        for i in range(n_params):
            popt_plus = popt.copy()
            popt_plus[i] += eps
            y_plus = model_func(x, *popt_plus)
            J[:, i] = (y_plus - y_base) / eps

        # Variance at each point: diag(J @ pcov @ J.T)
        # More efficient: sum((J @ pcov) * J, axis=1)
        var = np.sum((J @ pcov) * J, axis=1)
        std = np.sqrt(np.maximum(var, 0))

        # 95% confidence: ±1.96 * std
        conf_lower = y_base - 1.96 * std
        conf_upper = y_base + 1.96 * std

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
