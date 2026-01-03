"""
NLSQ Qt GUI Fitting Options Page

This page allows users to configure fitting parameters and execute
the curve fitting operation with real-time progress feedback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nlsq.gui_qt.plots.live_cost_plot import LiveCostPlotWidget
from nlsq.gui_qt.widgets.advanced_options import AdvancedOptionsWidget
from nlsq.gui_qt.widgets.iteration_table import IterationTableWidget
from nlsq.gui_qt.widgets.param_config import ParamConfigWidget

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["FittingOptionsPage"]

# Preset configurations
PRESETS = {
    "Fast": {
        "gtol": 1e-6,
        "ftol": 1e-6,
        "xtol": 1e-6,
        "max_iterations": 100,
        "enable_multistart": False,
    },
    "Robust": {
        "gtol": 1e-8,
        "ftol": 1e-8,
        "xtol": 1e-8,
        "max_iterations": 200,
        "enable_multistart": True,
        "n_starts": 5,
    },
    "Quality": {
        "gtol": 1e-10,
        "ftol": 1e-10,
        "xtol": 1e-10,
        "max_iterations": 500,
        "enable_multistart": True,
        "n_starts": 10,
    },
}


class FitWorker(QObject):
    """Worker for running fit in background thread."""

    # Signals
    progress = Signal(int, float)  # iteration, cost
    finished = Signal(object)  # result
    error = Signal(str)  # error message

    def __init__(self, state: Any) -> None:
        """Initialize the worker.

        Args:
            state: Session state with fit configuration
        """
        super().__init__()
        self._state = state
        self._abort_flag = False

    def run(self) -> None:
        """Execute the fitting operation."""
        from nlsq.gui_qt.adapters.fit_adapter import run_fit

        try:
            # Create callback for progress updates
            def progress_callback(iteration: int, cost: float) -> bool:
                self.progress.emit(iteration, cost)
                return not self._abort_flag

            # Run the fit
            result = run_fit(self._state, progress_callback=progress_callback)

            if self._abort_flag:
                self.error.emit("Fit aborted by user")
            else:
                self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def abort(self) -> None:
        """Request abort of the fit."""
        self._abort_flag = True


class FittingOptionsPage(QWidget):
    """Page for configuring and running fits.

    Provides:
    - Guided/Advanced mode toggle
    - Preset selector (Fast/Robust/Quality) in Guided mode
    - Parameter configuration widget
    - Advanced options (tolerances, multi-start, streaming)
    - Run/Abort buttons
    - Live cost function plot
    - Iteration history table
    """

    # Signals
    fit_started = Signal()
    fit_progress = Signal(int, float)  # iteration, cost
    fit_completed = Signal(object)  # result
    fit_aborted = Signal()

    def __init__(self, app_state: AppState) -> None:
        """Initialize the fitting options page.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._fit_thread: QThread | None = None
        self._fit_worker: FitWorker | None = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Fitting Options")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Mode selector
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_group)

        self._mode_button_group = QButtonGroup()
        self._guided_radio = QRadioButton("Guided")
        self._advanced_radio = QRadioButton("Advanced")
        self._mode_button_group.addButton(self._guided_radio, 0)
        self._mode_button_group.addButton(self._advanced_radio, 1)
        self._guided_radio.setChecked(True)

        mode_layout.addWidget(self._guided_radio)
        mode_layout.addWidget(self._advanced_radio)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # Main content with splitter
        splitter = QSplitter()

        # Left panel - Configuration
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Progress
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 1)

        # Action buttons and progress
        self._create_action_section(layout)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with configuration options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        # Preset selector (Guided mode)
        self._preset_group = QGroupBox("Preset")
        preset_layout = QHBoxLayout(self._preset_group)

        self._preset_combo = QComboBox()
        self._preset_combo.addItems(list(PRESETS.keys()))
        self._preset_combo.setCurrentText("Robust")
        preset_layout.addWidget(self._preset_combo)
        preset_layout.addStretch()

        layout.addWidget(self._preset_group)

        # Parameter configuration
        param_group = QGroupBox("Parameter Configuration")
        param_layout = QVBoxLayout(param_group)
        self._param_config = ParamConfigWidget()
        param_layout.addWidget(self._param_config)
        layout.addWidget(param_group)

        # Advanced options (Advanced mode)
        self._advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QVBoxLayout(self._advanced_group)
        self._advanced_options = AdvancedOptionsWidget()
        advanced_layout.addWidget(self._advanced_options)
        layout.addWidget(self._advanced_group)
        self._advanced_group.setVisible(False)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with progress displays."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)

        # Live cost plot
        plot_group = QGroupBox("Cost Function")
        plot_layout = QVBoxLayout(plot_group)
        self._cost_plot = LiveCostPlotWidget()
        plot_layout.addWidget(self._cost_plot)
        layout.addWidget(plot_group, 1)

        # Iteration table
        table_group = QGroupBox("Iteration History")
        table_layout = QVBoxLayout(table_group)
        self._iteration_table = IterationTableWidget()
        table_layout.addWidget(self._iteration_table)
        layout.addWidget(table_group, 1)

        return panel

    def _create_action_section(self, parent_layout: QVBoxLayout) -> None:
        """Create the action buttons and progress section."""
        # Progress bar
        progress_row = QHBoxLayout()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        self._progress_bar.setVisible(False)
        progress_row.addWidget(self._progress_bar)

        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet("color: gray;")
        progress_row.addWidget(self._status_label)
        parent_layout.addLayout(progress_row)

        # Buttons
        button_row = QHBoxLayout()
        button_row.addStretch()

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset all options to defaults")
        button_row.addWidget(self._reset_btn)

        self._abort_btn = QPushButton("Abort")
        self._abort_btn.setToolTip("Abort the running fit")
        self._abort_btn.setEnabled(False)
        self._abort_btn.setStyleSheet("background-color: #f44336; color: white;")
        button_row.addWidget(self._abort_btn)

        self._run_btn = QPushButton("Run Fit")
        self._run_btn.setToolTip("Start the curve fitting")
        self._run_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        button_row.addWidget(self._run_btn)

        parent_layout.addLayout(button_row)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._mode_button_group.idClicked.connect(self._on_mode_changed)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self._reset_btn.clicked.connect(self.reset)
        self._run_btn.clicked.connect(self.run_fit)
        self._abort_btn.clicked.connect(self.abort_fit)

        # Connect to app state model changes
        self._app_state.model_changed.connect(self._on_model_changed)

    def _on_mode_changed(self, mode_id: int) -> None:
        """Handle mode change.

        Args:
            mode_id: Mode button ID (0=Guided, 1=Advanced)
        """
        is_advanced = mode_id == 1
        self._preset_group.setVisible(not is_advanced)
        self._advanced_group.setVisible(is_advanced)

    def _on_preset_changed(self, preset_name: str) -> None:
        """Handle preset change.

        Args:
            preset_name: Selected preset name
        """
        if preset_name in PRESETS:
            preset = PRESETS[preset_name]
            self._advanced_options.set_options(preset)

    def _on_model_changed(self) -> None:
        """Handle model change to update parameter config."""
        from nlsq.gui_qt.adapters.model_adapter import get_model_info

        state = self._app_state.state
        if state.model_func is not None:
            info = get_model_info(state.model_func)
            param_names = info.get("param_names", [])
            self._param_config.set_param_names(param_names)

    def run_fit(self) -> None:
        """Execute the curve fitting operation."""
        # Check prerequisites
        state = self._app_state.state
        if state.xdata is None or state.ydata is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please load data before running fit.",
            )
            return

        if state.model_func is None:
            QMessageBox.warning(
                self,
                "No Model",
                "Please select a model before running fit.",
            )
            return

        # Update state with current options
        if self._guided_radio.isChecked():
            preset = PRESETS.get(self._preset_combo.currentText(), {})
            for key, value in preset.items():
                if hasattr(state, key):
                    setattr(state, key, value)
        else:
            options = self._advanced_options.get_options()
            for key, value in options.items():
                if hasattr(state, key):
                    setattr(state, key, value)

        # Update p0 and bounds from param config
        if not self._param_config.is_auto_p0():
            p0, bounds = self._param_config.get_values()
            state.p0 = p0 if p0 else None
            state.bounds = bounds
            state.auto_p0 = False
        else:
            state.auto_p0 = True
            state.p0 = None
            state.bounds = None

        # Clear previous results
        self._cost_plot.reset()
        self._iteration_table.clear()

        # Update UI state
        self._set_running_state(True)

        # Create and start worker thread
        self._fit_thread = QThread()
        self._fit_worker = FitWorker(state)
        self._fit_worker.moveToThread(self._fit_thread)

        # Connect signals
        self._fit_thread.started.connect(self._fit_worker.run)
        self._fit_worker.progress.connect(self._on_fit_progress)
        self._fit_worker.finished.connect(self._on_fit_finished)
        self._fit_worker.error.connect(self._on_fit_error)
        self._fit_worker.finished.connect(self._cleanup_thread)
        self._fit_worker.error.connect(self._cleanup_thread)

        # Start
        self._fit_thread.start()
        self.fit_started.emit()
        self._app_state.set_fit_running(True)

    def abort_fit(self) -> None:
        """Abort the running fit operation."""
        if self._fit_worker is not None:
            self._fit_worker.abort()
            self._status_label.setText("Aborting...")

    def _on_fit_progress(self, iteration: int, cost: float) -> None:
        """Handle fit progress update.

        Args:
            iteration: Current iteration
            cost: Current cost value
        """
        self._cost_plot.add_point(iteration, cost)
        self._iteration_table.add_iteration(iteration, cost)
        self._status_label.setText(f"Iteration {iteration}: cost = {cost:.6g}")
        self.fit_progress.emit(iteration, cost)
        self._app_state.emit_fit_progress(iteration, cost)

    def _on_fit_finished(self, result: object) -> None:
        """Handle fit completion.

        Args:
            result: Fit result object
        """
        self._set_running_state(False)
        self._status_label.setText("Fit completed")
        self._status_label.setStyleSheet("color: #4CAF50;")

        # Update app state
        self._app_state.set_fit_result(result)
        self.fit_completed.emit(result)

        QMessageBox.information(
            self,
            "Fit Complete",
            "Curve fitting completed successfully.\n"
            "Go to Results page to view the fitted parameters.",
        )

    def _on_fit_error(self, message: str) -> None:
        """Handle fit error.

        Args:
            message: Error message
        """
        self._set_running_state(False)

        if "aborted" in message.lower():
            self._status_label.setText("Fit aborted")
            self._status_label.setStyleSheet("color: #FF9800;")
            self._app_state.set_fit_aborted()
            self.fit_aborted.emit()
        else:
            self._status_label.setText("Fit failed")
            self._status_label.setStyleSheet("color: #f44336;")
            QMessageBox.critical(
                self,
                "Fit Error",
                f"Curve fitting failed:\n\n{message}",
            )

    def _cleanup_thread(self) -> None:
        """Clean up the worker thread."""
        if self._fit_thread is not None:
            self._fit_thread.quit()
            self._fit_thread.wait()
            self._fit_thread = None
            self._fit_worker = None

    def _set_running_state(self, running: bool) -> None:
        """Update UI for running/stopped state.

        Args:
            running: Whether fit is running
        """
        self._run_btn.setEnabled(not running)
        self._abort_btn.setEnabled(running)
        self._reset_btn.setEnabled(not running)
        self._progress_bar.setVisible(running)
        self._preset_combo.setEnabled(not running)
        self._param_config.setEnabled(not running)
        self._advanced_options.setEnabled(not running)

        if running:
            self._status_label.setText("Running...")
            self._status_label.setStyleSheet("color: #2196F3;")

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
        self._cost_plot.set_theme(theme)
        self._iteration_table.set_theme(theme)
        self._param_config.set_theme(theme)
        self._advanced_options.set_theme(theme)

    def reset(self) -> None:
        """Reset the page to initial state."""
        self._guided_radio.setChecked(True)
        self._on_mode_changed(0)
        self._preset_combo.setCurrentText("Robust")
        self._cost_plot.reset()
        self._iteration_table.clear()
        self._status_label.setText("Ready")
        self._status_label.setStyleSheet("color: gray;")
