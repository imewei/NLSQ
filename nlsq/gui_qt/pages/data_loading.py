"""
NLSQ Qt GUI Data Loading Page

This page allows users to load data from files or clipboard,
assign columns to roles (x, y, sigma), and preview the data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from nlsq.gui_qt.widgets.column_selector import ColumnSelectorWidget

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["DataLoadingPage"]

# File format options
FILE_FORMATS = [
    ("Auto-detect", "auto"),
    ("CSV (Comma-separated)", "csv"),
    ("ASCII (Whitespace)", "ascii"),
    ("NumPy NPZ", "npz"),
    ("HDF5", "hdf5"),
]

# File filters for dialog
FILE_FILTERS = (
    "All Supported Files (*.csv *.txt *.dat *.npz *.npy *.h5 *.hdf5);;"
    "CSV Files (*.csv);;"
    "Text Files (*.txt *.dat);;"
    "NumPy Files (*.npz *.npy);;"
    "HDF5 Files (*.h5 *.hdf5);;"
    "All Files (*)"
)


class DataLoadingPage(QWidget):
    """Page for loading and configuring data.

    Provides:
    - File upload via QFileDialog
    - Clipboard paste functionality
    - Format selection (CSV, ASCII, NPZ, HDF5)
    - Column assignment widget
    - Data preview table
    - Statistics display
    """

    # Signals
    data_loaded = Signal(object, object, object)  # xdata, ydata, sigma

    def __init__(self, app_state: AppState) -> None:
        """Initialize the data loading page.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state

        # Data storage
        self._raw_data: NDArray[np.float64] | None = None
        self._xdata: NDArray[np.float64] | None = None
        self._ydata: NDArray[np.float64] | None = None
        self._sigma: NDArray[np.float64] | None = None
        self._column_names: list[str] = []

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Data Loading")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Data input
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)

        # File upload section
        self._create_file_section(left_layout)

        # Clipboard section
        self._create_clipboard_section(left_layout)

        # Column selector
        self._create_column_section(left_layout)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Right panel - Preview and statistics
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)

        # Data preview
        self._create_preview_section(right_layout)

        # Statistics
        self._create_statistics_section(right_layout)

        splitter.addWidget(right_panel)

        # Set splitter proportions (40% left, 60% right)
        splitter.setSizes([400, 600])
        layout.addWidget(splitter, 1)

        # Action buttons
        self._create_action_buttons(layout)

    def _create_file_section(self, parent_layout: QVBoxLayout) -> None:
        """Create file upload section."""
        group = QGroupBox("Load from File")
        layout = QVBoxLayout(group)

        # Format selector row
        format_row = QHBoxLayout()
        format_label = QLabel("Format:")
        self._format_combo = QComboBox()
        for display_name, value in FILE_FORMATS:
            self._format_combo.addItem(display_name, value)
        format_row.addWidget(format_label)
        format_row.addWidget(self._format_combo, 1)
        layout.addLayout(format_row)

        # File path display
        self._file_path_label = QLabel("No file selected")
        self._file_path_label.setStyleSheet("color: gray; font-style: italic;")
        self._file_path_label.setWordWrap(True)
        layout.addWidget(self._file_path_label)

        # Browse button
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setToolTip("Open file browser to select a data file")
        layout.addWidget(self._browse_btn)

        parent_layout.addWidget(group)

    def _create_clipboard_section(self, parent_layout: QVBoxLayout) -> None:
        """Create clipboard paste section."""
        group = QGroupBox("Paste from Clipboard")
        layout = QVBoxLayout(group)

        # Info label
        info_label = QLabel("Paste tabular data from Excel, Google Sheets, or text:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Text area
        self._clipboard_text = QPlainTextEdit()
        self._clipboard_text.setPlaceholderText(
            "Paste data here (tab or comma separated)...\n\n"
            "Example:\n"
            "1.0    2.5\n"
            "2.0    5.1\n"
            "3.0    7.8"
        )
        self._clipboard_text.setMaximumHeight(120)
        layout.addWidget(self._clipboard_text)

        # Parse button
        self._parse_btn = QPushButton("Parse Clipboard Data")
        self._parse_btn.setToolTip("Parse the pasted text as tabular data")
        layout.addWidget(self._parse_btn)

        parent_layout.addWidget(group)

    def _create_column_section(self, parent_layout: QVBoxLayout) -> None:
        """Create column assignment section."""
        group = QGroupBox("Column Assignment")
        layout = QVBoxLayout(group)

        self._column_selector = ColumnSelectorWidget()
        layout.addWidget(self._column_selector)

        parent_layout.addWidget(group)

    def _create_preview_section(self, parent_layout: QVBoxLayout) -> None:
        """Create data preview table section."""
        group = QGroupBox("Data Preview")
        layout = QVBoxLayout(group)

        # Table view with model
        self._preview_table = QTableView()
        self._preview_model = QStandardItemModel()
        self._preview_table.setModel(self._preview_model)
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)

        # Configure header
        header = self._preview_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self._preview_table)

        # Preview info
        self._preview_info = QLabel("Load data to see preview")
        self._preview_info.setStyleSheet("color: gray;")
        layout.addWidget(self._preview_info)

        parent_layout.addWidget(group, 1)

    def _create_statistics_section(self, parent_layout: QVBoxLayout) -> None:
        """Create statistics display section."""
        group = QGroupBox("Data Statistics")
        layout = QVBoxLayout(group)

        # Statistics labels
        self._stats_points = QLabel("Points: -")
        self._stats_x_range = QLabel("X Range: -")
        self._stats_y_range = QLabel("Y Range: -")
        self._stats_sigma = QLabel("Sigma: -")

        for label in [
            self._stats_points,
            self._stats_x_range,
            self._stats_y_range,
            self._stats_sigma,
        ]:
            layout.addWidget(label)

        # Validation status
        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        layout.addWidget(self._validation_label)

        parent_layout.addWidget(group)

    def _create_action_buttons(self, parent_layout: QVBoxLayout) -> None:
        """Create action buttons row."""
        button_row = QHBoxLayout()
        button_row.addStretch()

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear all loaded data")
        button_row.addWidget(self._clear_btn)

        self._apply_btn = QPushButton("Apply Data")
        self._apply_btn.setToolTip("Apply the loaded data for curve fitting")
        self._apply_btn.setEnabled(False)
        button_row.addWidget(self._apply_btn)

        parent_layout.addLayout(button_row)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._browse_btn.clicked.connect(self._on_browse)
        self._parse_btn.clicked.connect(self._on_parse_clipboard)
        self._column_selector.selection_changed.connect(
            self._on_column_selection_changed
        )
        self._clear_btn.clicked.connect(self.reset)
        self._apply_btn.clicked.connect(self._on_apply)

    def _on_browse(self) -> None:
        """Handle browse button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Data File",
            "",
            FILE_FILTERS,
        )

        if file_path:
            self._load_file(Path(file_path))

    def _load_file(self, file_path: Path) -> None:
        """Load data from a file.

        Args:
            file_path: Path to the file to load
        """
        from nlsq.gui_qt.adapters.data_adapter import load_from_file

        format_value = self._format_combo.currentData()
        config: dict[str, Any] = {
            "format": format_value,
            "columns": {"x": 0, "y": 1},
        }

        try:
            xdata, ydata, sigma = load_from_file(str(file_path), config)

            # Store raw data for preview
            if sigma is not None:
                self._raw_data = np.column_stack(
                    [xdata.T if xdata.ndim == 2 else xdata, ydata, sigma]
                )
            else:
                self._raw_data = np.column_stack(
                    [xdata.T if xdata.ndim == 2 else xdata, ydata]
                )

            # Generate column names
            n_cols = self._raw_data.shape[1]
            self._column_names = [f"Column {i}" for i in range(n_cols)]

            # Update UI
            self._file_path_label.setText(str(file_path))
            self._file_path_label.setStyleSheet("")
            self._column_selector.set_columns(self._column_names)
            self._update_preview()

        except Exception as e:
            self._show_error("File Load Error", str(e))

    def _on_parse_clipboard(self) -> None:
        """Handle parse clipboard button click."""
        from nlsq.gui_qt.adapters.data_adapter import detect_delimiter

        text = self._clipboard_text.toPlainText().strip()
        if not text:
            self._show_error("No Data", "Please paste data into the text area first.")
            return

        try:
            # Detect delimiter and parse
            delimiter = detect_delimiter(text)
            lines = text.split("\n")
            rows: list[list[float]] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                if delimiter is None:
                    parts = stripped.split()
                else:
                    parts = stripped.split(delimiter)

                row = [float(p.strip()) for p in parts if p.strip()]
                if row:
                    rows.append(row)

            if not rows:
                self._show_error("Parse Error", "No valid numeric data found.")
                return

            # Convert to numpy array
            self._raw_data = np.array(rows, dtype=np.float64)
            n_cols = self._raw_data.shape[1]
            self._column_names = [f"Column {i}" for i in range(n_cols)]

            # Update UI
            self._file_path_label.setText("(Clipboard data)")
            self._file_path_label.setStyleSheet("color: green;")
            self._column_selector.set_columns(self._column_names)
            self._update_preview()

        except ValueError as e:
            self._show_error("Parse Error", str(e))

    def _on_column_selection_changed(self, selection: dict[str, int | None]) -> None:
        """Handle column selection change.

        Args:
            selection: Column selection dictionary
        """
        if self._raw_data is None:
            return

        # Validate selection
        is_valid, message = self._column_selector.validate()

        if not is_valid:
            self._validation_label.setText(f"Invalid: {message}")
            self._validation_label.setStyleSheet("color: #f44336;")
            self._apply_btn.setEnabled(False)
            return

        # Extract data based on selection
        try:
            x_idx = selection["x"]
            y_idx = selection["y"]
            z_idx = selection.get("z")
            sigma_idx = selection.get("sigma")

            if z_idx is not None:
                # 2D mode
                x_coords = self._raw_data[:, x_idx] if x_idx is not None else None
                y_coords = self._raw_data[:, y_idx] if y_idx is not None else None
                if x_coords is not None and y_coords is not None:
                    self._xdata = np.vstack([x_coords, y_coords])
                self._ydata = self._raw_data[:, z_idx]
            else:
                # 1D mode
                self._xdata = self._raw_data[:, x_idx] if x_idx is not None else None
                self._ydata = self._raw_data[:, y_idx] if y_idx is not None else None

            self._sigma = (
                self._raw_data[:, sigma_idx] if sigma_idx is not None else None
            )

            # Validate data
            self._validate_and_update_stats()

        except IndexError as e:
            self._validation_label.setText(f"Column error: {e}")
            self._validation_label.setStyleSheet("color: #f44336;")
            self._apply_btn.setEnabled(False)

    def _validate_and_update_stats(self) -> None:
        """Validate data and update statistics display."""
        from nlsq.gui_qt.adapters.data_adapter import compute_statistics, validate_data

        if self._xdata is None or self._ydata is None:
            return

        # Validate
        result = validate_data(self._xdata, self._ydata, self._sigma)

        if not result.is_valid:
            self._validation_label.setText(f"Validation: {result.message}")
            self._validation_label.setStyleSheet("color: #f44336;")
            self._apply_btn.setEnabled(False)

            # Show warning for NaN/Inf
            if result.nan_count > 0 or result.inf_count > 0:
                details = []
                if result.nan_count > 0:
                    details.append(f"{result.nan_count} NaN")
                if result.inf_count > 0:
                    details.append(f"{result.inf_count} Inf")
                QMessageBox.warning(
                    self,
                    "Data Warning",
                    f"Data contains non-finite values: {', '.join(details)}.\n"
                    "Please clean your data before fitting.",
                )
            return

        # Compute statistics
        stats = compute_statistics(self._xdata, self._ydata, self._sigma)

        # Update display
        self._stats_points.setText(f"Points: {stats['point_count']:,}")
        self._stats_x_range.setText(
            f"X Range: [{stats['x_min']:.4g}, {stats['x_max']:.4g}]"
        )
        self._stats_y_range.setText(
            f"Y Range: [{stats['y_min']:.4g}, {stats['y_max']:.4g}]"
        )

        if stats.get("has_sigma"):
            self._stats_sigma.setText(
                f"Sigma Range: [{stats['sigma_min']:.4g}, {stats['sigma_max']:.4g}]"
            )
        else:
            self._stats_sigma.setText("Sigma: (not provided)")

        self._validation_label.setText("Data valid")
        self._validation_label.setStyleSheet("color: #4CAF50;")
        self._apply_btn.setEnabled(True)

    def _update_preview(self) -> None:
        """Update the data preview table."""
        self._preview_model.clear()

        if self._raw_data is None:
            self._preview_info.setText("Load data to see preview")
            return

        n_rows, n_cols = self._raw_data.shape

        # Set headers
        self._preview_model.setHorizontalHeaderLabels(self._column_names)

        # Limit preview rows for performance
        preview_rows = min(n_rows, 100)

        for row_idx in range(preview_rows):
            row_items = []
            for col_idx in range(n_cols):
                value = self._raw_data[row_idx, col_idx]
                item = QStandardItem(f"{value:.6g}")
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                row_items.append(item)
            self._preview_model.appendRow(row_items)

        # Update info
        if n_rows > preview_rows:
            self._preview_info.setText(
                f"Showing first {preview_rows} of {n_rows:,} rows"
            )
        else:
            self._preview_info.setText(f"{n_rows:,} rows, {n_cols} columns")

    def _on_apply(self) -> None:
        """Apply the loaded data to app state."""
        if self._xdata is None or self._ydata is None:
            return

        # Emit signal and update app state
        self._app_state.set_data(self._xdata, self._ydata, self._sigma)
        self.data_loaded.emit(self._xdata, self._ydata, self._sigma)

        QMessageBox.information(
            self,
            "Data Applied",
            f"Successfully loaded {len(self._ydata):,} data points.\n"
            "You can now proceed to Model Selection.",
        )

    def _show_error(self, title: str, message: str) -> None:
        """Show an error dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)

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
        self._column_selector.set_theme(theme)

    def reset(self) -> None:
        """Reset the page to initial state."""
        self._raw_data = None
        self._xdata = None
        self._ydata = None
        self._sigma = None
        self._column_names = []

        self._file_path_label.setText("No file selected")
        self._file_path_label.setStyleSheet("color: gray; font-style: italic;")
        self._clipboard_text.clear()
        self._preview_model.clear()
        self._preview_info.setText("Load data to see preview")

        self._stats_points.setText("Points: -")
        self._stats_x_range.setText("X Range: -")
        self._stats_y_range.setText("Y Range: -")
        self._stats_sigma.setText("Sigma: -")

        self._validation_label.setText("")
        self._apply_btn.setEnabled(False)

    def load_yaml_config(self, config_path: Path) -> None:
        """Load data configuration from YAML file (FR-020).

        Args:
            config_path: Path to YAML configuration file
        """
        import yaml

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Check for data file path in config
            if "data" in config and "file" in config["data"]:
                data_file = Path(config["data"]["file"])
                if not data_file.is_absolute():
                    data_file = config_path.parent / data_file

                # Set format if specified
                if "format" in config["data"]:
                    format_value = config["data"]["format"]
                    for i in range(self._format_combo.count()):
                        if self._format_combo.itemData(i) == format_value:
                            self._format_combo.setCurrentIndex(i)
                            break

                # Load the file
                self._load_file(data_file)

                # Set column assignments if specified
                if "columns" in config["data"]:
                    # Column selector will be updated when file loads
                    # Additional column configuration can be applied here
                    pass

        except Exception as e:
            self._show_error("YAML Config Error", f"Failed to load configuration: {e}")
