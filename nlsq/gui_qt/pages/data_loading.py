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

        # Title — 24px bold, consistent across all pages
        title = QLabel("Data Loading")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Data input (40% config, 60% visualization is the standard)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(8)

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
        right_layout.setSpacing(8)

        # Data preview
        self._create_preview_section(right_layout)

        # Statistics
        self._create_statistics_section(right_layout)

        splitter.addWidget(right_panel)

        # Standard 40/60 split: config panel / data visualization
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
            "3.0    7.8",
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
        button_row.setSpacing(8)
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
            self._on_column_selection_changed,
        )
        self._clear_btn.clicked.connect(self._on_clear)
        self._apply_btn.clicked.connect(self._on_apply)
        self._app_state.data_changed.connect(self._on_app_state_data_changed)

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
        from nlsq.cli.data_loaders import DataLoader
        from nlsq.gui_qt.adapters.data_adapter import (
            load_from_file,
            parse_table_text,
        )

        format_value = self._format_combo.currentData()
        config: dict[str, Any] = {
            "format": format_value,
            "columns": {"x": 0, "y": 1},
        }

        try:
            file_format = DataLoader().detect_format(file_path, config)
            if file_format in ("ascii", "csv"):
                # Load the whole numeric table so the ColumnSelector can pick
                # sigma / z columns (same parsing as the clipboard path).
                rows, header = parse_table_text(
                    file_path.read_text(encoding="utf-8-sig")
                )
                if not rows:
                    raise ValueError("No valid numeric data found.")
                self._set_table(rows, header)
            else:
                # npz / hdf5: keys, not columns, define the arrays
                xdata, ydata, sigma = load_from_file(str(file_path), config)
                cols = [xdata.T if xdata.ndim == 2 else xdata, ydata]
                if sigma is not None:
                    cols.append(sigma)
                self._raw_data = np.column_stack(cols)
                self._column_names = [
                    f"Column {i}" for i in range(self._raw_data.shape[1])
                ]

            # Update UI
            self._file_path_label.setText(str(file_path))
            self._file_path_label.setStyleSheet("")
            self._column_selector.set_columns(self._column_names)
            self._update_preview()

        except Exception as e:
            self._show_error("File Load Error", str(e))

    def _set_table(self, rows: list[list[float]], header: list[str] | None) -> None:
        """Store parsed rows; drop all-NaN columns with no header (trailing delimiter)."""
        raw = np.array(rows, dtype=np.float64)
        names = header if header and len(header) == raw.shape[1] else None
        keep = [
            i
            for i in range(raw.shape[1])
            if not (np.isnan(raw[:, i]).all() and not (names and names[i]))
        ]
        self._raw_data = raw[:, keep]
        self._column_names = (
            [names[i] for i in keep]
            if names
            else [f"Column {i}" for i in range(len(keep))]
        )

    def _on_parse_clipboard(self) -> None:
        """Handle parse clipboard button click."""
        from nlsq.gui_qt.adapters.data_adapter import parse_table_text

        raw_text = self._clipboard_text.toPlainText()
        text = raw_text.strip()
        if not text:
            self._show_error("No Data", "Please paste data into the text area first.")
            return

        try:
            rows, header = parse_table_text(raw_text)

            if not rows:
                self._show_error("Parse Error", "No valid numeric data found.")
                return

            # Convert to numpy array
            self._set_table(rows, header)

            # Update UI
            self._file_path_label.setText("(Clipboard data)")
            self._file_path_label.setStyleSheet("color: green;")
            self._column_selector.set_columns(self._column_names)
            self._update_preview()

        except Exception as e:
            # Broadened from ValueError: numpy/str parsing on arbitrary
            # pasted text can also raise other exception types (matches the
            # broad `except Exception` used elsewhere in this file), and an
            # uncaught one here would crash the page instead of showing the
            # existing "Parse Error" dialog.
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

    def _clear_stats(self) -> None:
        self._stats_points.setText("Points: -")
        self._stats_x_range.setText("X Range: -")
        self._stats_y_range.setText("Y Range: -")
        self._stats_sigma.setText("Sigma: -")

    def _validate_and_update_stats(self) -> None:
        """Validate data and update statistics display."""
        from nlsq.gui_qt.adapters.data_adapter import compute_statistics, validate_data

        if self._xdata is None or self._ydata is None:
            self._clear_stats()
            return

        # Validate
        result = validate_data(self._xdata, self._ydata, self._sigma)

        if not result.is_valid:
            self._clear_stats()  # never leave the previous dataset's numbers up
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
            f"X Range: [{stats['x_min']:.4g}, {stats['x_max']:.4g}]",
        )
        self._stats_y_range.setText(
            f"Y Range: [{stats['y_min']:.4g}, {stats['y_max']:.4g}]",
        )

        if stats.get("has_sigma"):
            self._stats_sigma.setText(
                f"Sigma Range: [{stats['sigma_min']:.4g}, {stats['sigma_max']:.4g}]",
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
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                )
                row_items.append(item)
            self._preview_model.appendRow(row_items)

        # Update info
        if n_rows > preview_rows:
            self._preview_info.setText(
                f"Showing first {preview_rows} of {n_rows:,} rows",
            )
        else:
            self._preview_info.setText(f"{n_rows:,} rows, {n_cols} columns")

    def _on_clear(self) -> None:
        """Clear button: drop the page's data and the applied AppState data."""
        self.reset()
        self._app_state.set_data(None, None)

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

    def _on_app_state_data_changed(self) -> None:
        """Refresh the page when AppState data changes from outside this page.

        This page's own `_on_apply` also triggers `data_changed`, but in that
        case the state arrays are the exact objects already displayed (same
        identity), so this is a no-op then. It only does real work for
        external changes such as crash-recovery restoring a session.
        """
        state = self._app_state.state
        if state.xdata is None or state.ydata is None:
            return
        if state.xdata is self._xdata and state.ydata is self._ydata:
            return

        self._xdata = state.xdata
        self._ydata = state.ydata
        self._sigma = state.sigma

        # 2-D xdata is stored (k, n); the preview table wants one column per x
        x_cols = self._xdata.T if self._xdata.ndim == 2 else self._xdata
        is_surface = self._xdata.ndim == 2 and self._xdata.shape[0] == 2
        if self._sigma is not None:
            self._raw_data = np.column_stack([x_cols, self._ydata, self._sigma])
            self._column_names = (
                ["x", "y", "z", "sigma"] if is_surface else ["x", "y", "sigma"]
            )
        else:
            self._raw_data = np.column_stack([x_cols, self._ydata])
            self._column_names = ["x", "y", "z"] if is_surface else ["x", "y"]
        if is_surface:
            # columns are [x0, x1, z(, sigma)]: use the selector's 2-D layout so
            # a later combo change keeps the 2-D mapping instead of x0/x1/z -> x/y/sigma
            self._column_selector.set_mode("2d")
        else:
            self._column_selector.set_mode("1d")

        self._file_path_label.setText(state.data_file_name or "(Restored session)")
        self._file_path_label.setStyleSheet("color: green;")
        self._column_selector.set_columns(self._column_names)
        # set_columns re-runs the default column selection, which would keep
        # only column 0 of a 2-D x; show exactly what the session holds.
        self._xdata = state.xdata
        self._ydata = state.ydata
        self._sigma = state.sigma
        self._update_preview()
        self._validate_and_update_stats()

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

            data_config = config.get("data", {})
            input_file = data_config.get("input_file", data_config.get("file"))
            if input_file:
                data_file = Path(input_file)
                if not data_file.is_absolute():
                    data_file = config_path.parent / data_file

                # Set format if specified
                if "format" in data_config:
                    format_value = data_config["format"]
                    for i in range(self._format_combo.count()):
                        if self._format_combo.itemData(i) == format_value:
                            self._format_combo.setCurrentIndex(i)
                            break

                # Load the file
                self._load_file(data_file)

                columns = {
                    "x": 0,
                    "y": 1,
                    "z": None,
                    "sigma": None,
                    **data_config.get("columns", {}),
                }
                self._column_selector.set_mode(
                    "2d" if columns["z"] is not None else "1d"
                )
                self._column_selector.set_selection(columns)

        except Exception as e:
            self._show_error("YAML Config Error", f"Failed to load configuration: {e}")
