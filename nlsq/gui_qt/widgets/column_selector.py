"""
NLSQ Qt GUI Column Selector Widget

This widget allows users to assign data columns to roles (x, y, sigma, z)
for curve fitting. Supports both 1D and 2D data modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ColumnSelectorWidget"]


class ColumnSelectorWidget(QWidget):
    """Widget for assigning data columns to roles.

    Provides:
    - Mode selector (1D curve / 2D surface)
    - Column dropdowns for x, y, z (2D only), sigma (optional)
    - Validation feedback
    """

    # Signal emitted when column selection changes
    selection_changed = Signal(
        dict
    )  # {"x": int, "y": int, "sigma": int|None, "z": int|None}

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the column selector widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._column_names: list[str] = []
        self._mode: str = "1d"  # "1d" or "2d"
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Mode selector group
        mode_group = QGroupBox("Data Mode")
        mode_layout = QHBoxLayout(mode_group)

        self._mode_1d = QRadioButton("1D Curve")
        self._mode_2d = QRadioButton("2D Surface")
        self._mode_1d.setChecked(True)

        mode_layout.addWidget(self._mode_1d)
        mode_layout.addWidget(self._mode_2d)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # Column assignment group
        columns_group = QGroupBox("Column Assignment")
        columns_layout = QFormLayout(columns_group)
        columns_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        # X column selector
        self._x_combo = QComboBox()
        self._x_combo.setToolTip(
            "Select the column containing X values (independent variable)"
        )
        columns_layout.addRow("X Column:", self._x_combo)

        # Y column selector
        self._y_combo = QComboBox()
        self._y_combo.setToolTip(
            "Select the column containing Y values (dependent variable for 1D, second coordinate for 2D)"
        )
        columns_layout.addRow("Y Column:", self._y_combo)

        # Z column selector (2D mode only)
        self._z_combo = QComboBox()
        self._z_combo.setToolTip(
            "Select the column containing Z values (dependent variable for 2D surface)"
        )
        self._z_label = QLabel("Z Column:")
        columns_layout.addRow(self._z_label, self._z_combo)

        # Sigma column selector (optional)
        self._sigma_combo = QComboBox()
        self._sigma_combo.setToolTip(
            "Select the column containing measurement uncertainties (optional)"
        )
        columns_layout.addRow("Sigma (optional):", self._sigma_combo)

        layout.addWidget(columns_group)

        # Validation message
        self._validation_label = QLabel("")
        self._validation_label.setWordWrap(True)
        layout.addWidget(self._validation_label)

        # Initial state - hide 2D elements
        self._update_mode_visibility()

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._mode_1d.toggled.connect(self._on_mode_changed)
        self._x_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._y_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._z_combo.currentIndexChanged.connect(self._on_selection_changed)
        self._sigma_combo.currentIndexChanged.connect(self._on_selection_changed)

    def _on_mode_changed(self, checked: bool) -> None:
        """Handle mode radio button change.

        Args:
            checked: Whether 1D mode is checked
        """
        self._mode = "1d" if checked else "2d"
        self._update_mode_visibility()
        self._on_selection_changed()

    def _update_mode_visibility(self) -> None:
        """Update visibility of Z column based on mode."""
        is_2d = self._mode == "2d"
        self._z_label.setVisible(is_2d)
        self._z_combo.setVisible(is_2d)

        # Update Y column tooltip based on mode
        if is_2d:
            self._y_combo.setToolTip(
                "Select the column containing Y coordinates (second independent variable)"
            )
        else:
            self._y_combo.setToolTip(
                "Select the column containing Y values (dependent variable)"
            )

    def _on_selection_changed(self) -> None:
        """Handle column selection change."""
        selection = self.get_selection()
        is_valid, message = self.validate()

        # Update validation display
        if is_valid:
            self._validation_label.setText("")
            self._validation_label.setStyleSheet("")
        else:
            self._validation_label.setText(f"⚠ {message}")
            self._validation_label.setStyleSheet("color: #FF9800;")

        self.selection_changed.emit(selection)

    def set_columns(self, column_names: list[str]) -> None:
        """Set the available column names.

        Populates all dropdowns with the column names and attempts
        to auto-select sensible defaults.

        Args:
            column_names: List of column names from loaded data
        """
        self._column_names = column_names

        # Block signals during update
        for combo in [self._x_combo, self._y_combo, self._z_combo, self._sigma_combo]:
            combo.blockSignals(True)
            combo.clear()

        # Add "None" option for sigma
        self._sigma_combo.addItem("(None)", None)

        # Add column names to all combos
        for i, name in enumerate(column_names):
            display_name = f"{i}: {name}" if name else f"Column {i}"
            self._x_combo.addItem(display_name, i)
            self._y_combo.addItem(display_name, i)
            self._z_combo.addItem(display_name, i)
            self._sigma_combo.addItem(display_name, i)

        # Auto-select defaults based on number of columns
        n_cols = len(column_names)
        if n_cols >= 1:
            self._x_combo.setCurrentIndex(0)
        if n_cols >= 2:
            self._y_combo.setCurrentIndex(1)
        if n_cols >= 3:
            # For 3 columns in 1D mode, assume 3rd is sigma
            if self._mode == "1d":
                self._sigma_combo.setCurrentIndex(3)  # Index 3 because of "(None)" at 0
            else:
                self._z_combo.setCurrentIndex(2)
        if n_cols >= 4 and self._mode == "2d":
            self._sigma_combo.setCurrentIndex(4)  # Index 4 because of "(None)" at 0

        # Unblock signals and emit change
        for combo in [self._x_combo, self._y_combo, self._z_combo, self._sigma_combo]:
            combo.blockSignals(False)

        self._on_selection_changed()

    def get_selection(self) -> dict[str, int | None]:
        """Get the current column selection.

        Returns:
            Dictionary with column indices:
            - "x": int - X column index
            - "y": int - Y column index
            - "z": int | None - Z column index (2D mode only)
            - "sigma": int | None - Sigma column index (optional)
        """
        selection: dict[str, int | None] = {
            "x": self._x_combo.currentData(),
            "y": self._y_combo.currentData(),
            "z": None,
            "sigma": self._sigma_combo.currentData(),
        }

        if self._mode == "2d":
            selection["z"] = self._z_combo.currentData()

        return selection

    def set_mode(self, mode: str) -> None:
        """Set the data mode.

        Args:
            mode: Either "1d" for curve data or "2d" for surface data
        """
        if mode not in ("1d", "2d"):
            raise ValueError(f"Invalid mode: {mode}. Must be '1d' or '2d'.")

        self._mode = mode
        self._mode_1d.setChecked(mode == "1d")
        self._mode_2d.setChecked(mode == "2d")
        self._update_mode_visibility()

    def validate(self) -> tuple[bool, str]:
        """Validate the current column selection.

        Checks that:
        - X and Y columns are different
        - In 2D mode, Z column is different from X and Y
        - Sigma column (if set) is different from data columns

        Returns:
            Tuple of (is_valid, error_message)
        """
        selection = self.get_selection()

        x_idx = selection["x"]
        y_idx = selection["y"]
        z_idx = selection["z"]
        sigma_idx = selection["sigma"]

        # Check that columns are set
        if x_idx is None:
            return False, "X column not selected"
        if y_idx is None:
            return False, "Y column not selected"

        # Check X and Y are different
        if x_idx == y_idx:
            return False, "X and Y must be different columns"

        # Check 2D mode constraints
        if self._mode == "2d":
            if z_idx is None:
                return False, "Z column required for 2D mode"
            if z_idx in {x_idx, y_idx}:
                return False, "Z must be different from X and Y columns"

        # Check sigma uniqueness if set
        if sigma_idx is not None:
            data_cols = {x_idx, y_idx}
            if z_idx is not None:
                data_cols.add(z_idx)
            if sigma_idx in data_cols:
                return False, "Sigma must be different from data columns"

        return True, "Selection valid"

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        # Update validation label color based on theme
        # Warning color is consistent across themes
        pass  # Theme is applied globally via Qt color scheme
