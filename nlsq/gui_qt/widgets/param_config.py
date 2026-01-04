"""
NLSQ Qt GUI Parameter Configuration Widget

This widget allows users to configure parameter initial values and bounds
for curve fitting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ParamConfigWidget"]


class ParamConfigWidget(QWidget):
    """Widget for configuring parameter initial values and bounds.

    Provides:
    - Table with parameter names
    - Initial value inputs
    - Lower/upper bound inputs
    - Auto-estimate checkbox
    """

    # Signal emitted when configuration changes
    config_changed = Signal(list, tuple)  # p0, bounds

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the parameter config widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._param_names: list[str] = []
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Auto-estimate checkbox
        auto_row = QHBoxLayout()
        self._auto_p0_check = QCheckBox("Auto-estimate initial values")
        self._auto_p0_check.setChecked(True)
        self._auto_p0_check.setToolTip(
            "Use model's built-in p0 estimation if available"
        )
        auto_row.addWidget(self._auto_p0_check)
        auto_row.addStretch()
        layout.addLayout(auto_row)

        # Parameter table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["Parameter", "Initial Value", "Lower Bound", "Upper Bound"]
        )

        # Configure header
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self._table)

        # Info label
        self._info_label = QLabel("Set parameters to configure fit")
        self._info_label.setStyleSheet("color: gray;")
        layout.addWidget(self._info_label)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._auto_p0_check.toggled.connect(self._on_auto_toggled)

    def _on_auto_toggled(self, checked: bool) -> None:
        """Handle auto-estimate toggle.

        Args:
            checked: Whether auto-estimate is enabled
        """
        self._table.setEnabled(not checked)
        self._emit_config()

    def _emit_config(self) -> None:
        """Emit the current configuration."""
        p0, bounds = self.get_values()
        self.config_changed.emit(p0, bounds)

    def set_param_names(self, names: list[str]) -> None:
        """Set the parameter names.

        Args:
            names: List of parameter names
        """
        self._param_names = names
        self._table.setRowCount(len(names))

        for row, name in enumerate(names):
            # Parameter name (read-only)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Initial value spinbox
            p0_spin = QDoubleSpinBox()
            p0_spin.setRange(-1e10, 1e10)
            p0_spin.setDecimals(6)
            p0_spin.setValue(1.0)
            p0_spin.valueChanged.connect(self._emit_config)
            self._table.setCellWidget(row, 1, p0_spin)

            # Lower bound spinbox
            lower_spin = QDoubleSpinBox()
            lower_spin.setRange(-1e10, 1e10)
            lower_spin.setDecimals(6)
            lower_spin.setValue(-1e10)
            lower_spin.setSpecialValueText("-inf")
            lower_spin.valueChanged.connect(self._emit_config)
            self._table.setCellWidget(row, 2, lower_spin)

            # Upper bound spinbox
            upper_spin = QDoubleSpinBox()
            upper_spin.setRange(-1e10, 1e10)
            upper_spin.setDecimals(6)
            upper_spin.setValue(1e10)
            upper_spin.setSpecialValueText("+inf")
            upper_spin.valueChanged.connect(self._emit_config)
            self._table.setCellWidget(row, 3, upper_spin)

        self._info_label.setText(f"{len(names)} parameters")
        self._emit_config()

    def set_values(
        self, p0: list[float], bounds: tuple[list[float], list[float]] | None
    ) -> None:
        """Set the parameter values.

        Args:
            p0: Initial parameter values
            bounds: Parameter bounds as (lower, upper) or None
        """
        for row, val in enumerate(p0):
            if row >= self._table.rowCount():
                break

            p0_spin = self._table.cellWidget(row, 1)
            if isinstance(p0_spin, QDoubleSpinBox):
                p0_spin.setValue(val)

        if bounds is not None:
            lower, upper = bounds
            for row in range(min(len(lower), self._table.rowCount())):
                lower_spin = self._table.cellWidget(row, 2)
                if isinstance(lower_spin, QDoubleSpinBox):
                    lower_spin.setValue(lower[row])

                upper_spin = self._table.cellWidget(row, 3)
                if isinstance(upper_spin, QDoubleSpinBox):
                    upper_spin.setValue(upper[row])

    def get_values(self) -> tuple[list[float], tuple[list[float], list[float]] | None]:
        """Get the current parameter values.

        Returns:
            Tuple of (p0, bounds) where bounds is (lower, upper) or None
        """
        if self._auto_p0_check.isChecked():
            return [], None

        p0: list[float] = []
        lower: list[float] = []
        upper: list[float] = []

        for row in range(self._table.rowCount()):
            p0_spin = self._table.cellWidget(row, 1)
            lower_spin = self._table.cellWidget(row, 2)
            upper_spin = self._table.cellWidget(row, 3)

            if isinstance(p0_spin, QDoubleSpinBox):
                p0.append(p0_spin.value())
            if isinstance(lower_spin, QDoubleSpinBox):
                lower.append(lower_spin.value())
            if isinstance(upper_spin, QDoubleSpinBox):
                upper.append(upper_spin.value())

        bounds = (lower, upper) if lower and upper else None
        return p0, bounds

    def is_auto_p0(self) -> bool:
        """Check if auto-estimate is enabled.

        Returns:
            True if auto-estimate is checked
        """
        return self._auto_p0_check.isChecked()

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        pass  # Theme is applied globally via Qt color scheme
