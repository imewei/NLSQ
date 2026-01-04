"""
NLSQ Qt GUI Advanced Options Widget

This widget provides tabbed interface for advanced fitting options
including tolerances, multi-start, streaming, and defense layers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["AdvancedOptionsWidget"]


class AdvancedOptionsWidget(QWidget):
    """Widget for advanced fitting options with tabbed interface.

    Provides tabs for:
    - Termination criteria (gtol, ftol, xtol, max_iter)
    - Multi-start configuration
    - Streaming options (chunk_size, normalize)
    - Defense layers (layer toggles and thresholds)
    """

    # Signal emitted when options change
    options_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize the advanced options widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tab widget
        self._tabs = QTabWidget()

        # Termination tab
        self._tabs.addTab(self._create_termination_tab(), "Termination")

        # Multi-start tab
        self._tabs.addTab(self._create_multistart_tab(), "Multi-Start")

        # Streaming tab
        self._tabs.addTab(self._create_streaming_tab(), "Streaming")

        # Defense layers tab
        self._tabs.addTab(self._create_defense_tab(), "Defense Layers")

        layout.addWidget(self._tabs)

    def _create_termination_tab(self) -> QWidget:
        """Create the termination criteria tab."""
        widget = QWidget()
        layout = QFormLayout(widget)

        # gtol
        self._gtol_spin = QDoubleSpinBox()
        self._gtol_spin.setDecimals(12)
        self._gtol_spin.setRange(1e-15, 1.0)
        self._gtol_spin.setValue(1e-8)
        self._gtol_spin.setToolTip("Gradient tolerance for convergence")
        layout.addRow("Gradient Tolerance (gtol):", self._gtol_spin)

        # ftol
        self._ftol_spin = QDoubleSpinBox()
        self._ftol_spin.setDecimals(12)
        self._ftol_spin.setRange(1e-15, 1.0)
        self._ftol_spin.setValue(1e-8)
        self._ftol_spin.setToolTip("Function tolerance for convergence")
        layout.addRow("Function Tolerance (ftol):", self._ftol_spin)

        # xtol
        self._xtol_spin = QDoubleSpinBox()
        self._xtol_spin.setDecimals(12)
        self._xtol_spin.setRange(1e-15, 1.0)
        self._xtol_spin.setValue(1e-8)
        self._xtol_spin.setToolTip("Parameter tolerance for convergence")
        layout.addRow("Parameter Tolerance (xtol):", self._xtol_spin)

        # max_iterations
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 100000)
        self._max_iter_spin.setValue(200)
        self._max_iter_spin.setToolTip("Maximum number of iterations")
        layout.addRow("Max Iterations:", self._max_iter_spin)

        # max_function_evals
        self._max_fev_spin = QSpinBox()
        self._max_fev_spin.setRange(1, 1000000)
        self._max_fev_spin.setValue(2000)
        self._max_fev_spin.setToolTip("Maximum number of function evaluations")
        layout.addRow("Max Function Evals:", self._max_fev_spin)

        return widget

    def _create_multistart_tab(self) -> QWidget:
        """Create the multi-start options tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Enable checkbox
        self._multistart_check = QCheckBox("Enable Multi-Start Optimization")
        self._multistart_check.setToolTip(
            "Run optimization from multiple starting points"
        )
        layout.addWidget(self._multistart_check)

        # Options group (disabled when unchecked)
        self._multistart_group = QGroupBox("Multi-Start Options")
        self._multistart_group.setEnabled(False)
        group_layout = QFormLayout(self._multistart_group)

        # n_starts
        self._n_starts_spin = QSpinBox()
        self._n_starts_spin.setRange(2, 1000)
        self._n_starts_spin.setValue(10)
        self._n_starts_spin.setToolTip("Number of starting points")
        group_layout.addRow("Number of Starts:", self._n_starts_spin)

        # sampler
        self._sampler_combo = QComboBox()
        self._sampler_combo.addItems(["lhs", "sobol", "halton", "random"])
        self._sampler_combo.setToolTip("Sampling method for starting points")
        group_layout.addRow("Sampler:", self._sampler_combo)

        # center_on_p0
        self._center_check = QCheckBox("Center on initial guess")
        self._center_check.setChecked(True)
        group_layout.addRow("", self._center_check)

        # scale_factor
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.01, 100.0)
        self._scale_spin.setValue(1.0)
        self._scale_spin.setDecimals(2)
        self._scale_spin.setToolTip("Scale factor for sampling region")
        group_layout.addRow("Scale Factor:", self._scale_spin)

        layout.addWidget(self._multistart_group)
        layout.addStretch()

        # Connect enable checkbox
        self._multistart_check.toggled.connect(self._multistart_group.setEnabled)

        return widget

    def _create_streaming_tab(self) -> QWidget:
        """Create the streaming options tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Streaming mode is automatically enabled for large datasets (>500K points)."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        # Options
        form = QFormLayout()

        # chunk_size
        self._chunk_spin = QSpinBox()
        self._chunk_spin.setRange(1000, 1000000)
        self._chunk_spin.setValue(10000)
        self._chunk_spin.setSingleStep(1000)
        self._chunk_spin.setToolTip("Number of points per chunk")
        form.addRow("Chunk Size:", self._chunk_spin)

        # normalize
        self._normalize_check = QCheckBox("Normalize Parameters")
        self._normalize_check.setChecked(True)
        self._normalize_check.setToolTip("Normalize parameters for streaming")
        form.addRow("", self._normalize_check)

        # warmup_iterations
        self._warmup_spin = QSpinBox()
        self._warmup_spin.setRange(10, 2000)
        self._warmup_spin.setValue(200)
        self._warmup_spin.setToolTip("Warmup iterations on first chunk")
        form.addRow("Warmup Iterations:", self._warmup_spin)

        layout.addLayout(form)
        layout.addStretch()

        return widget

    def _create_defense_tab(self) -> QWidget:
        """Create the defense layers tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Defense layers provide stability for streaming optimization."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        # Preset selector
        preset_layout = QHBoxLayout()
        preset_label = QLabel("Preset:")
        self._defense_preset = QComboBox()
        self._defense_preset.addItems(
            ["default", "conservative", "aggressive", "custom"]
        )
        self._defense_preset.setToolTip("Defense layer preset configuration")
        preset_layout.addWidget(preset_label)
        preset_layout.addWidget(self._defense_preset, 1)
        layout.addLayout(preset_layout)

        # Layer toggles
        self._layer1_check = QCheckBox("Layer 1: Warm Start Detection")
        self._layer1_check.setChecked(True)
        layout.addWidget(self._layer1_check)

        self._layer2_check = QCheckBox("Layer 2: Adaptive Learning Rate")
        self._layer2_check.setChecked(True)
        layout.addWidget(self._layer2_check)

        self._layer3_check = QCheckBox("Layer 3: Cost Guard")
        self._layer3_check.setChecked(True)
        layout.addWidget(self._layer3_check)

        self._layer4_check = QCheckBox("Layer 4: Step Clipping")
        self._layer4_check.setChecked(True)
        layout.addWidget(self._layer4_check)

        layout.addStretch()

        return widget

    def _connect_signals(self) -> None:
        """Connect internal signals to emit options_changed."""
        # Termination
        self._gtol_spin.valueChanged.connect(self._emit_options)
        self._ftol_spin.valueChanged.connect(self._emit_options)
        self._xtol_spin.valueChanged.connect(self._emit_options)
        self._max_iter_spin.valueChanged.connect(self._emit_options)
        self._max_fev_spin.valueChanged.connect(self._emit_options)

        # Multi-start
        self._multistart_check.toggled.connect(self._emit_options)
        self._n_starts_spin.valueChanged.connect(self._emit_options)
        self._sampler_combo.currentIndexChanged.connect(self._emit_options)
        self._center_check.toggled.connect(self._emit_options)
        self._scale_spin.valueChanged.connect(self._emit_options)

        # Streaming
        self._chunk_spin.valueChanged.connect(self._emit_options)
        self._normalize_check.toggled.connect(self._emit_options)
        self._warmup_spin.valueChanged.connect(self._emit_options)

        # Defense
        self._defense_preset.currentIndexChanged.connect(self._emit_options)
        self._layer1_check.toggled.connect(self._emit_options)
        self._layer2_check.toggled.connect(self._emit_options)
        self._layer3_check.toggled.connect(self._emit_options)
        self._layer4_check.toggled.connect(self._emit_options)

    def _emit_options(self) -> None:
        """Emit the current options."""
        self.options_changed.emit(self.get_options())

    def get_options(self) -> dict[str, Any]:
        """Get the current options.

        Returns:
            Dictionary with all option values
        """
        return {
            # Termination
            "gtol": self._gtol_spin.value(),
            "ftol": self._ftol_spin.value(),
            "xtol": self._xtol_spin.value(),
            "max_iterations": self._max_iter_spin.value(),
            "max_function_evals": self._max_fev_spin.value(),
            # Multi-start
            "enable_multistart": self._multistart_check.isChecked(),
            "n_starts": self._n_starts_spin.value(),
            "sampler": self._sampler_combo.currentText(),
            "center_on_p0": self._center_check.isChecked(),
            "scale_factor": self._scale_spin.value(),
            # Streaming
            "chunk_size": self._chunk_spin.value(),
            "normalize": self._normalize_check.isChecked(),
            "warmup_iterations": self._warmup_spin.value(),
            # Defense
            "defense_preset": self._defense_preset.currentText(),
            "layer1_enabled": self._layer1_check.isChecked(),
            "layer2_enabled": self._layer2_check.isChecked(),
            "layer3_enabled": self._layer3_check.isChecked(),
            "layer4_enabled": self._layer4_check.isChecked(),
        }

    def set_options(self, options: dict[str, Any]) -> None:
        """Set the options.

        Args:
            options: Dictionary with option values
        """
        self._set_spinbox_options(options)
        self._set_checkbox_options(options)
        self._set_combo_options(options)

    def _set_spinbox_options(self, options: dict[str, Any]) -> None:
        """Set spinbox values from options dict."""
        spinbox_mapping = {
            "gtol": self._gtol_spin,
            "ftol": self._ftol_spin,
            "xtol": self._xtol_spin,
            "max_iterations": self._max_iter_spin,
            "max_function_evals": self._max_fev_spin,
            "n_starts": self._n_starts_spin,
            "scale_factor": self._scale_spin,
            "chunk_size": self._chunk_spin,
            "warmup_iterations": self._warmup_spin,
        }
        for key, spinbox in spinbox_mapping.items():
            if key in options:
                spinbox.setValue(options[key])

    def _set_checkbox_options(self, options: dict[str, Any]) -> None:
        """Set checkbox values from options dict."""
        checkbox_mapping = {
            "enable_multistart": self._multistart_check,
            "center_on_p0": self._center_check,
            "normalize": self._normalize_check,
            "layer1_enabled": self._layer1_check,
            "layer2_enabled": self._layer2_check,
            "layer3_enabled": self._layer3_check,
            "layer4_enabled": self._layer4_check,
        }
        for key, checkbox in checkbox_mapping.items():
            if key in options:
                checkbox.setChecked(options[key])

    def _set_combo_options(self, options: dict[str, Any]) -> None:
        """Set combobox values from options dict."""
        combo_mapping = {
            "sampler": self._sampler_combo,
            "defense_preset": self._defense_preset,
        }
        for key, combo in combo_mapping.items():
            if key in options:
                idx = combo.findText(options[key])
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def set_theme(self, theme: ThemeConfig) -> None:
        """Apply theme to this widget.

        Args:
            theme: Theme configuration
        """
        pass  # Theme is applied globally via Qt color scheme
