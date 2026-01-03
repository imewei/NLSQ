"""
NLSQ Qt GUI Model Selection Page

This page allows users to select a mathematical model for fitting,
including built-in models, polynomial models, or custom Python functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from nlsq.gui_qt.widgets.code_editor import CodeEditorWidget

if TYPE_CHECKING:
    from collections.abc import Callable

    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ModelSelectionPage"]

# Built-in model display names
BUILTIN_MODELS = [
    ("Exponential Decay", "exponential_decay"),
    ("Gaussian", "gaussian"),
    ("Lorentzian", "lorentzian"),
    ("Double Exponential", "double_exponential"),
    ("Damped Oscillation", "damped_oscillation"),
    ("Power Law", "power_law"),
    ("Logistic (Sigmoid)", "logistic"),
]


class ModelSelectionPage(QWidget):
    """Page for selecting and configuring model function.

    Provides:
    - Model type selector (Built-in, Polynomial, Custom)
    - Built-in model dropdown with 7 models
    - Polynomial degree slider (0-10)
    - Custom code editor with syntax highlighting
    - Equation preview with LaTeX rendering
    - Parameter names display
    """

    # Signals
    model_selected = Signal(str, dict)  # model_type, config

    def __init__(self, app_state: AppState) -> None:
        """Initialize the model selection page.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._current_model: Callable | None = None
        self._current_model_type: str = "builtin"
        self._current_config: dict[str, Any] = {}

        self._setup_ui()
        self._connect_signals()

        # Initialize with first built-in model
        self._on_builtin_model_changed(0)

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Model Selection")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Model selection
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Preview
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions (50% left, 50% right)
        splitter.setSizes([500, 500])
        layout.addWidget(splitter, 1)

        # Action buttons
        self._create_action_buttons(layout)

    def _create_left_panel(self) -> QWidget:
        """Create the left panel with model selection options."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        # Model type selector
        type_group = QGroupBox("Model Type")
        type_layout = QVBoxLayout(type_group)

        self._type_button_group = QButtonGroup()
        self._builtin_radio = QRadioButton("Built-in Model")
        self._polynomial_radio = QRadioButton("Polynomial")
        self._custom_radio = QRadioButton("Custom Function")

        self._type_button_group.addButton(self._builtin_radio, 0)
        self._type_button_group.addButton(self._polynomial_radio, 1)
        self._type_button_group.addButton(self._custom_radio, 2)

        self._builtin_radio.setChecked(True)

        type_layout.addWidget(self._builtin_radio)
        type_layout.addWidget(self._polynomial_radio)
        type_layout.addWidget(self._custom_radio)
        layout.addWidget(type_group)

        # Built-in model section
        self._builtin_group = QGroupBox("Built-in Models")
        builtin_layout = QVBoxLayout(self._builtin_group)

        self._builtin_combo = QComboBox()
        for display_name, value in BUILTIN_MODELS:
            self._builtin_combo.addItem(display_name, value)
        builtin_layout.addWidget(self._builtin_combo)

        # Model description
        self._builtin_desc = QLabel("")
        self._builtin_desc.setWordWrap(True)
        self._builtin_desc.setStyleSheet("color: gray;")
        builtin_layout.addWidget(self._builtin_desc)

        layout.addWidget(self._builtin_group)

        # Polynomial section
        self._polynomial_group = QGroupBox("Polynomial Degree")
        poly_layout = QVBoxLayout(self._polynomial_group)

        # Degree slider
        slider_row = QHBoxLayout()
        slider_label = QLabel("Degree:")
        self._degree_slider = QSlider(Qt.Orientation.Horizontal)
        self._degree_slider.setRange(0, 10)
        self._degree_slider.setValue(2)
        self._degree_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._degree_slider.setTickInterval(1)

        self._degree_value = QLabel("2")
        self._degree_value.setMinimumWidth(30)

        slider_row.addWidget(slider_label)
        slider_row.addWidget(self._degree_slider, 1)
        slider_row.addWidget(self._degree_value)
        poly_layout.addLayout(slider_row)

        layout.addWidget(self._polynomial_group)
        self._polynomial_group.setVisible(False)

        # Custom function section
        self._custom_group = QGroupBox("Custom Function")
        custom_layout = QVBoxLayout(self._custom_group)

        info_label = QLabel(
            "Define a model function using JAX-compatible NumPy operations.\n"
            "The first parameter must be x (independent variable)."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        custom_layout.addWidget(info_label)

        self._code_editor = CodeEditorWidget()
        self._code_editor.setMaximumHeight(200)
        custom_layout.addWidget(self._code_editor)

        # Function selector
        func_row = QHBoxLayout()
        func_label = QLabel("Function:")
        self._function_combo = QComboBox()
        self._function_combo.setEnabled(False)
        func_row.addWidget(func_label)
        func_row.addWidget(self._function_combo, 1)
        custom_layout.addLayout(func_row)

        layout.addWidget(self._custom_group)
        self._custom_group.setVisible(False)

        layout.addStretch()
        return panel

    def _create_right_panel(self) -> QWidget:
        """Create the right panel with model preview."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)

        # Equation preview
        eq_group = QGroupBox("Equation")
        eq_layout = QVBoxLayout(eq_group)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(100)

        self._equation_label = QLabel("")
        self._equation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._equation_label.setStyleSheet("font-size: 16px; padding: 16px;")
        self._equation_label.setTextFormat(Qt.TextFormat.RichText)
        scroll.setWidget(self._equation_label)
        eq_layout.addWidget(scroll)

        layout.addWidget(eq_group)

        # Parameter names
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)

        self._param_list = QLabel("")
        self._param_list.setWordWrap(True)
        param_layout.addWidget(self._param_list)

        layout.addWidget(param_group)

        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout(info_group)

        self._info_p0 = QLabel("Auto initial guess: -")
        self._info_bounds = QLabel("Auto bounds: -")
        self._info_jit = QLabel("JIT compatible: -")

        info_layout.addWidget(self._info_p0)
        info_layout.addWidget(self._info_bounds)
        info_layout.addWidget(self._info_jit)

        layout.addWidget(info_group)

        layout.addStretch()
        return panel

    def _create_action_buttons(self, parent_layout: QVBoxLayout) -> None:
        """Create action buttons row."""
        button_row = QHBoxLayout()
        button_row.addStretch()

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset to default model")
        button_row.addWidget(self._reset_btn)

        self._apply_btn = QPushButton("Apply Model")
        self._apply_btn.setToolTip("Apply the selected model for curve fitting")
        button_row.addWidget(self._apply_btn)

        parent_layout.addLayout(button_row)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        self._type_button_group.idClicked.connect(self._on_type_changed)
        self._builtin_combo.currentIndexChanged.connect(self._on_builtin_model_changed)
        self._degree_slider.valueChanged.connect(self._on_degree_changed)
        self._code_editor.code_changed.connect(self._on_code_changed)
        self._function_combo.currentIndexChanged.connect(self._on_function_selected)
        self._reset_btn.clicked.connect(self.reset)
        self._apply_btn.clicked.connect(self._on_apply)

    def _on_type_changed(self, type_id: int) -> None:
        """Handle model type change.

        Args:
            type_id: The type button ID (0=builtin, 1=polynomial, 2=custom)
        """
        # Show/hide relevant sections
        self._builtin_group.setVisible(type_id == 0)
        self._polynomial_group.setVisible(type_id == 1)
        self._custom_group.setVisible(type_id == 2)

        # Update current type
        type_map = {0: "builtin", 1: "polynomial", 2: "custom"}
        self._current_model_type = type_map[type_id]

        # Trigger appropriate update
        if type_id == 0:
            self._on_builtin_model_changed(self._builtin_combo.currentIndex())
        elif type_id == 1:
            self._on_degree_changed(self._degree_slider.value())
        else:
            self._on_code_changed(self._code_editor.get_code())

    def _on_builtin_model_changed(self, index: int) -> None:
        """Handle built-in model selection change.

        Args:
            index: The selected combo box index
        """
        from nlsq.gui_qt.adapters.model_adapter import (
            get_latex_equation,
            get_model,
            get_model_info,
        )

        model_name = self._builtin_combo.itemData(index)
        if not model_name:
            return

        self._current_model_type = "builtin"
        self._current_config = {"name": model_name}

        try:
            model = get_model("builtin", self._current_config)
            self._current_model = model

            # Get model info
            info = get_model_info(model)

            # Update equation preview
            equation = get_latex_equation(model_name)
            self._equation_label.setText(f"<pre>{equation}</pre>")

            # Update parameter list
            params = info.get("param_names", [])
            if params:
                self._param_list.setText(f"Parameters: {', '.join(params)}")
            else:
                self._param_list.setText("Parameters: (none)")

            # Update info
            has_p0 = info.get("has_estimate_p0", False)
            has_bounds = info.get("has_bounds", False)

            self._info_p0.setText(f"Auto initial guess: {'Yes' if has_p0 else 'No'}")
            self._info_bounds.setText(f"Auto bounds: {'Yes' if has_bounds else 'No'}")
            self._info_jit.setText("JIT compatible: Yes")

            # Update description
            descriptions = {
                "exponential_decay": "y = a * exp(-b * x) + c",
                "gaussian": "y = A * exp(-(x-mu)^2 / (2*sigma^2))",
                "lorentzian": "y = A / (1 + ((x-x0)/gamma)^2)",
                "double_exponential": "y = a1*exp(-b1*x) + a2*exp(-b2*x) + c",
                "damped_oscillation": "y = A * exp(-gamma*x) * cos(omega*x + phi)",
                "power_law": "y = a * x^b",
                "logistic": "y = L / (1 + exp(-k*(x-x0))) + b",
            }
            self._builtin_desc.setText(descriptions.get(model_name, ""))

        except Exception as e:
            self._show_error("Model Error", str(e))

    def _on_degree_changed(self, degree: int) -> None:
        """Handle polynomial degree change.

        Args:
            degree: The polynomial degree
        """
        from nlsq.gui_qt.adapters.model_adapter import get_model, get_polynomial_latex

        self._degree_value.setText(str(degree))
        self._current_model_type = "polynomial"
        self._current_config = {"degree": degree}

        try:
            model = get_model("polynomial", self._current_config)
            self._current_model = model

            # Update equation preview
            equation = get_polynomial_latex(degree)
            self._equation_label.setText(f"<pre>{equation}</pre>")

            # Update parameter list
            params = [f"c{i}" for i in range(degree + 1)]
            self._param_list.setText(f"Parameters: {', '.join(params)}")

            # Update info
            self._info_p0.setText("Auto initial guess: Yes")
            self._info_bounds.setText("Auto bounds: Yes")
            self._info_jit.setText("JIT compatible: Yes")

        except Exception as e:
            self._show_error("Model Error", str(e))

    def _on_code_changed(self, code: str) -> None:
        """Handle custom code change.

        Args:
            code: The code content
        """
        # Update function combo
        funcs = self._code_editor.get_function_names()
        self._function_combo.clear()

        if funcs:
            self._function_combo.setEnabled(True)
            for func in funcs:
                self._function_combo.addItem(func)
            self._on_function_selected(0)
        else:
            self._function_combo.setEnabled(False)
            self._equation_label.setText("")
            self._param_list.setText("Parameters: (define a function)")
            self._info_p0.setText("Auto initial guess: No")
            self._info_bounds.setText("Auto bounds: No")
            self._info_jit.setText("JIT compatible: -")

    def _on_function_selected(self, index: int) -> None:
        """Handle function selection from custom code.

        Args:
            index: The selected function index
        """
        from nlsq.gui_qt.adapters.model_adapter import (
            parse_custom_model_string,
            validate_jit_compatibility,
        )

        if index < 0:
            return

        func_name = self._function_combo.currentText()
        if not func_name:
            return

        code = self._code_editor.get_code()
        self._current_model_type = "custom"
        self._current_config = {"code": code, "function": func_name}

        try:
            func, param_names = parse_custom_model_string(code, func_name)
            self._current_model = func

            # Update equation preview
            self._equation_label.setText(f"<pre>y = {func_name}(x, ...)</pre>")

            # Update parameter list
            if param_names:
                self._param_list.setText(f"Parameters: {', '.join(param_names)}")
            else:
                self._param_list.setText("Parameters: (none after x)")

            # Check JIT compatibility
            is_jit = validate_jit_compatibility(code)
            self._info_p0.setText("Auto initial guess: No")
            self._info_bounds.setText("Auto bounds: No")
            self._info_jit.setText(f"JIT compatible: {'Likely' if is_jit else 'Check'}")

        except Exception as e:
            self._equation_label.setText(f"<span style='color: red;'>Error: {e}</span>")
            self._param_list.setText("Parameters: (error)")

    def _on_apply(self) -> None:
        """Apply the selected model to app state."""
        if self._current_model is None:
            self._show_error("No Model", "Please select or define a model function.")
            return

        # Validate custom code syntax
        if self._current_model_type == "custom":
            is_valid, message = self._code_editor.validate_syntax()
            if not is_valid:
                self._show_error("Syntax Error", message)
                return

        # Update app state
        self._app_state.set_model(
            self._current_model_type,
            self._current_config,
            self._current_model,
        )

        # Emit signal
        self.model_selected.emit(self._current_model_type, self._current_config)

        QMessageBox.information(
            self,
            "Model Applied",
            f"Model '{self._current_model_type}' applied successfully.\n"
            "You can now proceed to Fitting Options.",
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
        self._code_editor.set_theme(theme)

    def reset(self) -> None:
        """Reset the page to initial state."""
        self._builtin_radio.setChecked(True)
        self._on_type_changed(0)
        self._builtin_combo.setCurrentIndex(0)
        self._degree_slider.setValue(2)
        self._code_editor.set_code("")
        self._function_combo.clear()
