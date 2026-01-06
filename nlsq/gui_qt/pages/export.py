"""
NLSQ Qt GUI Export Page

This page allows users to export fitting results in multiple formats
including session bundles, JSON, CSV, and Python code generation.
"""

from __future__ import annotations

import contextlib
import json
import zipfile
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

__all__ = ["ExportPage"]


class ExportPage(QWidget):
    """Page for exporting results.

    Provides:
    - Session bundle export (ZIP with data, config, results, plots)
    - JSON export for programmatic access
    - CSV export for spreadsheet compatibility
    - Python code generation for reproducibility
    - Copy-to-clipboard functionality
    """

    # Signals
    export_completed = Signal(str)  # file path

    def __init__(self, app_state: AppState) -> None:
        """Initialize the export page.

        Args:
            app_state: Application state manager
        """
        super().__init__()
        self._app_state = app_state
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up the page UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("Export")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Main content splitter
        splitter = QSplitter()

        # Left panel - Export options
        left_panel = self._create_export_options()
        splitter.addWidget(left_panel)

        # Right panel - Code preview
        right_panel = self._create_code_preview()
        splitter.addWidget(right_panel)

        splitter.setSizes([400, 600])
        layout.addWidget(splitter, 1)

        # Status row
        status_row = QHBoxLayout()
        self._status_label = QLabel("Ready to export")
        self._status_label.setStyleSheet("color: gray;")
        status_row.addWidget(self._status_label)
        status_row.addStretch()
        layout.addLayout(status_row)

    def _create_export_options(self) -> QWidget:
        """Create the export options panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)

        # Session bundle export
        bundle_group = QGroupBox("Session Bundle (ZIP)")
        bundle_layout = QVBoxLayout(bundle_group)

        bundle_desc = QLabel(
            "Export complete session including data, configuration, "
            "fit results, and plots as a ZIP archive."
        )
        bundle_desc.setWordWrap(True)
        bundle_desc.setStyleSheet("color: gray; font-size: 11px;")
        bundle_layout.addWidget(bundle_desc)

        self._export_bundle_btn = QPushButton("Export Session Bundle...")
        self._export_bundle_btn.setToolTip("Save complete session as ZIP file")
        bundle_layout.addWidget(self._export_bundle_btn)

        layout.addWidget(bundle_group)

        # JSON export
        json_group = QGroupBox("JSON Export")
        json_layout = QVBoxLayout(json_group)

        json_desc = QLabel(
            "Export fit results as structured JSON for programmatic access "
            "or integration with other tools."
        )
        json_desc.setWordWrap(True)
        json_desc.setStyleSheet("color: gray; font-size: 11px;")
        json_layout.addWidget(json_desc)

        json_btn_row = QHBoxLayout()
        self._export_json_btn = QPushButton("Export JSON...")
        self._export_json_btn.setToolTip("Save results as JSON file")
        json_btn_row.addWidget(self._export_json_btn)

        self._copy_json_btn = QPushButton("Copy to Clipboard")
        self._copy_json_btn.setToolTip("Copy JSON to clipboard")
        json_btn_row.addWidget(self._copy_json_btn)

        json_layout.addLayout(json_btn_row)
        layout.addWidget(json_group)

        # CSV export
        csv_group = QGroupBox("CSV Export")
        csv_layout = QVBoxLayout(csv_group)

        csv_desc = QLabel(
            "Export fitted parameters and data as CSV for spreadsheet "
            "applications like Excel or Google Sheets."
        )
        csv_desc.setWordWrap(True)
        csv_desc.setStyleSheet("color: gray; font-size: 11px;")
        csv_layout.addWidget(csv_desc)

        csv_btn_row = QHBoxLayout()
        self._export_params_csv_btn = QPushButton("Export Parameters...")
        self._export_params_csv_btn.setToolTip("Save fitted parameters as CSV")
        csv_btn_row.addWidget(self._export_params_csv_btn)

        self._export_data_csv_btn = QPushButton("Export Data...")
        self._export_data_csv_btn.setToolTip("Save data with fit as CSV")
        csv_btn_row.addWidget(self._export_data_csv_btn)

        csv_layout.addLayout(csv_btn_row)
        layout.addWidget(csv_group)

        # Python code generation
        code_group = QGroupBox("Python Code")
        code_layout = QVBoxLayout(code_group)

        code_desc = QLabel(
            "Generate reproducible Python code that recreates this fit "
            "using NLSQ library."
        )
        code_desc.setWordWrap(True)
        code_desc.setStyleSheet("color: gray; font-size: 11px;")
        code_layout.addWidget(code_desc)

        code_btn_row = QHBoxLayout()
        self._generate_code_btn = QPushButton("Generate Code")
        self._generate_code_btn.setToolTip("Generate Python code")
        code_btn_row.addWidget(self._generate_code_btn)

        self._copy_code_btn = QPushButton("Copy Code")
        self._copy_code_btn.setToolTip("Copy Python code to clipboard")
        code_btn_row.addWidget(self._copy_code_btn)

        self._export_code_btn = QPushButton("Save as .py...")
        self._export_code_btn.setToolTip("Save Python code to file")
        code_btn_row.addWidget(self._export_code_btn)

        code_layout.addLayout(code_btn_row)
        layout.addWidget(code_group)

        layout.addStretch()
        return panel

    def _create_code_preview(self) -> QWidget:
        """Create the code preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)

        code_group = QGroupBox("Code Preview")
        code_layout = QVBoxLayout(code_group)

        self._code_preview = QTextEdit()
        self._code_preview.setReadOnly(True)
        self._code_preview.setStyleSheet(
            "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
            "font-size: 12px;"
        )
        self._code_preview.setPlaceholderText(
            "Click 'Generate Code' to preview Python code..."
        )
        code_layout.addWidget(self._code_preview)

        layout.addWidget(code_group)
        return panel

    def _connect_signals(self) -> None:
        """Connect button signals."""
        self._export_bundle_btn.clicked.connect(self._on_export_bundle)
        self._export_json_btn.clicked.connect(self._on_export_json)
        self._copy_json_btn.clicked.connect(self._on_copy_json)
        self._export_params_csv_btn.clicked.connect(self._on_export_params_csv)
        self._export_data_csv_btn.clicked.connect(self._on_export_data_csv)
        self._generate_code_btn.clicked.connect(self._on_generate_code)
        self._copy_code_btn.clicked.connect(self._on_copy_code)
        self._export_code_btn.clicked.connect(self._on_export_code)

    def _on_export_bundle(self) -> None:
        """Handle export session bundle."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Session Bundle",
            "nlsq_session.zip",
            "ZIP Files (*.zip)",
        )
        if path:
            try:
                self.export_session_bundle(path)
                self._status_label.setText(f"Exported to {path}")
                self._status_label.setStyleSheet("color: #4CAF50;")
                self.export_completed.emit(path)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
                self._status_label.setText("Export failed")
                self._status_label.setStyleSheet("color: #f44336;")

    def _on_export_json(self) -> None:
        """Handle export JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export JSON",
            "fit_results.json",
            "JSON Files (*.json)",
        )
        if path:
            try:
                self.export_json(path)
                self._status_label.setText(f"Exported to {path}")
                self._status_label.setStyleSheet("color: #4CAF50;")
                self.export_completed.emit(path)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _on_copy_json(self) -> None:
        """Handle copy JSON to clipboard."""
        try:
            json_str = self._generate_json()
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(json_str)
                self._status_label.setText("JSON copied to clipboard")
                self._status_label.setStyleSheet("color: #4CAF50;")
        except Exception as e:
            QMessageBox.critical(self, "Copy Error", str(e))

    def _on_export_params_csv(self) -> None:
        """Handle export parameters CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Parameters",
            "fit_parameters.csv",
            "CSV Files (*.csv)",
        )
        if path:
            try:
                self.export_csv(path, include_data=False)
                self._status_label.setText(f"Exported to {path}")
                self._status_label.setStyleSheet("color: #4CAF50;")
                self.export_completed.emit(path)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _on_export_data_csv(self) -> None:
        """Handle export data CSV."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data with Fit",
            "fit_data.csv",
            "CSV Files (*.csv)",
        )
        if path:
            try:
                self.export_csv(path, include_data=True)
                self._status_label.setText(f"Exported to {path}")
                self._status_label.setStyleSheet("color: #4CAF50;")
                self.export_completed.emit(path)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _on_generate_code(self) -> None:
        """Handle generate code."""
        code = self.generate_python_code()
        self._code_preview.setPlainText(code)
        self._status_label.setText("Code generated")
        self._status_label.setStyleSheet("color: #4CAF50;")

    def _on_copy_code(self) -> None:
        """Handle copy code to clipboard."""
        code = self._code_preview.toPlainText()
        if not code:
            code = self.generate_python_code()
            self._code_preview.setPlainText(code)

        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(code)
            self._status_label.setText("Code copied to clipboard")
            self._status_label.setStyleSheet("color: #4CAF50;")

    def _on_export_code(self) -> None:
        """Handle export code to file."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python Code",
            "fit_script.py",
            "Python Files (*.py)",
        )
        if path:
            code = self._code_preview.toPlainText()
            if not code:
                code = self.generate_python_code()

            try:
                Path(path).write_text(code, encoding="utf-8")
                self._status_label.setText(f"Exported to {path}")
                self._status_label.setStyleSheet("color: #4CAF50;")
                self.export_completed.emit(path)
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

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
        if theme.is_dark:
            self._code_preview.setStyleSheet(
                "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
                "font-size: 12px; "
                "background-color: #1e1e1e; color: #d4d4d4;"
            )
        else:
            self._code_preview.setStyleSheet(
                "font-family: 'Consolas', 'Monaco', 'Courier New', monospace; "
                "font-size: 12px; "
                "background-color: #ffffff; color: #000000;"
            )

    def export_session_bundle(self, path: str) -> None:
        """Export session as a ZIP bundle.

        Args:
            path: Output file path
        """
        state = self._app_state.state

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Export data as CSV
            if state.xdata is not None and state.ydata is not None:
                data_csv = StringIO()
                data_csv.write("x,y,y_fit,residuals\n")
                y_fit = self._compute_y_fit()
                if y_fit is not None:
                    for x, y, yf in zip(state.xdata, state.ydata, y_fit, strict=False):
                        data_csv.write(f"{x},{y},{yf},{y - yf}\n")
                else:
                    for x, y in zip(state.xdata, state.ydata, strict=False):
                        data_csv.write(f"{x},{y},,\n")
                zf.writestr("data.csv", data_csv.getvalue())

            # Export results as JSON
            zf.writestr("results.json", self._generate_json())

            # Export Python code
            zf.writestr("fit_script.py", self.generate_python_code())

            # Export config as YAML
            config_yaml = self._generate_config_yaml()
            zf.writestr("config.yaml", config_yaml)

    def export_json(self, path: str) -> None:
        """Export results as JSON.

        Args:
            path: Output file path
        """
        json_str = self._generate_json()
        Path(path).write_text(json_str, encoding="utf-8")

    def export_csv(self, path: str, include_data: bool = False) -> None:
        """Export parameters/data as CSV.

        Args:
            path: Output file path
            include_data: If True, export data with fit; if False, export parameters only
        """
        state = self._app_state.state

        if include_data:
            # Export data with fit
            if state.xdata is None or state.ydata is None:
                raise ValueError("No data available")

            lines = ["x,y,y_fit,residuals"]
            y_fit = self._compute_y_fit()
            if y_fit is not None:
                for x, y, yf in zip(state.xdata, state.ydata, y_fit, strict=False):
                    lines.append(f"{x},{y},{yf},{y - yf}")
            else:
                for x, y in zip(state.xdata, state.ydata, strict=False):
                    lines.append(f"{x},{y},,")

            Path(path).write_text("\n".join(lines), encoding="utf-8")
        else:
            # Export parameters only
            result = state.fit_result
            if result is None:
                raise ValueError("No fit results available")

            param_names = self._get_param_names()
            popt = getattr(result, "x", None) or getattr(result, "popt", [])
            pcov = getattr(result, "pcov", None)

            lines = ["parameter,value,uncertainty,ci_lower,ci_upper"]
            for i, name in enumerate(param_names):
                value = popt[i] if i < len(popt) else 0.0
                if pcov is not None:
                    try:
                        uncert = np.sqrt(pcov[i, i])
                        ci_lower = value - 1.96 * uncert
                        ci_upper = value + 1.96 * uncert
                    except Exception:
                        uncert = 0.0
                        ci_lower = ci_upper = value
                else:
                    uncert = 0.0
                    ci_lower = ci_upper = value

                lines.append(f"{name},{value},{uncert},{ci_lower},{ci_upper}")

            Path(path).write_text("\n".join(lines), encoding="utf-8")

    def generate_python_code(self) -> str:
        """Generate Python code for reproducibility.

        Uses the fit() function with workflow presets for consistency
        with NLSQ's recommended approach.

        Returns:
            Python code string
        """
        state = self._app_state.state

        lines = [
            '"""',
            "NLSQ Curve Fitting Script",
            "Generated by NLSQ Qt GUI",
            '"""',
            "",
            "import numpy as np",
            "from nlsq import fit",
            "",
        ]

        # Add model function
        if state.model_config:
            model_type = state.model_config.get("type", "unknown")
            if model_type == "builtin":
                model_name = state.model_config.get("name", "linear")
                lines.extend(
                    [
                        f"# Using built-in model: {model_name}",
                        f"from nlsq.core.functions import {model_name}",
                        f"model = {model_name}",
                        "",
                    ]
                )
            elif model_type == "polynomial":
                degree = state.model_config.get("degree", 2)
                lines.extend(
                    [
                        f"# Polynomial model of degree {degree}",
                        "def polynomial(x, *coeffs):",
                        "    result = 0.0",
                        "    for i, c in enumerate(coeffs):",
                        "        result += c * x**i",
                        "    return result",
                        "",
                        f"# degree = {degree}",
                        "model = polynomial",
                        "",
                    ]
                )
            elif model_type == "custom":
                code = state.model_config.get("code", "")
                lines.extend(
                    [
                        "# Custom model function",
                        code,
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "# Define your model function",
                    "def model(x, a, b):",
                    "    return a * x + b",
                    "",
                ]
            )

        # Add data
        if state.xdata is not None and state.ydata is not None:
            # Use repr for small arrays, file reference for large
            n = len(state.xdata)
            if n <= 20:
                x_str = np.array2string(state.xdata, separator=", ", max_line_width=80)
                y_str = np.array2string(state.ydata, separator=", ", max_line_width=80)
                lines.extend(
                    [
                        "# Data",
                        f"xdata = np.array({x_str})",
                        f"ydata = np.array({y_str})",
                        "",
                    ]
                )
            else:
                lines.extend(
                    [
                        "# Data (load from file)",
                        "# xdata = np.loadtxt('data.csv', delimiter=',', usecols=0)",
                        "# ydata = np.loadtxt('data.csv', delimiter=',', usecols=1)",
                        f"xdata = np.linspace(0, 10, {n})  # Replace with your data",
                        f"ydata = np.zeros({n})  # Replace with your data",
                        "",
                    ]
                )

        # Determine workflow preset
        workflow = getattr(state, "preset", "standard") or "standard"

        # Add fitting options
        lines.extend(
            [
                "# Fitting options",
            ]
        )

        if state.p0:
            p0_str = str(list(state.p0))
            lines.append(f"p0 = {p0_str}")
        else:
            lines.append("p0 = None  # Auto-detect initial parameters")

        if state.bounds:
            lines.append(f"bounds = {state.bounds}")
        else:
            lines.append("bounds = (-np.inf, np.inf)")

        lines.append(
            f'workflow = "{workflow}"  # Options: standard, fast, quality, large_robust'
        )

        lines.extend(
            [
                "",
                "# Run curve fitting using workflow preset",
                "result = fit(",
                "    model,",
                "    xdata,",
                "    ydata,",
                "    p0=p0,",
                "    bounds=bounds,",
                "    workflow=workflow,",
                ")",
                "",
                "# Extract parameters and covariance",
                "popt = result.popt",
                "pcov = result.pcov",
                "",
                "# Print results",
                'print("Fitted parameters:")',
                "for i, val in enumerate(popt):",
                "    err = np.sqrt(pcov[i, i])",
                '    print(f"  p{i} = {val:.6g} +/- {err:.6g}")',
                "",
                "# Print fit statistics (computed by NLSQ)",
                'print(f"R-squared: {result.r_squared:.6f}")',
                'print(f"RMSE: {result.rmse:.6g}")',
                'print(f"Converged: {result.success}")',
            ]
        )

        return "\n".join(lines)

    def _generate_json(self) -> str:
        """Generate JSON representation of results.

        Returns:
            JSON string
        """
        state = self._app_state.state
        result = state.fit_result

        data: dict[str, Any] = {
            "nlsq_version": "0.6.0",
            "export_type": "fit_results",
        }

        # Parameters
        if result is not None:
            popt = getattr(result, "x", None) or getattr(result, "popt", None)
            pcov = getattr(result, "pcov", None)
            param_names = self._get_param_names()

            if popt is not None:
                params = {}
                for i, name in enumerate(param_names):
                    value = float(popt[i]) if i < len(popt) else 0.0
                    uncert = 0.0
                    if pcov is not None:
                        with contextlib.suppress(Exception):
                            uncert = float(np.sqrt(pcov[i, i]))
                    params[name] = {
                        "value": value,
                        "uncertainty": uncert,
                        "ci_lower": value - 1.96 * uncert,
                        "ci_upper": value + 1.96 * uncert,
                    }
                data["parameters"] = params

            # Statistics
            data["statistics"] = {
                "converged": getattr(result, "success", True),
                "n_evaluations": getattr(result, "nfev", 0),
            }

            # Add R², RMSE if data available
            if state.xdata is not None and state.ydata is not None and popt is not None:
                y_fit = self._compute_y_fit()
                if y_fit is not None:
                    residuals = state.ydata - y_fit
                    ss_res = float(np.sum(residuals**2))
                    ss_tot = float(np.sum((state.ydata - np.mean(state.ydata)) ** 2))
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    rmse = float(np.sqrt(np.mean(residuals**2)))

                    data["statistics"]["r_squared"] = r_squared
                    data["statistics"]["rmse"] = rmse
                    data["statistics"]["n_points"] = len(state.xdata)

        # Model info
        if state.model_config:
            data["model"] = state.model_config

        return json.dumps(data, indent=2)

    def _generate_config_yaml(self) -> str:
        """Generate YAML configuration.

        Returns:
            YAML string
        """
        state = self._app_state.state

        lines = [
            "# NLSQ Configuration",
            "# Generated by NLSQ Qt GUI",
            "",
        ]

        if state.model_config:
            lines.append("model:")
            for key, value in state.model_config.items():
                if isinstance(value, str) and "\n" in value:
                    lines.append(f"  {key}: |")
                    lines.extend(f"    {line}" for line in value.split("\n"))
                else:
                    lines.append(f"  {key}: {value}")
            lines.append("")

        lines.append("fitting:")
        lines.append(f"  gtol: {state.gtol if hasattr(state, 'gtol') else 1e-8}")
        lines.append(f"  ftol: {state.ftol if hasattr(state, 'ftol') else 1e-8}")
        lines.append(f"  xtol: {state.xtol if hasattr(state, 'xtol') else 1e-8}")
        lines.append(
            f"  max_iterations: {state.max_iterations if hasattr(state, 'max_iterations') else 200}"
        )

        return "\n".join(lines)

    def _compute_y_fit(self) -> np.ndarray | None:
        """Compute fitted y values.

        Returns:
            Fitted y values or None
        """
        state = self._app_state.state
        result = state.fit_result

        if result is None or state.xdata is None:
            return None

        popt = getattr(result, "x", None) or getattr(result, "popt", None)
        if popt is None:
            return None

        try:
            if state.model_func is not None:
                return state.model_func(state.xdata, *popt)
        except Exception:
            pass

        return None

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
