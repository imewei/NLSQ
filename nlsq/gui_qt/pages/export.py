"""
NLSQ Qt GUI Export Page

This page allows users to export fitting results in multiple formats
including session bundles, JSON, CSV, and Python code generation.
"""

from __future__ import annotations

import ast
import contextlib
import json
import stat
import tempfile
import warnings
import zipfile
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QFontDatabase, QGuiApplication
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
from scipy import stats

from nlsq.cli.result_exporter import NumpyJSONEncoder, _atomic_write
from nlsq.gui_qt.adapters.fit_adapter import _resolve_workflow
from nlsq.gui_qt.session_state import n_points

if TYPE_CHECKING:
    from nlsq.gui_qt.app_state import AppState
    from nlsq.gui_qt.theme import ThemeConfig

# Largest dataset embedded literally in a standalone script (no truncation:
# above this the script loads data.csv instead).
_INLINE_MAX_POINTS = 5000

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

        # Title — 24px bold, consistent across all pages
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
        layout.setSpacing(8)

        # Session bundle export
        bundle_group = QGroupBox("Session Bundle (ZIP)")
        bundle_layout = QVBoxLayout(bundle_group)

        bundle_desc = QLabel(
            "Export complete session including data, configuration, "
            "fit results, and plots as a ZIP archive.",
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
            "or integration with other tools.",
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
            "applications like Excel or Google Sheets.",
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
            "using NLSQ library.",
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
        layout.setSpacing(8)

        code_group = QGroupBox("Code Preview")
        code_layout = QVBoxLayout(code_group)

        self._code_preview = QTextEdit()
        self._code_preview.setReadOnly(True)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        family = fixed_font.family()
        self._code_preview.setStyleSheet(
            f"font-family: '{family}', monospace; font-size: 12px;",
        )
        self._code_preview.setPlaceholderText(
            "Click 'Generate Code' to preview Python code...",
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
            "data.csv",
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
        code = self.generate_python_code()  # never reuse a stale preview
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
            try:
                code = self.generate_python_code()  # never reuse a stale preview
                self._code_preview.setPlainText(code)
                _atomic_write(Path(path).resolve(), code)
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
            fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            family = fixed_font.family()
            self._code_preview.setStyleSheet(
                f"font-family: '{family}', monospace; "
                "font-size: 12px; "
                "background-color: #1e1e1e; color: #d4d4d4;",
            )
        else:
            fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            family = fixed_font.family()
            self._code_preview.setStyleSheet(
                f"font-family: '{family}', monospace; "
                "font-size: 12px; "
                "background-color: #ffffff; color: #000000;",
            )

    def export_session_bundle(self, path: str) -> None:
        """Export session as a ZIP bundle.

        Args:
            path: Output file path
        """
        state = self._app_state.state

        destination = Path(path)
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp_dir:
            temporary = Path(tmp_dir) / destination.name
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as zf:
                if state.xdata is not None and state.ydata is not None:
                    zf.writestr("data.csv", "\n".join(self._data_csv_lines()) + "\n")
                zf.writestr("results.json", self._generate_json())
                zf.writestr("fit_script.py", self.generate_python_code(for_bundle=True))
                zf.writestr("config.yaml", self._generate_config_yaml())
            # shortcut: same stat-then-replace mode race as cli.result_exporter._atomic_write
            # (uncoordinated second process); upgrade to a lock file if exports get sensitive.
            if destination.exists():
                temporary.chmod(stat.S_IMODE(destination.stat().st_mode))
            temporary.replace(destination)

    def export_json(self, path: str) -> None:
        """Export results as JSON.

        Args:
            path: Output file path
        """
        json_str = self._generate_json()
        _atomic_write(Path(path).resolve(), json_str)

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

            _atomic_write(Path(path).resolve(), "\n".join(self._data_csv_lines()))
        else:
            # Export parameters only
            result = state.fit_result
            if result is None:
                raise ValueError("No fit results available")

            param_names = self._get_param_names()
            popt = getattr(result, "x", None)
            if popt is None:
                popt = getattr(result, "popt", [])
            pcov = getattr(result, "pcov", None)

            # Compute t-value for 95% CI
            n_data = len(state.ydata) if state.ydata is not None else 0
            dof = max(n_data - len(popt), 1)
            t_val = stats.t.ppf(0.975, dof)

            lines = ["parameter,value,uncertainty,ci_lower,ci_upper"]
            for i, name in enumerate(param_names):
                value = popt[i] if i < len(popt) else 0.0
                if pcov is not None:
                    try:
                        uncert = np.sqrt(pcov[i, i])
                        ci_lower = value - t_val * uncert
                        ci_upper = value + t_val * uncert
                    except Exception:
                        uncert = ci_lower = ci_upper = ""
                else:
                    uncert = ci_lower = ci_upper = ""

                lines.append(f"{name},{value},{uncert},{ci_lower},{ci_upper}")

            _atomic_write(Path(path).resolve(), "\n".join(lines))

    def _data_csv_lines(self) -> list[str]:
        """CSV lines (header first) for data with fit and residuals.

        2-D surface data (``xdata`` shape ``(n_coords, n)``) gets one column
        per coordinate: ``x0,x1,...``; 1-D data keeps the single ``x`` column.
        """
        state = self._app_state.state
        xdata = np.asarray(state.xdata)
        x_cols = xdata[None, :] if xdata.ndim == 1 else xdata
        x_head = ["x"] if xdata.ndim == 1 else [f"x{i}" for i in range(len(x_cols))]

        # sigma (when present) is the last column so the fit can be reproduced
        sigma = None if state.sigma is None else np.asarray(state.sigma)
        sig_head = [] if sigma is None else ["sigma"]
        sig_col = [] if sigma is None else [sigma]

        y = np.asarray(state.ydata)
        y_fit = self._compute_y_fit()
        fit_cols = (
            [y_fit, y - np.asarray(y_fit)] if y_fit is not None else [[""] * len(y)] * 2
        )
        lines = [",".join([*x_head, "y", "y_fit", "residuals", *sig_head])]
        lines.extend(
            ",".join(map(str, row))
            for row in zip(*x_cols, y, *fit_cols, *sig_col, strict=False)
        )
        return lines

    def _inline_data(self, for_bundle: bool) -> bool:
        """Whether the script embeds the data literally instead of loading it."""
        n = n_points(self._app_state.state.xdata)
        return n <= 20 or (not for_bundle and n <= _INLINE_MAX_POINTS)

    def _script_data_lines(self, for_bundle: bool = False) -> list[str]:
        """Data section of the reproducibility script.

        Standalone scripts embed up to ``_INLINE_MAX_POINTS`` points; larger
        data (and the ZIP bundle, which ships data.csv) is loaded from file.
        """
        state = self._app_state.state
        lines: list[str] = []
        has_sigma = state.sigma is not None
        if state.xdata is not None and state.ydata is not None:
            # Use repr for small arrays, file reference for large
            if self._inline_data(for_bundle):
                # repr(tolist()) keeps full float64 precision, unlike array2string
                x_str = repr(np.asarray(state.xdata).tolist())
                y_str = repr(np.asarray(state.ydata).tolist())
                lines.extend(
                    [
                        "# Data",
                        f"xdata = np.array({x_str})",
                        f"ydata = np.array({y_str})",
                    ],
                )
                if has_sigma:
                    s_str = repr(np.asarray(state.sigma).tolist())
                    lines.append(f"sigma = np.array({s_str})")
                lines.append("")
            else:
                # data.csv layout: x column(s), y, y_fit, residuals, [sigma]
                k = 1 if np.ndim(state.xdata) == 1 else len(state.xdata)
                x_cols = "0" if k == 1 else str(list(range(k)))
                x_t = "" if k == 1 else ".T"
                lines.extend(
                    [
                        "# Data (load from file): data.csv must sit next to this script",
                        "# (use 'Export Data...' in the GUI, which defaults to that name)",
                        "from pathlib import Path",
                        "data_file = Path(__file__).with_name('data.csv')",
                        f"xdata = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols={x_cols}){x_t}",
                        f"ydata = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols={k})",
                    ],
                )
                if has_sigma:
                    lines.append(
                        f"sigma = np.loadtxt(data_file, delimiter=',', skiprows=1, usecols={k + 3})"
                    )
                lines.append("")
        return lines

    def _script_fit_kwargs(self, workflow: str, multistart: bool = False) -> list[str]:
        """fit() keyword lines mirroring the settings the GUI fit used."""
        state = self._app_state.state
        # Mirror the settings the GUI fit used so a rerun reproduces its result
        extra = [
            f"    gtol={state.gtol!r},",
            f"    ftol={state.ftol!r},",
            f"    xtol={state.xtol!r},",
            f"    loss={state.loss!r},",
        ]
        if state.method and state.method != "auto":
            extra.append(f"    method={state.method!r},")
        extra.append(
            f"    max_nfev={min(state.max_function_evals, state.max_iterations)},"
        )
        if (workflow == "auto_global" or multistart) and state.n_starts > 0:
            if multistart:
                extra.append("    multistart=True,")
            extra += [
                f"    n_starts={state.n_starts},",
                f"    sampler={state.sampler!r},",
                f"    center_on_p0={state.center_on_p0!r},",
                f"    scale_factor={state.scale_factor!r},",
            ]
        if state.sigma is not None:
            extra.insert(0, "    sigma=sigma,")
            if state.absolute_sigma:  # fit()'s default is False
                extra.insert(1, "    absolute_sigma=True,")
        return extra

    def generate_python_code(self, for_bundle: bool = False) -> str:
        """Generate Python code for reproducibility.

        Uses the fit() function with workflow presets for consistency
        with NLSQ's recommended approach.

        Args:
            for_bundle: True when the script ships next to data.csv in the
                ZIP bundle, so data is read from that file instead of inlined.

        Returns:
            Python code string
        """
        state = self._app_state.state

        # repr() of nan/inf is a bare name: the script must define it
        nonfinite = any(
            a is not None and not np.all(np.isfinite(np.asarray(a, dtype=float)))
            for a in (state.xdata, state.ydata, state.sigma, state.p0)
        )
        needs_file = state.xdata is not None and not self._inline_data(for_bundle)
        lines = [
            '"""',
            "NLSQ Curve Fitting Script",
            "Generated by NLSQ Qt GUI",
            *(
                [
                    "",
                    "Requires data.csv next to this script ('Export Data...' in the GUI).",
                ]
                if needs_file
                else []
            ),
            '"""',
            "",
            "import numpy as np",
            *(
                ["from numpy import inf, nan  # inlined non-finite values"]
                if nonfinite
                else []
            ),
            "from nlsq import fit",
            "",
        ]

        # Add model function
        if state.model_config:
            model_type = state.model_type
            if model_type == "builtin":
                model_name = state.model_config.get("name", "linear")
                if not (isinstance(model_name, str) and model_name.isidentifier()):
                    raise ValueError(
                        f"Model name must be an identifier: {model_name!r}"
                    )
                lines.extend(
                    [
                        f"# Using built-in model: {model_name}",
                        f"from nlsq.core.functions import {model_name}",
                        f"model = {model_name}",
                        "",
                    ],
                )
            elif model_type == "polynomial":
                degree = int(state.model_config.get("degree", 2))
                lines.extend(
                    [
                        f"# Polynomial model of degree {degree}",
                        "import jax.numpy as jnp",
                        "",
                        "def polynomial(x, *coeffs):",
                        "    # descending powers, same as the GUI model (c0*x^n + ... + cn)",
                        "    return jnp.polyval(jnp.array(coeffs), x)",
                        "",
                        f"# degree = {degree}",
                        "model = polynomial",
                        "",
                    ],
                )
            elif model_type == "custom":
                code = state.model_config.get("code", "")
                model_path = state.model_config.get("path")
                if not code and model_path:  # YAML-imported custom model file
                    code = self._read_model_file(model_path)
                func_name = state.model_config.get(
                    "function"
                ) or self._first_function_name(code)
                if func_name and not (
                    isinstance(func_name, str) and func_name.isidentifier()
                ):
                    raise ValueError(
                        f"Model function name must be an identifier: {func_name!r}"
                    )
                lines.extend(["# Custom model function", code, ""])
                if func_name and func_name != "model":
                    lines.extend([f"model = {func_name}", ""])
        else:
            lines.extend(
                [
                    "# Define your model function",
                    "def model(x, a, b):",
                    "    return a * x + b",
                    "",
                ],
            )

        lines.extend(self._script_data_lines(for_bundle))

        # Determine workflow (3-workflow system v0.6.3)
        # Export the workflow that actually ran (auto_global degrades to auto
        # without finite bounds), not the one merely selected in state.
        bounds_arr = None
        if state.bounds:
            bounds_arr = tuple(
                np.array([fill if v is None else float(v) for v in b])
                for b, fill in zip(state.bounds, (-np.inf, np.inf), strict=True)
            )
        selected = getattr(state, "workflow", "auto")
        # mirrors execute_fit: p0 present -> multi-start runs inside inferred bounds
        multistart = bool(state.p0) and state.n_starts > 0
        workflow = _resolve_workflow(selected, bounds_arr, [], multistart)
        multistart = multistart and selected == "auto_global" and workflow == "auto"

        # Add fitting options
        lines.extend(
            [
                "# Fitting options",
            ],
        )

        if state.p0:
            p0_str = str([float(v) for v in state.p0])
            lines.append(f"p0 = {p0_str}")
        elif state.model_type == "polynomial" and state.model_config:
            # polynomial(x, *coeffs) has no fixed arity: p0 sets the parameter count
            degree = int(state.model_config.get("degree", 2))
            lines.append(f"p0 = [1.0] * {degree + 1}  # degree + 1 coefficients")
        else:
            lines.append("p0 = None  # Auto-detect initial parameters")

        if state.bounds:
            # GUI bounds use None for "unbounded"; the script needs +/-np.inf
            def _bound(v: Any, inf: str) -> str:
                if v is None or not np.isfinite(float(v)):
                    return inf
                return repr(float(v))

            lo = ", ".join(_bound(v, "-np.inf") for v in state.bounds[0])
            hi = ", ".join(_bound(v, "np.inf") for v in state.bounds[1])
            lines.append(f"bounds = ([{lo}], [{hi}])")
        else:
            lines.append("bounds = (-np.inf, np.inf)")

        lines.append(f"workflow = {workflow!r}  # Options: auto, auto_global, hpc")

        extra = self._script_fit_kwargs(workflow, multistart)

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
                *extra,
                ")",
                "",
                "# Extract parameters and covariance",
                "popt = result.popt",
                "pcov = result.pcov",
                "",
                "# Print results",
                f"param_names = {[str(n) for n in self._get_param_names()]!r}",
                'print("Fitted parameters:")',
                "for i, val in enumerate(popt):",
                "    err = np.sqrt(pcov[i, i])",
                '    name = param_names[i] if i < len(param_names) else f"p{i}"',
                '    print(f"  {name} = {val:.6g} +/- {err:.6g}")',
                "",
                "# Print fit statistics (computed by NLSQ)",
                'print(f"R-squared: {result.r_squared:.6f}")',
                'print(f"RMSE: {result.rmse:.6g}")',
                'print(f"Converged: {result.success}")',
            ],
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
            "nlsq_version": _pkg_version("nlsq"),
            "export_type": "fit_results",
        }

        # Parameters
        if result is not None:
            popt = getattr(result, "x", None)
            if popt is None:
                popt = getattr(result, "popt", None)
            pcov = getattr(result, "pcov", None)
            param_names = self._get_param_names()

            if popt is not None:
                # Compute t-value for 95% CI
                n_data = len(state.ydata) if state.ydata is not None else 0
                dof = max(n_data - len(popt), 1)
                t_val = stats.t.ppf(0.975, dof)

                params = {}
                for i, name in enumerate(param_names):
                    value = float(popt[i]) if i < len(popt) else 0.0
                    uncert = float("nan")
                    if pcov is not None:
                        with contextlib.suppress(Exception):
                            uncert = float(np.sqrt(pcov[i, i]))
                    ci_lower = value - t_val * uncert
                    ci_upper = value + t_val * uncert
                    # NLSQ deliberately reports +inf uncertainty for a
                    # singular covariance, which makes ci_lower/ci_upper
                    # +/-inf too. json.dumps emits non-standard NaN/Infinity
                    # tokens for non-finite floats -- normalize to null here
                    # the same way the statistics block below already does.
                    params[name] = {
                        "value": value if np.isfinite(value) else None,
                        "uncertainty": uncert if np.isfinite(uncert) else None,
                        "ci_lower": ci_lower if np.isfinite(ci_lower) else None,
                        "ci_upper": ci_upper if np.isfinite(ci_upper) else None,
                    }
                data["parameters"] = params

            # Statistics
            data["statistics"] = {
                "converged": getattr(result, "success", True),
                "n_evaluations": getattr(result, "nfev", 0),
            }

            for key in ("aic", "bic"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        val = float(getattr(result, key))
                    except (AttributeError, TypeError, ValueError):
                        continue
                if np.isfinite(val):  # undefined (NaN) is omitted, not nulled
                    data["statistics"][key] = val
            if state.sigma is not None:
                data["statistics"]["absolute_sigma"] = bool(state.absolute_sigma)

            # Add R², RMSE if data available
            if state.xdata is not None and state.ydata is not None and popt is not None:
                y_fit = self._compute_y_fit()
                if y_fit is not None:
                    ydata = np.asarray(state.ydata)
                    residuals = ydata - np.asarray(y_fit)
                    ss_res = float(np.sum(residuals**2))
                    ss_tot = float(np.sum((ydata - np.mean(ydata)) ** 2))
                    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    rmse = float(np.sqrt(np.mean(residuals**2)))

                    # json.dumps emits non-standard NaN/Infinity tokens for
                    # non-finite floats; normalize to null for a diverged fit.
                    data["statistics"]["r_squared"] = (
                        r_squared if np.isfinite(r_squared) else None
                    )
                    data["statistics"]["rmse"] = rmse if np.isfinite(rmse) else None
                    data["statistics"]["n_points"] = n_points(state.xdata)

        # Model info
        if state.model_config:
            data["model"] = state.model_config

        return json.dumps(data, indent=2, cls=NumpyJSONEncoder)

    def _generate_config_yaml(self) -> str:
        """Generate YAML configuration (same schema ``load_yaml_config`` reads)."""
        from nlsq.gui_qt.adapters.config_adapter import export_yaml_config

        return export_yaml_config(self._app_state.state)

    @staticmethod
    def _read_model_file(model_path: Any) -> str:
        """Inline a custom-model ``.py`` file; never raise, never copy other files.

        Only small ``.py`` files are inlined (a YAML config could name any
        readable file, e.g. credentials); otherwise a stub comment is emitted.
        """
        path = Path(str(model_path))
        try:
            real = path.resolve()  # a symlink must not smuggle in another file
            if (
                real.suffix != ".py"
                or not real.is_file()  # also rejects FIFOs/devices (read would block)
                or real.stat().st_size > 256 * 1024
            ):
                raise ValueError("not a regular .py file of at most 256 KiB")
            return real.read_text(encoding="utf-8")
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            return (
                f"# Custom model file {path.name!r} could not be inlined "
                f"({type(exc).__name__}); define the model function here.\n"
            )

    @staticmethod
    def _first_function_name(code: str) -> str | None:
        """Name of the first function defined in ``code`` (None if unparsable)."""
        try:
            for node in ast.parse(code).body:
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except SyntaxError:
            pass
        return None

    def _compute_y_fit(self) -> np.ndarray | None:
        """Compute fitted y values.

        Returns:
            Fitted y values or None
        """
        state = self._app_state.state
        result = state.fit_result

        if result is None or state.xdata is None:
            return None

        popt = getattr(result, "x", None)
        if popt is None:
            popt = getattr(result, "popt", None)
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
        names: list[str] = []
        if state.model_func is not None:
            names = list(get_model_info(state.model_func).get("param_names", []))
        # Varargs models (``lambda x, *c``) expose no names: pad to len(popt)
        # with the same ``c{i}`` convention get_model_info uses for polynomials.
        popt = getattr(state.fit_result, "x", None)
        if popt is None:
            popt = getattr(state.fit_result, "popt", None)
        n_fit = 0 if popt is None else len(popt)
        names.extend(f"c{i}" for i in range(len(names), n_fit))
        return names
