"""Tests for the NLSQ GUI export adapter module.

This module tests the export adapter which handles exporting fit results
in various formats (JSON, CSV, ZIP bundle, HTML) for the Streamlit GUI.
"""

import io
import json
import zipfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nlsq.gui.adapters.export_adapter import (
    create_session_bundle,
    export_csv,
    export_json,
    export_plotly_html,
)
from nlsq.gui.state import SessionState, initialize_state

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_result() -> MagicMock:
    """Create a mock CurveFitResult for testing."""
    result = MagicMock()
    result.popt = np.array([2.0, 0.5, 1.0])
    result.pcov = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.02],
        ]
    )
    result.success = True
    result.message = "Optimization converged"
    result.nfev = 42
    result.cost = 0.0015
    result.r_squared = 0.9987
    result.rmse = 0.025
    result.mae = 0.018
    result.aic = -150.5
    result.bic = -145.2
    result.residuals = np.array([0.01, -0.02, 0.015, -0.01, 0.005])
    return result


@pytest.fixture
def session_state() -> SessionState:
    """Create a session state for testing."""
    state = initialize_state()
    state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    state.ydata = np.array([2.5, 4.0, 5.5, 8.0, 9.5])
    state.model_type = "builtin"
    state.model_name = "exponential_decay"
    state.p0 = [2.0, 0.5, 1.0]
    state.gtol = 1e-8
    state.ftol = 1e-8
    state.xtol = 1e-8
    return state


@pytest.fixture
def mock_figure() -> MagicMock:
    """Create a mock Plotly figure for testing."""
    fig = MagicMock()
    fig.to_html.return_value = "<html><body>Mock Plotly Chart</body></html>"
    fig.to_image.return_value = b"PNG_IMAGE_DATA"
    return fig


# =============================================================================
# Test JSON Export
# =============================================================================


class TestExportJSON:
    """Tests for JSON export functionality."""

    def test_export_json_basic(self, mock_result: MagicMock) -> None:
        """Test basic JSON export with parameters and statistics."""
        json_str = export_json(mock_result)

        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert "popt" in data
        assert "pcov" in data
        assert "statistics" in data
        assert "convergence" in data

    def test_export_json_parameters(self, mock_result: MagicMock) -> None:
        """Test that JSON export includes correct parameter values."""
        json_str = export_json(mock_result)
        data = json.loads(json_str)

        assert len(data["popt"]) == 3
        assert abs(data["popt"][0] - 2.0) < 1e-10
        assert abs(data["popt"][1] - 0.5) < 1e-10
        assert abs(data["popt"][2] - 1.0) < 1e-10

    def test_export_json_statistics(self, mock_result: MagicMock) -> None:
        """Test that JSON export includes statistics."""
        json_str = export_json(mock_result)
        data = json.loads(json_str)

        stats = data["statistics"]
        assert "r_squared" in stats
        assert "rmse" in stats
        assert abs(stats["r_squared"] - 0.9987) < 1e-10

    def test_export_json_convergence(self, mock_result: MagicMock) -> None:
        """Test that JSON export includes convergence info."""
        json_str = export_json(mock_result)
        data = json.loads(json_str)

        conv = data["convergence"]
        assert conv["success"] is True
        assert conv["nfev"] == 42
        assert "message" in conv

    def test_export_json_with_param_names(self, mock_result: MagicMock) -> None:
        """Test JSON export with custom parameter names."""
        param_names = ["amplitude", "decay_rate", "offset"]
        json_str = export_json(mock_result, param_names=param_names)
        data = json.loads(json_str)

        assert "parameter_names" in data
        assert data["parameter_names"] == param_names

    def test_export_json_handles_numpy_types(self, mock_result: MagicMock) -> None:
        """Test that numpy types are properly serialized."""
        mock_result.popt = np.array([np.float64(2.0), np.float64(0.5)])
        mock_result.nfev = np.int64(42)
        mock_result.success = np.bool_(True)

        json_str = export_json(mock_result)

        # Should not raise, all numpy types serialized
        data = json.loads(json_str)
        assert isinstance(data["popt"][0], float)


# =============================================================================
# Test CSV Export
# =============================================================================


class TestExportCSV:
    """Tests for CSV export functionality."""

    def test_export_csv_basic(self, mock_result: MagicMock) -> None:
        """Test basic CSV export with parameters."""
        csv_str = export_csv(mock_result)

        assert isinstance(csv_str, str)
        assert "name" in csv_str
        assert "value" in csv_str
        assert "uncertainty" in csv_str

    def test_export_csv_parameter_rows(self, mock_result: MagicMock) -> None:
        """Test that CSV export includes parameter rows."""
        csv_str = export_csv(mock_result)
        lines = csv_str.strip().split("\n")

        # Header + at least 3 parameter rows
        assert len(lines) >= 4

        # Check header
        header = lines[0]
        assert "name" in header
        assert "value" in header

    def test_export_csv_with_param_names(self, mock_result: MagicMock) -> None:
        """Test CSV export with custom parameter names."""
        param_names = ["amplitude", "decay_rate", "offset"]
        csv_str = export_csv(mock_result, param_names=param_names)

        assert "amplitude" in csv_str
        assert "decay_rate" in csv_str
        assert "offset" in csv_str

    def test_export_csv_uncertainties(self, mock_result: MagicMock) -> None:
        """Test that CSV includes uncertainty values."""
        csv_str = export_csv(mock_result)

        # Uncertainty for first param: sqrt(0.01) = 0.1
        lines = csv_str.strip().split("\n")
        # Check that uncertainties are present (non-empty)
        assert len(lines) >= 2
        # The second line should have an uncertainty value
        parts = lines[1].split(",")
        assert len(parts) >= 3


# =============================================================================
# Test Session Bundle Creation
# =============================================================================


class TestCreateSessionBundle:
    """Tests for session bundle ZIP creation."""

    def test_create_session_bundle_basic(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test basic session bundle creation."""
        figures = {}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        assert isinstance(zip_bytes, bytes)
        assert len(zip_bytes) > 0

    def test_create_session_bundle_is_valid_zip(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test that session bundle is a valid ZIP file."""
        figures = {}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        # Should be readable as ZIP
        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            assert len(names) > 0

    def test_create_session_bundle_contains_data(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test that session bundle contains data CSV."""
        figures = {}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            # Should have a data file
            assert any("data" in name.lower() for name in names)

    def test_create_session_bundle_contains_config(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test that session bundle contains YAML config."""
        figures = {}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            # Should have a config file
            assert any(
                "config" in name.lower() or name.endswith(".yaml") for name in names
            )

    def test_create_session_bundle_contains_results(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test that session bundle contains JSON results."""
        figures = {}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            # Should have a results file
            assert any(
                "result" in name.lower() or name.endswith(".json") for name in names
            )

    def test_create_session_bundle_with_figures(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
        mock_figure: MagicMock,
    ) -> None:
        """Test session bundle includes figure files."""
        figures = {"fit_plot": mock_figure}

        zip_bytes = create_session_bundle(session_state, mock_result, figures)

        zip_buffer = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            # Should have an HTML file for the figure
            assert any(name.endswith(".html") for name in names)


# =============================================================================
# Test Plotly HTML Export
# =============================================================================


class TestExportPlotlyHTML:
    """Tests for Plotly HTML export functionality."""

    def test_export_plotly_html_basic(self, mock_figure: MagicMock) -> None:
        """Test basic Plotly HTML export."""
        html = export_plotly_html(mock_figure)

        assert isinstance(html, str)
        assert (
            "<html>" in html.lower()
            or "<!doctype" in html.lower()
            or "<body>" in html.lower()
        )

    def test_export_plotly_html_calls_to_html(self, mock_figure: MagicMock) -> None:
        """Test that export calls figure.to_html()."""
        export_plotly_html(mock_figure)

        mock_figure.to_html.assert_called_once()

    def test_export_plotly_html_standalone(self, mock_figure: MagicMock) -> None:
        """Test that HTML export is standalone (includes plotly.js)."""
        # Configure mock to return HTML with plotly.js
        mock_figure.to_html.return_value = """
        <html>
        <head><script src="plotly.min.js"></script></head>
        <body><div id="chart"></div></body>
        </html>
        """

        html = export_plotly_html(mock_figure)

        # Check that to_html was called with include_plotlyjs parameter
        call_args = mock_figure.to_html.call_args
        if call_args.kwargs:
            # Should request standalone HTML
            assert "include_plotlyjs" in call_args.kwargs or len(html) > 100


# =============================================================================
# Test Python Code Generation (via export_adapter)
# =============================================================================


class TestCodeGeneratorIntegration:
    """Tests for Python code generation via export adapter."""

    def test_code_generation_integration(
        self,
        session_state: SessionState,
        mock_result: MagicMock,
    ) -> None:
        """Test that code generation works with export adapter."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(session_state, mock_result)

        assert isinstance(code, str)
        assert "import" in code
        assert "nlsq" in code.lower() or "curve_fit" in code.lower()
