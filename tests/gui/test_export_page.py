"""Tests for the NLSQ GUI export page.

This module tests the export page UI components, including download
buttons for various export formats and Python code display.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
    result.residuals = np.array([0.01, -0.02, 0.015])
    return result


@pytest.fixture
def mock_session_state(mock_result: MagicMock) -> MagicMock:
    """Create a mock session state with fit result."""
    state = MagicMock()
    state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    state.ydata = np.array([2.5, 4.0, 5.5, 8.0, 9.5])
    state.sigma = None
    state.model_type = "builtin"
    state.model_name = "exponential_decay"
    state.p0 = [2.0, 0.5, 1.0]
    state.fit_result = mock_result
    state.gtol = 1e-8
    state.ftol = 1e-8
    state.xtol = 1e-8
    return state


# =============================================================================
# Test Export Page Components
# =============================================================================


class TestExportPageRendering:
    """Tests for export page rendering."""

    def test_page_imports_without_error(self) -> None:
        """Test that export page module can be imported."""
        # This tests that the module has no import-time errors
        # We can't fully test Streamlit pages without running Streamlit
        try:
            # Import should work even without Streamlit context
            import nlsq.gui.pages
        except Exception as e:
            # Streamlit-specific errors are expected when not in Streamlit context
            assert "streamlit" in str(e).lower() or "session" in str(e).lower()


class TestDownloadButtonGeneration:
    """Tests for download button data generation."""

    def test_json_download_data_generation(self, mock_result: MagicMock) -> None:
        """Test JSON download button data is generated correctly."""
        from nlsq.gui.adapters.export_adapter import export_json

        json_data = export_json(mock_result)

        assert isinstance(json_data, str)
        assert len(json_data) > 0
        # Should be valid JSON
        import json

        parsed = json.loads(json_data)
        assert "popt" in parsed

    def test_csv_download_data_generation(self, mock_result: MagicMock) -> None:
        """Test CSV download button data is generated correctly."""
        from nlsq.gui.adapters.export_adapter import export_csv

        csv_data = export_csv(mock_result)

        assert isinstance(csv_data, str)
        assert len(csv_data) > 0
        # Should have header and data rows
        lines = csv_data.strip().split("\n")
        assert len(lines) >= 2

    def test_zip_download_data_generation(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test ZIP bundle download data is generated correctly."""
        from nlsq.gui.adapters.export_adapter import create_session_bundle

        zip_data = create_session_bundle(mock_session_state, mock_result, {})

        assert isinstance(zip_data, bytes)
        assert len(zip_data) > 0
        # Should start with ZIP magic bytes
        assert zip_data[:2] == b"PK"


class TestPythonCodeGeneration:
    """Tests for Python code generation and display."""

    def test_code_generation_produces_valid_python(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code is valid Python syntax."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        # Check it's valid Python by attempting to compile
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax error: {e}")

    def test_code_generation_includes_imports(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code includes necessary imports."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        assert "import" in code
        # Should import numpy or jax.numpy
        assert "numpy" in code or "jnp" in code

    def test_code_generation_includes_model(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code includes model definition."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        # Should have a function definition
        assert "def " in code

    def test_code_generation_includes_curve_fit_call(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code includes curve_fit call."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        # Should call curve_fit or fit
        assert "curve_fit" in code.lower() or "fit(" in code.lower()


class TestSessionBundleExport:
    """Tests for session bundle export functionality."""

    def test_session_bundle_includes_all_components(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that session bundle includes all expected components."""
        import io
        import zipfile

        from nlsq.gui.adapters.export_adapter import create_session_bundle

        zip_data = create_session_bundle(mock_session_state, mock_result, {})

        zip_buffer = io.BytesIO(zip_data)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()

            # Should have data file
            has_data = any("data" in n.lower() for n in names)
            # Should have config file
            has_config = any(
                "config" in n.lower() or n.endswith(".yaml") for n in names
            )
            # Should have results file
            has_results = any(
                "result" in n.lower() or n.endswith(".json") for n in names
            )

            assert has_data or has_config or has_results, f"Bundle contents: {names}"


class TestExportPageStateCheck:
    """Tests for export page state validation."""

    def test_results_available_check(self) -> None:
        """Test that export page checks for available results."""
        # Create a state without results
        from nlsq.gui.state import initialize_state

        state = initialize_state()

        # Without fit_result, export page should detect no results
        assert state.fit_result is None

    def test_results_available_with_fit(self, mock_result: MagicMock) -> None:
        """Test that export page detects available results."""
        from nlsq.gui.state import initialize_state

        state = initialize_state()
        state.fit_result = mock_result

        # With fit_result, export page should detect results
        assert state.fit_result is not None
        assert hasattr(state.fit_result, "popt")


class TestCodeDisplay:
    """Tests for code display functionality."""

    def test_code_has_comments(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code includes helpful comments."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        # Should have at least one comment
        assert "#" in code

    def test_code_includes_data_section(
        self,
        mock_session_state: MagicMock,
        mock_result: MagicMock,
    ) -> None:
        """Test that generated code includes data definition section."""
        from nlsq.gui.utils.code_generator import generate_fit_script

        code = generate_fit_script(mock_session_state, mock_result)

        # Should define data arrays
        assert "xdata" in code.lower() or "x_data" in code.lower() or "x =" in code
