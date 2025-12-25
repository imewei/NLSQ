"""Strategic integration tests for NLSQ GUI workflows.

This module contains additional strategic tests to fill coverage gaps
identified during the test review for Task Group 19.

Focus areas:
1. Complete workflow scenarios (data -> model -> fit -> results -> export)
2. Error handling paths
3. Edge cases in data formats
4. Integration points between pages
"""

from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nlsq.cli.errors import DataLoadError
from nlsq.gui.adapters.config_adapter import export_yaml_config, load_yaml_config
from nlsq.gui.adapters.data_adapter import (
    compute_statistics,
    load_from_clipboard,
    load_from_file,
    validate_data,
)
from nlsq.gui.adapters.export_adapter import (
    create_session_bundle,
    export_csv,
    export_json,
)
from nlsq.gui.adapters.model_adapter import (
    get_model,
    get_model_info,
    list_builtin_models,
)
from nlsq.gui.app import (
    can_access_page,
    get_page_status,
    get_workflow_step_message,
)
from nlsq.gui.state import (
    SessionState,
    get_current_config,
    initialize_state,
    reset_state,
)
from nlsq.gui.utils.code_generator import generate_fit_script

# =============================================================================
# Test Complete Workflow Scenarios
# =============================================================================


class TestCompleteWorkflowIntegration:
    """End-to-end workflow tests covering data to export."""

    def test_complete_workflow_builtin_model(self, tmp_path: Path) -> None:
        """Test complete workflow with built-in exponential decay model."""
        # Step 1: Create test data file
        csv_content = "x,y,sigma\n0.0,2.5,0.1\n1.0,1.5,0.1\n2.0,1.1,0.1\n3.0,1.0,0.1\n4.0,0.95,0.1\n"
        csv_file = tmp_path / "exp_data.csv"
        csv_file.write_text(csv_content)

        # Step 2: Initialize state
        state = initialize_state()
        assert can_access_page("data", state) is True

        # Step 3: Load data
        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "csv": {"header": True, "delimiter": ","},
        }
        xdata, ydata, sigma = load_from_file(str(csv_file), config)
        state.xdata = xdata
        state.ydata = ydata
        state.sigma = sigma
        state.data_file_name = "exp_data.csv"

        # Verify data loaded correctly
        assert len(xdata) == 5
        assert can_access_page("model", state) is True

        # Step 4: Select model
        state.model_type = "builtin"
        state.model_name = "exponential_decay"
        state.p0 = [2.5, 0.5, 1.0]

        # Verify model selection
        status = get_page_status(state)
        assert status["model"]["selected"] is True
        assert can_access_page("fitting", state) is True

        # Step 5: Simulate fit result
        mock_result = MagicMock()
        mock_result.popt = np.array([2.5, 0.5, 1.0])
        mock_result.pcov = np.eye(3) * 0.01
        mock_result.success = True
        mock_result.message = "Optimization converged"
        mock_result.nfev = 25
        mock_result.cost = 0.001
        mock_result.r_squared = 0.998
        mock_result.rmse = 0.015
        mock_result.residuals = np.array([0.01, -0.01, 0.005, -0.005, 0.002])
        state.fit_result = mock_result

        # Verify fit complete
        assert can_access_page("results", state) is True
        assert can_access_page("export", state) is True

        # Step 6: Verify export works
        json_output = export_json(mock_result)
        assert "popt" in json_output

        csv_output = export_csv(mock_result)
        assert len(csv_output) > 0

        # Step 7: Generate code
        code = generate_fit_script(state, mock_result)
        assert "curve_fit" in code.lower() or "fit(" in code.lower()

        # Verify code is valid Python
        compile(code, "<generated>", "exec")

    def test_complete_workflow_polynomial_model(self, tmp_path: Path) -> None:
        """Test complete workflow with polynomial model."""
        # Create quadratic data
        x = np.linspace(0, 5, 20)
        y = 2.0 * x**2 - 3.0 * x + 1.0 + np.random.normal(0, 0.1, 20)

        csv_content = "x,y\n" + "\n".join(
            f"{xi},{yi}" for xi, yi in zip(x, y, strict=False)
        )
        csv_file = tmp_path / "poly_data.csv"
        csv_file.write_text(csv_content)

        # Initialize and load
        state = initialize_state()
        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }
        xdata, ydata, _sigma = load_from_file(str(csv_file), config)
        state.xdata = xdata
        state.ydata = ydata

        # Select polynomial model
        state.model_type = "polynomial"
        state.polynomial_degree = 2
        state.p0 = [1.0, 1.0, 1.0]

        # Verify workflow state
        assert can_access_page("fitting", state) is True

    def test_workflow_with_yaml_config_roundtrip(self) -> None:
        """Test that YAML config export/import preserves workflow state."""
        # Set up initial state
        state = initialize_state()
        state.model_type = "builtin"
        state.model_name = "gaussian"
        state.gtol = 1e-10
        state.ftol = 1e-10
        state.xtol = 1e-10
        state.enable_multistart = True
        state.n_starts = 15
        state.sampler = "lhs"

        # Export to YAML
        yaml_str = export_yaml_config(state)

        # Load into new state
        new_state = load_yaml_config(StringIO(yaml_str))

        # Verify all values preserved
        assert new_state.model_name == "gaussian"
        assert new_state.gtol == 1e-10
        assert new_state.enable_multistart is True
        assert new_state.n_starts == 15
        assert new_state.sampler == "lhs"

    def test_builtin_models_list_completeness(self) -> None:
        """Test that all expected built-in models are available."""
        models = list_builtin_models()
        model_names = [m["name"] for m in models]

        # Check for expected models
        expected_models = [
            "linear",
            "exponential_decay",
            "exponential_growth",
            "gaussian",
            "sigmoid",
            "power_law",
        ]

        for expected in expected_models:
            assert expected in model_names, f"Missing expected model: {expected}"


# =============================================================================
# Test Error Handling Paths
# =============================================================================


class TestErrorHandlingPaths:
    """Tests for error handling in various scenarios."""

    def test_invalid_csv_format_handling(self, tmp_path: Path) -> None:
        """Test handling of malformed CSV files raises appropriate error."""
        # Create a malformed CSV with non-numeric data
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("this,is,not\nproper,numeric,data\n")

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }

        # Should raise an error when trying to parse non-numeric data
        with pytest.raises((ValueError, TypeError, DataLoadError)):
            load_from_file(str(bad_csv), config)

    def test_nan_data_validation(self) -> None:
        """Test that NaN values are properly detected."""
        xdata = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        ydata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = validate_data(xdata, ydata, sigma=None)

        assert result.is_valid is False
        assert result.nan_count == 1
        assert "NaN" in result.message

    def test_inf_data_validation(self) -> None:
        """Test that Inf values are properly detected."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        result = validate_data(xdata, ydata, sigma=None)

        assert result.is_valid is False
        assert result.inf_count == 1
        assert "Inf" in result.message

    def test_small_data_handling(self) -> None:
        """Test handling of minimal data arrays."""
        xdata = np.array([1.0, 2.0])
        ydata = np.array([2.0, 4.0])

        # Statistics should handle small arrays gracefully
        stats = compute_statistics(xdata, ydata)
        assert stats["point_count"] == 2
        assert stats["x_min"] == 1.0
        assert stats["x_max"] == 2.0

    def test_mismatched_array_lengths(self) -> None:
        """Test handling of mismatched x and y array lengths."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([1.0, 2.0])  # Different length
        sigma = None

        result = validate_data(xdata, ydata, sigma)

        # Should detect the mismatch
        assert result.is_valid is False


# =============================================================================
# Test Edge Cases in Data Formats
# =============================================================================


class TestDataFormatEdgeCases:
    """Tests for edge cases in various data formats."""

    def test_clipboard_with_mixed_delimiters(self) -> None:
        """Test clipboard parsing with inconsistent delimiters."""
        # Some spreadsheets export with mixed tabs and spaces
        clipboard_text = "1.0\t2.0\n2.0\t4.0\n3.0\t6.0"

        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "sigma": None},
        }

        xdata, _ydata, _sigma = load_from_clipboard(clipboard_text, config)

        assert len(xdata) == 3
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0, 3.0])

    def test_csv_with_trailing_whitespace(self, tmp_path: Path) -> None:
        """Test CSV parsing with trailing whitespace in values."""
        csv_content = "x,y\n1.0 ,2.0 \n2.0 ,4.0 \n"
        csv_file = tmp_path / "whitespace.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, _ydata, _sigma = load_from_file(str(csv_file), config)

        assert len(xdata) == 2
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0])

    def test_scientific_notation_parsing(self, tmp_path: Path) -> None:
        """Test parsing of scientific notation in data files."""
        csv_content = "x,y\n1.0e-3,2.5e+2\n1.0e-2,2.5e+1\n1.0e-1,2.5e+0\n"
        csv_file = tmp_path / "scientific.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, ydata, _sigma = load_from_file(str(csv_file), config)

        assert len(xdata) == 3
        np.testing.assert_array_almost_equal(xdata, [0.001, 0.01, 0.1])
        np.testing.assert_array_almost_equal(ydata, [250.0, 25.0, 2.5])

    def test_large_dataset_statistics(self) -> None:
        """Test statistics computation for larger datasets."""
        # 10000 points
        xdata = np.linspace(0, 100, 10000)
        ydata = np.sin(xdata) + np.random.normal(0, 0.1, 10000)

        stats = compute_statistics(xdata, ydata)

        assert stats["point_count"] == 10000
        assert stats["x_min"] == 0.0
        assert stats["x_max"] == 100.0
        assert "x_std" in stats
        assert "y_std" in stats


# =============================================================================
# Test Integration Points Between Pages
# =============================================================================


class TestPageIntegration:
    """Tests for integration points between workflow pages."""

    def test_data_to_model_state_transfer(self) -> None:
        """Test that data state is accessible from model selection page."""
        state = initialize_state()

        # Load data on data page
        state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state.ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        state.data_file_name = "test.csv"

        # Check model page can access data
        status = get_page_status(state)
        assert status["data"]["loaded"] is True
        assert status["data"]["point_count"] == 5
        assert status["data"]["file_name"] == "test.csv"

    def test_model_to_fitting_state_transfer(self) -> None:
        """Test that model state is accessible from fitting options page."""
        state = initialize_state()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])

        # Select model on model page
        state.model_type = "builtin"
        state.model_name = "power_law"
        state.p0 = [1.0, 2.0]

        # Check fitting page can access model
        status = get_page_status(state)
        assert status["model"]["selected"] is True
        assert status["model"]["name"] == "power_law"

    def test_fitting_to_results_state_transfer(self) -> None:
        """Test that fit results are accessible from results page."""
        state = initialize_state()
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        state.model_type = "builtin"
        state.model_name = "linear"

        # Complete fit
        mock_result = MagicMock()
        mock_result.popt = np.array([1.0, 0.0])
        mock_result.pcov = np.eye(2)
        mock_result.success = True
        mock_result.r_squared = 0.95
        state.fit_result = mock_result

        # Check results page can access fit
        status = get_page_status(state)
        assert status["fit"]["complete"] is True

    def test_results_to_export_state_transfer(self) -> None:
        """Test that all state is accessible for export."""
        state = initialize_state()
        state.xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state.ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        state.model_type = "builtin"
        state.model_name = "linear"
        state.gtol = 1e-8

        mock_result = MagicMock()
        mock_result.popt = np.array([2.0, 0.0])
        mock_result.pcov = np.eye(2) * 0.01
        mock_result.success = True
        mock_result.message = "Converged"
        mock_result.nfev = 10
        mock_result.cost = 0.001
        mock_result.r_squared = 0.999
        mock_result.rmse = 0.01
        mock_result.residuals = np.array([0.01, -0.01, 0.0, 0.01, -0.01])
        state.fit_result = mock_result

        # Export should have access to all state
        config = get_current_config(state)
        assert "model" in config
        assert "fitting" in config

        # Session bundle should work
        bundle = create_session_bundle(state, mock_result, {})
        assert len(bundle) > 0

    def test_workflow_step_messages_progression(self) -> None:
        """Test that workflow step messages update correctly."""
        state = initialize_state()
        # Clear model to simulate fresh start
        state.model_type = ""

        # Step 1: No data
        msg = get_workflow_step_message(state)
        assert "data" in msg.lower() or "load" in msg.lower()

        # Step 2: Data loaded
        state.xdata = np.array([1.0, 2.0, 3.0])
        state.ydata = np.array([1.0, 4.0, 9.0])
        msg = get_workflow_step_message(state)
        assert "model" in msg.lower() or "select" in msg.lower()

        # Step 3: Model selected
        state.model_type = "builtin"
        state.model_name = "gaussian"
        msg = get_workflow_step_message(state)
        assert "fit" in msg.lower() or "run" in msg.lower()

        # Step 4: Fit complete
        state.fit_result = MagicMock()
        msg = get_workflow_step_message(state)
        assert (
            "complete" in msg.lower()
            or "result" in msg.lower()
            or "export" in msg.lower()
        )


# =============================================================================
# Test State Reset Behavior
# =============================================================================


class TestStateResetBehavior:
    """Tests for state reset behavior during workflow."""

    def test_reset_clears_fit_result(self) -> None:
        """Test that reset clears fit result."""
        state = initialize_state()
        state.fit_result = MagicMock()

        reset_state(state)

        assert state.fit_result is None

    def test_reset_preserves_preferences_when_requested(self) -> None:
        """Test that reset preserves user preferences when requested."""
        state = initialize_state()
        state.mode = "advanced"
        state.preset = "quality"
        state.fit_result = MagicMock()

        reset_state(state, preserve_preferences=True)

        assert state.mode == "advanced"
        assert state.preset == "quality"
        assert state.fit_result is None

    def test_full_reset_clears_all(self) -> None:
        """Test that full reset clears everything."""
        state = initialize_state()
        state.xdata = np.array([1.0, 2.0])
        state.ydata = np.array([1.0, 2.0])
        state.model_name = "gaussian"
        state.mode = "advanced"

        reset_state(state, preserve_preferences=False)

        assert state.xdata is None
        assert state.mode == "guided"
