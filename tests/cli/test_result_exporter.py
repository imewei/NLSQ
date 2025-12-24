"""Tests for result exporter module.

Tests for JSON export, CSV export, and stdout export functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestResultExporter:
    """Tests for ResultExporter class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample fit result for testing."""
        return {
            "popt": np.array([1.5, 0.3, 2.0]),
            "pcov": np.array(
                [
                    [0.01, 0.001, 0.002],
                    [0.001, 0.02, 0.003],
                    [0.002, 0.003, 0.03],
                ]
            ),
            "success": True,
            "message": "Optimization converged",
            "nfev": 42,
            "njev": 15,
            "cost": 0.0025,
            "fun": np.array([0.01, -0.02, 0.015, -0.005]),
        }

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a sample export configuration."""
        return {
            "metadata": {
                "workflow_name": "test_workflow",
                "dataset_id": "dataset_001",
            },
            "model": {
                "name": "exponential_decay",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

    def test_json_export_with_full_metadata(
        self, sample_result, sample_config, tmp_path
    ):
        """Test JSON export includes all required metadata."""
        from nlsq.cli.result_exporter import ResultExporter

        exporter = ResultExporter()
        sample_config["export"]["results_file"] = str(tmp_path / "results.json")

        exporter.export(sample_result, sample_config)

        # Read the exported file
        output_path = tmp_path / "results.json"
        assert output_path.exists()

        with open(output_path) as f:
            exported = json.load(f)

        # Verify popt is present and correct
        assert "popt" in exported
        assert exported["popt"] == pytest.approx([1.5, 0.3, 2.0], rel=1e-6)

        # Verify pcov is present and correct shape
        assert "pcov" in exported
        assert len(exported["pcov"]) == 3
        assert len(exported["pcov"][0]) == 3

        # Verify uncertainties are calculated
        assert "uncertainties" in exported
        expected_uncertainties = np.sqrt(np.diag(sample_result["pcov"]))
        assert exported["uncertainties"] == pytest.approx(
            expected_uncertainties.tolist(), rel=1e-6
        )

        # Verify statistics are present
        assert "statistics" in exported
        stats = exported["statistics"]
        assert "r_squared" in stats or "rmse" in stats

        # Verify convergence info
        assert "convergence" in exported
        convergence = exported["convergence"]
        assert convergence["iterations"] == 42
        assert convergence["function_evals"] == 42  # nfev
        assert convergence["status"] == "success"

        # Verify workflow metadata
        assert "metadata" in exported
        metadata = exported["metadata"]
        assert metadata["workflow_name"] == "test_workflow"
        assert metadata["dataset_id"] == "dataset_001"
        assert metadata["model_id"] == "exponential_decay"

    def test_csv_export_with_flattened_rows(
        self, sample_result, sample_config, tmp_path
    ):
        """Test CSV export with flattened parameter name/value/uncertainty rows."""
        from nlsq.cli.result_exporter import ResultExporter

        exporter = ResultExporter()
        sample_config["export"]["results_file"] = str(tmp_path / "results.csv")
        sample_config["export"]["format"] = "csv"

        exporter.export(sample_result, sample_config)

        # Read the exported file
        output_path = tmp_path / "results.csv"
        assert output_path.exists()

        with open(output_path) as f:
            content = f.read()

        # Verify CSV structure
        lines = content.strip().split("\n")
        assert len(lines) > 1  # Header + data rows

        # Verify header
        header = lines[0]
        assert "parameter" in header.lower() or "name" in header.lower()
        assert "value" in header.lower()
        assert "uncertainty" in header.lower() or "error" in header.lower()

        # Verify parameter rows exist
        # Should have rows for p0, p1, p2 at minimum
        assert len(lines) >= 4  # Header + 3 parameters

    def test_stdout_json_output_for_piping(self, sample_result, sample_config, capsys):
        """Test stdout JSON output for piping to other tools."""
        from nlsq.cli.result_exporter import ResultExporter

        exporter = ResultExporter()
        sample_config["export"]["stdout"] = True

        exporter.export(sample_result, sample_config)

        # Capture stdout
        captured = capsys.readouterr()

        # Verify JSON was written to stdout
        output = captured.out
        assert output.strip()  # Not empty

        # Should be valid JSON
        parsed = json.loads(output)
        assert "popt" in parsed
        assert parsed["popt"] == pytest.approx([1.5, 0.3, 2.0], rel=1e-6)

    def test_numpy_arrays_converted_to_lists(
        self, sample_result, sample_config, tmp_path
    ):
        """Test that numpy arrays are properly converted to lists in JSON."""
        from nlsq.cli.result_exporter import ResultExporter

        exporter = ResultExporter()
        sample_config["export"]["results_file"] = str(tmp_path / "results.json")

        # Ensure result contains numpy arrays
        assert isinstance(sample_result["popt"], np.ndarray)
        assert isinstance(sample_result["pcov"], np.ndarray)

        exporter.export(sample_result, sample_config)

        # Read and parse JSON
        with open(tmp_path / "results.json") as f:
            exported = json.load(f)

        # Verify values are lists, not numpy arrays
        assert isinstance(exported["popt"], list)
        assert isinstance(exported["pcov"], list)
        assert isinstance(exported["pcov"][0], list)

    def test_stdout_mode_skips_file_writing(
        self, sample_result, sample_config, tmp_path, capsys
    ):
        """Test that stdout mode skips file writing when file path is also provided."""
        from nlsq.cli.result_exporter import ResultExporter

        exporter = ResultExporter()
        output_file = tmp_path / "should_not_exist.json"
        sample_config["export"]["results_file"] = str(output_file)
        sample_config["export"]["stdout"] = True
        sample_config["export"]["skip_file_on_stdout"] = True

        exporter.export(sample_result, sample_config)

        # Verify stdout was written
        captured = capsys.readouterr()
        assert captured.out.strip()

        # Verify file was NOT written when skip_file_on_stdout is True
        assert not output_file.exists()


class TestWorkflowRunner:
    """Tests for WorkflowRunner class."""

    @pytest.fixture
    def sample_ascii_data(self, tmp_path):
        """Create sample ASCII data file with exponential decay data."""
        # Generate exponential decay: y = 2.5 * exp(-0.3 * x) + 0.5
        x = np.linspace(0, 10, 50)
        np.random.seed(42)
        y_true = 2.5 * np.exp(-0.3 * x) + 0.5
        y_noisy = y_true + np.random.normal(0, 0.05, len(x))

        data_file = tmp_path / "data.txt"
        np.savetxt(data_file, np.column_stack([x, y_noisy]), delimiter=" ")
        return data_file

    @pytest.fixture
    def builtin_model_config(self, sample_ascii_data, tmp_path):
        """Create config for builtin model workflow."""
        return {
            "metadata": {
                "workflow_name": "test_builtin_workflow",
                "dataset_id": "test_dataset",
            },
            "data": {
                "input_file": str(sample_ascii_data),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": None},
            },
            "model": {
                "type": "builtin",
                "name": "exponential_decay",
            },
            "fitting": {
                "p0": "auto",
                "method": "trf",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

    def test_workflow_execution_with_builtin_model(
        self, builtin_model_config, tmp_path
    ):
        """Test complete workflow execution with a builtin model."""
        from nlsq.cli.workflow_runner import WorkflowRunner

        runner = WorkflowRunner()
        result = runner.run(builtin_model_config)

        # Verify result structure
        assert "popt" in result
        assert "pcov" in result
        assert result["success"] is True

        # Verify fitted parameters are reasonable
        # Expected: a ~ 2.5, b ~ 0.3, c ~ 0.5
        popt = result["popt"]
        assert len(popt) == 3  # exponential_decay has 3 parameters

        # Parameters should be in reasonable range
        assert 1.5 < popt[0] < 4.0  # amplitude
        assert 0.1 < popt[1] < 0.6  # decay rate
        assert 0.0 < popt[2] < 1.5  # offset

    def test_workflow_execution_with_custom_model(self, tmp_path):
        """Test workflow execution with a custom model from file."""
        from nlsq.cli.workflow_runner import WorkflowRunner

        # Create custom model file
        custom_model_file = tmp_path / "my_model.py"
        custom_model_file.write_text('''
import jax.numpy as jnp
import numpy as np

def linear_model(x, a, b):
    """Simple linear model: y = a * x + b"""
    return a * x + b

def estimate_p0(xdata, ydata):
    """Estimate initial parameters."""
    return [1.0, 0.0]

def bounds():
    """Return parameter bounds."""
    return ([-np.inf, -np.inf], [np.inf, np.inf])
''')

        # Create data file with linear data
        x = np.linspace(0, 10, 30)
        np.random.seed(42)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, len(x))
        data_file = tmp_path / "linear_data.txt"
        np.savetxt(data_file, np.column_stack([x, y]), delimiter=" ")

        config = {
            "metadata": {
                "workflow_name": "custom_model_test",
                "dataset_id": "linear_dataset",
            },
            "data": {
                "input_file": str(data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": None},
            },
            "model": {
                "type": "custom",
                "path": str(custom_model_file),
                "function": "linear_model",
            },
            "fitting": {
                "p0": [1.0, 0.0],
                "method": "trf",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

        runner = WorkflowRunner()
        result = runner.run(config)

        # Verify fit succeeded
        assert result["success"] is True
        assert len(result["popt"]) == 2

        # Check parameters are close to expected
        # Expected: a ~ 2.0, b ~ 1.0
        assert 1.8 < result["popt"][0] < 2.2  # slope
        assert 0.5 < result["popt"][1] < 1.5  # intercept

    def test_workflow_execution_with_bounds_and_p0_from_config(
        self, sample_ascii_data, tmp_path
    ):
        """Test workflow execution with explicit bounds and p0 from config."""
        from nlsq.cli.workflow_runner import WorkflowRunner

        config = {
            "metadata": {
                "workflow_name": "bounds_test",
                "dataset_id": "test_data",
            },
            "data": {
                "input_file": str(sample_ascii_data),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": None},
            },
            "model": {
                "type": "builtin",
                "name": "exponential_decay",
            },
            "fitting": {
                "p0": [3.0, 0.4, 0.3],  # Explicit initial guess
                "bounds": {
                    "lower": [0.0, 0.0, 0.0],
                    "upper": [10.0, 2.0, 5.0],
                },
                "method": "trf",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

        runner = WorkflowRunner()
        result = runner.run(config)

        # Verify fit succeeded
        assert result["success"] is True

        # Verify parameters are within bounds
        popt = result["popt"]
        assert 0.0 <= popt[0] <= 10.0
        assert 0.0 <= popt[1] <= 2.0
        assert 0.0 <= popt[2] <= 5.0

    def test_error_propagation_from_data_load_failures(self, tmp_path):
        """Test that data load failures are properly propagated as DataLoadError."""
        from nlsq.cli.errors import DataLoadError
        from nlsq.cli.workflow_runner import WorkflowRunner

        # Reference a non-existent file
        config = {
            "metadata": {
                "workflow_name": "fail_test",
                "dataset_id": "bad_data",
            },
            "data": {
                "input_file": str(tmp_path / "nonexistent_file.txt"),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": None},
            },
            "model": {
                "type": "builtin",
                "name": "linear",
            },
            "fitting": {
                "p0": [1.0, 0.0],
                "method": "trf",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

        runner = WorkflowRunner()

        # This should raise a DataLoadError due to missing file
        with pytest.raises(DataLoadError):
            runner.run(config)

    def test_error_propagation_from_model_failures(self, sample_ascii_data, tmp_path):
        """Test that model resolution failures are properly propagated as ModelError."""
        from nlsq.cli.errors import ModelError
        from nlsq.cli.workflow_runner import WorkflowRunner

        config = {
            "metadata": {
                "workflow_name": "fail_test",
                "dataset_id": "test_data",
            },
            "data": {
                "input_file": str(sample_ascii_data),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": None},
            },
            "model": {
                "type": "builtin",
                "name": "nonexistent_model_xyz",  # This model doesn't exist
            },
            "fitting": {
                "p0": [1.0, 0.0],
                "method": "trf",
            },
            "export": {
                "results_file": str(tmp_path / "results.json"),
                "format": "json",
            },
        }

        runner = WorkflowRunner()

        # This should raise a ModelError due to invalid model name
        with pytest.raises(ModelError):
            runner.run(config)
