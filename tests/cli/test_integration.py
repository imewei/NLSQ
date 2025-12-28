"""Integration tests for NLSQ CLI feature.

This module contains end-to-end integration tests for the CLI feature,
focusing on critical user workflows that span multiple components.

Test Categories
---------------
1. End-to-end: nlsq fit with ASCII data, builtin model, JSON export
2. End-to-end: nlsq fit with CSV data, custom model, visualization
3. End-to-end: nlsq batch with multiple workflows, summary generation
4. Integration: Error handling across all components
5. Integration: YAML config parsing edge cases
6. Integration: Workflow with bounds and parameter estimation
7. Integration: Complete CLI invocation via subprocess
8. Integration: Batch processing with mixed success/failure workflows
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixture: Generate test data files
# =============================================================================


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with test data and configs."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create output directories
    (workspace / "output").mkdir()
    (workspace / "figures").mkdir()

    return workspace


@pytest.fixture
def linear_data_file(temp_workspace):
    """Create ASCII data file with linear data for testing."""
    data_path = temp_workspace / "linear_data.txt"
    np.random.seed(42)

    x = np.linspace(0, 10, 50)
    # True parameters: a=2.0, b=1.0
    y_true = 2.0 * x + 1.0
    y = y_true + np.random.normal(0, 0.1, 50)
    sigma = np.full(50, 0.1)

    # Write with header comment
    with open(data_path, "w") as f:
        f.write("# x y sigma\n")
        f.writelines(f"{x[i]:.6f} {y[i]:.6f} {sigma[i]:.6f}\n" for i in range(len(x)))

    return data_path


@pytest.fixture
def exponential_csv_data(temp_workspace):
    """Create CSV data file with exponential decay data."""
    data_path = temp_workspace / "exponential_data.csv"
    np.random.seed(123)

    x = np.linspace(0, 10, 40)
    # True parameters: a=3.0, b=0.4, c=0.5
    y_true = 3.0 * np.exp(-0.4 * x) + 0.5
    y = y_true + np.random.normal(0, 0.05, 40)
    sigma = np.full(40, 0.05)

    # Write CSV with header
    with open(data_path, "w") as f:
        f.write("x,y,sigma\n")
        f.writelines(f"{x[i]:.6f},{y[i]:.6f},{sigma[i]:.6f}\n" for i in range(len(x)))

    return data_path


# =============================================================================
# Test 1: End-to-end nlsq fit with ASCII data, builtin model, JSON export
# =============================================================================


class TestEndToEndASCIIFit:
    """End-to-end test: nlsq fit with ASCII data, builtin model, JSON export."""

    def test_fit_ascii_builtin_json_complete_workflow(
        self, temp_workspace, linear_data_file
    ):
        """Test complete workflow: load ASCII -> fit with linear model -> export JSON."""
        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.workflow_runner import WorkflowRunner

        # Create workflow config
        output_file = temp_workspace / "output" / "results.json"
        config = {
            "metadata": {
                "workflow_name": "test_ascii_linear",
                "dataset_id": "linear_test",
            },
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": 2},
                "ascii": {"comment_char": "#"},
            },
            "model": {
                "type": "builtin",
                "name": "linear",
            },
            "fitting": {
                "p0": "auto",
                "method": "trf",
            },
            "export": {
                "results_file": str(output_file),
                "format": "json",
            },
            "visualization": {
                "enabled": False,
            },
        }

        # Write config to YAML
        import yaml

        config_path = temp_workspace / "workflow.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Execute workflow
        result = run_fit(str(config_path))

        # Verify result structure
        assert result is not None
        assert "popt" in result
        assert "pcov" in result
        assert result["success"] is True

        # Verify fitted parameters are close to true values (a=2.0, b=1.0)
        popt = result["popt"]
        assert len(popt) == 2
        assert abs(popt[0] - 2.0) < 0.2  # slope
        assert abs(popt[1] - 1.0) < 0.5  # intercept

        # Verify JSON export was created
        assert output_file.exists()
        with open(output_file) as f:
            exported = json.load(f)

        assert "popt" in exported
        assert "uncertainties" in exported
        assert "metadata" in exported
        assert exported["metadata"]["workflow_name"] == "test_ascii_linear"


# =============================================================================
# Test 2: End-to-end nlsq fit with CSV data, custom model, visualization
# =============================================================================


class TestEndToEndCSVVisualization:
    """End-to-end test: nlsq fit with CSV data, custom model, visualization."""

    def test_fit_csv_custom_model_with_visualization(
        self, temp_workspace, exponential_csv_data
    ):
        """Test workflow with CSV data, custom model, and visualization output."""
        from nlsq.cli.commands.fit import run_fit

        # Use the custom model from fixtures
        custom_model_path = FIXTURES_DIR / "sample_custom_model.py"

        # Create workflow config with visualization
        output_file = temp_workspace / "output" / "results_custom.json"
        figures_dir = temp_workspace / "figures"

        config = {
            "metadata": {
                "workflow_name": "test_csv_custom",
                "dataset_id": "custom_test",
            },
            "data": {
                "input_file": str(exponential_csv_data),
                "format": "csv",
                "columns": {"x": "x", "y": "y", "sigma": "sigma"},
                "csv": {"header": True, "delimiter": ","},
            },
            "model": {
                "type": "custom",
                "path": str(custom_model_path),
                "function": "custom_linear",
            },
            "fitting": {
                "p0": [1.0, 0.0],
                "method": "trf",
            },
            "export": {
                "results_file": str(output_file),
                "format": "json",
            },
            "visualization": {
                "enabled": True,
                "output_dir": str(figures_dir),
                "filename_prefix": "custom_fit",
                "formats": ["png"],
                "dpi": 100,
                "figure_size": [6.0, 4.5],
                "style": "publication",
                "layout": "combined",
                "main_plot": {
                    "show_grid": True,
                    "data": {
                        "marker": "o",
                        "color": "#1f77b4",
                        "size": 20,
                        "alpha": 0.7,
                        "show_errorbars": True,
                    },
                    "fit": {"color": "#d62728", "linewidth": 1.5, "n_points": 100},
                    "confidence_band": {"enabled": False},
                    "legend": {"enabled": True, "location": "best"},
                    "annotation": {
                        "enabled": True,
                        "show_r_squared": True,
                        "show_rmse": True,
                        "location": "upper right",
                        "fontsize": 9,
                    },
                },
                "residuals_plot": {
                    "enabled": True,
                    "type": "scatter",
                    "show_zero_line": True,
                    "std_bands": {"enabled": True, "levels": [1, 2]},
                },
                "histogram": {"enabled": False},
                "color_schemes": {
                    "default": {
                        "data": "#1f77b4",
                        "fit": "#d62728",
                        "residuals": "#2ca02c",
                        "confidence": "#ff7f0e",
                    },
                },
                "active_scheme": "default",
            },
        }

        # Write config to YAML
        import yaml

        config_path = temp_workspace / "workflow_viz.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Execute workflow
        result = run_fit(str(config_path))

        # Verify result
        assert result is not None
        assert result["success"] is True

        # Verify JSON export
        assert output_file.exists()

        # Verify visualization was created
        png_files = list(figures_dir.glob("*.png"))
        assert len(png_files) > 0, "No PNG files generated"


# =============================================================================
# Test 3: End-to-end nlsq batch with multiple workflows, summary generation
# =============================================================================


class TestEndToEndBatchProcessing:
    """End-to-end test: nlsq batch with multiple workflows."""

    def test_batch_multiple_workflows_with_summary(
        self, temp_workspace, linear_data_file
    ):
        """Test batch processing multiple workflows with summary generation."""
        import yaml

        from nlsq.cli.commands.batch import run_batch

        # Create multiple workflow configs
        workflow_paths = []
        output_files = []

        for i in range(3):
            output_file = temp_workspace / "output" / f"results_{i}.json"
            output_files.append(output_file)

            config = {
                "metadata": {
                    "workflow_name": f"batch_workflow_{i}",
                    "dataset_id": f"dataset_{i}",
                },
                "data": {
                    "input_file": str(linear_data_file),
                    "format": "ascii",
                    "columns": {"x": 0, "y": 1, "sigma": 2},
                    "ascii": {"comment_char": "#"},
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
                    "results_file": str(output_file),
                    "format": "json",
                },
                "visualization": {
                    "enabled": False,
                },
            }

            config_path = temp_workspace / f"workflow_{i}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            workflow_paths.append(str(config_path))

        # Run batch
        summary_file = temp_workspace / "output" / "batch_summary.json"
        results = run_batch(
            workflow_paths,
            summary_file=str(summary_file),
            max_workers=2,
        )

        # Verify all workflows completed
        assert len(results) == 3

        # Verify all output files exist
        for output_file in output_files:
            assert output_file.exists()

        # Verify summary file
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["total"] == 3
        assert summary["succeeded"] == 3
        assert summary["failed"] == 0


# =============================================================================
# Test 4: Integration test for error handling across components
# =============================================================================


class TestErrorHandlingIntegration:
    """Integration test: Error handling across all components."""

    def test_data_load_error_propagates_to_cli(self, temp_workspace):
        """Test that DataLoadError from data loader propagates correctly."""
        import yaml

        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.errors import DataLoadError

        # Create config with non-existent file
        config = {
            "metadata": {"workflow_name": "error_test"},
            "data": {
                "input_file": str(temp_workspace / "nonexistent_file.txt"),
                "format": "ascii",
                "columns": {"x": 0, "y": 1},
            },
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {"p0": [1.0, 0.0]},
            "export": {"results_file": str(temp_workspace / "output" / "results.json")},
        }

        config_path = temp_workspace / "error_workflow.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Should raise DataLoadError
        with pytest.raises(DataLoadError) as exc_info:
            run_fit(str(config_path))

        assert (
            "not found" in str(exc_info.value).lower()
            or "nonexistent" in str(exc_info.value).lower()
        )

    def test_model_error_propagates_to_cli(self, temp_workspace, linear_data_file):
        """Test that ModelError from model registry propagates correctly."""
        import yaml

        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.errors import ModelError

        # Create config with non-existent model
        config = {
            "metadata": {"workflow_name": "model_error_test"},
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": 2},
                "ascii": {"comment_char": "#"},
            },
            "model": {"type": "builtin", "name": "nonexistent_model_xyz"},
            "fitting": {"p0": [1.0, 0.0]},
            "export": {"results_file": str(temp_workspace / "output" / "results.json")},
        }

        config_path = temp_workspace / "model_error_workflow.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Should raise ModelError
        with pytest.raises(ModelError) as exc_info:
            run_fit(str(config_path))

        assert "nonexistent_model_xyz" in str(exc_info.value)


# =============================================================================
# Test 5: YAML config parsing edge cases
# =============================================================================


class TestYAMLConfigEdgeCases:
    """Integration test: YAML config parsing edge cases."""

    def test_missing_required_data_section(self, temp_workspace):
        """Test that missing required 'data' section raises CLIError."""
        import yaml

        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.errors import CLIError, ConfigError

        # Create config without required 'data' section
        config = {
            "metadata": {"workflow_name": "missing_data"},
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {"p0": [1.0, 0.0]},
        }

        config_path = temp_workspace / "missing_data.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Implementation raises CLIError for missing data.input_file
        with pytest.raises((CLIError, ConfigError, KeyError)):
            run_fit(str(config_path))

    def test_missing_required_model_section(self, temp_workspace, linear_data_file):
        """Test that missing required 'model' section raises CLIError."""
        import yaml

        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.errors import CLIError, ConfigError

        # Create config without required 'model' section
        config = {
            "metadata": {"workflow_name": "missing_model"},
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1},
            },
            "fitting": {"p0": [1.0, 0.0]},
        }

        config_path = temp_workspace / "missing_model.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Implementation raises CLIError for missing model section
        with pytest.raises((CLIError, ConfigError, KeyError)):
            run_fit(str(config_path))

    def test_invalid_yaml_syntax(self, temp_workspace):
        """Test that invalid YAML syntax raises appropriate error."""
        import yaml

        from nlsq.cli.commands.fit import run_fit
        from nlsq.cli.errors import ConfigError

        # Write invalid YAML
        config_path = temp_workspace / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("metadata:\n  workflow_name: test\n  invalid yaml: [unterminated")

        with pytest.raises((ConfigError, yaml.YAMLError)):
            run_fit(str(config_path))


# =============================================================================
# Test 6: Workflow with bounds and auto parameter estimation
# =============================================================================


class TestWorkflowWithBounds:
    """Integration test: Workflow with bounds and parameter estimation."""

    def test_workflow_with_explicit_bounds(self, temp_workspace, linear_data_file):
        """Test workflow with explicit parameter bounds."""
        import yaml

        from nlsq.cli.workflow_runner import WorkflowRunner

        config = {
            "metadata": {"workflow_name": "bounds_test"},
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": 2},
                "ascii": {"comment_char": "#"},
            },
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {
                "p0": [1.5, 0.5],
                "bounds": {
                    "lower": [0.0, -10.0],
                    "upper": [10.0, 10.0],
                },
                "method": "trf",
            },
            "export": {
                "results_file": str(temp_workspace / "output" / "bounds_results.json"),
                "format": "json",
            },
            "visualization": {"enabled": False},
        }

        runner = WorkflowRunner()
        result = runner.run(config)

        # Verify fit succeeded
        assert result["success"] is True

        # Verify parameters are within bounds
        popt = result["popt"]
        assert 0.0 <= popt[0] <= 10.0
        assert -10.0 <= popt[1] <= 10.0


# =============================================================================
# Test 7: Complete CLI invocation via subprocess
# =============================================================================


@pytest.mark.slow  # Skip in fast tests (-m "not slow")
@pytest.mark.serial  # Run on single xdist worker to prevent resource contention
class TestCLISubprocessInvocation:
    """Integration test: Complete CLI invocation via subprocess.

    These tests spawn external Python processes that initialize JAX.
    They are marked for serial execution to prevent resource contention
    when running with pytest-xdist parallel execution.

    Root Cause Analysis (2025-12-27):
    - Each subprocess spawns a process that initializes JAX (~620ms + 500MB memory)
    - With -n 4 workers, parallel JAX initializations cause compilation cache deadlocks
    - Serial execution prevents resource contention and system freezes
    """

    def test_cli_fit_command_via_subprocess(self, temp_workspace, linear_data_file):
        """Test invoking 'nlsq fit' via subprocess."""
        import yaml

        # Create workflow config
        output_file = temp_workspace / "output" / "subprocess_results.json"
        config = {
            "metadata": {"workflow_name": "subprocess_test"},
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": 2},
                "ascii": {"comment_char": "#"},
            },
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {"p0": [1.0, 0.0], "method": "trf"},
            "export": {"results_file": str(output_file), "format": "json"},
            "visualization": {"enabled": False},
        }

        config_path = temp_workspace / "subprocess_workflow.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Run via subprocess
        result = subprocess.run(
            [sys.executable, "-m", "nlsq.cli.main", "fit", str(config_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Check return code
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        # Verify output file exists
        assert output_file.exists()

    def test_cli_info_command_via_subprocess(self):
        """Test invoking 'nlsq info' via subprocess."""
        result = subprocess.run(
            [sys.executable, "-m", "nlsq.cli.main", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check return code
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"

        # Verify output contains expected sections
        output = result.stdout
        assert "NLSQ" in output or "nlsq" in output.lower()
        assert "Python" in output


# =============================================================================
# Test 8: Batch processing with mixed success/failure workflows
# =============================================================================


class TestBatchMixedResults:
    """Integration test: Batch processing with mixed success/failure."""

    def test_batch_continue_on_error_collects_failures(
        self, temp_workspace, linear_data_file
    ):
        """Test that batch continues after errors and collects all failures."""
        import yaml

        from nlsq.cli.commands.batch import run_batch

        workflow_paths = []

        # Create one valid workflow
        valid_config = {
            "metadata": {"workflow_name": "valid_workflow"},
            "data": {
                "input_file": str(linear_data_file),
                "format": "ascii",
                "columns": {"x": 0, "y": 1, "sigma": 2},
                "ascii": {"comment_char": "#"},
            },
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {"p0": [1.0, 0.0]},
            "export": {
                "results_file": str(temp_workspace / "output" / "valid_results.json"),
                "format": "json",
            },
            "visualization": {"enabled": False},
        }

        valid_path = temp_workspace / "valid_workflow.yaml"
        with open(valid_path, "w") as f:
            yaml.dump(valid_config, f)
        workflow_paths.append(str(valid_path))

        # Create one failing workflow (non-existent data file)
        failing_config = {
            "metadata": {"workflow_name": "failing_workflow"},
            "data": {
                "input_file": str(temp_workspace / "nonexistent_data.txt"),
                "format": "ascii",
                "columns": {"x": 0, "y": 1},
            },
            "model": {"type": "builtin", "name": "linear"},
            "fitting": {"p0": [1.0, 0.0]},
            "export": {
                "results_file": str(temp_workspace / "output" / "failing_results.json"),
                "format": "json",
            },
        }

        failing_path = temp_workspace / "failing_workflow.yaml"
        with open(failing_path, "w") as f:
            yaml.dump(failing_config, f)
        workflow_paths.append(str(failing_path))

        # Run batch with continue_on_error=True
        summary_file = temp_workspace / "output" / "mixed_summary.json"
        results = run_batch(
            workflow_paths,
            summary_file=str(summary_file),
            continue_on_error=True,
            max_workers=1,
        )

        # Verify batch completed (did not abort on first error)
        assert len(results) == 2

        # Verify summary contains both successes and failures
        assert summary_file.exists()
        with open(summary_file) as f:
            summary = json.load(f)

        assert summary["total"] == 2
        assert summary["succeeded"] == 1
        assert summary["failed"] == 1
        assert len(summary.get("failures", [])) == 1

        # Verify valid workflow output exists
        assert (temp_workspace / "output" / "valid_results.json").exists()
