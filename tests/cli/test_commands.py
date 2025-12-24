"""Tests for CLI commands (fit, batch, info).

This module tests the CLI entry point and all three commands:
- nlsq fit workflow.yaml - Execute single curve fit
- nlsq batch w1.yaml w2.yaml - Parallel batch fitting
- nlsq info - Display system information

Test Structure
--------------
- test_fit_command_executes_workflow: Basic fit execution
- test_fit_command_output_override: --output flag override
- test_fit_command_stdout: --stdout JSON output
- test_batch_command_parallel_execution: Multiple workflows in parallel
- test_batch_command_continue_on_error: Collect failures at end
- test_info_command: Displays system information
- test_version_flag: --version flag
- test_verbose_flag: --verbose flag
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.commands import batch as batch_command

# Import CLI modules for direct testing
from nlsq.cli.commands import fit as fit_command
from nlsq.cli.commands import info as info_command
from nlsq.cli.main import create_parser, main

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_data_file(temp_dir):
    """Create a sample ASCII data file for testing."""
    data_path = temp_dir / "test_data.txt"
    x = np.linspace(0, 10, 50)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, 50)
    np.savetxt(data_path, np.column_stack([x, y]), header="x y")
    return data_path


@pytest.fixture
def sample_workflow_yaml(temp_dir, sample_data_file):
    """Create a sample workflow YAML configuration."""
    yaml_path = temp_dir / "workflow.yaml"
    output_path = temp_dir / "results.json"

    # Use .as_posix() for cross-platform YAML compatibility (Windows backslashes
    # are interpreted as escape sequences in YAML double-quoted strings)
    data_file_posix = sample_data_file.as_posix()
    output_path_posix = output_path.as_posix()

    yaml_content = f"""
metadata:
  workflow_name: test_workflow
  dataset_id: test_dataset

data:
  input_file: "{data_file_posix}"
  format: ascii
  columns:
    x: 0
    y: 1

model:
  type: builtin
  name: linear

fitting:
  p0: [1.0, 0.0]
  method: trf

export:
  results_file: "{output_path_posix}"
  format: json

visualization:
  enabled: false
"""
    yaml_path.write_text(yaml_content)
    return yaml_path, output_path


@pytest.fixture
def sample_failing_workflow_yaml(temp_dir):
    """Create a workflow YAML that will fail (missing data file)."""
    yaml_path = temp_dir / "failing_workflow.yaml"

    yaml_content = """
metadata:
  workflow_name: failing_workflow

data:
  input_file: "/nonexistent/path/to/data.txt"
  format: ascii
  columns:
    x: 0
    y: 1

model:
  type: builtin
  name: linear

fitting:
  p0: [1.0, 0.0]
"""
    yaml_path.write_text(yaml_content)
    return yaml_path


# =============================================================================
# Test: nlsq fit workflow.yaml
# =============================================================================


def test_fit_command_executes_workflow(sample_workflow_yaml):
    """Test that 'nlsq fit workflow.yaml' executes workflow and produces output."""
    yaml_path, output_path = sample_workflow_yaml

    # Execute fit command handler directly
    result = fit_command.run_fit(str(yaml_path))

    # Verify result structure
    assert result is not None
    assert "popt" in result
    assert "pcov" in result
    assert len(result["popt"]) == 2  # Linear has 2 parameters

    # Verify output file was created
    assert output_path.exists(), "Results file should be created"

    # Verify output file content
    with open(output_path) as f:
        export_data = json.load(f)

    assert "popt" in export_data
    assert "uncertainties" in export_data
    assert "metadata" in export_data


def test_fit_command_output_override(sample_workflow_yaml, temp_dir):
    """Test that '--output /path/to/output.json' overrides export path."""
    yaml_path, _ = sample_workflow_yaml
    override_output = temp_dir / "custom_output.json"

    # Execute with output override
    result = fit_command.run_fit(str(yaml_path), output_override=str(override_output))

    assert result is not None

    # Verify custom output file was created
    assert override_output.exists(), "Custom output file should be created"

    # Verify content
    with open(override_output) as f:
        export_data = json.load(f)

    assert "popt" in export_data


def test_fit_command_stdout(sample_workflow_yaml, capsys):
    """Test that '--stdout' outputs JSON to stdout."""
    yaml_path, _ = sample_workflow_yaml

    # Execute with stdout mode
    result = fit_command.run_fit(str(yaml_path), stdout=True)

    assert result is not None

    # Capture stdout
    captured = capsys.readouterr()

    # Parse stdout as JSON
    stdout_data = json.loads(captured.out)
    assert "popt" in stdout_data
    assert "uncertainties" in stdout_data


# =============================================================================
# Test: nlsq batch w1.yaml w2.yaml
# =============================================================================


def test_batch_command_parallel_execution(temp_dir, sample_data_file):
    """Test that 'nlsq batch w1.yaml w2.yaml' executes workflows in parallel."""
    # Create multiple workflow files
    workflow_paths = []
    output_paths = []

    # Use .as_posix() for cross-platform YAML compatibility
    data_file_posix = sample_data_file.as_posix()

    for i in range(3):
        yaml_path = temp_dir / f"workflow_{i}.yaml"
        output_path = temp_dir / f"results_{i}.json"
        output_paths.append(output_path)
        output_path_posix = output_path.as_posix()

        yaml_content = f"""
metadata:
  workflow_name: test_workflow_{i}
  dataset_id: dataset_{i}

data:
  input_file: "{data_file_posix}"
  format: ascii
  columns:
    x: 0
    y: 1

model:
  type: builtin
  name: linear

fitting:
  p0: [1.0, 0.0]

export:
  results_file: "{output_path_posix}"
  format: json

visualization:
  enabled: false
"""
        yaml_path.write_text(yaml_content)
        workflow_paths.append(str(yaml_path))

    # Execute batch command
    summary_path = temp_dir / "batch_summary.json"
    results = batch_command.run_batch(
        workflow_paths, summary_file=str(summary_path), max_workers=2
    )

    # Verify all workflows executed
    assert len(results) == 3

    # Verify all output files created
    for output_path in output_paths:
        assert output_path.exists(), f"Output {output_path} should exist"

    # Verify summary file
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)

    assert "total" in summary
    assert summary["total"] == 3
    assert summary["succeeded"] == 3


def test_batch_command_continue_on_error(
    temp_dir, sample_data_file, sample_failing_workflow_yaml
):
    """Test that batch continues on error and collects failures at end."""
    # Create one valid workflow
    valid_yaml = temp_dir / "valid_workflow.yaml"
    valid_output = temp_dir / "valid_output.json"

    # Use .as_posix() for cross-platform YAML compatibility
    data_file_posix = sample_data_file.as_posix()
    valid_output_posix = valid_output.as_posix()

    yaml_content = f"""
metadata:
  workflow_name: valid_workflow

data:
  input_file: "{data_file_posix}"
  format: ascii
  columns:
    x: 0
    y: 1

model:
  type: builtin
  name: linear

fitting:
  p0: [1.0, 0.0]

export:
  results_file: "{valid_output_posix}"
  format: json

visualization:
  enabled: false
"""
    valid_yaml.write_text(yaml_content)

    # Execute batch with one failing and one passing workflow
    workflow_paths = [str(sample_failing_workflow_yaml), str(valid_yaml)]
    summary_path = temp_dir / "batch_summary.json"

    results = batch_command.run_batch(
        workflow_paths,
        summary_file=str(summary_path),
        continue_on_error=True,
        max_workers=1,
    )

    # Verify batch completed (did not abort on error)
    assert len(results) == 2

    # Verify summary includes failures
    assert summary_path.exists()
    with open(summary_path) as f:
        summary = json.load(f)

    assert summary["total"] == 2
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert len(summary.get("failures", [])) == 1


# =============================================================================
# Test: nlsq info
# =============================================================================


def test_info_command(capsys):
    """Test that 'nlsq info' displays version, Python, JAX backend, GPU info."""
    # Execute info command
    info_command.run_info()

    # Capture output
    captured = capsys.readouterr()
    output = captured.out

    # Verify key sections are present
    assert "NLSQ Version" in output or "nlsq" in output.lower()
    assert "Python" in output
    assert "JAX" in output or "jax" in output.lower()
    # GPU info may vary - just verify command runs successfully


# =============================================================================
# Test: nlsq --version
# =============================================================================


def test_version_flag(capsys):
    """Test that 'nlsq --version' displays NLSQ version."""
    parser = create_parser()

    # argparse exits on --version, so we catch SystemExit
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0

    # Check version was printed
    captured = capsys.readouterr()
    # Version output goes to stdout
    assert "nlsq" in captured.out.lower() or captured.out.strip() != ""


# =============================================================================
# Test: nlsq --verbose
# =============================================================================


def test_verbose_flag():
    """Test that '--verbose' flag is recognized and increases verbosity."""
    parser = create_parser()

    # Parse with verbose flag
    args = parser.parse_args(["--verbose", "info"])

    assert hasattr(args, "verbose")
    assert args.verbose is True

    # Test short form
    args_short = parser.parse_args(["-v", "info"])
    assert args_short.verbose is True


# =============================================================================
# Test: argparse structure
# =============================================================================


def test_parser_subcommands():
    """Test that parser has correct subcommands."""
    parser = create_parser()

    # Test fit subcommand
    args = parser.parse_args(["fit", "test.yaml"])
    assert args.command == "fit"
    assert args.workflow == "test.yaml"

    # Test batch subcommand
    args = parser.parse_args(["batch", "a.yaml", "b.yaml"])
    assert args.command == "batch"
    assert args.workflows == ["a.yaml", "b.yaml"]

    # Test info subcommand
    args = parser.parse_args(["info"])
    assert args.command == "info"


def test_fit_subcommand_arguments():
    """Test fit subcommand argument parsing."""
    parser = create_parser()

    # Test with --output
    args = parser.parse_args(["fit", "test.yaml", "--output", "out.json"])
    assert args.output == "out.json"

    # Test with --stdout
    args = parser.parse_args(["fit", "test.yaml", "--stdout"])
    assert args.stdout is True


def test_batch_subcommand_arguments():
    """Test batch subcommand argument parsing."""
    parser = create_parser()

    # Test with multiple files
    args = parser.parse_args(["batch", "a.yaml", "b.yaml", "c.yaml"])
    assert len(args.workflows) == 3

    # Test with --summary
    args = parser.parse_args(["batch", "a.yaml", "--summary", "summary.json"])
    assert args.summary == "summary.json"

    # Test with --workers
    args = parser.parse_args(["batch", "a.yaml", "--workers", "4"])
    assert args.workers == 4
