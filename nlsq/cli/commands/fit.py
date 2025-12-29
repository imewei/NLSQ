"""Fit command handler for NLSQ CLI.

This module provides the 'nlsq fit' command for executing single curve fits
from YAML workflow configuration files.

Example Usage
-------------
From command line:
    $ nlsq fit workflow.yaml
    $ nlsq fit workflow.yaml --output results.json
    $ nlsq fit workflow.yaml --stdout

From Python:
    >>> from nlsq.cli.commands.fit import run_fit
    >>> result = run_fit("workflow.yaml")
    >>> result = run_fit("workflow.yaml", output_override="results.json")
    >>> result = run_fit("workflow.yaml", stdout=True)
"""

from pathlib import Path
from typing import Any

import yaml

from nlsq.cli.errors import ConfigError, setup_logging
from nlsq.cli.visualization import FitVisualizer
from nlsq.cli.workflow_runner import WorkflowRunner


def run_fit(
    workflow_path: str,
    output_override: str | None = None,
    stdout: bool = False,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """Execute a single curve fit from a YAML workflow configuration.

    Parameters
    ----------
    workflow_path : str
        Path to the workflow YAML configuration file.
    output_override : str, optional
        Override the export.results_file path.
    stdout : bool
        If True, output results as JSON to stdout.
    verbose : bool
        If True, enable verbose logging.

    Returns
    -------
    dict or None
        Fit result dictionary if successful, None if failed.

    Raises
    ------
    ConfigError
        If the YAML file cannot be parsed or is invalid.
    DataLoadError
        If the data file cannot be loaded.
    ModelError
        If the model function cannot be resolved.
    FitError
        If curve fitting fails.
    """
    # Set up logging
    logger = setup_logging(
        verbosity=2 if verbose else 1,
        console=True,
    )

    # Validate workflow path
    workflow_file = Path(workflow_path)
    if not workflow_file.exists():
        raise ConfigError(
            f"Workflow file not found: {workflow_path}",
            config_file=workflow_path,
            suggestion="Check that the file path is correct",
        )

    if not workflow_file.is_file():
        raise ConfigError(
            f"Workflow path is not a file: {workflow_path}",
            config_file=workflow_path,
            suggestion="Provide a path to a YAML file, not a directory",
        )

    # Load YAML configuration
    try:
        with open(workflow_file) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Failed to parse YAML: {e}",
            config_file=workflow_path,
            suggestion="Check YAML syntax for errors",
        ) from e

    if not isinstance(config, dict):
        raise ConfigError(
            "Invalid YAML: configuration must be a dictionary",
            config_file=workflow_path,
            suggestion="Ensure the YAML file contains a valid configuration object",
        )

    # Apply output override
    if output_override is not None:
        if "export" not in config:
            config["export"] = {}
        config["export"]["results_file"] = output_override

    # Apply stdout mode
    if stdout:
        if "export" not in config:
            config["export"] = {}
        config["export"]["stdout"] = True
        config["export"]["skip_file_on_stdout"] = True

    # Log start
    workflow_name = config.get("metadata", {}).get("workflow_name", "unnamed")
    if verbose:
        logger.info(f"Starting workflow: {workflow_name}")
        logger.info(f"Configuration file: {workflow_path}")

    # Execute workflow
    runner = WorkflowRunner()
    result = runner.run(config)

    # Generate visualization if enabled
    vis_config = config.get("visualization", {})
    if vis_config.get("enabled", False):
        _generate_visualization(result, config, runner, verbose, logger)

    if verbose:
        logger.info(f"Workflow completed: {workflow_name}")

    return result


def _generate_visualization(
    result: dict[str, Any],
    config: dict[str, Any],
    runner: WorkflowRunner,
    verbose: bool,
    logger: Any,
) -> None:
    """Generate visualization for fit results.

    Parameters
    ----------
    result : dict
        Fit result dictionary.
    config : dict
        Workflow configuration.
    runner : WorkflowRunner
        The workflow runner instance.
    verbose : bool
        Verbose logging flag.
    logger : CLILogger
        Logger instance.
    """
    try:
        # Re-load data for visualization
        data_config = config.get("data", {})
        xdata, ydata, sigma = runner.data_loader.load(
            data_config.get("input_file", ""),
            data_config,
        )

        # Prepare data dict for visualizer
        data_dict = {
            "xdata": xdata,
            "ydata": ydata,
            "sigma": sigma,
        }

        # Resolve model for visualization
        model_config = config.get("model", {})
        model_name = model_config.get("name", model_config.get("path", ""))
        model = runner.model_registry.get_model(model_name, model_config)

        # Generate visualization
        visualizer = FitVisualizer()
        output_paths = visualizer.generate(result, data_dict, model, config)

        if verbose and output_paths:
            logger.info(f"Generated {len(output_paths)} visualization files")
            for path in output_paths:
                logger.info(f"  - {path}")

    except Exception as e:
        # Visualization errors should not fail the fit
        if verbose:
            logger.warning(f"Visualization generation failed: {e}")
