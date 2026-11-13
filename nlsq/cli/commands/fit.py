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

import contextlib
import sys
import threading
from pathlib import Path
from typing import Any

import yaml

from nlsq.cli.errors import (
    CLIError,
    ConfigError,
    beside_workflow_file,
    cfg_get,
    setup_logging,
)
from nlsq.cli.result_exporter import ResultExporter, _dumps
from nlsq.cli.visualization import (
    FitVisualizer,
    check_filename_prefix,
    normalize_formats,
)
from nlsq.cli.workflow_runner import WorkflowRunner, validate_config_sections

_STDOUT_REDIRECT_LOCK = threading.Lock()


class _StdoutExporter(ResultExporter):
    """Exporter whose JSON goes to the real stdout while the workflow's own
    prints (hybrid banners, etc.) are diverted to stderr."""

    def __init__(self, stream: Any) -> None:
        super().__init__()
        self._stream = stream

    def _export_stdout(self, data: dict[str, Any]) -> None:
        print(_dumps(data), file=self._stream)


def run_fit(
    workflow_path: str,
    output_override: str | None = None,
    style_override: str | None = None,
    stdout: bool = False,
    verbose: bool = False,
    default_figure_prefix: str | None = None,
) -> dict[str, Any] | None:
    """Execute a single curve fit from a YAML workflow configuration.

    Parameters
    ----------
    workflow_path : str
        Path to the workflow YAML configuration file.
    output_override : str, optional
        Override the export.results_file path.
    style_override : str, optional
        Override the visualization style preset.
    stdout : bool
        If True, output results as JSON to stdout.
    verbose : bool
        If True, enable verbose logging.
    default_figure_prefix : str, optional
        Figure filename prefix used when ``visualization.filename_prefix`` is
        not set (``run_batch`` passes the workflow file stem so parallel
        workflows do not overwrite each other's figures).

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
        with open(workflow_file, encoding="utf-8-sig") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Failed to parse YAML: {e}",
            config_file=workflow_path,
            suggestion="Check YAML syntax for errors",
        ) from e
    except (UnicodeDecodeError, OSError) as e:
        raise ConfigError(
            f"Cannot read workflow file: {e}",
            config_file=workflow_path,
            suggestion="Provide a UTF-8 encoded YAML text file",
        ) from e

    if not isinstance(config, dict):
        raise ConfigError(
            "Invalid YAML: configuration must be a dictionary",
            config_file=workflow_path,
            suggestion="Ensure the YAML file contains a valid configuration object",
        )

    validate_config_sections(config, workflow_path)

    # Apply output override (`or {}`: a bare YAML key parses as None)
    if output_override is not None:
        if not output_override.strip() or Path(output_override).is_dir():
            raise ConfigError(
                f"Invalid --output path {output_override!r}: give a results file name",
                config_file=workflow_path,
            )
        config["export"] = config.get("export") or {}
        config["export"]["results_file"] = output_override

    # Apply style override
    if style_override is not None:
        config["visualization"] = config.get("visualization") or {}
        config["visualization"]["style"] = style_override
        # Ensure enabled if style is explicitly set
        config["visualization"]["enabled"] = True

    # Apply stdout mode
    if stdout:
        config["export"] = config.get("export") or {}
        config["export"]["stdout"] = True
        config["export"]["skip_file_on_stdout"] = True

    if not config.get("export"):
        raise ConfigError(
            "No output file specified for export",
            config_file=workflow_path,
            suggestion="Add export.results_file, or pass --output / --stdout",
        )

    export_config = config.get("export") or {}
    results_file = export_config.get("results_file")
    json_stdout = bool(cfg_get(export_config, "stdout", False))
    skip_file = json_stdout and bool(
        cfg_get(export_config, "skip_file_on_stdout", False)
    )
    if results_file is not None and not skip_file and Path(results_file).is_dir():
        raise ConfigError(
            f"export.results_file {str(results_file)!r} is a directory",
            config_file=workflow_path,
            suggestion="Give a results file name",
        )
    if (
        results_file
        and not (
            cfg_get(export_config, "stdout", False)
            and cfg_get(export_config, "skip_file_on_stdout", False)
        )
        and Path(results_file).resolve() == workflow_file.resolve()
    ):
        raise CLIError("Results file must not overwrite the workflow configuration")

    # Log start
    workflow_name = (config.get("metadata") or {}).get("workflow_name", "unnamed")
    if verbose:
        logger.info(f"Starting workflow: {workflow_name}")
        logger.info(f"Configuration file: {workflow_path}")

    if default_figure_prefix is not None and isinstance(
        config.get("visualization"), dict
    ):
        # A bare `filename_prefix:` is present-but-None; treat it as missing.
        if config["visualization"].get("filename_prefix") is None:
            config["visualization"]["filename_prefix"] = default_figure_prefix

    # A relative data.input_file next to the YAML wins over a same-named file in
    # the CWD (matches the GUI; `nlsq batch a/w.yaml b/w.yaml` would otherwise
    # fit ./data.csv for every workflow). Fall back to the CWD-relative path.
    _data_cfg = config.get("data")
    if isinstance(_data_cfg, dict) and isinstance(_data_cfg.get("input_file"), str):
        _beside = beside_workflow_file(workflow_path, _data_cfg["input_file"])
        if _beside is not None:
            _data_cfg["input_file"] = str(_beside)
    # Same lookup order for a custom model file (read-only input).
    _model_cfg = config.get("model")
    if isinstance(_model_cfg, dict) and isinstance(_model_cfg.get("path"), str):
        _beside = beside_workflow_file(workflow_path, _model_cfg["path"])
        if _beside is not None:
            _model_cfg["path"] = str(_beside)

    vis_config = config.get("visualization") or {}
    if "formats" in vis_config:
        vis_config["formats"] = normalize_formats(vis_config["formats"])
    if cfg_get(vis_config, "enabled", False):
        check_filename_prefix(cfg_get(vis_config, "filename_prefix", "fit"))
        sources = {workflow_file.resolve()}
        input_file = (config.get("data") or {}).get("input_file")
        model = config.get("model") or {}
        if input_file:
            sources.add(Path(input_file).resolve())
        if model.get("type") == "custom" and model.get("path"):
            sources.add(Path(model["path"]).resolve())
        if results_file and not (
            cfg_get(export_config, "stdout", False)
            and cfg_get(export_config, "skip_file_on_stdout", False)
        ):
            sources.add(Path(results_file).resolve())
        output_dir = Path(cfg_get(vis_config, "output_dir", "figures"))
        # POSIX resolve() raises on NUL; Windows does not, so reject it here
        if "\x00" in str(output_dir):
            raise CLIError("Visualization output_dir contains a NUL character")
        prefix = cfg_get(vis_config, "filename_prefix", "fit")
        suffixes = ["combined"]
        hist = vis_config.get("histogram")  # may be a bare bool (`histogram: true`)
        if isinstance(hist, dict) and cfg_get(hist, "enabled", False):
            suffixes.append("histogram")
        for suffix in suffixes:
            for fmt in cfg_get(vis_config, "formats", ["pdf", "png"]):
                if (output_dir / f"{prefix}_{suffix}.{fmt}").resolve() in sources:
                    raise CLIError(
                        "Visualization must not overwrite workflow input files"
                    )

    # Execute workflow. In JSON-stdout mode everything except the JSON document
    # (banners, progress prints) goes to stderr so stdout stays machine-readable.
    runner = WorkflowRunner()
    runner.model_registry.allowed_model_dirs.append(
        Path(workflow_path).resolve().parent
    )
    overwrites = (
        bool(results_file) and not skip_file and Path(str(results_file)).is_file()
    )
    # redirect_stdout swaps a process-global: serialise it so concurrent batch
    # workers cannot capture each other's redirect and leave sys.stdout on stderr.
    with _STDOUT_REDIRECT_LOCK if json_stdout else contextlib.nullcontext():
        real_stdout = sys.stdout  # captured before the redirect below
        if json_stdout:
            runner.result_exporter = _StdoutExporter(real_stdout)
        with (
            contextlib.redirect_stdout(sys.stderr)
            if json_stdout
            else contextlib.nullcontext()
        ):
            result = runner.run(config)

            # Generate visualization if enabled
            vis_config = config.get("visualization") or {}
            if cfg_get(vis_config, "enabled", False):
                _generate_visualization(result, config, runner, verbose, logger)

    if overwrites and result.get("success") is False:
        print(
            f"Warning: overwrote existing results file {results_file} "
            "with a non-converged fit",
            file=sys.stderr,
        )

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
        # Use data already loaded by runner.run() — avoids a second file read
        # and prevents failures if the source file was moved after fitting.
        if runner.last_data is not None:
            xdata, ydata, sigma = runner.last_data
        else:
            data_config = config.get("data") or {}
            xdata, ydata, sigma = runner.data_loader.load(
                cfg_get(data_config, "input_file", ""),
                data_config,
            )

        # Prepare data dict for visualizer
        data_dict = {
            "xdata": xdata,
            "ydata": ydata,
            "sigma": sigma,
        }

        # Resolve model for visualization
        model_config = config.get("model") or {}
        model_name = cfg_get(model_config, "name", cfg_get(model_config, "path", ""))
        model = runner.model_registry.get_model(model_name, model_config)

        # Generate visualization
        visualizer = FitVisualizer()
        # The runner result carries no `statistics`; the annotation needs them.
        vis_result = {
            **result,
            "statistics": ResultExporter()._calculate_statistics(result),
        }
        output_paths = visualizer.generate(vis_result, data_dict, model, config)

        if verbose and output_paths:
            logger.info(f"Generated {len(output_paths)} visualization files")
            for path in output_paths:
                logger.info(f"  - {path}")

    except ValueError as e:
        # A ValueError from the visualizer signals a structural mismatch
        # (e.g. multi-dimensional xdata) that will never succeed on retry,
        # unlike a transient plotting glitch - log it at error level so it
        # isn't mistaken for a one-off warning.
        logger.error(f"Visualization generation failed: {e}")
    except Exception as e:
        # Visualization errors should not fail the fit, but must still be
        # visible: gating this behind `verbose` meant a bad `export.formats`
        # entry (e.g. an unsupported extension) silently produced zero plots
        # on a default run with no indication anything went wrong.
        logger.warning(f"Visualization generation failed: {e}")
