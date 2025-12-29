"""Result exporter module for NLSQ CLI.

This module provides the ResultExporter class for exporting curve fitting
results in multiple formats (JSON, CSV, stdout).

Supported Export Formats
------------------------
- JSON: Full metadata with nested structure
- CSV: Flattened parameter name/value/uncertainty rows
- stdout: JSON format for piping to other tools

Example Usage
-------------
>>> from nlsq.cli.result_exporter import ResultExporter
>>>
>>> exporter = ResultExporter()
>>> result = {"popt": [1.0, 0.5], "pcov": [[0.01, 0], [0, 0.02]], ...}
>>> config = {"export": {"results_file": "output.json", "format": "json"}}
>>> exporter.export(result, config)
"""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from nlsq.cli.errors import CLIError


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and types."""

    def default(self, obj: Any) -> Any:
        """Convert numpy types to JSON-serializable types.

        Parameters
        ----------
        obj : Any
            Object to convert.

        Returns
        -------
        Any
            JSON-serializable object.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class ResultExporter:
    """Exporter for curve fitting results.

    Exports fit results to JSON, CSV, or stdout formats with full
    metadata including parameters, covariance, uncertainties, statistics,
    and convergence information.

    Attributes
    ----------
    None

    Methods
    -------
    export(result, config)
        Export fit result to configured format(s).

    Examples
    --------
    >>> exporter = ResultExporter()
    >>> result = {"popt": [1.0, 0.5], "pcov": [[0.01, 0], [0, 0.02]]}
    >>> config = {"export": {"results_file": "results.json", "format": "json"}}
    >>> exporter.export(result, config)
    """

    def export(self, result: dict[str, Any], config: dict[str, Any]) -> None:
        """Export fit result to configured format(s).

        Parameters
        ----------
        result : dict
            Fit result dictionary containing:
            - popt: Fitted parameters (ndarray or list)
            - pcov: Covariance matrix (ndarray or list)
            - success: bool indicating fit success
            - message: str with convergence message
            - nfev: Number of function evaluations
            - njev: Number of Jacobian evaluations (optional)
            - cost: Final cost value (optional)
            - fun: Residual vector (optional)
        config : dict
            Export configuration containing:
            - export.results_file: Output file path
            - export.format: "json" or "csv"
            - export.stdout: bool to output to stdout
            - export.skip_file_on_stdout: bool to skip file when stdout active
            - metadata: Workflow metadata (optional)
            - model: Model configuration (optional)

        Returns
        -------
        None

        Raises
        ------
        CLIError
            If export fails due to file or format issues.
        """
        export_config = config.get("export", {})
        output_format = export_config.get("format", "json").lower()
        stdout_mode = export_config.get("stdout", False)
        skip_file = export_config.get("skip_file_on_stdout", False)

        # Prepare export data
        export_data = self._prepare_export_data(result, config)

        # Export to stdout if requested
        if stdout_mode:
            self._export_stdout(export_data)

            # Skip file writing if configured
            if skip_file:
                return

        # Get output file path
        results_file = export_config.get("results_file")
        if results_file is None and not stdout_mode:
            raise CLIError(
                "No output file specified for export",
                suggestion="Set export.results_file in config or use export.stdout: true",
            )

        if results_file is not None:
            output_path = Path(results_file)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Export based on format
            if output_format == "json":
                self._export_json(export_data, output_path)
            elif output_format == "csv":
                self._export_csv(export_data, output_path)
            else:
                raise CLIError(
                    f"Unsupported export format: {output_format}",
                    context={"format": output_format},
                    suggestion="Supported formats are: json, csv",
                )

    def _prepare_export_data(
        self, result: dict[str, Any], config: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare export data with full metadata.

        Parameters
        ----------
        result : dict
            Fit result dictionary.
        config : dict
            Export configuration.

        Returns
        -------
        dict
            Prepared export data with all metadata.
        """
        # Extract parameters
        popt = result.get("popt", result.get("x", []))
        if isinstance(popt, np.ndarray):
            popt = popt.tolist()

        pcov = result.get("pcov", [])
        if isinstance(pcov, np.ndarray):
            pcov = pcov.tolist()

        # Calculate uncertainties from covariance diagonal
        uncertainties = []
        if pcov:
            pcov_arr = np.asarray(pcov)
            if pcov_arr.ndim == 2:
                uncertainties = np.sqrt(np.diag(pcov_arr)).tolist()

        # Calculate statistics
        statistics = self._calculate_statistics(result)

        # Extract convergence info
        convergence = {
            "iterations": result.get("nfev", 0),
            "function_evals": result.get("nfev", 0),
            "jacobian_evals": result.get("njev", 0),
            "status": "success" if result.get("success", False) else "failed",
            "message": result.get("message", ""),
        }

        if "cost" in result:
            convergence["final_cost"] = (
                float(result["cost"]) if result["cost"] is not None else None
            )

        # Extract metadata from config
        metadata_config = config.get("metadata", {})
        model_config = config.get("model", {})

        metadata = {
            "workflow_name": metadata_config.get("workflow_name", "unknown"),
            "dataset_id": metadata_config.get("dataset_id", "unknown"),
            "model_id": model_config.get(
                "name", model_config.get("function", "custom")
            ),
        }

        # Build export data
        export_data = {
            "popt": popt,
            "pcov": pcov,
            "uncertainties": uncertainties,
            "statistics": statistics,
            "convergence": convergence,
            "metadata": metadata,
        }

        # Include parameter names if available
        param_names = config.get("model", {}).get("parameter_names")
        if param_names:
            export_data["parameter_names"] = param_names

        return export_data

    def _calculate_statistics(self, result: dict[str, Any]) -> dict[str, Any]:
        """Calculate fit statistics from result.

        Parameters
        ----------
        result : dict
            Fit result dictionary.

        Returns
        -------
        dict
            Statistics dictionary with r_squared, rmse, chi_squared, etc.
        """
        statistics: dict[str, Any] = {}

        # Get residuals if available
        fun = result.get("fun")
        if fun is not None:
            residuals = np.asarray(fun)

            # RMSE
            statistics["rmse"] = float(np.sqrt(np.mean(residuals**2)))

            # Chi-squared (sum of squared residuals)
            statistics["chi_squared"] = float(np.sum(residuals**2))

            # R-squared if ydata is available
            ydata = result.get("ydata")
            if ydata is not None:
                y = np.asarray(ydata)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot > 0:
                    statistics["r_squared"] = float(1 - ss_res / ss_tot)

        # Copy any existing statistics
        if "r_squared" in result:
            statistics["r_squared"] = float(result["r_squared"])
        if "rmse" in result:
            statistics["rmse"] = float(result["rmse"])
        if "chi_squared" in result:
            statistics["chi_squared"] = float(result["chi_squared"])

        # Cost is related to sum of squared residuals
        if "cost" in result and result["cost"] is not None:
            # cost = 0.5 * sum(residuals**2) for least squares
            statistics["cost"] = float(result["cost"])

        return statistics

    def _export_json(self, data: dict[str, Any], output_path: Path) -> None:
        """Export data to JSON format.

        Parameters
        ----------
        data : dict
            Export data dictionary.
        output_path : Path
            Output file path.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, cls=NumpyJSONEncoder, indent=2)
        except (OSError, TypeError) as e:
            raise CLIError(
                f"Failed to write JSON file: {e}",
                context={"output_path": str(output_path)},
            ) from e

    def _export_csv(self, data: dict[str, Any], output_path: Path) -> None:
        """Export data to CSV format with flattened rows.

        Creates a CSV with parameter name/value/uncertainty rows,
        plus statistics as separate rows.

        Parameters
        ----------
        data : dict
            Export data dictionary.
        output_path : Path
            Output file path.
        """
        popt = data.get("popt", [])
        uncertainties = data.get("uncertainties", [])
        statistics = data.get("statistics", {})
        param_names = data.get("parameter_names", [])

        # Ensure uncertainties list matches popt length
        while len(uncertainties) < len(popt):
            uncertainties.append(float("nan"))

        # Generate parameter names if not provided
        if len(param_names) < len(popt):
            param_names = [f"p{i}" for i in range(len(popt))]

        try:
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

                # Write header
                writer.writerow(["name", "value", "uncertainty", "type"])

                # Write parameter rows
                for i, (name, value, uncertainty) in enumerate(
                    zip(param_names, popt, uncertainties, strict=False)
                ):
                    writer.writerow([name, value, uncertainty, "parameter"])

                # Write statistics rows
                for stat_name, stat_value in statistics.items():
                    writer.writerow([stat_name, stat_value, "", "statistic"])

                # Write metadata
                metadata = data.get("metadata", {})
                for meta_name, meta_value in metadata.items():
                    writer.writerow([meta_name, meta_value, "", "metadata"])

                # Write convergence info
                convergence = data.get("convergence", {})
                for conv_name, conv_value in convergence.items():
                    writer.writerow([conv_name, conv_value, "", "convergence"])

        except OSError as e:
            raise CLIError(
                f"Failed to write CSV file: {e}",
                context={"output_path": str(output_path)},
            ) from e

    def _export_stdout(self, data: dict[str, Any]) -> None:
        """Export data to stdout in JSON format.

        Parameters
        ----------
        data : dict
            Export data dictionary.
        """
        json_str = json.dumps(data, cls=NumpyJSONEncoder, indent=2)
        print(json_str)
