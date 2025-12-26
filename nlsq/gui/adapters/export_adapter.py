"""Export adapter for NLSQ GUI.

This module provides functions for exporting fit results in various formats
including JSON, CSV, ZIP session bundles, and standalone HTML plots.

The adapter wraps and extends the functionality from nlsq.cli.result_exporter
for use in the Streamlit GUI.

Functions
---------
export_json
    Export fit result as JSON string.
export_csv
    Export fit parameters as CSV string.
create_session_bundle
    Create a ZIP archive with all session artifacts.
export_plotly_html
    Export a Plotly figure as standalone HTML.
"""

import csv
import io
import json
import zipfile
from typing import Any

import numpy as np

from nlsq.cli.result_exporter import NumpyJSONEncoder


def export_json(
    result: Any,
    param_names: list[str] | None = None,
) -> str:
    """Export fit result as a JSON string.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result containing popt, pcov, statistics, etc.
    param_names : list[str] | None, optional
        Custom parameter names. If None, uses p0, p1, p2, etc.

    Returns
    -------
    str
        JSON string with parameters, covariance, statistics, and convergence info.

    Examples
    --------
    >>> json_str = export_json(result)
    >>> data = json.loads(json_str)
    >>> print(data["popt"])
    [2.0, 0.5, 1.0]
    """
    # Extract parameters
    popt = _to_list(getattr(result, "popt", []))
    pcov = _to_nested_list(getattr(result, "pcov", None))

    # Calculate uncertainties from covariance diagonal
    uncertainties = []
    if pcov is not None:
        pcov_arr = np.asarray(pcov)
        if pcov_arr.ndim == 2 and pcov_arr.shape[0] == pcov_arr.shape[1]:
            uncertainties = [float(np.sqrt(pcov_arr[i, i])) for i in range(len(popt))]

    # Extract statistics
    statistics = _extract_statistics(result)

    # Extract convergence info
    convergence = _extract_convergence(result)

    # Build export data
    export_data: dict[str, Any] = {
        "popt": popt,
        "pcov": pcov,
        "uncertainties": uncertainties,
        "statistics": statistics,
        "convergence": convergence,
    }

    # Add parameter names if provided
    if param_names is not None:
        export_data["parameter_names"] = param_names

    return json.dumps(export_data, cls=NumpyJSONEncoder, indent=2)


def export_csv(
    result: Any,
    param_names: list[str] | None = None,
) -> str:
    """Export fit parameters as a CSV string.

    Parameters
    ----------
    result : CurveFitResult
        The fitting result containing popt and pcov.
    param_names : list[str] | None, optional
        Custom parameter names. If None, uses p0, p1, p2, etc.

    Returns
    -------
    str
        CSV string with parameter name, value, and uncertainty columns.

    Examples
    --------
    >>> csv_str = export_csv(result, param_names=["amplitude", "decay"])
    >>> print(csv_str)
    name,value,uncertainty
    amplitude,2.0,0.1
    decay,0.5,0.032
    """
    # Extract parameters
    popt = _to_list(getattr(result, "popt", []))
    pcov = getattr(result, "pcov", None)

    # Calculate uncertainties
    uncertainties = []
    if pcov is not None:
        pcov_arr = np.asarray(pcov)
        if pcov_arr.ndim == 2 and pcov_arr.shape[0] == pcov_arr.shape[1]:
            uncertainties = [float(np.sqrt(pcov_arr[i, i])) for i in range(len(popt))]
        else:
            uncertainties = [float("nan")] * len(popt)
    else:
        uncertainties = [float("nan")] * len(popt)

    # Generate parameter names if not provided
    if param_names is None or len(param_names) < len(popt):
        param_names = [f"p{i}" for i in range(len(popt))]

    # Write CSV to string buffer
    output = io.StringIO()
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow(["name", "value", "uncertainty"])

    # Write parameter rows
    for name, value, uncertainty in zip(param_names, popt, uncertainties, strict=False):
        writer.writerow([name, value, uncertainty])

    return output.getvalue()


def create_session_bundle(
    state: Any,
    result: Any,
    figures: dict[str, Any],
) -> bytes:
    """Create a ZIP archive containing all session artifacts.

    The bundle includes:
    - data.csv: Data snapshot
    - config.yaml: Workflow configuration
    - results.json: Full fit results
    - ``*.html``: Interactive Plotly figures (if provided)

    Parameters
    ----------
    state : SessionState
        The GUI session state containing data and configuration.
    result : CurveFitResult
        The fitting result.
    figures : dict[str, Any]
        Dictionary of Plotly figures to include, keyed by name.

    Returns
    -------
    bytes
        ZIP file contents as bytes.

    Examples
    --------
    >>> zip_bytes = create_session_bundle(state, result, {"fit_plot": fig})
    >>> with open("session.zip", "wb") as f:
    ...     f.write(zip_bytes)
    """
    from nlsq.gui.adapters.config_adapter import export_yaml_config

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add data snapshot
        data_csv = _create_data_csv(state)
        zf.writestr("data.csv", data_csv)

        # Add configuration - pass state directly to export_yaml_config
        # which internally calls get_current_config
        try:
            config_yaml = export_yaml_config(state)
            zf.writestr("config.yaml", config_yaml)
        except Exception:
            # If config export fails, skip it
            pass

        # Add results JSON
        results_json = export_json(result)
        zf.writestr("results.json", results_json)

        # Add figures as HTML
        for name, fig in figures.items():
            if fig is not None:
                try:
                    html = export_plotly_html(fig)
                    zf.writestr(f"{name}.html", html)
                except Exception:
                    # Skip figures that fail to export
                    pass

    return zip_buffer.getvalue()


def export_plotly_html(figure: Any) -> str:
    """Export a Plotly figure as standalone HTML.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
        The Plotly figure to export.

    Returns
    -------
    str
        Standalone HTML string with embedded plotly.js.

    Examples
    --------
    >>> html = export_plotly_html(fig)
    >>> with open("plot.html", "w") as f:
    ...     f.write(html)
    """
    # Use Plotly's built-in HTML export with embedded JS
    return figure.to_html(
        include_plotlyjs=True,
        full_html=True,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _to_list(arr: Any) -> list:
    """Convert array-like to Python list."""
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _to_nested_list(arr: Any) -> list | None:
    """Convert 2D array-like to nested Python list."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if hasattr(arr, "tolist"):
        return arr.tolist()
    return list(arr)


def _extract_statistics(result: Any) -> dict[str, float]:
    """Extract statistics from fit result."""
    stats: dict[str, float] = {}

    for stat_name in ["r_squared", "rmse", "mae", "aic", "bic", "adj_r_squared"]:
        try:
            value = getattr(result, stat_name, None)
            if value is not None:
                stats[stat_name] = float(value)
        except (AttributeError, TypeError, ValueError):
            pass

    # Also try cost
    try:
        cost = getattr(result, "cost", None)
        if cost is not None:
            stats["cost"] = float(cost)
    except (AttributeError, TypeError, ValueError):
        pass

    return stats


def _extract_convergence(result: Any) -> dict[str, Any]:
    """Extract convergence info from fit result."""
    conv: dict[str, Any] = {}

    conv["success"] = bool(getattr(result, "success", False))
    conv["message"] = str(getattr(result, "message", ""))

    try:
        nfev = getattr(result, "nfev", None)
        if nfev is not None:
            conv["nfev"] = int(nfev)
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        cost = getattr(result, "cost", None)
        if cost is not None:
            conv["final_cost"] = float(cost)
    except (AttributeError, TypeError, ValueError):
        pass

    return conv


def _create_data_csv(state: Any) -> str:
    """Create CSV string from state data."""
    output = io.StringIO()
    writer = csv.writer(output)

    xdata = getattr(state, "xdata", None)
    ydata = getattr(state, "ydata", None)
    sigma = getattr(state, "sigma", None)

    if xdata is None or ydata is None:
        # Return empty CSV with header only
        writer.writerow(["x", "y"])
        return output.getvalue()

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)

    # Write header
    if sigma is not None:
        writer.writerow(["x", "y", "sigma"])
    else:
        writer.writerow(["x", "y"])

    # Write data rows
    n = min(len(xdata), len(ydata))
    for i in range(n):
        if sigma is not None:
            sigma_arr = np.asarray(sigma)
            if i < len(sigma_arr):
                writer.writerow([xdata[i], ydata[i], sigma_arr[i]])
            else:
                writer.writerow([xdata[i], ydata[i], ""])
        else:
            writer.writerow([xdata[i], ydata[i]])

    return output.getvalue()
