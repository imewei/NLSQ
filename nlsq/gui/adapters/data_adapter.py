"""Data loading adapter for NLSQ GUI.

This module provides a wrapper around the CLI DataLoader class for use in the
Streamlit GUI. It supports loading data from files (via Streamlit UploadedFile),
parsing clipboard text, auto-detecting columns, validating data, and computing
statistics.

The adapter handles:
- All CLI-supported formats (CSV, ASCII, NPZ, HDF5)
- Clipboard paste with auto-delimiter detection
- 2D surface data mode detection
- NaN/Inf validation
- Basic data statistics computation
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.typing import NDArray

from nlsq.cli.data_loaders import EXTENSION_FORMAT_MAP, DataLoader

# =============================================================================
# Data Validation Result
# =============================================================================


@dataclass(slots=True)
class ValidationResult:
    """Result of data validation.

    Attributes
    ----------
    is_valid : bool
        Whether the data passed all validation checks.
    message : str
        Human-readable message describing the validation result.
    nan_count : int
        Number of NaN values found in the data.
    inf_count : int
        Number of Inf values found in the data.
    point_count : int
        Total number of data points.
    """

    is_valid: bool
    message: str
    nan_count: int = 0
    inf_count: int = 0
    point_count: int = 0


# =============================================================================
# Public API Functions
# =============================================================================


def load_from_file(
    file_path_or_uploaded: str | Path | BinaryIO,
    config: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Load data from a file or Streamlit UploadedFile.

    This function wraps the CLI DataLoader to handle both filesystem paths
    and Streamlit UploadedFile objects (which are file-like objects).

    Parameters
    ----------
    file_path_or_uploaded : str, Path, or file-like object
        Either a filesystem path (str/Path) or a Streamlit UploadedFile.
        For UploadedFile, the data is written to a temporary file first.
    config : dict
        Configuration dictionary for the DataLoader. See DataLoader.load()
        for expected keys:
        - "format": str - Format override or "auto" for detection
        - "columns": dict - Column assignments {"x": int/str, "y": int/str, ...}
        - "csv": dict - CSV-specific options
        - "ascii": dict - ASCII-specific options
        - "npz": dict - NPZ-specific options
        - "hdf5": dict - HDF5-specific options

    Returns
    -------
    tuple[ndarray, ndarray, ndarray | None]
        (xdata, ydata, sigma) tuple. For 1D data, xdata/ydata are 1D arrays.
        For 2D surface data, xdata is shape (2, n_points) and ydata is (n_points,).
        sigma may be None if not provided.

    Raises
    ------
    DataLoadError
        If the file cannot be loaded or parsed.

    Examples
    --------
    Load from filesystem path:
    >>> config = {"format": "csv", "columns": {"x": 0, "y": 1}}
    >>> xdata, ydata, sigma = load_from_file("data.csv", config)

    Load from Streamlit UploadedFile:
    >>> uploaded_file = st.file_uploader("Upload data")
    >>> if uploaded_file:
    ...     xdata, ydata, sigma = load_from_file(uploaded_file, config)
    """
    loader = DataLoader()

    # Check if this is a file-like object (Streamlit UploadedFile)
    if hasattr(file_path_or_uploaded, "read"):
        # It's a file-like object - write to temp file
        file_obj: BinaryIO = file_path_or_uploaded  # type: ignore[assignment]

        # Get the filename if available (Streamlit UploadedFile has .name)
        filename = getattr(file_obj, "name", "uploaded_data.txt")
        suffix = Path(filename).suffix

        # Detect format from extension if not specified
        if (
            config.get("format", "auto") == "auto"
            and suffix.lower() in EXTENSION_FORMAT_MAP
        ):
            config = {**config, "format": EXTENSION_FORMAT_MAP[suffix.lower()]}

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_obj.read())
            tmp_path = Path(tmp.name)

        try:
            # Reset file position for potential re-reads
            file_obj.seek(0)
            return loader.load(tmp_path, config)
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)
    else:
        # It's a path - load directly
        return loader.load(Path(file_path_or_uploaded), config)


def load_from_clipboard(
    text: str,
    config: dict[str, Any],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Parse tabular data from clipboard text.

    This function handles data pasted from spreadsheets (Excel, Google Sheets)
    or text editors. It auto-detects the delimiter (tab, comma, or whitespace).

    Parameters
    ----------
    text : str
        Raw text content from clipboard. Expected to be tabular data with
        rows separated by newlines and columns separated by tabs, commas,
        or whitespace.
    config : dict
        Configuration dictionary with:
        - "columns": dict - Column assignments {"x": int, "y": int, "sigma": int | None}
        - "has_header": bool - Whether first row is a header (default: False)
        - "delimiter": str | None - Force a specific delimiter (default: auto-detect)

    Returns
    -------
    tuple[ndarray, ndarray, ndarray | None]
        (xdata, ydata, sigma) tuple. sigma is None if not specified in columns.

    Raises
    ------
    ValueError
        If text cannot be parsed as tabular data.

    Examples
    --------
    Parse tab-separated data:
    >>> text = "1.0\\t2.0\\n2.0\\t4.0\\n3.0\\t6.0"
    >>> config = {"columns": {"x": 0, "y": 1}}
    >>> xdata, ydata, sigma = load_from_clipboard(text, config)

    Parse comma-separated with header:
    >>> text = "x,y\\n1,2\\n3,4"
    >>> config = {"columns": {"x": 0, "y": 1}, "has_header": True}
    >>> xdata, ydata, sigma = load_from_clipboard(text, config)
    """
    # Get configuration options
    columns_config = config.get("columns", {"x": 0, "y": 1, "sigma": None})
    has_header = config.get("has_header", False)
    forced_delimiter = config.get("delimiter")

    # Auto-detect delimiter if not specified
    if forced_delimiter is not None:
        delimiter = forced_delimiter
    else:
        delimiter = detect_delimiter(text)

    # Parse the text
    lines = text.strip().split("\n")

    if has_header and len(lines) > 0:
        # Skip header row
        lines = lines[1:]

    if not lines:
        raise ValueError("No data rows found in clipboard text")

    # Parse rows
    rows: list[list[float]] = []
    for line_num, line in enumerate(lines, start=1):
        stripped_line = line.strip()
        if not stripped_line:
            continue

        # Split by delimiter
        if delimiter is None:
            # Whitespace splitting
            parts = stripped_line.split()
        else:
            parts = stripped_line.split(delimiter)

        try:
            row = [float(p.strip()) for p in parts if p.strip()]
            if row:
                rows.append(row)
        except ValueError as e:
            raise ValueError(
                f"Cannot parse line {line_num} as numeric data: '{stripped_line}'"
            ) from e

    if not rows:
        raise ValueError("No valid numeric data found in clipboard text")

    # Convert to numpy array
    data = np.array(rows, dtype=np.float64)

    # Extract columns
    x_col = columns_config.get("x", 0)
    y_col = columns_config.get("y", 1)
    z_col = columns_config.get("z")
    sigma_col = columns_config.get("sigma")

    is_2d = z_col is not None

    try:
        if is_2d and z_col is not None:
            # 2D surface data
            x_coords = data[:, x_col]
            y_coords = data[:, y_col]
            xdata = np.vstack([x_coords, y_coords])
            ydata = data[:, z_col]
        else:
            # 1D curve data
            xdata = data[:, x_col]
            ydata = data[:, y_col]

        sigma = data[:, sigma_col] if sigma_col is not None else None

    except IndexError as e:
        num_cols = data.shape[1] if data.ndim > 1 else 1
        raise ValueError(
            f"Column index out of range. Data has {num_cols} columns."
        ) from e

    return xdata, ydata, sigma


def detect_delimiter(text: str) -> str | None:
    """Auto-detect delimiter in tabular text.

    Checks for the presence of common delimiters (tab, comma) and returns
    the most likely one. If neither is found, returns None to indicate
    whitespace splitting should be used.

    Parameters
    ----------
    text : str
        Text to analyze for delimiters.

    Returns
    -------
    str or None
        Detected delimiter character, or None for whitespace.
    """
    # Check first few lines
    lines = text.strip().split("\n")[:5]

    tab_count = 0
    comma_count = 0

    for line in lines:
        tab_count += line.count("\t")
        comma_count += line.count(",")

    # If tabs are present and more common than commas, use tab
    if tab_count > 0 and tab_count >= comma_count:
        return "\t"

    # If commas are present, use comma
    if comma_count > 0:
        return ","

    # Default to whitespace
    return None


def is_2d_mode(config: dict[str, Any]) -> bool:
    """Check if configuration specifies 2D surface data mode.

    2D mode is enabled when a 'z' column/key is specified in the
    configuration. In 2D mode, x and y are independent variables
    (coordinates) and z is the dependent variable.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    bool
        True if 2D surface data mode, False for 1D curve data.
    """
    loader = DataLoader()
    return loader.is_2d_data(config)


def detect_columns(data: NDArray[np.float64]) -> dict[str, Any]:
    """Auto-detect column assignments from data array.

    Provides sensible defaults for column assignments based on the
    number of columns in the data:
    - 2 columns: x=0, y=1
    - 3 columns: x=0, y=1, sigma=2
    - 4+ columns: x=0, y=1, z=2 (if 2D mode suspected), sigma=3

    Parameters
    ----------
    data : ndarray
        2D numpy array with shape (n_rows, n_cols).

    Returns
    -------
    dict
        Dictionary with detected column assignments:
        - "x_column": int - X column index
        - "y_column": int - Y column index
        - "z_column": int | None - Z column index (if 4+ columns)
        - "sigma_column": int | None - Sigma column index (if 3+ columns)
        - "num_columns": int - Total number of columns
        - "suggested_mode": str - "1d" or "2d"
    """
    if data.ndim == 1:
        # Single column - can't do anything useful
        return {
            "x_column": 0,
            "y_column": 0,
            "z_column": None,
            "sigma_column": None,
            "num_columns": 1,
            "suggested_mode": "1d",
        }

    num_cols = data.shape[1]

    result: dict[str, Any] = {
        "x_column": 0,
        "y_column": 1 if num_cols > 1 else 0,
        "z_column": None,
        "sigma_column": None,
        "num_columns": num_cols,
        "suggested_mode": "1d",
    }

    if num_cols == 3:
        # Likely x, y, sigma
        result["sigma_column"] = 2

    elif num_cols >= 4:
        # Could be 2D surface data: x, y, z, sigma
        result["z_column"] = 2
        result["sigma_column"] = 3
        result["suggested_mode"] = "2d"

    return result


def validate_data(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    sigma: NDArray[np.float64] | None = None,
    min_points: int = 2,
) -> ValidationResult:
    """Validate data for NaN/Inf values and minimum points.

    Checks that:
    - All data values are finite (not NaN or Inf)
    - There are at least min_points data points
    - Array lengths match

    Parameters
    ----------
    xdata : ndarray
        X data array. For 1D: shape (n,). For 2D: shape (2, n).
    ydata : ndarray
        Y data array. Shape (n,).
    sigma : ndarray or None
        Optional sigma/uncertainty array. Shape (n,).
    min_points : int
        Minimum number of data points required (default: 2).

    Returns
    -------
    ValidationResult
        Dataclass with validation results including is_valid flag,
        message, and counts of NaN/Inf values.

    Examples
    --------
    >>> xdata = np.array([1.0, 2.0, 3.0])
    >>> ydata = np.array([2.0, 4.0, 6.0])
    >>> result = validate_data(xdata, ydata)
    >>> result.is_valid
    True
    """
    # Get point count (handle 2D xdata)
    if xdata.ndim == 2:
        n_points = xdata.shape[1]
    else:
        n_points = len(xdata)

    # Count non-finite values
    x_nan = int(np.sum(np.isnan(xdata)))
    x_inf = int(np.sum(np.isinf(xdata)))
    y_nan = int(np.sum(np.isnan(ydata)))
    y_inf = int(np.sum(np.isinf(ydata)))

    sigma_nan = 0
    sigma_inf = 0
    if sigma is not None:
        sigma_nan = int(np.sum(np.isnan(sigma)))
        sigma_inf = int(np.sum(np.isinf(sigma)))

    total_nan = x_nan + y_nan + sigma_nan
    total_inf = x_inf + y_inf + sigma_inf

    # Check for insufficient points
    if n_points < min_points:
        return ValidationResult(
            is_valid=False,
            message=f"Insufficient data points: got {n_points}, minimum required is {min_points}",
            nan_count=total_nan,
            inf_count=total_inf,
            point_count=n_points,
        )

    # Check for array length mismatch
    if len(ydata) != n_points:
        return ValidationResult(
            is_valid=False,
            message=f"Array length mismatch: xdata has {n_points} points, ydata has {len(ydata)}",
            nan_count=total_nan,
            inf_count=total_inf,
            point_count=n_points,
        )

    if sigma is not None and len(sigma) != n_points:
        return ValidationResult(
            is_valid=False,
            message=f"Array length mismatch: sigma has {len(sigma)} points, expected {n_points}",
            nan_count=total_nan,
            inf_count=total_inf,
            point_count=n_points,
        )

    # Check for NaN/Inf
    if total_nan > 0 or total_inf > 0:
        details = []
        if total_nan > 0:
            details.append(f"{total_nan} NaN values")
        if total_inf > 0:
            details.append(f"{total_inf} Inf values")

        return ValidationResult(
            is_valid=False,
            message=f"Data contains non-finite values: {', '.join(details)}",
            nan_count=total_nan,
            inf_count=total_inf,
            point_count=n_points,
        )

    return ValidationResult(
        is_valid=True,
        message="Data validation passed",
        nan_count=0,
        inf_count=0,
        point_count=n_points,
    )


def compute_statistics(
    xdata: NDArray[np.float64],
    ydata: NDArray[np.float64],
    sigma: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """Compute basic statistics for loaded data.

    Parameters
    ----------
    xdata : ndarray
        X data array. For 1D: shape (n,). For 2D: shape (2, n).
    ydata : ndarray
        Y data array. Shape (n,).
    sigma : ndarray or None
        Optional sigma/uncertainty array.

    Returns
    -------
    dict
        Dictionary with statistics:
        - "point_count": int - Number of data points
        - "is_2d": bool - Whether this is 2D surface data
        - "x_min", "x_max": float - X range
        - "y_min", "y_max": float - Y range
        - "x_mean", "y_mean": float - Mean values
        - "x_std", "y_std": float - Standard deviations
        - "sigma_min", "sigma_max": float - Sigma range (if provided)
    """
    is_2d = xdata.ndim == 2

    if is_2d:
        n_points = xdata.shape[1]
        x_coords = xdata[0]
        y_coords = xdata[1] if xdata.shape[0] > 1 else xdata[0]
    else:
        n_points = len(xdata)
        x_coords = xdata
        y_coords = None

    stats: dict[str, Any] = {
        "point_count": n_points,
        "is_2d": is_2d,
    }

    # X statistics
    stats["x_min"] = float(np.min(x_coords))
    stats["x_max"] = float(np.max(x_coords))
    stats["x_mean"] = float(np.mean(x_coords))
    stats["x_std"] = float(np.std(x_coords))

    # Y statistics (for 1D this is the dependent variable, for 2D this is second coordinate)
    if is_2d and y_coords is not None:
        # For 2D, report y coordinate stats separately
        stats["y_coord_min"] = float(np.min(y_coords))
        stats["y_coord_max"] = float(np.max(y_coords))
        stats["y_coord_mean"] = float(np.mean(y_coords))
        stats["y_coord_std"] = float(np.std(y_coords))

    # ydata statistics (the dependent variable - z in 2D case)
    stats["y_min"] = float(np.min(ydata))
    stats["y_max"] = float(np.max(ydata))
    stats["y_mean"] = float(np.mean(ydata))
    stats["y_std"] = float(np.std(ydata))

    # Sigma statistics
    if sigma is not None:
        stats["sigma_min"] = float(np.min(sigma))
        stats["sigma_max"] = float(np.max(sigma))
        stats["sigma_mean"] = float(np.mean(sigma))
        stats["has_sigma"] = True
    else:
        stats["has_sigma"] = False

    return stats
