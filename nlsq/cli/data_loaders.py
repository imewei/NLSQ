"""Data loading module for NLSQ CLI.

This module provides a DataLoader class for loading data from multiple formats:
- ASCII text files (.txt, .dat, .asc)
- CSV files (.csv)
- NumPy compressed archives (.npz)
- HDF5 files (.h5, .hdf5)

The module supports:
- Automatic format detection from file extension
- Configurable column/key selection
- Optional sigma/uncertainty loading
- Data validation (NaN/Inf rejection, minimum points)

Example Usage
-------------
>>> from nlsq.cli.data_loaders import DataLoader
>>> loader = DataLoader()
>>> config = {
...     "format": "auto",
...     "columns": {"x": 0, "y": 1, "sigma": 2},
...     "ascii": {"comment_char": "#"},
... }
>>> xdata, ydata, sigma = loader.load("data.txt", config)
"""

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from nlsq.cli.errors import DataLoadError


# =============================================================================
# Format Extension Mappings
# =============================================================================

EXTENSION_FORMAT_MAP: dict[str, str] = {
    ".txt": "ascii",
    ".dat": "ascii",
    ".asc": "ascii",
    ".csv": "csv",
    ".npz": "npz",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
}

SUPPORTED_FORMATS = {"ascii", "csv", "npz", "hdf5"}


# =============================================================================
# DataLoader Class
# =============================================================================


class DataLoader:
    """Data loader supporting multiple file formats.

    Loads data from ASCII, CSV, NPZ, and HDF5 formats with automatic
    format detection and configurable column/key selection.

    Attributes
    ----------
    None

    Methods
    -------
    load(file_path, config)
        Load data from file and return (xdata, ydata, sigma) tuple.
    detect_format(file_path, config)
        Detect file format from extension or config.

    Examples
    --------
    >>> loader = DataLoader()
    >>> config = {
    ...     "format": "csv",
    ...     "columns": {"x": "time", "y": "signal", "sigma": None},
    ...     "csv": {"header": True, "delimiter": ","},
    ... }
    >>> x, y, sigma = loader.load("experiment.csv", config)
    """

    def load(
        self,
        file_path: str | Path,
        config: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load data from file.

        Parameters
        ----------
        file_path : str or Path
            Path to the data file.
        config : dict
            Configuration dictionary containing format-specific options.
            Required keys depend on format:
            - All formats: "format" (or "auto" for detection)
            - ASCII/CSV: "columns" dict with "x", "y", "sigma" keys
            - NPZ: "npz" dict with "x_key", "y_key", "sigma_key"
            - HDF5: "hdf5" dict with "x_path", "y_path", "sigma_path"

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Tuple of (xdata, ydata, sigma) where sigma may be None.

        Raises
        ------
        DataLoadError
            If file cannot be loaded or data is invalid.
        """
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            raise DataLoadError(
                f"Data file not found: {file_path}",
                file_path=file_path,
                suggestion="Check that the file path is correct and the file exists.",
            )

        # Detect or get format
        file_format = self.detect_format(file_path, config)

        # Load data based on format
        if file_format == "ascii":
            xdata, ydata, sigma = self._load_ascii(file_path, config)
        elif file_format == "csv":
            xdata, ydata, sigma = self._load_csv(file_path, config)
        elif file_format == "npz":
            xdata, ydata, sigma = self._load_npz(file_path, config)
        elif file_format == "hdf5":
            xdata, ydata, sigma = self._load_hdf5(file_path, config)
        else:
            raise DataLoadError(
                f"Unsupported format: {file_format}",
                file_path=file_path,
                file_format=file_format,
                suggestion=f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}",
            )

        # Validate data
        self._validate_data(xdata, ydata, sigma, file_path, config)

        return xdata, ydata, sigma

    def detect_format(self, file_path: Path, config: dict[str, Any]) -> str:
        """Detect file format from extension or config.

        Parameters
        ----------
        file_path : Path
            Path to the data file.
        config : dict
            Configuration dictionary with optional "format" key.

        Returns
        -------
        str
            Detected format string ("ascii", "csv", "npz", "hdf5").

        Raises
        ------
        DataLoadError
            If format cannot be determined.
        """
        config_format = config.get("format", "auto")

        if config_format != "auto":
            if config_format not in SUPPORTED_FORMATS:
                raise DataLoadError(
                    f"Unknown format: {config_format}",
                    file_path=file_path,
                    file_format=config_format,
                    suggestion=f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}",
                )
            return config_format

        # Auto-detect from extension
        suffix = file_path.suffix.lower()
        if suffix not in EXTENSION_FORMAT_MAP:
            raise DataLoadError(
                f"Cannot auto-detect format for extension '{suffix}'",
                file_path=file_path,
                context={"extension": suffix},
                suggestion=f"Supported extensions are: {', '.join(sorted(EXTENSION_FORMAT_MAP.keys()))}",
            )

        return EXTENSION_FORMAT_MAP[suffix]

    # =========================================================================
    # ASCII Format Loader
    # =========================================================================

    def _load_ascii(
        self,
        file_path: Path,
        config: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load ASCII format file (.txt, .dat, .asc).

        Uses numpy.loadtxt() with configurable delimiter, comment char,
        and header skipping.

        Parameters
        ----------
        file_path : Path
            Path to the ASCII file.
        config : dict
            Configuration with:
            - "columns": {"x": int, "y": int, "sigma": int | None}
            - "ascii": {"delimiter": str | None, "comment_char": str, "skip_header": int}

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Loaded data arrays.
        """
        ascii_config = config.get("ascii", {})
        columns_config = config.get("columns", {"x": 0, "y": 1, "sigma": None})

        delimiter = ascii_config.get("delimiter", None)  # None = whitespace
        comment_char = ascii_config.get("comment_char", "#")
        skip_header = ascii_config.get("skip_header", 0)
        skip_footer = ascii_config.get("skip_footer", 0)
        usecols = ascii_config.get("usecols", None)
        dtype = ascii_config.get("dtype", "float64")

        try:
            data = np.loadtxt(
                file_path,
                delimiter=delimiter,
                comments=comment_char,
                skiprows=skip_header,
                dtype=dtype,
                ndmin=2,
            )

            # Handle skip_footer
            if skip_footer > 0:
                data = data[:-skip_footer]

        except (ValueError, OSError) as e:
            raise DataLoadError(
                f"Failed to parse ASCII file: {e}",
                file_path=file_path,
                file_format="ascii",
                suggestion="Check that the file format matches the configuration "
                "(delimiter, comment character, etc.)",
            ) from e

        # Extract columns
        x_col = columns_config.get("x", 0)
        y_col = columns_config.get("y", 1)
        sigma_col = columns_config.get("sigma")

        try:
            xdata = data[:, x_col].astype(np.float64)
            ydata = data[:, y_col].astype(np.float64)
            sigma = data[:, sigma_col].astype(np.float64) if sigma_col is not None else None
        except IndexError as e:
            max_col = max(x_col, y_col, sigma_col or 0)
            raise DataLoadError(
                f"Column index {max_col} out of range (file has {data.shape[1]} columns)",
                file_path=file_path,
                file_format="ascii",
                context={"num_columns": data.shape[1], "requested_columns": [x_col, y_col, sigma_col]},
                suggestion="Check that column indices are correct (0-based indexing).",
            ) from e

        return xdata, ydata, sigma

    # =========================================================================
    # CSV Format Loader
    # =========================================================================

    def _load_csv(
        self,
        file_path: Path,
        config: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load CSV format file.

        Uses numpy.genfromtxt() with header detection, encoding,
        and missing value handling.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.
        config : dict
            Configuration with:
            - "columns": {"x": str | int, "y": str | int, "sigma": str | int | None}
            - "csv": {"delimiter": str, "header": bool, "encoding": str, "missing_values": list}

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Loaded data arrays.
        """
        csv_config = config.get("csv", {})
        columns_config = config.get("columns", {"x": 0, "y": 1, "sigma": None})

        delimiter = csv_config.get("delimiter", ",")
        has_header = csv_config.get("header", True)
        encoding = csv_config.get("encoding", "utf-8")
        missing_values = csv_config.get("missing_values", ["", "NA", "null", "NaN"])
        skip_header = csv_config.get("skip_header", 0)

        # Read header names if present
        header_names: list[str] | None = None
        if has_header:
            with open(file_path, encoding=encoding) as f:
                # Skip additional header lines if specified
                for _ in range(skip_header):
                    f.readline()
                header_line = f.readline().strip()
                header_names = [col.strip() for col in header_line.split(delimiter)]

        try:
            # Calculate skip rows: additional skip_header + 1 for header row
            total_skip = skip_header + (1 if has_header else 0)

            data = np.genfromtxt(
                file_path,
                delimiter=delimiter,
                skip_header=total_skip,
                encoding=encoding,
                missing_values=missing_values,
                filling_values=np.nan,
                dtype=np.float64,
            )

            # Ensure 2D array
            if data.ndim == 1:
                data = data.reshape(-1, 1)

        except (ValueError, OSError) as e:
            raise DataLoadError(
                f"Failed to parse CSV file: {e}",
                file_path=file_path,
                file_format="csv",
                suggestion="Check that the file format matches the configuration "
                "(delimiter, encoding, etc.)",
            ) from e

        # Extract columns by name or index
        x_col = columns_config.get("x", 0)
        y_col = columns_config.get("y", 1)
        sigma_col = columns_config.get("sigma")

        # Convert column names to indices if needed
        x_idx = self._resolve_column_index(x_col, header_names, file_path)
        y_idx = self._resolve_column_index(y_col, header_names, file_path)
        sigma_idx = (
            self._resolve_column_index(sigma_col, header_names, file_path)
            if sigma_col is not None
            else None
        )

        try:
            xdata = data[:, x_idx].astype(np.float64)
            ydata = data[:, y_idx].astype(np.float64)
            sigma = data[:, sigma_idx].astype(np.float64) if sigma_idx is not None else None
        except IndexError as e:
            raise DataLoadError(
                f"Column index out of range",
                file_path=file_path,
                file_format="csv",
                context={"num_columns": data.shape[1] if data.ndim > 1 else 1},
                suggestion="Check that column indices or names are correct.",
            ) from e

        return xdata, ydata, sigma

    def _resolve_column_index(
        self,
        col: str | int,
        header_names: list[str] | None,
        file_path: Path,
    ) -> int:
        """Resolve column name to index.

        Parameters
        ----------
        col : str or int
            Column name or index.
        header_names : list of str, optional
            Header names from CSV file.
        file_path : Path
            Path for error reporting.

        Returns
        -------
        int
            Column index.

        Raises
        ------
        DataLoadError
            If column name not found.
        """
        if isinstance(col, int):
            return col

        if header_names is None:
            raise DataLoadError(
                f"Cannot use column name '{col}' without header row",
                file_path=file_path,
                file_format="csv",
                suggestion="Set csv.header: true or use column indices instead.",
            )

        try:
            return header_names.index(col)
        except ValueError:
            raise DataLoadError(
                f"Column '{col}' not found in CSV header",
                file_path=file_path,
                file_format="csv",
                context={"available_columns": header_names},
                suggestion=f"Available columns are: {', '.join(header_names)}",
            ) from None

    # =========================================================================
    # NPZ Format Loader
    # =========================================================================

    def _load_npz(
        self,
        file_path: Path,
        config: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load NPZ format file.

        Uses numpy.load() with configurable array keys.

        Parameters
        ----------
        file_path : Path
            Path to the NPZ file.
        config : dict
            Configuration with:
            - "npz": {"x_key": str, "y_key": str, "sigma_key": str | None}

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Loaded data arrays.
        """
        npz_config = config.get("npz", {})

        x_key = npz_config.get("x_key", "x")
        y_key = npz_config.get("y_key", "y")
        sigma_key = npz_config.get("sigma_key")

        try:
            with np.load(file_path) as data:
                available_keys = list(data.keys())

                # Load x data
                if x_key not in data:
                    raise DataLoadError(
                        f"Array key '{x_key}' not found in NPZ archive",
                        file_path=file_path,
                        file_format="npz",
                        context={"available_keys": available_keys},
                        suggestion=f"Available keys are: {', '.join(available_keys)}",
                    )
                xdata = data[x_key].astype(np.float64)

                # Load y data
                if y_key not in data:
                    raise DataLoadError(
                        f"Array key '{y_key}' not found in NPZ archive",
                        file_path=file_path,
                        file_format="npz",
                        context={"available_keys": available_keys},
                        suggestion=f"Available keys are: {', '.join(available_keys)}",
                    )
                ydata = data[y_key].astype(np.float64)

                # Load sigma data if key specified
                sigma: NDArray[np.float64] | None = None
                if sigma_key is not None:
                    if sigma_key not in data:
                        raise DataLoadError(
                            f"Array key '{sigma_key}' not found in NPZ archive",
                            file_path=file_path,
                            file_format="npz",
                            context={"available_keys": available_keys},
                            suggestion=f"Available keys are: {', '.join(available_keys)}",
                        )
                    sigma = data[sigma_key].astype(np.float64)

        except (ValueError, OSError) as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(
                f"Failed to load NPZ file: {e}",
                file_path=file_path,
                file_format="npz",
                suggestion="Check that the file is a valid NumPy NPZ archive.",
            ) from e

        return xdata, ydata, sigma

    # =========================================================================
    # HDF5 Format Loader
    # =========================================================================

    def _load_hdf5(
        self,
        file_path: Path,
        config: dict[str, Any],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load HDF5 format file.

        Uses h5py.File() with dataset path specification.

        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.
        config : dict
            Configuration with:
            - "hdf5": {"x_path": str, "y_path": str, "sigma_path": str | None}

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            Loaded data arrays.
        """
        try:
            import h5py
        except ImportError:
            raise DataLoadError(
                "h5py package required for HDF5 file loading",
                file_path=file_path,
                file_format="hdf5",
                suggestion="Install h5py: pip install h5py",
            ) from None

        hdf5_config = config.get("hdf5", {})

        x_path = hdf5_config.get("x_path", "/data/x")
        y_path = hdf5_config.get("y_path", "/data/y")
        sigma_path = hdf5_config.get("sigma_path")

        try:
            with h5py.File(file_path, "r") as f:
                # Load x data
                if x_path not in f:
                    available_paths = self._list_hdf5_datasets(f)
                    raise DataLoadError(
                        f"Dataset path '{x_path}' not found in HDF5 file",
                        file_path=file_path,
                        file_format="hdf5",
                        context={"available_paths": available_paths},
                        suggestion=f"Available dataset paths are: {', '.join(available_paths)}",
                    )
                xdata = np.asarray(f[x_path], dtype=np.float64)

                # Load y data
                if y_path not in f:
                    available_paths = self._list_hdf5_datasets(f)
                    raise DataLoadError(
                        f"Dataset path '{y_path}' not found in HDF5 file",
                        file_path=file_path,
                        file_format="hdf5",
                        context={"available_paths": available_paths},
                        suggestion=f"Available dataset paths are: {', '.join(available_paths)}",
                    )
                ydata = np.asarray(f[y_path], dtype=np.float64)

                # Load sigma data if path specified
                sigma: NDArray[np.float64] | None = None
                if sigma_path is not None:
                    if sigma_path not in f:
                        available_paths = self._list_hdf5_datasets(f)
                        raise DataLoadError(
                            f"Dataset path '{sigma_path}' not found in HDF5 file",
                            file_path=file_path,
                            file_format="hdf5",
                            context={"available_paths": available_paths},
                            suggestion=f"Available dataset paths are: {', '.join(available_paths)}",
                        )
                    sigma = np.asarray(f[sigma_path], dtype=np.float64)

        except (ValueError, OSError) as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(
                f"Failed to load HDF5 file: {e}",
                file_path=file_path,
                file_format="hdf5",
                suggestion="Check that the file is a valid HDF5 file.",
            ) from e

        return xdata, ydata, sigma

    def _list_hdf5_datasets(self, h5_file: Any) -> list[str]:
        """List all dataset paths in an HDF5 file.

        Parameters
        ----------
        h5_file : h5py.File
            Open HDF5 file handle.

        Returns
        -------
        list of str
            List of dataset paths.
        """
        import h5py

        datasets: list[str] = []

        def visitor(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                datasets.append(f"/{name}")

        h5_file.visititems(visitor)
        return datasets

    # =========================================================================
    # Data Validation
    # =========================================================================

    def _validate_data(
        self,
        xdata: NDArray[np.float64],
        ydata: NDArray[np.float64],
        sigma: NDArray[np.float64] | None,
        file_path: Path,
        config: dict[str, Any],
    ) -> None:
        """Validate loaded data.

        Parameters
        ----------
        xdata : ndarray
            X data array.
        ydata : ndarray
            Y data array.
        sigma : ndarray or None
            Sigma data array.
        file_path : Path
            Path for error reporting.
        config : dict
            Configuration with validation settings.

        Raises
        ------
        DataLoadError
            If data fails validation.
        """
        validation_config = config.get("validation", {})
        require_finite = validation_config.get("require_finite", True)
        min_points = validation_config.get("min_points", 2)

        # Check array lengths match
        if len(xdata) != len(ydata):
            raise DataLoadError(
                f"Array length mismatch: xdata has {len(xdata)} points, ydata has {len(ydata)}",
                file_path=file_path,
                context={"x_length": len(xdata), "y_length": len(ydata)},
                suggestion="Ensure x and y data have the same number of points.",
            )

        if sigma is not None and len(sigma) != len(xdata):
            raise DataLoadError(
                f"Array length mismatch: sigma has {len(sigma)} points, xdata has {len(xdata)}",
                file_path=file_path,
                context={"sigma_length": len(sigma), "x_length": len(xdata)},
                suggestion="Ensure sigma data has the same number of points as x and y.",
            )

        # Check minimum points
        if len(xdata) < min_points:
            raise DataLoadError(
                f"Insufficient data points: got {len(xdata)}, minimum required is {min_points}",
                file_path=file_path,
                context={"num_points": len(xdata), "min_points": min_points},
                suggestion="Provide more data points or reduce validation.min_points.",
            )

        # Check for NaN/Inf values
        if require_finite:
            x_nan = np.sum(~np.isfinite(xdata))
            y_nan = np.sum(~np.isfinite(ydata))
            sigma_nan = np.sum(~np.isfinite(sigma)) if sigma is not None else 0

            total_nonfinite = x_nan + y_nan + sigma_nan

            if total_nonfinite > 0:
                details = []
                if x_nan > 0:
                    details.append(f"xdata: {x_nan}")
                if y_nan > 0:
                    details.append(f"ydata: {y_nan}")
                if sigma_nan > 0:
                    details.append(f"sigma: {sigma_nan}")

                raise DataLoadError(
                    f"Data contains {total_nonfinite} non-finite values (NaN or Inf)",
                    file_path=file_path,
                    context={
                        "x_nonfinite": int(x_nan),
                        "y_nonfinite": int(y_nan),
                        "sigma_nonfinite": int(sigma_nan),
                    },
                    suggestion="Set validation.require_finite: false or clean the data to remove NaN/Inf values.",
                )
