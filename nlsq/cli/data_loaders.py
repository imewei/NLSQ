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
- Both 1D (x, y, sigma) and 2D/3D surface (x, y, z, sigma) data

Example Usage
-------------
1D Data (curve fitting):
>>> from nlsq.cli.data_loaders import DataLoader
>>> loader = DataLoader()
>>> config = {
...     "format": "auto",
...     "columns": {"x": 0, "y": 1, "sigma": 2},
...     "ascii": {"comment_char": "#"},
... }
>>> xdata, ydata, sigma = loader.load("data.txt", config)

2D Data (surface fitting):
>>> config = {
...     "format": "auto",
...     "columns": {"x": 0, "y": 1, "z": 2, "sigma": 3},  # z present = 2D mode
... }
>>> xdata, ydata, sigma = loader.load("surface.txt", config)
>>> # xdata shape: (2, n_points) - stacked x, y coordinates
>>> # ydata shape: (n_points,) - z values (dependent variable)
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
    """Data loader supporting multiple file formats and data dimensions.

    Loads data from ASCII, CSV, NPZ, and HDF5 formats with automatic
    format detection and configurable column/key selection. Supports both
    1D curve fitting data (x, y, sigma) and 2D surface fitting data
    (x, y, z, sigma).

    Data Modes
    ----------
    1D Mode (default):
        - Columns: x, y, sigma (optional)
        - xdata: 1D array of shape (n_points,)
        - ydata: 1D array of shape (n_points,)
        - Model signature: ``f(x, *params)``

    2D Mode (when z column is specified):
        - Columns: x, y, z, sigma (optional)
        - xdata: 2D array of shape (2, n_points) where xdata[0]=x, xdata[1]=y
        - ydata: 1D array of shape (n_points,) containing z values
        - Model signature: ``f(xy, *params)`` where xy[0]=x, xy[1]=y

    Methods
    -------
    load(file_path, config)
        Load data from file and return (xdata, ydata, sigma) tuple.
    detect_format(file_path, config)
        Detect file format from extension or config.
    is_2d_data(config)
        Check if configuration specifies 2D surface data.

    Examples
    --------
    1D data (curve fitting):
    >>> loader = DataLoader()
    >>> config = {
    ...     "format": "csv",
    ...     "columns": {"x": "time", "y": "signal", "sigma": None},
    ...     "csv": {"header": True, "delimiter": ","},
    ... }
    >>> x, y, sigma = loader.load("experiment.csv", config)

    2D data (surface fitting):
    >>> config = {
    ...     "format": "csv",
    ...     "columns": {"x": "pos_x", "y": "pos_y", "z": "intensity", "sigma": "error"},
    ... }
    >>> xy, z, sigma = loader.load("surface.csv", config)
    >>> # xy.shape = (2, n_points), z.shape = (n_points,)
    """

    def is_2d_data(self, config: dict[str, Any]) -> bool:
        """Check if configuration specifies 2D surface data.

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
        # Check columns config (ASCII/CSV)
        columns_config = config.get("columns", {})
        if columns_config.get("z") is not None:
            return True

        # Check NPZ config
        npz_config = config.get("npz", {})
        if npz_config.get("z_key") is not None:
            return True

        # Check HDF5 config
        hdf5_config = config.get("hdf5", {})
        return hdf5_config.get("z_path") is not None

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
            - ASCII/CSV: "columns" dict with "x", "y", "z" (optional), "sigma" keys
            - NPZ: "npz" dict with "x_key", "y_key", "z_key" (optional), "sigma_key"
            - HDF5: "hdf5" dict with "x_path", "y_path", "z_path" (optional), "sigma_path"

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            For 1D data: (xdata, ydata, sigma) where xdata/ydata are 1D arrays.
            For 2D data: (xdata, ydata, sigma) where xdata is shape (2, n_points)
            with xdata[0]=x, xdata[1]=y and ydata is shape (n_points,) containing
            z values. sigma may be None if not provided.

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

        # Determine if this is 2D surface data
        is_2d = self.is_2d_data(config)

        # Load data based on format
        if file_format == "ascii":
            xdata, ydata, sigma = self._load_ascii(file_path, config, is_2d)
        elif file_format == "csv":
            xdata, ydata, sigma = self._load_csv(file_path, config, is_2d)
        elif file_format == "npz":
            xdata, ydata, sigma = self._load_npz(file_path, config, is_2d)
        elif file_format == "hdf5":
            xdata, ydata, sigma = self._load_hdf5(file_path, config, is_2d)
        else:
            raise DataLoadError(
                f"Unsupported format: {file_format}",
                file_path=file_path,
                file_format=file_format,
                suggestion=f"Supported formats are: {', '.join(sorted(SUPPORTED_FORMATS))}",
            )

        # Validate data
        self._validate_data(xdata, ydata, sigma, file_path, config, is_2d)

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
        is_2d: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load ASCII format file (.txt, .dat, .asc).

        Uses numpy.loadtxt() with configurable delimiter, comment char,
        and header skipping. Supports both 1D and 2D data.

        Parameters
        ----------
        file_path : Path
            Path to the ASCII file.
        config : dict
            Configuration with:
            - "columns": {"x": int, "y": int, "z": int | None, "sigma": int | None}
            - "ascii": {"delimiter": str | None, "comment_char": str, "skip_header": int}
        is_2d : bool
            If True, load as 2D surface data (x, y, z, sigma).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            For 1D: (xdata, ydata, sigma) where xdata/ydata are 1D.
            For 2D: (xdata, ydata, sigma) where xdata is (2, n), ydata is (n,).
        """
        ascii_config = config.get("ascii", {})
        columns_config = config.get(
            "columns", {"x": 0, "y": 1, "z": None, "sigma": None}
        )

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
        z_col = columns_config.get("z")
        sigma_col = columns_config.get("sigma")

        try:
            if is_2d and z_col is not None:
                # 2D surface data: x, y are coordinates, z is dependent variable
                x_coords = data[:, x_col].astype(np.float64)
                y_coords = data[:, y_col].astype(np.float64)
                # Stack x, y into shape (2, n_points) for curve_fit
                xdata = np.vstack([x_coords, y_coords])
                # z values become ydata
                ydata = data[:, z_col].astype(np.float64)
            else:
                # 1D curve data (original behavior)
                xdata = data[:, x_col].astype(np.float64)
                ydata = data[:, y_col].astype(np.float64)

            sigma = (
                data[:, sigma_col].astype(np.float64) if sigma_col is not None else None
            )
        except IndexError as e:
            cols_requested = [x_col, y_col]
            if is_2d and z_col is not None:
                cols_requested.append(z_col)
            if sigma_col is not None:
                cols_requested.append(sigma_col)
            max_col = max(cols_requested)
            raise DataLoadError(
                f"Column index {max_col} out of range (file has {data.shape[1]} columns)",
                file_path=file_path,
                file_format="ascii",
                context={
                    "num_columns": data.shape[1],
                    "requested_columns": cols_requested,
                    "is_2d": is_2d,
                },
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
        is_2d: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load CSV format file.

        Uses numpy.genfromtxt() with header detection, encoding,
        and missing value handling. Supports both 1D and 2D data.

        Parameters
        ----------
        file_path : Path
            Path to the CSV file.
        config : dict
            Configuration with:
            - "columns": {"x": str|int, "y": str|int, "z": str|int|None, "sigma": str|int|None}
            - "csv": {"delimiter": str, "header": bool, "encoding": str, "missing_values": list}
        is_2d : bool
            If True, load as 2D surface data (x, y, z, sigma).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            For 1D: (xdata, ydata, sigma) where xdata/ydata are 1D.
            For 2D: (xdata, ydata, sigma) where xdata is (2, n), ydata is (n,).
        """
        csv_config = config.get("csv", {})
        columns_config = config.get(
            "columns", {"x": 0, "y": 1, "z": None, "sigma": None}
        )

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
        z_col = columns_config.get("z")
        sigma_col = columns_config.get("sigma")

        # Convert column names to indices if needed
        x_idx = self._resolve_column_index(x_col, header_names, file_path)
        y_idx = self._resolve_column_index(y_col, header_names, file_path)
        z_idx = (
            self._resolve_column_index(z_col, header_names, file_path)
            if z_col is not None
            else None
        )
        sigma_idx = (
            self._resolve_column_index(sigma_col, header_names, file_path)
            if sigma_col is not None
            else None
        )

        try:
            if is_2d and z_idx is not None:
                # 2D surface data: x, y are coordinates, z is dependent variable
                x_coords = data[:, x_idx].astype(np.float64)
                y_coords = data[:, y_idx].astype(np.float64)
                # Stack x, y into shape (2, n_points) for curve_fit
                xdata = np.vstack([x_coords, y_coords])
                # z values become ydata
                ydata = data[:, z_idx].astype(np.float64)
            else:
                # 1D curve data (original behavior)
                xdata = data[:, x_idx].astype(np.float64)
                ydata = data[:, y_idx].astype(np.float64)

            sigma = (
                data[:, sigma_idx].astype(np.float64) if sigma_idx is not None else None
            )
        except IndexError as e:
            raise DataLoadError(
                "Column index out of range",
                file_path=file_path,
                file_format="csv",
                context={
                    "num_columns": data.shape[1] if data.ndim > 1 else 1,
                    "is_2d": is_2d,
                },
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
        is_2d: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load NPZ format file.

        Uses numpy.load() with configurable array keys. Supports both 1D and 2D data.

        Parameters
        ----------
        file_path : Path
            Path to the NPZ file.
        config : dict
            Configuration with:
            - "npz": {"x_key": str, "y_key": str, "z_key": str | None, "sigma_key": str | None}
        is_2d : bool
            If True, load as 2D surface data (x, y, z, sigma).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            For 1D: (xdata, ydata, sigma) where xdata/ydata are 1D.
            For 2D: (xdata, ydata, sigma) where xdata is (2, n), ydata is (n,).
        """
        npz_config = config.get("npz", {})

        x_key = npz_config.get("x_key", "x")
        y_key = npz_config.get("y_key", "y")
        z_key = npz_config.get("z_key")
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
                x_arr = data[x_key].astype(np.float64).flatten()

                # Load y data
                if y_key not in data:
                    raise DataLoadError(
                        f"Array key '{y_key}' not found in NPZ archive",
                        file_path=file_path,
                        file_format="npz",
                        context={"available_keys": available_keys},
                        suggestion=f"Available keys are: {', '.join(available_keys)}",
                    )
                y_arr = data[y_key].astype(np.float64).flatten()

                # Handle 2D vs 1D data
                if is_2d and z_key is not None:
                    # Load z data for 2D mode
                    if z_key not in data:
                        raise DataLoadError(
                            f"Array key '{z_key}' not found in NPZ archive",
                            file_path=file_path,
                            file_format="npz",
                            context={"available_keys": available_keys},
                            suggestion=f"Available keys are: {', '.join(available_keys)}",
                        )
                    z_arr = data[z_key].astype(np.float64).flatten()

                    # 2D surface data: x, y are coordinates, z is dependent variable
                    # Stack x, y into shape (2, n_points)
                    xdata = np.vstack([x_arr, y_arr])
                    ydata = z_arr
                else:
                    # 1D curve data (original behavior)
                    xdata = x_arr
                    ydata = y_arr

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
                    sigma = data[sigma_key].astype(np.float64).flatten()

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
        is_2d: bool = False,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
        """Load HDF5 format file.

        Uses h5py.File() with dataset path specification. Supports both 1D and 2D data.

        Parameters
        ----------
        file_path : Path
            Path to the HDF5 file.
        config : dict
            Configuration with:
            - "hdf5": {"x_path": str, "y_path": str, "z_path": str | None, "sigma_path": str | None}
        is_2d : bool
            If True, load as 2D surface data (x, y, z, sigma).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray | None]
            For 1D: (xdata, ydata, sigma) where xdata/ydata are 1D.
            For 2D: (xdata, ydata, sigma) where xdata is (2, n), ydata is (n,).
        """
        try:
            import h5py  # type: ignore[import-not-found]
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
        z_path = hdf5_config.get("z_path")
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
                x_arr = np.asarray(f[x_path], dtype=np.float64).flatten()

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
                y_arr = np.asarray(f[y_path], dtype=np.float64).flatten()

                # Handle 2D vs 1D data
                if is_2d and z_path is not None:
                    # Load z data for 2D mode
                    if z_path not in f:
                        available_paths = self._list_hdf5_datasets(f)
                        raise DataLoadError(
                            f"Dataset path '{z_path}' not found in HDF5 file",
                            file_path=file_path,
                            file_format="hdf5",
                            context={"available_paths": available_paths},
                            suggestion=f"Available dataset paths are: {', '.join(available_paths)}",
                        )
                    z_arr = np.asarray(f[z_path], dtype=np.float64).flatten()

                    # 2D surface data: x, y are coordinates, z is dependent variable
                    # Stack x, y into shape (2, n_points)
                    xdata = np.vstack([x_arr, y_arr])
                    ydata = z_arr
                else:
                    # 1D curve data (original behavior)
                    xdata = x_arr
                    ydata = y_arr

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
                    sigma = np.asarray(f[sigma_path], dtype=np.float64).flatten()

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
        is_2d: bool = False,
    ) -> None:
        """Validate loaded data.

        Parameters
        ----------
        xdata : ndarray
            X data array. For 1D: shape (n,). For 2D: shape (2, n).
        ydata : ndarray
            Y data array. Shape (n,) for both 1D and 2D.
        sigma : ndarray or None
            Sigma data array. Shape (n,) if provided.
        file_path : Path
            Path for error reporting.
        config : dict
            Configuration with validation settings.
        is_2d : bool
            If True, validate as 2D surface data.

        Raises
        ------
        DataLoadError
            If data fails validation.
        """
        validation_config = config.get("validation", {})
        require_finite = validation_config.get("require_finite", True)
        min_points = validation_config.get("min_points", 2)

        # Get number of data points (accounting for 2D xdata shape)
        if is_2d and xdata.ndim == 2:
            # xdata shape is (2, n_points)
            n_points = xdata.shape[1]
        else:
            # xdata shape is (n_points,)
            n_points = len(xdata)

        # Check array lengths match
        if len(ydata) != n_points:
            raise DataLoadError(
                f"Array length mismatch: xdata has {n_points} points, ydata has {len(ydata)}",
                file_path=file_path,
                context={
                    "x_points": n_points,
                    "y_points": len(ydata),
                    "is_2d": is_2d,
                },
                suggestion="Ensure all data arrays have the same number of points.",
            )

        if sigma is not None and len(sigma) != n_points:
            raise DataLoadError(
                f"Array length mismatch: sigma has {len(sigma)} points, expected {n_points}",
                file_path=file_path,
                context={
                    "sigma_points": len(sigma),
                    "expected_points": n_points,
                    "is_2d": is_2d,
                },
                suggestion="Ensure sigma data has the same number of points as other data.",
            )

        # Check minimum points
        if n_points < min_points:
            raise DataLoadError(
                f"Insufficient data points: got {n_points}, minimum required is {min_points}",
                file_path=file_path,
                context={"num_points": n_points, "min_points": min_points},
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
                        "is_2d": is_2d,
                    },
                    suggestion="Set validation.require_finite: false or clean the data to remove NaN/Inf values.",
                )
