"""Tests for NLSQ CLI data loader module.

This module tests:
- ASCII format loading (.txt, .dat, .asc) with configurable delimiter/comments
- CSV format loading with header detection and missing values
- NPZ format loading with configurable array keys
- HDF5 format loading with dataset path specification
- Automatic format detection from file extension
- Sigma/uncertainty column loading (optional)
- Data validation (require_finite, NaN/Inf rejection)
- DataLoadError for malformed data

Test Categories
---------------
1. ASCII format loading with various delimiters and comments
2. CSV format loading with headers and missing value handling
3. NPZ format loading with configurable keys
4. HDF5 format loading with hierarchical paths
5. Automatic format detection from extensions
6. Sigma/uncertainty column handling
7. Data validation (NaN/Inf, min_points)
8. DataLoadError for malformed data
"""

from pathlib import Path

import numpy as np
import pytest

from nlsq.cli.data_loaders import DataLoader
from nlsq.cli.errors import DataLoadError

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Test 1: ASCII Format Loading (.txt, .dat, .asc)
# =============================================================================


class TestASCIIFormatLoading:
    """Tests for ASCII format loading with various configurations."""

    def test_ascii_load_basic_whitespace_delimiter(self):
        """Test loading ASCII file with whitespace delimiter."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {
                "delimiter": None,  # Whitespace
                "comment_char": "#",
                "skip_header": 0,
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(FIXTURES_DIR / "sample_ascii.txt", config)

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert len(sigma) == 10
        np.testing.assert_allclose(xdata[0], 0.0)
        np.testing.assert_allclose(ydata[0], 1.0)
        np.testing.assert_allclose(sigma[0], 0.05)

    def test_ascii_load_tab_delimiter(self):
        """Test loading ASCII file with tab delimiter (.dat extension)."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {
                "delimiter": "\t",
                "comment_char": "#",
                "skip_header": 0,
            },
        }
        loader = DataLoader()
        xdata, ydata, _sigma = loader.load(
            FIXTURES_DIR / "sample_tab_delimited.dat", config
        )

        assert len(xdata) == 5
        np.testing.assert_allclose(xdata[0], 0.0)
        np.testing.assert_allclose(ydata[0], 1.0)

    def test_ascii_load_without_sigma(self):
        """Test loading ASCII file without sigma column."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "ascii": {
                "delimiter": None,
                "comment_char": "#",
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(FIXTURES_DIR / "sample_ascii.txt", config)

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert sigma is None


# =============================================================================
# Test 2: CSV Format Loading
# =============================================================================


class TestCSVFormatLoading:
    """Tests for CSV format loading with header detection."""

    def test_csv_load_with_header(self):
        """Test loading CSV file with header row."""
        config = {
            "format": "csv",
            "columns": {"x": "x", "y": "y", "sigma": "sigma"},
            "csv": {
                "delimiter": ",",
                "header": True,
                "encoding": "utf-8",
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(FIXTURES_DIR / "sample_data.csv", config)

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert len(sigma) == 10
        np.testing.assert_allclose(xdata[0], 0.0)
        np.testing.assert_allclose(ydata[0], 1.0)

    def test_csv_load_by_column_index(self):
        """Test loading CSV file using column indices."""
        config = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "csv": {
                "delimiter": ",",
                "header": True,
                "encoding": "utf-8",
            },
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_data.csv", config)

        assert len(xdata) == 10
        np.testing.assert_allclose(xdata[0], 0.0)

    def test_csv_load_with_missing_values(self):
        """Test loading CSV file with missing values (NA, NaN, empty)."""
        config = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "csv": {
                "delimiter": ",",
                "header": True,
                "missing_values": ["", "NA", "null", "NaN"],
            },
            "validation": {"require_finite": False},  # Allow NaN for this test
        }
        loader = DataLoader()
        xdata, ydata, _sigma = loader.load(
            FIXTURES_DIR / "sample_data_missing.csv", config
        )

        # Should load successfully even with missing values
        assert len(xdata) == 6
        # y[3] should be NaN (was "NaN" in the file)
        assert np.isnan(ydata[3])


# =============================================================================
# Test 3: NPZ Format Loading
# =============================================================================


class TestNPZFormatLoading:
    """Tests for NPZ format loading with configurable keys."""

    def test_npz_load_default_keys(self):
        """Test loading NPZ file with default x, y, sigma keys."""
        config = {
            "format": "npz",
            "npz": {
                "x_key": "x",
                "y_key": "y",
                "sigma_key": "sigma",
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(FIXTURES_DIR / "sample_data.npz", config)

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert len(sigma) == 10
        np.testing.assert_allclose(xdata[0], 0.0)

    def test_npz_load_custom_keys(self):
        """Test loading NPZ file with custom array keys."""
        config = {
            "format": "npz",
            "npz": {
                "x_key": "xdata",
                "y_key": "ydata",
                "sigma_key": "uncertainty",
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(
            FIXTURES_DIR / "sample_custom_keys.npz", config
        )

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert len(sigma) == 10

    def test_npz_load_without_sigma_key(self):
        """Test loading NPZ file without sigma key."""
        config = {
            "format": "npz",
            "npz": {
                "x_key": "x",
                "y_key": "y",
                "sigma_key": None,
            },
        }
        loader = DataLoader()
        xdata, _ydata, sigma = loader.load(FIXTURES_DIR / "sample_data.npz", config)

        assert len(xdata) == 10
        assert sigma is None

    def test_npz_load_missing_key_raises_error(self):
        """Test loading NPZ file with missing key raises DataLoadError."""
        config = {
            "format": "npz",
            "npz": {
                "x_key": "nonexistent_key",
                "y_key": "y",
                "sigma_key": None,
            },
        }
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "sample_data.npz", config)
        assert "nonexistent_key" in str(exc_info.value)


# =============================================================================
# Test 4: HDF5 Format Loading
# =============================================================================


class TestHDF5FormatLoading:
    """Tests for HDF5 format loading with dataset paths."""

    def test_hdf5_load_hierarchical_path(self):
        """Test loading HDF5 file with hierarchical dataset paths."""
        config = {
            "format": "hdf5",
            "hdf5": {
                "x_path": "/data/x",
                "y_path": "/data/y",
                "sigma_path": "/data/sigma",
            },
        }
        loader = DataLoader()
        xdata, ydata, sigma = loader.load(FIXTURES_DIR / "sample_data.h5", config)

        assert len(xdata) == 10
        assert len(ydata) == 10
        assert len(sigma) == 10
        np.testing.assert_allclose(xdata[0], 0.0)

    def test_hdf5_load_flat_structure(self):
        """Test loading HDF5 file with flat dataset structure."""
        config = {
            "format": "hdf5",
            "hdf5": {
                "x_path": "/x",
                "y_path": "/y",
                "sigma_path": None,
            },
        }
        loader = DataLoader()
        xdata, _ydata, sigma = loader.load(FIXTURES_DIR / "sample_flat.hdf5", config)

        assert len(xdata) == 10
        assert sigma is None

    def test_hdf5_load_missing_path_raises_error(self):
        """Test loading HDF5 file with missing dataset path raises DataLoadError."""
        config = {
            "format": "hdf5",
            "hdf5": {
                "x_path": "/nonexistent/path",
                "y_path": "/data/y",
                "sigma_path": None,
            },
        }
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "sample_data.h5", config)
        assert "nonexistent" in str(exc_info.value).lower()


# =============================================================================
# Test 5: Automatic Format Detection from Extension
# =============================================================================


class TestFormatAutoDetection:
    """Tests for automatic format detection from file extension."""

    def test_auto_detect_txt_extension(self):
        """Test format auto-detection for .txt extension."""
        config = {
            "format": "auto",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {"comment_char": "#"},
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_ascii.txt", config)

        assert len(xdata) == 10

    def test_auto_detect_dat_extension(self):
        """Test format auto-detection for .dat extension."""
        config = {
            "format": "auto",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {"comment_char": "#"},
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(
            FIXTURES_DIR / "sample_tab_delimited.dat", config
        )

        assert len(xdata) == 5

    def test_auto_detect_csv_extension(self):
        """Test format auto-detection for .csv extension."""
        config = {
            "format": "auto",
            "columns": {"x": "x", "y": "y", "sigma": "sigma"},
            "csv": {"header": True},
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_data.csv", config)

        assert len(xdata) == 10

    def test_auto_detect_npz_extension(self):
        """Test format auto-detection for .npz extension."""
        config = {
            "format": "auto",
            "npz": {"x_key": "x", "y_key": "y", "sigma_key": "sigma"},
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_data.npz", config)

        assert len(xdata) == 10

    def test_auto_detect_h5_extension(self):
        """Test format auto-detection for .h5 extension."""
        config = {
            "format": "auto",
            "hdf5": {
                "x_path": "/data/x",
                "y_path": "/data/y",
                "sigma_path": "/data/sigma",
            },
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_data.h5", config)

        assert len(xdata) == 10

    def test_auto_detect_hdf5_extension(self):
        """Test format auto-detection for .hdf5 extension."""
        config = {
            "format": "auto",
            "hdf5": {"x_path": "/x", "y_path": "/y", "sigma_path": None},
        }
        loader = DataLoader()
        xdata, _ydata, _sigma = loader.load(FIXTURES_DIR / "sample_flat.hdf5", config)

        assert len(xdata) == 10


# =============================================================================
# Test 6: Data Validation (require_finite, NaN/Inf rejection)
# =============================================================================


class TestDataValidation:
    """Tests for data validation functionality."""

    def test_require_finite_rejects_nan(self):
        """Test require_finite validation rejects NaN values."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {"comment_char": "#"},
            "validation": {"require_finite": True},
        }
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "sample_data_with_nan.txt", config)
        error_msg = str(exc_info.value).lower()
        assert "nan" in error_msg or "finite" in error_msg

    def test_require_finite_disabled_allows_nan(self):
        """Test require_finite=False allows NaN values."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {"comment_char": "#"},
            "validation": {"require_finite": False},
        }
        loader = DataLoader()
        # Should not raise
        xdata, _ydata, _sigma = loader.load(
            FIXTURES_DIR / "sample_data_with_nan.txt", config
        )
        assert len(xdata) == 4

    def test_min_points_validation(self):
        """Test min_points validation."""
        config = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "ascii": {"comment_char": "#"},
            "validation": {"require_finite": False, "min_points": 100},
        }
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "sample_ascii.txt", config)
        assert (
            "min" in str(exc_info.value).lower()
            or "points" in str(exc_info.value).lower()
        )


# =============================================================================
# Test 7: DataLoadError for Malformed Data
# =============================================================================


class TestDataLoadErrors:
    """Tests for DataLoadError handling."""

    def test_file_not_found_raises_error(self):
        """Test loading nonexistent file raises DataLoadError."""
        config = {"format": "ascii", "columns": {"x": 0, "y": 1}}
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "nonexistent_file.txt", config)
        assert (
            "not found" in str(exc_info.value).lower()
            or "exist" in str(exc_info.value).lower()
        )

    def test_unknown_format_raises_error(self):
        """Test unknown format raises DataLoadError."""
        config = {"format": "unknown_format"}
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "sample_ascii.txt", config)
        assert "format" in str(exc_info.value).lower()

    def test_unknown_extension_for_auto_raises_error(self):
        """Test unknown extension with format='auto' raises DataLoadError."""
        config = {"format": "auto"}
        loader = DataLoader()
        # Create a temporary file with unknown extension
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test data")
            temp_path = f.name
        try:
            with pytest.raises(DataLoadError) as exc_info:
                loader.load(Path(temp_path), config)
            assert (
                "extension" in str(exc_info.value).lower()
                or "format" in str(exc_info.value).lower()
            )
        finally:
            Path(temp_path).unlink()

    def test_data_load_error_contains_file_path(self):
        """Test DataLoadError contains the file path in context."""
        config = {"format": "ascii", "columns": {"x": 0, "y": 1}}
        loader = DataLoader()
        with pytest.raises(DataLoadError) as exc_info:
            loader.load(FIXTURES_DIR / "nonexistent_file.txt", config)
        # Verify file path is in the error context or message
        error = exc_info.value
        assert "nonexistent_file" in str(error) or "file_path" in error.context
