"""Tests for NLSQ GUI data loading adapter.

This module tests the data loading adapter for the Streamlit GUI,
including file loading, clipboard parsing, column detection, and data validation.
"""

import io
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nlsq.gui.adapters.data_adapter import (
    compute_statistics,
    detect_columns,
    detect_delimiter,
    is_2d_mode,
    load_from_clipboard,
    load_from_file,
    validate_data,
)


class TestCSVLoading:
    """Tests for CSV file loading."""

    def test_load_csv_basic(self, tmp_path: Path) -> None:
        """Test loading a basic CSV file."""
        csv_content = "x,y\n1.0,2.0\n2.0,4.0\n3.0,6.0\n"
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, ydata, sigma = load_from_file(str(csv_file), config)

        assert len(xdata) == 3
        assert len(ydata) == 3
        assert sigma is None
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(ydata, [2.0, 4.0, 6.0])

    def test_load_csv_with_sigma(self, tmp_path: Path) -> None:
        """Test loading CSV with sigma column."""
        csv_content = "x,y,error\n1.0,2.0,0.1\n2.0,4.0,0.2\n3.0,6.0,0.3\n"
        csv_file = tmp_path / "test_sigma.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "sigma": 2},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, ydata, sigma = load_from_file(str(csv_file), config)

        assert len(xdata) == 3
        assert len(ydata) == 3
        assert sigma is not None
        assert len(sigma) == 3
        np.testing.assert_array_almost_equal(sigma, [0.1, 0.2, 0.3])

    def test_load_csv_with_column_names(self, tmp_path: Path) -> None:
        """Test loading CSV using column names instead of indices."""
        csv_content = "time,signal,noise\n0.0,1.0,0.05\n1.0,2.0,0.10\n"
        csv_file = tmp_path / "test_named.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": "time", "y": "signal", "sigma": "noise"},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, ydata, _sigma = load_from_file(str(csv_file), config)

        assert len(xdata) == 2
        np.testing.assert_array_almost_equal(xdata, [0.0, 1.0])
        np.testing.assert_array_almost_equal(ydata, [1.0, 2.0])


class TestASCIILoading:
    """Tests for ASCII format loading."""

    def test_load_ascii_whitespace_delimited(self, tmp_path: Path) -> None:
        """Test loading ASCII file with whitespace delimiter."""
        ascii_content = "# Comment line\n1.0  2.0\n2.0  4.0\n3.0  6.0\n"
        ascii_file = tmp_path / "test.dat"
        ascii_file.write_text(ascii_content)

        config: dict[str, Any] = {
            "format": "ascii",
            "columns": {"x": 0, "y": 1, "sigma": None},
            "ascii": {"comment_char": "#", "delimiter": None},
        }

        xdata, ydata, _sigma = load_from_file(str(ascii_file), config)

        assert len(xdata) == 3
        assert len(ydata) == 3
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(ydata, [2.0, 4.0, 6.0])

    def test_load_ascii_with_multiple_columns(self, tmp_path: Path) -> None:
        """Test loading ASCII file with multiple columns."""
        ascii_content = "1.0 2.0 3.0 0.1\n2.0 4.0 6.0 0.2\n"
        ascii_file = tmp_path / "test_multi.txt"
        ascii_file.write_text(ascii_content)

        config: dict[str, Any] = {
            "format": "ascii",
            "columns": {"x": 0, "y": 2, "sigma": 3},
            "ascii": {"comment_char": "#"},
        }

        xdata, ydata, sigma = load_from_file(str(ascii_file), config)

        assert len(xdata) == 2
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0])
        np.testing.assert_array_almost_equal(ydata, [3.0, 6.0])
        assert sigma is not None
        np.testing.assert_array_almost_equal(sigma, [0.1, 0.2])


class TestClipboardParsing:
    """Tests for clipboard text parsing."""

    def test_parse_tab_separated_data(self) -> None:
        """Test parsing tab-separated clipboard data."""
        clipboard_text = "1.0\t2.0\n2.0\t4.0\n3.0\t6.0\n"

        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "sigma": None},
        }

        xdata, ydata, _sigma = load_from_clipboard(clipboard_text, config)

        assert len(xdata) == 3
        assert len(ydata) == 3
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(ydata, [2.0, 4.0, 6.0])

    def test_parse_comma_separated_data(self) -> None:
        """Test parsing comma-separated clipboard data."""
        clipboard_text = "1.0,2.0\n2.0,4.0\n3.0,6.0\n"

        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "sigma": None},
        }

        xdata, ydata, _sigma = load_from_clipboard(clipboard_text, config)

        assert len(xdata) == 3
        assert len(ydata) == 3
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0, 3.0])

    def test_parse_clipboard_with_header(self) -> None:
        """Test parsing clipboard data that has a header row."""
        clipboard_text = "x\ty\n1.0\t2.0\n2.0\t4.0\n"

        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "sigma": None},
            "has_header": True,
        }

        xdata, _ydata, _sigma = load_from_clipboard(clipboard_text, config)

        assert len(xdata) == 2
        np.testing.assert_array_almost_equal(xdata, [1.0, 2.0])

    def test_detect_delimiter_tab(self) -> None:
        """Test auto-detection of tab delimiter."""
        text = "1.0\t2.0\n3.0\t4.0\n"
        delimiter = detect_delimiter(text)
        assert delimiter == "\t"

    def test_detect_delimiter_comma(self) -> None:
        """Test auto-detection of comma delimiter."""
        text = "1.0,2.0\n3.0,4.0\n"
        delimiter = detect_delimiter(text)
        assert delimiter == ","

    def test_detect_delimiter_whitespace(self) -> None:
        """Test auto-detection of whitespace delimiter."""
        text = "1.0 2.0\n3.0 4.0\n"
        delimiter = detect_delimiter(text)
        assert delimiter is None  # None means whitespace


class TestSurfaceDataMode:
    """Tests for 2D surface data mode detection."""

    def test_detect_2d_mode_with_z_column(self) -> None:
        """Test 2D mode detection when z column is specified."""
        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "z": 2, "sigma": None},
        }
        assert is_2d_mode(config) is True

    def test_detect_1d_mode_without_z(self) -> None:
        """Test 1D mode detection when no z column."""
        config: dict[str, Any] = {
            "columns": {"x": 0, "y": 1, "sigma": None},
        }
        assert is_2d_mode(config) is False

    def test_detect_2d_mode_npz_config(self) -> None:
        """Test 2D mode detection with NPZ config."""
        config: dict[str, Any] = {
            "npz": {"x_key": "x", "y_key": "y", "z_key": "z"},
        }
        assert is_2d_mode(config) is True

    def test_load_2d_surface_data(self, tmp_path: Path) -> None:
        """Test loading 2D surface data from file."""
        csv_content = "x,y,z\n0.0,0.0,1.0\n1.0,0.0,2.0\n0.0,1.0,3.0\n1.0,1.0,4.0\n"
        csv_file = tmp_path / "surface.csv"
        csv_file.write_text(csv_content)

        config: dict[str, Any] = {
            "format": "csv",
            "columns": {"x": 0, "y": 1, "z": 2, "sigma": None},
            "csv": {"header": True, "delimiter": ","},
        }

        xdata, ydata, _sigma = load_from_file(str(csv_file), config)

        # For 2D data, xdata should have shape (2, n_points)
        assert xdata.shape == (2, 4)
        # ydata contains the z values
        assert ydata.shape == (4,)
        np.testing.assert_array_almost_equal(ydata, [1.0, 2.0, 3.0, 4.0])


class TestDataValidation:
    """Tests for NaN/Inf validation."""

    def test_validate_finite_data_passes(self) -> None:
        """Test that finite data passes validation."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])

        result = validate_data(xdata, ydata, sigma=None)

        assert result.is_valid is True
        assert result.nan_count == 0
        assert result.inf_count == 0

    def test_validate_data_with_nan_fails(self) -> None:
        """Test that data with NaN fails validation."""
        xdata = np.array([1.0, np.nan, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])

        result = validate_data(xdata, ydata, sigma=None)

        assert result.is_valid is False
        assert result.nan_count == 1
        assert "NaN" in result.message

    def test_validate_data_with_inf_fails(self) -> None:
        """Test that data with Inf fails validation."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, np.inf, 6.0])

        result = validate_data(xdata, ydata, sigma=None)

        assert result.is_valid is False
        assert result.inf_count == 1
        assert "Inf" in result.message

    def test_validate_sigma_with_nan(self) -> None:
        """Test validation of sigma array with NaN."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        sigma = np.array([0.1, np.nan, 0.3])

        result = validate_data(xdata, ydata, sigma=sigma)

        assert result.is_valid is False
        assert result.nan_count == 1

    def test_validate_minimum_points(self) -> None:
        """Test validation for minimum number of data points."""
        xdata = np.array([1.0])
        ydata = np.array([2.0])

        result = validate_data(xdata, ydata, sigma=None, min_points=2)

        assert result.is_valid is False
        assert (
            "minimum" in result.message.lower()
            or "insufficient" in result.message.lower()
        )


class TestColumnDetection:
    """Tests for automatic column detection."""

    def test_detect_columns_simple_2d_array(self) -> None:
        """Test column detection on simple 2D array."""
        data = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

        result = detect_columns(data)

        assert result["x_column"] == 0
        assert result["y_column"] == 1
        assert result["num_columns"] == 2

    def test_detect_columns_with_three_columns(self) -> None:
        """Test column detection suggests sigma for third column."""
        data = np.array([[1.0, 2.0, 0.1], [2.0, 4.0, 0.2], [3.0, 6.0, 0.3]])

        result = detect_columns(data)

        assert result["x_column"] == 0
        assert result["y_column"] == 1
        assert result["sigma_column"] == 2
        assert result["num_columns"] == 3


class TestComputeStatistics:
    """Tests for data statistics computation."""

    def test_compute_statistics_basic(self) -> None:
        """Test basic statistics computation."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

        stats = compute_statistics(xdata, ydata)

        assert stats["point_count"] == 5
        assert stats["x_min"] == 1.0
        assert stats["x_max"] == 5.0
        assert stats["y_min"] == 2.0
        assert stats["y_max"] == 10.0
        assert stats["x_mean"] == 3.0
        assert stats["y_mean"] == 6.0

    def test_compute_statistics_with_std(self) -> None:
        """Test standard deviation computation."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([1.0, 2.0, 3.0])

        stats = compute_statistics(xdata, ydata)

        assert "x_std" in stats
        assert "y_std" in stats
        assert stats["x_std"] > 0
        assert stats["y_std"] > 0

    def test_compute_statistics_2d_data(self) -> None:
        """Test statistics for 2D surface data."""
        # xdata shape (2, n_points)
        xdata = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        ydata = np.array([1.0, 2.0, 3.0, 4.0])

        stats = compute_statistics(xdata, ydata)

        assert stats["point_count"] == 4
        assert stats["is_2d"] is True
