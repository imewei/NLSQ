"""Tests for NLSQ GUI data loading page components.

This module tests the data loading page UI components including column selector,
data preview, statistics display, and 1D/2D mode handling.
"""

from typing import Any

import numpy as np
import pytest

from nlsq.gui.components.column_selector import (
    compute_data_preview_stats,
    create_column_assignments,
    get_column_color,
    get_role_display_name,
    validate_column_selections,
)


class TestColumnSelectorColors:
    """Tests for column color assignment."""

    def test_get_column_color_x(self) -> None:
        """Test that X column gets a distinct color."""
        color = get_column_color("x")
        assert color is not None
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_get_column_color_y(self) -> None:
        """Test that Y column gets a distinct color."""
        color = get_column_color("y")
        assert color is not None
        assert isinstance(color, str)
        assert color.startswith("#")

    def test_get_column_color_z(self) -> None:
        """Test that Z column gets a distinct color."""
        color = get_column_color("z")
        assert color is not None
        assert isinstance(color, str)

    def test_get_column_color_sigma(self) -> None:
        """Test that sigma column gets a distinct color."""
        color = get_column_color("sigma")
        assert color is not None
        assert isinstance(color, str)

    def test_get_column_color_unassigned(self) -> None:
        """Test that unassigned columns get no color (None)."""
        color = get_column_color("unassigned")
        assert color is None

    def test_get_column_color_unknown_role(self) -> None:
        """Test that unknown roles get no color."""
        color = get_column_color("unknown_role")
        assert color is None

    def test_column_colors_are_distinct(self) -> None:
        """Test that all role colors are distinct from each other."""
        x_color = get_column_color("x")
        y_color = get_column_color("y")
        z_color = get_column_color("z")
        sigma_color = get_column_color("sigma")

        # All should be distinct
        colors = {x_color, y_color, z_color, sigma_color}
        assert len(colors) == 4


class TestRoleDisplayName:
    """Tests for role display name formatting."""

    def test_get_role_display_name_x(self) -> None:
        """Test display name for X role."""
        name = get_role_display_name("x")
        assert "X" in name.upper() or "x" in name

    def test_get_role_display_name_sigma(self) -> None:
        """Test display name for sigma role."""
        name = get_role_display_name("sigma")
        assert (
            "sigma" in name.lower()
            or "error" in name.lower()
            or "uncertainty" in name.lower()
        )


class TestColumnAssignments:
    """Tests for column assignment state management."""

    def test_create_column_assignments_1d_mode(self) -> None:
        """Test column assignments in 1D mode."""
        columns: dict[str, int | None] = {
            "x": 0,
            "y": 1,
            "sigma": None,
        }
        assignments = create_column_assignments(columns, mode="1d")

        assert assignments["x"] == 0
        assert assignments["y"] == 1
        assert assignments.get("z") is None
        assert assignments.get("sigma") is None

    def test_create_column_assignments_2d_mode(self) -> None:
        """Test column assignments in 2D mode."""
        columns: dict[str, int | None] = {
            "x": 0,
            "y": 1,
            "z": 2,
            "sigma": 3,
        }
        assignments = create_column_assignments(columns, mode="2d")

        assert assignments["x"] == 0
        assert assignments["y"] == 1
        assert assignments["z"] == 2
        assert assignments["sigma"] == 3

    def test_create_column_assignments_default_values(self) -> None:
        """Test default column assignments with empty input."""
        assignments = create_column_assignments({}, mode="1d")

        assert "x" in assignments
        assert "y" in assignments


class TestDataPreviewStats:
    """Tests for data statistics computation for preview display."""

    def test_compute_data_preview_stats_basic(self) -> None:
        """Test basic statistics computation."""
        data = np.array(
            [
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 6.0],
                [4.0, 8.0],
                [5.0, 10.0],
            ]
        )

        stats = compute_data_preview_stats(data)

        assert stats["num_rows"] == 5
        assert stats["num_columns"] == 2

    def test_compute_data_preview_stats_with_nan(self) -> None:
        """Test statistics with NaN values present."""
        data = np.array(
            [
                [1.0, 2.0],
                [np.nan, 4.0],
                [3.0, 6.0],
            ]
        )

        stats = compute_data_preview_stats(data)

        assert stats["num_rows"] == 3
        assert stats["nan_count"] > 0

    def test_compute_data_preview_stats_column_ranges(self) -> None:
        """Test that column ranges are computed correctly."""
        data = np.array(
            [
                [0.0, 100.0],
                [5.0, 200.0],
                [10.0, 300.0],
            ]
        )

        stats = compute_data_preview_stats(data)

        assert "column_stats" in stats
        assert len(stats["column_stats"]) == 2
        assert stats["column_stats"][0]["min"] == 0.0
        assert stats["column_stats"][0]["max"] == 10.0
        assert stats["column_stats"][1]["min"] == 100.0
        assert stats["column_stats"][1]["max"] == 300.0

    def test_compute_data_preview_stats_empty_data(self) -> None:
        """Test statistics computation with empty data."""
        data = np.array([]).reshape(0, 2)

        stats = compute_data_preview_stats(data)

        assert stats["num_rows"] == 0
        assert stats["num_columns"] == 2


class TestColumnValidation:
    """Tests for column selection validation."""

    def test_validate_column_selections_valid_1d(self) -> None:
        """Test validation of valid 1D column selections."""
        columns = {"x": 0, "y": 1, "sigma": None}
        num_columns = 3

        result = validate_column_selections(columns, num_columns, mode="1d")

        assert result["is_valid"] is True
        assert result["message"] == ""

    def test_validate_column_selections_valid_2d(self) -> None:
        """Test validation of valid 2D column selections."""
        columns = {"x": 0, "y": 1, "z": 2, "sigma": 3}
        num_columns = 4

        result = validate_column_selections(columns, num_columns, mode="2d")

        assert result["is_valid"] is True

    def test_validate_column_selections_missing_required_1d(self) -> None:
        """Test validation fails when required columns missing in 1D mode."""
        columns = {"x": 0, "y": None}  # y is required
        num_columns = 2

        result = validate_column_selections(columns, num_columns, mode="1d")

        assert result["is_valid"] is False
        assert (
            "y" in result["message"].lower() or "required" in result["message"].lower()
        )

    def test_validate_column_selections_missing_z_in_2d(self) -> None:
        """Test validation fails when z column missing in 2D mode."""
        columns = {"x": 0, "y": 1, "z": None}  # z is required for 2D
        num_columns = 3

        result = validate_column_selections(columns, num_columns, mode="2d")

        assert result["is_valid"] is False
        assert (
            "z" in result["message"].lower() or "required" in result["message"].lower()
        )

    def test_validate_column_selections_out_of_range(self) -> None:
        """Test validation fails when column index is out of range."""
        columns = {"x": 0, "y": 5}  # 5 is out of range for 3 columns
        num_columns = 3

        result = validate_column_selections(columns, num_columns, mode="1d")

        assert result["is_valid"] is False
        assert (
            "range" in result["message"].lower()
            or "invalid" in result["message"].lower()
        )

    def test_validate_column_selections_duplicate_assignment(self) -> None:
        """Test validation fails when same column assigned to multiple roles."""
        columns = {"x": 0, "y": 0}  # Both x and y assigned to column 0
        num_columns = 2

        result = validate_column_selections(columns, num_columns, mode="1d")

        assert result["is_valid"] is False
        assert (
            "duplicate" in result["message"].lower()
            or "same" in result["message"].lower()
        )


class TestModeHandling:
    """Tests for 1D/2D mode handling in column selector."""

    def test_1d_mode_excludes_z_column(self) -> None:
        """Test that 1D mode does not include z column in required columns."""
        columns = {"x": 0, "y": 1}  # No z column
        num_columns = 2

        result = validate_column_selections(columns, num_columns, mode="1d")

        assert result["is_valid"] is True

    def test_2d_mode_requires_z_column(self) -> None:
        """Test that 2D mode requires z column."""
        columns = {"x": 0, "y": 1}  # No z column
        num_columns = 3

        result = validate_column_selections(columns, num_columns, mode="2d")

        assert result["is_valid"] is False

    def test_mode_affects_available_roles(self) -> None:
        """Test that mode affects which roles are available."""
        assignments_1d = create_column_assignments({"x": 0, "y": 1}, mode="1d")
        assignments_2d = create_column_assignments({"x": 0, "y": 1, "z": 2}, mode="2d")

        # In 1D mode, z should not be required
        assert assignments_1d.get("z") is None

        # In 2D mode, z should be present if provided
        assert assignments_2d.get("z") == 2
