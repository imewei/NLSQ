"""Unit tests for DataPreprocessor component.

Tests for data validation, array conversion, and preprocessing functionality
extracted from CurveFit class.

Reference: specs/017-curve-fit-decomposition/spec.md FR-001, FR-019
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq.core.orchestration.data_preprocessor import DataPreprocessor
from nlsq.interfaces.orchestration_protocol import PreprocessedData

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def preprocessor() -> DataPreprocessor:
    """Create a DataPreprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def linear_model():
    """Simple linear model function."""

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * x + b

    return model


@pytest.fixture
def exponential_model():
    """Exponential decay model function."""

    def model(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.exp(-b * x)

    return model


@pytest.fixture
def simple_data() -> tuple[np.ndarray, np.ndarray]:
    """Simple test data."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.5, 4.8, 7.1, 9.5, 11.8])
    return x, y


@pytest.fixture
def data_with_sigma() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test data with uncertainties."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.5, 4.8, 7.1, 9.5, 11.8])
    sigma = np.array([0.1, 0.2, 0.15, 0.1, 0.2])
    return x, y, sigma


# =============================================================================
# Test PreprocessedData Result
# =============================================================================


class TestPreprocessedDataResult:
    """Tests for PreprocessedData result object."""

    def test_returns_preprocessed_data(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test preprocess returns PreprocessedData instance."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert isinstance(result, PreprocessedData)

    def test_preprocessed_data_has_required_fields(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test PreprocessedData has all required attributes."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert hasattr(result, "xdata")
        assert hasattr(result, "ydata")
        assert hasattr(result, "sigma")
        assert hasattr(result, "mask")
        assert hasattr(result, "n_points")
        assert hasattr(result, "is_padded")
        assert hasattr(result, "original_length")

    def test_n_points_matches_data(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test n_points reflects actual data size."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_points == len(y)


# =============================================================================
# Test Array Conversion
# =============================================================================


class TestArrayConversion:
    """Tests for array conversion functionality."""

    def test_converts_lists_to_arrays(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test lists are converted to arrays."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.5, 4.8, 7.1, 9.5, 11.8]

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert hasattr(result.xdata, "shape")
        assert hasattr(result.ydata, "shape")

    def test_converts_tuples_to_arrays(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test tuples are converted to arrays."""
        x = (1.0, 2.0, 3.0, 4.0, 5.0)
        y = (2.5, 4.8, 7.1, 9.5, 11.8)

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert hasattr(result.xdata, "shape")
        assert hasattr(result.ydata, "shape")

    def test_preserves_numpy_arrays(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test numpy arrays are preserved."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert_allclose(np.asarray(result.xdata), x)
        assert_allclose(np.asarray(result.ydata), y)

    def test_handles_2d_xdata(self, preprocessor: DataPreprocessor) -> None:
        """Test preprocessing handles 2D x data."""

        def surface_model(xy, a, b, c):
            x, y = xy
            return a * x + b * y + c

        x = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([10, 20, 30])

        result = preprocessor.preprocess(
            f=surface_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_points == 3


# =============================================================================
# Test Finite Value Checking
# =============================================================================


class TestFiniteValueChecking:
    """Tests for check_finite functionality."""

    def test_raises_on_nan_in_ydata(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test raises ValueError for NaN in ydata."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, np.nan, 7.1, 9.5, 11.8])

        with pytest.raises(ValueError, match=r"[Nn]aN|[Ff]inite"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
                check_finite=True,
            )

    def test_raises_on_nan_in_xdata(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test raises ValueError for NaN in xdata."""
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y = np.array([2.5, 4.8, 7.1, 9.5, 11.8])

        with pytest.raises(ValueError, match=r"[Nn]aN|[Ff]inite"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
                check_finite=True,
            )

    def test_raises_on_inf_in_ydata(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test raises ValueError for Inf in ydata."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, np.inf, 7.1, 9.5, 11.8])

        with pytest.raises(ValueError, match=r"[Ii]nf|[Ff]inite"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
                check_finite=True,
            )

    def test_allows_nan_when_check_finite_false(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test NaN is allowed when check_finite=False."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, np.nan, 7.1, 9.5, 11.8])

        # Should not raise
        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
            check_finite=False,
        )

        assert result is not None


# =============================================================================
# Test Data Length Validation
# =============================================================================


class TestDataLengthValidation:
    """Tests for data length validation."""

    def test_raises_on_length_mismatch(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test raises ValueError when x and y lengths differ."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.5, 4.8, 7.1, 9.5])

        with pytest.raises(ValueError, match=r"[Ll]ength|[Mm]atch"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
            )

    def test_raises_on_empty_ydata(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test raises ValueError for empty ydata."""
        x = np.array([])
        y = np.array([])

        with pytest.raises(ValueError, match=r"[Ee]mpty"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
            )


# =============================================================================
# Test Sigma Handling
# =============================================================================


class TestSigmaHandling:
    """Tests for sigma (uncertainty) handling."""

    def test_processes_sigma(
        self, preprocessor: DataPreprocessor, linear_model, data_with_sigma
    ) -> None:
        """Test sigma is processed correctly."""
        x, y, sigma = data_with_sigma

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
            sigma=sigma,
        )

        assert result.sigma is not None
        assert_allclose(np.asarray(result.sigma), sigma)

    def test_sigma_none_when_not_provided(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test sigma is None when not provided."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.sigma is None

    def test_validates_sigma_length(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test raises when sigma length doesn't match ydata."""
        x, y = simple_data
        sigma = np.array([0.1, 0.2])  # Wrong length

        with pytest.raises(ValueError, match=r"[Ss]igma|[Ll]ength"):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
                sigma=sigma,
            )


# =============================================================================
# Test Mask Generation
# =============================================================================


class TestMaskGeneration:
    """Tests for data mask generation."""

    def test_generates_all_true_mask(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test default mask is all True."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert np.all(result.mask)
        assert len(result.mask) == len(y)


# =============================================================================
# Test NaN Policy
# =============================================================================


class TestNanPolicy:
    """Tests for nan_policy handling."""

    def test_nan_policy_raise(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test nan_policy='raise' raises on NaN."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, np.nan, 7.1, 9.5, 11.8])

        with pytest.raises(ValueError):
            preprocessor.preprocess(
                f=linear_model,
                xdata=x,
                ydata=y,
                nan_policy="raise",
                check_finite=True,
            )

    def test_nan_policy_omit(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test nan_policy='omit' removes NaN values."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.5, np.nan, 7.1, 9.5, 11.8])

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
            nan_policy="omit",
            check_finite=False,  # Don't raise, let nan_policy handle it
        )

        # Should have one fewer point
        assert result.n_points == 4
        assert result.has_nans_removed  # Could be np.True_ or True


# =============================================================================
# Test Immutability
# =============================================================================


class TestImmutability:
    """Tests for PreprocessedData immutability."""

    def test_preprocessed_data_is_frozen(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test PreprocessedData cannot be modified."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.n_points = 100  # type: ignore[misc]


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_data_point(
        self, preprocessor: DataPreprocessor, linear_model
    ) -> None:
        """Test preprocessing single data point."""
        x = np.array([1.0])
        y = np.array([2.5])

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_points == 1

    def test_large_dataset(self, preprocessor: DataPreprocessor, linear_model) -> None:
        """Test preprocessing large dataset."""
        n = 10000
        x = np.linspace(0, 10, n)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.1, n)

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        assert result.n_points == n

    def test_preserves_data_values(
        self, preprocessor: DataPreprocessor, linear_model, simple_data
    ) -> None:
        """Test preprocessing preserves data values."""
        x, y = simple_data

        result = preprocessor.preprocess(
            f=linear_model,
            xdata=x,
            ydata=y,
        )

        # Values should be preserved
        assert_allclose(np.asarray(result.xdata)[: len(x)], x)
        assert_allclose(np.asarray(result.ydata)[: len(y)], y)
