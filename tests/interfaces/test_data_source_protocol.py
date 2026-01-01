"""Protocol contract tests for DataSourceProtocol.

This module tests that data source implementations conform to the
DataSourceProtocol defined in nlsq.interfaces.
"""

import numpy as np
import pytest

from nlsq.interfaces.data_source_protocol import (
    ArrayDataSource,
    DataSourceProtocol,
    StreamingDataSourceProtocol,
)


class TestDataSourceProtocolDefinition:
    """Test that DataSourceProtocol is correctly defined."""

    def test_protocol_is_runtime_checkable(self):
        """DataSourceProtocol should be runtime_checkable."""

        class MockDataSource:
            @property
            def n_points(self) -> int:
                return 100

            @property
            def n_dims(self) -> int:
                return 1

            @property
            def dtype(self) -> np.dtype:
                return np.float64

            def get_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
                return np.zeros(end - start), np.zeros(end - start)

            def __len__(self) -> int:
                return 100

        assert isinstance(MockDataSource(), DataSourceProtocol)

    def test_protocol_requires_all_properties(self):
        """Classes missing properties should not satisfy protocol."""

        class MissingNPoints:
            @property
            def n_dims(self):
                return 1

            @property
            def dtype(self):
                return np.float64

            def get_chunk(self, start, end):
                return np.zeros(end - start), np.zeros(end - start)

            def __len__(self):
                return 100

        assert not isinstance(MissingNPoints(), DataSourceProtocol)


class TestArrayDataSourceConformance:
    """Test that ArrayDataSource conforms to DataSourceProtocol."""

    def test_array_data_source_satisfies_protocol(self):
        """ArrayDataSource should satisfy DataSourceProtocol."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        source = ArrayDataSource(xdata, ydata)
        assert isinstance(source, DataSourceProtocol)

    def test_n_points(self):
        """n_points should return correct count."""
        xdata = np.linspace(0, 10, 100)
        ydata = np.sin(xdata)
        source = ArrayDataSource(xdata, ydata)
        assert source.n_points == 100

    def test_n_dims_1d(self):
        """n_dims should be 1 for 1D x data."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        source = ArrayDataSource(xdata, ydata)
        assert source.n_dims == 1

    def test_n_dims_2d(self):
        """n_dims should reflect 2D x data shape."""
        xdata = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ydata = np.array([1.0, 2.0, 3.0])
        source = ArrayDataSource(xdata, ydata)
        assert source.n_dims == 2

    def test_dtype(self):
        """dtype should reflect ydata dtype."""
        xdata = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ydata = np.array([2.0, 4.0, 6.0], dtype=np.float64)
        source = ArrayDataSource(xdata, ydata)
        assert source.dtype == np.float64

    def test_get_chunk(self):
        """get_chunk should return correct slice."""
        xdata = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ydata = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        source = ArrayDataSource(xdata, ydata)

        x_chunk, y_chunk = source.get_chunk(1, 4)
        np.testing.assert_array_equal(x_chunk, [2.0, 3.0, 4.0])
        np.testing.assert_array_equal(y_chunk, [4.0, 6.0, 8.0])

    def test_len(self):
        """__len__ should return n_points."""
        xdata = np.linspace(0, 10, 50)
        ydata = np.cos(xdata)
        source = ArrayDataSource(xdata, ydata)
        assert len(source) == 50


class TestStreamingDataSourceProtocol:
    """Test StreamingDataSourceProtocol requirements."""

    def test_protocol_is_runtime_checkable(self):
        """StreamingDataSourceProtocol should be runtime_checkable."""

        class MockStreamingSource:
            def __init__(self):
                self._idx = 0

            @property
            def n_points(self) -> int:
                return 100

            @property
            def batch_size(self) -> int:
                return 10

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= 100:
                    raise StopIteration
                self._idx += 10
                return np.zeros(10), np.zeros(10)

            def reset(self):
                self._idx = 0

        assert isinstance(MockStreamingSource(), StreamingDataSourceProtocol)


class TestDataSourceUsagePatterns:
    """Test common data source usage patterns."""

    def test_iterate_over_chunks(self):
        """Data source should support chunk iteration."""
        xdata = np.linspace(0, 10, 100)
        ydata = np.sin(xdata)
        source = ArrayDataSource(xdata, ydata)

        chunk_size = 25
        chunks_collected = []

        for start in range(0, len(source), chunk_size):
            end = min(start + chunk_size, len(source))
            x_chunk, y_chunk = source.get_chunk(start, end)
            chunks_collected.append((x_chunk, y_chunk))

        assert len(chunks_collected) == 4
        assert all(len(x) == 25 for x, y in chunks_collected)

    def test_full_data_access(self):
        """Should be able to get all data at once."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        source = ArrayDataSource(xdata, ydata)

        x_all, y_all = source.get_chunk(0, len(source))
        np.testing.assert_array_equal(x_all, xdata)
        np.testing.assert_array_equal(y_all, ydata)

    def test_empty_chunk(self):
        """Empty chunk should return empty arrays."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([2.0, 4.0, 6.0])
        source = ArrayDataSource(xdata, ydata)

        x_empty, y_empty = source.get_chunk(2, 2)
        assert len(x_empty) == 0
        assert len(y_empty) == 0


class TestCustomDataSource:
    """Test custom data source implementations."""

    def test_lazy_data_source(self):
        """Custom lazy data source should work."""

        class LazyDataSource:
            """Data source that generates data on demand."""

            def __init__(self, n: int):
                self._n = n

            @property
            def n_points(self) -> int:
                return self._n

            @property
            def n_dims(self) -> int:
                return 1

            @property
            def dtype(self) -> np.dtype:
                return np.dtype(np.float64)

            def get_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
                x = np.arange(start, end, dtype=np.float64)
                y = np.sin(x)
                return x, y

            def __len__(self) -> int:
                return self._n

        source = LazyDataSource(1000)
        assert isinstance(source, DataSourceProtocol)
        assert len(source) == 1000

        x, y = source.get_chunk(10, 20)
        np.testing.assert_allclose(y, np.sin(x), rtol=1e-10)
