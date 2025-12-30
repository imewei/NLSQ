"""Protocol definition for data sources.

This module defines the DataSourceProtocol that data providers should implement,
enabling support for different data backends (arrays, HDF5, streaming).
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DataSourceProtocol(Protocol):
    """Protocol for data sources.

    This protocol defines the interface for data providers, allowing
    support for various backends like NumPy arrays, HDF5 files, or
    streaming data without coupling to specific implementations.

    Properties
    ----------
    n_points : int
        Total number of data points.
    n_dims : int
        Number of dimensions in x data (1 for scalar x).
    dtype : np.dtype
        Data type of the arrays.

    Methods
    -------
    get_chunk(start, end)
        Get a chunk of data from start to end indices.
    __len__()
        Return the total number of data points.
    """

    @property
    def n_points(self) -> int:
        """Total number of data points."""
        ...

    @property
    def n_dims(self) -> int:
        """Number of dimensions in x data."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Data type of the arrays."""
        ...

    def get_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a chunk of data.

        Parameters
        ----------
        start : int
            Start index (inclusive).
        end : int
            End index (exclusive).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (xdata, ydata) arrays for the requested chunk.
        """
        ...

    def __len__(self) -> int:
        """Return total number of data points."""
        ...


@runtime_checkable
class StreamingDataSourceProtocol(Protocol):
    """Protocol for streaming data sources.

    Extended protocol for data sources that support streaming iteration
    over batches, useful for datasets that don't fit in memory.
    """

    @property
    def n_points(self) -> int:
        """Total number of data points."""
        ...

    @property
    def batch_size(self) -> int:
        """Size of each batch."""
        ...

    def __iter__(self) -> "StreamingDataSourceProtocol":
        """Return iterator over batches."""
        ...

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """Get next batch of (xdata, ydata)."""
        ...

    def reset(self) -> None:
        """Reset iterator to beginning."""
        ...


class ArrayDataSource:
    """Concrete implementation of DataSourceProtocol for NumPy arrays.

    This is the default data source for in-memory arrays.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data.
    """

    __slots__ = ("_xdata", "_ydata")

    def __init__(self, xdata: np.ndarray, ydata: np.ndarray) -> None:
        self._xdata = np.asarray(xdata)
        self._ydata = np.asarray(ydata)

    @property
    def n_points(self) -> int:
        """Total number of data points."""
        return len(self._ydata)

    @property
    def n_dims(self) -> int:
        """Number of dimensions in x data."""
        return 1 if self._xdata.ndim == 1 else self._xdata.shape[1]

    @property
    def dtype(self) -> np.dtype:
        """Data type of the arrays."""
        return self._ydata.dtype

    def get_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Get a chunk of data."""
        return self._xdata[start:end], self._ydata[start:end]

    def __len__(self) -> int:
        """Return total number of data points."""
        return self.n_points
