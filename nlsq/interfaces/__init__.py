"""Protocol interfaces for NLSQ dependency injection.

This package provides protocol definitions that enable loose coupling between
modules and break circular dependencies. Modules should depend on these
abstractions rather than concrete implementations.

Protocols
---------
OptimizerProtocol
    Protocol for optimization algorithms.
LeastSquaresOptimizerProtocol
    Extended protocol for least squares optimizers.
CurveFitProtocol
    Protocol for curve fitting interfaces.
DataSourceProtocol
    Protocol for data sources (arrays, HDF5, streaming).
StreamingDataSourceProtocol
    Extended protocol for streaming data sources.
ResultProtocol
    Protocol for optimization results.
JacobianProtocol
    Protocol for Jacobian computation strategies.
SparseJacobianProtocol
    Extended protocol for sparse Jacobians.
CacheProtocol
    Protocol for caching mechanisms.
BoundedCacheProtocol
    Extended protocol for memory-bounded caches.

Concrete Implementations
------------------------
ArrayDataSource
    Data source for NumPy arrays.
AutodiffJacobian
    Jacobian computation using JAX autodiff.
DictCache
    Simple dictionary-based cache.
"""

from nlsq.interfaces.cache_protocol import (
    BoundedCacheProtocol,
    CacheProtocol,
    DictCache,
)
from nlsq.interfaces.data_source_protocol import (
    ArrayDataSource,
    DataSourceProtocol,
    StreamingDataSourceProtocol,
)
from nlsq.interfaces.jacobian_protocol import (
    AutodiffJacobian,
    JacobianProtocol,
    SparseJacobianProtocol,
)
from nlsq.interfaces.optimizer_protocol import (
    CurveFitProtocol,
    LeastSquaresOptimizerProtocol,
    OptimizerProtocol,
)
from nlsq.interfaces.result_protocol import ResultProtocol

__all__ = [
    "ArrayDataSource",
    "AutodiffJacobian",
    "BoundedCacheProtocol",
    "CacheProtocol",
    "CurveFitProtocol",
    "DataSourceProtocol",
    "DictCache",
    "JacobianProtocol",
    "LeastSquaresOptimizerProtocol",
    "OptimizerProtocol",
    "ResultProtocol",
    "SparseJacobianProtocol",
    "StreamingDataSourceProtocol",
]
