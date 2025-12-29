"""Protocol interfaces for NLSQ dependency injection.

This package provides protocol definitions that enable loose coupling between
modules and break circular dependencies. Modules should depend on these
abstractions rather than concrete implementations.

Protocols
---------
OptimizerProtocol
    Protocol for optimization algorithms.
DataSourceProtocol
    Protocol for data sources (arrays, HDF5, streaming).
ResultProtocol
    Protocol for optimization results.
JacobianProtocol
    Protocol for Jacobian computation strategies.
CacheProtocol
    Protocol for caching mechanisms.
"""

from nlsq.interfaces.cache_protocol import (
    BoundedCacheProtocol,
    CacheProtocol,
    DictCache,
)
from nlsq.interfaces.data_source_protocol import DataSourceProtocol
from nlsq.interfaces.jacobian_protocol import JacobianProtocol
from nlsq.interfaces.optimizer_protocol import OptimizerProtocol
from nlsq.interfaces.result_protocol import ResultProtocol

__all__ = [
    "BoundedCacheProtocol",
    "CacheProtocol",
    "DataSourceProtocol",
    "DictCache",
    "JacobianProtocol",
    "OptimizerProtocol",
    "ResultProtocol",
]
