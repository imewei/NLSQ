nlsq.interfaces
===============

Protocol definitions for dependency injection and loose coupling.

.. versionadded:: 1.2.0
   Extracted from core modules to enable cleaner architecture.

This package provides Protocol classes that define interfaces for key
components in the NLSQ system. Using protocols enables:

- Dependency injection for testing
- Loose coupling between modules
- Clear contracts for implementations

All protocols are re-exported from the ``nlsq.interfaces`` package for
convenient imports:

.. code-block:: python

   from nlsq.interfaces import CacheProtocol, OptimizerProtocol

Available Protocols
-------------------

CacheProtocol
~~~~~~~~~~~~~

Protocol for cache implementations.

.. automodule:: nlsq.interfaces.cache_protocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OptimizerProtocol
~~~~~~~~~~~~~~~~~

Protocol for optimizer implementations.

.. automodule:: nlsq.interfaces.optimizer_protocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DataSourceProtocol
~~~~~~~~~~~~~~~~~~

Protocol for data source implementations.

.. automodule:: nlsq.interfaces.data_source_protocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

JacobianProtocol
~~~~~~~~~~~~~~~~

Protocol for Jacobian computation implementations.

.. automodule:: nlsq.interfaces.jacobian_protocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ResultProtocol
~~~~~~~~~~~~~~

Protocol for optimization result containers.

.. automodule:: nlsq.interfaces.result_protocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
