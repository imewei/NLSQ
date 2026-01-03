nlsq.core.adapters
==================

Protocol adapters for dependency injection in the NLSQ core module.

This subpackage provides adapter classes that implement the protocols defined
in ``nlsq.interfaces``, enabling flexible dependency injection and loose coupling
between components.

.. automodule:: nlsq.core.adapters
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   from nlsq.core.adapters import CurveFitAdapter
   from nlsq.interfaces import CurveFitProtocol

   # Create adapter instance
   adapter = CurveFitAdapter()

   # Use as CurveFitProtocol implementation
   result = adapter.fit(model, xdata, ydata, p0=initial_params)
