Facades
=======

.. versionadded:: 0.6.4

Facades provide lazy-loading wrappers that break circular dependencies
while maintaining clean import paths. Components load only when accessed.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Facade
     - Purpose
   * - :class:`~nlsq.facades.OptimizationFacade`
     - CMA-ES and multi-start optimizers
   * - :class:`~nlsq.facades.StabilityFacade`
     - SVD fallback and stability guards
   * - :class:`~nlsq.facades.DiagnosticsFacade`
     - Convergence monitoring and diagnostics

OptimizationFacade
------------------

.. autoclass:: nlsq.facades.OptimizationFacade
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.facades import OptimizationFacade

   facade = OptimizationFacade()

   # Get CMA-ES optimizer (loads only when accessed)
   CMAESOptimizer = facade.get_cmaes_optimizer()

   # Get multi-start optimizer
   MultiStartOptimizer = facade.get_multistart_optimizer()

StabilityFacade
---------------

.. autoclass:: nlsq.facades.StabilityFacade
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.facades import StabilityFacade

   facade = StabilityFacade()

   # Get SVD fallback function
   svd_with_fallback = facade.get_fallback_svd()

   # Get stability guard
   StabilityGuard = facade.get_stability_guard()

DiagnosticsFacade
-----------------

.. autoclass:: nlsq.facades.DiagnosticsFacade
   :members:
   :show-inheritance:

**Example:**

.. code-block:: python

   from nlsq.facades import DiagnosticsFacade

   facade = DiagnosticsFacade()

   # Get diagnostic level enum
   DiagnosticLevel = facade.get_diagnostic_level()

   # Get convergence monitor
   ConvergenceMonitor = facade.get_convergence_monitor()

Benefits
--------

1. **Reduced Import Time**: Dependencies load only when needed
2. **Circular Dependency Breaking**: Clean separation between modules
3. **Testing Isolation**: Easy to mock for unit tests
4. **Gradual Migration**: Supports feature flag-based rollout

See Also
--------

- :doc:`orchestration` - Orchestration components
- :doc:`global_optimization` - Global optimization methods
- :doc:`/tutorials/advanced/architecture/index` - Architecture overview
