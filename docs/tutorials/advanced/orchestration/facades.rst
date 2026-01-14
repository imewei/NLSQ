Facades
=======

.. versionadded:: 0.6.4

Facades provide lazy-loading wrappers that break circular dependencies
while providing convenient access to components.

Why Facades?
------------

1. **Circular Dependencies**: Some modules depend on each other
2. **Lazy Loading**: Load expensive modules only when needed
3. **Convenience**: Single import point for related functionality

Available Facades
-----------------

Located in ``nlsq/facades/``:

- ``OptimizationFacade``: Global optimization components
- ``StabilityFacade``: Numerical stability components
- ``DiagnosticsFacade``: Diagnostics and monitoring

OptimizationFacade
------------------

Access global optimization without circular imports:

.. code-block:: python

   from nlsq.facades import OptimizationFacade

   facade = OptimizationFacade()

   # Get CMA-ES optimizer (lazy loaded)
   CMAESOptimizer = facade.get_cmaes_optimizer()
   optimizer = facade.create_cmaes_optimizer(bounds=([0], [10]))

   # Get multi-start orchestrator
   MultiStart = facade.get_multistart_optimizer()
   multistart = facade.create_multistart_orchestrator(n_starts=20)

**Methods:**

.. code-block:: python

   facade.get_cmaes_optimizer()  # Returns CMAESOptimizer class
   facade.create_cmaes_optimizer(bounds)  # Creates configured instance
   facade.get_multistart_optimizer()  # Returns MultiStartOrchestrator class
   facade.create_multistart_orchestrator(n_starts)  # Creates instance

StabilityFacade
---------------

Access stability components:

.. code-block:: python

   from nlsq.facades import StabilityFacade

   facade = StabilityFacade()

   # Get fallback orchestrator
   fallback = facade.get_fallback_orchestrator()

   # Get optimization recovery
   recovery = facade.get_optimization_recovery()

   # Get numerical guard
   guard = facade.get_numerical_guard()

DiagnosticsFacade
-----------------

Access diagnostics components:

.. code-block:: python

   from nlsq.facades import DiagnosticsFacade

   facade = DiagnosticsFacade()

   # Get convergence monitor
   monitor = facade.get_convergence_monitor()

   # Get diagnostics configuration
   config = facade.get_diagnostics_config()

Lazy Loading Behavior
---------------------

Components are loaded on first access:

.. code-block:: python

   from nlsq.facades import OptimizationFacade

   # Fast: no heavy imports yet
   facade = OptimizationFacade()

   # First access: imports nlsq.global_optimization
   CMAESOptimizer = facade.get_cmaes_optimizer()

   # Subsequent access: cached
   CMAESOptimizer2 = facade.get_cmaes_optimizer()  # Same object

Example Use Case
----------------

Breaking circular dependencies:

.. code-block:: python

   # In nlsq/core/minpack.py (can't import global_optimization directly)

   from nlsq.facades import OptimizationFacade


   def fit_with_global(model, x, y, bounds, n_starts=10):
       facade = OptimizationFacade()

       # Create multi-start optimizer (lazy loaded)
       orchestrator = facade.create_multistart_orchestrator(n_starts=n_starts)

       # Run global optimization
       results = orchestrator.optimize(lambda p: model(x, *p), bounds=bounds)

       return results.best

Complete Example
----------------

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from nlsq.facades import OptimizationFacade, StabilityFacade


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Generate data
   np.random.seed(42)
   x = np.linspace(0, 10, 100)
   y = 2.5 * np.exp(-0.5 * x) + 0.3 + 0.1 * np.random.randn(100)

   # Use facades for components
   opt_facade = OptimizationFacade()
   stab_facade = StabilityFacade()

   # Get multistart optimizer
   orchestrator = opt_facade.create_multistart_orchestrator(n_starts=10)


   # Define objective
   def objective(params):
       return jnp.sum((model(x, *params) - y) ** 2)


   # Define bounds
   bounds = ([0, 0, -1], [10, 5, 1])

   # Run multi-start optimization
   # (This is a simplified example - actual API may differ)
   print("Running multi-start optimization via facade...")

   # Get stability guard for checking
   guard = stab_facade.get_numerical_guard()
   print(f"Stability guard type: {type(guard)}")

Next Steps
----------

- :doc:`../custom_workflows/index` - Building custom pipelines
- :doc:`../performance/index` - Performance optimization
