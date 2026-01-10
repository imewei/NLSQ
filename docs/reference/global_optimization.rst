Global Optimization
===================

This reference covers NLSQ's global optimization capabilities, including
multi-start optimization and CMA-ES (Covariance Matrix Adaptation Evolution
Strategy).

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

NLSQ provides two main approaches for global optimization:

1. **Multi-start optimization**: Run multiple local optimizations from
   different starting points using Latin Hypercube Sampling or other
   quasi-random samplers.

2. **CMA-ES (Evolution Strategy)**: A gradient-free evolutionary algorithm
   that adapts the search covariance matrix, particularly effective for
   multi-scale parameter problems.

Installation
------------

Multi-start optimization works out of the box. For CMA-ES, install the
optional ``evosax`` dependency:

.. code-block:: bash

   pip install "nlsq[global]"

CMA-ES Global Optimization
--------------------------

CMA-ES is recommended when:

- Parameters span many orders of magnitude (>1000x scale ratio)
- The fitness landscape has multiple local minima
- Gradient information is unreliable
- You want robust convergence without sensitivity to initialization

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from nlsq.global_optimization import CMAESOptimizer, CMAESConfig
   import jax.numpy as jnp


   # Define model
   def model(x, a, b):
       return a * jnp.exp(-b * x)


   # Generate data
   x = jnp.linspace(0, 5, 100)
   y = 2.5 * jnp.exp(-0.5 * x)

   # Bounds are required for CMA-ES
   bounds = ([0.1, 0.01], [10.0, 2.0])

   # Create optimizer (uses default BIPOP configuration)
   optimizer = CMAESOptimizer()

   # Run optimization
   result = optimizer.fit(model, x, y, bounds=bounds)

   print(f"Optimal parameters: {result['popt']}")
   print(f"Parameter covariance: {result['pcov']}")

Using Presets
^^^^^^^^^^^^^

Three presets are available for common use cases:

.. code-block:: python

   # Fast preset: no restarts, 50 generations
   optimizer = CMAESOptimizer.from_preset("cmaes-fast")

   # Standard preset: BIPOP with 9 restarts, 100 generations
   optimizer = CMAESOptimizer.from_preset("cmaes")

   # Global preset: BIPOP with 9 restarts, 200 generations, 2x population
   optimizer = CMAESOptimizer.from_preset("cmaes-global")

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

For fine-grained control, create a custom ``CMAESConfig``:

.. code-block:: python

   from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

   config = CMAESConfig(
       popsize=32,  # Population size (None = auto)
       max_generations=150,  # Max generations per run
       sigma=0.3,  # Initial step size
       tol_fun=1e-10,  # Fitness tolerance
       tol_x=1e-10,  # Parameter tolerance
       restart_strategy="bipop",  # 'none' or 'bipop'
       max_restarts=5,  # Max BIPOP restarts
       refine_with_nlsq=True,  # Refine with Trust Region
       seed=42,  # For reproducibility
   )

   optimizer = CMAESOptimizer(config=config)

Integration with curve_fit
^^^^^^^^^^^^^^^^^^^^^^^^^^

CMA-ES can be used directly through ``curve_fit`` with the ``method`` parameter:

.. code-block:: python

   from nlsq import curve_fit

   result = curve_fit(
       model,
       x,
       y,
       bounds=bounds,
       method="cmaes",  # Explicitly request CMA-ES
   )

   # Or with custom config
   from nlsq.global_optimization import CMAESConfig

   config = CMAESConfig(max_generations=200, seed=42)
   result = curve_fit(
       model,
       x,
       y,
       bounds=bounds,
       method="cmaes",
       cmaes_config=config,
   )

Auto Method Selection
^^^^^^^^^^^^^^^^^^^^^

Use ``method="auto"`` to let NLSQ choose based on the problem:

.. code-block:: python

   from nlsq import curve_fit

   # NLSQ checks scale ratio and evosax availability
   result = curve_fit(model, x, y, bounds=bounds, method="auto")

The ``MethodSelector`` class handles the logic:

- If scale ratio > 1000x and evosax available: CMA-ES
- Otherwise: multi-start optimization

Diagnostics
^^^^^^^^^^^

CMA-ES returns detailed diagnostics:

.. code-block:: python

   result = optimizer.fit(model, x, y, bounds=bounds)
   diag = result["cmaes_diagnostics"]

   print(f"Total generations: {diag['total_generations']}")
   print(f"Total restarts: {diag['total_restarts']}")
   print(f"Final sigma: {diag['final_sigma']}")
   print(f"Best fitness: {diag['best_fitness']}")
   print(f"Convergence reason: {diag['convergence_reason']}")
   print(f"Wall time: {diag['wall_time']}s")

The ``CMAESDiagnostics`` class provides analysis methods:

.. code-block:: python

   from nlsq.global_optimization import CMAESDiagnostics

   diag = CMAESDiagnostics.from_dict(result["cmaes_diagnostics"])
   print(diag.summary())
   print(f"Fitness improvement: {diag.get_fitness_improvement()}")

BIPOP Restart Strategy
^^^^^^^^^^^^^^^^^^^^^^

BIPOP (Bi-Population) alternates between large and small population runs:

- **Large population**: More exploration, broader search
- **Small population**: More exploitation, faster convergence

The ``BIPOPRestarter`` class manages this:

.. code-block:: python

   from nlsq.global_optimization import BIPOPRestarter

   restarter = BIPOPRestarter(
       base_popsize=16,
       n_params=3,
       max_restarts=9,
       min_fitness_spread=1e-12,
   )

   while not restarter.exhausted:
       popsize = restarter.get_next_popsize()  # Alternates large/small
       # ... run CMA-ES ...
       restarter.register_restart()

Multi-Start Optimization
------------------------

For problems where CMA-ES is not needed, multi-start optimization provides
robust global search using quasi-random sampling.

See the ``MultiStartOrchestrator`` API for details.

API Reference
-------------

.. autoclass:: nlsq.global_optimization.CMAESConfig
   :members:
   :undoc-members:

.. autoclass:: nlsq.global_optimization.CMAESOptimizer
   :members:
   :undoc-members:

.. autoclass:: nlsq.global_optimization.CMAESDiagnostics
   :members:
   :undoc-members:

.. autoclass:: nlsq.global_optimization.BIPOPRestarter
   :members:
   :undoc-members:

.. autoclass:: nlsq.global_optimization.MethodSelector
   :members:
   :undoc-members:
