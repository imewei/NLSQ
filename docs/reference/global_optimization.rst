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

Memory Management for Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMA-ES can encounter out-of-memory (OOM) issues with large datasets (>10M points)
because each fitness evaluation processes the full dataset across all population
members. NLSQ provides two strategies to manage memory:

**Population Batching**: Evaluates population members in smaller groups instead
of all at once.

.. code-block:: python

   config = CMAESConfig(
       population_batch_size=4,  # Evaluate 4 candidates at a time
   )

**Data Streaming**: Processes the dataset in chunks, accumulating the sum of
squared residuals across chunks.

.. code-block:: python

   config = CMAESConfig(
       data_chunk_size=50000,  # Process 50K points per chunk
   )

**Combined Configuration** for maximum memory efficiency:

.. code-block:: python

   from nlsq.global_optimization import CMAESConfig, CMAESOptimizer

   # For 100M+ point datasets
   config = CMAESConfig(
       population_batch_size=4,  # Batch population evaluation
       data_chunk_size=50000,  # Stream data in 50K chunks
       max_generations=100,
   )

   optimizer = CMAESOptimizer(config=config)

Memory Estimation
"""""""""""""""""

Use the helper functions to estimate and auto-configure memory usage:

.. code-block:: python

   from nlsq.global_optimization import (
       estimate_cmaes_memory_gb,
       auto_configure_cmaes_memory,
   )

   # Estimate memory for a configuration
   memory_gb = estimate_cmaes_memory_gb(
       n_data=100_000_000,
       popsize=16,
       population_batch_size=4,
       data_chunk_size=50000,
   )
   print(f"Estimated memory: {memory_gb:.3f} GB")  # ~0.005 GB

   # Auto-configure based on available memory
   pop_batch, data_chunk = auto_configure_cmaes_memory(
       n_data=100_000_000,
       popsize=16,
       available_memory_gb=4.0,  # Target GPU memory
   )
   config = CMAESConfig(
       population_batch_size=pop_batch,
       data_chunk_size=data_chunk,
   )

Memory comparison for 100M points with popsize=16:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Configuration
     - Peak Memory
     - Reduction
   * - No batching (default)
     - 12.8 GB
     - --
   * - ``population_batch_size=4``
     - 3.2 GB
     - 75%
   * - ``data_chunk_size=50000``
     - ~18 MB
     - 99.9%
   * - Both combined
     - ~5 MB
     - 99.96%

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

Configuration
^^^^^^^^^^^^^

.. autoclass:: nlsq.global_optimization.CMAESConfig
   :members:
   :undoc-members:

Optimizer
^^^^^^^^^

.. autoclass:: nlsq.global_optimization.CMAESOptimizer
   :members:
   :undoc-members:

Diagnostics
^^^^^^^^^^^

.. autoclass:: nlsq.global_optimization.CMAESDiagnostics
   :members:
   :undoc-members:

Restart Strategy
^^^^^^^^^^^^^^^^

.. autoclass:: nlsq.global_optimization.BIPOPRestarter
   :members:
   :undoc-members:

Method Selection
^^^^^^^^^^^^^^^^

.. autoclass:: nlsq.global_optimization.MethodSelector
   :members:
   :undoc-members:

Memory Helpers
^^^^^^^^^^^^^^

.. autofunction:: nlsq.global_optimization.estimate_cmaes_memory_gb

.. autofunction:: nlsq.global_optimization.auto_configure_cmaes_memory
