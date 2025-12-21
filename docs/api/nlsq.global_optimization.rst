nlsq.global_optimization
=========================

.. module:: nlsq.global_optimization
   :synopsis: Multi-start global optimization with Latin Hypercube Sampling

Multi-start optimization with Latin Hypercube Sampling (LHS) and quasi-random
samplers for global search in nonlinear least squares fitting.

This module provides tools for exploring parameter space to find global optima,
which is particularly useful for problems with multiple local minima.

Key Features
------------

- **Latin Hypercube Sampling (LHS)**: Stratified random sampling for better coverage
- **Quasi-random sequences**: Sobol and Halton for deterministic, low-discrepancy sampling
- **Multi-start orchestration**: Evaluate multiple starting points and select best
- **Tournament selection**: Memory-efficient selection for streaming datasets
- **Preset configurations**: 'fast', 'robust', 'global', 'thorough', 'streaming'

Quick Start
-----------

Basic multi-start optimization:

.. code-block:: python

   from nlsq import fit, curve_fit
   from nlsq.global_optimization import MultiStartOrchestrator
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 5, 100)
   y = 3.0 * np.exp(-0.5 * x) + 1.0 + np.random.normal(0, 0.1, 100)

   # Method 1: Use fit() with preset
   popt, pcov = fit(model, x, y, preset="robust", bounds=([0, 0, 0], [10, 5, 10]))

   # Method 2: Use MultiStartOrchestrator directly
   orchestrator = MultiStartOrchestrator.from_preset("global")
   result = orchestrator.fit(model, x, y, bounds=([0, 0, 0], [10, 5, 10]))

Configuration
-------------

.. autoclass:: GlobalOptimizationConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. data:: PRESETS
   :annotation: = {'fast': {...}, 'robust': {...}, 'global': {...}, 'thorough': {...}, 'streaming': {...}}

   Preset configurations for common use cases:

   - **fast**: n_starts=0, multi-start disabled for maximum speed
   - **robust**: n_starts=5, light multi-start for robustness
   - **global**: n_starts=20, thorough global search
   - **thorough**: n_starts=50, exhaustive search
   - **streaming**: n_starts=10, tournament selection for large datasets

Multi-Start Orchestrator
------------------------

.. autoclass:: MultiStartOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Sampling Functions
------------------

Latin Hypercube Sampling
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: latin_hypercube_sample

Sobol Sequence
^^^^^^^^^^^^^^

.. autofunction:: sobol_sample

Halton Sequence
^^^^^^^^^^^^^^^

.. autofunction:: halton_sample

Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: scale_samples_to_bounds

.. autofunction:: center_samples_around_p0

.. autofunction:: get_sampler

Tournament Selection
--------------------

For large/streaming datasets where evaluating all candidates on the full dataset
is impractical:

.. autoclass:: TournamentSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Integration with NLSQ
---------------------

Multi-start optimization is integrated with the existing NLSQ infrastructure:

- **Small datasets (<1M points)**: Full multi-start on complete data
- **Medium datasets (1M-100M points)**: Full multi-start, then chunked fit
- **Large datasets (>100M points)**: Tournament selection during streaming warmup

Usage Examples
--------------

Using curve_fit with multi-start
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nlsq import curve_fit

   popt, pcov = curve_fit(
       model,
       x,
       y,
       p0=[1, 1, 1],
       bounds=([0, 0, 0], [10, 10, 10]),
       multistart=True,
       n_starts=10,
       sampler="lhs",
       center_on_p0=True,
   )

Custom configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nlsq.global_optimization import GlobalOptimizationConfig, MultiStartOrchestrator

   config = GlobalOptimizationConfig(
       n_starts=30,
       sampler="sobol",
       center_on_p0=True,
       scale_factor=0.5,  # Tighter exploration around p0
   )

   orchestrator = MultiStartOrchestrator(config=config)
   result = orchestrator.fit(model, x, y, bounds=bounds)

Tournament selection for streaming
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from nlsq.global_optimization import (
       TournamentSelector,
       GlobalOptimizationConfig,
       latin_hypercube_sample,
   )

   # Generate candidates
   candidates = latin_hypercube_sample(20, 3)

   # Configure tournament
   config = GlobalOptimizationConfig(
       n_starts=20,
       elimination_rounds=3,
       elimination_fraction=0.5,
       batches_per_round=50,
   )

   selector = TournamentSelector(candidates, config)


   def data_generator():
       for _ in range(200):
           x_batch = np.random.randn(100)
           y_batch = model(x_batch, *true_params)
           yield x_batch, y_batch


   best_candidates = selector.run_tournament(data_generator(), model, top_m=1)

Interactive Notebooks
---------------------

Hands-on tutorials for global optimization:

- `Multistart Basics <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/07_global_optimization/01_multistart_basics.ipynb>`_ - Local minima traps and multi-start solutions
- `Sampling Strategies <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/07_global_optimization/02_sampling_strategies.ipynb>`_ - LHS, Sobol, Halton comparison
- `Presets and Config <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/07_global_optimization/03_presets_and_config.ipynb>`_ - Using built-in presets
- `Tournament Selection <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/07_global_optimization/04_tournament_selection.ipynb>`_ - Memory-efficient selection for streaming
- `Multistart Integration <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/07_global_optimization/05_multistart_integration.ipynb>`_ - Integration with curve_fit workflows

See Also
--------

- :func:`nlsq.curve_fit` : Main curve fitting function with multi-start support
- :class:`nlsq.LargeDatasetFitter` : Chunked processing for medium-large datasets
- :class:`nlsq.AdaptiveHybridStreamingOptimizer` : Streaming for very large datasets
