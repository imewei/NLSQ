Workflow System Overview
========================

.. versionchanged:: 0.6.3
   The workflow system was simplified from 9 presets to 3 smart workflows:
   ``auto``, ``auto_global``, and ``hpc``. The system now automatically selects
   the optimal strategy based on memory constraints and problem characteristics.

NLSQ provides automatic workflow selection based on memory constraints and dataset
characteristics. The system analyzes available memory and data size to choose the
optimal fitting strategy, preventing out-of-memory errors while maximizing performance.

The Three Workflows
-------------------

NLSQ v0.6.3 provides three workflows that cover all use cases:

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Workflow
     - Description
     - Bounds
     - Use Case
   * - ``auto``
     - Memory-aware local optimization
     - Optional
     - **Default**. Standard curve fitting.
   * - ``auto_global``
     - Memory-aware global optimization
     - Required
     - Multi-modal problems, unknown initial guess.
   * - ``hpc``
     - ``auto_global`` + checkpointing
     - Required
     - Long-running HPC jobs.

workflow="auto" (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~

The default workflow for local optimization. It automatically selects the
best memory strategy based on your data size:

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Default: workflow="auto"
   result = fit(model, x, y, p0=[1.0, 0.5, 0.1])

   # Explicit workflow selection
   result = fit(model, x, y, p0=[1.0, 0.5, 0.1], workflow="auto")

   # With optional bounds (constrains solution to valid range)
   result = fit(
       model, x, y, p0=[1.0, 0.5, 0.1], workflow="auto", bounds=([0, 0, -1], [10, 5, 1])
   )

workflow="auto_global"
~~~~~~~~~~~~~~~~~~~~~~

For problems with multiple local minima or unknown initial guesses. Requires
bounds to define the search space.

The system automatically selects between:

- **CMA-ES**: When parameter scale ratio > 1000 (wide bounds relative to typical values)
- **Multi-Start**: Otherwise, using Latin Hypercube Sampling

.. code-block:: python

   from nlsq import fit

   # Global optimization with automatic method selection
   result = fit(
       model,
       x,
       y,
       p0=[1.0, 0.5, 0.1],
       workflow="auto_global",
       bounds=([0, 0, 0], [10, 5, 1]),
       n_starts=10,  # For multi-start (default: 10)
   )

workflow="hpc"
~~~~~~~~~~~~~~

For long-running jobs on HPC clusters. Wraps ``auto_global`` with automatic
checkpointing for crash recovery.

.. code-block:: python

   from nlsq import fit

   result = fit(
       model,
       x,
       y,
       p0=[1.0, 0.5, 0.1],
       workflow="hpc",
       bounds=([0, 0, 0], [10, 5, 1]),
       checkpoint_dir="/scratch/my_job/checkpoints",
       checkpoint_interval=10,  # Save every 10 generations/starts
   )

Memory Strategy Selection
-------------------------

Both ``auto`` and ``auto_global`` workflows use the ``MemoryBudgetSelector``
to choose the optimal memory strategy. The selector uses 75% of available
RAM as the threshold.

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    MEMORY BUDGET COMPUTATION                     │
   ├─────────────────────────────────────────────────────────────────┤
   │ available_gb = psutil.virtual_memory().available / 1e9          │
   │ threshold_gb = available_gb × 0.75  (safety factor)             │
   │                                                                  │
   │ # Memory estimates (float64 = 8 bytes)                          │
   │ data_gb     = n_points × 2 × 8 / 1e9  (x + y)                   │
   │ jacobian_gb = n_points × n_params × 8 / 1e9                     │
   │ peak_gb     = data_gb + 1.3 × jacobian_gb + solver_overhead     │
   └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     data_gb > threshold_gb ?  │
               └───────────────────────────────┘
                       │ YES              │ NO
                       ▼                  ▼
          ┌──────────────────┐    ┌───────────────────────────┐
          │ STREAMING        │    │ peak_gb > threshold_gb?   │
          │ HybridStreaming  │    └───────────────────────────┘
          │ with adaptive    │          │ YES           │ NO
          │ batch_size       │          ▼               ▼
          └──────────────────┘   ┌─────────────┐  ┌─────────────┐
                                 │ CHUNKED     │  │ STANDARD    │
                                 │ LDMemory    │  │ Direct TRF  │
                                 │ with auto   │  │ curve_fit() │
                                 │ chunk_size  │  └─────────────┘
                                 └─────────────┘

Strategy × Method Matrix
~~~~~~~~~~~~~~~~~~~~~~~~

The ``auto_global`` workflow produces 6 combinations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Memory Strategy
     - Multi-Start
     - CMA-ES
   * - **standard**
     - MultiStartOrchestrator + n_starts × TRF
     - CMAESOptimizer + BIPOP + TRF refine
   * - **chunked**
     - LargeDatasetFitter + multi-start
     - CMAESOptimizer + data_chunk_size
   * - **streaming**
     - AdaptiveHybridStreaming + multi-start
     - CMAESOptimizer + data streaming

Method Selection (CMA-ES vs Multi-Start)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``MethodSelector`` chooses between CMA-ES and Multi-Start based on
parameter scale ratio:

.. code-block:: python

   from nlsq.global_optimization.method_selector import MethodSelector

   selector = MethodSelector()
   method = selector.select("auto", lower_bounds, upper_bounds)
   # Returns "cmaes" or "multi-start"

- **CMA-ES**: Selected when ``scale_ratio > 1000`` AND ``evosax`` is available
- **Multi-Start**: Selected otherwise

The scale ratio is computed as:

.. code-block:: python

   scale_ratio = max(upper - lower) / min(upper - lower)

Memory Override
~~~~~~~~~~~~~~~

You can override automatic memory detection:

.. code-block:: python

   # Force smaller memory footprint
   result = fit(
       model,
       x,
       y,
       p0=[1, 2],
       workflow="auto",
       memory_limit_gb=4.0,  # Pretend only 4GB available
   )

Tolerance Configuration
-----------------------

Tolerances are set directly, not via presets:

.. code-block:: python

   # Fast fitting with looser tolerances
   result = fit(model, x, y, p0=[1, 2], gtol=1e-6, ftol=1e-6, xtol=1e-6)

   # High precision fitting
   result = fit(model, x, y, p0=[1, 2], gtol=1e-10, ftol=1e-10, xtol=1e-10)

Migration from Old Presets
--------------------------

.. versionchanged:: 0.6.3
   The following presets were removed. Using them will raise ``ValueError``
   with a migration hint.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Old Preset
     - New Equivalent
   * - ``standard``
     - ``workflow="auto"``
   * - ``fast``
     - ``workflow="auto", gtol=1e-6, ftol=1e-6, xtol=1e-6``
   * - ``quality``
     - ``workflow="auto_global", n_starts=20``
   * - ``large_robust``
     - ``workflow="auto"`` (auto-detects large data)
   * - ``streaming``
     - ``workflow="auto"`` (auto-detects memory pressure)
   * - ``hpc_distributed``
     - ``workflow="hpc"``
   * - ``cmaes``
     - ``workflow="auto_global"`` (auto-selects CMA-ES)
   * - ``cmaes-global``
     - ``workflow="auto_global", cmaes_config=CMAESConfig(n_generations=200)``
   * - ``global_auto``
     - ``workflow="auto_global"``

4-Layer Defense Strategy
------------------------

All workflows using ``hybrid_streaming`` or ``AdaptiveHybridStreamingOptimizer``
include a 4-layer defense against L-BFGS warmup divergence. This is particularly
important for **warm-start refinement** scenarios where initial parameters are
already near optimal.

The layers activate automatically:

1. **Warm Start Detection**: Skips warmup if initial loss < 1% of data variance
2. **Adaptive Step Size**: Scales step size based on fit quality (1e-6 to 0.001)
3. **Cost-Increase Guard**: Aborts if loss increases > 5%
4. **Step Clipping**: Limits parameter update magnitude (max norm 0.1)

Where to go next
----------------

- API reference: :doc:`../api/nlsq.workflow`
- Configuration options: :doc:`../reference/configuration`
- Common workflow patterns: :doc:`../howto/common_workflows`
- Large dataset handling: :doc:`../howto/handle_large_data`
