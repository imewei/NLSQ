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
     - ``auto_global`` + logged cluster detection (checkpointing on the CMA-ES route only)
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

For long-running jobs on HPC clusters. ``hpc`` runs ``auto_global`` after logging
what ``ClusterDetector`` found (a PBS allocation read from ``$PBS_NODEFILE``, or the
local GPUs; the result is informational and does not change the fit).

Checkpoint and resume work on the CMA-ES route only. On the multi-start, chunked
and streaming routes, passing ``checkpoint_dir`` emits a ``UserWarning`` and writes
nothing. To checkpoint, select CMA-ES explicitly:

.. code-block:: python

   from nlsq import fit
   from nlsq.global_optimization import CMAESConfig

   result = fit(
       model,
       x,
       y,
       p0=[1.0, 0.5, 0.1],
       workflow="hpc",
       bounds=([0, 0, 0], [10, 5, 1]),
       method="cmaes",
       cmaes_config=CMAESConfig(restart_strategy="none"),
       checkpoint_dir="/scratch/my_job/checkpoints",
       checkpoint_interval=10,  # Save every 10 generations
       run_id="exp-42",
       model_id="exp_decay_v1",
       seed=42,
   )

See :doc:`../tutorials/routine/three_workflows/hpc_workflow` for the full
checkpointing walkthrough.

How ``fit``, ``curve_fit`` and ``curve_fit_large`` relate
----------------------------------------------------------

``curve_fit`` is the in-memory base case. The other entry points either call it
or replace it, and they pick their engine by **different rules**, so the same
data can take different engines depending on which one you call:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Call
     - Decides by
     - Engines it can pick
   * - ``fit(...)`` (no ``workflow``)
     - Point count against ``size_threshold`` (default 1,000,000)
     - ``curve_fit`` below the threshold, ``curve_fit_large`` at or above it
   * - ``curve_fit_large(...)``
     - Point count against ``size_threshold``, via the ``Auto`` strategy
     - ``curve_fit`` below the threshold; exact streaming LM at or above it
       (opt out with ``strategy=Chunked()``); chunked or hybrid when you ask
   * - ``fit(..., workflow=...)``
     - Estimated memory (``MemoryBudgetSelector``, below), not point count
     - ``curve_fit`` (standard), the chunked engine, or the hybrid streaming
       optimizer; plus a global method for ``auto_global`` and ``hpc``

Three consequences:

- ``workflow="auto"`` never selects the exact streaming solver. Exact streaming is
  reachable only through ``curve_fit_large`` (or ``fit`` without ``workflow`` on
  data at or above ``size_threshold``).
- On the ``workflow=`` path, ``size_threshold``, ``chunk_size`` and
  ``show_progress`` are ignored with a warning. Use ``memory_limit_gb`` to steer
  the memory strategy instead.
- ``fit(..., method="exact_streaming")`` is rejected with an error that points to
  ``curve_fit_large``.

For example, 10 million points with 3 parameters estimate to roughly 0.54 GB
peak (see the formulas below). On a machine with ample free RAM, ``workflow="auto"``
therefore runs an in-memory ``curve_fit``, while ``curve_fit_large`` on the same
data runs exact streaming. The estimate uses detected available memory (8 GB is
the fallback when detection fails), so the outcome depends on the machine.

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
   │ data_gb     = n_points × (n_features + 1) × 8 / 1024³           │
   │ jacobian_gb = n_points × n_params × 8 / 1024³                   │
   │ peak_gb     = data_gb + 1.3 × jacobian_gb + 0.1  (overhead, GB) │
   └─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     data_gb > threshold_gb ?  │
               └───────────────────────────────┘
                       │ YES              │ NO
                       ▼                  ▼
          ┌──────────────────┐    ┌───────────────────────────┐
          │ STREAMING        │    │ peak > 0.9×threshold_gb?  │
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
