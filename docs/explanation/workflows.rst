Workflow System Overview
========================

NLSQ provides automatic workflow selection based on memory constraints and dataset
characteristics. The system analyzes available memory and data size to choose the
optimal fitting strategy, preventing out-of-memory errors while maximizing performance.

.. versionchanged:: 0.5.5
   The tier-based workflow system was replaced with a unified memory-based approach.
   ``MemoryBudgetSelector`` replaces ``auto_select_workflow()``, and strategy selection
   is now driven entirely by memory budget computation.

Unified Memory-Based Strategy
-----------------------------

The memory-based approach uses a simple decision tree based on memory budget:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                    MEMORY BUDGET COMPUTATION                     │
   ├─────────────────────────────────────────────────────────────────┤
   │ available_gb = min(cpu_available, gpu_available_if_used)        │
   │ threshold_gb = available_gb × 0.75  (safety factor)             │
   │                                                                  │
   │ # Memory estimates (float64 = 8 bytes)                          │
   │ data_gb     = n_points × (n_features + 1) × 8 / 1e9            │
   │ jacobian_gb = n_points × n_params × 8 / 1e9    ← THE BIG ONE   │
   │ solver_gb   = n_params² × 8 / 1e9 + svd_overhead               │
   │ peak_gb     = data_gb + 1.3 × jacobian_gb + solver_gb          │
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

Memory Estimation Details
~~~~~~~~~~~~~~~~~~~~~~~~~

The system estimates memory requirements for each component:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Component
     - Formula
     - Example (10M pts, 10 params)
   * - Data (x, y)
     - n × (features + 1) × 8
     - 160 MB
   * - Jacobian
     - n × p × 8
     - 800 MB
   * - J\ :sup:`T`\ J
     - p² × 8
     - 0.8 KB
   * - SVD working
     - ~0.3 × jacobian
     - 240 MB
   * - **Peak**
     - data + 1.3×J + solver
     - **~1.3 GB**

The Jacobian matrix dominates memory usage for most problems. The 1.3× multiplier
accounts for temporary arrays during computation.

Automatic Memory-Based Selection
--------------------------------

When using ``workflow="auto"`` or ``method="auto"``, the ``MemoryBudgetSelector``
automatically picks the best strategy:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Strategy
     - When Selected
   * - **streaming**
     - Data alone exceeds 75% of available memory
   * - **chunked**
     - Data fits, but peak memory (data + Jacobian) exceeds threshold
   * - **standard**
     - Everything fits comfortably in memory

.. code-block:: python

   from nlsq import fit, curve_fit
   from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector

   # Automatic selection via fit()
   result = fit(model, x, y, p0=[1, 2], workflow="auto")

   # Automatic selection via curve_fit()
   popt, pcov = curve_fit(model, x, y, p0=[1, 2], method="auto")

   # Direct use of MemoryBudgetSelector
   selector = MemoryBudgetSelector(safety_factor=0.75)
   strategy, config = selector.select(
       n_points=10_000_000,
       n_params=10,
       memory_limit_gb=16.0,  # Optional override
   )
   print(f"Selected: {strategy}")  # "streaming", "chunked", or "standard"

   # Inspect memory budget
   budget = MemoryBudget.compute(n_points=10_000_000, n_params=10)
   print(f"Peak memory: {budget.peak_gb:.2f} GB")
   print(f"Fits in memory: {budget.fits_in_memory}")

Named Workflow Presets
----------------------

The ``fit()`` function accepts named presets for common use cases:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Workflow
     - Description
     - Use Case
   * - ``"auto"``
     - Memory-based automatic selection
     - Default - picks best strategy based on dataset size and memory
   * - ``"standard"``
     - Standard curve_fit with default tolerances (1e-8)
     - Small to medium datasets
   * - ``"quality"``
     - Multi-start (20 starts) with tight tolerances (1e-10)
     - When accuracy matters most
   * - ``"fast"``
     - Looser tolerances (1e-6), no multi-start
     - Quick exploratory fits
   * - ``"large_robust"``
     - Chunked processing with multi-start (10 starts)
     - Large datasets needing robustness
   * - ``"streaming"``
     - AdaptiveHybridStreamingOptimizer with tolerances (1e-7)
     - Huge datasets (10M+ points)
   * - ``"hpc_distributed"``
     - Multi-GPU/node with checkpoints
     - HPC cluster deployments

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: python

   from nlsq import fit, curve_fit

   # Automatic selection (recommended)
   result = fit(model, x, y, p0=[1, 2], workflow="auto")

   # Named preset
   result = fit(model, x, y, p0=[1, 2], workflow="quality")

   # With method='auto' in curve_fit
   popt, pcov = curve_fit(model, x, y, p0=[1, 2], method="auto")

   # Direct memory control
   result = fit(model, x, y, p0=[1, 2], workflow="auto", memory_limit_gb=8.0)

Optimization Goals
------------------

The ``OptimizationGoal`` enum controls tolerance scaling:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Goal
     - Description
   * - **FAST**
     - Prioritize speed. Uses one tier looser tolerances, skips multi-start.
       Best for quick exploration or well-conditioned problems.
   * - **ROBUST**
     - Standard tolerances with multi-start for better global optimum.
       Uses ``MultiStartOrchestrator`` for reliability. Best for production use.
   * - **GLOBAL**
     - Synonym for ROBUST. Emphasizes global optimization.
   * - **MEMORY_EFFICIENT**
     - Minimize memory usage with standard tolerances.
       Prioritizes streaming/chunking with smaller chunk sizes.
   * - **QUALITY**
     - Highest precision as TOP PRIORITY. Uses one tier tighter tolerances,
       enables multi-start, runs validation passes. Best for publication-quality results.

.. code-block:: python

   from nlsq import fit
   from nlsq.core.workflow import OptimizationGoal

   # Quality-focused fit
   result = fit(model, x, y, p0=[1, 2], goal=OptimizationGoal.QUALITY)

   # Speed-focused fit
   result = fit(model, x, y, p0=[1, 2], goal=OptimizationGoal.FAST)

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

Defense presets for common scenarios:

.. code-block:: python

   from nlsq import HybridStreamingConfig

   # For warm-start refinement (strictest)
   config = HybridStreamingConfig.defense_strict()

   # For exploration (more aggressive learning)
   config = HybridStreamingConfig.defense_relaxed()

   # For production scientific computing
   config = HybridStreamingConfig.scientific_default()

   # To disable (pre-0.3.6 behavior)
   config = HybridStreamingConfig.defense_disabled()

Where to go next
----------------

- API reference: :doc:`../api/nlsq.workflow`
- Configuration options: :doc:`../reference/configuration`
- Common workflow patterns: :doc:`../howto/common_workflows`
- Large dataset handling: :doc:`../howto/handle_large_data`
