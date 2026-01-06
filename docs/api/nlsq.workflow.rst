nlsq.workflow
=============

Memory-based workflow system for automatic optimization strategy selection.

.. versionchanged:: 0.5.5
   The tier-based workflow system was replaced with a unified memory-based approach.
   ``MemoryBudgetSelector`` replaces ``auto_select_workflow()``, and strategy selection
   is now driven entirely by memory budget computation.

Overview
--------

The workflow module provides:

* **MemoryBudget**: Dataclass for computing and storing memory estimates
* **MemoryBudgetSelector**: Automatic strategy selection based on memory analysis
* **OptimizationGoal**: Optimization objectives (FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY)
* **calculate_adaptive_tolerances**: Dataset-size-aware tolerance computation

Quick Start
-----------

.. code-block:: python

   from nlsq import fit, curve_fit
   from nlsq.core.workflow import MemoryBudget, MemoryBudgetSelector, OptimizationGoal
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 10, 1_000_000)
   y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, len(x))

   # Automatic selection via fit() (recommended)
   result = fit(model, x, y, p0=[1, 1, 0], workflow="auto")

   # Automatic selection via curve_fit()
   popt, pcov = curve_fit(model, x, y, p0=[1, 1, 0], method="auto")

   # Direct use of MemoryBudgetSelector
   selector = MemoryBudgetSelector(safety_factor=0.75)
   strategy, config = selector.select(
       n_points=len(x),
       n_params=3,
       memory_limit_gb=16.0,  # Optional override
   )
   print(f"Selected strategy: {strategy}")

   # Inspect memory budget
   budget = MemoryBudget.compute(n_points=len(x), n_params=3)
   print(f"Peak memory: {budget.peak_gb:.2f} GB")
   print(f"Fits in memory: {budget.fits_in_memory}")

Memory Budget Classes
---------------------

MemoryBudget
~~~~~~~~~~~~

.. autoclass:: nlsq.core.workflow.MemoryBudget
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

**Fields:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - Description
   * - ``available_gb``
     - Total available memory (CPU or GPU) in GB
   * - ``threshold_gb``
     - Safe threshold (available × safety_factor)
   * - ``data_gb``
     - Estimated memory for data arrays (x, y)
   * - ``jacobian_gb``
     - Estimated memory for Jacobian matrix
   * - ``peak_gb``
     - Total peak memory estimate

**Computed Properties:**

* ``fits_in_memory``: True if peak_gb <= threshold_gb
* ``data_fits``: True if data_gb <= threshold_gb

MemoryBudgetSelector
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.core.workflow.MemoryBudgetSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

**Strategy Selection Logic:**

.. code-block:: text

   if data_gb > threshold_gb:
       return "streaming"  # Data too large for memory
   elif peak_gb > threshold_gb:
       return "chunked"    # Jacobian too large, chunk the computation
   else:
       return "standard"   # Everything fits, use direct curve_fit()

Enumerations
------------

OptimizationGoal
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.core.workflow.OptimizationGoal
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Goal
     - Description
   * - **FAST**
     - Prioritize speed. Uses one tier looser tolerances, skips multi-start.
   * - **ROBUST**
     - Standard tolerances with multi-start for better global optimum.
   * - **GLOBAL**
     - Synonym for ROBUST. Emphasizes global optimization.
   * - **MEMORY_EFFICIENT**
     - Minimize memory usage with standard tolerances.
   * - **QUALITY**
     - Highest precision. Uses one tier tighter tolerances, enables multi-start.

Named Workflow Presets
----------------------

The ``fit()`` function accepts named presets via the ``workflow`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Preset
     - Strategy
     - Tolerance
     - Description
   * - ``"auto"``
     - Memory-based
     - Adaptive
     - Automatic selection based on memory budget
   * - ``"standard"``
     - standard
     - 1e-8
     - Default curve_fit() behavior, no multi-start
   * - ``"quality"``
     - standard
     - 1e-10
     - Highest precision with 20-point multi-start
   * - ``"fast"``
     - standard
     - 1e-6
     - Speed-optimized, no multi-start
   * - ``"large_robust"``
     - chunked
     - 1e-8
     - Chunked processing with 10-point multi-start
   * - ``"streaming"``
     - streaming
     - 1e-7
     - AdaptiveHybridStreamingOptimizer for huge datasets
   * - ``"hpc_distributed"``
     - streaming
     - 1e-6
     - Multi-GPU/node HPC configuration with checkpointing

**Usage:**

.. code-block:: python

   from nlsq import fit

   # Use automatic memory-based selection
   result = fit(model, x, y, p0=[1, 1, 0], workflow="auto")

   # Use a named preset
   result = fit(model, x, y, p0=[1, 1, 0], workflow="quality")

   # Override memory detection
   result = fit(model, x, y, p0=[1, 1, 0], workflow="auto", memory_limit_gb=8.0)

Adaptive Tolerances
-------------------

The workflow system uses adaptive tolerances based on dataset size:

.. list-table::
   :header-rows: 1
   :widths: 20 25 20 35

   * - Dataset Size
     - Points
     - Default Tolerance
     - Notes
   * - TINY
     - < 1,000
     - 1e-12
     - Maximum precision
   * - SMALL
     - 1,000 - 10,000
     - 1e-10
     - High precision
   * - MEDIUM
     - 10,000 - 100,000
     - 1e-9
     - Balanced
   * - LARGE
     - 100,000 - 1,000,000
     - 1e-8
     - Standard (NLSQ default)
   * - VERY_LARGE
     - 1M - 10M
     - 1e-7
     - Reduced precision
   * - HUGE
     - 10M - 100M
     - 1e-6
     - Streaming mode
   * - MASSIVE
     - > 100M
     - 1e-5
     - Streaming with checkpoints

**Goal-Based Adjustments:**

* ``QUALITY``: Uses one tier tighter tolerances
* ``FAST``: Uses one tier looser tolerances
* ``ROBUST``/``GLOBAL``/``MEMORY_EFFICIENT``: Uses standard tolerances

.. code-block:: python

   from nlsq.core.workflow import calculate_adaptive_tolerances, OptimizationGoal

   # 5M points with QUALITY goal
   tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.QUALITY)
   print(tols)  # {'gtol': 1e-08, 'ftol': 1e-08, 'xtol': 1e-08}

   # 5M points with FAST goal
   tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.FAST)
   print(tols)  # {'gtol': 1e-06, 'ftol': 1e-06, 'xtol': 1e-06}

Memory Estimation Details
-------------------------

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

The Jacobian matrix dominates memory usage for most problems.

Utility Functions
-----------------

calculate_adaptive_tolerances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.core.workflow.calculate_adaptive_tolerances
   :no-index:

create_checkpoint_directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.core.workflow.create_checkpoint_directory
   :no-index:

Module Contents
---------------

.. automodule:: nlsq.core.workflow
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :exclude-members: MemoryBudget, MemoryBudgetSelector, OptimizationGoal, calculate_adaptive_tolerances, create_checkpoint_directory

See Also
--------

- :doc:`/explanation/workflows` - Workflow system overview
- :doc:`/howto/common_workflows` - Common workflow patterns
- :doc:`/reference/configuration` - Configuration reference
