nlsq.workflow
=============

Unified workflow system for automatic optimization strategy selection.

The workflow module provides a unified ``fit()`` entry point that automatically
selects the optimal fitting strategy based on dataset size and available memory.

Overview
--------

The workflow system introduces:

* **WorkflowTier**: Processing strategies (STANDARD, CHUNKED, STREAMING, STREAMING_CHECKPOINT)
* **OptimizationGoal**: Optimization objectives (FAST, ROBUST, GLOBAL, MEMORY_EFFICIENT, QUALITY)
* **WorkflowConfig**: Configuration dataclass for fine-grained control
* **WorkflowSelector**: Automatic tier selection based on dataset/memory analysis
* **WORKFLOW_PRESETS**: Named configurations for common use cases
* **ClusterDetector**: HPC cluster and multi-GPU detection

Quick Start
-----------

.. code-block:: python

   from nlsq.core.workflow import (
       WorkflowConfig,
       WorkflowTier,
       OptimizationGoal,
       auto_select_workflow,
   )
   from nlsq import curve_fit
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 10, 1_000_000)
   y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, len(x))

   # Auto-select workflow based on dataset size and available memory
   config = auto_select_workflow(n_points=len(x), n_params=3, goal=OptimizationGoal.QUALITY)

   # Use a preset
   config = WorkflowConfig.from_preset("quality")

   # Custom configuration
   config = WorkflowConfig(
       tier=WorkflowTier.STREAMING,
       goal=OptimizationGoal.QUALITY,
       enable_multistart=True,
       n_starts=20,
   )

   # Adapt tolerances based on dataset size
   adapted_config = config.with_adaptive_tolerances(n_points=len(x))

Enumerations
------------

WorkflowTier
~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.WorkflowTier
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

OptimizationGoal
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.OptimizationGoal
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DatasetSizeTier
~~~~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.DatasetSizeTier
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

MemoryTier
~~~~~~~~~~

.. autoclass:: nlsq.workflow.MemoryTier
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Configuration
-------------

WorkflowConfig
~~~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.WorkflowConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

WORKFLOW_PRESETS
~~~~~~~~~~~~~~~~

Pre-defined workflow configurations for common use cases.

**Core Presets:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Preset
     - Tier
     - Tolerance
     - Description
   * - ``standard``
     - STANDARD
     - 1e-8
     - Default curve_fit() behavior, no multi-start
   * - ``quality``
     - STANDARD
     - 1e-10
     - Highest precision with 20-point multi-start
   * - ``fast``
     - STANDARD
     - 1e-6
     - Speed-optimized, no multi-start
   * - ``large_robust``
     - CHUNKED
     - 1e-8
     - Chunked processing with 10-point multi-start
   * - ``streaming``
     - STREAMING
     - 1e-7
     - AdaptiveHybridStreamingOptimizer for huge datasets
   * - ``hpc_distributed``
     - STREAMING_CHECKPOINT
     - 1e-6
     - Multi-GPU/node HPC configuration with checkpointing
   * - ``memory_efficient``
     - STREAMING
     - 1e-7
     - Minimize memory footprint with small chunks

**Precision Presets:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Preset
     - Tier
     - Tolerance
     - Description
   * - ``precision_high``
     - STANDARD
     - 1e-10
     - Maximum numerical precision, 15 LHS starts
   * - ``precision_standard``
     - STANDARD
     - 1e-8
     - Standard precision, 10 LHS starts

**Scale Presets:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Preset
     - Tier
     - Tolerance
     - Description
   * - ``streaming_large``
     - STREAMING
     - 1e-7
     - Large-scale streaming with checkpointing enabled

**Global Optimization Presets:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Preset
     - Tier
     - Tolerance
     - Description
   * - ``global_multimodal``
     - STANDARD
     - 1e-8
     - 30 Sobol-sampled starting points for multimodal problems
   * - ``multimodal``
     - STANDARD
     - 1e-8
     - Alias for global_multimodal

**Domain-Specific Presets:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Preset
     - Tier
     - Tolerance
     - Description
   * - ``spectroscopy``
     - STANDARD
     - 1e-10
     - Peak fitting (Gaussian/Lorentzian/Voigt), 15 LHS starts
   * - ``timeseries``
     - STREAMING
     - 1e-7
     - Time series with streaming and checkpointing

**Usage:**

.. code-block:: python

   from nlsq.core.workflow import WorkflowConfig

   # Load a preset
   config = WorkflowConfig.from_preset("quality")
   print(config.gtol)  # 1e-10
   print(config.enable_multistart)  # True
   print(config.n_starts)  # 20

   # Override preset values
   config = config.with_overrides(n_starts=30)

Workflow Selection
------------------

WorkflowSelector
~~~~~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.WorkflowSelector
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

auto_select_workflow
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.workflow.auto_select_workflow
   :no-index:

Memory and Resource Detection
-----------------------------

The workflow system automatically detects available memory and GPU resources
to select the appropriate processing tier.

MemoryTier
~~~~~~~~~~

Memory availability is classified into tiers:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Tier
     - Memory Range
     - Recommended Use
   * - LOW
     - < 16 GB
     - Streaming/small chunks, prioritize memory efficiency
   * - MEDIUM
     - 16-64 GB
     - Standard workstation, moderate chunk sizes
   * - HIGH
     - 64-128 GB
     - High-memory workstation, larger chunks
   * - VERY_HIGH
     - > 128 GB
     - HPC/server, can handle large in-memory operations

.. autoclass:: nlsq.workflow.MemoryTier
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Cluster Detection
-----------------

ClusterInfo
~~~~~~~~~~~

.. autoclass:: nlsq.workflow.ClusterInfo
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ClusterDetector
~~~~~~~~~~~~~~~

.. autoclass:: nlsq.workflow.ClusterDetector
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

YAML Configuration
------------------

The workflow system supports YAML configuration files (``nlsq.yaml``):

.. code-block:: yaml

   workflow:
     goal: robust
     memory_limit_gb: 16.0
     enable_checkpointing: true
     checkpoint_dir: ./checkpoints

   tolerances:
     ftol: 1e-10
     xtol: 1e-10
     gtol: 1e-10

   cluster:
     type: pbs
     nodes: 4
     gpus_per_node: 2

Environment Variables
---------------------

Override configuration via environment variables:

* ``NLSQ_WORKFLOW_GOAL``: Set optimization goal (fast, robust, global, etc.)
* ``NLSQ_MEMORY_LIMIT_GB``: Set memory limit in GB
* ``NLSQ_CHECKPOINT_DIR``: Set checkpoint directory path
* ``NLSQ_ENABLE_CHECKPOINTING``: Enable/disable checkpointing (1/0)

Adaptive Tolerances
-------------------

The workflow system uses adaptive tolerances based on dataset size. Larger
datasets use progressively looser tolerances to balance precision with
computation time.

.. list-table:: Adaptive Tolerance Values
   :header-rows: 1
   :widths: 20 25 20 35

   * - Dataset Size
     - Points
     - Default Tolerance
     - Notes
   * - TINY
     - < 1,000
     - 1e-12
     - Maximum precision, negligible compute cost
   * - SMALL
     - 1,000 - 10,000
     - 1e-10
     - High precision, minimal overhead
   * - MEDIUM
     - 10,000 - 100,000
     - 1e-9
     - Balanced precision/performance
   * - LARGE
     - 100,000 - 1,000,000
     - 1e-8
     - Standard tolerances (NLSQ default)
   * - VERY_LARGE
     - 1M - 10M
     - 1e-7
     - Reduced precision, chunked processing
   * - HUGE
     - 10M - 100M
     - 1e-6
     - Streaming mode, practical limits
   * - MASSIVE
     - > 100M
     - 1e-5
     - Streaming with checkpoints

**Goal-Based Adjustments:**

The ``OptimizationGoal`` shifts tolerances by one tier:

- ``QUALITY``: Uses one tier tighter (e.g., LARGE dataset uses 1e-9 instead of 1e-8)
- ``FAST``: Uses one tier looser (e.g., LARGE dataset uses 1e-7 instead of 1e-8)
- ``ROBUST``/``GLOBAL``/``MEMORY_EFFICIENT``: Uses standard tolerances for dataset size

.. code-block:: python

   from nlsq.core.workflow import calculate_adaptive_tolerances, OptimizationGoal

   # 5M points with QUALITY goal
   tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.QUALITY)
   print(tols)  # {'gtol': 1e-08, 'ftol': 1e-08, 'xtol': 1e-08}

   # 5M points with FAST goal
   tols = calculate_adaptive_tolerances(5_000_000, goal=OptimizationGoal.FAST)
   print(tols)  # {'gtol': 1e-06, 'ftol': 1e-06, 'xtol': 1e-06}

Workflow Selection Matrix
-------------------------

The ``WorkflowSelector`` uses this matrix to choose the appropriate tier based
on dataset size and available memory:

.. list-table:: Workflow Selection Matrix
   :header-rows: 1
   :widths: 18 18 18 18 18 18

   * - Dataset Size
     - Low (<16GB)
     - Medium (16-64GB)
     - High (64-128GB)
     - Very High (>128GB)
     - Notes
   * - Small (<10K)
     - standard
     - standard
     - standard
     - standard+ms
     - ms = multi-start for QUALITY goal
   * - Medium (10K-1M)
     - chunked
     - standard
     - standard+ms
     - standard+ms
     - Chunking only for low memory
   * - Large (1M-10M)
     - streaming
     - chunked
     - chunked+ms
     - chunked+ms
     - Streaming for memory-constrained
   * - Huge (10M-100M)
     - stream+ckpt
     - streaming
     - chunked
     - chunked+ms
     - ckpt = checkpointing enabled
   * - Massive (>100M)
     - stream+ckpt
     - stream+ckpt
     - streaming
     - streaming+ms
     - Always uses streaming

**Legend:**

- **ms**: Multi-start enabled (for QUALITY or ROBUST goals)
- **ckpt**: Checkpointing enabled for fault tolerance

Module Contents
---------------

.. automodule:: nlsq.core.workflow
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :exclude-members: WorkflowTier, OptimizationGoal, DatasetSizeTier, MemoryTier, WorkflowConfig, WorkflowSelector, ClusterInfo, ClusterDetector, auto_select_workflow

See Also
--------

- :doc:`/explanation/workflows` - Workflow system overview
- :doc:`/howto/common_workflows` - Common workflow patterns
- :doc:`/reference/configuration` - Configuration reference
