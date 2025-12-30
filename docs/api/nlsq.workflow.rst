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

Quick Start
-----------

.. code-block:: python

   from nlsq import fit, WorkflowConfig, WorkflowTier, OptimizationGoal
   import jax.numpy as jnp
   import numpy as np


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   x = np.linspace(0, 10, 1_000_000)
   y = 2.0 * np.exp(-0.5 * x) + 0.3 + np.random.normal(0, 0.05, len(x))

   # Auto-select workflow
   popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2])

   # Use preset
   popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2], preset="robust")

   # Custom configuration
   config = WorkflowConfig(
       tier=WorkflowTier.STREAMING,
       goal=OptimizationGoal.QUALITY,
   )
   popt, pcov = fit(model, x, y, p0=[2.5, 0.6, 0.2], config=config)

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

Pre-defined workflow configurations for common use cases:

* ``'fast'``: Minimum iterations, relaxed tolerances for quick results
* ``'robust'``: Multi-start with 5 starting points for reliability
* ``'global'``: Thorough global search with 20 starting points
* ``'memory_efficient'``: Aggressive chunking and streaming fallback
* ``'quality'``: Tight tolerances with validation passes
* ``'hpc'``: PBS Pro cluster configuration
* ``'streaming'``: Tournament selection for streaming data

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

Memory Detection
----------------

Functions for detecting system memory and GPU resources.

get_total_available_memory_gb
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: nlsq.workflow.get_total_available_memory_gb
   :no-index:

get_memory_tier
~~~~~~~~~~~~~~~

.. autofunction:: nlsq.workflow.get_memory_tier
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

The workflow system uses adaptive tolerances based on dataset size:

.. list-table:: Adaptive Tolerance Values
   :header-rows: 1

   * - Dataset Size
     - Points
     - Default Tolerance
   * - TINY
     - < 1,000
     - 1e-12
   * - SMALL
     - 1,000 - 10,000
     - 1e-10
   * - MEDIUM
     - 10,000 - 100,000
     - 1e-8
   * - LARGE
     - 100,000 - 1,000,000
     - 1e-7
   * - VERY_LARGE
     - 1M - 10M
     - 1e-6
   * - MASSIVE
     - > 10M
     - 1e-5

Module Contents
---------------

.. automodule:: nlsq.core.workflow
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :exclude-members: WorkflowTier, OptimizationGoal, DatasetSizeTier, MemoryTier, WorkflowConfig, WorkflowSelector, ClusterInfo, ClusterDetector, auto_select_workflow, get_total_available_memory_gb, get_memory_tier
