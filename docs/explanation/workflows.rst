Workflow System Overview
========================

The workflow system lets you run an end-to-end analysis with a single
configuration file. It provides a consistent way to define data inputs,
models, fitting options, and outputs without writing custom pipeline code.

This page is high-level by design. For the exact configuration fields, see
:doc:`../howto/configure_yaml`.

Why use the workflow system?
----------------------------

- **Automatic optimization**: Selects the best fitting strategy based on dataset size and memory
- Reproducible runs driven by versioned configuration
- Consistent defaults across team members and machines
- Clear separation of data, model, fitting, and outputs
- Minimal glue code for batch or automated execution
- **Built-in numerical safeguards** via the 4-Layer Defense Strategy (v0.3.6+)

Workflow Tiers
--------------

NLSQ automatically selects one of four processing tiers based on your dataset
size and available memory:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Tier
     - Dataset Size
     - Description
   * - **STANDARD**
     - < 10K points
     - Standard ``curve_fit()`` for small datasets that fit in memory.
       Uses O(N) memory where N is number of data points.
   * - **CHUNKED**
     - 10K - 10M points
     - ``LargeDatasetFitter`` with automatic chunking. Processes data in
       sequential chunks with O(chunk_size) memory complexity.
   * - **STREAMING**
     - 10M - 100M points
     - ``AdaptiveHybridStreamingOptimizer`` with O(batch_size) memory.
       Uses mini-batch gradient descent for memory efficiency.
   * - **STREAMING_CHECKPOINT**
     - > 100M points
     - Streaming with automatic checkpointing for massive datasets.
       Enables resume capability for multi-hour fits.

Optimization Goals
------------------

The ``OptimizationGoal`` enum controls the optimization priority:

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

Available Presets
-----------------

NLSQ provides pre-configured workflow presets for common use cases:

**Core Presets:**

- ``standard`` - Default curve_fit() behavior, no multi-start
- ``quality`` - Highest precision (1e-10 tolerance, 20-point multi-start)
- ``fast`` - Speed-optimized (1e-6 tolerance, no multi-start)
- ``large_robust`` - Chunked processing with 10-point multi-start
- ``streaming`` - AdaptiveHybridStreamingOptimizer for huge datasets
- ``hpc_distributed`` - Multi-GPU/node HPC configuration with checkpointing
- ``memory_efficient`` - Minimize memory footprint with small chunks

**Specialized Presets:**

- ``precision_high`` / ``precision_standard`` - Precision-focused configurations
- ``global_multimodal`` / ``multimodal`` - 30 Sobol-sampled starts for multimodal problems
- ``spectroscopy`` - Peak fitting (Gaussian/Lorentzian/Voigt)
- ``timeseries`` - Time series with streaming and checkpointing

.. code-block:: python

   from nlsq.core.workflow import WorkflowConfig

   # Load a preset
   config = WorkflowConfig.from_preset("quality")

   # Check preset settings
   print(config.gtol)  # 1e-10
   print(config.enable_multistart)  # True
   print(config.n_starts)  # 20

Automatic Workflow Selection
----------------------------

Use ``auto_select_workflow()`` to automatically choose the best configuration
based on your dataset size and available system memory:

.. code-block:: python

   from nlsq.core.workflow import auto_select_workflow, OptimizationGoal

   # Auto-select based on dataset characteristics
   config = auto_select_workflow(
       n_points=5_000_000,
       n_params=5,
       goal=OptimizationGoal.QUALITY,
   )

   # The returned config is ready to use
   print(config)  # HybridStreamingConfig or LDMemoryConfig

Typical workflow lifecycle
--------------------------

1. Prepare a YAML configuration file for your dataset and model.
2. Run the workflow from the CLI or a job runner.
3. Inspect logs and result artifacts.
4. Iterate on configuration parameters as needed.

4-Layer Defense Strategy
------------------------

Starting in v0.3.6, all workflows using ``hybrid_streaming`` or
``AdaptiveHybridStreamingOptimizer`` include a 4-layer defense against L-BFGS
warmup divergence. This is particularly important for **warm-start refinement**
scenarios where initial parameters are already near optimal.

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

See :doc:`../reference/configuration` for detailed configuration options.

Where to go next
----------------

- API reference: :doc:`../api/nlsq.workflow`
- Configuration layout and examples: :doc:`../howto/configure_yaml`
- Configuration options: :doc:`../reference/configuration`
- Common workflow patterns: :doc:`../howto/common_workflows`

Interactive Notebooks
---------------------

Hands-on tutorials for the workflow system:

- `fit() Quickstart <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/01_fit_quickstart.ipynb>`_ - Using fit() with automatic workflow selection
- `Workflow Tiers <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/02_workflow_tiers.ipynb>`_ - Understanding the four workflow tiers
- `Optimization Goals <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/03_optimization_goals.ipynb>`_ - All 5 OptimizationGoal values
- `Workflow Presets <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/04_workflow_presets.ipynb>`_ - Using built-in presets
- `YAML Configuration <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/05_yaml_configuration.ipynb>`_ - Configuration files
- `Auto Selection <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/06_auto_selection.ipynb>`_ - Automatic workflow selection
- `HPC and Checkpointing <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/08_workflow_system/07_hpc_and_checkpointing.ipynb>`_ - Cluster computing and fault tolerance
- `Defense Layers Demo <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/02_features/defense_layers_demo.ipynb>`_ - 4-layer defense strategy for warm-start refinement
