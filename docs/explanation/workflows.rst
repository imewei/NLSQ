Workflow System Overview
========================

The workflow system lets you run an end-to-end analysis with a single
configuration file. It provides a consistent way to define data inputs,
models, fitting options, and outputs without writing custom pipeline code.

This page is high-level by design. For the exact configuration fields, see
:doc:`../howto/configure_yaml`.

Why use the workflow system?
----------------------------

- Reproducible runs driven by versioned configuration
- Consistent defaults across team members and machines
- Clear separation of data, model, fitting, and outputs
- Minimal glue code for batch or automated execution
- **Built-in numerical safeguards** via the 4-Layer Defense Strategy (v0.3.6+)

Typical workflow lifecycle
--------------------------

1. Prepare a YAML configuration file for your dataset and model.
2. Run the workflow from the CLI or a job runner.
3. Inspect logs and result artifacts.
4. Iterate on configuration parameters as needed.

4-Layer Defense Strategy
------------------------

Starting in v0.3.6, all workflows using ``hybrid_streaming`` or
``AdaptiveHybridStreamingOptimizer`` include a 4-layer defense against Adam
warmup divergence. This is particularly important for **warm-start refinement**
scenarios where initial parameters are already near optimal.

The layers activate automatically:

1. **Warm Start Detection**: Skips warmup if initial loss < 1% of data variance
2. **Adaptive Learning Rate**: Scales LR based on fit quality (1e-6 to 0.001)
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

- Configuration layout and examples: :doc:`../howto/configure_yaml`
- Configuration options: :doc:`../reference/configuration`

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
