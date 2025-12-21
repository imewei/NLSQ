Workflow System Overview
========================

The workflow system lets you run an end-to-end analysis with a single
configuration file. It provides a consistent way to define data inputs,
models, fitting options, and outputs without writing custom pipeline code.

This page is high-level by design. For the exact configuration fields, see
:doc:`yaml_configuration`.

Why use the workflow system?
----------------------------

- Reproducible runs driven by versioned configuration
- Consistent defaults across team members and machines
- Clear separation of data, model, fitting, and outputs
- Minimal glue code for batch or automated execution

Typical workflow lifecycle
--------------------------

1. Prepare a YAML configuration file for your dataset and model.
2. Run the workflow from the CLI or a job runner.
3. Inspect logs and result artifacts.
4. Iterate on configuration parameters as needed.

Where to go next
----------------

- Configuration layout and examples: :doc:`yaml_configuration`
- Practical workflow options: :doc:`../guides/workflow_options`
- Results and outputs: :doc:`results_outputs`

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
