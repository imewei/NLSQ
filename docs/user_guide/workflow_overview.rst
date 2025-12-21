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
