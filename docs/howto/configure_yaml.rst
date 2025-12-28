YAML Configuration Structure
============================

This page explains the common structure of a workflow configuration file
and points to the full template for reference.

Start with the template
-----------------------

The repository ships a full template:

- `workflow_config_template.yaml <https://github.com/imewei/nlsq/blob/main/templates/workflow_config_template.yaml>`_

Copy the template and edit only the fields you need for your run.

Common sections
---------------

Most workflows use a subset of these sections:

- ``paths``: input data locations and output directories
- ``data``: dataset-specific settings (ranges, filtering, batching)
- ``model``: model name, parameters, and bounds
- ``fitting``: solver selection, stopping criteria, and tolerances
- ``multistart``: global search options (LHS, Sobol, Halton)
- ``hybrid_streaming``: streaming optimizer and defense layer configuration (v0.3.6+)
- ``resources``: memory and device controls
- ``logging``: verbosity and log file destinations

Minimal example
---------------

.. code-block:: yaml

   paths:
     input: ./data/experiment_01.csv
     output_dir: ./runs/experiment_01

   model:
     name: exponential_decay
     parameters:
       p0: [2.0, 0.5]

   fitting:
     solver: auto
     max_nfev: 200

Defense layers example (v0.3.6+)
--------------------------------

Configure the 4-Layer Defense Strategy for warm-start refinement:

.. code-block:: yaml

   hybrid_streaming:
     normalize: true
     warmup_iterations: 300

     defense_layers:
       preset: strict  # or "relaxed", "scientific", "disabled"

       # Fine-tune individual layers (optional)
       layer1_warm_start:
         enabled: true
         threshold: 0.01

       layer2_adaptive_lr:
         enabled: true
         lr_refinement: 1.0e-6
         lr_careful: 1.0e-5
         lr_exploration: 0.001

       layer3_cost_guard:
         enabled: true
         tolerance: 0.05

       layer4_step_clipping:
         enabled: true
         max_step_size: 0.1

     telemetry:
       enabled: true
       export_format: prometheus

See :doc:`../reference/configuration` for the complete configuration reference.

Workflow options
----------------

For user-level options like loss functions, callbacks, and solver choices,
see :doc:`../reference/configuration`.

Advanced customization
----------------------

If you need programmatic workflow construction or custom models, start
with :doc:`../developer/index`.
