YAML Configuration Structure
============================

This page explains the common structure of a workflow configuration file
and points to the full template for reference.

Start with the template
-----------------------

The repository ships a full template:

- `workflow_config_template.yaml <https://github.com/imewei/nlsq/blob/main/workflow_config_template.yaml>`_

Copy the template and edit only the fields you need for your run.

Common sections
---------------

Most workflows use a subset of these sections:

- ``paths``: input data locations and output directories
- ``data``: dataset-specific settings (ranges, filtering, batching)
- ``model``: model name, parameters, and bounds
- ``fitting``: solver selection, stopping criteria, and tolerances
- ``multistart``: global search options (LHS, Sobol, Halton)
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

Workflow options
----------------

For user-level options like loss functions, callbacks, and solver choices,
see :doc:`../guides/workflow_options`.

Advanced customization
----------------------

If you need programmatic workflow construction or custom models, start
with :doc:`../advanced/index`.
