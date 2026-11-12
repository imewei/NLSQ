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

The CLI accepts these top-level keys (anything else is rejected with a
"did you mean" hint):

- ``data``: input file, column mapping, and loader options
- ``model``: model type (``builtin``, ``polynomial``, ``custom``) and name
- ``fitting``: ``p0``, ``bounds``, ``method``, ``ftol``/``xtol``/``gtol``,
  ``max_nfev``, ``loss``, ``absolute_sigma``, ``check_finite``
- ``export``: output files; ``nlsq fit`` errors when this section is missing
- ``metadata``: free-form run description
- ``validation``: input and result checks
- ``visualization``: plot settings
- ``hybrid_streaming``: streaming optimizer and defense layer configuration.
  Read only by the GUI; ``nlsq fit`` accepts the key but does not apply it
  (it logs a warning and fits with defaults)

Multi-start search and memory controls have no YAML equivalent. Use the
Python API instead (``curve_fit(..., multistart=True)`` and
``curve_fit_large``); see :doc:`common_workflows`.

Minimal example
---------------

.. code-block:: yaml

   data:
     input_file: ./data/experiment_01.csv
     columns: { x: 0, y: 1 }

   model:
     type: builtin
     name: exponential_decay

   fitting:
     p0: [2.0, 0.5, 0.0]
     max_nfev: 200

   export:
     results_file: ./runs/experiment_01/results.json

Run it with ``nlsq fit workflow.yaml``.

Defense layers example (GUI only)
---------------------------------

.. note::

   ``nlsq fit`` does **not** apply the ``hybrid_streaming`` section; it logs
   a warning and runs with default settings. The section is read only by the
   GUI. From Python, pass the same settings as a ``HybridStreamingConfig`` to
   ``curve_fit_large(..., strategy=Hybrid(config=config))``; see
   :doc:`common_workflows`.

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
