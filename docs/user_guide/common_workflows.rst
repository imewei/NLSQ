Common Workflows
================

This page provides runnable, end-to-end YAML patterns built from the
`workflow_config_template.yaml <https://github.com/imewei/nlsq/blob/main/workflow_config_template.yaml>`_.

Start by copying the template, then replace the sections shown here. Each
example focuses on a small set of fields so you can compose them easily.

Quick single-fit workflow
-------------------------

Use this when you have a single dataset and a simple model.

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

Multi-start global search (LHS)
-------------------------------

Use this when you want robust initialization for nonconvex fits.

.. code-block:: yaml

   model:
     name: exponential_decay
     parameters:
       bounds:
         lower: [0.0, 0.0]
         upper: [10.0, 5.0]

   multistart:
     enabled: true
     sampler: lhs
     n_starts: 32

Large dataset workflow
----------------------

Use this when you need chunking and memory controls for big data.

.. code-block:: yaml

   data:
     batch_size: 1_000_000

   resources:
     memory_limit_gb: 4.0

   fitting:
     solver: cg
     max_nfev: 100

Reproducible batch runs
-----------------------

Use this when running multiple datasets in a batch or on a scheduler.

.. code-block:: yaml

   paths:
     input: ./data/batch/*.csv
     output_dir: ./runs/batch

   logging:
     level: INFO
     save_config: true

   fitting:
     solver: auto
     max_nfev: 150

Multi-dataset with per-file outputs
-----------------------------------

Use this when you want each input file to write to its own output folder.

.. code-block:: yaml

   paths:
     input: ./data/batch/*.csv
     output_dir: ./runs/{stem}

   logging:
     level: INFO
     save_config: true

   fitting:
     solver: auto
     max_nfev: 150

CLI run
-------

Use this when you want a single command to launch a workflow run.

.. code-block:: bash

   nlsq fit --config ./configs/experiment_01.yaml

Next steps
----------

- Full configuration layout: :doc:`yaml_configuration`
- Workflow overview: :doc:`workflow_overview`
- Advanced customization: :doc:`../guides/advanced_customization`
