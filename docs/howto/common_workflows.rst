Common Workflows
================

This page provides runnable, end-to-end YAML patterns built from the
`workflow_config_template.yaml <https://github.com/imewei/nlsq/blob/main/templates/workflow_config_template.yaml>`_.

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

Warm-start refinement (v0.3.6+)
-------------------------------

Use this when refining parameters from a previous fit. The 4-Layer Defense
Strategy prevents Adam warmup from overshooting when starting near the optimum.

.. code-block:: yaml

   hybrid_streaming:
     normalize: true
     warmup_iterations: 300
     gauss_newton_tol: 1e-8

     defense_layers:
       preset: strict  # strictest protection for warm-start

       layer1_warm_start:
         enabled: true
         threshold: 0.01  # 1% of data variance

       layer2_adaptive_lr:
         enabled: true
         lr_refinement: 1.0e-6  # very conservative
         lr_careful: 1.0e-5
         lr_exploration: 0.001

       layer3_cost_guard:
         enabled: true
         tolerance: 0.05  # 5% increase allowed

       layer4_step_clipping:
         enabled: true
         max_step_size: 0.1

Or configure programmatically:

.. code-block:: python

   from nlsq import fit, HybridStreamingConfig

   # Use the strict defense preset for warm-start refinement
   config = HybridStreamingConfig.defense_strict()
   popt, pcov = fit(model, x, y, p0=previous_popt, method="hybrid_streaming")

   # Monitor defense layer activations
   from nlsq import get_defense_telemetry

   telemetry = get_defense_telemetry()
   print(telemetry.get_summary())

See :doc:`../reference/configuration` for detailed configuration.

Python Script Run
-----------------

Use this when you want to run a workflow from a Python script using YAML configuration.

.. code-block:: python

   from nlsq import fit
   from nlsq.core.workflow import load_yaml_config

   # Load configuration from YAML file
   config = load_yaml_config("./configs/experiment_01.yaml")

   # Use the configuration
   popt, pcov = fit(
       model_func,
       xdata,
       ydata,
       p0=config.get("model", {}).get("parameters", {}).get("p0"),
       workflow=config.get("default_workflow", "standard"),
   )

Or use a preset directly:

.. code-block:: python

   from nlsq import fit

   # Using built-in presets (no YAML needed)
   popt, pcov = fit(model_func, xdata, ydata, p0=[2.0, 0.5], preset="robust")

   # Scientific application presets
   popt, pcov = fit(model_func, xdata, ydata, p0=[2.0, 0.5], preset="spectroscopy")
   popt, pcov = fit(model_func, xdata, ydata, p0=[2.0, 0.5], preset="kinetics")

Domain-Specific Examples
------------------------

Interactive notebooks organized by scientific domain:

**Biology:**

- `Dose-Response Curves <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/biology/dose_response.ipynb>`_ - EC50, Hill slopes
- `Enzyme Kinetics <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/biology/enzyme_kinetics.ipynb>`_ - Michaelis-Menten kinetics
- `Growth Curves <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/biology/growth_curves.ipynb>`_ - Logistic growth models

**Chemistry:**

- `Reaction Kinetics <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/chemistry/reaction_kinetics.ipynb>`_ - Rate laws and kinetics
- `Titration Curves <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/chemistry/titration_curves.ipynb>`_ - pH curves

**Physics:**

- `Damped Oscillation <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/physics/damped_oscillation.ipynb>`_ - Pendulums and resonance
- `Radioactive Decay <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/physics/radioactive_decay.ipynb>`_ - Half-lives
- `Spectroscopy Peaks <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/physics/spectroscopy_peaks.ipynb>`_ - Peak fitting

**Engineering:**

- `Sensor Calibration <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/engineering/sensor_calibration.ipynb>`_ - Calibration curves
- `Materials Characterization <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/engineering/materials_characterization.ipynb>`_ - Materials science
- `System Identification <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/04_gallery/engineering/system_identification.ipynb>`_ - System ID

**Learning Map:**

- `Learning Map <https://github.com/imewei/NLSQ/blob/main/examples/notebooks/00_learning_map.ipynb>`_ - Find the right tutorial for your needs

Next steps
----------

- Full configuration layout: :doc:`configure_yaml`
- Configuration reference: :doc:`../reference/configuration`
- Advanced options: :doc:`advanced_api`
