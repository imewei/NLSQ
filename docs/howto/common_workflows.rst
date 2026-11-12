Common Workflows
================

This page provides runnable, end-to-end YAML patterns built from the
`workflow_config_template.yaml <https://github.com/imewei/nlsq/blob/main/templates/workflow_config_template.yaml>`_.

Start by copying the template, then replace the sections shown here. Each
example focuses on a small set of fields so you can compose them easily.

Quick single-fit workflow
-------------------------

Use this when you have a single dataset and a simple model. Run it with
``nlsq fit workflow.yaml``.

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

Bounded fit
-----------

Use this to constrain parameters. ``fitting.bounds`` takes ``lower`` and
``upper`` lists, one entry per parameter.

.. code-block:: yaml

   data:
     input_file: ./data/experiment_01.csv
     columns: { x: 0, y: 1 }

   model:
     type: builtin
     name: exponential_decay

   fitting:
     p0: [2.0, 0.5, 0.0]
     bounds:
       lower: [0.0, 0.0, -1.0]
       upper: [10.0, 5.0, 1.0]

   export:
     results_file: ./runs/experiment_01/results.json

Multi-start global search (LHS)
-------------------------------

Multi-start is not available from YAML. Use the Python API for robust
initialization of nonconvex fits (bounds are required for the sampler):

.. code-block:: python

   import jax.numpy as jnp
   from nlsq import curve_fit


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   popt, pcov = curve_fit(
       model,
       x,
       y,
       bounds=([0.0, 0.0], [10.0, 5.0]),
       multistart=True,
       n_starts=32,
       sampler="lhs",
   )

Large dataset workflow
----------------------

Chunking and memory limits are not available from YAML. Use
``curve_fit_large`` from Python (see :doc:`handle_large_data`):

.. code-block:: python

   from nlsq import curve_fit_large

   popt, pcov = curve_fit_large(
       model, x, y, p0=[2.0, 0.5], memory_limit_gb=4.0, max_nfev=100
   )

Reproducible batch runs
-----------------------

``nlsq batch`` takes several workflow files, one per dataset; it does not
expand glob patterns or placeholders inside a single YAML. Give each file its
own ``data.input_file`` and ``export.results_file`` (batch warns when two
workflows write the same output file):

.. code-block:: bash

   nlsq batch runs/exp_01.yaml runs/exp_02.yaml --summary batch_summary.json

Each workflow file is an ordinary single-fit configuration:

.. code-block:: yaml

   # runs/exp_01.yaml
   data:
     input_file: ./data/batch/exp_01.csv
     columns: { x: 0, y: 1 }

   model:
     type: builtin
     name: exponential_decay

   fitting:
     max_nfev: 150

   export:
     results_file: ./runs/exp_01/results.json

Warm-start refinement
---------------------

Use this when refining parameters from a previous fit. The 4-Layer Defense
Strategy prevents L-BFGS warmup from overshooting when starting near the optimum.

.. note::

   This is a Python-only workflow. ``nlsq fit`` does not apply a
   ``hybrid_streaming`` YAML section (it is read only by the GUI).

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np

   from nlsq import HybridStreamingConfig, curve_fit_large, get_defense_telemetry
   from nlsq.streaming import Hybrid


   def model(x, a, b):
       return a * jnp.exp(-b * x)


   x = np.linspace(0, 4, 5000)
   y = 2.5 * np.exp(-1.3 * x) + np.random.default_rng(0).normal(0, 0.01, x.size)
   previous_popt = np.array([2.4, 1.25])

   # Use the strict defense preset for warm-start refinement
   config = HybridStreamingConfig.defense_strict()
   popt, pcov = curve_fit_large(
       model, x, y, p0=previous_popt, strategy=Hybrid(config=config)
   )

   # Monitor defense layer activations
   telemetry = get_defense_telemetry()
   print(telemetry.get_summary())

See :doc:`../reference/configuration` for detailed configuration.

Python Script Run
-----------------

Use this when you want to run a workflow from a Python script using built-in presets.

.. code-block:: python

   from nlsq import fit

   # Using built-in presets (no YAML needed): "fast", "robust", "global",
   # "streaming", "large" -- no domain-specific presets exist yet.
   popt, pcov = fit(model_func, xdata, ydata, p0=[2.0, 0.5], preset="robust")

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
