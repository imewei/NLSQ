CLI Reference
=============

NLSQ provides a command-line interface for curve fitting workflows, batch processing,
and system information.

Entry Points
------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``nlsq``
     - Main CLI entry point
   * - ``nlsq-gui``
     - Direct Qt GUI launcher

Global Options
--------------

.. code-block:: text

   nlsq --version         Show version and exit
   nlsq -v, --verbose     Enable verbose output
   nlsq --help            Show help and available commands

Commands
--------

nlsq gui
~~~~~~~~

Launch the interactive Qt desktop GUI for visual curve fitting.

.. code-block:: bash

   nlsq gui

The GUI provides a 5-page workflow: Data Loading → Model Selection → Fitting Options
→ Results → Export. Requires the ``gui`` extra (PySide6, pyqtgraph), which the
core install does not include:

.. code-block:: bash

   pip install "nlsq[gui]"

nlsq fit
~~~~~~~~

Execute single curve fit from a YAML workflow configuration.

**Syntax:**

.. code-block:: text

   nlsq fit <workflow.yaml> [OPTIONS]

   Arguments:
     workflow.yaml        Path to workflow YAML configuration file

   Options:
     --style PRESET       Override visualization style (publication, nature, science, presentation, minimal)
     -o, --output FILE    Override export.results_file path
     --stdout             Output results as JSON to stdout (for piping)

**Examples:**

.. code-block:: bash

   # Basic fit
   nlsq fit workflow.yaml

   # Generate Nature-style figure
   nlsq fit workflow.yaml --style nature

   # Override output file
   nlsq fit workflow.yaml --output results.json

   # Output to stdout for piping
   nlsq fit workflow.yaml --stdout

   # Verbose mode
   nlsq --verbose fit workflow.yaml

nlsq batch
~~~~~~~~~~

Execute parallel batch fitting from multiple YAML workflow files.

**Syntax:**

.. code-block:: text

   nlsq batch <files...> [OPTIONS]

   Arguments:
     files...             Paths to workflow YAML configuration files

   Options:
     -s, --summary FILE      Path for aggregate summary file
     -w, --workers N         Maximum parallel workers (default: auto-detect)
     --continue-on-error     Continue processing on individual failures (default: true)

**Examples:**

.. code-block:: bash

   # Multiple files
   nlsq batch w1.yaml w2.yaml w3.yaml

   # Using shell glob expansion
   nlsq batch configs/*.yaml

   # With worker limit and summary file
   nlsq batch configs/*.yaml --workers 4 --summary batch_results.json

   # Verbose mode
   nlsq --verbose batch *.yaml

nlsq info
~~~~~~~~~

Display system and environment information including NLSQ version, Python version,
JAX backend, GPU info, and available builtin models.

**Syntax:**

.. code-block:: bash

   nlsq info
   nlsq --verbose info    # More detailed output

**Sample Output:**

.. code-block:: text

   NLSQ Information
   ================
   Version: 0.7.6
   Python: 3.12.0
   JAX: 0.8.2
   Device: cuda:0 (NVIDIA RTX 4090)
   Memory: 24.0 GB available

   Builtin Models:
   - linear, exponential_decay, exponential_growth
   - gaussian, sigmoid, power_law, polynomial

nlsq config
~~~~~~~~~~~

Copy configuration templates to current directory to start a new project.

**Syntax:**

.. code-block:: text

   nlsq config [OPTIONS]

   Options:
     --workflow          Copy only the workflow configuration template (workflow_config.yaml)
     --model             Copy only the custom model template (custom_model.py)
     -o, --output FILE   Custom output filename (only valid with --workflow or --model)
     -f, --force         Overwrite existing files without prompting

**Examples:**

.. code-block:: bash

   # Copy both templates
   nlsq config

   # Workflow template only
   nlsq config --workflow

   # Model template only
   nlsq config --model

   # Custom filename for model template
   nlsq config --model -o my_model.py

   # Force overwrite existing files
   nlsq config -f

Quick Reference
---------------

.. code-block:: bash

   nlsq                    # Show help
   nlsq gui                # Launch Qt GUI
   nlsq fit w.yaml         # Single fit
   nlsq batch *.yaml       # Batch fit
   nlsq info               # System info
   nlsq config             # Get templates

Workflow Configuration
----------------------

The ``nlsq fit`` and ``nlsq batch`` commands use YAML workflow configurations.
See :doc:`configuration` for the full configuration reference.

**Minimal Example:**

.. code-block:: yaml

   data:
     input_file: "data/experiment.csv"
     columns: { x: 0, y: 1 }
   model:
     type: "builtin"
     name: "exponential_decay"
   export:
     results_file: "results.json"

**Fitting settings:**

The CLI runner calls ``curve_fit``. Set initial guesses, bounds, tolerances and
evaluation limits directly under ``fitting``:

.. code-block:: yaml

   fitting:
     p0: [2.0, 0.5, 0.0]
     bounds:
       lower: [0.0, 0.0, -1.0]
       upper: [10.0, 5.0, 1.0]
     ftol: 1.0e-8
     xtol: 1.0e-8
     gtol: 1.0e-8
     max_nfev: 200

``fitting.termination`` and ``model.parameters`` do not configure CLI fits.
When ``fitting.p0`` is omitted or ``auto``, the runner tries the model's
``estimate_p0`` helper. Omitted bounds use the model's ``bounds`` helper when
available. ``model.auto_p0`` and ``model.auto_bounds`` do not control this behavior.
Global workflow presets, multi-start, CMA-ES and distributed execution are
available through the Python API; the CLI does not dispatch these workflows
from ``workflow`` or ``global_optimization`` YAML keys.

Custom Models and the Security Validator
----------------------------------------

``model.type: custom`` loads a Python file and runs it. Before it runs, the file
is scanned with an AST-based validator that rejects code which could execute
other code, start processes, reach the network or write files. Blocked
categories include:

* ``exec``/``eval``/``compile`` and ``__import__``, reflection helpers
  (``getattr``, ``setattr``); any ``_``-prefixed attribute except ``__name__``,
  ``__doc__`` and ``__version__`` (so a model class using ``self._k`` needs
  ``trusted: true``); and attributes named like blocked modules (for example
  ``.string``, ``.profiler``);
* process, thread and interpreter control (``os``, ``subprocess``,
  ``multiprocessing``, ``threading``, ``signal``, ``cProfile``, ``pdb`` ...);
* network modules (``socket``, ``urllib``, ``http``, ``requests`` ...);
* unsafe deserialisation (``pickle``, ``yaml.unsafe_load``,
  ``pandas.read_pickle``, ``numpy.load`` with ``allow_pickle``);
* file writes (``open`` in a write mode, ``numpy.save*``, ``tofile`` ...).

The validator is a **denylist, not a sandbox**. It reduces accidental and
casual abuse but cannot prove an arbitrary file is safe, so only run custom
model files you wrote or fully trust.

A file that is rejected, but that you trust, can bypass the scan with the
``trusted`` key in the workflow configuration:

.. code-block:: yaml

   model:
     type: "custom"
     path: "my_model.py"
     function: "my_model"
     trusted: true   # must be a YAML boolean; skips the security scan

With ``trusted: true`` the file runs with the full privileges of the user
running ``nlsq``, with no static checks. Use it only for models you authored or
have reviewed line by line, never for files received from others. A quoted
value such as ``"false"`` is rejected rather than treated as a boolean.

Strict allowlist mode (opt-in)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because a denylist cannot be made complete, a stricter **allowlist** mode is
available. It runs every denylist check and additionally rejects any import or
attribute chain outside a short list of numeric modules. Enable it per model:

.. code-block:: yaml

   model:
     type: "custom"
     path: "my_model.py"
     function: "my_model"
     strict: true   # must be a YAML boolean

or globally with the environment variable ``NLSQ_STRICT_MODELS=1``. In strict
mode only these modules (and their submodules) may be imported:
``math``, ``cmath``, ``functools``, ``typing``, ``collections.abc``,
``dataclasses``, ``numpy``, ``jax``, ``jax.numpy``, ``jax.scipy``, ``jax.lax``,
``scipy.special``, ``scipy.stats``, ``scipy.constants`` (plus ``__future__``).
Relative and wildcard imports are rejected, so is ``import scipy`` on its own,
and attribute chains are checked against their full dotted path
(``jax.config``, ``jax.experimental``, ``numpy.load``, ``scipy.io`` ... are
blocked). ``open``, ``eval``, ``exec``, ``compile``, ``getattr`` and similar
names are rejected wherever they appear, and module objects may not be passed
around as values (use ``jnp.sin``, not ``xp=jnp``). Violation messages begin
with ``strict allowlist``. ``trusted: true`` still bypasses strict mode (a
warning is logged).

Strict mode narrows the attack surface but, like the default validator, **is
not a sandbox**. It is opt-in for one release to measure how many real models
it rejects; it is planned to become the default after that release, with
``trusted: true`` as the escape hatch for models that import other packages.

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Code
     - Meaning
   * - 0
     - Success
   * - 1
     - General error (configuration, data loading, model, or fitting error), or a fit that did not converge
   * - 130
     - ``nlsq batch`` interrupted (Ctrl-C); completed results are still reported

Scripting Examples
------------------

**Batch Processing with Shell:**

.. code-block:: bash

   #!/bin/bash
   for config in configs/*.yaml; do
       nlsq fit "$config" || echo "Failed: $config"
   done

**Pipeline Integration:**

.. code-block:: bash

   # Fit and extract parameter with jq
   nlsq fit workflow.yaml --stdout | jq '.popt[0]'

**Parallel Batch with Custom Summary:**

.. code-block:: bash

   nlsq batch experiments/*.yaml \
       --workers 8 \
       --summary results/batch_summary.json

JSON Output Format
------------------

When using ``--stdout``, results are output as JSON:

.. code-block:: javascript

   {
     "popt": [2.5, 10.2, 0.05],
     "pcov": [[0.01, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.0004]],
     "success": true,
     "message": "Optimization converged",
     "nfev": 42,
     "cost": 0.0025
   }

See Also
--------

- :doc:`configuration` - Full configuration options reference
- :doc:`/howto/configure_yaml` - How to write workflow YAML files
- :doc:`/howto/common_workflows` - Common usage patterns
- :doc:`/gui/index` - Qt GUI documentation
