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
→ Results → Export. Requires the ``gui_qt`` extra:

.. code-block:: bash

   pip install "nlsq[gui_qt]"

nlsq fit
~~~~~~~~

Execute single curve fit from a YAML workflow configuration.

**Syntax:**

.. code-block:: text

   nlsq fit <workflow.yaml> [OPTIONS]

   Arguments:
     workflow.yaml        Path to workflow YAML configuration file

   Options:
     -o, --output FILE    Override export.results_file path
     --stdout             Output results as JSON to stdout (for piping)

**Examples:**

.. code-block:: bash

   # Basic fit
   nlsq fit workflow.yaml

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
   Version: 0.6.2
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
     auto_p0: true
   export:
     results_file: "results.json"

**Built-in Workflow Presets:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Preset
     - Description
   * - ``standard``
     - Standard curve_fit() with default tolerances (1e-8)
   * - ``quality``
     - Highest precision with multi-start (tolerances 1e-10)
   * - ``fast``
     - Speed-optimized with looser tolerances (1e-6)
   * - ``large_robust``
     - Chunked processing with multi-start for large datasets
   * - ``streaming``
     - AdaptiveHybridStreamingOptimizer for huge datasets
   * - ``hpc_distributed``
     - Multi-GPU/node configuration for HPC clusters (PBS)

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
     - General error (configuration, data loading, model, or fitting error)

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
