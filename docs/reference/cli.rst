CLI Reference
=============

NLSQ provides a command-line interface for common operations.

Installation Check
------------------

.. code-block:: bash

   nlsq --version
   nlsq info

Basic Usage
-----------

nlsq fit
~~~~~~~~

Fit data from a file:

.. code-block:: bash

   nlsq fit data.csv --model exponential --output results.json

**Options:**

.. code-block:: text

   nlsq fit [OPTIONS] DATA_FILE

   Arguments:
     DATA_FILE    Path to data file (CSV, HDF5, NPY)

   Options:
     --model TEXT           Model type or path to custom model
     --p0 TEXT              Initial parameters (comma-separated)
     --bounds TEXT          Parameter bounds (JSON format)
     --output PATH          Output file for results
     --format [json|yaml]   Output format
     --plot                 Generate fit plot
     --quiet                Suppress output
     -h, --help             Show this help

**Examples:**

.. code-block:: bash

   # Fit with Gaussian model
   nlsq fit spectrum.csv --model gaussian --p0 "1,0,1"

   # With bounds
   nlsq fit decay.csv --model exponential \
       --bounds '[[0,0,-1],[10,100,1]]' \
       --output fit.json

   # Custom model from file
   nlsq fit data.csv --model ./my_model.py:model_func

nlsq benchmark
~~~~~~~~~~~~~~

Run performance benchmarks:

.. code-block:: bash

   nlsq benchmark

**Options:**

.. code-block:: text

   nlsq benchmark [OPTIONS]

   Options:
     --sizes TEXT    Dataset sizes to test (comma-separated)
     --models TEXT   Models to benchmark
     --output PATH   Save results to file
     --compare-scipy Compare with SciPy

**Example:**

.. code-block:: bash

   nlsq benchmark --sizes "1000,10000,100000,1000000" --compare-scipy

nlsq info
~~~~~~~~~

Display system and configuration info:

.. code-block:: bash

   nlsq info

**Output:**

.. code-block:: text

   NLSQ Information
   ================
   Version: 0.1.0
   Python: 3.12.0
   JAX: 0.4.25
   Device: cuda:0 (NVIDIA RTX 4090)
   Memory: 24.0 GB available

   Configuration:
   - Config file: ~/.config/nlsq/config.yaml
   - Cache: ~/.cache/nlsq (1.2 GB used)

   Environment:
   - NLSQ_DEBUG: not set
   - CUDA_VISIBLE_DEVICES: 0

nlsq cache
~~~~~~~~~~

Manage the compilation cache:

.. code-block:: bash

   # Show cache status
   nlsq cache status

   # Clear cache
   nlsq cache clear

   # Set cache size limit
   nlsq cache limit 2GB

nlsq gui
~~~~~~~~

Launch the interactive GUI:

.. code-block:: bash

   nlsq gui

**Options:**

.. code-block:: text

   nlsq gui [OPTIONS]

   Options:
     --port INT      Port number (default: 8501)
     --no-browser    Don't open browser automatically

Configuration via CLI
---------------------

Override configuration from command line:

.. code-block:: bash

   # Set tolerance
   nlsq fit data.csv --model gaussian --gtol 1e-10

   # Use preset
   nlsq fit data.csv --model gaussian --preset precise

   # Set memory limit
   nlsq fit large_data.h5 --model gaussian --memory-limit 8GB

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
     - General error
   * - 2
     - Invalid arguments
   * - 3
     - Fit did not converge
   * - 4
     - File not found
   * - 5
     - GPU not available (when required)

Scripting Examples
------------------

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   for file in data/*.csv; do
       nlsq fit "$file" --model exponential --output "${file%.csv}_fit.json"
   done

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Fit and extract parameter
   tau=$(nlsq fit decay.csv --model exponential --format json | jq '.params.tau')
   echo "Decay constant: $tau"

JSON Output Format
~~~~~~~~~~~~~~~~~~

.. code-block:: javascript

   {
     "success": true,
     "params": {
       "a": 2.5,
       "tau": 10.2,
       "c": 0.05
     },
     "errors": {
       "a": 0.1,
       "tau": 0.3,
       "c": 0.02
     },
     "covariance": [[0.01, 0.0, 0.0], [0.0, 0.09, 0.0], [0.0, 0.0, 0.0004]],
     "r_squared": 0.998,
     "residual_std": 0.05,
     "nfev": 42,
     "message": "Optimization terminated successfully"
   }

See Also
--------

- :doc:`configuration` - Configuration options
- :doc:`/howto/common_workflows` - Common usage patterns
