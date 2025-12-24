CLI Reference
=============

NLSQ provides a command-line interface for running curve fitting workflows
defined in YAML configuration files. This enables reproducible, scriptable
fitting pipelines without writing Python code.

Installation
------------

The CLI is installed automatically with NLSQ::

    pip install nlsq

Verify installation::

    nlsq --version
    nlsq info

Commands Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Command
     - Description
   * - ``nlsq fit``
     - Execute a single fitting workflow from a YAML file
   * - ``nlsq batch``
     - Run multiple workflows in parallel with summary reporting
   * - ``nlsq info``
     - Display system information, JAX backend, and available models

Global Options
--------------

All commands support these global options:

``--version``
    Show NLSQ version and exit.

``-v, --verbose``
    Enable verbose output for debugging.

``-h, --help``
    Show help message and exit.


nlsq fit
--------

Execute a single curve fitting workflow from a YAML configuration file.

**Synopsis**::

    nlsq fit WORKFLOW_FILE [OPTIONS]

**Arguments**

``WORKFLOW_FILE``
    Path to the workflow YAML configuration file (required).

**Options**

``-o, --output PATH``
    Override the ``export.results_file`` path specified in the YAML file.

``--stdout``
    Output results as JSON to stdout (for piping to other tools).

``-v, --verbose``
    Enable verbose output.

**Examples**

Basic usage::

    nlsq fit experiment.yaml

Override output file::

    nlsq fit experiment.yaml --output results/fit_001.json

Pipe results to jq for processing::

    nlsq fit experiment.yaml --stdout | jq '.popt'

**Exit Codes**

- ``0``: Success
- ``1``: Configuration, data loading, model, or fitting error


nlsq batch
----------

Execute multiple fitting workflows in parallel with aggregate summary.

**Synopsis**::

    nlsq batch WORKFLOW_FILE [WORKFLOW_FILE ...] [OPTIONS]

**Arguments**

``WORKFLOW_FILES``
    One or more paths to workflow YAML files. Supports shell glob patterns.

**Options**

``-s, --summary FILE``
    Path for aggregate summary file (JSON format).

``-w, --workers N``
    Maximum number of parallel workers. Default: auto-detect
    (limited to min(4, cpu_count, n_workflows) for JAX compatibility).

``--continue-on-error``
    Continue processing remaining workflows if one fails (default: true).

``-v, --verbose``
    Enable verbose output.

**Examples**

Fit multiple workflows::

    nlsq batch workflow1.yaml workflow2.yaml workflow3.yaml

Use glob patterns::

    nlsq batch configs/*.yaml

Generate summary report::

    nlsq batch configs/*.yaml --summary batch_summary.json

Limit parallelism::

    nlsq batch configs/*.yaml --workers 2

**Summary Output**

When ``--summary`` is specified, a JSON file is written with:

.. code-block:: json

    {
        "total": 10,
        "succeeded": 9,
        "failed": 1,
        "duration_seconds": 45.2,
        "start_time": "2024-01-15T10:30:00",
        "end_time": "2024-01-15T10:30:45",
        "successes": ["workflow1.yaml", "workflow2.yaml"],
        "failures": [{"file": "workflow3.yaml", "error": "..."}]
    }

**Exit Codes**

- ``0``: All workflows succeeded
- ``1``: One or more workflows failed


nlsq info
---------

Display system information including NLSQ version, JAX configuration,
memory status, and available builtin models.

**Synopsis**::

    nlsq info [OPTIONS]

**Options**

``-v, --verbose``
    Show detailed parameter information for each model.

**Examples**

Basic info::

    nlsq info

Verbose with model docstrings::

    nlsq info --verbose

**Output**

.. code-block:: text

    NLSQ Version: 0.4.1

    Python Version: 3.12.0

    JAX Configuration:
      Version: 0.8.0
      Backend: gpu
      Devices: 1
        - cuda:0 (NVIDIA RTX 4090)

    System Memory:
      Total: 64.0 GB
      Available: 48.2 GB
      Used: 15.8 GB (24.7%)

    Builtin Models:
      - linear(x, a, b)
      - exponential_decay(x, a, b, c)
      - gaussian(x, amp, mu, sigma)
      - sigmoid(x, L, x0, k, b)
      - power_law(x, a, b)
      ...


YAML Workflow Configuration
---------------------------

Workflow files define the complete fitting pipeline.

Minimal Example
~~~~~~~~~~~~~~~

.. code-block:: yaml

    metadata:
      workflow_name: "my_experiment"

    data:
      input_file: "data/experiment.csv"
      format: "auto"
      columns:
        x: 0
        y: 1

    model:
      type: "builtin"
      name: "exponential_decay"
      auto_p0: true

    fitting:
      method: "trf"

    export:
      results_file: "output/results.json"

Complete Reference
~~~~~~~~~~~~~~~~~~

**metadata** (optional)
    Workflow identification.

    .. code-block:: yaml

        metadata:
          workflow_name: "experiment_001"
          dataset_id: "sample_A"
          description: "Fitting decay curve"

**data** (required)
    Data source configuration.

    .. code-block:: yaml

        data:
          input_file: "data.csv"      # Path to data file
          format: "auto"              # auto, ascii, csv, npz, hdf5
          columns:
            x: 0                      # Column index or name
            y: 1
            sigma: null               # Optional uncertainties

          # Format-specific options
          ascii:
            delimiter: null           # Any whitespace
            comment_char: "#"
            skip_header: 0
            dtype: "float64"

          csv:
            delimiter: ","
            header: true

          npz:
            x_key: "x"
            y_key: "y"
            sigma_key: null

          hdf5:
            x_path: "/data/x"
            y_path: "/data/y"

**model** (required)
    Model function specification.

    .. code-block:: yaml

        model:
          type: "builtin"             # builtin, custom, polynomial
          name: "gaussian"            # Builtin model name
          auto_p0: true               # Auto-estimate initial parameters

          # For custom models
          path: "models/my_model.py"  # Python file path
          function: "my_function"     # Function name in file

          # For polynomial
          degree: 3                   # Polynomial degree

**fitting** (optional)
    Fitting algorithm configuration.

    .. code-block:: yaml

        fitting:
          method: "trf"               # Trust Region Reflective
          p0: [1.0, 0.5, 0.1]        # Initial parameters
          bounds:
            lower: [0, -inf, 0]
            upper: [10, inf, 5]
          ftol: 1.0e-8                # Function tolerance
          xtol: 1.0e-8                # Parameter tolerance
          gtol: 1.0e-8                # Gradient tolerance
          max_nfev: null              # Max function evaluations
          absolute_sigma: false       # Interpret sigma as absolute

**export** (optional)
    Results output configuration.

    .. code-block:: yaml

        export:
          results_file: "results.json"
          format: "json"              # json, csv, npz

          include:
            parameters: true
            covariance: true
            uncertainties: true
            statistics: true          # R-squared, RMSE
            residuals: false
            fitted_values: false
            convergence_info: true

**visualization** (optional)
    Plot generation settings.

    .. code-block:: yaml

        visualization:
          enabled: true
          output_dir: "figures"
          formats: ["png", "pdf"]
          dpi: 300
          figsize: [8, 6]

          main_plot:
            show_data: true
            show_fit: true
            show_residuals: true

          confidence_bands:
            enabled: true
            levels: [0.68, 0.95]


Builtin Models
--------------

The following models are available with ``type: "builtin"``:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Name
     - Function
     - Parameters
   * - ``linear``
     - :math:`f(x) = a \cdot x + b`
     - a, b
   * - ``exponential_decay``
     - :math:`f(x) = a \cdot e^{-bx} + c`
     - a, b, c
   * - ``exponential_growth``
     - :math:`f(x) = a \cdot e^{bx} + c`
     - a, b, c
   * - ``gaussian``
     - :math:`f(x) = A \cdot e^{-(x-\mu)^2/(2\sigma^2)}`
     - amp, mu, sigma
   * - ``sigmoid``
     - :math:`f(x) = L / (1 + e^{-k(x-x_0)}) + b`
     - L, x0, k, b
   * - ``power_law``
     - :math:`f(x) = a \cdot x^b`
     - a, b
   * - ``polynomial``
     - :math:`f(x) = \sum_{i=0}^{n} a_i x^i`
     - a0, a1, ..., an


Custom Models
-------------

Define custom models in Python files:

.. code-block:: python

    # models/my_model.py
    import jax.numpy as jnp


    def double_exponential(x, a1, t1, a2, t2, c):
        """Double exponential decay model."""
        return a1 * jnp.exp(-x / t1) + a2 * jnp.exp(-x / t2) + c


    def estimate_p0(x, y):
        """Optional: Auto-estimate initial parameters."""
        return [y.max() / 2, 1.0, y.max() / 2, 10.0, y.min()]


    def bounds():
        """Optional: Return default parameter bounds."""
        return ([0, 0, 0, 0, -1], [10, 100, 10, 1000, 1])

Reference in YAML:

.. code-block:: yaml

    model:
      type: "custom"
      path: "models/my_model.py"
      function: "double_exponential"


Error Handling
--------------

The CLI provides structured error messages with actionable suggestions:

**ConfigError**
    YAML parsing or validation failures. Check syntax and required fields.

**DataLoadError**
    File not found, column missing, or format issues. Verify paths and column specs.

**ModelError**
    Model resolution failures. Check model name or custom file path.

**FitError**
    Curve fitting failures. Check initial parameters, bounds, or data quality.

Example error output:

.. code-block:: text

    ERROR: DataLoadError
    Message: Column 'temperature' not found in data file
    Context:
      file: data/experiment.csv
      available_columns: ['time', 'signal', 'error']
    Suggestion: Check column name spelling or use column index instead


See Also
--------

- :doc:`yaml_configuration` - YAML file format details
- :doc:`common_workflows` - Workflow examples
- :doc:`../guides/practical_workflows` - Real-world fitting scenarios
