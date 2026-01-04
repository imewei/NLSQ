nlsq.cli
========

Command-line interface for YAML-based curve fitting workflows.

.. versionadded:: 0.4.1

The CLI module provides commands for executing curve fitting workflows
defined in YAML configuration files, enabling reproducible and scriptable
fitting pipelines.

Quick Start
-----------

.. code-block:: bash

    # Launch Qt desktop GUI
    nlsq gui

    # Run a single workflow
    nlsq fit experiment.yaml

    # Output results to stdout for piping
    nlsq fit experiment.yaml --stdout

    # Batch process multiple workflows
    nlsq batch configs/*.yaml --summary results.json

    # Show system information
    nlsq info

    # Copy configuration templates
    nlsq config

Commands
--------

**nlsq gui**
    Launch the interactive Qt desktop GUI for visual curve fitting.
    Provides a 5-page workflow: Data Loading → Model Selection →
    Fitting Options → Results → Export.

**nlsq fit**
    Execute a single curve fitting workflow from YAML configuration.
    Options: ``-o/--output`` to override results path, ``--stdout`` for JSON output.

**nlsq batch**
    Run multiple workflows in parallel with aggregate summary reporting.
    Options: ``-w/--workers`` for parallelism, ``-s/--summary`` for output path.

**nlsq info**
    Display system information including NLSQ version, JAX backend, and
    available builtin models.

**nlsq config**
    Generate configuration templates for new workflows.
    Options: ``--workflow``, ``--model``, ``-o/--output``, ``-f/--force``.

See Also
--------

- :doc:`/reference/cli` - Complete CLI reference documentation
- :doc:`/reference/configuration` - Configuration and YAML file format
- :doc:`/gui/index` - Qt GUI documentation

Module Structure
----------------

nlsq.cli.main
~~~~~~~~~~~~~

.. automodule:: nlsq.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

nlsq.cli.workflow_runner
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.workflow_runner
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

nlsq.cli.data_loaders
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.data_loaders
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

nlsq.cli.model_registry
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.model_registry
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

nlsq.cli.visualization
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.visualization
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

nlsq.cli.result_exporter
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.result_exporter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

nlsq.cli.errors
~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.errors
   :members:
   :undoc-members:
   :show-inheritance:

nlsq.cli.commands
~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:
