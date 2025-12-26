Reference
=========

Complete reference documentation for all NLSQ modules, functions, and classes.

.. note::

   This section is for looking up specific functions and parameters.
   For learning how to use NLSQ, see :doc:`/tutorials/index`.

.. toctree::
   :maxdepth: 2
   :caption: Reference Sections

   core_api
   functions
   large_data
   configuration
   cli

Core API
--------

The main functions you'll use for curve fitting:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`nlsq.fit`
     - Unified fit function with preset-based configuration (recommended)
   * - :func:`nlsq.curve_fit`
     - Drop-in replacement for scipy.optimize.curve_fit
   * - :func:`nlsq.curve_fit_large`
     - Automatic handling for large datasets
   * - :class:`nlsq.LeastSquares`
     - Low-level least squares solver class

See :doc:`core_api` for detailed API documentation.

Quick Links
-----------

**Core Modules**

- :doc:`/api/nlsq.minpack` - curve_fit implementation
- :doc:`/api/nlsq.least_squares` - Core optimization
- :doc:`/api/nlsq.functions` - Model functions library

**Large Datasets**

- :doc:`/api/nlsq.large_dataset` - Large dataset handling
- :doc:`/api/nlsq.streaming_optimizer` - Streaming optimization
- :doc:`/api/nlsq.adaptive_hybrid_streaming` - Hybrid optimizer

**Utilities**

- :doc:`/api/nlsq.validators` - Input validation
- :doc:`/api/nlsq.diagnostics` - Convergence monitoring
- :doc:`/api/nlsq.callbacks` - Progress callbacks

Full Module Documentation
-------------------------

For auto-generated documentation of all modules:

.. toctree::
   :maxdepth: 1

   /api/modules

See Also
--------

- :doc:`/tutorials/index` - Step-by-step tutorials
- :doc:`/howto/index` - Task-oriented guides
- :doc:`/explanation/index` - Conceptual documentation
