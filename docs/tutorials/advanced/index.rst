Advanced User Tutorials
=======================

These tutorials are for developers and scientists who want to leverage NLSQ's
full API capabilities: custom optimization pipelines, protocol-based design,
performance tuning, and extending the library.

.. note::
   **Prerequisites**: Complete the :doc:`/tutorials/routine/index` first.
   Understanding the 3-workflow system is essential before diving into the
   API layer.

Who This Guide Is For
---------------------

- Developers building custom optimization workflows
- Scientists with specialized fitting requirements
- Researchers extending NLSQ for new algorithms
- Power users wanting maximum control

What You'll Learn
-----------------

- NLSQ's internal architecture and design patterns
- How to use core API classes directly
- Performance tuning and profiling
- Creating custom optimizers and extensions

.. toctree::
   :maxdepth: 2
   :numbered:

   architecture/index
   core_apis/index
   custom_workflows/index
   performance/index
   stability/index
   extension/index

Learning Path
-------------

.. list-table::
   :header-rows: 1
   :widths: 5 20 50 10

   * - #
     - Chapter
     - What You'll Learn
     - Time
   * - 1
     - :doc:`architecture/index`
     - Package structure, optimization pipeline, JAX patterns
     - 30 min
   * - 2
     - :doc:`core_apis/index`
     - CurveFit, LeastSquares, TRF classes
     - 45 min
   * - 3
     - :doc:`custom_workflows/index`
     - Build your own optimization pipelines
     - 45 min
   * - 4
     - :doc:`performance/index`
     - JIT caching, memory management, profiling
     - 30 min
   * - 5
     - :doc:`stability/index`
     - Numerical guards, SVD fallback, recovery
     - 30 min
   * - 6
     - :doc:`extension/index`
     - Custom protocols, plugins, testing
     - 30 min

Prerequisites
-------------

- Completed routine user tutorials
- Python experience (classes, decorators, type hints)
- Understanding of optimization concepts
- Familiarity with JAX (helpful but not required)

Quick Reference: API Levels
---------------------------

NLSQ provides multiple API levels:

.. code-block:: text

   High Level    fit() function
        │           │
        │           ▼
        │     CurveFit class
        │           │
        │           ▼
   Mid Level  LeastSquares class
        │           │
        │           ▼
   Low Level  TrustRegionReflective

**Choose based on needs:**

- ``fit()``: Simple, automatic (routine users)
- ``CurveFit``: Reusable, stateful fitting
- ``LeastSquares``: Direct optimizer control
- ``TRF``: Algorithm-level customization

Import Patterns
---------------

.. code-block:: python

   # High-level API
   from nlsq import fit, curve_fit, CurveFit

   # Core classes
   from nlsq.core.least_squares import LeastSquares
   from nlsq.core.trf import TrustRegionReflective

   # Facades
   from nlsq.facades import (
       OptimizationFacade,
       StabilityFacade,
       DiagnosticsFacade,
   )
