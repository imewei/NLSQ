Routine User Tutorials
======================

Welcome to NLSQ! This tutorial series will guide you through curve fitting using
NLSQ's **3-Workflow System** - a simple but powerful approach that handles
everything from small datasets to 100M+ points automatically.

.. note::
   **Coming from SciPy?** NLSQ is a drop-in replacement. Your existing code
   will work with minimal changes. See :doc:`/howto/migration`.

The 3-Workflow System
---------------------

NLSQ provides three workflows that cover all curve fitting use cases:

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Workflow
     - When to Use
     - Bounds
     - Speed
   * - ``auto``
     - Standard fitting (default)
     - Optional
     - Fastest
   * - ``auto_global``
     - Multiple local minima, unknown initial guess
     - **Required**
     - Moderate
   * - ``hpc``
     - HPC clusters, long-running jobs
     - **Required**
     - Varies

Quick Start
-----------

Here's all you need to fit data with NLSQ:

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   # Define your model
   def exponential(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Fit with default workflow (auto)
   popt, pcov = fit(exponential, xdata, ydata, p0=[1.0, 0.5, 0.0])

   # For global optimization (requires bounds)
   popt, pcov = fit(
       exponential,
       xdata,
       ydata,
       p0=[1.0, 0.5, 0.0],
       workflow="auto_global",
       bounds=([0, 0, -1], [10, 5, 1]),
   )

That's it! NLSQ automatically handles memory management, GPU acceleration,
and optimization strategy selection.

Tutorial Series
---------------

.. toctree::
   :maxdepth: 2
   :numbered:

   getting_started/index
   three_workflows/index
   model_selection/index
   data_handling/index
   gpu_acceleration/index
   gui_desktop/index
   troubleshooting/index

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
     - :doc:`getting_started/index`
     - Install NLSQ, fit your first curve, interpret results
     - 20 min
   * - 2
     - :doc:`three_workflows/index`
     - Master the 3-workflow system: auto, auto_global, hpc
     - 30 min
   * - 3
     - :doc:`model_selection/index`
     - Built-in models, custom models, model validation
     - 25 min
   * - 4
     - :doc:`data_handling/index`
     - Data input, uncertainties, bounds, large datasets
     - 25 min
   * - 5
     - :doc:`gpu_acceleration/index`
     - GPU setup, usage, multi-GPU
     - 20 min
   * - 6
     - :doc:`gui_desktop/index`
     - Desktop GUI application for interactive fitting
     - 15 min
   * - 7
     - :doc:`troubleshooting/index`
     - Common issues and solutions
     - 15 min

Prerequisites
-------------

- Python 3.12 or higher
- Basic Python knowledge (variables, functions, imports)
- Basic understanding of scientific data (x, y coordinates)

No prior curve fitting experience is required.

Next Steps After Tutorials
--------------------------

- :doc:`/howto/index` - Solve specific problems
- :doc:`/reference/index` - API reference documentation
- :doc:`/tutorials/advanced/index` - Advanced API tutorials for custom workflows
