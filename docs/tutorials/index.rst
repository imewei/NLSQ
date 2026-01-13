Tutorials
=========

Learn curve fitting with NLSQ through comprehensive, hands-on tutorials
organized for two types of users:

- **Routine Users**: Use the 3-workflow system (auto, auto_global, hpc) via ``fit()``
- **Advanced Users**: Design custom workflows using NLSQ's API layer

.. note::
   **Coming from SciPy?** Check out :doc:`/howto/migration` for a quick migration guide.

Choose Your Path
----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Routine User Tutorials
      :link: routine/index
      :link-type: doc

      Learn the 3-workflow system for standard curve fitting tasks.
      Covers fit(), GUI, GPU acceleration, and troubleshooting.

      **~2.5 hours** | Beginner-friendly

   .. grid-item-card:: Advanced User Tutorials
      :link: advanced/index
      :link-type: doc

      Design custom optimization pipelines using NLSQ's API layer.
      Covers architecture, protocols, orchestration components.

      **~5 hours** | Requires Python experience

Tutorial Sections
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User-Focused Tutorials

   routine/index
   advanced/index

.. toctree::
   :maxdepth: 1
   :caption: Quick Start (Legacy)

   01_first_fit
   02_understanding_results
   03_fitting_with_bounds
   04_multiple_parameters
   05_large_datasets
   06_gpu_acceleration

The 3-Workflow System
---------------------

NLSQ's core feature is the **3-workflow system** that automatically handles
memory management, optimization strategy, and GPU acceleration:

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Workflow
     - Description
     - Bounds
     - Use Case
   * - ``auto``
     - Memory-aware local optimization
     - Optional
     - **Default** - standard fitting
   * - ``auto_global``
     - Memory-aware global optimization
     - **Required**
     - Multiple minima, unknown guess
   * - ``hpc``
     - Global + checkpointing
     - **Required**
     - Long HPC jobs

Quick Example
-------------

.. code-block:: python

   from nlsq import fit
   import jax.numpy as jnp


   def model(x, a, b, c):
       return a * jnp.exp(-b * x) + c


   # Default workflow (auto)
   popt, pcov = fit(model, x, y, p0=[1.0, 0.5, 0.0])

   # Global optimization
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[1.0, 0.5, 0.0],
       workflow="auto_global",
       bounds=([0, 0, -1], [10, 5, 1]),
   )

Prerequisites
-------------

**Routine Tutorials:**

- Python 3.12 or higher
- Basic Python knowledge
- No curve fitting experience required

**Advanced Tutorials:**

- Completed routine tutorials
- Python classes and type hints
- Basic optimization concepts

Next Steps
----------

After completing the tutorials:

- :doc:`/howto/index` - Solve specific problems
- :doc:`/explanation/index` - Understand concepts in depth
- :doc:`/reference/index` - API reference documentation
