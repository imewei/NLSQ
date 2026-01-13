Routine User Guide
==================

This guide is designed for users who want to perform standard curve fitting
analysis efficiently using NLSQ's **3-workflow system** (auto, auto_global, hpc),
the CLI, or the desktop GUI.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Routine Tutorials
      :link: tutorials/routine/index
      :link-type: doc

      Comprehensive tutorials covering the 3-workflow system.

   .. grid-item-card:: Desktop GUI
      :link: gui/index
      :link-type: doc

      Interactive curve fitting without writing code.

The 3-Workflow System
---------------------

NLSQ provides three workflows that cover all curve fitting needs:

- **auto**: Memory-aware local optimization (default)
- **auto_global**: Global optimization with bounds
- **hpc**: HPC clusters with checkpointing

.. code-block:: python

   from nlsq import fit

   # Default: auto workflow
   popt, pcov = fit(model, x, y, p0=[1.0, 0.5, 0.0])

   # Global optimization (requires bounds)
   popt, pcov = fit(
       model,
       x,
       y,
       p0=[1.0, 0.5, 0.0],
       workflow="auto_global",
       bounds=([0, 0, -1], [10, 5, 1]),
   )

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/routine/index
   tutorials/routine/three_workflows/index

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   tutorials/routine/getting_started/index
   howto/migration
   howto/choose_model

.. toctree::
   :maxdepth: 2
   :caption: Workflow System

   explanation/workflows
   howto/common_workflows
   reference/configuration

.. toctree::
   :maxdepth: 2
   :caption: Interfaces

   reference/cli
   gui/index
   tutorials/routine/gui_desktop/index
