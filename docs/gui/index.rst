GUI User Guide
==============

NLSQ provides a native desktop GUI built with PySide6 (Qt) and pyqtgraph
for GPU-accelerated scientific plotting. The GUI allows you to fit data
without writing code.

The complete GUI documentation lives in the structured tutorial:

.. toctree::
   :maxdepth: 2

   /tutorials/routine/gui_desktop/index

Quick Start
-----------

Launch the GUI from the command line:

.. code-block:: bash

   nlsq-gui

Or from Python:

.. code-block:: python

   from nlsq.gui_qt import run_desktop

   run_desktop()

When to Use the GUI
-------------------

The GUI is ideal for:

- **Exploratory analysis**: Quickly try different models
- **Interactive fitting**: Tune parameters visually
- **Teaching and demos**: Show curve fitting concepts
- **One-off fits**: No code setup required

For batch processing, scripting, or production pipelines,
use the Python API instead (see :doc:`/tutorials/index`).

See Also
--------

- :doc:`/tutorials/routine/gui_desktop/index` - Complete GUI guide
- :doc:`/tutorials/routine/getting_started/first_fit` - Python API tutorial
- :doc:`/reference/index` - API reference
