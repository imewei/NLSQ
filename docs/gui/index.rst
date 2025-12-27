GUI User Guide
==============

NLSQ provides an interactive graphical interface for curve fitting,
built with Streamlit. The GUI allows you to fit data without writing code.

.. toctree::
   :maxdepth: 2

   user_guide

Quick Start
-----------

Launch the GUI from the command line:

.. code-block:: bash

   nlsq gui

Or from Python:

.. code-block:: python

   from nlsq.gui import launch_gui

   launch_gui()

Features
--------

- **Data Import**: Load CSV, Excel, or paste data directly
- **Model Selection**: Choose from built-in models or define custom functions
- **Interactive Fitting**: Adjust initial guesses and bounds with sliders
- **Visualization**: Real-time plots of data, fit, and residuals
- **Results Export**: Download fitted parameters and statistics

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

- :doc:`user_guide` - Complete GUI documentation
- :doc:`/tutorials/01_first_fit` - Python API tutorial
- :doc:`/reference/index` - API reference
