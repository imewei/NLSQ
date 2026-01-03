Qt Desktop GUI (nlsq.gui_qt)
============================

The ``nlsq.gui_qt`` module provides a native desktop application built with
PySide6 (Qt) and pyqtgraph for GPU-accelerated scientific plotting.

.. note::

   The Qt GUI requires optional dependencies. Install with:

   .. code-block:: bash

      pip install "nlsq[gui]"

Launching the GUI
-----------------

From the command line:

.. code-block:: bash

   nlsq-gui

Or from Python:

.. code-block:: python

   from nlsq.gui_qt import run_desktop

   run_desktop()

Module Overview
---------------

Entry Point
~~~~~~~~~~~

.. autofunction:: nlsq.gui_qt.run_desktop

Main Window
~~~~~~~~~~~

The main window manages the 5-page workflow:

1. **Data Loading** - Import CSV, ASCII, NPZ, or HDF5 files
2. **Model Selection** - Choose built-in, polynomial, or custom models
3. **Fitting Options** - Configure tolerances and algorithms
4. **Results** - View parameters, statistics, and plots
5. **Export** - Save results in various formats

Theme Support
~~~~~~~~~~~~~

The GUI supports light and dark themes via ``qdarktheme``. Toggle with ``Ctrl+T``.

Keyboard Shortcuts
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Shortcut
     - Action
   * - Ctrl+1 to Ctrl+5
     - Switch to page 1-5
   * - Ctrl+R
     - Run fit
   * - Ctrl+O
     - Open file
   * - Ctrl+T
     - Toggle theme
   * - Ctrl+Q
     - Quit

Submodules
----------

Pages
~~~~~

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.pages.data_loading
   nlsq.gui_qt.pages.model_selection
   nlsq.gui_qt.pages.fitting_options
   nlsq.gui_qt.pages.results
   nlsq.gui_qt.pages.export

Widgets
~~~~~~~

Reusable Qt widgets for the fitting workflow:

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.widgets.advanced_options
   nlsq.gui_qt.widgets.column_selector
   nlsq.gui_qt.widgets.param_config
   nlsq.gui_qt.widgets.param_results
   nlsq.gui_qt.widgets.fit_statistics
   nlsq.gui_qt.widgets.iteration_table
   nlsq.gui_qt.widgets.code_editor

Plots
~~~~~

pyqtgraph-based scientific plotting widgets:

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.plots.base_plot
   nlsq.gui_qt.plots.fit_plot
   nlsq.gui_qt.plots.residuals_plot
   nlsq.gui_qt.plots.histogram_plot
   nlsq.gui_qt.plots.live_cost_plot

Adapters
~~~~~~~~

Data adapters for the GUI workflow:

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.adapters.data_adapter
   nlsq.gui_qt.adapters.fit_adapter
   nlsq.gui_qt.adapters.config_adapter
   nlsq.gui_qt.adapters.export_adapter

State Management
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.session_state
   nlsq.gui_qt.app_state

Theme
~~~~~

.. autosummary::
   :toctree: generated/

   nlsq.gui_qt.theme

See Also
--------

- :doc:`/gui/index` - GUI user guide
- :doc:`/gui/user_guide` - Complete GUI documentation
