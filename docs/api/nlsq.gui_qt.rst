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

The GUI supports light and dark themes via Qt's built-in color scheme API. Toggle with ``Ctrl+T``.

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

Workflow pages for the 5-step fitting process:

- ``nlsq.gui_qt.pages.data_loading`` - Data import and column selection
- ``nlsq.gui_qt.pages.model_selection`` - Model function selection
- ``nlsq.gui_qt.pages.fitting_options`` - Fit configuration
- ``nlsq.gui_qt.pages.results`` - Fit results and visualization
- ``nlsq.gui_qt.pages.export`` - Export results in various formats

Widgets
~~~~~~~

Reusable Qt widgets for the fitting workflow:

- ``nlsq.gui_qt.widgets.advanced_options`` - Advanced fitting options panel
- ``nlsq.gui_qt.widgets.column_selector`` - Data column assignment widget
- ``nlsq.gui_qt.widgets.param_config`` - Parameter configuration widget
- ``nlsq.gui_qt.widgets.param_results`` - Fitted parameter display
- ``nlsq.gui_qt.widgets.fit_statistics`` - Fit quality statistics
- ``nlsq.gui_qt.widgets.iteration_table`` - Optimization iteration history
- ``nlsq.gui_qt.widgets.code_editor`` - Syntax-highlighted code editor

Plots
~~~~~

pyqtgraph-based scientific plotting widgets:

- ``nlsq.gui_qt.plots.base_plot`` - Base plot widget class
- ``nlsq.gui_qt.plots.fit_plot`` - Data and fit curve visualization
- ``nlsq.gui_qt.plots.residuals_plot`` - Residuals visualization
- ``nlsq.gui_qt.plots.histogram_plot`` - Residual histogram
- ``nlsq.gui_qt.plots.live_cost_plot`` - Live cost function during fitting

Adapters
~~~~~~~~

Data adapters for the GUI workflow:

- ``nlsq.gui_qt.adapters.data_adapter`` - Data loading and validation
- ``nlsq.gui_qt.adapters.fit_adapter`` - Fit execution wrapper
- ``nlsq.gui_qt.adapters.config_adapter`` - YAML configuration import/export
- ``nlsq.gui_qt.adapters.export_adapter`` - Results export utilities

State Management
~~~~~~~~~~~~~~~~

- ``nlsq.gui_qt.session_state`` - Session state dataclass
- ``nlsq.gui_qt.app_state`` - Qt signal-based application state

Theme
~~~~~

- ``nlsq.gui_qt.theme`` - Theme configuration and management

See Also
--------

- :doc:`/gui/index` - GUI user guide
- :doc:`/gui/user_guide` - Complete GUI reference
