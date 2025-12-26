nlsq.gui
========

The GUI module provides an interactive Streamlit-based graphical interface for
curve fitting with GPU/TPU acceleration. This module is optional and requires
the ``gui`` extra dependencies.

Installation
------------

.. code-block:: bash

    pip install nlsq[gui]

Launching
---------

.. code-block:: bash

    # Via CLI command
    nlsq gui

    # Via Python
    python -m nlsq.gui.app

    # As desktop application
    python -m nlsq.gui.run_desktop

Module Overview
---------------

The GUI is organized into layers:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Layer
     - Description
   * - Pages
     - Streamlit page components (Data Loading, Model Selection, etc.)
   * - Adapters
     - Bridge between GUI and core NLSQ fitting engine
   * - Components
     - Reusable UI elements (parameter config, plots, tables)
   * - State
     - SessionState management via dataclass

Public API
----------

The following classes and functions are the main entry points for the GUI module.
They are re-exported at the ``nlsq.gui`` package level for convenient access.

.. automodule:: nlsq.gui
   :members:
   :show-inheritance:

Adapters (Implementation Details)
---------------------------------

The adapter layer transforms GUI inputs into NLSQ API calls and formats
outputs for display. These modules contain implementation details; the
public API is documented above.

nlsq.gui.adapters.data_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.gui.adapters.data_adapter
   :no-index:

nlsq.gui.adapters.model_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.gui.adapters.model_adapter
   :no-index:

nlsq.gui.adapters.fit_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.gui.adapters.fit_adapter
   :no-index:

nlsq.gui.adapters.export_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.gui.adapters.export_adapter
   :no-index:

nlsq.gui.adapters.config_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: nlsq.gui.adapters.config_adapter
   :no-index:

Components (Implementation Details)
-----------------------------------

Reusable UI components for building the GUI pages.

.. automodule:: nlsq.gui.components
   :no-index:

Presets (Implementation Details)
--------------------------------

.. automodule:: nlsq.gui.presets
   :no-index:

Desktop Application (Implementation Details)
---------------------------------------------

.. automodule:: nlsq.gui.run_desktop
   :no-index:

.. automodule:: nlsq.gui.desktop_config
   :no-index:

See Also
--------

- :doc:`../gui/user_guide` - User guide for the GUI
- :doc:`../developer/gui_development_guide` - Developer guide for GUI development
- :doc:`../reference/cli` - CLI documentation for ``nlsq gui`` command
