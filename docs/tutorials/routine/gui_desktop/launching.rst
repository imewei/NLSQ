Launching the GUI
=================

This guide covers installing and starting the NLSQ desktop application.

Installation
------------

Install NLSQ with GUI support:

.. code-block:: bash

   pip install "nlsq[gui_qt]"

This installs:

- PySide6 (Qt bindings)
- pyqtgraph (GPU-accelerated plotting)

Starting the Application
------------------------

**From command line:**

.. code-block:: bash

   nlsq-gui

**From Python:**

.. code-block:: python

   from nlsq.gui_qt import run_desktop

   run_desktop()

**As a module:**

.. code-block:: bash

   python -m nlsq.gui_qt

First Launch
------------

On first launch:

1. The main window opens with the Data Loading page
2. Default theme is light (toggle with Ctrl+T)
3. Window position and size are saved for next launch

Command Line Options
--------------------

.. code-block:: bash

   # Open a specific file
   nlsq-gui /path/to/data.csv

   # Force dark theme
   nlsq-gui --dark

   # Show version
   nlsq-gui --version

System Requirements
-------------------

- Python 3.12+
- Qt 6.5+ (via PySide6)
- OpenGL support (for GPU-accelerated plots)
- ~100MB disk space for dependencies

Platform Notes
--------------

**Linux:**

Install OpenGL libraries if needed:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt install libgl1-mesa-glx

**macOS:**

Works out of the box on macOS 12+.

**Windows:**

Requires Visual C++ Redistributable. Usually already installed.

Troubleshooting Launch Issues
-----------------------------

**"PySide6 not found":**

.. code-block:: bash

   pip install "nlsq[gui_qt]"

**"Cannot find OpenGL":**

Install graphics drivers or OpenGL libraries.

**Application doesn't start:**

.. code-block:: bash

   # Check for errors
   python -c "from nlsq.gui_qt import run_desktop; run_desktop()"

**Blank window:**

Try disabling OpenGL acceleration:

.. code-block:: bash

   export QT_QUICK_BACKEND=software
   nlsq-gui

Next Steps
----------

- :doc:`workflow_pages` - Learn the 5-page workflow
- :doc:`presets` - Understand fitting presets
