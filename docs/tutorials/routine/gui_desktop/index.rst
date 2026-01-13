Desktop GUI Application
=======================

NLSQ includes a native Qt desktop application for interactive curve fitting
without writing code.

.. toctree::
   :maxdepth: 1

   launching
   workflow_pages
   presets

Chapter Overview
----------------

**Launching** (3 min)
   Install and start the desktop application.

**Workflow Pages** (10 min)
   Navigate through the 5-page fitting workflow.

**Presets** (5 min)
   Use Fast, Robust, and Quality presets.

Quick Start
-----------

.. code-block:: bash

   # Install with GUI support
   pip install "nlsq[gui_qt]"

   # Launch the application
   nlsq-gui

Features
--------

- **5-page workflow**: Data → Model → Fitting → Results → Export
- **GPU-accelerated plots**: Handle 500K+ points smoothly
- **Theme support**: Light and dark themes (Ctrl+T to toggle)
- **Keyboard shortcuts**: Fast navigation and actions
- **Autosave**: Automatic crash recovery
- **Export options**: ZIP, JSON, CSV, Python code

Screenshot Overview
-------------------

The GUI provides an intuitive workflow:

1. **Data Loading**: Import CSV files or paste from clipboard
2. **Model Selection**: Choose built-in or custom models
3. **Fitting Options**: Select presets (Fast/Robust/Quality)
4. **Results**: View parameters, statistics, and fit quality
5. **Export**: Save results in various formats

Keyboard Shortcuts
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Shortcut
     - Action
   * - Ctrl+1 to Ctrl+5
     - Jump to pages 1-5
   * - Ctrl+R
     - Run fit
   * - Ctrl+O
     - Open file
   * - Ctrl+S
     - Save session
   * - Ctrl+T
     - Toggle dark/light theme
   * - Ctrl+Q
     - Quit application
