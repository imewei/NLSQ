How-To Guides
=============

Practical guides for solving specific problems with NLSQ.
Each guide focuses on a particular task and provides step-by-step instructions.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   migrate_from_scipy
   choose_model

Working with Data
-----------------

.. toctree::
   :maxdepth: 1

   configure_yaml
   common_workflows
   handle_large_data

Optimization
------------

.. toctree::
   :maxdepth: 1

   optimize_performance
   advanced_api
   streaming_checkpoints

Troubleshooting
---------------

.. toctree::
   :maxdepth: 1

   debug_bad_fits
   troubleshooting

Domain Applications
-------------------

.. toctree::
   :maxdepth: 1

   domain_applications/index

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Task
     - Guide
   * - Migrate from scipy.optimize.curve_fit
     - :doc:`migrate_from_scipy`
   * - Choose the right model function
     - :doc:`choose_model`
   * - Handle datasets > 100K points
     - :doc:`handle_large_data`
   * - Speed up fitting with GPU
     - :doc:`optimize_performance`
   * - Fix convergence failures
     - :doc:`debug_bad_fits`
   * - Use YAML configuration
     - :doc:`configure_yaml`

See Also
--------

- :doc:`/tutorials/index` - Step-by-step learning path
- :doc:`/explanation/index` - Understand concepts in depth
- :doc:`/reference/index` - Complete API documentation
