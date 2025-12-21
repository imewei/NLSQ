User Guides
===========

Comprehensive guides for using NLSQ effectively in your projects.

.. toctree::
   :maxdepth: 2

   migration_scipy
   workflow_options
   advanced_customization
   stability
   performance_guide
   large_datasets
   troubleshooting

Guide Overview
--------------

Migrating from SciPy
~~~~~~~~~~~~~~~~~~~~

:doc:`migration_scipy`

Complete guide for transitioning from ``scipy.optimize.curve_fit`` to NLSQ:

- Minimal code changes required
- API compatibility reference
- Common migration patterns
- Performance considerations
- Troubleshooting migration issues

Workflow Options
~~~~~~~~~~~~~~~~

:doc:`workflow_options`

Common configuration choices for workflow-driven usage:

- Callbacks and progress monitoring
- Robust loss functions
- Solver and algorithm selection
- Memory management

Advanced Customization
~~~~~~~~~~~~~~~~~~~~~~

:doc:`advanced_customization`

API-level customization and extension points:

- Custom callbacks
- Mixed precision control
- Diagnostics and sparse Jacobians
- Streaming and online optimization

Numerical Stability
~~~~~~~~~~~~~~~~~~~

:doc:`stability`

Prevent optimization divergence:

- Stability modes (auto, check, off)
- Physics applications (XPCS, scattering)
- Large Jacobian optimization
- Rescale data options
- Condition number monitoring

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`performance_guide`

Maximize fitting speed and efficiency:

- GPU/TPU acceleration strategies
- Batch processing techniques
- Memory optimization
- JIT compilation tips
- Benchmarking and profiling

Large Datasets
~~~~~~~~~~~~~~

:doc:`large_datasets`

Handle datasets with millions of points:

- Automatic data chunking
- Streaming optimization
- Memory-efficient processing
- Parallel fitting strategies

Troubleshooting
~~~~~~~~~~~~~~~

:doc:`troubleshooting`

Solutions to common issues:

- Convergence failures
- Memory errors
- JAX compatibility issues
- Performance problems
- Installation issues
