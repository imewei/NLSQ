User Guides
===========

Comprehensive guides for using NLSQ effectively in your projects.

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

4-Layer Defense Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`defense_layers`

**New in version 0.3.6**: Prevent Adam warmup divergence:

- Layer 1: Warm start detection (skip warmup when near optimal)
- Layer 2: Adaptive learning rate selection
- Layer 3: Cost-increase guard (abort on divergence)
- Layer 4: Step clipping (limit update magnitude)
- Defense telemetry and monitoring
- Sensitivity presets (strict, relaxed, disabled, scientific)

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

Streaming Optimizer Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`streaming_optimizer_comparison`

Choose the right streaming optimizer for your use case:

- StreamingOptimizer vs AdaptiveHybridStreamingOptimizer
- Feature comparison and decision guide
- Parameter normalization and multi-scale problems
- Multi-start optimization and exact covariance

Group Variance Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:doc:`group_variance_regularization`

Prevent per-group parameters from absorbing physical signals:

- Mathematical formulation (MSE + lambda * Var penalty)
- XPCS laminar flow fitting example
- Choosing regularization strength (L-curve method)
- Phase 1 (Adam warmup) and Phase 2 (Gauss-Newton) implementation

Practical Workflows
~~~~~~~~~~~~~~~~~~~

:doc:`practical_workflows`

Ready-to-use workflows for scientific applications:

- Spectroscopy (peak fitting, fluorescence lifetime)
- X-ray scattering (SAXS, WAXS, XPCS)
- Kinetics and binding (Michaelis-Menten, isotherms)
- Imaging (2D Gaussian, PSF fitting)
- Materials science (stress-strain, thermal)
- Hardware-specific configurations (CPU, GPU, HPC)

Troubleshooting
~~~~~~~~~~~~~~~

:doc:`troubleshooting`

Solutions to common issues:

- Convergence failures
- Memory errors
- JAX compatibility issues
- Performance problems
- Installation issues

Guide Index
-----------

.. toctree::
   :maxdepth: 1
   :caption: Core Guides

   migration_scipy
   workflow_options
   advanced_customization
   stability
   defense_layers
   performance_guide
   large_datasets
   streaming_optimizer_comparison
   group_variance_regularization
   practical_workflows
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Migration Guides

   /migration/v0.3.6_defense_layers
   /migration/curve_fit_to_fit
   /migration/streaming_fault_tolerance

Migration Guides
~~~~~~~~~~~~~~~~

Guides for upgrading to newer NLSQ versions and APIs:

:doc:`/migration/v0.3.6_defense_layers`
    **New in v0.3.6**: Understand the 4-layer defense strategy and behavioral changes.

:doc:`/migration/curve_fit_to_fit`
    Transition from ``curve_fit()`` to the unified ``fit()`` function with preset-based configuration.

:doc:`/migration/streaming_fault_tolerance`
    Enable fault-tolerant streaming optimization for large datasets with checkpoint/resume support.
