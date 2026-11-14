API Reference
=============

Complete API documentation for NLSQ modules and functions.

.. toctree::
   :maxdepth: 2

   modules
   notebook_utils

Core API
--------

The main NLSQ API provides drop-in replacements for SciPy's curve fitting functions:

Main Functions
~~~~~~~~~~~~~~

- :func:`nlsq.fit` - **Unified curve fitting with preset-based configuration** (recommended)
- :func:`nlsq.curve_fit` - High-level curve fitting interface (SciPy-compatible)
- :func:`nlsq.curve_fit_large` - Exact streaming fit for large datasets (chunked on request)
- :class:`nlsq.LeastSquares` - Low-level least squares solver class
- :class:`nlsq.CurveFit` - Reusable curve fitting class (JIT-compiled)

See :doc:`modules` for complete module documentation.

Large Dataset API
-----------------

Specialized functions for large-scale fitting:

- :func:`nlsq.curve_fit_large` - Exact full-data streaming by default; chunked and hybrid fits via ``strategy=``
- :class:`nlsq.LargeDatasetFitter` - Advanced dataset management

See :doc:`large_datasets_api` for detailed documentation.

Adaptive Hybrid Streaming API
-----------------------------

Four-phase hybrid optimizer combining parameter normalization, L-BFGS warmup,
streaming Gauss-Newton, and exact covariance computation:

- :class:`nlsq.AdaptiveHybridStreamingOptimizer` - Main optimizer class
- :class:`nlsq.HybridStreamingConfig` - Configuration with presets
- :class:`nlsq.ParameterNormalizer` - Parameter normalization

See:

- :doc:`nlsq.adaptive_hybrid_streaming` - Main optimizer documentation
- :doc:`nlsq.hybrid_streaming_config` - Configuration options
- :doc:`nlsq.parameter_normalizer` - Parameter normalization

Module Organization
-------------------

Core Modules
~~~~~~~~~~~~

- ``nlsq.core.minpack`` - Main curve_fit implementation
- ``nlsq.core.least_squares`` - Least squares solver
- ``nlsq.core.trf`` - Trust Region Reflective algorithm

Advanced Features
~~~~~~~~~~~~~~~~~

- ``nlsq.streaming.large_dataset`` - Large dataset handling
- ``nlsq.caching.memory_manager`` - Memory management
- ``nlsq.caching.unified_cache`` - JIT compilation caching
- ``nlsq.diagnostics`` - Model Health Diagnostics System (identifiability, gradient health, sloppy model analysis)
- ``nlsq.streaming.adaptive_hybrid`` - Four-phase hybrid optimizer
- ``nlsq.streaming.hybrid_config`` - Hybrid streaming configuration
- ``nlsq.precision.parameter_normalizer`` - Parameter normalization

Utilities
~~~~~~~~~

- ``nlsq.utils.validators`` - Input validation
- ``nlsq.core.loss_functions`` - Loss function library
- ``nlsq.config`` - Configuration management
- ``nlsq.utils.logging`` - Logging utilities

See :doc:`modules` for complete documentation of all modules.

Development Tools
-----------------

Notebook Configuration Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modern framework for transforming Jupyter notebooks with automated configurations:

- :mod:`notebook_utils` - Notebook transformation package
- :class:`~notebook_utils.pipeline.TransformationPipeline` - Pipeline orchestration
- :class:`~notebook_utils.tracking.ProcessingTracker` - Incremental processing

See :doc:`notebook_utils` for complete API documentation and :doc:`../developer/notebook_utilities` for usage guide.

Performance Benchmarks
----------------------

See :doc:`performance_benchmarks` for measured timings (same-device comparisons).
