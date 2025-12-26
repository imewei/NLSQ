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
- :func:`nlsq.curve_fit_large` - Automatic chunking for large datasets
- :class:`nlsq.LeastSquares` - Low-level least squares solver class
- :class:`nlsq.CurveFit` - Reusable curve fitting class (JIT-compiled)

See :doc:`modules` for complete module documentation.

Large Dataset API
-----------------

Specialized functions for large-scale fitting:

- :func:`nlsq.curve_fit_large` - Automatic chunking and memory management
- :class:`nlsq.LargeDatasetHandler` - Advanced dataset management
- :class:`nlsq.StreamingOptimizer` - Stream-based optimization

See :doc:`large_datasets_api` for detailed documentation.

Adaptive Hybrid Streaming API
-----------------------------

Four-phase hybrid optimizer combining parameter normalization, Adam warmup,
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

- ``nlsq.minpack`` - Main curve_fit implementation
- ``nlsq.least_squares`` - Least squares solver
- ``nlsq.trf`` - Trust Region Reflective algorithm

Advanced Features
~~~~~~~~~~~~~~~~~

- ``nlsq.large_dataset`` - Large dataset handling
- ``nlsq.memory_manager`` - Memory management
- ``nlsq.mixed_precision`` - Automatic mixed precision management
- ``nlsq.smart_cache`` - Intelligent caching
- ``nlsq.diagnostics`` - Performance monitoring
- ``nlsq.adaptive_hybrid_streaming`` - Four-phase hybrid optimizer
- ``nlsq.hybrid_streaming_config`` - Hybrid streaming configuration
- ``nlsq.parameter_normalizer`` - Parameter normalization

Utilities
~~~~~~~~~

- ``nlsq.validators`` - Input validation
- ``nlsq.loss_functions`` - Loss function library
- ``nlsq.config`` - Configuration management
- ``nlsq.logging`` - Logging utilities

GUI Module
~~~~~~~~~~

- ``nlsq.gui`` - Interactive Streamlit-based GUI for curve fitting
- ``nlsq.gui.adapters`` - Bridge between GUI and core fitting engine
- ``nlsq.gui.components`` - Reusable UI components

See :doc:`nlsq.gui` for complete GUI API documentation.

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

See :doc:`../developer/optimization_case_study` for detailed performance analysis.
