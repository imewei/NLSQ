NLSQ API Reference
==================

Complete API reference for all NLSQ modules. For most use cases, start with:

- :doc:`nlsq.minpack` - Main curve fitting interface (SciPy-compatible)
- :doc:`nlsq.functions` - Pre-built fit functions library
- :doc:`nlsq.large_dataset` - Large dataset handling

Core API
--------

Main interface for curve fitting:

.. toctree::
   :maxdepth: 2

   nlsq.minpack
   nlsq.least_squares

Pre-Built Functions
-------------------

Library of common fit functions with automatic parameter estimation:

.. toctree::
   :maxdepth: 2

   nlsq.functions

Large Dataset Support
---------------------

Tools for fitting very large datasets (10M+ points):

.. toctree::
   :maxdepth: 2

   nlsq.large_dataset
   nlsq.streaming_optimizer
   nlsq.streaming_config
   nlsq.memory_manager
   large_datasets_api

Enhanced Features (v0.1.1)
--------------------------

New features added in version 0.1.1:

.. toctree::
   :maxdepth: 2

   nlsq.callbacks
   nlsq.stability
   nlsq.fallback
   nlsq.recovery
   nlsq.bound_inference

Algorithms & Optimization
--------------------------

Low-level optimization algorithms:

.. toctree::
   :maxdepth: 2

   nlsq.trf
   nlsq.optimizer_base
   nlsq.loss_functions

Utilities & Infrastructure
---------------------------

Support modules for configuration, caching, and diagnostics:

.. toctree::
   :maxdepth: 2

   nlsq.config
   nlsq.validators
   nlsq.diagnostics
   nlsq.caching
   nlsq.logging
   nlsq.common_jax
   nlsq.common_scipy

Performance & Benchmarking
---------------------------

Performance analysis and benchmarking tools:

.. toctree::
   :maxdepth: 2

   performance_benchmarks

Module Index
------------

.. toctree::
   :maxdepth: 1

   nlsq

Complete Module Listing
------------------------

**Core Modules**:
- :doc:`nlsq.minpack` - Main ``curve_fit()`` API
- :doc:`nlsq.least_squares` - ``least_squares()`` solver
- :doc:`nlsq.trf` - Trust Region Reflective algorithm

**Feature Modules**:
- :doc:`nlsq.functions` - Pre-built fit functions (NEW in v0.1.1)
- :doc:`nlsq.callbacks` - Progress monitoring & early stopping (NEW in v0.1.1)
- :doc:`nlsq.stability` - Numerical stability analysis (NEW in v0.1.1)
- :doc:`nlsq.fallback` - Automatic retry strategies (NEW in v0.1.1)
- :doc:`nlsq.recovery` - Optimization failure recovery (NEW in v0.1.1)
- :doc:`nlsq.bound_inference` - Smart parameter bounds (NEW in v0.1.1)

**Large Dataset Modules**:
- :doc:`nlsq.large_dataset` - Chunked fitting for large data
- :doc:`nlsq.streaming_optimizer` - Streaming optimization for unlimited data (NEW in v0.1.1)
- :doc:`nlsq.streaming_config` - Configuration for streaming optimizer (v0.2.0+)
- :doc:`nlsq.memory_manager` - Intelligent memory management (NEW in v0.1.1)
- :doc:`large_datasets_api` - Comprehensive large dataset guide

**Utility Modules**:
- :doc:`nlsq.config` - Configuration management
- :doc:`nlsq.validators` - Input validation (NEW in v0.1.1)
- :doc:`nlsq.diagnostics` - Optimization diagnostics (NEW in v0.1.1)
- :doc:`nlsq.caching` - JIT and result caching
- :doc:`nlsq.logging` - Logging and debugging
- :doc:`nlsq.loss_functions` - Robust loss functions
- :doc:`nlsq.optimizer_base` - Base optimizer classes
- :doc:`nlsq.common_jax` - JAX utilities
- :doc:`nlsq.common_scipy` - SciPy compatibility layer

**Benchmarking**:
- :doc:`performance_benchmarks` - Performance analysis tools
