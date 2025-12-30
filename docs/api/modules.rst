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
   nlsq.memory_pool
   nlsq.mixed_precision
   large_datasets_api

Adaptive Hybrid Streaming (v0.3.0+)
-----------------------------------

Four-phase hybrid optimizer with parameter normalization, Adam warmup,
streaming Gauss-Newton, and exact covariance computation:

.. toctree::
   :maxdepth: 2

   nlsq.adaptive_hybrid_streaming
   nlsq.hybrid_streaming_config
   nlsq.parameter_normalizer
   nlsq.streaming.telemetry
   nlsq.streaming.validators

Global Optimization (v0.3.3+)
-----------------------------

Multi-start optimization with Latin Hypercube Sampling (LHS) for finding global
optima in problems with multiple local minima:

.. toctree::
   :maxdepth: 2

   nlsq.global_optimization

Workflow System (v0.3.4+)
-------------------------

Unified workflow system with automatic strategy selection based on dataset size
and available memory:

.. toctree::
   :maxdepth: 2

   nlsq.workflow

Command-Line Interface (v0.4.1+)
--------------------------------

YAML-based workflow execution from the command line:

.. toctree::
   :maxdepth: 2

   nlsq.cli

GUI Interface (v0.4.1+)
-----------------------

Interactive Streamlit-based graphical interface for curve fitting:

.. toctree::
   :maxdepth: 2

   nlsq.gui

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
   nlsq.parameter_estimation

Interfaces & Protocols (v0.4.2+)
---------------------------------

Protocol definitions for dependency injection:

.. toctree::
   :maxdepth: 2

   nlsq.interfaces

Algorithms & Optimization
--------------------------

Low-level optimization algorithms:

.. toctree::
   :maxdepth: 2

   nlsq.trf
   nlsq.core.trf_jit
   nlsq.core.profiler
   nlsq.optimizer_base
   nlsq.algorithm_selector
   nlsq.loss_functions
   nlsq.sparse_jacobian
   nlsq.robust_decomposition
   nlsq.svd_fallback

Utilities & Infrastructure
---------------------------

Support modules for configuration, caching, and diagnostics:

.. toctree::
   :maxdepth: 2

   nlsq.config
   nlsq.device
   nlsq.validators
   nlsq.diagnostics
   nlsq.caching
   nlsq.unified_cache
   nlsq.compilation_cache
   nlsq.smart_cache
   nlsq.logging
   nlsq.error_messages
   nlsq.constants
   nlsq.types
   nlsq.result
   nlsq.common_jax
   nlsq.common_scipy

Performance & Profiling
-----------------------

Performance analysis, profiling, and benchmarking tools (NEW in v0.3.0-beta.2+):

.. toctree::
   :maxdepth: 2

   nlsq.async_logger
   nlsq.profiling
   nlsq.profiler
   nlsq.profiler_visualization
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
- :doc:`nlsq.core.trf_jit` - JIT-compiled TRF functions (NEW in v0.4.2)
- :doc:`nlsq.core.profiler` - TRF performance profiling (NEW in v0.4.2)

**Feature Modules**:
- :doc:`nlsq.functions` - Pre-built fit functions (NEW in v0.1.1)
- :doc:`nlsq.callbacks` - Progress monitoring & early stopping (NEW in v0.1.1)
- :doc:`nlsq.stability` - Numerical stability analysis (NEW in v0.1.1)
- :doc:`nlsq.fallback` - Automatic retry strategies (NEW in v0.1.1)
- :doc:`nlsq.recovery` - Optimization failure recovery (NEW in v0.1.1)
- :doc:`nlsq.bound_inference` - Smart parameter bounds (NEW in v0.1.1)
- :doc:`nlsq.parameter_estimation` - Initial parameter estimation (NEW in v0.3.0-beta.2)
- :doc:`nlsq.algorithm_selector` - Automatic algorithm selection (NEW in v0.3.0-beta.2)

**Large Dataset Modules**:
- :doc:`nlsq.large_dataset` - Chunked fitting for large data
- :doc:`nlsq.streaming_optimizer` - Streaming optimization for unlimited data (NEW in v0.1.1)
- :doc:`nlsq.streaming_config` - Configuration for streaming optimizer (v0.2.0+)
- :doc:`nlsq.memory_manager` - Intelligent memory management (NEW in v0.1.1)
- :doc:`nlsq.memory_pool` - Memory pool allocation (NEW in v0.3.0-beta.2)
- :doc:`nlsq.mixed_precision` - Automatic mixed precision management (NEW in v0.1.6)
- :doc:`large_datasets_api` - Comprehensive large dataset guide

**Adaptive Hybrid Streaming Modules** (NEW in v0.3.0+):
- :doc:`nlsq.adaptive_hybrid_streaming` - Four-phase hybrid optimizer
- :doc:`nlsq.hybrid_streaming_config` - Configuration with presets
- :doc:`nlsq.parameter_normalizer` - Parameter normalization for gradient balance
- :doc:`nlsq.streaming.telemetry` - Defense layer telemetry (NEW in v0.4.2)
- :doc:`nlsq.streaming.validators` - Configuration validators (NEW in v0.4.2)

**Interfaces & Protocols** (NEW in v0.4.2):
- :doc:`nlsq.interfaces` - Protocol definitions for dependency injection

**Global Optimization Modules** (NEW in v0.3.3+):
- :doc:`nlsq.global_optimization` - Multi-start with LHS, Sobol, Halton samplers

**Workflow System Modules** (NEW in v0.3.4+):
- :doc:`nlsq.workflow` - Unified fit() with auto-selection, presets, YAML config

**Command-Line Interface Modules** (NEW in v0.4.1+):
- :doc:`nlsq.cli` - CLI commands: ``nlsq fit``, ``nlsq batch``, ``nlsq info``, ``nlsq gui``

**GUI Modules** (NEW in v0.4.1+):
- :doc:`nlsq.gui` - Interactive Streamlit-based GUI for curve fitting

**Utility Modules**:
- :doc:`nlsq.config` - Configuration management
- :doc:`nlsq.device` - GPU detection and warnings (NEW in v0.1.6)
- :doc:`nlsq.validators` - Input validation (NEW in v0.1.1)
- :doc:`nlsq.diagnostics` - Optimization diagnostics (NEW in v0.1.1)
- :doc:`nlsq.caching` - JIT and result caching
- :doc:`nlsq.unified_cache` - Unified compilation cache (NEW in v0.3.0-beta.2)
- :doc:`nlsq.compilation_cache` - Legacy compilation cache
- :doc:`nlsq.smart_cache` - Smart adaptive caching
- :doc:`nlsq.logging` - Logging and debugging
- :doc:`nlsq.error_messages` - Standardized error messages (NEW in v0.3.0-beta.2)
- :doc:`nlsq.constants` - Numerical constants (NEW in v0.3.0-beta.2)
- :doc:`nlsq.types` - Type definitions (NEW in v0.3.0-beta.2)
- :doc:`nlsq.result` - Result containers (NEW in v0.3.0-beta.2)
- :doc:`nlsq.loss_functions` - Robust loss functions
- :doc:`nlsq.optimizer_base` - Base optimizer classes
- :doc:`nlsq.robust_decomposition` - Robust matrix decomposition (NEW in v0.3.0-beta.2)
- :doc:`nlsq.svd_fallback` - SVD fallback strategies (NEW in v0.3.0-beta.2)
- :doc:`nlsq.sparse_jacobian` - Sparse Jacobian support (NEW in v0.3.0-beta.2)
- :doc:`nlsq.common_jax` - JAX utilities
- :doc:`nlsq.common_scipy` - SciPy compatibility layer

**Performance & Profiling**:
- :doc:`nlsq.async_logger` - Async logging infrastructure (NEW in v0.3.0-beta.3)
- :doc:`nlsq.profiling` - JAX profiler integration and static analysis (v0.3.0-beta.2+)
- :doc:`nlsq.profiler` - Performance profiler (NEW in v0.3.0-beta.2)
- :doc:`nlsq.profiler_visualization` - Profiling visualization (NEW in v0.3.0-beta.2)
- :doc:`performance_benchmarks` - Performance analysis tools
