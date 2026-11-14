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
   nlsq.core.problem
   nlsq.core.covariance
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
   nlsq.streaming.strategy
   nlsq.memory_manager
   large_datasets_api

Adaptive Hybrid Streaming
-----------------------------------

Four-phase hybrid optimizer with parameter normalization, L-BFGS warmup,
streaming Gauss-Newton, and exact covariance computation:

.. toctree::
   :maxdepth: 2

   nlsq.adaptive_hybrid_streaming
   nlsq.hybrid_streaming_config
   nlsq.parameter_normalizer
   nlsq.streaming.telemetry
   nlsq.streaming.validators
   nlsq.streaming.phases
   nlsq.streaming.exact_lm
   nlsq.streaming.large_fit

Global Optimization
-----------------------------

Multi-start optimization with Latin Hypercube Sampling (LHS) for finding global
optima in problems with multiple local minima:

.. toctree::
   :maxdepth: 2

   nlsq.global_optimization
   nlsq.global_optimization.config
   nlsq.global_optimization.sampling

Workflow System
-------------------------

Memory-based workflow system with automatic strategy selection based on
available memory and dataset characteristics:

.. toctree::
   :maxdepth: 2

   nlsq.workflow

Command-Line Interface
--------------------------------

YAML-based workflow execution from the command line:

.. toctree::
   :maxdepth: 2

   nlsq.cli

Qt Desktop GUI
------------------------

Native desktop application with PySide6 and pyqtgraph:

.. toctree::
   :maxdepth: 2

   nlsq.gui_qt

Feature Modules
---------------

Callbacks, numerical stability, fallbacks, recovery, bounds and parameter estimation:

.. toctree::
   :maxdepth: 2

   nlsq.callbacks
   nlsq.stability
   nlsq.stability.guard
   nlsq.fallback
   nlsq.recovery
   nlsq.bound_inference
   nlsq.parameter_estimation

Algorithms & Optimization
--------------------------

Low-level optimization algorithms:

.. toctree::
   :maxdepth: 2

   nlsq.trf
   nlsq.core.trf_jit
   nlsq.core.profiler
   nlsq.algorithm_selector
   nlsq.loss_functions
   nlsq.sparse_jacobian
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
   nlsq.utils.diagnostics
   nlsq.caching
   nlsq.unified_cache
   nlsq.logging
   nlsq.error_messages
   nlsq.constants
   nlsq.types
   nlsq.result
   nlsq.common_jax
   nlsq.common_scipy

Performance & Profiling
-----------------------

Performance analysis, profiling, and benchmarking tools:

.. toctree::
   :maxdepth: 2

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
- :doc:`nlsq.core.trf_jit` - JIT-compiled TRF functions
- :doc:`nlsq.core.profiler` - TRF performance profiling

**Feature Modules**:
- :doc:`nlsq.functions` - Pre-built fit functions
- :doc:`nlsq.callbacks` - Progress monitoring & early stopping
- :doc:`nlsq.stability` - Numerical stability analysis
- :doc:`nlsq.stability.guard` - Stability guard, condition estimation, collinearity detection
- :doc:`nlsq.fallback` - Automatic retry strategies
- :doc:`nlsq.recovery` - Optimization failure recovery
- :doc:`nlsq.bound_inference` - Smart parameter bounds
- :doc:`nlsq.parameter_estimation` - Initial parameter estimation
- :doc:`nlsq.algorithm_selector` - Automatic algorithm selection

**Large Dataset Modules**:
- :doc:`nlsq.large_dataset` - Chunked fitting for large data
- :doc:`nlsq.memory_manager` - Intelligent memory management
- :doc:`large_datasets_api` - Comprehensive large dataset guide

**Adaptive Hybrid Streaming Modules**:
- :doc:`nlsq.adaptive_hybrid_streaming` - Four-phase hybrid optimizer
- :doc:`nlsq.hybrid_streaming_config` - Configuration with presets
- :doc:`nlsq.parameter_normalizer` - Parameter normalization for gradient balance
- :doc:`nlsq.streaming.telemetry` - Defense layer telemetry
- :doc:`nlsq.streaming.validators` - Configuration validators
- :doc:`nlsq.streaming.phases` - Phase classes (WarmupPhase, GaussNewtonPhase, etc.)
- :doc:`nlsq.streaming.exact_lm` - ``ExactSolverError`` for ``curve_fit_large(method="exact_streaming")``
- :doc:`nlsq.streaming.large_fit` - Five strategy types (``Auto``, ``InMemory``, ``Chunked``, ``Hybrid``, ``Exact``), the ``Multistart`` settings object, and ``fit_with_strategy`` behind ``curve_fit_large(strategy=...)``

**Global Optimization Modules**:
- :doc:`nlsq.global_optimization` - Multi-start with LHS, Sobol, Halton samplers

**Workflow System Modules**:
- :doc:`nlsq.workflow` - Unified fit() with auto-selection, presets, YAML config

**Command-Line Interface Modules**:
- :doc:`nlsq.cli` - CLI commands: ``nlsq fit``, ``nlsq batch``, ``nlsq info``, ``nlsq gui``

**Qt Desktop GUI**:
- :doc:`nlsq.gui_qt` - Native desktop application with PySide6 and pyqtgraph

**Utility Modules**:
- :doc:`nlsq.config` - Configuration management
- :doc:`nlsq.device` - GPU detection and warnings
- :doc:`nlsq.validators` - Input validation
- :doc:`nlsq.diagnostics` - Model Health Diagnostics System (identifiability, gradient health, parameter sensitivity)
- :doc:`nlsq.utils.diagnostics` - Convergence monitoring (ConvergenceMonitor, OptimizationDiagnostics)
- :doc:`nlsq.caching` - JIT and result caching
- :doc:`nlsq.unified_cache` - Unified compilation cache
- :doc:`nlsq.logging` - Logging and debugging
- :doc:`nlsq.error_messages` - Standardized error messages
- :doc:`nlsq.constants` - Numerical constants
- :doc:`nlsq.types` - Type definitions
- :doc:`nlsq.result` - Result containers
- :doc:`nlsq.loss_functions` - Robust loss functions
- :doc:`nlsq.svd_fallback` - SVD fallback strategies
- :doc:`nlsq.sparse_jacobian` - Sparse Jacobian support
- :doc:`nlsq.common_jax` - JAX utilities
- :doc:`nlsq.common_scipy` - SciPy compatibility layer

**Performance & Profiling**:
- :doc:`nlsq.profiling` - JAX profiler integration and static analysis
- :doc:`nlsq.profiler` - Performance profiler
- :doc:`nlsq.profiler_visualization` - Profiling visualization
- :doc:`performance_benchmarks` - Measured timings (same-device comparisons)
