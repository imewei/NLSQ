Architecture Overview
=====================

This document provides a comprehensive architectural overview of NLSQ, a GPU/TPU-accelerated
nonlinear least squares curve fitting library built on JAX. The codebase consists of
approximately 78,000 lines of Python code organized into well-separated modules.

Package Structure
-----------------

The ``nlsq`` package is organized into logical subpackages:

.. code-block:: text

   nlsq/ (~78,000 lines)
   ├── core/           15,149 lines - Core optimization (curve_fit, TRF, LeastSquares)
   │   ├── minpack.py        curve_fit(), fit(), CurveFit
   │   ├── problem.py        prepare_problem(): input validation, p0/bounds, padding
   │   ├── covariance.py     pcov_from_jacobian() / pcov_from_normal()
   │   ├── least_squares.py  LeastSquares, AutoDiffJacobian
   │   ├── trf.py, trf_jit.py  Trust Region Reflective optimizer and JIT kernels
   │   ├── workflow.py       Workflow system (auto / auto_global / hpc)
   │   └── device_policy.py  Host-device placement for small SVDs under a GPU backend
   ├── streaming/      14,680 lines - Large dataset handling, 4-phase optimizer
   │   ├── specs.py, strategy.py  Strategy specs and build_large_strategy()
   │   ├── normal_equations.py, exact_lm.py  Exact chunk-accumulated LM
   │   └── phases/      2,748 lines - WarmupPhase, GaussNewtonPhase, Checkpoint
   ├── gui_qt/         12,348 lines - Native Qt desktop GUI (PySide6/pyqtgraph)
   ├── cli/             8,085 lines - Command-line interface with security
   │   ├── commands/     Subcommands (fit, batch, config, info)
   │   └── templates/    Custom model scaffolding
   ├── utils/           5,446 lines - Validators, logging, profiling, serialization
   ├── global_optimization/ 5,501 lines - CMA-ES, multi-start, tournament selection
   ├── diagnostics/     4,665 lines - Model health analysis, plugin system
   ├── stability/       2,598 lines - Numerical robustness, fallbacks, recovery
   ├── precision/       2,200 lines - Algorithm selection, parameter normalization
   ├── caching/         1,521 lines - JIT caching, memory management
   ├── result/          1,180 lines - OptimizeResult, CurveFitResult
   └── (root)          ~4,911 lines - Config, callbacks, types, device, constants


Architectural Layers
--------------------

The following diagram illustrates the layered architecture of NLSQ:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                             USER INTERFACES                                  │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  Qt GUI (PySide6)       CLI (argparse)         Python API                    │
   │  ├── 5-page workflow    ├── Model validation   ├── curve_fit(), fit()        │
   │  ├── pyqtgraph plots    ├── Batch fitting      ├── CurveFit class            │
   │  └── Native desktop     └── Export formats     └── LargeDatasetFitter        │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                        OPTIMIZATION ORCHESTRATION                            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  Problem preparation      Global Optimization    Streaming Optimizer         │
   │  ├── prepare_problem()    ├── CMAESOptimizer     ├── AdaptiveHybrid          │
   │  └── pcov_from_*()        ├── MultiStartOrch.    ├── 4-Phase Pipeline:       │
   │                           ├── TournamentSelect   │   0: Normalization        │
   │  Large-fit strategies     ├── LHS/Sobol/Halton   │   1: L-BFGS warmup       │
   │  ├── specs.py             └── MethodSelector      │   2: Gauss-Newton         │
   │  ├── strategy.py          Presets: fast/robust/   └── 3: Denormalization      │
   │  └── exact_lm.py          global/thorough/streaming                          │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                          CORE OPTIMIZATION ENGINE                            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  curve_fit() ──→ CurveFit ──→ LeastSquares ──→ TrustRegionReflective        │
   │  (minpack.py)    (minpack.py)  (least_squares.py)  (trf.py)                  │
   │       │                │                │                │                    │
   │       ▼                ▼                ▼                ▼                    │
   │  API Wrapper      Cache + State   Orchestrator + AD   SVD-based TRF          │
   │  (SciPy-compat)   (UnifiedCache)  (AutoDiffJacobian)  (trf_jit.py)           │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                          SUPPORT SUBSYSTEMS                                  │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  stability/           precision/          caching/          diagnostics/     │
   │  ├── guard.py         ├── algorithm_sel   ├── unified_cache ├── identifiab.  │
   │  │   NumericalStab.   │   Problem-size    │   Shape-relaxed ├── gradient     │
   │  │   Guard (3 modes)  │   aware selection │   LRU, weak refs├── param_sens.  │
   │  ├── svd_fallback     ├── bound_inference ├── memory_mgr    ├── health_rep.  │
   │  ├── fallback         └── normalizer      └── closure keys  └── plugin sys.  │
   │  └── recovery                                                                │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                            INFRASTRUCTURE                                    │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  config.py (Singleton)       Security                                        │
   │  ├── JAXConfig               ├── safe_serialize (JSON-based)                 │
   │  │   (x64, GPU config)       ├── model_validation (AST-based)                │
   │  ├── MemoryConfig            └── resource limits                             │
   │  └── LargeDatasetConfig                                                      │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                         JAX RUNTIME (0.8.0)                                  │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  ├── x64 enabled (double precision)  ├── JIT compilation with cache          │
   │  ├── Automatic differentiation       └── GPU/TPU backend (optional)          │
   └──────────────────────────────────────────────────────────────────────────────┘


Core Optimization Pipeline
--------------------------

Class Hierarchy
~~~~~~~~~~~~~~~

The core optimization pipeline follows this class hierarchy:

.. code-block:: text

   curve_fit() / fit()           Entry points (minpack.py)
            │
            ▼
       CurveFit                  Main curve fitting class (minpack.py)
            │                    - SciPy-compatible API wrapper
            ├── UnifiedCache     - Fixed-length padding for JIT
            │                    - Stability/recovery options
            ▼
       LeastSquares              Optimization orchestrator (least_squares.py)
            │                    - Algorithm selection (TRF, LM)
            ├── AutoDiffJacobian - Three Jacobian handlers (no sigma, 1D sigma, 2D cov)
            ├── LossFunctionsJIT - Bound constraint processing
            │
            ▼
       TrustRegionReflective     Main optimizer (trf.py)
            │                    - Inherits: TrustRegionJITFunctions + TrustRegionOptimizerBase
            ├── CommonJIT        - Variable scaling for bounds
            ├── trf_jit.py       - Exact (SVD) and iterative (CG) solvers
            │   (493 lines)
            ▼
       SVD-based trust region subproblem solver

Problem Preparation and Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two single-purpose modules sit on either side of the optimizer:

.. code-block:: text

   curve_fit() call
       │
       ▼
   prepare_problem()          core/problem.py
       │                      Fixed step order: parameter count / p0 / bounds,
       │                      validation and NaN handling, method selection,
       │                      mask construction and padding → PreparedProblem
       ▼
   LeastSquares.least_squares()  → OptimizeResult
       │
       ▼
   pcov_from_jacobian()       core/covariance.py
                              SVD-based pcov with rank cutoff, sigma transform

``pcov_from_normal()`` (same module) computes the covariance from the normal matrix
for the exact streaming solver; it Jacobi-equilibrates ``J^T J`` before the rank cutoff
(when full rank after scaling) so the result does not depend on parameter units. ``core/device_policy.py`` runs small exact SVDs on the
host CPU device when the active backend is a GPU, where launch overhead dominates.

Key Files
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - File
     - Lines
     - Purpose
   * - ``core/minpack.py``
     - 4,922
     - curve_fit(), CurveFit class, fit() unified entry point
   * - ``core/trf.py``
     - 2,676
     - TrustRegionReflective algorithm
   * - ``core/least_squares.py``
     - 2,188
     - LeastSquares orchestrator
   * - ``core/workflow.py``
     - 1,318
     - Workflow system for automatic strategy selection
   * - ``core/trf_jit.py``
     - 493
     - JIT-compiled TRF helper functions
   * - ``core/problem.py``
     - 1,037
     - ``prepare_problem()``: input validation, p0/bounds, NaN handling, padding
   * - ``core/covariance.py``
     - 169
     - ``pcov_from_jacobian()`` / ``pcov_from_normal()``
   * - ``core/loss_functions.py``
     - 405
     - JIT-compiled loss functions


Streaming Optimization
----------------------

Four-Phase Pipeline
~~~~~~~~~~~~~~~~~~~

The streaming subsystem implements a sophisticated four-phase optimization strategy
for datasets up to 100M+ points:

.. list-table::
   :header-rows: 1
   :widths: 10 20 25 45

   * - Phase
     - Name
     - Algorithm
     - Purpose
   * - 0
     - Normalization
     - ParameterNormalizer
     - Scale parameters to similar ranges
   * - 1
     - Warmup
     - L-BFGS (optax)
     - Fast initial convergence
   * - 2
     - Gauss-Newton
     - Streaming J^T J
     - Precision near optimum
   * - 3
     - Finalization
     - Denormalization
     - Covariance transform

Key Components
~~~~~~~~~~~~~~

- **AdaptiveHybridStreamingOptimizer** (``adaptive_hybrid.py``, 4,498 lines): Main 4-phase optimizer
- **LargeDatasetFitter** (``large_dataset.py``, 3,245 lines): Memory-aware automatic chunking
- **HybridStreamingConfig** (``hybrid_config.py``, 881 lines): Extensive configuration
- **exact_streaming** (``exact_lm.py``, 29K; ``normal_equations.py``): exact full-data
  Levenberg-Marquardt that accumulates ``J^T J`` and ``J^T r`` over chunks
  (``curve_fit_large(..., method="exact_streaming")``)
- **streaming/phases/** subpackage (2,748 lines):

  - ``WarmupPhase``: L-BFGS warmup with adaptive switching
  - ``GaussNewtonPhase``: Chunked J^T J accumulation
  - ``CheckpointManager``: Fault-tolerant state persistence
  - ``PhaseOrchestrator``: Phase transition management

Memory Management
~~~~~~~~~~~~~~~~~

- Power-of-2 bucket sizes eliminate JIT recompilation: 1024, 2048, 4096, ..., 131072
- psutil for system memory detection with 16GB default fallback
- Automatic chunk size calculation based on available memory


Large-Dataset Strategies
------------------------

Large problems run through one of several engines; two small modules describe and
build the chosen path:

- ``streaming/specs.py``: frozen dataclasses (``Multistart``, ``InMemory``,
  ``Chunked``, ``Hybrid``, ``Exact``, ``Auto``) that only declare configuration and
  import no engines.
- ``streaming/strategy.py``: ``build_large_strategy(kind, config, **common)``
  constructs the engine for a path the caller has already chosen and never decides
  routing:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - ``kind``
     - Engine
   * - ``"hybrid"``
     - ``AdaptiveHybridStreamingOptimizer``
   * - ``"chunked"``
     - ``LargeDatasetFitter``
   * - ``"exact"``
     - ``exact_lm_fit`` (exact chunk-accumulated LM)
   * - ``"in_memory"``
     - ``nlsq.curve_fit``


Result Types
------------

The ``result/`` package (1,180 lines) provides optimization result containers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Purpose
   * - ``OptimizeResult``
     - dict subclass with attribute access, SciPy-compatible
   * - ``CurveFitResult``
     - Enhanced result with R-squared, RMSE, AIC/BIC, confidence intervals, prediction bands, ``summary()``, ``plot()``
   * - ``OptimizeWarning``
     - Custom UserWarning for non-critical optimization warnings


Caching and Performance
-----------------------

Multi-Tier Caching
~~~~~~~~~~~~~~~~~~

**UnifiedCache** (``unified_cache.py``, 543 lines):

- Shape-relaxed cache keys: ``(func_hash, dtype, rank)`` instead of full shapes
- LRU eviction with configurable maxsize (default: 128)
- Weak references to prevent memory leaks
- Target: 80%+ cache hit rate

**MemoryManager** (``memory_manager.py``, 895 lines):

- LRU array pooling via OrderedDict
- psutil for system memory detection
- Telemetry circular buffer (deque maxlen=1000) for multi-day runs


Numerical Stability
-------------------

Stability Guard
~~~~~~~~~~~~~~~

``NumericalStabilityGuard`` (``stability/guard.py``, 1,259 lines) provides three modes:

- ``stability=False``: No checks (maximum performance)
- ``stability='check'``: Warn only, no modifications
- ``stability='auto'``: Detect and fix numerical issues

Key thresholds:

- Condition number threshold: 1e12
- SVD skip for >10M Jacobian elements
- Tikhonov regularization factor: 1e-10

Fallback Chain
~~~~~~~~~~~~~~

The solver uses a JAX JIT-compatible fallback chain:

.. code-block:: text

   Cholesky decomposition
          │
          ▼ (if fails via NaN detection)
   Eigenvalue decomposition
          │
          ▼ (if ill-conditioned)
   Tikhonov regularization

Additional stability components:

- **SVD Fallback** (``svd_fallback.py``, 185 lines): GPU/CPU fallback SVD
- **Recovery** (``recovery.py``, 433 lines): Optimization recovery after trust region failures
- **FallbackOrchestrator** (``fallback.py``, 684 lines): Ordered retry strategies (alternative method, perturbed ``p0``, adjusted tolerances, bounds, robust loss, rescaling)


Precision Modules
-----------------

The ``precision/`` package provides solver selection and parameter management:

- **AlgorithmSelector** (``algorithm_selector.py``, 625 lines): Problem-size-aware solver selection
- **BoundInference** (``bound_inference.py``, 548 lines): Automatic bound inference
- **ParameterNormalizer** (``parameter_normalizer.py``): Parameter scaling for numerical stability


Global Optimization
-------------------

GPU-accelerated global optimization for escaping local minima:

CMA-ES Optimizer
~~~~~~~~~~~~~~~~

- **CMAESOptimizer** (``cmaes_optimizer.py``, 936 lines): evosax-based CMA-ES with BIPOP restart strategy
- CMA-ES global search followed by TRF local refinement for proper covariance estimation
- Presets: ``'cmaes-fast'``, ``'cmaes'``, ``'cmaes-global'``
- Sigmoid bounds transform for unbounded CMA-ES to bounded parameter space

Multi-Start Search
~~~~~~~~~~~~~~~~~~

- **MultiStartOrchestrator** (``multi_start.py``, 679 lines): Parallel evaluation of starting points
- Thread-safe with per-thread CurveFit isolation
- Adaptive worker count based on hardware (GPU count, CPU cores)
- Presets: ``'fast'``, ``'robust'``, ``'global'``, ``'thorough'``, ``'streaming'``

Tournament Selection
~~~~~~~~~~~~~~~~~~~~

- **TournamentSelector** (``tournament.py``, 559 lines): Progressive N to N/2 to top M elimination
- Memory-efficient evaluation on data batches without loading full dataset
- Checkpoint/resume support for fault tolerance

Samplers
~~~~~~~~

- **Latin Hypercube** (LHS): Stratified random sampling
- **Sobol**: Low-discrepancy quasi-random (up to 21 dimensions)
- **Halton**: Prime-base quasi-random (up to 20 dimensions)

**MethodSelector** (``method_selector.py``): Auto-selects CMA-ES vs multi-start based on
parameter scale ratio and evosax availability.

Integration strategy by dataset size:

- Small (<1M points): Full multi-start on complete data
- Medium (1M-100M): Full multi-start, then chunked fit
- Large (>100M): Tournament selection during streaming warmup


Security Architecture
---------------------

NLSQ implements comprehensive security measures:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Component
     - Location
     - Protection
   * - Safe Serialization
     - ``utils/safe_serialize.py``
     - JSON-based, CWE-502 mitigation
   * - Model Validation
     - ``cli/model_validation.py``
     - AST-based dangerous pattern denylist (best-effort, not a sandbox)
   * - Path Traversal
     - ``validate_path()``
     - Relative path containment
   * - Resource Limits
     - ``resource_limits()``
     - SIGALRM timeout; RLIMIT_AS cap (relative to current VSZ), skipped when a GPU/TPU backend is active
   * - Audit Logging
     - ``AuditLogger``
     - RotatingFileHandler (10MB per file, at most 10 backups)

Blocked patterns include: ``exec``, ``eval``, ``subprocess``, ``socket``, ``ctypes``,
and other dangerous builtins and module calls.

.. warning::

   Model validation is a best-effort static **denylist**, not a sandbox. It
   rejects known-dangerous names, modules, and write-mode ``open()`` calls, but
   it cannot prove that arbitrary Python is safe: Python is too dynamic for a
   name-based check to be complete, and resource limits only wrap module load
   time. Never load a custom model file from an untrusted source; treat
   ``trusted: true`` and a passing validation as equally "you vouched for this
   code", because a model file is ordinary Python executed with your privileges.


Diagnostics System
------------------

Post-fit model health analysis via the ``diagnostics/`` package (4,039 lines):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Analyzer
     - Purpose
   * - ``IdentifiabilityAnalyzer``
     - FIM condition number, rank, correlations
   * - ``GradientMonitor``
     - Vanishing, imbalance, stagnation detection
   * - ``ParameterSensitivityAnalyzer``
     - Eigenvalue spectrum, stiff/sloppy directions
   * - ``HealthReport``
     - Aggregated health summary from all analyzers
   * - ``PluginRegistry``
     - Domain-specific extensions

Usage:

.. code-block:: python

   result = curve_fit(model, x, y, compute_diagnostics=True)
   print(result.diagnostics.summary())


CLI System
----------

The ``cli/`` package (8,085 lines) provides an argparse-based command-line interface:

.. code-block:: text

   cli/
   ├── main.py              - argparse entry point (nlsq)
   ├── commands/
   │   ├── fit.py            - Single fit command
   │   ├── batch.py          - Batch fitting (multiple datasets)
   │   ├── config.py         - Configuration management
   │   └── info.py           - Environment/GPU info
   ├── data_loaders.py       - CSV/JSON/HDF5 data loading
   ├── model_registry.py     - Built-in + custom model discovery
   ├── model_validation.py   - AST-based security validation
   ├── workflow_runner.py     - Orchestration of fit workflows
   ├── visualization.py      - Terminal/matplotlib result display
   ├── result_exporter.py    - JSON/CSV/ZIP export
   ├── errors.py             - Structured error handling
   └── templates/            - Custom model scaffolding


Qt GUI System
-------------

The ``gui_qt/`` package (~12,300 lines) provides a native Qt desktop application:

.. code-block:: text

   gui_qt/
   ├── __init__.py         - run_desktop() entry point
   ├── main_window.py      - MainWindow with sidebar navigation
   ├── app_state.py        - AppState (Qt signals wrapping SessionState)
   ├── session_state.py    - SessionState dataclass
   ├── theme.py            - ThemeConfig, ThemeManager (light/dark)
   ├── autosave.py         - AutosaveManager for crash recovery
   ├── presets.py           - Workflow presets
   ├── pages/               - 5-page workflow (QWidget-based)
   │   ├── data_loading.py
   │   ├── model_selection.py
   │   ├── fitting_options.py
   │   ├── results.py
   │   └── export.py
   ├── widgets/             - Reusable Qt widgets
   │   ├── code_editor.py
   │   ├── advanced_options.py
   │   ├── column_selector.py
   │   ├── param_results.py
   │   ├── iteration_table.py
   │   └── fit_statistics.py
   ├── plots/               - pyqtgraph-based scientific plots
   │   ├── fit_plot.py
   │   ├── residuals_plot.py
   │   ├── histogram_plot.py
   │   └── live_cost_plot.py
   └── adapters/            - NLSQ-GUI bridge
       ├── data_adapter.py
       ├── fit_adapter.py
       ├── model_adapter.py
       └── config_adapter.py

Launch options:

.. code-block:: bash

   # Entry point command
   nlsq-gui

   # Python module
   python -m nlsq.gui_qt

   # Python API
   from nlsq.gui_qt import run_desktop
   run_desktop()


Design Patterns
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Pattern
     - Usage
   * - Strategy
     - Frozen specs in ``streaming/specs.py``; ``build_large_strategy()`` in ``streaming/strategy.py``
   * - Singleton
     - ``JAXConfig``, global caches
   * - Phased Pipeline
     - 4-phase streaming optimizer
   * - Lazy Loading
     - ``__getattr__`` in ``nlsq/__init__.py`` (specialty modules load on first access)
   * - Adapter
     - GUI adapters bridge NLSQ to Qt widgets


Data Flow Diagrams
------------------

Standard Optimization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit(f, x, y, p0)
       → CurveFit.curve_fit()
       → InputValidator.validate()
       → LeastSquares.least_squares()
       → AutoDiffJacobian (JAX autodiff)
       → TrustRegionReflective.trf()
           → JIT-compiled iteration loop
           → SVD for trust region subproblems
       → OptimizeResult → (popt, pcov)

Problem Preparation Path
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit(f, x, y, p0)
       → prepare_problem()                   → PreparedProblem
       → LeastSquares.least_squares()        → OptimizeResult
       → pcov_from_jacobian()                → pcov
       → CurveFitResult (popt, pcov, statistics)

Large Dataset (Streaming)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit_large(f, x, y, p0)
       → build_large_strategy('hybrid', config)
       → AdaptiveHybridStreamingOptimizer
           Phase 0: ParameterNormalizer.setup()
           Phase 1: WarmupPhase (L-BFGS via optax)
                    ├── Adaptive switching criteria
                    └── DefenseLayerTelemetry
           Phase 2: GaussNewtonPhase
                    ├── Chunked J^T J accumulation
                    └── CheckpointManager (fault tolerance)
           Phase 3: Denormalize + covariance transform
       → CurveFitResult

Global Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → fit(f, x, y, p0, bounds=bounds, workflow='auto_global')
       → MethodSelector.select()
           ├── CMAESOptimizer (evosax)
           │   ├── Sigmoid bounds transform
           │   ├── CMA-ES global search (optional BIPOP restarts)
           │   └── TRF local refinement → covariance
           └── MultiStartOrchestrator
               ├── LHS/Sobol/Halton sampling
               ├── Parallel ThreadPoolExecutor evaluation
               └── Best result selection


Performance Optimizations
-------------------------

1. **Lazy Imports**: Reduced cold import time via ``__getattr__`` in package init
2. **Shape-Relaxed Cache Keys**: Cache by ``(hash, dtype, rank)`` not exact shapes
3. **Power-of-2 Bucketing**: Static array shapes for JIT efficiency
4. **LRU Pooling**: Array reuse via OrderedDict (``memory_manager.py``)
5. **TTL-cached psutil**: Reduce memory detection overhead
6. **JAX Array Updates**: Functional ``jax.numpy`` updates instead of NumPy copies
7. **XLA Fusion**: Inlined residual functions for better GPU kernel fusion
8. **Gradient Caching**: Consolidated gradient norm computation to avoid redundant calculation
9. **Logging Guards**: ``isEnabledFor()`` checks prevent overhead when logging disabled


Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Variable
     - Effect
   * - ``NLSQ_FORCE_CPU=1``
     - Force CPU backend for testing
   * - ``NLSQ_SKIP_GPU_CHECK=1``
     - Suppress GPU availability warnings
   * - ``NLSQ_DISABLE_PERSISTENT_CACHE=1``
     - Disable JAX compilation cache
   * - ``NLSQ_DEBUG=1``
     - Enable debug logging

See :doc:`/reference/configuration` for the complete environment-variable reference.


Configuration System
--------------------

The ``config.py`` module (904 lines) provides a singleton ``JAXConfig`` that manages:

- **JAX initialization**: x64 enabled, GPU memory configuration
- **MemoryConfig**: Memory limits, chunk sizes, out-of-memory strategies
- **LargeDatasetConfig**: Solver selection thresholds (direct: 100K, iterative: 10M, chunked: 100M)

All configuration is validated at instantiation time with descriptive error messages.


See Also
--------

- :doc:`optimization_case_study` — Performance optimization deep dive
- :doc:`performance_tuning_guide` — Practical tuning recommendations
- `Architecture Decision Records <adr/README.html>`_
