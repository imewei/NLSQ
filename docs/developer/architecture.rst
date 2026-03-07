Architecture Overview
=====================

This document provides a comprehensive architectural overview of NLSQ, a GPU/TPU-accelerated
nonlinear least squares curve fitting library built on JAX. The codebase consists of
approximately 76,000 lines of Python code organized into well-separated modules.

Package Structure
-----------------

The ``nlsq`` package is organized into logical subpackages:

.. code-block:: text

   nlsq/ (~76,000 lines)
   ├── core/           13,705 lines - Core optimization (curve_fit, TRF, LeastSquares)
   │   ├── orchestration/  1,432 lines - Decomposed CurveFit components (v0.6.4)
   │   └── adapters/       Protocol adapters for dependency injection
   ├── streaming/       9,082 lines - Large dataset handling, 4-phase optimizer
   │   └── phases/      2,467 lines - WarmupPhase, GaussNewtonPhase, Checkpoint
   ├── gui_qt/        ~11,200 lines - Native Qt desktop GUI (PySide6/pyqtgraph)
   ├── cli/             7,176 lines - Command-line interface with security
   │   ├── commands/     Subcommands (fit, batch, config, info)
   │   └── templates/    Custom model scaffolding
   ├── utils/           5,127 lines - Validators, logging, profiling, serialization
   ├── global_optimization/ 4,256 lines - CMA-ES, multi-start, tournament selection
   ├── diagnostics/     4,039 lines - Model health analysis, plugin system
   ├── caching/         3,481 lines - JIT caching, memory management, pooling
   ├── precision/       3,338 lines - Algorithm selection, parameter normalization
   ├── stability/       2,760 lines - Numerical robustness, fallbacks, recovery
   ├── interfaces/      1,306 lines - Protocol definitions for DI
   ├── result/          1,221 lines - OptimizeResult, CurveFitResult
   ├── facades/           385 lines - Lazy-loading dependency breakers
   └── (root)          ~4,850 lines - Config, callbacks, types, device, constants


Architectural Layers
--------------------

The following diagram illustrates the layered architecture of NLSQ:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                             USER INTERFACES                                  │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  Qt GUI (PySide6)       CLI (Click)            Python API                    │
   │  ├── 5-page workflow    ├── Model validation   ├── curve_fit(), fit()        │
   │  ├── pyqtgraph plots    ├── Batch fitting      ├── CurveFit class            │
   │  └── Native desktop     └── Export formats     └── LargeDatasetFitter        │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                        OPTIMIZATION ORCHESTRATION                            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  Orchestration (v0.6.4)   Global Optimization    Streaming Optimizer         │
   │  ├── DataPreprocessor     ├── CMAESOptimizer     ├── AdaptiveHybrid (4550L)  │
   │  ├── OptimizationSelector ├── MultiStartOrch.    ├── 4-Phase Pipeline:       │
   │  ├── CovarianceComputer   ├── TournamentSelect   │   0: Normalization        │
   │  └── StreamingCoordinator ├── LHS/Sobol/Halton   │   1: L-BFGS warmup       │
   │                           └── MethodSelector      │   2: Gauss-Newton         │
   │  Facades (circular dep    Presets: fast/robust/   └── 3: Denormalization      │
   │   breakers):              global/thorough/                                    │
   │  ├── OptimizationFacade   streaming/cmaes-*                                  │
   │  ├── StabilityFacade                                                         │
   │  └── DiagnosticsFacade                                                       │
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
   │  ├── svd_fallback     ├── bound_inference ├── smart_cache   ├── health_rep.  │
   │  ├── recovery         └── normalizer      ├── memory_mgr    └── plugin sys.  │
   │  └── robust_decomp                        ├── memory_pool                    │
   │                                           └── compilation                    │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                            INFRASTRUCTURE                                    │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │  interfaces/ (Protocols)     config.py (Singleton)    Security               │
   │  ├── OptimizerProtocol       ├── JAXConfig            ├── safe_serialize     │
   │  ├── CurveFitProtocol        │   (x64, GPU config)    │   (JSON-based)       │
   │  ├── CacheProtocol           ├── MemoryConfig         ├── model_validation   │
   │  ├── DataSourceProtocol      ├── LargeDatasetConfig   │   (AST-based)        │
   │  ├── JacobianProtocol        └── LargeDatasetConfig   └── resource limits    │
   │  ├── Orchestration Protocols                                                 │
   │  │   (DataPreprocessor,      Feature Flags                                   │
   │  │    OptimizationSelector,  ├── NLSQ_*_IMPL envvars                         │
   │  │    CovarianceComputer,    ├── Hash-based rollout                           │
   │  │    StreamingCoordinator)  └── Safe defaults (old)                          │
   │  └── ResultProtocol                                                          │
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
            │   (460 lines)
            ▼
       SVD-based trust region subproblem solver

CurveFit Decomposition (v0.6.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The monolithic ``CurveFit`` class has been decomposed into 4 focused components,
enabled by feature flags for gradual rollout:

.. code-block:: text

   curve_fit() call
       │
       ▼
   FeatureFlags.from_env()          Check NLSQ_*_IMPL environment variables
       │
       ├── (old path) ──→ Original CurveFit monolith in minpack.py
       │
       └── (new path) ──→ Orchestration Components:
           │
           ├── 1. DataPreprocessor.preprocess()         (299 lines)
           │      Raw data → PreprocessedData (validated JAX arrays)
           │      NaN handling, sigma validation, finiteness checks
           │
           ├── 2. OptimizationSelector.select()         (343 lines)
           │      User params → OptimizationConfig
           │      Method selection, bounds prep, initial guess, solver choice
           │
           ├── 3. StreamingCoordinator.decide()          (356 lines)
           │      Dataset size → StreamingDecision
           │      Memory analysis, strategy routing (direct/chunked/hybrid)
           │
           ├── 4. [Optimization via LeastSquares]
           │
           └── 5. CovarianceComputer.compute()           (342 lines)
                  OptimizeResult → CovarianceResult
                  SVD-based pcov, sigma transform, condition estimation

Feature flags use ``NLSQ_PREPROCESSOR_IMPL``, ``NLSQ_SELECTOR_IMPL``,
``NLSQ_COVARIANCE_IMPL``, ``NLSQ_STREAMING_IMPL`` environment variables with values
``'old'``, ``'new'``, or ``'auto'`` (with hash-based rollout percentage).

Key Files
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - File
     - Lines
     - Purpose
   * - ``core/minpack.py``
     - 4,541
     - curve_fit(), CurveFit class, fit() unified entry point
   * - ``core/trf.py``
     - 2,806
     - TrustRegionReflective algorithm
   * - ``core/least_squares.py``
     - 1,559
     - LeastSquares orchestrator
   * - ``core/workflow.py``
     - 1,206
     - Workflow system for automatic strategy selection
   * - ``core/trf_jit.py``
     - 460
     - JIT-compiled TRF helper functions
   * - ``core/orchestration/``
     - 1,432
     - Decomposed CurveFit components (v0.6.4)
   * - ``core/feature_flags.py``
     - —
     - Feature flag system for gradual component rollout
   * - ``core/loss_functions.py``
     - 397
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

- **AdaptiveHybridStreamingOptimizer** (``adaptive_hybrid.py``, 4,550 lines): Main 4-phase optimizer
- **LargeDatasetFitter** (``large_dataset.py``, 2,629 lines): Memory-aware automatic chunking
- **HybridStreamingConfig** (``hybrid_config.py``, 893 lines): Extensive configuration
- **streaming/phases/** subpackage (2,467 lines):

  - ``WarmupPhase`` (885 lines): L-BFGS warmup with adaptive switching
  - ``GaussNewtonPhase`` (712 lines): Chunked J^T J accumulation
  - ``CheckpointManager`` (464 lines): Fault-tolerant state persistence
  - ``PhaseOrchestrator`` (338 lines): Phase transition management

Memory Management
~~~~~~~~~~~~~~~~~

- Power-of-2 bucket sizes eliminate JIT recompilation: 1024, 2048, 4096, ..., 131072
- psutil for system memory detection with 16GB default fallback
- Automatic chunk size calculation based on available memory


Facades
-------

The ``facades/`` package (385 lines) breaks circular import dependencies by deferring
heavy imports to function-call time. Each facade provides lazy accessors for a
subsystem that would otherwise create import cycles with ``core/minpack.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Facade
     - Deferred Subsystem
     - Key Accessors
   * - ``OptimizationFacade``
     - ``global_optimization/``
     - ``get_cmaes_optimizer()``, ``get_multistart_optimizer()``
   * - ``StabilityFacade``
     - ``stability/``
     - ``get_fallback_svd()``, ``get_stability_guard()``, ``get_recovery_handler()``
   * - ``DiagnosticsFacade``
     - ``diagnostics/``
     - ``get_convergence_monitor()``, ``get_diagnostics_config()``


Protocol-Based Dependency Injection
-----------------------------------

The ``interfaces/`` package (1,306 lines) provides Protocol definitions enabling loose coupling:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Protocol
     - Purpose
   * - ``OptimizerProtocol``
     - Base optimizer interface
   * - ``LeastSquaresOptimizerProtocol``
     - Extended for least squares problems
   * - ``CurveFitProtocol``
     - curve_fit-like interfaces
   * - ``CacheProtocol``
     - Caching mechanisms
   * - ``BoundedCacheProtocol``
     - Memory-bounded caches
   * - ``DataSourceProtocol``
     - Data sources (arrays, HDF5)
   * - ``StreamingDataSourceProtocol``
     - Streaming data sources with iterator
   * - ``JacobianProtocol``
     - Jacobian computation strategies
   * - ``SparseJacobianProtocol``
     - Sparse Jacobian handling
   * - ``ResultProtocol``
     - Optimization results
   * - ``LeastSquaresResultProtocol``
     - Extended result with cost, nfev, njev
   * - ``CurveFitResultProtocol``
     - popt/pcov with dict-like access

Orchestration Protocols (v0.6.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``orchestration_protocol.py`` module (503 lines) defines protocols and frozen
dataclasses for the decomposed CurveFit components:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Protocol
     - Output Dataclass
     - Purpose
   * - ``DataPreprocessorProtocol``
     - ``PreprocessedData``
     - Input validation, NaN handling
   * - ``OptimizationSelectorProtocol``
     - ``OptimizationConfig``
     - Method/solver selection
   * - ``CovarianceComputerProtocol``
     - ``CovarianceResult``
     - SVD-based covariance
   * - ``StreamingCoordinatorProtocol``
     - ``StreamingDecision``
     - Memory-aware strategy routing

All protocols use ``@runtime_checkable`` for structural subtyping without explicit inheritance.


Result Types
------------

The ``result/`` package (1,221 lines) provides optimization result containers:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Purpose
   * - ``OptimizeResult``
     - dict subclass with attribute access, SciPy-compatible
   * - ``OptimizeResultV2``
     - Frozen dataclass with ``__slots__`` (~40% memory reduction, ~2x faster access)
   * - ``CurveFitResult``
     - Enhanced result with R-squared, RMSE, AIC/BIC, confidence intervals, prediction bands, ``summary()``, ``plot()``
   * - ``OptimizeWarning``
     - Custom UserWarning for non-critical optimization warnings


Caching and Performance
-----------------------

Multi-Tier Caching
~~~~~~~~~~~~~~~~~~

**UnifiedCache** (``unified_cache.py``, 562 lines):

- Shape-relaxed cache keys: ``(func_hash, dtype, rank)`` instead of full shapes
- LRU eviction with configurable maxsize (default: 128)
- Weak references to prevent memory leaks
- Target: 80%+ cache hit rate

**SmartCache** (``smart_cache.py``, 713 lines):

- xxhash for 10x faster hashing than SHA256
- Stride-based sampling for arrays >10K elements
- Safe JSON serialization

**MemoryManager** (``memory_manager.py``, 932 lines):

- LRU array pooling via OrderedDict
- psutil for system memory detection
- Telemetry circular buffer (deque maxlen=1000) for multi-day runs

**MemoryPool** (``memory_pool.py``, 421 lines):

- Array reuse with power-of-2 bucket sizing


Numerical Stability
-------------------

Stability Guard
~~~~~~~~~~~~~~~

``NumericalStabilityGuard`` (``stability/guard.py``, 1,159 lines) provides three modes:

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

- **SVD Fallback** (``fallback.py``, 533 lines): GPU/CPU fallback SVD with randomized SVD for large matrices
- **Recovery** (``recovery.py``, 419 lines): Optimization recovery after trust region failures
- **Robust Decomposition** (``robust_decomposition.py``, 480 lines): Numerically robust matrix decompositions


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
     - AST-based dangerous pattern detection
   * - Path Traversal
     - ``validate_path()``
     - Relative path containment
   * - Resource Limits
     - ``resource_limits()``
     - RLIMIT_AS + SIGALRM timeout
   * - Audit Logging
     - ``AuditLogger``
     - RotatingFileHandler (10MB, 90 days)

Blocked patterns include: ``exec``, ``eval``, ``subprocess``, ``socket``, ``ctypes``,
and other dangerous builtins and module calls.


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

The ``cli/`` package (7,176 lines) provides a Click-based command-line interface:

.. code-block:: text

   cli/
   ├── main.py              - Click group entry point (nlsq)
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

The ``gui_qt/`` package (~11,200 lines) provides a native Qt desktop application:

.. code-block:: text

   gui_qt/
   ├── __init__.py         - run_desktop() entry point
   ├── main_window.py      - MainWindow with sidebar navigation (469 lines)
   ├── app_state.py        - AppState (Qt signals wrapping SessionState)
   ├── session_state.py    - SessionState dataclass (541 lines)
   ├── theme.py            - ThemeConfig, ThemeManager (light/dark)
   ├── autosave.py         - AutosaveManager for crash recovery
   ├── presets.py           - Workflow presets
   ├── pages/               - 5-page workflow (QWidget-based)
   │   ├── data_loading.py     (625 lines)
   │   ├── model_selection.py  (531 lines)
   │   ├── fitting_options.py  (505 lines)
   │   ├── results.py          (599 lines)
   │   └── export.py           (784 lines)
   ├── widgets/             - Reusable Qt widgets
   │   ├── code_editor.py      (460 lines)
   │   ├── advanced_options.py (383 lines)
   │   ├── column_selector.py  (306 lines)
   │   ├── param_results.py
   │   ├── iteration_table.py
   │   └── fit_statistics.py
   ├── plots/               - pyqtgraph-based scientific plots
   │   ├── fit_plot.py         (316 lines)
   │   ├── residuals_plot.py   (353 lines)
   │   ├── histogram_plot.py   (246 lines)
   │   └── base_plot.py
   └── adapters/            - NLSQ-GUI bridge
       ├── data_adapter.py     (571 lines)
       ├── fit_adapter.py      (725 lines)
       ├── model_adapter.py    (754 lines)
       ├── export_adapter.py   (353 lines)
       └── config_adapter.py   (387 lines)

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
   * - Protocol-Based DI
     - ``interfaces/`` — structural subtyping without inheritance
   * - Factory
     - ``create_optimizer()``, ``configure_curve_fit()``
   * - Singleton
     - ``JAXConfig``, ``FeatureFlags``, global caches
   * - State Machine
     - ``PrecisionState`` for optimization state tracking
   * - Phased Pipeline
     - 4-phase streaming optimizer
   * - Lazy Loading
     - ``__getattr__`` in ``__init__.py``, ``orchestration/``, ``facades/`` (50%+ import reduction)
   * - Facade
     - Break circular dependencies (``facades/``) with deferred imports
   * - Adapter
     - GUI adapters bridge NLSQ to Qt widgets; ``CurveFitAdapter`` for protocol compliance
   * - Feature Flags
     - ``FeatureFlags`` for gradual rollout of decomposed CurveFit components


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

Decomposed Path (v0.6.4, feature-flagged)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit(f, x, y, p0)
       → FeatureFlags.from_env()
       → DataPreprocessor.preprocess()       → PreprocessedData
       → OptimizationSelector.select()       → OptimizationConfig
       → StreamingCoordinator.decide()       → StreamingDecision
           ├── 'direct' → LeastSquares.least_squares()
           ├── 'chunked' → ChunkedOptimizer
           └── 'hybrid' → AdaptiveHybridStreamingOptimizer
       → CovarianceComputer.compute()        → CovarianceResult
       → CurveFitResult (popt, pcov, statistics)

Large Dataset (Streaming)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → fit(f, x, y, workflow='streaming')
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

   User Input → curve_fit(f, x, y, method='cmaes')
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

1. **Lazy Imports**: 50%+ reduction in cold import time via ``__getattr__`` in package init, orchestration, and facades
2. **Shape-Relaxed Cache Keys**: Cache by ``(hash, dtype, rank)`` not exact shapes
3. **Power-of-2 Bucketing**: Static array shapes for JIT efficiency
4. **xxhash**: 10x faster hashing than SHA256
5. **LRU Pooling**: Array reuse via OrderedDict (``memory_pool.py``)
6. **TTL-cached psutil**: Reduce memory detection overhead
7. **JAX Array Updates**: Functional ``jax.numpy`` updates instead of NumPy copies
8. **XLA Fusion**: Inlined residual functions for better GPU kernel fusion
9. **Gradient Caching**: Consolidated gradient norm computation to avoid redundant calculation
10. **Logging Guards**: ``isEnabledFor()`` checks prevent overhead when logging disabled


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
   * - ``NLSQ_PREPROCESSOR_IMPL``
     - DataPreprocessor: ``'old'``, ``'new'``, or ``'auto'``
   * - ``NLSQ_SELECTOR_IMPL``
     - OptimizationSelector: ``'old'``, ``'new'``, or ``'auto'``
   * - ``NLSQ_COVARIANCE_IMPL``
     - CovarianceComputer: ``'old'``, ``'new'``, or ``'auto'``
   * - ``NLSQ_STREAMING_IMPL``
     - StreamingCoordinator: ``'old'``, ``'new'``, or ``'auto'``


Configuration System
--------------------

The ``config.py`` module (1,159 lines) provides a singleton ``JAXConfig`` that manages:

- **JAX initialization**: x64 enabled, GPU memory configuration
- **MemoryConfig**: Memory limits, chunk sizes, out-of-memory strategies
- **LargeDatasetConfig**: Solver selection thresholds (direct: 100K, iterative: 10M, chunked: 100M)

All configuration is validated at instantiation time with descriptive error messages.


See Also
--------

- :doc:`optimization_case_study` — Performance optimization deep dive
- :doc:`performance_tuning_guide` — Practical tuning recommendations
- `Architecture Decision Records <adr/README.html>`_
