Architecture Overview
=====================

This document provides a comprehensive architectural overview of NLSQ, a GPU/TPU-accelerated
nonlinear least squares curve fitting library built on JAX. The codebase consists of
approximately 72,000 lines of Python code organized into well-separated modules.

Package Structure
-----------------

The ``nlsq`` package is organized into logical subpackages:

.. code-block:: text

   nlsq/ (~62,000 lines)
   ├── core/           13,667 lines - Core optimization (curve_fit, TRF, LeastSquares)
   ├── streaming/      11,260 lines - Large dataset handling, 4-phase optimizer
   ├── gui_qt/        ~10,000 lines - Native Qt desktop GUI (PySide6/pyqtgraph)
   ├── cli/            4,853 lines - Command-line interface with security
   ├── diagnostics/    4,082 lines - Model health analysis, plugin system
   ├── caching/        3,363 lines - JIT caching, memory management
   ├── precision/      3,312 lines - Mixed precision, algorithm selection
   ├── stability/      2,814 lines - Numerical robustness, fallbacks
   ├── global_optimization/ 2,099 lines - Multi-start search, sampling
   ├── interfaces/       785 lines - Protocol definitions for DI
   ├── result/          ~500 lines - OptimizeResult, CurveFitResult
   ├── utils/         ~3,500 lines - Validators, logging, serialization
   └── (root)         ~4,000 lines - Config, callbacks, types, device

Architectural Layers
--------------------

The following diagram illustrates the layered architecture of NLSQ:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                             USER INTERFACES                                  │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  Qt GUI (PySide6)       CLI (Click)            Python API                   │
   │  ├── 5-page workflow    ├── Model validation   ├── curve_fit(), fit()       │
   │  ├── pyqtgraph plots    ├── Security auditing  ├── CurveFit class           │
   │  └── Native desktop     └── Export formats     └── LargeDatasetFitter       │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                        OPTIMIZATION ORCHESTRATION                            │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  Workflow System         Global Optimization      Streaming Optimizer        │
   │  ├── WorkflowSelector    ├── MultiStartOrch.     ├── AdaptiveHybrid (4376L) │
   │  ├── Tier: STANDARD/     ├── TournamentSelect    ├── 4-Phase Pipeline:      │
   │  │   CHUNKED/STREAMING   ├── LHS/Sobol/Halton    │   0: Normalization       │
   │  └── Goal-based config   └── Sampling            │   1: L-BFGS warmup       │
   │                                                   │   2: Gauss-Newton        │
   │                                                   └── 3: Denormalization     │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                          CORE OPTIMIZATION ENGINE                            │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  curve_fit() ─→ CurveFit ─→ LeastSquares ─→ TrustRegionReflective           │
   │  (minpack.py)    (minpack.py)  (least_squares.py)  (trf.py)                 │
   │       │                │                │                │                   │
   │       ▼                ▼                ▼                ▼                   │
   │  API Wrapper      Cache + State   Orchestrator + AD   SVD-based TRF         │
   │  (SciPy-compat)   (UnifiedCache)  (AutoDiffJacobian)  (trf_jit.py)          │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                          SUPPORT SUBSYSTEMS                                  │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  stability/           precision/          caching/          diagnostics/    │
   │  ├── guard.py         ├── mixed_precision ├── unified_cache ├── identifiab. │
   │  │   NumericalStab.   │   5-state machine │   Shape-relaxed ├── gradient    │
   │  │   Guard (3 modes)  │   float32→float64 │   LRU, weak refs├── param_sens. │
   │  ├── svd_fallback     ├── algorithm_sel   ├── smart_cache   └── plugin sys. │
   │  ├── recovery         ├── bound_inference ├── memory_mgr                    │
   │  └── robust_decomp    └── normalizer      └── compilation                   │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                            INFRASTRUCTURE                                    │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  interfaces/ (Protocols)     config.py (Singleton)    Security              │
   │  ├── OptimizerProtocol       ├── JAXConfig            ├── safe_serialize    │
   │  ├── CurveFitProtocol        │   (x64, GPU config)    │   (JSON, no pickle) │
   │  ├── CacheProtocol           ├── MemoryConfig         ├── model_validation  │
   │  ├── DataSourceProtocol      ├── LargeDatasetConfig   │   (AST-based)       │
   │  └── JacobianProtocol        └── MixedPrecisionCfg    └── resource limits   │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                         JAX RUNTIME (0.8.0)                                  │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │  ├── x64 enabled (double precision)  ├── JIT compilation with cache         │
   │  ├── Automatic differentiation       └── GPU/TPU backend (optional)         │
   └─────────────────────────────────────────────────────────────────────────────┘


Core Optimization Pipeline
--------------------------

Class Hierarchy
~~~~~~~~~~~~~~~

The core optimization pipeline follows this class hierarchy:

.. code-block:: text

   curve_fit() / fit()           Entry points (minpack.py:1103, 155)
            │
            ▼
       CurveFit                  Main curve fitting class (minpack.py:1458)
            │                    - SciPy-compatible API wrapper
            ├── UnifiedCache     - Fixed-length padding for JIT
            │                    - Stability/recovery options
            ▼
       LeastSquares              Optimization orchestrator (least_squares.py:360)
            │                    - Algorithm selection (TRF, LM)
            ├── AutoDiffJacobian - Three Jacobian handlers (no σ, 1D σ, 2D cov)
            ├── LossFunctionsJIT - Bound constraint processing
            │
            ▼
       TrustRegionReflective     Main optimizer (trf.py:423)
            │                    - Inherits: TrustRegionJITFunctions + TrustRegionOptimizerBase
            ├── CommonJIT        - Variable scaling for bounds
            ├── trf_jit.py       - Exact (SVD) and iterative (CG) solvers
            │   (478 lines)
            ▼
       SVD-based trust region subproblem solver

Key Files
~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - File
     - Lines
     - Purpose
   * - ``core/minpack.py``
     - 3,135
     - curve_fit(), CurveFit class, fit() unified entry point
   * - ``core/trf.py``
     - 3,272
     - TrustRegionReflective algorithm
   * - ``core/workflow.py``
     - 2,437
     - Workflow system for automatic strategy selection
   * - ``core/least_squares.py``
     - 1,523
     - LeastSquares orchestrator
   * - ``core/trf_jit.py``
     - 478
     - JIT-compiled TRF helper functions


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

- **AdaptiveHybridStreamingOptimizer** (``adaptive_hybrid.py``, 4,376 lines): Main 4-phase optimizer
- **LargeDatasetFitter** (``large_dataset.py``, 2,539 lines): Memory-aware automatic chunking
- **HybridStreamingConfig** (``hybrid_config.py``, 878 lines): Extensive configuration
- **streaming/phases/** subpackage: WarmupPhase, GaussNewtonPhase, CheckpointManager, PhaseOrchestrator

Memory Management
~~~~~~~~~~~~~~~~~

- Power-of-2 bucket sizes eliminate JIT recompilation: 1024, 2048, 4096, ..., 131072
- psutil for system memory detection with 16GB default fallback
- Automatic chunk size calculation based on available memory


Protocol-Based Dependency Injection
-----------------------------------

The ``interfaces/`` package provides Protocol definitions enabling loose coupling:

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
     - Streaming data sources
   * - ``JacobianProtocol``
     - Jacobian computation strategies
   * - ``SparseJacobianProtocol``
     - Sparse Jacobian handling
   * - ``ResultProtocol``
     - Optimization results

All protocols use ``@runtime_checkable`` for structural subtyping without explicit inheritance.


Caching and Performance
-----------------------

Multi-Tier Caching
~~~~~~~~~~~~~~~~~~

**UnifiedCache** (``unified_cache.py``):

- Shape-relaxed cache keys: ``(func_hash, dtype, rank)`` instead of full shapes
- LRU eviction with configurable maxsize (default: 128)
- Weak references to prevent memory leaks
- Target: 80%+ cache hit rate

**SmartCache** (``smart_cache.py``):

- xxhash for 10x faster hashing than SHA256
- Stride-based sampling for arrays >10K elements
- Safe JSON serialization (no pickle)

**MemoryManager** (``memory_manager.py``):

- LRU array pooling via OrderedDict
- psutil for system memory detection
- Telemetry circular buffer (deque maxlen=1000) for multi-day runs
- Mixed precision coordination


Numerical Stability
-------------------

Stability Guard
~~~~~~~~~~~~~~~

``NumericalStabilityGuard`` (``stability/guard.py``, 1,148 lines) provides three modes:

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


Mixed Precision
---------------

The ``MixedPrecisionManager`` implements a 5-state machine for automatic precision management:

.. code-block:: text

   FLOAT32_ACTIVE → MONITORING_DEGRADATION → UPGRADING_TO_FLOAT64 → FLOAT64_ACTIVE
                                                                          │
                                              RELAXED_FLOAT32_FALLBACK ←──┘

Key features:

- 50% memory savings when using float32
- Monitors 5 convergence metrics
- Zero-iteration-loss state transfer during upgrades


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

Blocked patterns include: ``exec``, ``eval``, ``os.system``, ``subprocess``, ``socket``, ``ctypes``, etc.


Diagnostics System
------------------

Post-fit model health analysis via the ``diagnostics/`` package:

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
   * - ``PluginRegistry``
     - Domain-specific extensions

Usage:

.. code-block:: python

   result = curve_fit(model, x, y, compute_diagnostics=True)
   print(result.diagnostics.summary())


Global Optimization
-------------------

Multi-start search for escaping local minima:

- **MultiStartOrchestrator**: Parallel evaluation of starting points
- **TournamentSelector**: Memory-efficient selection for streaming
- **Samplers**: Latin Hypercube (LHS), Sobol, Halton quasi-random sequences
- **Presets**: ``'fast'``, ``'robust'``, ``'global'``, ``'thorough'``, ``'streaming'``

Integration strategy by dataset size:

- Small (<1M points): Full multi-start on complete data
- Medium (1M-100M): Full multi-start, then chunked fit
- Large (>100M): Tournament selection during streaming warmup


Qt GUI System
-------------

The ``gui_qt/`` package provides a native Qt desktop application:

.. code-block:: text

   gui_qt/
   ├── __init__.py         - run_desktop() entry point
   ├── main_window.py      - MainWindow with sidebar navigation
   ├── app_state.py        - AppState (Qt signals wrapping SessionState)
   ├── session_state.py    - SessionState dataclass
   ├── theme.py            - ThemeConfig, ThemeManager (light/dark)
   ├── autosave.py         - AutosaveManager for crash recovery
   ├── pages/              - 5-page workflow (QWidget-based)
   │   ├── data_loading.py
   │   ├── model_selection.py
   │   ├── fitting_options.py
   │   ├── results.py
   │   └── export.py
   ├── widgets/            - Reusable Qt widgets
   ├── plots/              - pyqtgraph-based scientific plots
   └── adapters/           - NLSQ-GUI bridge (data, fit, model, export)

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
     - ``interfaces/`` - structural subtyping without inheritance
   * - Factory
     - ``create_optimizer()``, ``configure_curve_fit()``
   * - Singleton
     - ``JAXConfig``, global caches
   * - State Machine
     - ``PrecisionState`` for mixed precision management
   * - Phased Pipeline
     - 4-phase streaming optimizer
   * - Lazy Loading
     - ``__getattr__`` for specialty modules (50%+ import time reduction)
   * - Adapter
     - GUI adapters bridge NLSQ ↔ Qt widgets


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


Performance Optimizations
-------------------------

1. **Lazy Imports**: 50%+ reduction in cold import time via ``__getattr__``
2. **Shape-Relaxed Cache Keys**: Cache by ``(hash, dtype, rank)`` not exact shapes
3. **Power-of-2 Bucketing**: Static array shapes for JIT efficiency
4. **xxhash**: 10x faster hashing than SHA256
5. **LRU Pooling**: Array reuse via OrderedDict
6. **TTL-cached psutil**: Reduce memory detection overhead


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


Configuration System
--------------------

The ``config.py`` module provides a singleton ``JAXConfig`` that manages:

- **JAX initialization**: x64 enabled, GPU memory configuration
- **MemoryConfig**: Memory limits, chunk sizes, out-of-memory strategies
- **LargeDatasetConfig**: Solver selection thresholds (direct: 100K, iterative: 10M, chunked: 100M)
- **MixedPrecisionConfig**: Precision management settings

All configuration is validated at instantiation time with descriptive error messages.


See Also
--------

- :doc:`optimization_case_study` - Performance optimization deep dive
- :doc:`performance_tuning_guide` - Practical tuning recommendations
- :doc:`adr/README` - Architecture Decision Records
