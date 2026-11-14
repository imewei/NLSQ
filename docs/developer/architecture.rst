Architecture Overview
=====================

This document provides an architectural overview of NLSQ, a nonlinear least squares curve
fitting library built on JAX that runs on CPU, GPU and TPU backends. The package consists of
approximately 83,400 lines of Python code organized into well-separated modules.

Package Structure
-----------------

The ``nlsq`` package is organized into logical subpackages:

.. code-block:: text

   nlsq/ (~83,400 lines)
   ├── core/            16,000 lines - Core optimization (curve_fit, TRF, LeastSquares)
   │   ├── minpack.py        curve_fit(), CurveFit, fit() workflow dispatch
   │   ├── problem.py        prepare_problem(): validation, p0/bounds, NaN handling, padding
   │   ├── covariance.py     pcov_from_jacobian() / pcov_from_normal()
   │   ├── least_squares.py  LeastSquares, AutoDiffJacobian, compiled-wrapper cache
   │   ├── trf.py, trf_jit.py  Trust Region Reflective optimizer and JIT kernels
   │   ├── workflow.py       Workflow configuration (auto / auto_global / hpc)
   │   ├── loss_functions.py, functions.py  Robust losses, built-in model functions
   │   ├── sparse_jacobian.py, profiler.py  Sparse Jacobian support, TRF profiling
   │   ├── _option_checks.py Shared option validators for the fitting entry points
   │   └── device_policy.py  Host-device placement for small SVDs under a GPU backend
   ├── streaming/      15,900 lines - Large dataset handling
   │   ├── large_fit.py      fit_with_strategy(): runs the strategy chosen by curve_fit_large
   │   ├── specs.py, strategy.py  Strategy specs and build_large_strategy()
   │   ├── normal_equations.py, exact_lm.py  Exact chunk-accumulated LM
   │   ├── large_dataset.py  LargeDatasetFitter (chunked fit)
   │   ├── adaptive_hybrid.py, hybrid_config.py  Hybrid 4-phase streaming optimizer
   │   ├── telemetry.py, validators.py  Defense-layer telemetry, config validation
   │   └── phases/      3,000 lines - WarmupPhase, GaussNewtonPhase, Checkpoint
   ├── gui_qt/         13,000 lines - Optional Qt desktop GUI (nlsq[gui] extra)
   ├── cli/             9,200 lines - Command-line interface with security
   │   ├── commands/     Subcommands (fit, batch, config, info); gui is handled in main.py
   │   └── templates/    Custom model scaffolding
   ├── utils/           5,500 lines - Validators, logging, profiling, serialization
   ├── global_optimization/ 5,900 lines - CMA-ES, multi-start, tournament selection
   ├── diagnostics/     4,800 lines - Model health analysis, plugin system
   ├── stability/       2,700 lines - Numerical robustness, fallbacks, recovery
   ├── precision/       2,200 lines - Algorithm selection, parameter normalization/estimation
   ├── caching/         1,500 lines - JIT caching, memory management
   ├── result/          1,300 lines - OptimizeResult, CurveFitResult
   └── (root)          ~5,200 lines - Config, callbacks, types, device, constants,
                                      common_jax / common_scipy

Line counts are recursive per package, rounded to the nearest hundred
(``find nlsq/<pkg> -name '*.py' | xargs wc -l``). They are not tracked per file because
they drift with every change.



Architectural Layers
--------------------

The following diagram illustrates the layered architecture of NLSQ. Large-data routing is
described in :ref:`large-dataset-routing`.

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────────────────────┐
   │                               USER INTERFACES                                │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ Qt GUI (nlsq[gui] extra)  CLI (argparse)              Python API             │
   │ ├── 5-page workflow       ├── fit, batch, info        ├── curve_fit(), fit() │
   │ ├── pyqtgraph plots       ├── config, gui             ├── curve_fit_large()  │
   │ └── PySide6 desktop       └── Model validation (AST)  └── CurveFit class     │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                           ENTRY POINTS AND ROUTING                           │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ fit(workflow=...)                  curve_fit_large(strategy=...)             │
   │ ├── auto, auto_global, hpc         Auto: InMemory < size_threshold,          │
   │ └── MemoryBudgetSelector picks:          else Exact (default)                │
   │     standard  → curve_fit          Chunked → LargeDatasetFitter              │
   │     chunked   → LargeDatasetFitter Hybrid → AdaptiveHybridStreamingOptimizer │
   │     streaming → hybrid             all via fit_with_strategy()               │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                             OPTIMIZATION ENGINES                             │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ Global optimization        Exact streaming LM      Hybrid (opt-in)           │
   │ ├── MethodSelector         ├── exact_lm.py         ├── 0: normalization      │
   │ ├── CMAESOptimizer         ├── normal_equations.py ├── 1: L-BFGS             │
   │ ├── MultiStartOrchestrator └── sums J^T J, J^T r   ├── 2: Gauss-Newton       │
   │ ├── TournamentSelector         over data chunks    └── 3: denormalization    │
   │ └── LHS / Sobol / Halton                                                     │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                           CORE OPTIMIZATION ENGINE                           │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ 1. curve_fit() / CurveFit              core/minpack.py                       │
   │ 2. prepare_problem()                   core/problem.py                       │
   │ 3. LeastSquares + AutoDiffJacobian     core/least_squares.py                 │
   │ 4. TrustRegionReflective (Python loop) core/trf.py, trf_jit.py               │
   │ 5. pcov_from_jacobian()                core/covariance.py                    │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                              SUPPORT SUBSYSTEMS                              │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ stability/                             precision/                            │
   │ ├── guard.py (3 modes)                 ├── algorithm_selector                │
   │ ├── svd_fallback, fallback             ├── bound_inference                   │
   │ └── recovery                           ├── parameter_normalizer              │
   │                                        └── parameter_estimation              │
   │ caching/                               diagnostics/                          │
   │ ├── unified_cache                      ├── identifiability                   │
   │ ├── memory_manager                     ├── gradient_health                   │
   │ └── _closure_serial                    ├── parameter_sensitivity             │
   │                                        ├── health_report                     │
   │                                        └── plugin                            │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                                INFRASTRUCTURE                                │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ config.py (singleton)                Security                                │
   │ ├── JAXConfig (x64, GPU)             ├── utils/safe_serialize (JSON)         │
   │ ├── MemoryConfig                     ├── cli/model_validation (AST denylist) │
   │ └── LargeDatasetConfig               └── path/resource limits, audit         │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │                  JAX RUNTIME (>=0.8.0, tested on 0.12.0.1)                   │
   ├──────────────────────────────────────────────────────────────────────────────┤
   │ ├── x64 enabled (double precision)     ├── JIT compilation, persistent cache │
   │ └── Automatic differentiation          └── CPU, GPU or TPU backend           │
   └──────────────────────────────────────────────────────────────────────────────┘

Core Optimization Pipeline
--------------------------

Class Hierarchy
~~~~~~~~~~~~~~~

The core optimization pipeline follows this class hierarchy:

.. code-block:: text

   curve_fit() / fit()           Entry points (nlsq/__init__.py, core/minpack.py)
            │
            ▼
       CurveFit                  Main curve fitting class (minpack.py)
            │                    - SciPy-compatible API wrapper
            ├── UnifiedCache     - Fixed-length padding for JIT
            │                    - Stability/recovery options
            ▼
       prepare_problem()         Validation, p0/bounds, NaN handling (problem.py)
            │
            ▼
       LeastSquares              Optimization orchestrator (least_squares.py)
            │                    - method 'trf' (default); 'lm' is an alias for TRF
            ├── AutoDiffJacobian - Three Jacobian handlers (no sigma, 1D sigma, 2D cov)
            ├── LossFunctionsJIT - Robust loss functions (loss_functions.py)
            ├── _wrapper_cache   - LRU cache (32 entries) of compiled wrappers
            ▼
       TrustRegionReflective     Main optimizer (trf.py)
            │                    - Inherits TrustRegionJITFunctions (trf_jit.py)
            ├── CommonJIT        - Variable scaling for bounds (common_jax.py)
            │                    - tr_solver: None / 'exact' (SVD, default) / 'lsmr' / 'cg'
            ▼                    - Python ``while`` loop over JIT-compiled kernels
       Trust-region step         (SVD-based by default)

Problem Preparation and Covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two single-purpose modules sit on either side of the optimizer:

.. code-block:: text

   curve_fit() call
       │
       ▼
   prepare_problem()          core/problem.py
       │                      Fixed step order: parameter count / p0 / bounds,
       │                      validation (``InputValidator``) and NaN handling, method selection,
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
   :widths: 35 65

   * - File
     - Purpose
   * - ``core/minpack.py``
     - curve_fit(), CurveFit class, fit() workflow dispatch
   * - ``core/problem.py``
     - ``prepare_problem()``: input validation, p0/bounds, NaN handling, padding
   * - ``core/least_squares.py``
     - LeastSquares orchestrator, AutoDiffJacobian, compiled-wrapper cache
   * - ``core/trf.py``
     - TrustRegionReflective algorithm
   * - ``core/trf_jit.py``
     - JIT-compiled TRF helper functions
   * - ``core/covariance.py``
     - ``pcov_from_jacobian()`` / ``pcov_from_normal()``
   * - ``core/workflow.py``
     - Workflow configuration for automatic strategy selection
   * - ``core/loss_functions.py``
     - JIT-compiled robust loss functions
   * - ``core/sparse_jacobian.py``
     - Sparse Jacobian detection and construction
   * - ``core/_option_checks.py``
     - Option validators shared by ``curve_fit``, ``fit`` and ``curve_fit_large``
   * - ``streaming/large_fit.py``
     - ``fit_with_strategy()``: runs the engine chosen for ``curve_fit_large``


.. _large-dataset-routing:

Large-Dataset Routing
---------------------

Two public entry points decide how a large problem is run. Both build on the engines
described below.

Entry Points
~~~~~~~~~~~~

- ``curve_fit_large(f, x, y, ..., strategy=...)`` (``nlsq/__init__.py``) translates its
  options into a spec from ``streaming/specs.py`` and calls ``fit_with_strategy()``
  (``streaming/large_fit.py``), which runs the engine for that spec.
- ``fit(f, x, y, workflow=...)`` (``core/minpack.py``):

  - ``auto``: ``MemoryBudgetSelector`` picks standard (``curve_fit``), chunked
    (``build_large_strategy("chunked")``, ``LargeDatasetFitter``) or streaming
    (``build_large_strategy("hybrid")``, ``AdaptiveHybridStreamingOptimizer``) from the
    memory budget.
  - ``auto_global``: ``MemoryBudgetSelector`` plus ``MethodSelector`` (CMA-ES or
    multi-start; requires bounds).
  - ``hpc``: ``auto_global`` plus checkpointing and cluster detection.
  - ``fit()`` never selects the exact engine and rejects ``method="exact_streaming"``.
  - The deprecated ``preset=`` argument (fast, robust, global, streaming, large) emits a
    ``FutureWarning`` and is scheduled for removal in v0.8.0.

Strategy Specs and Builder
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

``curve_fit_large`` Routing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Auto`` (the default) resolves on the number of points: ``InMemory`` below
  ``size_threshold`` (default 1,000,000), ``Exact`` at or above it.
- An explicit ``Chunked()`` runs ``LargeDatasetFitter``; an explicit ``Hybrid()`` runs
  ``AdaptiveHybridStreamingOptimizer``.
- ``method="exact_streaming"`` selects the exact engine explicitly, at any size.
- When the exact engine cannot honor the request for an *engine* reason (non-default
  ``method``, multistart, unsupported keyword arguments), it emits a ``UserWarning``
  and falls back to the chunked fit.
- When the cause is the *data* (2-D ``xdata``, a model that is not pointwise, non-finite
  values with ``check_finite=False``, invalid ``sigma``), it raises ``ValueError``
  naming ``strategy=Chunked()``.

Exact Engine
~~~~~~~~~~~~

``exact_lm.py`` is a Levenberg-Marquardt solver over the full data. Each iteration runs
one pass over the chunks (``normal_equations.py``, shared with the hybrid Gauss-Newton
phase) accumulating ``J^T J`` and ``J^T r``.

- ``max_nfev`` counts full-data passes; a stalled solve raises ``ExactSolverError``.
- ``pcov`` is adaptive: from the normal equations when ``sqrt(cond(H)) <= TSQR_KAPPA``
  (1e3), otherwise from one TSQR pass; ``compute_covariance=False`` skips it.
- The chunk kernel is cached by jaxpr digest in ``_wrapper_cache``. Models with a custom
  derivative rule (``custom_jvp`` / ``custom_vjp``) or more than 64 MiB of captured
  constants are never cached.


Hybrid Streaming Optimizer
--------------------------

``AdaptiveHybridStreamingOptimizer`` is one of the engines above, selected with
``strategy=Hybrid()`` or by the streaming workflow of ``fit()``. It is not the default
path of ``curve_fit_large``.

Four-Phase Pipeline
~~~~~~~~~~~~~~~~~~~

The optimizer runs four phases for datasets up to 100M+ points:

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

- **AdaptiveHybridStreamingOptimizer** (``adaptive_hybrid.py``): Main 4-phase optimizer
- **LargeDatasetFitter** (``large_dataset.py``): Memory-aware automatic chunking (the chunked engine)
- **HybridStreamingConfig** (``hybrid_config.py``): Extensive configuration
- **DefenseLayerTelemetry** (``telemetry.py``): Monitoring of the four defense layers
- **streaming/phases/** subpackage:

  - ``WarmupPhase``: L-BFGS warmup with adaptive switching
  - ``GaussNewtonPhase``: Chunked J^T J accumulation
  - ``CheckpointManager``: Fault-tolerant state persistence
  - ``PhaseOrchestrator``: Phase transition management

- **TournamentSelector** (``global_optimization/tournament.py``): runs inside the warmup
  when multistart is enabled

Memory Management
~~~~~~~~~~~~~~~~~

- Power-of-2 bucket sizes eliminate JIT recompilation: 1024, 2048, 4096, ..., 131072
- psutil for system memory detection with 16GB default fallback
- Automatic chunk size calculation based on available memory


Result Types
------------

The ``result/`` package provides optimization result containers:

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

**UnifiedCache** (``unified_cache.py``):

- Shape-relaxed cache keys: ``(func_hash, dtype, rank)`` instead of full shapes
- LRU eviction with configurable maxsize (default: 128)
- Weak references to prevent memory leaks
- Target: 80%+ cache hit rate

**MemoryManager** (``memory_manager.py``):

- LRU array pooling via OrderedDict
- psutil for system memory detection
- Telemetry circular buffer (deque maxlen=1000) for multi-day runs

**Compiled-wrapper cache** (``core/least_squares.py``):

- ``_wrapper_cache`` is an LRU ``OrderedDict`` limited to 32 entries; it holds the hot-path
  compiled wrappers, and the exact streaming kernel is cached in it by jaxpr digest
- ``NLSQ_DISABLE_WRAPPER_CACHE`` disables it
- ``caching/_closure_serial.py`` supplies the per-object identity used when closures are
  hashed into cache keys


Numerical Stability
-------------------

Stability Guard
~~~~~~~~~~~~~~~

``NumericalStabilityGuard`` (``stability/guard.py``) provides three modes:

- ``stability=False``: No checks (maximum performance)
- ``stability='check'``: Warn only, no modifications
- ``stability='auto'``: Detect and fix numerical issues

Key thresholds:

- Condition number threshold: 1e12
- SVD skip for >10M Jacobian elements
- Diagonal regularization of ill-conditioned Jacobians (``check_and_fix_jacobian``):
  factor 1e-10 times the largest singular value

Fallback Chain
~~~~~~~~~~~~~~

The JIT-compatible linear solve (``solve_with_cholesky_fallback``) has two stages:

.. code-block:: text

   Cholesky decomposition
          │
          ▼ (if it fails, detected via NaN)
   Eigenvalue decomposition

Diagonal regularization belongs to the guard (``check_and_fix_jacobian``, and
``regularize_hessian`` for Hessians) and is not a stage of this chain.

Additional stability components:

- **SVD Fallback** (``svd_fallback.py``): GPU/CPU fallback SVD
- **Recovery** (``recovery.py``): Optimization recovery after trust region failures
- **FallbackOrchestrator** (``fallback.py``): Ordered retry strategies (alternative method, perturbed ``p0``, adjusted tolerances, bounds, robust loss, rescaling)


Precision Modules
-----------------

The ``precision/`` package provides solver selection and parameter management:

- **AlgorithmSelector** (``algorithm_selector.py``): Problem-size-aware solver selection
- **BoundsInference** (``bound_inference.py``): Automatic bound inference
- **ParameterNormalizer** (``parameter_normalizer.py``): Parameter scaling for numerical stability
- **parameter_estimation.py**: Automatic initial parameter estimation for the built-in models



Global Optimization
-------------------

GPU-accelerated global optimization for escaping local minima:

CMA-ES Optimizer
~~~~~~~~~~~~~~~~

- **CMAESOptimizer** (``cmaes_optimizer.py``): evosax-based CMA-ES with BIPOP restart strategy
- CMA-ES global search followed by TRF local refinement for proper covariance estimation
- Presets: ``'cmaes-fast'``, ``'cmaes'``, ``'cmaes-global'``
- Sigmoid bounds transform (``bounds_transform.py``) maps unbounded CMA-ES to the bounded parameter space
- ``bipop.py`` (restart strategy), ``checkpoint.py`` (resume state), ``cmaes_config.py`` and ``cmaes_diagnostics.py``

Multi-Start Search
~~~~~~~~~~~~~~~~~~

- **MultiStartOrchestrator** (``multi_start.py``): Parallel evaluation of starting points
- Thread-safe with per-thread CurveFit isolation
- Worker count by hardware: one per device on multi-GPU, ``min(n, 4)`` on a single GPU, ``min(n, cpu_count, 16)`` on CPU
- Presets: ``'fast'``, ``'robust'``, ``'global'``, ``'thorough'``, ``'streaming'``

Tournament Selection
~~~~~~~~~~~~~~~~~~~~

- **TournamentSelector** (``tournament.py``): Progressive N to N/2 to top M elimination
- Memory-efficient evaluation on data batches without loading full dataset
- Checkpoint/resume support for fault tolerance

Samplers
~~~~~~~~

- **Latin Hypercube** (LHS): Stratified random sampling
- **Sobol**: Low-discrepancy quasi-random (up to 21 dimensions)
- **Halton**: Prime-base quasi-random (up to 20 dimensions)

**MethodSelector** (``method_selector.py``): Auto-selects CMA-ES vs multi-start based on
parameter scale ratio and evosax availability.

Dataset size does not choose the global strategy. ``MemoryBudgetSelector`` picks the local
fit engine from the memory budget, and tournament selection runs inside the hybrid
streaming warmup when multistart is enabled for that engine.



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
   * - ``ModelHealthReport``
     - Aggregated health summary from all analyzers
   * - ``PluginRegistry``
     - Domain-specific extensions

Usage:

.. code-block:: python

   result = curve_fit(model, x, y, compute_diagnostics=True)
   print(result.diagnostics.summary())


CLI System
----------

The ``cli/`` package provides an argparse-based command-line interface with the
subcommands ``fit``, ``batch``, ``info``, ``config`` and ``gui``:

.. code-block:: text

   cli/
   ├── main.py              - argparse entry point (nlsq); also handles the gui subcommand
   ├── commands/
   │   ├── fit.py            - Single fit command
   │   ├── batch.py          - Batch fitting (multiple datasets)
   │   ├── config.py         - Configuration management
   │   └── info.py           - Environment/GPU info
   ├── data_loaders.py       - ASCII (.txt/.dat/.asc), CSV, NPZ and HDF5 data loading
   ├── model_registry.py     - Built-in + custom model discovery
   ├── model_validation.py   - AST denylist validation, path/resource limits, audit log
   ├── workflow_runner.py     - Orchestration of fit workflows
   ├── visualization.py      - Terminal/matplotlib result display
   ├── result_exporter.py    - JSON/CSV export
   ├── errors.py             - Structured error handling
   └── templates/            - Custom model scaffolding


Qt GUI System
-------------

The ``gui_qt/`` package (~13,000 lines) provides a native Qt desktop application. It is an
optional extra (``pip install "nlsq[gui]"``; PySide6 and pyqtgraph):

.. code-block:: text

   gui_qt/
   ├── __init__.py         - run_desktop() entry point
   ├── __main__.py         - python -m nlsq.gui_qt
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
   │   ├── param_config.py
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
     - 4-phase hybrid streaming optimizer
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
       → prepare_problem()                   → PreparedProblem
           (InputValidator, p0/bounds, NaN handling, method selection, padding)
       → LeastSquares.least_squares()
       → AutoDiffJacobian (JAX autodiff)
       → TrustRegionReflective.trf()
           → Python while loop over JIT-compiled kernels
           → trust-region subproblem: SVD (default), LSMR or CG
       → OptimizeResult
       → pcov_from_jacobian()                → pcov
       → CurveFitResult (popt, pcov, statistics)

``fit()`` Workflow Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → fit(f, x, y, p0, workflow='auto')
       → core.minpack.fit()
       → MemoryBudgetSelector
           ├── standard  → curve_fit()
           ├── chunked   → build_large_strategy('chunked') → LargeDatasetFitter
           └── streaming → build_large_strategy('hybrid')
                           → AdaptiveHybridStreamingOptimizer

Large Dataset (``curve_fit_large``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit_large(f, x, y, p0)
       → spec from strategy= (default Auto)
           Auto.resolve(n_points): InMemory below size_threshold, else Exact
       → fit_with_strategy()                 streaming/large_fit.py
           ├── InMemory → curve_fit()
           ├── Exact    → exact_lm_fit(): one pass over chunks per iteration,
           │              accumulating J^T J and J^T r; adaptive pcov
           ├── Chunked  → LargeDatasetFitter
           └── Hybrid   → AdaptiveHybridStreamingOptimizer (flow below)
       → CurveFitResult

Hybrid Streaming
~~~~~~~~~~~~~~~~

.. code-block:: text

   User Input → curve_fit_large(f, x, y, p0, strategy=Hybrid())
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
       → MemoryBudgetSelector (local fit engine)
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
   * - ``NLSQ_DISABLE_WRAPPER_CACHE=1``
     - Disable the compiled-wrapper cache
   * - ``NLSQ_SAVE_ITERATIONS=1``
     - Record and save per-fit iteration history

Only the variables most relevant to the architecture are listed. See
:doc:`/reference/configuration` for the complete environment-variable reference.


Configuration System
--------------------

The ``config.py`` module provides a singleton ``JAXConfig`` that manages:

- **JAX initialization**: x64 enabled, GPU memory configuration
- **MemoryConfig**: Memory limits, chunk sizes, out-of-memory strategies
- **LargeDatasetConfig**: Solver selection thresholds (direct: 100K, iterative: 10M, chunked: 100M)

All configuration is validated at instantiation time with descriptive error messages.


See Also
--------

- :doc:`optimization_case_study` — Performance optimization deep dive
- :doc:`performance_tuning_guide` — Practical tuning recommendations
- :doc:`Architecture Decision Records <adr/README>`
