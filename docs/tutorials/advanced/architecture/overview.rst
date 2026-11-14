Package Overview
================

This page describes NLSQ's package structure and module organization.

Module Hierarchy
----------------

.. code-block:: text

   nlsq/
   ├── __init__.py            # Public API exports (lazy-loaded)
   │
   ├── core/                  # Core optimization
   │   ├── minpack.py              # SciPy-compatible curve_fit() API
   │   ├── problem.py              # prepare_problem(): input preparation
   │   ├── covariance.py           # pcov_from_jacobian() / pcov_from_normal()
   │   ├── least_squares.py        # LeastSquares class
   │   ├── trf.py                  # TRF algorithm
   │   ├── trf_jit.py              # JIT-compiled helpers
   │   ├── profiler.py             # TRFProfiler / NullProfiler
   │   ├── workflow.py             # 3-workflow system + MemoryBudgetSelector
   │   ├── functions.py            # Built-in models
   │   ├── loss_functions.py       # Robust loss functions
   │   ├── _option_checks.py       # Option validators shared by entry points
   │   └── sparse_jacobian.py      # Sparse Jacobian support
   │
   ├── diagnostics/           # Optimization health & identifiability
   │   ├── types.py                 # DiagnosticsConfig, IdentifiabilityReport
   │   ├── gradient_health.py       # GradientHealthReport
   │   ├── identifiability.py       # IdentifiabilityAnalyzer
   │   ├── parameter_sensitivity.py # ParameterSensitivityAnalyzer
   │   ├── health_report.py         # ModelHealthIssue aggregation
   │   └── recommendations.py
   │
   ├── result/                # Result types (consolidated)
   │   ├── optimize_result.py      # OptimizeResult
   │   ├── optimize_warning.py     # OptimizeWarning
   │   └── curve_fit_result.py     # CurveFitResult
   │
   ├── streaming/             # Large datasets
   │   ├── large_fit.py            # fit_with_strategy(): runs the chosen engine
   │   ├── exact_lm.py             # Exact chunk-accumulated LM (default >= size_threshold)
   │   ├── normal_equations.py     # J^T J / J^T r accumulation passes
   │   ├── specs.py, strategy.py   # Strategy specs, build_large_strategy()
   │   ├── large_dataset.py        # LargeDatasetFitter
   │   ├── adaptive_hybrid.py      # AdaptiveHybridStreamingOptimizer
   │   ├── telemetry.py            # DefenseLayerTelemetry
   │   ├── validators.py           # Config validation
   │   ├── hybrid_config.py        # HybridStreamingConfig
   │   └── phases/                 # Warmup / Gauss-Newton phase pipeline
   │
   ├── caching/               # Performance
   │   ├── memory_manager.py       # Memory pooling
   │   ├── unified_cache.py        # JIT compilation caching
   │   └── _closure_serial.py      # Closure identity for cache keys
   │
   ├── stability/             # Numerical stability
   │   ├── guard.py                # NumericalStabilityGuard
   │   ├── svd_fallback.py         # SVD fallback
   │   ├── fallback.py             # FallbackOrchestrator + strategies
   │   └── recovery.py             # OptimizationRecovery
   │
   ├── precision/             # Precision control
   │   ├── algorithm_selector.py   # Problem-size-aware solver selection
   │   ├── bound_inference.py      # BoundsInference
   │   ├── parameter_estimation.py # Initial parameter estimation
   │   └── parameter_normalizer.py
   │
   ├── global_optimization/   # Global search
   │   ├── multi_start.py          # MultiStartOrchestrator
   │   ├── cmaes_optimizer.py      # CMAESOptimizer
   │   ├── cmaes_config.py
   │   ├── cmaes_diagnostics.py
   │   ├── bounds_transform.py     # Sigmoid bounds transform
   │   ├── checkpoint.py           # CMA-ES resume state
   │   ├── bipop.py                # BIPOP restarts
   │   ├── tournament.py           # TournamentSelector
   │   └── method_selector.py      # MethodSelector (Multi-Start vs CMA-ES)
   │
   ├── utils/                 # Utilities
   │   ├── validators.py           # Input validation
   │   ├── diagnostics.py          # OptimizationDiagnostics (convergence monitor)
   │   ├── safe_serialize.py       # Secure JSON serialization (replaces pickle)
   │   └── logging.py
   │
   ├── cli/                   # Command-line interface
   │   ├── main.py                 # Subcommands: fit, batch, info, config, gui
   │   ├── data_loaders.py         # ASCII, CSV, NPZ, HDF5
   │   ├── model_registry.py       # Model loading with security validation
   │   ├── model_validation.py     # AST denylist validation (not a sandbox)
   │   ├── workflow_runner.py
   │   └── result_exporter.py
   │
   └── gui_qt/                # Desktop GUI (optional nlsq[gui] extra)
       └── ...

Import Patterns
---------------

**Public API (recommended):**

.. code-block:: python

   from nlsq import fit, curve_fit, CurveFit
   from nlsq import OptimizeResult, OptimizeWarning

**Core classes:**

.. code-block:: python

   from nlsq.core.least_squares import LeastSquares
   from nlsq.core.trf import TrustRegionReflective
   from nlsq.core.workflow import MemoryBudgetSelector

**Diagnostics:**

.. code-block:: python

   from nlsq.diagnostics.types import DiagnosticsConfig, DiagnosticLevel
   from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer

Lazy Loading
------------

NLSQ uses lazy loading to minimize import time:

.. code-block:: python

   # Fast import (~620ms including JAX)
   import nlsq

   # Specialty exports load their module on first access
   nlsq.LargeDatasetFitter  # Loads the streaming module
   nlsq.GlobalOptimizationConfig  # Loads global optimization

This reduces memory usage and startup time for simple use cases.

Dependency Graph
----------------

.. code-block:: text

   curve_fit() ──► CurveFit ──► prepare_problem() ──► LeastSquares
                                                        │
                                                        └──► TrustRegionReflective

   fit(workflow=...) ──► MemoryBudgetSelector
     ├──► standard  ──► curve_fit()
     ├──► chunked   ──► LargeDatasetFitter
     ├──► streaming ──► AdaptiveHybridStreamingOptimizer
     └──► auto_global / hpc ──► MethodSelector
                                  ├──► MultiStartOrchestrator
                                  └──► CMAESOptimizer

   curve_fit_large() ──► spec (default Auto) ──► fit_with_strategy()
     ├──► InMemory ──► curve_fit()
     ├──► Exact    ──► exact_lm_fit()
     ├──► Chunked  ──► LargeDatasetFitter
     └──► Hybrid   ──► AdaptiveHybridStreamingOptimizer

   Diagnostics (optional, compute_diagnostics=True)
     └──► analyzers in nlsq.diagnostics, reported on ``result.diagnostics``

Circular dependencies are broken via:

1. **Lazy imports**: Import at function call time
2. **TYPE_CHECKING**: Type hints without runtime import

Next Steps
----------

- :doc:`jax_patterns` - JAX programming patterns
