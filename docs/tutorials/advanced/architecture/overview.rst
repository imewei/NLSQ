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
   │   ├── least_squares.py        # LeastSquares class
   │   ├── trf.py                  # TRF algorithm (~2500 lines)
   │   ├── trf_jit.py              # JIT-compiled helpers
   │   ├── profiler.py             # TRFProfiler / NullProfiler
   │   ├── workflow.py             # 3-workflow system + MemoryBudgetSelector
   │   ├── functions.py            # Built-in models
   │   ├── factories.py            # Factory functions
   │   ├── sparse_jacobian.py      # Sparse Jacobian support
   │   ├── orchestration/          # CurveFit God Class decomposition (v0.6.4)
   │   │   ├── data_preprocessor.py
   │   │   ├── optimization_selector.py
   │   │   ├── covariance_computer.py
   │   │   ├── streaming_coordinator.py
   │   │   └── entities.py
   │   └── adapters/               # Protocol adapters
   │       └── curve_fit_adapter.py
   │
   ├── interfaces/            # Protocol definitions
   │   ├── optimizer_protocol.py   # OptimizerProtocol, CurveFitProtocol
   │   ├── cache_protocol.py       # CacheProtocol, BoundedCacheProtocol
   │   ├── data_source_protocol.py
   │   ├── jacobian_protocol.py
   │   ├── orchestration_protocol.py
   │   └── result_protocol.py
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
   │   ├── optimizer.py            # Base streaming
   │   ├── large_dataset.py        # LargeDatasetFitter
   │   ├── adaptive_hybrid.py      # AdaptiveHybridStreamingOptimizer
   │   ├── telemetry.py            # DefenseLayerTelemetry
   │   ├── validators.py           # Config validation
   │   ├── hybrid_config.py        # HybridStreamingConfig
   │   └── phases/                 # Warmup / Gauss-Newton phase pipeline
   │
   ├── caching/               # Performance
   │   ├── memory_manager.py       # Memory pooling
   │   ├── smart_cache.py          # JIT caching (xxhash)
   │   └── compilation_cache.py    # Persistent cache
   │
   ├── stability/             # Numerical stability
   │   ├── guard.py                # NumericalStabilityGuard
   │   ├── svd_fallback.py         # SVD fallback
   │   ├── fallback.py             # FallbackOrchestrator + strategies
   │   ├── recovery.py             # OptimizationRecovery
   │   └── condition_monitor.py    # Condition tracking
   │
   ├── precision/             # Precision control
   │   └── parameter_normalizer.py
   │
   ├── facades/               # Lazy loading
   │   ├── optimization_facade.py
   │   ├── stability_facade.py
   │   └── diagnostics_facade.py
   │
   ├── global_optimization/   # Global search
   │   ├── multi_start.py          # MultiStartOrchestrator
   │   ├── cmaes_optimizer.py      # CMAESOptimizer
   │   ├── cmaes_config.py
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
   │   ├── main.py
   │   ├── model_registry.py       # Model loading with security validation
   │   ├── model_validation.py     # AST-based model validation
   │   ├── workflow_runner.py
   │   └── result_exporter.py
   │
   └── gui_qt/                # Desktop GUI
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

**Protocols:**

.. code-block:: python

   from nlsq.interfaces.optimizer_protocol import OptimizerProtocol
   from nlsq.interfaces.cache_protocol import CacheProtocol

**Diagnostics:**

.. code-block:: python

   from nlsq.diagnostics.types import DiagnosticsConfig, DiagnosticLevel
   from nlsq.diagnostics.identifiability import IdentifiabilityAnalyzer

**Facades (lazy loading):**

.. code-block:: python

   from nlsq.facades import OptimizationFacade, StabilityFacade

Lazy Loading
------------

NLSQ uses lazy loading to minimize import time:

.. code-block:: python

   # Fast import (~620ms including JAX)
   import nlsq

   # Specialty modules load on first access
   nlsq.streaming  # Loads streaming module
   nlsq.global_optimization  # Loads global optimization

This reduces memory usage and startup time for simple use cases.

Dependency Graph
----------------

.. code-block:: text

   fit()
     │
     ├──► CurveFit
     │       │
     │       ├──► DataPreprocessor
     │       ├──► OptimizationSelector
     │       ├──► LeastSquares
     │       │       │
     │       │       └──► TrustRegionReflective
     │       │
     │       ├──► CovarianceComputer
     │       └──► StreamingCoordinator
     │
     ├──► MemoryBudgetSelector
     │
     ├──► Diagnostics (optional, compute_diagnostics=True)
     │       ├──► IdentifiabilityAnalyzer
     │       └──► GradientHealthReport
     │
     └──► GlobalOptimization (optional)
             │
             ├──► MultiStartOrchestrator
             └──► CMAESOptimizer

Circular dependencies are broken via:

1. **Lazy imports**: Import at function call time
2. **TYPE_CHECKING**: Type hints without runtime import
3. **Facades**: Lazy-loading wrappers

Next Steps
----------

- :doc:`optimization_pipeline` - Data flow through the system
- :doc:`jax_patterns` - JAX programming patterns
