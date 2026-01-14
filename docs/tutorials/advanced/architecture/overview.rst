Package Overview
================

This page describes NLSQ's package structure and module organization.

Module Hierarchy
----------------

.. code-block:: text

   nlsq/
   ├── __init__.py            # Public API exports
   │
   ├── core/                  # Core optimization
   │   ├── minpack.py              # 2500+ lines
   │   ├── least_squares.py        # LeastSquares class
   │   ├── trf.py                  # TRF algorithm (2544 lines)
   │   ├── trf_jit.py              # JIT-compiled helpers
   │   ├── profiler.py             # TRFProfiler
   │   ├── workflow.py             # 3-workflow system
   │   ├── functions.py            # Built-in models
   │   ├── factories.py            # Factory functions
   │   ├── sparse_jacobian.py      # Sparse Jacobian support
   │   ├── orchestration/          # v0.6.4 components
   │   │   ├── data_preprocessor.py
   │   │   ├── optimization_selector.py
   │   │   ├── covariance_computer.py
   │   │   └── streaming_coordinator.py
   │   └── adapters/               # Protocol adapters
   │       └── curve_fit_adapter.py
   │
   ├── interfaces/            # Protocol definitions
   │   ├── optimizer_protocol.py   # OptimizerProtocol
   │   ├── cache_protocol.py       # CacheProtocol
   │   ├── orchestration_protocol.py
   │   └── ...
   │
   ├── streaming/             # Large datasets
   │   ├── optimizer.py            # Base streaming
   │   ├── large_dataset.py        # LargeDatasetFitter
   │   ├── adaptive_hybrid.py      # Hybrid streaming
   │   ├── telemetry.py            # Defense monitoring
   │   └── validators.py           # Config validation
   │
   ├── caching/               # Performance
   │   ├── memory_manager.py       # Memory pooling
   │   ├── smart_cache.py          # JIT caching
   │   └── compilation_cache.py    # Persistent cache
   │
   ├── stability/             # Numerical stability
   │   ├── guard.py                # NumericalStabilityGuard
   │   ├── svd_fallback.py         # SVD fallback
   │   └── condition_monitor.py    # Condition tracking
   │
   ├── precision/             # Precision control
   │   ├── mixed_precision.py
   │   └── parameter_normalizer.py
   │
   ├── facades/               # Lazy loading
   │   ├── optimization_facade.py
   │   ├── stability_facade.py
   │   └── diagnostics_facade.py
   │
   ├── global_optimization/   # Global search
   │   ├── multi_start.py
   │   ├── cmaes_optimizer.py
   │   └── method_selector.py
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
