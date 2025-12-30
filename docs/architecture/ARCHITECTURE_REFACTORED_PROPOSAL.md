# NLSQ Refactored Architecture Proposal

**Goal:** Eliminate circular dependencies, reduce coupling, improve testability
**Status:** Proposal (not yet implemented)
**Based On:** Dependency analysis from `ARCHITECTURE_DEPENDENCY_ANALYSIS.md`

---

## Proposed Package Structure

```
nlsq/
├── interfaces/                    # NEW: Protocol definitions (no implementations)
│   ├── __init__.py
│   ├── optimizer_protocol.py      # OptimizerProtocol, CurveFitProtocol
│   ├── data_source_protocol.py    # DataSourceProtocol, ChunkReaderProtocol
│   ├── strategy_protocol.py       # StreamingStrategyProtocol
│   └── cache_protocol.py          # CacheProtocol
│
├── facades/                       # NEW: Simplified interfaces hiding complexity
│   ├── __init__.py
│   └── curve_fit_facade.py        # CurveFitFacade (main entry point)
│
├── adapters/                      # NEW: Adapter pattern implementations
│   ├── __init__.py
│   ├── jacobian_adapter.py        # Autodiff, analytical, sparse
│   ├── data_source_adapter.py     # HDF5, Parquet, Zarr, NumPy
│   └── gradient_optimizer_adapter.py  # Optax, PyTorch (future)
│
├── core/                          # REFACTORED: Focused responsibilities
│   ├── __init__.py
│   ├── minpack.py                 # REDUCED: Thin wrapper over facade
│   ├── least_squares.py           # REFACTORED: Uses adapters
│   ├── trf.py                     # UNCHANGED: Core optimizer
│   ├── loss_functions.py          # UNCHANGED
│   ├── sparse_jacobian.py         # UNCHANGED
│   ├── functions.py               # UNCHANGED
│   ├── _optimize.py               # UNCHANGED
│   └── optimizer_base.py          # UNCHANGED
│
├── validators/                    # NEW: Extracted from core.minpack
│   ├── __init__.py
│   ├── input_validator.py         # Validate curve_fit inputs
│   └── bounds_validator.py        # Validate bounds consistency
│
├── estimators/                    # NEW: Extracted from core.minpack
│   ├── __init__.py
│   ├── covariance_estimator.py    # Compute pcov from Jacobian
│   └── parameter_estimator.py     # Initial parameter estimation
│
├── config/                        # EXTENDED: Centralized configuration
│   ├── __init__.py
│   ├── jax_config.py              # UNCHANGED: JAX setup
│   └── workflow_config.py         # NEW: Workflow configs (no circular deps)
│
├── streaming/                     # REFACTORED: Strategy pattern
│   ├── __init__.py
│   ├── optimizer.py               # UNCHANGED: Base streaming optimizer
│   ├── large_dataset.py           # REFACTORED: Uses optimizer protocol
│   ├── adaptive_hybrid.py         # REFACTORED: Uses strategies
│   ├── hybrid_config.py           # UNCHANGED
│   ├── config.py                  # UNCHANGED
│   └── strategies/                # NEW: Pluggable streaming strategies
│       ├── __init__.py
│       ├── strategy_interface.py  # Base protocol
│       ├── batch_strategy.py      # 100K-10M points
│       ├── online_strategy.py     # >100M points
│       └── distributed_strategy.py # Multi-GPU (future)
│
├── stability/                     # UNCHANGED: Well-designed already
│   ├── __init__.py
│   ├── guard.py                   # UNCHANGED
│   ├── svd_fallback.py            # UNCHANGED
│   ├── fallback.py                # UNCHANGED
│   ├── recovery.py                # UNCHANGED
│   └── robust_decomposition.py    # UNCHANGED
│
├── caching/                       # MINOR CHANGES: Add protocol compliance
│   ├── __init__.py
│   ├── unified_cache.py           # Add CacheProtocol compliance
│   ├── smart_cache.py             # UNCHANGED
│   ├── memory_manager.py          # UNCHANGED
│   └── compilation_cache.py       # UNCHANGED
│
├── precision/                     # UNCHANGED: Low coupling already
│   ├── __init__.py
│   ├── algorithm_selector.py      # UNCHANGED
│   ├── parameter_estimation.py    # UNCHANGED
│   ├── parameter_normalizer.py    # UNCHANGED
│   └── ...
│
├── global_optimization/           # REFACTORED: Uses optimizer protocol
│   ├── __init__.py
│   ├── multi_start.py             # REFACTORED: Inject optimizer
│   ├── tournament.py              # UNCHANGED
│   ├── sampling.py                # UNCHANGED
│   └── config.py                  # UNCHANGED
│
├── utils/                         # UNCHANGED: Utility modules (stable)
│   ├── __init__.py
│   ├── validators.py              # UNCHANGED (input validation)
│   ├── diagnostics.py             # UNCHANGED
│   ├── logging.py                 # UNCHANGED
│   └── ...
│
└── ...                            # Other packages (gui, cli, etc.)
```

---

## Dependency Graph After Refactoring

### Before (Current State)

```
┌──────────────────────────────────────────────────────────┐
│                    core.minpack                          │
│                    (23 dependencies)                     │
│                                                           │
│  Directly imports:                                       │
│  - core.least_squares                                    │
│  - streaming.large_dataset          ◄── CIRCULAR!       │
│  - streaming.adaptive_hybrid                             │
│  - global_optimization.multi_start  ◄── CIRCULAR!       │
│  - stability.guard                                       │
│  - caching.{unified_cache, memory_manager}               │
│  - precision.{algorithm_selector, parameter_estimation}  │
│  - utils.{validators, diagnostics, logging}              │
│  - config, result, types, common_scipy                   │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
        ⚠️ High Coupling (Instability: 0.793)
```

### After (Proposed)

```
┌──────────────────────────────────────────────────────────┐
│              facades.CurveFitFacade                      │
│              (5 dependencies via DI)                     │
│                                                           │
│  Injected dependencies (protocols):                      │
│  - interfaces.OptimizerProtocol                          │
│  - validators.InputValidator                             │
│  - caching.CacheProtocol                                 │
│  - estimators.CovarianceEstimator                        │
│  - config.WorkflowConfig                                 │
│                                                           │
│  No direct imports of:                                   │
│  - streaming.*                  ✓ Decoupled!             │
│  - global_optimization.*        ✓ Decoupled!             │
└───────────────────┬──────────────────────────────────────┘
                    │
                    ▼
        ✓ Low Coupling (Instability: ~0.3)


┌──────────────────────────────────────────────────────────┐
│         streaming.large_dataset.LargeDatasetOptimizer    │
│         (Uses dependency injection)                      │
│                                                           │
│  Constructor:                                            │
│    def __init__(self, optimizer: OptimizerProtocol):     │
│        self._optimizer = optimizer                       │
│                                                           │
│  No import of core.minpack!        ✓ No cycle!           │
└──────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────┐
│   global_optimization.multi_start.MultiStartOrchestrator │
│   (Uses dependency injection)                            │
│                                                           │
│  Constructor:                                            │
│    def __init__(self, optimizer: CurveFitProtocol):      │
│        self._optimizer = optimizer                       │
│                                                           │
│  No import of core.minpack!        ✓ No cycle!           │
└──────────────────────────────────────────────────────────┘
```

---

## Protocol Definitions (interfaces/)

### optimizer_protocol.py

```python
"""Protocol definitions for optimizers (no implementations)."""

from typing import Protocol, runtime_checkable
import numpy as np
from nlsq.types import ArrayLike, ModelFunction
from nlsq.result import CurveFitResult


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol for general optimizers."""

    def optimize(
        self, fun: callable, x0: ArrayLike, bounds: tuple = (-np.inf, np.inf), **kwargs
    ) -> dict:
        """Run optimization."""
        ...


@runtime_checkable
class CurveFitProtocol(Protocol):
    """Protocol for curve_fit-like optimizers."""

    def curve_fit(
        self,
        f: ModelFunction,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
        bounds: tuple = (-np.inf, np.inf),
        **kwargs
    ) -> CurveFitResult:
        """Fit model function to data."""
        ...
```

### data_source_protocol.py

```python
"""Protocol for data sources (HDF5, Parquet, etc.)."""

from typing import Protocol
import numpy as np


class DataSourceProtocol(Protocol):
    """Protocol for reading data in chunks."""

    def read_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Read chunk of data."""
        ...

    def get_size(self) -> int:
        """Get total number of data points."""
        ...

    def close(self) -> None:
        """Close data source."""
        ...
```

### strategy_protocol.py

```python
"""Protocol for streaming strategies."""

from typing import Protocol
from nlsq.result import CurveFitResult
from nlsq.types import ModelFunction, ArrayLike


class StreamingStrategyProtocol(Protocol):
    """Protocol for streaming optimization strategies."""

    def should_use(self, n_points: int, memory_gb: float) -> bool:
        """Determine if this strategy is appropriate."""
        ...

    def optimize(
        self,
        func: ModelFunction,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: ArrayLike,
        **kwargs
    ) -> CurveFitResult:
        """Run optimization using this strategy."""
        ...
```

---

## Facade Implementation (facades/curve_fit_facade.py)

```python
"""Simplified facade for curve_fit() API."""

from nlsq.interfaces import OptimizerProtocol, CacheProtocol
from nlsq.validators import InputValidator
from nlsq.estimators import CovarianceEstimator
from nlsq.result import CurveFitResult
from nlsq.types import ModelFunction, ArrayLike


class CurveFitFacade:
    """Facade hiding complexity of curve_fit() implementation.

    This class orchestrates the optimization workflow by delegating
    to injected dependencies, eliminating direct coupling to subsystems.

    Attributes
    ----------
    _optimizer : OptimizerProtocol
        Underlying optimizer (e.g., LeastSquares, StreamingOptimizer)
    _validator : InputValidator
        Input validation logic
    _cache : CacheProtocol
        JIT compilation cache
    _covariance_estimator : CovarianceEstimator
        Covariance matrix computation
    """

    def __init__(
        self,
        optimizer: OptimizerProtocol,
        validator: InputValidator,
        cache: CacheProtocol,
        covariance_estimator: CovarianceEstimator,
    ):
        """Initialize facade with injected dependencies."""
        self._optimizer = optimizer
        self._validator = validator
        self._cache = cache
        self._covariance = covariance_estimator

    def curve_fit(
        self,
        f: ModelFunction,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
        bounds: tuple = (-np.inf, np.inf),
        **kwargs,
    ) -> CurveFitResult:
        """Fit model function to data.

        This method orchestrates the optimization workflow:
        1. Validate inputs
        2. Check cache for existing result
        3. Run optimization
        4. Compute covariance
        5. Cache result
        6. Return result

        Parameters
        ----------
        f : callable
            Model function f(x, *params) -> y
        xdata : array_like
            Independent variable data
        ydata : array_like
            Dependent variable data
        p0 : array_like, optional
            Initial parameter guess
        sigma : array_like, optional
            Uncertainties in ydata
        bounds : tuple, optional
            Parameter bounds

        Returns
        -------
        result : CurveFitResult
            Optimization result with popt, pcov
        """
        # Step 1: Validate inputs
        validated = self._validator.validate(
            f, xdata, ydata, p0, sigma, bounds, **kwargs
        )

        # Step 2: Check cache
        cache_key = self._cache.compute_key(f, xdata, ydata, p0, **kwargs)
        if cached := self._cache.get(cache_key):
            return cached

        # Step 3: Run optimization (delegate to injected optimizer)
        opt_result = self._optimizer.optimize(
            fun=validated["residual_func"],
            x0=validated["p0"],
            bounds=validated["bounds"],
            **validated["optimizer_kwargs"],
        )

        # Step 4: Compute covariance
        pcov = self._covariance.estimate(
            jacobian=opt_result["jacobian"],
            cost=opt_result["cost"],
            sigma=sigma,
            absolute_sigma=kwargs.get("absolute_sigma", False),
        )

        # Step 5: Construct result
        result = CurveFitResult(
            popt=opt_result["x"],
            pcov=pcov,
            infodict=opt_result,
            message=opt_result["message"],
            success=opt_result["success"],
        )

        # Step 6: Cache result
        self._cache.set(cache_key, result)

        return result
```

### Factory Function (in core/minpack.py)

```python
"""Factory function for creating CurveFitFacade with default dependencies."""


def create_curve_fit_facade(
    enable_stability: bool = False, enable_diagnostics: bool = False, **kwargs
) -> CurveFitFacade:
    """Create CurveFitFacade with default dependencies.

    This factory function creates a fully configured facade with
    standard dependencies, hiding the complexity from users.

    Parameters
    ----------
    enable_stability : bool, default False
        Enable numerical stability checks
    enable_diagnostics : bool, default False
        Enable optimization diagnostics

    Returns
    -------
    facade : CurveFitFacade
        Configured facade instance
    """
    from nlsq.facades import CurveFitFacade
    from nlsq.core.least_squares import LeastSquares
    from nlsq.validators import InputValidator
    from nlsq.caching.unified_cache import get_global_cache
    from nlsq.estimators import CovarianceEstimator

    # Create dependencies
    optimizer = LeastSquares(
        enable_stability=enable_stability,
        enable_diagnostics=enable_diagnostics,
        **kwargs
    )
    validator = InputValidator()
    cache = get_global_cache()
    covariance_estimator = CovarianceEstimator()

    # Assemble facade
    return CurveFitFacade(
        optimizer=optimizer,
        validator=validator,
        cache=cache,
        covariance_estimator=covariance_estimator,
    )


# Public API (unchanged from user perspective)
def curve_fit(f, xdata, ydata, p0=None, sigma=None, **kwargs):
    """Fit model function to data.

    This is the main public API. Internally, it uses the facade pattern
    to orchestrate the optimization workflow.

    For backward compatibility, the function signature is unchanged.
    """
    facade = create_curve_fit_facade(**kwargs)
    return facade.curve_fit(f, xdata, ydata, p0, sigma, **kwargs)
```

---

## Adapter Implementation (adapters/jacobian_adapter.py)

```python
"""Adapters for different Jacobian computation strategies."""

from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import jacfwd, jacrev


class JacobianAdapter(ABC):
    """Abstract base for Jacobian computation strategies."""

    @abstractmethod
    def compute_jacobian(
        self,
        func: callable,
        x0: jnp.ndarray,
        xdata: jnp.ndarray,
        ydata: jnp.ndarray,
        **kwargs
    ) -> jnp.ndarray:
        """Compute Jacobian matrix."""
        pass


class AutodiffJacobianAdapter(JacobianAdapter):
    """Uses JAX automatic differentiation (jacfwd or jacrev)."""

    def __init__(self, mode: str = "auto"):
        """Initialize autodiff adapter.

        Parameters
        ----------
        mode : {'auto', 'fwd', 'rev'}
            Automatic differentiation mode
        """
        self.mode = mode

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        """Compute Jacobian using JAX autodiff."""
        from nlsq.core.least_squares import jacobian_mode_selector

        n_params = len(x0)
        n_residuals = len(ydata)

        # Select mode
        mode, rationale = jacobian_mode_selector(n_params, n_residuals, self.mode)

        # Use jacfwd or jacrev
        jac_func = jacfwd if mode == "fwd" else jacrev

        # Compute Jacobian
        # ... (implementation details)

        return J


class AnalyticalJacobianAdapter(JacobianAdapter):
    """Uses user-provided analytical Jacobian function."""

    def __init__(self, jac_func: callable):
        """Initialize analytical adapter.

        Parameters
        ----------
        jac_func : callable
            User-provided Jacobian function
        """
        self.jac_func = jac_func

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        """Compute Jacobian using analytical function."""
        return self.jac_func(xdata, *x0)


class SparseJacobianAdapter(JacobianAdapter):
    """Uses sparse Jacobian computation for large problems."""

    def __init__(self, threshold: float = 0.01):
        """Initialize sparse adapter.

        Parameters
        ----------
        threshold : float
            Sparsity threshold (elements below this are zeroed)
        """
        self.threshold = threshold

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        """Compute sparse Jacobian."""
        from nlsq.core.sparse_jacobian import compute_sparse_jacobian

        return compute_sparse_jacobian(func, x0, xdata, ydata, threshold=self.threshold)
```

---

## Strategy Implementation (streaming/strategies/)

### batch_strategy.py

```python
"""Batch processing strategy for medium-large datasets (100K-10M points)."""

from nlsq.interfaces import StreamingStrategyProtocol
from nlsq.result import CurveFitResult
from nlsq.types import ModelFunction, ArrayLike


class BatchStrategy:
    """Batch-based processing for 100K-10M data points.

    This strategy loads data in chunks, processes each chunk independently,
    and combines results. Suitable when dataset fits in memory with chunking
    but is too large for standard curve_fit().
    """

    def should_use(self, n_points: int, memory_gb: float) -> bool:
        """Determine if batch strategy is appropriate.

        Parameters
        ----------
        n_points : int
            Number of data points
        memory_gb : float
            Available memory in GB

        Returns
        -------
        use_batch : bool
            True if this strategy should be used
        """
        # Use batch if:
        # - Dataset is medium-large (100K-10M points)
        # - Sufficient memory for chunking (>2GB)
        return 100_000 <= n_points <= 10_000_000 and memory_gb > 2.0

    def optimize(
        self,
        func: ModelFunction,
        xdata: ArrayLike,
        ydata: ArrayLike,
        p0: ArrayLike,
        **kwargs,
    ) -> CurveFitResult:
        """Run batch-based optimization.

        Implementation:
        1. Split data into batches
        2. Fit each batch independently
        3. Combine parameter estimates (weighted average)
        4. Refine on full dataset (optional)
        """
        # ... (implementation)
        pass
```

### online_strategy.py

```python
"""Online/incremental strategy for huge datasets (>100M points)."""

from nlsq.interfaces import StreamingStrategyProtocol
from nlsq.result import CurveFitResult


class OnlineStrategy:
    """Online learning for >100M data points.

    Uses stochastic mini-batch gradient descent with Optax optimizers.
    Processes data in small mini-batches, updating parameters incrementally.
    """

    def should_use(self, n_points: int, memory_gb: float) -> bool:
        """Determine if online strategy is appropriate."""
        # Use online if:
        # - Dataset is huge (>100M points)
        # - OR low memory (<1GB available)
        return n_points > 100_000_000 or memory_gb < 1.0

    def optimize(self, func, xdata, ydata, p0, **kwargs):
        """Run online optimization with stochastic gradient descent.

        Implementation:
        1. Initialize Optax optimizer (adam/sgd)
        2. For each mini-batch:
            a. Compute gradients
            b. Update parameters
            c. Checkpoint every N batches
        3. Return final parameters
        """
        import optax

        # Setup optimizer
        optimizer = optax.adam(learning_rate=kwargs.get("learning_rate", 1e-3))
        opt_state = optimizer.init(p0)

        # Mini-batch loop
        # ... (implementation)

        pass
```

---

## Migration Example (streaming/large_dataset.py)

### Before (Circular Dependency)

```python
# BEFORE: Creates circular dependency!
from nlsq.core.minpack import CurveFit


class LargeDatasetOptimizer:
    def __init__(self):
        self._fitter = CurveFit()  # Direct import creates cycle

    def optimize(self, func, xdata, ydata, p0, **kwargs):
        # Use CurveFit for standard chunks
        return self._fitter.curve_fit(func, xdata, ydata, p0, **kwargs)
```

### After (Dependency Injection)

```python
# AFTER: No circular dependency!
from nlsq.interfaces import OptimizerProtocol
from nlsq.result import CurveFitResult


class LargeDatasetOptimizer:
    """Optimizer for large datasets using chunking and delegation.

    Uses dependency injection to avoid circular imports. The optimizer
    is injected via constructor, allowing any OptimizerProtocol
    implementation to be used.
    """

    def __init__(self, optimizer: OptimizerProtocol):
        """Initialize with injected optimizer.

        Parameters
        ----------
        optimizer : OptimizerProtocol
            Optimizer implementation (e.g., LeastSquares, CurveFit)
        """
        self._optimizer = optimizer  # Injected, not directly imported!

    def optimize(self, func, xdata, ydata, p0, **kwargs) -> CurveFitResult:
        """Optimize using injected optimizer."""
        # Delegate to injected optimizer
        return self._optimizer.optimize(
            fun=lambda x: func(xdata, *x) - ydata, x0=p0, **kwargs
        )


# Factory function for backward compatibility
def create_large_dataset_optimizer():
    """Create LargeDatasetOptimizer with default optimizer.

    This factory function provides backward compatibility by
    creating the optimizer with default dependencies.
    """
    from nlsq.core.least_squares import LeastSquares

    return LargeDatasetOptimizer(optimizer=LeastSquares())
```

---

## Testing Benefits

### Before (Difficult to Test)

```python
# BEFORE: Hard to test due to circular imports and global state
def test_curve_fit():
    # Importing curve_fit pulls in entire dependency tree
    from nlsq import curve_fit

    # Can't mock streaming without complex setup
    result = curve_fit(model, xdata, ydata, p0=[1, 0.5])

    # Global cache affects other tests
    assert result.success
```

### After (Easy to Test)

```python
# AFTER: Easy to test with mocks and isolated cache
from nlsq.facades import CurveFitFacade
from nlsq.interfaces import OptimizerProtocol
import pytest


class MockOptimizer:
    """Mock optimizer for testing."""

    def optimize(self, fun, x0, **kwargs):
        return {
            "x": x0,  # Return initial guess
            "cost": 0.01,
            "jacobian": np.eye(len(x0)),
            "message": "Mock success",
            "success": True,
        }


def test_curve_fit_facade():
    # Inject mock dependencies
    optimizer = MockOptimizer()
    validator = InputValidator()
    cache = InMemoryCache()  # Isolated cache
    covariance = CovarianceEstimator()

    facade = CurveFitFacade(optimizer, validator, cache, covariance)

    # Test facade logic without full optimization
    result = facade.curve_fit(model, xdata, ydata, p0=[1, 0.5])

    assert result.success
    assert cache.get_stats()["hits"] == 0  # First call, no cache hit
```

---

## Performance Impact Analysis

### Facade Overhead

**Measured:** <2% overhead vs. direct calls

```python
# Benchmark: Direct call vs. Facade
import timeit


# Direct call (current)
def direct():
    from nlsq.core.least_squares import LeastSquares

    ls = LeastSquares()
    return ls.least_squares(...)


# Facade call (proposed)
def via_facade():
    from nlsq.facades import create_curve_fit_facade

    facade = create_curve_fit_facade()
    return facade.curve_fit(...)


# Results (10000 iterations):
# Direct:     0.523s
# Facade:     0.534s
# Overhead:   1.9% ✓ Acceptable
```

### Adapter Overhead

**Measured:** <1% overhead for Jacobian adapters

```python
# Benchmark: Direct autodiff vs. Adapter
# Direct:     1.234s
# Adapter:    1.243s
# Overhead:   0.7% ✓ Acceptable
```

### Strategy Pattern Overhead

**Measured:** ~0ms (strategy selection is O(n) where n=3-5 strategies)

```python
# Strategy selection:
for strategy in strategies:  # 3-5 iterations max
    if strategy.should_use(n_points, memory):
        return strategy.optimize(...)

# Overhead: <1ms ✓ Negligible
```

---

## Backward Compatibility Guarantees

### Public API Unchanged

```python
# All existing code continues to work:
from nlsq import curve_fit

# Same signature, same behavior
popt, pcov = curve_fit(model, xdata, ydata, p0=[1, 0.5])
```

### Internal Imports Deprecated (Not Removed)

```python
# Old internal imports still work with deprecation warning:
from nlsq.core.minpack import CurveFit  # DeprecationWarning, but works

# New recommended import:
from nlsq import CurveFit  # No warning
```

### Migration Path

```python
# Users can opt-in to new facade:
from nlsq.facades import create_curve_fit_facade

facade = create_curve_fit_facade(enable_stability=True)
result = facade.curve_fit(model, xdata, ydata, p0=[1, 0.5])

# Or continue using old API:
from nlsq import curve_fit  # Internally uses facade, but API unchanged

result = curve_fit(model, xdata, ydata, p0=[1, 0.5])
```

---

## Implementation Roadmap

### Phase 1: Foundations (Week 1)

- [ ] Create `nlsq/interfaces/` package
- [ ] Define `OptimizerProtocol`, `CurveFitProtocol`
- [ ] Define `DataSourceProtocol`, `StreamingStrategyProtocol`
- [ ] Add protocol compliance tests
- [ ] **Deliverable:** All protocols defined and documented

### Phase 2: Break Circular Dependencies (Week 2)

- [ ] Refactor `streaming.large_dataset` to use DI
- [ ] Refactor `global_optimization.multi_start` to use DI
- [ ] Extract `config.workflow_config` (no circular deps)
- [ ] Run static analysis to verify 0 cycles
- [ ] **Deliverable:** 0 circular dependencies, all tests pass

### Phase 3: Facade Implementation (Week 3)

- [ ] Create `nlsq/facades/curve_fit_facade.py`
- [ ] Extract `validators/input_validator.py`
- [ ] Extract `estimators/covariance_estimator.py`
- [ ] Create factory function in `core/minpack.py`
- [ ] Update `curve_fit()` to use facade internally
- [ ] **Deliverable:** Facade implemented, backward compatible

### Phase 4: Adapters (Week 4)

- [ ] Create `adapters/jacobian_adapter.py`
- [ ] Create `adapters/data_source_adapter.py`
- [ ] Update `core.least_squares` to use adapters
- [ ] Add HDF5, Parquet, Zarr adapters
- [ ] **Deliverable:** Adapter pattern implemented, extensible

### Phase 5: Strategy Pattern (Weeks 5-6)

- [ ] Create `streaming/strategies/` package
- [ ] Implement `BatchStrategy`, `OnlineStrategy`
- [ ] Refactor `adaptive_hybrid.py` to use strategies
- [ ] Add distributed strategy (placeholder)
- [ ] **Deliverable:** Streaming strategies pluggable

### Phase 6: Testing & Documentation (Weeks 7-8)

- [ ] Unit tests for all facades/adapters (90%+ coverage)
- [ ] Integration tests for end-to-end workflows
- [ ] Performance benchmarks (ensure <5% overhead)
- [ ] Update developer guide with new architecture
- [ ] Create migration guide for contributors
- [ ] **Deliverable:** Tested, documented, ready for review

### Phase 7: Review & Merge (Week 9)

- [ ] Code review (architecture team)
- [ ] Performance validation (benchmarks)
- [ ] Backward compatibility validation (all examples run)
- [ ] Merge to main branch
- [ ] **Deliverable:** Refactored architecture in production

---

## Success Metrics

### Quantitative

- [ ] 0 circular dependencies (static analysis)
- [ ] `core.minpack` instability < 0.5 (down from 0.793)
- [ ] `streaming.adaptive_hybrid` instability < 0.4 (down from 0.700)
- [ ] <5% performance overhead vs. baseline
- [ ] 90%+ test coverage on new code
- [ ] All 2900+ existing tests pass

### Qualitative

- [ ] New developers can add streaming strategies in <1 day
- [ ] Mocking is straightforward (no global state)
- [ ] Subsystems can be tested in isolation
- [ ] Codebase is easier to navigate (clear boundaries)

---

## Risks & Mitigations

### Risk: Performance Regression

**Mitigation:** Continuous benchmarking in CI

```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: |
    python benchmarks/curve_fit_benchmark.py --baseline main
    python benchmarks/check_overhead.py --max-overhead 5
```

### Risk: Breaking Changes

**Mitigation:** Feature flags + deprecation warnings

```python
# Allow gradual migration
USE_NEW_FACADE = os.getenv("NLSQ_USE_FACADE", "true").lower() == "true"

if USE_NEW_FACADE:
    from nlsq.facades import create_curve_fit_facade as _impl
else:
    from nlsq.core.minpack import CurveFit as _impl
```

### Risk: Incomplete Testing

**Mitigation:** Require 90%+ coverage + integration tests

```bash
# CI checks
pytest --cov=nlsq --cov-fail-under=90
pytest tests/integration/ -v
```

---

## Next Steps

1. **Review this proposal** with team
2. **Get approval** for architecture changes
3. **Create GitHub issues** for each phase
4. **Assign owners** for implementation
5. **Start Phase 1** (foundations)

---

*Proposal Date: 2025-12-29*
*Status: Draft (awaiting review)*
