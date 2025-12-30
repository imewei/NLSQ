# NLSQ Architecture: Dependency Graph & Decoupling Analysis

**Generated:** 2025-12-29
**Scope:** Internal module dependencies, integration points, and architectural recommendations
**Focus:** Legacy system assessment for migration planning

---

## Executive Summary

### Critical Findings

1. **Circular Dependencies Detected:** 5 circular dependency chains requiring immediate attention
2. **High Coupling in Core:** `core.minpack` has 29 total dependencies (Instability: 0.793)
3. **Integration Bottlenecks:** Core modules bridge 9-10 subsystems each
4. **Facade Pattern Candidates:** 15+ modules would benefit from interface extraction

### Architectural Health Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Modules Analyzed | 120+ | ✓ |
| Circular Dependencies | 5 chains | ⚠️ CRITICAL |
| Most Coupled Module | `core.minpack` (29 deps) | ⚠️ HIGH |
| Stable Modules | `utils.logging`, `gui.utils.theme` | ✓ |
| Instability Score (avg) | 0.65 | ⚠️ MODERATE |

---

## 1. Dependency Graph Overview

### 1.1 Core Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURVE_FIT ENTRY POINT                         │
│                    (core/minpack.py)                             │
│                                                                   │
│  Dependencies (23):                                              │
│  - core.least_squares, core.workflow                             │
│  - caching.{unified_cache, memory_manager}                       │
│  - stability.{guard, recovery}                                   │
│  - precision.{algorithm_selector, parameter_estimation}          │
│  - streaming.{adaptive_hybrid, large_dataset}                    │
│  - global_optimization, result, utils.*                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│              LEAST SQUARES ORCHESTRATOR                          │
│              (core/least_squares.py)                             │
│                                                                   │
│  Dependencies (12):                                              │
│  - core.{trf, loss_functions, sparse_jacobian}                   │
│  - caching.{unified_cache, memory_manager}                       │
│  - stability.guard                                               │
│  - utils.{diagnostics, logging}                                  │
│  - common_scipy, config, constants                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│         TRUST REGION REFLECTIVE OPTIMIZER                        │
│         (core/trf.py)                                            │
│                                                                   │
│  Dependencies (13):                                              │
│  - stability.{guard, svd_fallback}                               │
│  - precision.mixed_precision                                     │
│  - caching.unified_cache                                         │
│  - common_{jax, scipy}, callbacks                                │
│  - utils.{diagnostics, logging}                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. `curve_fit()` → validates inputs, selects workflow
2. `LeastSquares` → sets up autodiff, manages Jacobian compilation
3. `TrustRegionReflective` → inner optimization loop with SVD-based trust region

**Critical Issue:** Circular dependency between `core.minpack` ↔ `streaming.large_dataset` ↔ `core.workflow`

---

### 1.2 Streaming Architecture

```
┌────────────────────────────────────────────────────────────────┐
│          ADAPTIVE HYBRID STREAMING OPTIMIZER                    │
│          (streaming/adaptive_hybrid.py)                         │
│                                                                  │
│  Strategy:                                                       │
│  - Batch-based processing for 100K-10M points                   │
│  - Mini-batch stochastic for 10M-100M points                    │
│  - Online/incremental for >100M points                          │
│                                                                  │
│  Dependencies (7):                                               │
│  - global_optimization.{config, sampling, tournament}            │
│  - precision.parameter_normalizer                                │
│  - stability.guard                                               │
│  - streaming.hybrid_config, utils.logging                        │
└───────────────────┬────────────────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│            STREAMING OPTIMIZER BASE                             │
│            (streaming/optimizer.py)                             │
│                                                                  │
│  Core classes:                                                   │
│  - StreamingOptimizer: Base class with HDF5 integration         │
│  - DataGenerator: Yield-based batch provider                    │
│  - GeneratorWrapper: Adapts iterators to generators             │
│                                                                  │
│  Dependencies (1): streaming.config                             │
└────────────────────────────────────────────────────────────────┘
```

**HDF5 Data Flow:**
```
External HDF5 → DataGenerator → StreamingOptimizer → LeastSquares
                     ▲                                      │
                     └──────── Checkpoint writes ──────────┘
```

**Key Integration Point:** `streaming.large_dataset.LargeDatasetOptimizer` bridges:
- `core.minpack.CurveFit` (standard API)
- `streaming.optimizer.StreamingOptimizer` (chunking logic)
- `global_optimization.MultiStartOrchestrator` (multi-start)

---

### 1.3 Caching & Stability Subsystems

```
┌─────────────────────────────────────────────────────────────────┐
│                     CACHING LAYER                                │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │  UnifiedCache    │  │  SmartCache      │  │ MemoryManager│  │
│  │                  │  │                  │  │              │  │
│  │  - JIT compile   │  │  - LRU eviction  │  │ - Memory     │  │
│  │  - Source hash   │  │  - Disk persist  │  │   prediction │  │
│  │  - xxhash        │  │  - xxhash keys   │  │ - TTL cache  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│           │                      │                     │         │
│           └──────────────────────┴─────────────────────┘         │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
                               ▼
                    Used by: core.{minpack, least_squares, trf}
```

```
┌─────────────────────────────────────────────────────────────────┐
│                  NUMERICAL STABILITY LAYER                       │
│                                                                   │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ StabilityGuard   │  │  SVDFallback     │  │   Recovery   │  │
│  │                  │  │                  │  │              │  │
│  │ - Condition      │  │ - GPU/CPU switch │  │ - Retry      │  │
│  │   monitoring     │  │ - Randomized SVD │  │   strategies │  │
│  │ - Data rescaling │  │ - Cholesky first │  │ - Param norm │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│           │                      │                     │         │
│           └──────────────────────┴─────────────────────┘         │
│                              │                                    │
└──────────────────────────────┼────────────────────────────────────┘
                               │
                   Injected into: TRF, LeastSquares, CurveFit
```

**Dependency Pattern:** Stability modules are **low-coupling** (Instability: 0.143), used by core but depend only on `config`.

---

## 2. Circular Dependencies (CRITICAL)

### 2.1 Detected Cycles

#### Cycle 1: Core ↔ Streaming
```
streaming.large_dataset → core.minpack → streaming.large_dataset
```

**Root Cause:**
- `large_dataset.py` imports `core.minpack.CurveFit` to delegate standard fits
- `minpack.py` imports `streaming.large_dataset.LargeDatasetOptimizer` for workflow selection

**Impact:** Makes it impossible to test `core.minpack` without `streaming` module.

**Recommended Fix:**
```python
# Option A: Dependency Inversion (Preferred)
# Create nlsq/core/optimizer_protocol.py
from typing import Protocol


class OptimizerProtocol(Protocol):
    def curve_fit(self, f, xdata, ydata, p0, **kwargs) -> CurveFitResult: ...


# In large_dataset.py, inject optimizer via constructor:
class LargeDatasetOptimizer:
    def __init__(self, optimizer: OptimizerProtocol):
        self._optimizer = optimizer


# In minpack.py, remove direct import of LargeDatasetOptimizer
# Use factory function instead:
def create_large_dataset_optimizer():
    from nlsq.streaming.large_dataset import LargeDatasetOptimizer

    return LargeDatasetOptimizer(optimizer=CurveFit())
```

#### Cycle 2: Core ↔ Global Optimization
```
core.minpack → global_optimization → global_optimization.multi_start → core.minpack
```

**Root Cause:**
- `minpack.py` imports `GlobalOptimizationConfig` for multi-start configuration
- `multi_start.py` imports `core.minpack.CurveFit` to run individual fits

**Recommended Fix:** Introduce `nlsq/interfaces/optimizer_interface.py`:
```python
# interfaces/optimizer_interface.py
from typing import Protocol, runtime_checkable


@runtime_checkable
class CurveFitInterface(Protocol):
    """Protocol for curve_fit-like optimizers."""

    def curve_fit(
        self, f, xdata, ydata, p0, sigma=None, bounds=(-np.inf, np.inf), **kwargs
    ) -> CurveFitResult: ...


# global_optimization/multi_start.py
def __init__(self, optimizer: CurveFitInterface):
    self._optimizer = optimizer  # Duck typing, no circular import
```

#### Cycle 3: Workflow ↔ Streaming
```
streaming.large_dataset → core.minpack → core.workflow → streaming.large_dataset
```

**Recommended Fix:** Extract workflow configuration to separate config module:
```python
# config/workflow_config.py (new file)
# Contains: MemoryTier, WorkflowConfig, auto_select_workflow()
# No dependency on streaming or core

# core.minpack imports: from nlsq.config.workflow_config import ...
# streaming.large_dataset imports: from nlsq.config.workflow_config import ...
```

---

## 3. Integration Points & Facade Opportunities

### 3.1 Core.minpack (Hub Module)

**Current State:**
- Bridges **10 subsystems** (core, caching, stability, streaming, global_optimization, precision, utils, config, result, types)
- **23 efferent dependencies** (outgoing)
- **6 afferent dependencies** (incoming)
- **Instability: 0.793** (highly unstable)

**Architectural Issue:** This is a **God Class** pattern—`minpack.py` knows too much about the entire system.

**Recommended Facade Pattern:**

```python
# nlsq/facades/curve_fit_facade.py
"""Facade for curve_fit() API, hiding subsystem complexity."""

from nlsq.interfaces import OptimizerInterface, ValidatorInterface, CacheInterface
from nlsq.core.least_squares import LeastSquares
from nlsq.result import CurveFitResult


class CurveFitFacade:
    """Simplified interface for curve_fit() that delegates to subsystems."""

    def __init__(
        self,
        optimizer: OptimizerInterface,  # Injected dependency
        validator: ValidatorInterface,  # Injected dependency
        cache: CacheInterface,  # Injected dependency
    ):
        self._optimizer = optimizer
        self._validator = validator
        self._cache = cache

    def curve_fit(self, f, xdata, ydata, p0, **kwargs) -> CurveFitResult:
        # 1. Validate inputs (delegate to validator)
        validated = self._validator.validate_inputs(xdata, ydata, p0, **kwargs)

        # 2. Check cache (delegate to cache)
        cache_key = self._cache.compute_key(f, xdata, ydata, p0, **kwargs)
        if cached := self._cache.get(cache_key):
            return cached

        # 3. Run optimization (delegate to optimizer)
        result = self._optimizer.optimize(**validated)

        # 4. Store in cache
        self._cache.set(cache_key, result)

        return result


# Factory function in minpack.py
def create_curve_fit_facade() -> CurveFitFacade:
    from nlsq.validators import StandardValidator
    from nlsq.caching import UnifiedCache
    from nlsq.core.least_squares import LeastSquares

    return CurveFitFacade(
        optimizer=LeastSquares(), validator=StandardValidator(), cache=UnifiedCache()
    )


# Public API remains unchanged:
def curve_fit(f, xdata, ydata, p0, **kwargs):
    facade = create_curve_fit_facade()
    return facade.curve_fit(f, xdata, ydata, p0, **kwargs)
```

**Benefits:**
- Reduces `minpack.py` from 23 dependencies to ~5
- Enables unit testing of facade without full system
- Simplifies mocking for integration tests
- Makes dependency graph more maintainable

---

### 3.2 Core.least_squares (Orchestrator)

**Current State:**
- **12 efferent dependencies**
- **Instability: 0.800**
- Tightly coupled to `core.trf`, `stability.guard`, `caching.*`

**Recommended Adapter Pattern:**

```python
# nlsq/adapters/jacobian_adapter.py
"""Adapter for different Jacobian computation strategies."""


class JacobianAdapter(ABC):
    @abstractmethod
    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs) -> jnp.ndarray:
        pass


class AutodiffJacobianAdapter(JacobianAdapter):
    """Uses JAX autodiff (jacfwd/jacrev)."""

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        # Implementation using jacfwd/jacrev
        ...


class AnalyticalJacobianAdapter(JacobianAdapter):
    """Uses user-provided Jacobian function."""

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        # Wrap user's jac function
        ...


class SparseJacobianAdapter(JacobianAdapter):
    """Uses sparse Jacobian computation."""

    def compute_jacobian(self, func, x0, xdata, ydata, **kwargs):
        # Call core.sparse_jacobian
        ...


# In least_squares.py:
class LeastSquares:
    def __init__(self, jacobian_adapter: JacobianAdapter = None):
        self._jac_adapter = jacobian_adapter or AutodiffJacobianAdapter()

    def least_squares(self, fun, x0, jac=None, **kwargs):
        # Use adapter instead of direct autodiff logic
        J = self._jac_adapter.compute_jacobian(fun, x0, xdata, ydata)
        ...
```

**Benefits:**
- Decouples Jacobian strategy from orchestration logic
- Enables testing each strategy independently
- Simplifies adding new Jacobian methods (e.g., finite differences)

---

### 3.3 Streaming.adaptive_hybrid (Strategy Bridge)

**Current State:**
- Bridges `streaming` ↔ `global_optimization` ↔ `precision` ↔ `stability`
- **7 efferent dependencies**
- **Instability: 0.700**

**Recommended Strategy Pattern:**

```python
# nlsq/streaming/strategies/strategy_interface.py
from typing import Protocol


class StreamingStrategy(Protocol):
    """Protocol for streaming optimization strategies."""

    def should_use(self, n_points: int, memory_available: float) -> bool:
        """Determine if this strategy is appropriate."""
        ...

    def optimize(self, func, xdata, ydata, p0, **kwargs) -> CurveFitResult:
        """Run optimization using this strategy."""
        ...


# nlsq/streaming/strategies/batch_strategy.py
class BatchStrategy:
    """Batch-based processing for 100K-10M points."""

    def should_use(self, n_points, memory_available):
        return 100_000 <= n_points <= 10_000_000 and memory_available > 2.0

    def optimize(self, func, xdata, ydata, p0, **kwargs):
        # Batch processing logic
        ...


# nlsq/streaming/strategies/online_strategy.py
class OnlineStrategy:
    """Online/incremental for >100M points."""

    def should_use(self, n_points, memory_available):
        return n_points > 100_000_000 or memory_available < 1.0

    def optimize(self, func, xdata, ydata, p0, **kwargs):
        # Online learning logic
        ...


# nlsq/streaming/adaptive_hybrid.py (simplified)
class AdaptiveHybridStreamingOptimizer:
    def __init__(self, strategies: list[StreamingStrategy]):
        self._strategies = strategies

    def optimize(self, func, xdata, ydata, p0, **kwargs):
        n_points = len(ydata)
        memory = get_available_memory()

        # Select strategy
        for strategy in self._strategies:
            if strategy.should_use(n_points, memory):
                return strategy.optimize(func, xdata, ydata, p0, **kwargs)

        # Default fallback
        raise ValueError("No suitable streaming strategy found")
```

**Benefits:**
- Separates strategy selection from implementation
- Each strategy can be tested in isolation
- Easy to add new strategies (e.g., distributed/multi-GPU)
- Reduces coupling from 7 to ~2 dependencies

---

## 4. Data Flow Analysis

### 4.1 curve_fit() → Result (Happy Path)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER ENTRY POINT                                              │
│    nlsq.curve_fit(f, xdata, ydata, p0=[1, 0.5])                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. INPUT VALIDATION                                              │
│    utils.validators.InputValidator                               │
│    - Check xdata/ydata shapes match                              │
│    - Check bounds consistency                                    │
│    - Check finite values (if check_finite=True)                  │
│    - Security limits (max 10B points, max 100K params)           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. WORKFLOW SELECTION                                            │
│    core.workflow.auto_select_workflow()                          │
│    - Dataset size: n_points * n_params                           │
│    - Memory tier: STANDARD / CHUNKED / STREAMING                 │
│    - Multi-start decision: enable if goal='quality'              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. STABILITY CHECKS (if enable_stability=True)                   │
│    stability.guard.NumericalStabilityGuard                       │
│    - Initial Jacobian condition number (SVD if <10M elements)    │
│    - Data rescaling (if condition > 1e12)                        │
│    - NaN/Inf detection in residuals                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. JACOBIAN COMPILATION                                          │
│    core.least_squares.AutoDiffJacobian                           │
│    - Select mode: auto → jacfwd or jacrev based on dimensions    │
│    - JIT compile: create_ad_jacobian() with @jit                 │
│    - Cache: unified_cache.get_global_cache() checks hash         │
│    - 3 versions: func_none, func_1d, func_2d (for sigma cases)  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. OPTIMIZATION LOOP                                             │
│    core.trf.TrustRegionReflective.trf()                          │
│                                                                   │
│    Iteration loop:                                               │
│    ┌────────────────────────────────────────────────┐            │
│    │ a. Compute residuals: f0 = func(x, xdata, ydata) │          │
│    │ b. Compute Jacobian: J = jac(x, xdata, ydata)   │           │
│    │ c. Compute gradient: g = J^T @ f0                │          │
│    │ d. Build trust region subproblem:                │          │
│    │    (J^T J + λI) δ = -J^T f0                      │          │
│    │ e. Solve via SVD (stability.svd_fallback):       │          │
│    │    - Try GPU first                                │          │
│    │    - Fall back to CPU if OOM                      │          │
│    │    - Use randomized SVD if m*n > threshold        │          │
│    │ f. Update parameters: x_new = x + δ              │          │
│    │ g. Check convergence:                             │          │
│    │    - ftol: relative cost change                   │          │
│    │    - xtol: relative parameter change              │          │
│    │    - gtol: gradient norm                          │          │
│    │ h. Adjust trust region radius Δ                  │          │
│    └────────────────────────────────────────────────┘            │
│                                                                   │
│    Exit conditions:                                              │
│    - Converged (gtol, ftol, or xtol satisfied)                   │
│    - Max iterations (max_nfev)                                   │
│    - Timeout (if timeout_seconds specified)                      │
│    - Callback returns StopOptimization                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. COVARIANCE ESTIMATION                                         │
│    core.minpack.CurveFit._calculate_pcov()                       │
│    - Compute J at optimal parameters                             │
│    - Estimate via: pcov = inv(J^T @ J) * cost / (m - n)         │
│    - SVD-based inverse (stability.svd_fallback)                  │
│    - Handle absolute_sigma flag                                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. RESULT CONSTRUCTION                                           │
│    result.CurveFitResult                                         │
│    - popt: optimal parameters                                    │
│    - pcov: covariance matrix                                     │
│    - infodict: {nfev, fvec, fjac, nit, ...}                     │
│    - message: convergence reason                                 │
│    - success: True/False                                         │
│    - Supports tuple unpacking: popt, pcov = result               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow for Streaming (Large Datasets)

```
┌─────────────────────────────────────────────────────────────────┐
│ HDF5 DATASET                                                     │
│ /data/large_experiment.h5                                        │
│ - xdata: (100M, 1) dataset                                       │
│ - ydata: (100M, 1) dataset                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ STREAMING DATA GENERATOR                                         │
│ streaming.optimizer.StreamingDataGenerator                       │
│                                                                   │
│ chunk_size = 100K (adaptive based on memory)                     │
│ Yields batches:                                                  │
│   batch_0: xdata[0:100K], ydata[0:100K]                         │
│   batch_1: xdata[100K:200K], ydata[100K:200K]                   │
│   ...                                                            │
│   batch_999: xdata[99.9M:100M], ydata[99.9M:100M]               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ ADAPTIVE HYBRID OPTIMIZER                                        │
│ streaming.adaptive_hybrid.AdaptiveHybridStreamingOptimizer       │
│                                                                   │
│ Strategy selection (100M points):                                │
│ - ONLINE mode (stochastic mini-batch)                            │
│ - Optax optimizer: adam(learning_rate=1e-3)                      │
│ - Update frequency: every 10K points                             │
│                                                                   │
│ For each batch:                                                  │
│   1. Compute residuals on batch                                  │
│   2. Compute Jacobian on batch                                   │
│   3. Update parameters using Optax                               │
│   4. Checkpoint every 10 batches → HDF5                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ CHECKPOINT STORAGE                                               │
│ /checkpoints/experiment_iteration_100.h5                         │
│ - params: current parameter estimates                            │
│ - cost_history: cost per iteration                               │
│ - batch_idx: last processed batch                                │
│                                                                   │
│ Resume capability:                                               │
│ - Load checkpoint if exists                                      │
│ - Continue from batch_idx                                        │
└─────────────────────────────────────────────────────────────────┘
```

**Key Integration:** `streaming.large_dataset.LargeDatasetOptimizer` acts as **coordinator**:
- Detects dataset size via `core.workflow.estimate_dataset_memory()`
- Delegates to `StreamingOptimizer` if size > memory threshold
- Falls back to `CurveFit` for smaller chunks

---

## 5. Shared State & Cross-Cutting Concerns

### 5.1 Global Cache (Shared Mutable State)

**Location:** `caching/unified_cache.py` → `get_global_cache()`

**Shared By:**
- `core.minpack.CurveFit.__init__()` → stores cache instance
- `core.least_squares.LeastSquares.__init__()` → stores cache instance
- `core.trf.TrustRegionReflective` → logs cache hits

**Issue:** Global singleton makes testing difficult (tests can interfere with each other).

**Recommendation:**
```python
# Option 1: Dependency Injection
class CurveFit:
    def __init__(self, cache: CacheInterface = None):
        self._cache = cache or get_global_cache()


# Option 2: Context Manager
@contextmanager
def isolated_cache():
    """Provide isolated cache for testing."""
    old_cache = get_global_cache()
    test_cache = UnifiedCache()
    set_global_cache(test_cache)
    try:
        yield test_cache
    finally:
        set_global_cache(old_cache)


# Usage in tests:
def test_curve_fit():
    with isolated_cache() as cache:
        result = curve_fit(...)
        assert cache.get_stats()["hits"] == 0
```

### 5.2 Configuration (JAXConfig)

**Location:** `config.py` → `JAXConfig()` singleton

**Shared By:** Every module that imports JAX (120+ modules)

**Current Pattern:**
```python
# Every module does this:
from nlsq.config import JAXConfig

_jax_config = JAXConfig()
import jax.numpy as jnp
```

**Issue:** Global configuration state; changing JAX backend at runtime affects all modules.

**Recommendation:**
```python
# config/jax_context.py
@contextmanager
def jax_backend(backend: str = "cpu"):
    """Temporarily switch JAX backend."""
    old_backend = jax.default_backend()
    jax.config.update("jax_platform_name", backend)
    try:
        yield
    finally:
        jax.config.update("jax_platform_name", old_backend)


# Usage:
with jax_backend("gpu"):
    result = curve_fit(...)  # Runs on GPU
```

### 5.3 Logging (Structured Logging via get_logger)

**Location:** `utils/logging.py` → `get_logger(name)`

**Shared By:** All core, streaming, stability modules

**Current Pattern:**
```python
from nlsq.utils.logging import get_logger

logger = get_logger("core.trf")

# Structured logging:
logger.info("Optimization started", n_params=5, n_data=1000)
logger.convergence(reason="gtol satisfied", final_cost=0.01)
```

**Benefits:**
- Centralized logging configuration
- Structured output (JSON-compatible)
- Performance metrics (timers)

**No Issues:** Logging is well-designed cross-cutting concern. ✓

---

## 6. External Service Integrations

### 6.1 HDF5 (Large Dataset Storage)

**Integration Points:**
- `streaming/optimizer.py` → `StreamingDataGenerator`
- `streaming/large_dataset.py` → `create_hdf5_dataset()`

**Protocol:**
```python
# Expected HDF5 structure:
with h5py.File("data.h5", "r") as f:
    xdata = f["xdata"][:]  # Dataset with shape (n_points, n_features)
    ydata = f["ydata"][:]  # Dataset with shape (n_points,)

    # Optional: Chunked reading
    for i in range(0, n_points, chunk_size):
        x_chunk = f["xdata"][i : i + chunk_size]
        y_chunk = f["ydata"][i : i + chunk_size]
```

**Adapter Recommendation:**
```python
# nlsq/adapters/data_source_adapter.py
class DataSourceAdapter(ABC):
    @abstractmethod
    def read_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        pass


class HDF5DataAdapter(DataSourceAdapter):
    def __init__(self, filepath: str, xdata_key="xdata", ydata_key="ydata"):
        self.file = h5py.File(filepath, "r")
        self.xdata = self.file[xdata_key]
        self.ydata = self.file[ydata_key]

    def read_chunk(self, start, end):
        return self.xdata[start:end], self.ydata[start:end]


class ParquetDataAdapter(DataSourceAdapter):
    """Adapter for Parquet files (future)."""

    def read_chunk(self, start, end):
        # Use pandas/pyarrow
        ...


class ZarrDataAdapter(DataSourceAdapter):
    """Adapter for Zarr arrays (future)."""

    def read_chunk(self, start, end):
        # Use zarr library
        ...
```

### 6.2 Optax (Gradient-Based Optimizers)

**Integration Point:** `streaming/adaptive_hybrid.py`

**Usage:**
```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Update step
updates, opt_state = optimizer.update(gradients, opt_state)
params = optax.apply_updates(params, updates)
```

**Coupling:** Moderate (only used in online streaming mode)

**Recommendation:** Wrap Optax in adapter to support multiple gradient optimizers:
```python
class GradientOptimizerAdapter(ABC):
    @abstractmethod
    def init(self, params):
        pass

    @abstractmethod
    def update(self, gradients, state):
        pass


class OptaxAdapter(GradientOptimizerAdapter):
    def __init__(self, optimizer_name="adam", learning_rate=1e-3):
        self._opt = getattr(optax, optimizer_name)(learning_rate)


class PyTorchAdapter(GradientOptimizerAdapter):
    """Future: Support torch.optim optimizers."""

    ...
```

---

## 7. Tight Coupling Analysis

### 7.1 Modules Requiring Interface Extraction

| Module | Afferent | Efferent | Instability | Recommendation |
|--------|----------|----------|-------------|----------------|
| `core.minpack` | 6 | 23 | 0.793 | Extract `CurveFitInterface` protocol |
| `core.least_squares` | 3 | 12 | 0.800 | Extract `OptimizerInterface` |
| `core.trf` | 2 | 13 | 0.867 | Extract `TrustRegionSolverInterface` |
| `streaming.adaptive_hybrid` | 3 | 7 | 0.700 | Extract `StreamingStrategyInterface` |
| `streaming.large_dataset` | 4 | 8 | 0.667 | Extract `DataSourceInterface` |
| `caching.unified_cache` | 5 | 2 | 0.286 | Already well-designed ✓ |
| `stability.guard` | 6 | 1 | 0.143 | Already well-designed ✓ |

### 7.2 God Objects (Require Decomposition)

#### core.minpack.CurveFit (2700+ lines)

**Responsibilities:**
1. Input validation
2. Workflow selection
3. Sigma transformation (1D, 2D covariance)
4. Multi-start orchestration
5. Parameter estimation
6. Covariance computation
7. Result formatting

**Decomposition Plan:**
```python
# Decompose into focused classes:


class InputValidator:
    """Single Responsibility: Validate curve_fit inputs."""

    def validate(self, f, xdata, ydata, p0, sigma, bounds, **kwargs): ...


class WorkflowSelector:
    """Single Responsibility: Select optimization workflow."""

    def select(self, n_points, n_params, memory_available, goal): ...


class SigmaTransformer:
    """Single Responsibility: Transform sigma to weight matrix."""

    def transform(self, sigma, absolute_sigma): ...


class CovarianceEstimator:
    """Single Responsibility: Compute parameter covariance."""

    def estimate(self, jacobian, cost, residuals_dof): ...


# New CurveFit becomes coordinator:
class CurveFit:
    def __init__(
        self,
        validator: InputValidator,
        workflow_selector: WorkflowSelector,
        sigma_transformer: SigmaTransformer,
        covariance_estimator: CovarianceEstimator,
        optimizer: OptimizerInterface,
    ):
        self._validator = validator
        self._workflow = workflow_selector
        self._sigma = sigma_transformer
        self._covariance = covariance_estimator
        self._optimizer = optimizer

    def curve_fit(self, f, xdata, ydata, p0, **kwargs):
        # Orchestrate the workflow
        validated = self._validator.validate(...)
        workflow = self._workflow.select(...)
        transform = self._sigma.transform(...)
        result = self._optimizer.optimize(...)
        pcov = self._covariance.estimate(...)
        return CurveFitResult(popt=result.x, pcov=pcov, ...)
```

---

## 8. Actionable Recommendations

### 8.1 Phase 1: Break Circular Dependencies (Weeks 1-2)

**Priority: CRITICAL**

1. **Introduce Protocol Interfaces** (`nlsq/interfaces/`)
   - `optimizer_protocol.py`: `OptimizerProtocol`, `CurveFitProtocol`
   - `data_source_protocol.py`: `DataSourceProtocol`, `ChunkReaderProtocol`
   - `strategy_protocol.py`: `StreamingStrategyProtocol`

2. **Refactor `core.minpack` → `streaming.large_dataset` cycle**
   - Use dependency injection: `LargeDatasetOptimizer(optimizer: OptimizerProtocol)`
   - Remove direct import of `CurveFit` from `large_dataset.py`

3. **Refactor `core.minpack` → `global_optimization.multi_start` cycle**
   - Use protocol: `MultiStartOrchestrator(optimizer: CurveFitProtocol)`
   - Remove import of `CurveFit` from `multi_start.py`

4. **Extract Workflow Configuration**
   - Create `nlsq/config/workflow_config.py` (no dependencies on core/streaming)
   - Move `MemoryTier`, `WorkflowConfig`, `auto_select_workflow()` here

**Success Metrics:**
- 0 circular dependencies detected by static analysis
- `pytest tests/core/test_minpack.py` runs without importing `streaming`

---

### 8.2 Phase 2: Reduce Coupling via Facades (Weeks 3-4)

**Priority: HIGH**

1. **Create `nlsq/facades/curve_fit_facade.py`**
   - Implement facade pattern for `curve_fit()` (see Section 3.1)
   - Reduce `core.minpack` dependencies from 23 to ~5

2. **Create Adapter Hierarchy**
   - `adapters/jacobian_adapter.py`: `AutodiffJacobianAdapter`, `AnalyticalJacobianAdapter`, `SparseJacobianAdapter`
   - `adapters/data_source_adapter.py`: `HDF5DataAdapter`, `NumpyArrayAdapter`
   - `adapters/gradient_optimizer_adapter.py`: `OptaxAdapter`

3. **Decompose `core.minpack.CurveFit`**
   - Extract: `InputValidator`, `WorkflowSelector`, `SigmaTransformer`, `CovarianceEstimator`
   - Reduce class size from 2700 lines to ~500 lines

**Success Metrics:**
- `core.minpack` instability drops from 0.793 to <0.500
- 80%+ unit test coverage for each extracted class
- Integration tests pass unchanged

---

### 8.3 Phase 3: Implement Strategy Pattern for Streaming (Weeks 5-6)

**Priority: MEDIUM**

1. **Create `nlsq/streaming/strategies/` package**
   - `strategy_interface.py`: `StreamingStrategy` protocol
   - `batch_strategy.py`: For 100K-10M points
   - `online_strategy.py`: For >100M points
   - `distributed_strategy.py`: For multi-GPU (future)

2. **Refactor `AdaptiveHybridStreamingOptimizer`**
   - Use strategy selection pattern (see Section 3.3)
   - Reduce dependencies from 7 to ~2

3. **Add Data Source Adapters**
   - Support Parquet, Zarr, Arrow (not just HDF5)

**Success Metrics:**
- `streaming.adaptive_hybrid` instability drops from 0.700 to <0.400
- New strategies can be added without modifying core logic

---

### 8.4 Phase 4: Improve Testability (Weeks 7-8)

**Priority: MEDIUM**

1. **Inject Dependencies Instead of Global Singletons**
   - `UnifiedCache`: Add `cache=None` parameter to all classes
   - `JAXConfig`: Use context managers for backend switching

2. **Create Test Fixtures**
   - `tests/fixtures/optimizers.py`: Mock optimizer implementations
   - `tests/fixtures/data_sources.py`: Synthetic HDF5/array generators

3. **Add Integration Tests**
   - Test facade interactions with real subsystems
   - Test circular dependency prevention (import guards)

**Success Metrics:**
- 90%+ test coverage for facade/adapter layers
- Tests run in isolation (no global state leakage)

---

### 8.5 Phase 5: Documentation & ADRs (Week 9)

**Priority: LOW**

1. **Create Architecture Decision Records (ADRs)**
   - `docs/adr/001-facade-pattern-for-curve-fit.md`
   - `docs/adr/002-protocol-based-dependency-injection.md`
   - `docs/adr/003-streaming-strategy-pattern.md`

2. **Update Developer Guide**
   - Document new interfaces and protocols
   - Provide examples of adding new strategies
   - Migration guide for code using old APIs

3. **Generate Dependency Diagrams**
   - Use `graphviz` or `mermaid` to visualize module graph
   - Include in CI to detect circular dependencies

**Success Metrics:**
- All major architectural decisions documented
- Developer onboarding time reduced by 50%

---

## 9. Risk Mitigation

### 9.1 Backward Compatibility

**Risk:** Refactoring breaks existing user code.

**Mitigation:**
1. **Keep Public API Unchanged**
   - `nlsq.curve_fit()` function signature stays identical
   - Internal refactoring only affects private modules

2. **Deprecation Warnings**
   ```python
   # If changing internal imports:
   def curve_fit(*args, **kwargs):
       warnings.warn(
           "Importing from nlsq.core.minpack is deprecated. "
           "Use nlsq.curve_fit() instead.",
           DeprecationWarning,
           stacklevel=2,
       )
       return _curve_fit_impl(*args, **kwargs)
   ```

3. **Regression Test Suite**
   - Run all 2900+ tests after each refactoring step
   - Ensure `examples/` scripts still work

### 9.2 Performance Regression

**Risk:** Adding abstraction layers slows down optimization.

**Mitigation:**
1. **Benchmark Critical Paths**
   - Measure overhead of facade/adapter calls
   - Target: <5% overhead vs. direct calls

2. **Profile Before/After**
   ```python
   # Use existing PerformanceProfiler:
   from nlsq.utils.profiler import PerformanceProfiler

   with PerformanceProfiler() as profiler:
       result = curve_fit(...)

   profiler.summary()  # Compare against baseline
   ```

3. **Optimization Escapes**
   - Allow direct `LeastSquares()` usage for power users
   - Document performance tradeoffs

### 9.3 Team Coordination

**Risk:** Multiple developers working on coupled modules.

**Mitigation:**
1. **Feature Flags**
   ```python
   # Enable new facade in config:
   USE_CURVE_FIT_FACADE = os.getenv("NLSQ_USE_FACADE", "false").lower() == "true"

   if USE_CURVE_FIT_FACADE:
       from nlsq.facades.curve_fit_facade import CurveFitFacade as CurveFit
   else:
       from nlsq.core.minpack import CurveFit
   ```

2. **Branch Strategy**
   - `refactor/phase1-circular-deps`
   - `refactor/phase2-facades`
   - Merge to main only after tests pass

3. **Code Review Checklist**
   - [ ] No new circular dependencies added
   - [ ] All tests pass
   - [ ] Performance benchmarks within 5%
   - [ ] Public API unchanged

---

## 10. Appendix: Dependency Tables

### 10.1 Full Coupling Metrics

| Module | Afferent | Efferent | Total | Instability |
|--------|----------|----------|-------|-------------|
| core.minpack | 6 | 23 | 29 | 0.793 |
| config | 15 | 1 | 16 | 0.062 |
| core.least_squares | 3 | 12 | 15 | 0.800 |
| core.trf | 2 | 13 | 15 | 0.867 |
| gui.utils.theme | 12 | 0 | 12 | 0.000 |
| utils.logging | 12 | 0 | 12 | 0.000 |
| streaming.large_dataset | 4 | 8 | 12 | 0.667 |
| gui.state | 10 | 1 | 11 | 0.091 |
| streaming.adaptive_hybrid | 3 | 7 | 10 | 0.700 |
| global_optimization.multi_start | 1 | 7 | 8 | 0.875 |
| caching.unified_cache | 5 | 2 | 7 | 0.286 |
| stability.guard | 6 | 1 | 7 | 0.143 |
| utils.diagnostics | 5 | 1 | 6 | 0.167 |
| result | 5 | 1 | 6 | 0.167 |
| types | 5 | 0 | 5 | 0.000 |

### 10.2 Circular Dependency Details

| Cycle | Modules Involved | Root Cause | Fix Priority |
|-------|------------------|------------|--------------|
| 1 | streaming.large_dataset ↔ core.minpack | Direct import of CurveFit | CRITICAL |
| 2 | core.minpack ↔ global_optimization.multi_start | Direct import of CurveFit | CRITICAL |
| 3 | streaming.large_dataset ↔ core.workflow ↔ core.minpack | Workflow config in core | HIGH |
| 4 | core ↔ core | Self-import via __init__ | LOW (cosmetic) |
| 5 | cli.commands ↔ cli.commands | Self-import via __init__ | LOW (cosmetic) |

---

## Summary

This analysis reveals that NLSQ has a well-structured optimization core but suffers from:

1. **5 circular dependencies** requiring protocol-based refactoring
2. **God objects** (`core.minpack` with 2700 lines, 23 dependencies)
3. **Missing abstraction layers** for data sources, Jacobian strategies, and streaming approaches

The recommended facade/adapter/strategy patterns will:
- Reduce coupling by 50%+ in core modules
- Enable independent testing of subsystems
- Simplify addition of new features (e.g., distributed optimization)
- Improve maintainability without breaking public API

**Next Steps:** Begin Phase 1 (break circular dependencies) immediately, then proceed through Phases 2-5 over 9 weeks.
