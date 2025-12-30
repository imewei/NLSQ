# NLSQ Architecture Quick Reference

**Purpose:** Fast lookup for developers working with NLSQ's internal architecture
**See Also:** `ARCHITECTURE_DEPENDENCY_ANALYSIS.md` for full details

---

## Critical Issues (Fix First)

### 🔴 Circular Dependencies (5 detected)

1. **Core ↔ Streaming**
   ```
   streaming.large_dataset → core.minpack → streaming.large_dataset
   ```
   **Fix:** Dependency injection via `OptimizerProtocol`

2. **Core ↔ Global Optimization**
   ```
   core.minpack → global_optimization.multi_start → core.minpack
   ```
   **Fix:** Protocol-based interface for `CurveFit`

3. **Workflow ↔ Streaming**
   ```
   streaming.large_dataset → core.workflow → core.minpack → streaming.large_dataset
   ```
   **Fix:** Extract `workflow_config.py` to config package

---

## Module Coupling at a Glance

### High Coupling (Needs Refactoring)

| Module | Total Deps | Instability | Issue |
|--------|-----------|-------------|-------|
| `core.minpack` | 29 | 0.793 | God Class (2700 lines) |
| `core.least_squares` | 15 | 0.800 | Too many responsibilities |
| `core.trf` | 15 | 0.867 | Tight coupling to stability |
| `streaming.adaptive_hybrid` | 10 | 0.700 | Hard-coded strategies |

### Well-Designed (Keep Stable)

| Module | Total Deps | Instability | Strength |
|--------|-----------|-------------|----------|
| `stability.guard` | 7 | 0.143 | Single responsibility |
| `utils.logging` | 12 | 0.000 | Pure dependency |
| `config` | 16 | 0.062 | Stable foundation |
| `caching.unified_cache` | 7 | 0.286 | Good abstraction |

---

## Data Flow Cheat Sheet

### Standard Fit (Small Datasets)

```
curve_fit()
  → InputValidator
  → WorkflowSelector (selects STANDARD tier)
  → StabilityGuard (check initial Jacobian)
  → AutoDiffJacobian (JIT compile jacfwd/jacrev)
  → TrustRegionReflective (optimization loop)
  → CovarianceEstimator (compute pcov from J)
  → CurveFitResult
```

**Key Files:**
- Entry: `/home/wei/Documents/GitHub/NLSQ/nlsq/core/minpack.py` (L137-300)
- Orchestrator: `/home/wei/Documents/GitHub/NLSQ/nlsq/core/least_squares.py` (L966-1275)
- Optimizer: `/home/wei/Documents/GitHub/NLSQ/nlsq/core/trf.py` (L898-1500)

### Large Dataset Fit (Streaming)

```
fit(workflow='streaming')
  → auto_select_workflow() (selects STREAMING tier)
  → LargeDatasetOptimizer
  → AdaptiveHybridStreamingOptimizer
  → StreamingDataGenerator (HDF5 chunking)
  → OnlineStrategy (mini-batch SGD)
  → checkpoint every 10 batches
  → CurveFitResult
```

**Key Files:**
- Entry: `/home/wei/Documents/GitHub/NLSQ/nlsq/core/minpack.py` (L137-300)
- Workflow: `/home/wei/Documents/GitHub/NLSQ/nlsq/core/workflow.py` (L200-400)
- Streaming: `/home/wei/Documents/GitHub/NLSQ/nlsq/streaming/adaptive_hybrid.py` (L100-500)
- HDF5: `/home/wei/Documents/GitHub/NLSQ/nlsq/streaming/optimizer.py` (L444-800)

---

## Integration Points (Cross-Subsystem Bridges)

### core.minpack (Main Hub)

**Bridges 10 subsystems:**
```
core → caching, stability, streaming, global_optimization,
       precision, utils, config, result, types, common_scipy
```

**Most Critical Dependencies:**
- `core.least_squares.LeastSquares` - Main optimizer orchestrator
- `streaming.large_dataset.LargeDatasetOptimizer` - Large dataset delegation
- `stability.guard.NumericalStabilityGuard` - Stability checks
- `caching.unified_cache.get_global_cache()` - JIT caching
- `utils.validators.InputValidator` - Input validation

**Recommendation:** Extract facade to reduce from 23 to ~5 dependencies.

### streaming.adaptive_hybrid (Strategy Hub)

**Bridges 4 subsystems:**
```
streaming → global_optimization, precision, stability, utils
```

**Dependencies:**
- `global_optimization.{sampling, tournament}` - Multi-start
- `precision.parameter_normalizer` - Parameter scaling
- `stability.guard` - Numerical checks

**Recommendation:** Implement strategy pattern to isolate batch/online/distributed modes.

---

## External Service Integrations

### HDF5 (Large Datasets)

**Used by:** `streaming/optimizer.py`, `streaming/large_dataset.py`

**Protocol:**
```python
import h5py

# Expected structure:
with h5py.File("data.h5", "r") as f:
    xdata = f["xdata"][:]  # Shape: (n_points, n_features)
    ydata = f["ydata"][:]  # Shape: (n_points,)
```

**Recommended Adapter:**
```python
class HDF5DataAdapter(DataSourceAdapter):
    def read_chunk(self, start, end):
        return self.xdata[start:end], self.ydata[start:end]
```

### Optax (Gradient Optimizers)

**Used by:** `streaming/adaptive_hybrid.py` (online mode only)

**Integration:**
```python
import optax

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)
updates, opt_state = optimizer.update(gradients, opt_state)
```

**Coupling:** Moderate (only 1 module)

---

## Facade/Adapter Opportunities

### Immediate (High Impact)

1. **CurveFitFacade** (`nlsq/facades/curve_fit_facade.py`)
   - **Reduces:** `core.minpack` from 23 deps to ~5
   - **Pattern:** Facade + Dependency Injection
   - **Effort:** 2 weeks

2. **JacobianAdapter** (`nlsq/adapters/jacobian_adapter.py`)
   - **Decouples:** Autodiff, analytical, sparse strategies
   - **Pattern:** Strategy + Adapter
   - **Effort:** 1 week

3. **DataSourceAdapter** (`nlsq/adapters/data_source_adapter.py`)
   - **Supports:** HDF5, Parquet, Zarr, NumPy arrays
   - **Pattern:** Adapter
   - **Effort:** 1 week

### Future (Medium Impact)

4. **StreamingStrategy** (`nlsq/streaming/strategies/`)
   - **Isolates:** Batch, online, distributed modes
   - **Pattern:** Strategy
   - **Effort:** 2 weeks

5. **GradientOptimizerAdapter** (`nlsq/adapters/gradient_optimizer_adapter.py`)
   - **Supports:** Optax, PyTorch, custom optimizers
   - **Pattern:** Adapter
   - **Effort:** 1 week

---

## Testing Anti-Patterns to Avoid

### 🚫 Global Singleton Leakage

**Bad:**
```python
# Tests interfere with each other
def test_curve_fit_1():
    result = curve_fit(...)  # Uses global cache


def test_curve_fit_2():
    result = curve_fit(...)  # Cache polluted by test_1
```

**Good:**
```python
# Isolated cache per test
def test_curve_fit_1():
    with isolated_cache() as cache:
        result = curve_fit(...)
        assert cache.get_stats()["hits"] == 0
```

### 🚫 Importing Circular Dependencies

**Bad:**
```python
# In large_dataset.py
from nlsq.core.minpack import CurveFit  # Creates cycle!


class LargeDatasetOptimizer:
    def __init__(self):
        self._fitter = CurveFit()
```

**Good:**
```python
# In large_dataset.py
from nlsq.interfaces import OptimizerProtocol


class LargeDatasetOptimizer:
    def __init__(self, optimizer: OptimizerProtocol):
        self._optimizer = optimizer  # Injected dependency
```

### 🚫 Hard-Coded Strategy Selection

**Bad:**
```python
# In adaptive_hybrid.py
if n_points > 100_000_000:
    return self._online_optimize(...)  # Hard-coded
elif n_points > 10_000_000:
    return self._batch_optimize(...)
else:
    return self._standard_optimize(...)
```

**Good:**
```python
# In adaptive_hybrid.py
for strategy in self._strategies:
    if strategy.should_use(n_points, memory):
        return strategy.optimize(...)  # Pluggable strategies
```

---

## Common Import Paths

### Core Optimization
```python
from nlsq import curve_fit  # Public API
from nlsq.core.minpack import CurveFit  # Class-based
from nlsq.core.least_squares import LeastSquares  # Low-level
from nlsq.core.trf import TrustRegionReflective  # Algorithm
```

### Streaming
```python
from nlsq.streaming import AdaptiveHybridStreamingOptimizer
from nlsq.streaming import StreamingOptimizer
from nlsq.streaming.large_dataset import LargeDatasetOptimizer
```

### Stability & Caching
```python
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.stability.svd_fallback import safe_svd
from nlsq.caching.unified_cache import get_global_cache
from nlsq.caching.memory_manager import get_memory_manager
```

### Utilities
```python
from nlsq.utils.validators import InputValidator
from nlsq.utils.diagnostics import OptimizationDiagnostics
from nlsq.utils.logging import get_logger
```

---

## Migration Checklist (For Refactoring)

### Phase 1: Break Circular Dependencies (Weeks 1-2)

- [ ] Create `nlsq/interfaces/optimizer_protocol.py`
- [ ] Create `nlsq/interfaces/data_source_protocol.py`
- [ ] Refactor `streaming.large_dataset` to use dependency injection
- [ ] Refactor `global_optimization.multi_start` to use protocol
- [ ] Extract `config/workflow_config.py` (no core/streaming imports)
- [ ] Run static analysis: `python -m pytest --import-mode=importlib`
- [ ] Verify 0 circular dependencies

### Phase 2: Reduce Coupling (Weeks 3-4)

- [ ] Create `nlsq/facades/curve_fit_facade.py`
- [ ] Decompose `core.minpack.CurveFit` into focused classes
- [ ] Create `adapters/jacobian_adapter.py`
- [ ] Create `adapters/data_source_adapter.py`
- [ ] Run regression tests: `pytest tests/ -v`
- [ ] Verify `core.minpack` instability < 0.5

### Phase 3: Strategy Pattern (Weeks 5-6)

- [ ] Create `streaming/strategies/strategy_interface.py`
- [ ] Implement `BatchStrategy`, `OnlineStrategy`
- [ ] Refactor `AdaptiveHybridStreamingOptimizer`
- [ ] Add Parquet/Zarr adapters (optional)
- [ ] Run streaming tests: `pytest tests/streaming/ -v`
- [ ] Verify `streaming.adaptive_hybrid` instability < 0.4

### Phase 4: Testability (Weeks 7-8)

- [ ] Add `isolated_cache()` context manager
- [ ] Add `jax_backend()` context manager
- [ ] Create test fixtures for optimizers
- [ ] Create test fixtures for data sources
- [ ] Run full test suite: `pytest tests/ --cov=nlsq`
- [ ] Verify 90%+ coverage on new code

### Phase 5: Documentation (Week 9)

- [ ] Write ADR-001: Facade pattern
- [ ] Write ADR-002: Protocol-based DI
- [ ] Write ADR-003: Streaming strategies
- [ ] Update developer guide
- [ ] Generate dependency diagrams (CI)
- [ ] Code review and merge

---

## Quick Diagnosis Commands

### Check for Circular Dependencies
```bash
# Using pytest import mode
python -m pytest --collect-only --import-mode=importlib 2>&1 | grep -i circular

# Using custom analyzer
python scripts/analyze_dependencies.py --check-cycles
```

### Measure Module Coupling
```bash
python scripts/analyze_dependencies.py --coupling-report
```

### Profile Optimization Performance
```python
from nlsq.utils.profiler import PerformanceProfiler

with PerformanceProfiler() as profiler:
    result = curve_fit(f, xdata, ydata, p0=[1, 0.5])

profiler.summary()
```

### Check Cache Hit Rate
```python
from nlsq.caching.unified_cache import get_global_cache

cache = get_global_cache()
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
```

---

## Contact & Resources

- **Full Analysis:** `ARCHITECTURE_DEPENDENCY_ANALYSIS.md`
- **Dependency Graph:** `architecture_dependency_graph.png` (generate with GraphViz)
- **Data Flow:** `architecture_data_flow.png`
- **Streaming Flow:** `architecture_streaming_flow.png`

---

*Last Updated: 2025-12-29*
