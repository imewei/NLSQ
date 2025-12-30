# NLSQ Architecture Analysis - Index

**Generated:** 2025-12-29
**Purpose:** Comprehensive dependency graph analysis for legacy system migration planning

---

## Document Overview

This architecture analysis provides a complete dependency graph, identifies circular dependencies, highlights tight coupling, and proposes refactoring solutions for the NLSQ optimization library.

### Generated Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| **ARCHITECTURE_DEPENDENCY_ANALYSIS.md** | Full dependency analysis report (40+ pages) | Architects, Tech Leads |
| **ARCHITECTURE_QUICK_REFERENCE.md** | Quick lookup guide for developers | Developers, Contributors |
| **ARCHITECTURE_REFACTORED_PROPOSAL.md** | Proposed refactored architecture with code examples | Implementation Team |
| **ARCHITECTURE_ANALYSIS_INDEX.md** | This file - navigation guide | All stakeholders |

### Visual Diagrams (DOT files)

Generated GraphViz DOT files (can be rendered to PNG/SVG):

```bash
# Generate PNG images (requires graphviz):
dot -Tpng architecture_dependency_graph.dot -o architecture_dependency_graph.png
dot -Tpng architecture_data_flow.dot -o architecture_data_flow.png
dot -Tpng architecture_streaming_flow.dot -o architecture_streaming_flow.png
```

| File | Description |
|------|-------------|
| `architecture_dependency_graph.dot` | Module dependency graph showing circular dependencies |
| `architecture_data_flow.dot` | Data flow from curve_fit() to result |
| `architecture_streaming_flow.dot` | Streaming data flow (HDF5 → optimizer → checkpoint) |

---

## Quick Navigation

### For Architects & Tech Leads

**Start Here:** `ARCHITECTURE_DEPENDENCY_ANALYSIS.md`

Key sections:
- Section 1: Dependency Graph Overview
- Section 2: Circular Dependencies (CRITICAL)
- Section 3: Integration Points & Facade Opportunities
- Section 8: Actionable Recommendations (5-phase roadmap)

### For Developers

**Start Here:** `ARCHITECTURE_QUICK_REFERENCE.md`

Key sections:
- Critical Issues (Fix First)
- Module Coupling at a Glance
- Data Flow Cheat Sheet
- Common Import Paths
- Testing Anti-Patterns to Avoid

### For Implementation Team

**Start Here:** `ARCHITECTURE_REFACTORED_PROPOSAL.md`

Key sections:
- Proposed Package Structure
- Protocol Definitions (interfaces/)
- Facade Implementation
- Adapter Implementation
- Implementation Roadmap (9-week plan)

---

## Executive Summary

### Critical Findings

1. **5 Circular Dependencies Detected**
   - `core.minpack` ↔ `streaming.large_dataset`
   - `core.minpack` ↔ `global_optimization.multi_start`
   - `core.workflow` ↔ `streaming.large_dataset` ↔ `core.minpack`

2. **High Coupling in Core Modules**
   - `core.minpack`: 29 total dependencies (Instability: 0.793)
   - `core.least_squares`: 15 dependencies (Instability: 0.800)
   - `core.trf`: 15 dependencies (Instability: 0.867)

3. **God Class Anti-Pattern**
   - `core.minpack.CurveFit`: 2700+ lines, 23 efferent dependencies
   - Violates Single Responsibility Principle

### Recommended Solutions

1. **Protocol-Based Dependency Injection**
   - Create `nlsq/interfaces/` package with protocol definitions
   - Inject dependencies via constructor (no direct imports)
   - Eliminates all circular dependencies

2. **Facade Pattern for Main API**
   - Create `CurveFitFacade` to hide subsystem complexity
   - Reduces `core.minpack` from 23 to ~5 dependencies
   - Improves testability (easy mocking)

3. **Adapter Pattern for Extensibility**
   - `JacobianAdapter`: Autodiff, analytical, sparse strategies
   - `DataSourceAdapter`: HDF5, Parquet, Zarr, NumPy
   - `StreamingStrategyAdapter`: Batch, online, distributed

### Implementation Timeline

**9 weeks total:**
- Weeks 1-2: Break circular dependencies (protocols)
- Weeks 3-4: Implement facades & adapters
- Weeks 5-6: Refactor streaming with strategy pattern
- Weeks 7-8: Testing & documentation
- Week 9: Code review & merge

---

## Key Metrics

### Current State (Before Refactoring)

| Metric | Value | Status |
|--------|-------|--------|
| Circular Dependencies | 5 chains | 🔴 CRITICAL |
| Most Coupled Module | `core.minpack` (29 deps) | 🔴 HIGH |
| Average Instability | 0.65 | 🟡 MODERATE |
| God Classes | 1 (`CurveFit`: 2700 lines) | 🔴 HIGH |
| Integration Bottlenecks | 3 modules bridge 9-10 subsystems | 🟡 MODERATE |

### Target State (After Refactoring)

| Metric | Target | Expected Benefit |
|--------|--------|------------------|
| Circular Dependencies | 0 | ✅ Testable subsystems |
| `core.minpack` Instability | <0.5 (from 0.793) | ✅ 37% reduction |
| `streaming.adaptive_hybrid` Instability | <0.4 (from 0.700) | ✅ 43% reduction |
| Performance Overhead | <5% | ✅ Acceptable tradeoff |
| Test Coverage (new code) | >90% | ✅ High quality |

---

## Architecture Patterns Applied

### 1. Facade Pattern

**Purpose:** Simplify complex subsystem interactions

**Implementation:** `nlsq/facades/curve_fit_facade.py`

**Benefits:**
- Reduces coupling from 23 to ~5 dependencies
- Hides complexity from users
- Single point of entry for curve_fit()

### 2. Dependency Injection

**Purpose:** Eliminate circular dependencies

**Implementation:** Constructor injection with protocols

**Example:**
```python
class LargeDatasetOptimizer:
    def __init__(self, optimizer: OptimizerProtocol):
        self._optimizer = optimizer  # Injected, not imported
```

**Benefits:**
- No circular imports
- Easy mocking for tests
- Flexible swapping of implementations

### 3. Adapter Pattern

**Purpose:** Provide uniform interface for varied implementations

**Implementation:** `nlsq/adapters/jacobian_adapter.py`

**Example:**
```python
class JacobianAdapter(ABC):
    @abstractmethod
    def compute_jacobian(self, func, x0, xdata, ydata):
        pass


# Concrete adapters:
# - AutodiffJacobianAdapter
# - AnalyticalJacobianAdapter
# - SparseJacobianAdapter
```

**Benefits:**
- Uniform interface for different strategies
- Easy to add new adapters (Parquet, Zarr, etc.)
- Decouples strategy selection from implementation

### 4. Strategy Pattern

**Purpose:** Encapsulate algorithms and make them interchangeable

**Implementation:** `nlsq/streaming/strategies/`

**Example:**
```python
class StreamingStrategy(Protocol):
    def should_use(self, n_points, memory):
        pass

    def optimize(self, func, xdata, ydata, p0):
        pass


# Concrete strategies:
# - BatchStrategy (100K-10M points)
# - OnlineStrategy (>100M points)
# - DistributedStrategy (multi-GPU)
```

**Benefits:**
- Pluggable streaming strategies
- Each strategy tested independently
- Easy to add new strategies

---

## Dependency Graph Highlights

### Core Optimization Pipeline

```
curve_fit() → InputValidator → WorkflowSelector → StabilityGuard
    → AutoDiffJacobian (JIT) → TrustRegionReflective → CovarianceEstimator
    → CurveFitResult
```

**Files:**
- `/home/wei/Documents/GitHub/NLSQ/nlsq/core/minpack.py` (entry)
- `/home/wei/Documents/GitHub/NLSQ/nlsq/core/least_squares.py` (orchestrator)
- `/home/wei/Documents/GitHub/NLSQ/nlsq/core/trf.py` (optimizer)

### Streaming Architecture

```
HDF5 → StreamingDataGenerator → AdaptiveHybridOptimizer
    → [BatchStrategy | OnlineStrategy] → LeastSquares
    → Checkpoint (HDF5) → CurveFitResult
```

**Files:**
- `/home/wei/Documents/GitHub/NLSQ/nlsq/streaming/optimizer.py` (generator)
- `/home/wei/Documents/GitHub/NLSQ/nlsq/streaming/adaptive_hybrid.py` (strategy)
- `/home/wei/Documents/GitHub/NLSQ/nlsq/streaming/large_dataset.py` (coordinator)

---

## Integration Points

### Subsystem Bridges (Modules Connecting Multiple Subsystems)

| Module | Bridges | Dependencies | Recommendation |
|--------|---------|--------------|----------------|
| `core.minpack` | 10 subsystems | 23 efferent | Extract facade |
| `core.trf` | 9 subsystems | 13 efferent | Inject stability/precision |
| `core.least_squares` | 8 subsystems | 12 efferent | Use adapters |
| `streaming.adaptive_hybrid` | 4 subsystems | 7 efferent | Strategy pattern |

---

## External Service Integrations

### HDF5 (Large Dataset Storage)

**Usage:** Chunked reading for datasets >100M points

**Integration Points:**
- `streaming/optimizer.py` → `StreamingDataGenerator`
- `streaming/large_dataset.py` → `create_hdf5_dataset()`

**Recommended Adapter:**
```python
class HDF5DataAdapter(DataSourceAdapter):
    def read_chunk(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        return self.xdata[start:end], self.ydata[start:end]
```

### Optax (Gradient-Based Optimizers)

**Usage:** Online learning mode for >100M points

**Integration Points:**
- `streaming/adaptive_hybrid.py` → `OnlineStrategy`

**Coupling:** Low (only 1 module uses Optax)

**Recommended Adapter:**
```python
class OptaxAdapter(GradientOptimizerAdapter):
    def __init__(self, optimizer_name="adam", learning_rate=1e-3):
        self._opt = getattr(optax, optimizer_name)(learning_rate)
```

---

## Testing Strategy

### Unit Tests (Per-Module)

**Target:** 90%+ coverage for new code

**Focus Areas:**
- Facades: Test delegation logic without real optimizers
- Adapters: Test each strategy independently
- Protocols: Test compliance with runtime_checkable

**Example:**
```python
def test_curve_fit_facade():
    optimizer = MockOptimizer()
    facade = CurveFitFacade(optimizer, validator, cache, covariance)
    result = facade.curve_fit(model, xdata, ydata, p0=[1, 0.5])
    assert result.success
```

### Integration Tests (End-to-End)

**Target:** All major workflows covered

**Focus Areas:**
- Standard fit (small datasets)
- Chunked fit (large datasets)
- Streaming fit (huge datasets)
- Multi-start optimization

**Example:**
```python
def test_large_dataset_workflow():
    # End-to-end test with real HDF5 data
    result = fit(model, xdata_hdf5, ydata_hdf5, workflow="streaming")
    assert result.success
    assert result.sparsity_detected["solver"] == "streaming"
```

### Performance Tests (Benchmarking)

**Target:** <5% overhead vs. baseline

**Focus Areas:**
- Facade overhead
- Adapter overhead
- Strategy selection overhead

**Example:**
```python
def test_facade_performance():
    baseline_time = benchmark_direct_call()
    facade_time = benchmark_facade_call()
    overhead = (facade_time - baseline_time) / baseline_time
    assert overhead < 0.05  # <5% overhead
```

---

## Migration Checklist

Use this checklist to track refactoring progress:

### Phase 1: Foundations (Week 1)

- [ ] Create `nlsq/interfaces/optimizer_protocol.py`
- [ ] Create `nlsq/interfaces/data_source_protocol.py`
- [ ] Create `nlsq/interfaces/strategy_protocol.py`
- [ ] Create `nlsq/interfaces/cache_protocol.py`
- [ ] Add protocol compliance tests
- [ ] **Deliverable:** All protocols defined and documented

### Phase 2: Break Circular Dependencies (Week 2)

- [ ] Refactor `streaming.large_dataset` to use DI
- [ ] Refactor `global_optimization.multi_start` to use DI
- [ ] Extract `config.workflow_config` (no circular deps)
- [ ] Run static analysis: `pytest --import-mode=importlib`
- [ ] **Deliverable:** 0 circular dependencies

### Phase 3: Facade Implementation (Week 3)

- [ ] Create `nlsq/facades/curve_fit_facade.py`
- [ ] Create `nlsq/validators/input_validator.py`
- [ ] Create `nlsq/estimators/covariance_estimator.py`
- [ ] Update `core/minpack.py` factory function
- [ ] **Deliverable:** Facade implemented, backward compatible

### Phase 4: Adapters (Week 4)

- [ ] Create `nlsq/adapters/jacobian_adapter.py`
- [ ] Create `nlsq/adapters/data_source_adapter.py`
- [ ] Create `nlsq/adapters/gradient_optimizer_adapter.py`
- [ ] Update `core.least_squares` to use adapters
- [ ] **Deliverable:** Adapter pattern implemented

### Phase 5: Strategy Pattern (Weeks 5-6)

- [ ] Create `streaming/strategies/batch_strategy.py`
- [ ] Create `streaming/strategies/online_strategy.py`
- [ ] Create `streaming/strategies/distributed_strategy.py` (placeholder)
- [ ] Refactor `adaptive_hybrid.py`
- [ ] **Deliverable:** Streaming strategies pluggable

### Phase 6: Testing (Weeks 7-8)

- [ ] Unit tests (90%+ coverage)
- [ ] Integration tests (all workflows)
- [ ] Performance benchmarks (<5% overhead)
- [ ] Update developer guide
- [ ] **Deliverable:** Tested, documented

### Phase 7: Review & Merge (Week 9)

- [ ] Code review
- [ ] Performance validation
- [ ] Backward compatibility check
- [ ] Merge to main
- [ ] **Deliverable:** Production-ready

---

## Useful Commands

### Dependency Analysis

```bash
# Run custom dependency analyzer
python scripts/analyze_dependencies.py --check-cycles
python scripts/analyze_dependencies.py --coupling-report

# Check imports with pytest
python -m pytest --collect-only --import-mode=importlib 2>&1 | grep -i circular
```

### Generate Diagrams

```bash
# Generate PNG images (requires graphviz)
dot -Tpng architecture_dependency_graph.dot -o architecture_dependency_graph.png
dot -Tpng architecture_data_flow.dot -o architecture_data_flow.png
dot -Tpng architecture_streaming_flow.dot -o architecture_streaming_flow.png
```

### Run Tests

```bash
# Full test suite
pytest tests/ -v

# Coverage report
pytest --cov=nlsq --cov-report=html

# Performance benchmarks
python benchmarks/curve_fit_benchmark.py --baseline main
```

---

## Contact & Contribution

### Questions?

For questions about this analysis:
1. Review the relevant document (see navigation above)
2. Check the ARCHITECTURE_QUICK_REFERENCE for common patterns
3. Consult the ARCHITECTURE_REFACTORED_PROPOSAL for implementation details

### Contributing to Refactoring

1. Choose a phase from the migration checklist
2. Create a GitHub issue for the task
3. Create a feature branch (`refactor/phase1-protocols`)
4. Implement changes following the proposal
5. Run tests and benchmarks
6. Submit PR with detailed description

---

*Generated: 2025-12-29*
*Last Updated: 2025-12-29*
