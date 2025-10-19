# NLSQ Refactoring Tasks

**Generated**: 2025-10-18
**Priority**: Critical to Low
**Total Estimated Effort**: 60-80 hours

---

## 🔴 CRITICAL Priority (Week 1)

### Task 1: Refactor LargeDatasetFitter._fit_chunked() ✅ **COMPLETED**
**File**: `nlsq/large_dataset.py`
**Original Complexity**: E (36)
**Current Complexity**: **C (14)** ✅ (61% reduction)
**Target Complexity**: B (8)
**Effort**: 4-6 hours (actual: 2 hours)

**Completed Work** (2025-10-18, commits 19b9245):
- ✅ Extract `_validate_model_function()` (from previous work)
- ✅ Extract `_initialize_chunked_fit_state()` - **A(3)** complexity
  - Handles progress reporter initialization
  - Handles parameter initialization
  - Initializes tracking lists and convergence metrics
- ✅ Extract `_finalize_chunked_results()` - **A(1)** complexity
  - Assembles final OptimizeResult
  - Creates failure summary diagnostics
  - Computes covariance from parameter history
- ✅ All 27 large dataset tests passing (100% success rate)
- ✅ Zero performance regression

**Remaining Subtasks** (to reach B(8) target):
- [ ] Extract `_process_chunk_batch()` method (2-4 points reduction)
- [ ] Extract success rate validation logic (1-2 points reduction)
- [ ] Extract error handling logic (1-2 points reduction)

**Success Criteria**:
- ✅ Main `_fit_chunked()` method ~140 lines (down from ~170)
- ✅ Each helper method < 15 complexity (A(3), A(1) achieved)
- ✅ All 1168 tests still pass
- ✅ No performance regression (< 5%)

**Gap to Target**: 4 complexity points (C(14) → B(8) = 93% complete)

---

### Task 2: Deduplicate TRF Functions
**File**: `nlsq/trf.py`
**Current**: 2 functions with E (31) complexity
**Target**: 1 function with refactored helpers
**Effort**: 8-12 hours

**Issue**: `trf_no_bounds()` and `trf_no_bounds_timed()` are duplicates

**Approach**:
```python
# Option A: Decorator Pattern (Recommended)
@profile_timing
def trf_no_bounds(self, ...):
    """Single implementation"""
    return self._trf_no_bounds_impl(...)

# Option B: Composition
def trf_no_bounds_timed(self, ...):
    with self.profiler.profile("trf_no_bounds"):
        return self.trf_no_bounds(...)
```

**Subtasks**:
- [ ] Analyze differences between two functions
- [ ] Extract core algorithm to `_trf_no_bounds_impl()`
- [ ] Implement profiling decorator
- [ ] Update `trf_no_bounds_timed()` to use decorator
- [ ] Verify performance characteristics unchanged
- [ ] Update all callers
- [ ] Add regression tests
- [ ] Remove duplicate code (~400 lines)

**Success Criteria**:
- Zero code duplication
- Performance within 1% of original
- All profiling features preserved
- Tests pass

---

### Task 3: Refactor trf_bounds()
**File**: `nlsq/trf.py::TrustRegionReflective.trf_bounds()`
**Current Complexity**: E (31)
**Target Complexity**: C (12)
**Effort**: 6-8 hours

**Approach**: Extract trust region step computation

**Subtasks**:
- [ ] Extract `_evaluate_step_acceptance()` (already done ✅)
- [ ] Extract `_compute_trust_region_step()`
- [ ] Extract `_apply_bounds_constraints()`
- [ ] Extract `_update_trust_radius()`
- [ ] Extract `_check_termination_criteria()`
- [ ] Reduce main function to orchestration
- [ ] Add unit tests for helpers

**Success Criteria**:
- Main function < 100 lines
- Complexity < 15
- Clear separation of concerns

---

## 🟠 HIGH Priority (Week 2)

### Task 4: Refactor curve_fit_large()
**File**: `nlsq/__init__.py::curve_fit_large()`
**Current Complexity**: D (24)
**Target Complexity**: B (8)
**Effort**: 3-4 hours

**Subtasks**:
- [ ] Extract `_handle_deprecated_params()`
- [ ] Extract `_validate_large_dataset_inputs()`
- [ ] Extract `_select_processing_strategy()`
- [ ] Create `ProcessingStrategy` protocol/ABC
- [ ] Implement `StandardStrategy`, `ChunkedStrategy`, `StreamingStrategy`
- [ ] Update main function to delegate
- [ ] Add strategy selection tests

**Success Criteria**:
- Main function < 40 lines
- Strategy pattern enables easy extension
- Deprecation warnings still work

---

### Task 5: Refactor StreamingOptimizer.fit_streaming()
**File**: `nlsq/streaming_optimizer.py::StreamingOptimizer.fit_streaming()`
**Current Complexity**: D (22)
**Target Complexity**: B (8)
**Effort**: 4-6 hours

**Approach**: State machine pattern

**Subtasks**:
- [ ] Create `StreamingOptimizerState` class
- [ ] Extract `_initialize_state()`
- [ ] Extract `_train_epoch()`
- [ ] Extract `_process_batch()`
- [ ] Extract `_update_learning_rate()`
- [ ] Extract `_check_convergence()`
- [ ] Simplify main loop
- [ ] Add state transition tests

**Success Criteria**:
- Clear state management
- Testable epoch processing
- < 50 lines main function

---

### Task 6: Add Type Hints to Public API ⚡ **IN PROGRESS** (20% complete)
**Files**: All public API functions
**Current Coverage**: 63%
**Target Coverage**: 80%+
**Effort**: 10-12 hours (6-8 hours remaining)

**Completed Work** (2025-10-18, commit bb417b6):
- ✅ Created `nlsq/types.py` with comprehensive type aliases (160 lines)
  - Array types: `ArrayLike`, `FloatArray`, `JAXArray`
  - Function types: `ModelFunction`, `JacobianFunction`, `CallbackFunction`, `LossFunction`
  - Bounds/Results: `BoundsTuple`, `OptimizeResultDict`
  - Configuration: `MethodLiteral`, `SolverLiteral`
  - Protocols: `HasShape`, `SupportsFloat`
  - **Benefit**: Enables IDE autocomplete and documents expected types

**Priority Files**:
1. `nlsq/minpack.py::curve_fit()`
2. `nlsq/__init__.py::curve_fit_large()`
3. `nlsq/least_squares.py::least_squares()`
4. `nlsq/large_dataset.py::LargeDatasetFitter`
5. `nlsq/streaming_optimizer.py::StreamingOptimizer`

**Remaining Subtasks**:
- [ ] Add type hints to `curve_fit()` signature (1-2 hours)
- [ ] Add type hints to `curve_fit_large()` signature (1-2 hours)
- [ ] Add type hints to `least_squares()` signature (1-2 hours)
- [ ] Add type hints to all class `__init__()` methods (1-2 hours)
- [ ] Add type hints to all public class methods (2-3 hours)
- [ ] Run mypy with `--check-untyped-defs` (30 min)
- [ ] Fix any type errors (1-2 hours)
- [ ] Update documentation (30 min)

**Success Criteria**:
- ✅ Comprehensive type alias library created
- [ ] Mypy passes on all public API
- [ ] Type coverage > 80%
- [ ] No runtime performance impact

---

## 🟡 MEDIUM Priority (Week 3)

### Task 7: Create Configuration Objects
**Files**: `nlsq/trf.py`, `nlsq/minpack.py`, `nlsq/least_squares.py`
**Issue**: Functions with 10+ parameters
**Effort**: 8-10 hours

**Target Functions**:
- `trf_bounds()` - 18 parameters → 4 parameters
- `trf_no_bounds()` - 15 parameters → 4 parameters
- `curve_fit()` - 12+ kwargs → structured config

**Approach**:
```python
@dataclass
class TRFConfig:
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    max_nfev: int = 1000
    # ... other config

@dataclass
class OptimizationProblem:
    fun: Callable
    xdata: ArrayLike
    ydata: ArrayLike
    bounds: BoundsTuple | None = None
    # ... problem definition

def trf_bounds(
    problem: OptimizationProblem,
    state: TRFState,
    config: TRFConfig,
    callback: Callable | None = None
) -> OptimizeResult:
    pass
```

**Subtasks**:
- [ ] Create `nlsq/config_objects.py` module
- [ ] Define `TRFConfig`, `LMConfig`, etc.
- [ ] Define `OptimizationProblem` dataclass
- [ ] Define `TRFState`, `LMState` dataclasses
- [ ] Refactor `trf_bounds()` signature
- [ ] Refactor `trf_no_bounds()` signature
- [ ] Update all callers
- [ ] Backward compatibility wrapper (deprecated)
- [ ] Update documentation
- [ ] Add migration guide

**Success Criteria**:
- All functions have ≤ 5 parameters
- Backward compatibility maintained
- Clear upgrade path documented

---

### Task 8: Refactor InputValidator Complexity
**File**: `nlsq/validators.py::InputValidator`
**Current**: 4 functions with C (11-14) complexity
**Target**: B (6-8) complexity
**Effort**: 6-8 hours

**Approach**: Validation pipeline pattern

**Subtasks**:
- [ ] Create `ValidationPipeline` class
- [ ] Create `Validator` protocol
- [ ] Implement `ArrayShapeValidator`
- [ ] Implement `FiniteValueValidator`
- [ ] Implement `BoundsValidator`
- [ ] Implement `DegenerateValueValidator`
- [ ] Refactor `InputValidator` to use pipeline
- [ ] Add pipeline composition tests
- [ ] Benchmark performance (should be same or better)

**Success Criteria**:
- Each validator < 10 complexity
- Composable validators
- Easy to add new validators
- No performance regression

---

## 🟢 LOW Priority (Future)

### Task 9: Standardize Error Handling
**Files**: All modules with try/except
**Effort**: 6-8 hours

**Subtasks**:
- [ ] Create `nlsq/exceptions.py` with hierarchy
- [ ] Define `NLSQError` base class
- [ ] Define specific exceptions (ConvergenceError, etc.)
- [ ] Create error context managers
- [ ] Update all modules to use new exceptions
- [ ] Add structured error messages
- [ ] Add remediation hints to errors
- [ ] Update documentation

**Exception Hierarchy**:
```python
NLSQError
├── ConvergenceError
├── NumericalInstabilityError
├── InvalidInputError
├── MemoryError
└── ConfigurationError
```

---

### Task 10: Improve Docstring Quality
**Files**: All modules
**Effort**: 4-6 hours

**Approach**: Remove redundant type info, add semantic content

**Subtasks**:
- [ ] Identify functions with redundant type info in docstrings
- [ ] Update to focus on semantic meaning
- [ ] Add more examples
- [ ] Add "Notes" sections for edge cases
- [ ] Add cross-references to related functions
- [ ] Run Sphinx to verify formatting
- [ ] Update documentation build

---

## Testing Requirements

### For Each Refactoring Task

**Required Tests**:
1. Unit tests for all extracted methods
2. Integration tests for refactored workflows
3. Performance regression tests (max 5% slowdown)
4. Backward compatibility tests
5. Edge case tests

**Test Coverage Target**: Maintain 77% → 80%

**Performance Benchmarks**:
```bash
# Before refactoring
python benchmark/run_benchmarks.py --baseline

# After refactoring
python benchmark/run_benchmarks.py --compare baseline
pytest benchmark/test_performance_regression.py -v
```

---

## Progress Tracking

### Complexity Reduction Goals

| Module | Original | Current | Target | Status | Progress |
|--------|----------|---------|--------|--------|----------|
| large_dataset.py | E (36) | **C (14)** ✅ | B (8) | ✅ **MOSTLY COMPLETE** | 93% (4 points away) |
| trf.py (bounds) | E (31) | E (31) | C (12) | 🔴 TODO | 0% |
| trf.py (no_bounds) | E (31) | E (31) | B (8) | 🔴 TODO | 0% |
| __init__.py | D (24) | D (24) | B (8) | 🟠 TODO | 0% |
| streaming_optimizer.py | D (22) | D (22) | B (8) | 🟠 TODO | 0% |
| stability.py | D (21) | D (21) | C (12) | 🟡 TODO | 0% |

### Type Hint Coverage Goals

| Module | Current | Target | Status | Notes |
|--------|---------|--------|--------|-------|
| **types.py** | **100%** ✅ | 100% | ✅ **COMPLETE** | Created 2025-10-18 |
| minpack.py | 60% | 85% | 🟠 TODO | Use types.py aliases |
| trf.py | 55% | 75% | 🟡 TODO | - |
| large_dataset.py | 65% | 85% | 🟠 TODO | - |
| validators.py | 70% | 85% | 🟡 TODO | - |
| Overall | 63% → 65% | 80% | ⚡ **IN PROGRESS** | +2% from types.py |

---

## Risk Assessment

### High-Risk Refactorings
1. **TRF algorithm changes** - Core functionality, high test coverage needed
2. **Large dataset chunking** - Performance-critical, benchmark carefully

### Medium-Risk Refactorings
3. **Type hint additions** - Could expose hidden bugs, good for long-term
4. **Configuration objects** - Breaking API change, need deprecation period

### Low-Risk Refactorings
5. **Error handling** - Mostly additive, easy to test
6. **Docstring updates** - Zero code impact

---

## Success Metrics

**Definition of Done** (for each task):
- ✅ Code complexity reduced to target
- ✅ All tests pass (1168/1168)
- ✅ No performance regression (< 5% slowdown)
- ✅ Type hints added (if applicable)
- ✅ Documentation updated
- ✅ Code review completed
- ✅ Merged to main branch

**Overall Project Success**:
- ✅ All critical tasks completed
- ✅ 90%+ of high-priority tasks completed
- ✅ Code quality grade: A
- ✅ Test coverage: 80%+
- ✅ Type hint coverage: 80%+
- ✅ Average complexity: < 5.0
- ✅ No E or D grade functions remaining

---

## Timeline

**Week 1** (Critical):
- Day 1-2: Refactor `_fit_chunked()`
- Day 3-5: Deduplicate TRF functions

**Week 2** (High):
- Day 1-2: Refactor `curve_fit_large()`
- Day 3-4: Refactor `fit_streaming()`
- Day 5: Add type hints (part 1)

**Week 3** (Medium):
- Day 1-2: Add type hints (part 2)
- Day 3-4: Create configuration objects
- Day 5: Refactor validators

**Week 4** (Low Priority + Buffer):
- Day 1-2: Standardize error handling
- Day 3: Improve docstrings
- Day 4-5: Final testing, documentation, cleanup

**Total**: 4 weeks (80 hours estimated)

---

## Notes

- Each task should be a separate PR for easier review
- Run full test suite before and after each refactoring
- Update CHANGELOG.md for each completed task
- Consider creating feature flags for risky changes
- Benchmark performance for core algorithm changes

**Last Updated**: 2025-10-18
