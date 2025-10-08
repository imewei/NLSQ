# NLSQ Documentation TODO List

**Generated**: 2025-10-08
**Based on**: [CODE_STRUCTURE_ANALYSIS.md](./CODE_STRUCTURE_ANALYSIS.md)

This is an actionable checklist for completing NLSQ documentation.

## Critical (Do First)

### 1. Add Missing Module Docstrings (2 modules)

- [ ] **`nlsq/minpack.py`**
  - Main public API module
  - Exports: `CurveFit`, `curve_fit`
  - Add comprehensive module-level docstring explaining this is the SciPy-compatible interface

- [ ] **`nlsq/loss_functions.py`**
  - Already in Sphinx but missing module docstring
  - Add docstring explaining available loss functions and their use

### 2. Add Missing Class Docstring (1 class)

- [ ] **`OptimizeWarning` in `nlsq/_optimize.py`**
  - Simple UserWarning subclass
  - Add brief docstring explaining when it's raised

### 3. Fix Printing Functions (4 functions)

These are internal utilities but should have minimal docstrings:

- [ ] `print_header_nonlinear()` in `nlsq/common_scipy.py`
- [ ] `print_iteration_nonlinear()` in `nlsq/common_scipy.py`
- [ ] `print_header_linear()` in `nlsq/common_scipy.py`
- [ ] `print_iteration_linear()` in `nlsq/common_scipy.py`

**Time estimate**: 2-4 hours total

---

## High Priority (Week 1-2)

### Add Sphinx Documentation for High-Priority Modules (7 modules)

These modules have explicit `__all__` exports and are user-facing features.

#### 1. Large Dataset Support

- [ ] **`nlsq.large_dataset`** (8 exports)
  - Create `docs/api/nlsq.large_dataset.rst`
  - Document: `LargeDatasetFitter`, `fit_large_dataset`, `estimate_memory_requirements`
  - Add examples for >10M point datasets
  - Cross-reference with large dataset guide

#### 2. Common Fit Functions

- [ ] **`nlsq.functions`** (7 exports)
  - Create `docs/api/nlsq.functions.rst`
  - Document: `gaussian`, `exponential_decay`, `exponential_growth`, `sigmoid`, etc.
  - Add examples for each function
  - Show parameter estimation features

#### 3. Numerical Stability

- [ ] **`nlsq.stability`** (7 exports)
  - Create `docs/api/nlsq.stability.rst`
  - Document: `NumericalStabilityGuard`, `check_problem_stability`, etc.
  - Add troubleshooting guide
  - Cross-reference with stability guide

#### 4. Progress Callbacks

- [ ] **`nlsq.callbacks`** (6 exports)
  - Create `docs/api/nlsq.callbacks.rst`
  - Document: `ProgressBar`, `EarlyStopping`, `IterationLogger`, etc.
  - Add usage examples
  - Show custom callback creation

#### 5. Performance Profiling

- [ ] **`nlsq.profiler`** (4 exports)
  - Create `docs/api/nlsq.profiler.rst`
  - Document: `PerformanceProfiler`, `get_global_profiler`, etc.
  - Add profiling examples
  - Cross-reference with performance guide

- [ ] **`nlsq.profiler_visualization`** (2 exports)
  - Create `docs/api/nlsq.profiler_visualization.rst`
  - Document: `ProfilerVisualization`, `ProfilingDashboard`
  - Add visualization examples

#### 6. Bounds Inference

- [ ] **`nlsq.bound_inference`** (3 exports)
  - Create `docs/api/nlsq.bound_inference.rst`
  - Document: `BoundsInference`, `infer_bounds`, `merge_bounds`
  - Add automatic bounds examples

#### 7. Fallback Strategies

- [ ] **`nlsq.fallback`** (3 exports)
  - Create `docs/api/nlsq.fallback.rst`
  - Document: `FallbackOrchestrator`, `FallbackStrategy`, etc.
  - Add robustness examples

**Time estimate**: 14-21 hours (2-3 hours per module)

---

## Medium Priority (Week 3-4)

### Add Sphinx Documentation for Infrastructure Modules (17 modules)

These modules provide infrastructure and advanced features.

#### Algorithm & Optimization

- [ ] **`nlsq.algorithm_selector`**
  - Create `docs/api/nlsq.algorithm_selector.rst`
  - Document automatic algorithm selection

- [ ] **`nlsq.validators`**
  - Create `docs/api/nlsq.validators.rst`
  - Document input validation system

- [ ] **`nlsq.parameter_estimation`**
  - Create `docs/api/nlsq.parameter_estimation.rst`
  - Document automatic p0 estimation

#### Memory Management

- [ ] **`nlsq.memory_manager`**
  - Create `docs/api/nlsq.memory_manager.rst`
  - Document memory management features

- [ ] **`nlsq.memory_pool`**
  - Create `docs/api/nlsq.memory_pool.rst`
  - Document memory pooling

#### Caching & Compilation

- [ ] **`nlsq.smart_cache`**
  - Create `docs/api/nlsq.smart_cache.rst`
  - Document intelligent caching

- [ ] **`nlsq.compilation_cache`**
  - Create `docs/api/nlsq.compilation_cache.rst`
  - Document JIT compilation caching

#### Advanced Optimization

- [ ] **`nlsq.sparse_jacobian`**
  - Create `docs/api/nlsq.sparse_jacobian.rst`
  - Document sparse matrix support

- [ ] **`nlsq.streaming_optimizer`**
  - Create `docs/api/nlsq.streaming_optimizer.rst`
  - Document streaming optimization

- [ ] **`nlsq.recovery`**
  - Create `docs/api/nlsq.recovery.rst`
  - Document recovery strategies

#### Diagnostics & Monitoring

- [ ] **`nlsq.diagnostics`**
  - Create `docs/api/nlsq.diagnostics.rst`
  - Document convergence monitoring

- [ ] **`nlsq.error_messages`**
  - Create `docs/api/nlsq.error_messages.rst`
  - Document error handling

#### Results & Robustness

- [ ] **`nlsq.result`**
  - Create `docs/api/nlsq.result.rst`
  - Document enhanced result objects

- [ ] **`nlsq.robust_decomposition`**
  - Create `docs/api/nlsq.robust_decomposition.rst`
  - Document robust matrix decomposition

- [ ] **`nlsq.svd_fallback`**
  - Create `docs/api/nlsq.svd_fallback.rst`
  - Document SVD fallback mechanisms

#### Constants

- [ ] **`nlsq.constants`**
  - Create `docs/api/nlsq.constants.rst`
  - Document all optimization constants
  - Add configuration guide

**Time estimate**: 17-34 hours (1-2 hours per module)

---

## Low Priority (Week 5+)

### 1. Add `__all__` to Modules Without It (22 modules)

For each module, identify public API and add:

```python
__all__ = [
    'PublicClass1',
    'PublicClass2',
    'public_function1',
    'public_function2',
]
```

**Modules needing `__all__`**:
- `_optimize.py` (exports: `OptimizeResult`, `OptimizeWarning`)
- `algorithm_selector.py` (exports: `AlgorithmSelector`, `auto_select_algorithm`)
- `caching.py` (exports: `FunctionCache`, `cached_jit`, etc.)
- `common_jax.py` (exports: `CommonJIT`)
- `common_scipy.py` (exports: all public functions)
- `compilation_cache.py` (exports: `CompilationCache`, etc.)
- `config.py` (exports: `MemoryConfig`, `LargeDatasetConfig`, etc.)
- `diagnostics.py` (exports: `ConvergenceMonitor`, `OptimizationDiagnostics`)
- `error_messages.py` (exports: error classes and functions)
- `least_squares.py` (exports: `LeastSquares`, utility functions)
- `logging.py` (exports: `NLSQLogger`, `get_logger`, etc.)
- `loss_functions.py` (exports: `LossFunctionsJIT`)
- `memory_manager.py` (exports: `MemoryManager`, functions)
- `memory_pool.py` (exports: `MemoryPool`, `TRFMemoryPool`, functions)
- `optimizer_base.py` (exports: `OptimizerBase`, `TrustRegionOptimizerBase`)
- `parameter_estimation.py` (exports: all functions)
- `recovery.py` (exports: `OptimizationRecovery`)
- `result.py` (exports: `CurveFitResult`)
- `robust_decomposition.py` (exports: `RobustDecomposition`)
- `sparse_jacobian.py` (exports: `SparseJacobianComputer`, etc.)
- `streaming_optimizer.py` (exports: `StreamingOptimizer`, etc.)
- `svd_fallback.py` (exports: all functions)
- `trf.py` (exports: `TrustRegionReflective`)
- `validators.py` (exports: `InputValidator`, decorators)

**Time estimate**: 5-10 hours

### 2. Improve Type Hints (32 functions)

Add type hints to public functions that lack them. Focus on:
- Parameter types
- Return types
- Use `typing` module for complex types

**Current**: 66/98 functions (67%)
**Target**: 80+/98 functions (82%+)

**Time estimate**: 8-16 hours

### 3. Add Examples to Complex Functions

Add usage examples to docstrings for:
- Complex optimization functions
- Advanced features
- Configuration functions

**Time estimate**: 10-15 hours

---

## Summary

### Total Estimated Time

| Priority | Tasks | Time Estimate |
|----------|-------|---------------|
| Critical | 8 items | 2-4 hours |
| High | 7 modules | 14-21 hours |
| Medium | 17 modules | 17-34 hours |
| Low | 3 tasks | 23-41 hours |
| **Total** | **35+ tasks** | **56-100 hours** |

### Recommended Schedule

**Week 1**: Critical fixes (2-4 hours)
- Add missing module docstrings
- Fix class and function docstrings

**Weeks 2-3**: High-priority Sphinx docs (14-21 hours)
- Large dataset, functions, stability, callbacks
- Profiler, bounds inference, fallback

**Weeks 4-5**: Medium-priority Sphinx docs (17-34 hours)
- Infrastructure modules
- Advanced features

**Weeks 6-7**: Low-priority improvements (23-41 hours)
- Add `__all__` exports
- Improve type hints
- Add examples

### Success Metrics

| Metric | Current | Target | Progress |
|--------|---------|--------|----------|
| Module docstrings | 89% | 95% | Need 2 more |
| Sphinx coverage | 29% | 70%+ | Need 24 modules |
| Type hints | 67% | 80% | Need 32 functions |
| `__all__` exports | 32% | 80% | Need 22 modules |

---

## Quick Wins (Do Today)

1. Add module docstring to `minpack.py` (15 min)
2. Add module docstring to `loss_functions.py` (10 min)
3. Add class docstring to `OptimizeWarning` (5 min)

**Total**: 30 minutes for 3 critical fixes!

---

## Resources

- **Full Analysis**: [CODE_STRUCTURE_ANALYSIS.md](./CODE_STRUCTURE_ANALYSIS.md)
- **Summary**: [CODE_ANALYSIS_SUMMARY.md](./CODE_ANALYSIS_SUMMARY.md)
- **Sphinx Guide**: https://www.sphinx-doc.org/
- **NumPy Docstring Style**: https://numpydoc.readthedocs.io/

---

**Last Updated**: 2025-10-08
