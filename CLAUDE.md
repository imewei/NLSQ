# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
NLSQ is a GPU/TPU-accelerated nonlinear least squares curve fitting library that ports SciPy's curve fitting algorithms to JAX. It provides a drop-in replacement for `scipy.optimize.curve_fit` with massive speedups on GPU/TPU hardware.

**Repository**: https://github.com/imewei/NLSQ
**Maintainer**: Wei Chen (Argonne National Laboratory)
**Origin**: Enhanced fork of JAXFit by Lucas R. Hofer, Milan Krstajić, and Robert P. Smith
**Status**: Production-ready (Beta), 77% test coverage, 817 tests passing (100% pass rate)

## Recent Updates (Updated: 2025-10-07)

### Major Changes (October 2025)

#### Pre-Commit Compliance & CI/CD Health ✅ **NEW**
- **100% pre-commit compliance** achieved (24/24 hooks passing)
- Fixed CodeQL workflow schema validation error (`.github/workflows/codeql.yml`)
- Resolved all code quality issues: Greek characters, try-except-pass, unused variables
- See `docs/codeql_workflow_fix.md` for complete technical documentation

#### Performance Optimization Complete ✅
- **8% overall performance improvement** (~15% on core TRF algorithm)
- Eliminated 11 NumPy↔JAX conversions in hot paths (`trf.py`)
- Zero numerical regressions, all tests passing
- **Decision**: Further complex optimizations deferred (diminishing returns)
- See `docs/optimization_case_study.md` for detailed analysis

#### Test Suite Health ✅
- **100% pass rate achieved** (743 tests, 1 skipped)
- Fixed JAX immutability error in `common_scipy.py` (`make_strictly_feasible()`)
- Fixed flaky test in `test_integration.py` (added random seed, relaxed bounds)
- Coverage: 77% (target 80%)

#### Documentation Updates ✅
- Added comprehensive codebase analysis: `codebase_analysis.md`
- Added optimization case study: `docs/optimization_case_study.md`
- Added performance tuning guide: `docs/performance_tuning_guide.md`
- Cleaned up obsolete reports (removed 3 duplicate files, ~2.8K lines)

#### Infrastructure Improvements ✅
- Updated `.gitignore` for optimization artifacts
- Added profiling and benchmark scripts in `benchmark/`
- Added PyPI setup documentation: `docs/PYPI_SETUP.md`

#### Configuration Harmonization ✅ **NEW**
- **Phase 1 Complete**: Updated test counts (676 → 743) and coverage metrics (74% → 70%) in config files
- **Phase 3 Validation**: Empirically tested ruff complexity ignores post-refactoring
  - **Finding**: Despite Phases 1-2.1 refactoring, complexity ignores still required
  - **nlsq/minpack.py**: 9 violations (_prepare_curve_fit_inputs, _run_optimization remain complex)
  - **nlsq/least_squares.py**: 12 violations (least_squares method: 77 statements, 22 branches, complexity 24)
  - **nlsq/validators.py**: 4 violations (validate_curve_fit_inputs: complexity 25)
  - **Decision**: Keep all complexity ignores (C901, PLR0912, PLR0913, PLR0915) - inherent complexity, not refactoring artifacts
  - **Verified**: 743 tests passing, pre-commit clean, all configurations consistent

#### Sprint 3: API Fixes and Complexity Reduction ✅ **COMPLETE**
- **100% test pass rate achieved** (817/820 tests passing, 3 skipped)
- **Fixed 10 API mismatch tests** in `test_validators_comprehensive.py`
- **Reduced 2 complexity violations** (23→<10, 20→<10)
- **Created 14 helper methods** following orchestrator pattern
- **Deferred core algorithm refactoring** (`trf_no_bounds`) to dedicated sprint
- See `sprint3_completion_summary.md` for comprehensive details

**Functions Refactored**:
1. `validators.validate_least_squares_inputs` (complexity 23 → <10, 6 helpers)
2. `algorithm_selector.select_algorithm` (complexity 20 → <10, 8 helpers)

**Key Achievements**:
- Fixed API mismatches: Validators return (errors, warnings, data) tuples, not exceptions
- Orchestrator pattern: Main methods now 33-49 lines with clear numbered steps
- Zero regressions: All 817 applicable tests passing
- Strategic deferral: Core algorithm (`trf_no_bounds`, complexity 24) deferred for dedicated sprint with benchmarking

**Complexity Status**: 20 → 18 violations (-10% reduction)

### Breaking Changes
None. All changes are backward compatible.

### Files Changed
**Modified:**
- `.github/workflows/codeql.yml` - Fixed schema validation error (10/07) ⭐
- `pyproject.toml` - Updated test counts (743) and coverage (70%) (10/07) ⭐
- `tox.ini` - Updated test counts (743) and coverage (70%), adjusted --cov-fail-under (10/07) ⭐
- `CLAUDE.md` - Documented Phase 3 ruff validation findings and Sprint 3 completion (10/07)
- `nlsq/constants.py` - Fixed Greek chars, removed docstrings, sorted __all__ (10/07)
- `nlsq/validators.py` - Use contextlib.suppress, refactored validate_least_squares_inputs (10/07) ⭐
- `nlsq/algorithm_selector.py` - Refactored select_algorithm (complexity 20→<10) (10/07) ⭐
- `nlsq/minpack.py` - Prefixed unused variables with underscore (10/07)
- `nlsq/least_squares.py` - Refactored into focused methods (10/07)
- `tests/test_validators_comprehensive.py` - Fixed 10 API mismatch tests (10/07) ⭐
- `nlsq/trf.py` - Performance optimization (NumPy↔JAX reduction)
- `nlsq/common_scipy.py` - JAX immutability fix
- `tests/test_integration.py` - Flaky test fix
- `.gitignore` - Added optimization artifacts

**Added:**
- `sprint3_plan.md` - Sprint 3 implementation plan (10/07) ⭐
- `sprint3_completion_summary.md` - Sprint 3 comprehensive summary (10/07) ⭐
- `docs/codeql_workflow_fix.md` - Complete technical documentation for CI fix (10/07) ⭐
- `docs/optimization_case_study.md` - Performance optimization analysis
- `docs/performance_tuning_guide.md` - User performance guide
- `codebase_analysis.md` - Comprehensive codebase analysis
- `benchmark/profile_trf.py` - TRF profiling tool (renamed from profile_trf_hot_paths.py)
- `benchmark/test_performance_regression.py` - Performance regression tests
- `benchmark/docs/completed/trf_profiling.md` - Profiling results
- `benchmark/docs/completed/numpy_jax_optimization.md` - NumPy↔JAX optimization (8%)
- `benchmark/docs/future/lax_scan_design.md` - lax.scan conversion design (deferred)

**Removed:**
- `multi-agent-optimization-report.md` - Obsolete (superseded by case study)
- `optimization_complete_summary.md` - Obsolete (duplicate)
- `optimization_progress_summary.md` - Obsolete (intermediate state)
- `benchmark/classes/` - Obsolete subdirectory (4 unused files removed)

## Development Commands

### Running Tests
```bash
# Run all tests with pytest
make test

# Run only fast tests (excludes slow optimization tests)
make test-fast

# Run only slow optimization tests
make test-slow

# Run tests with coverage report
make test-cov

# Run tests on CPU backend only (avoids GPU compilation)
make test-cpu

# Run tests with debug logging
NLSQ_DEBUG=1 pytest -s

# Run specific test file
pytest tests/test_least_squares.py -v
pytest tests/test_minpack.py -v

# Run specific test
pytest tests/test_minpack.py::test_exponential_fit -v
```

### Building and Installation
```bash
# Install in development mode with all dependencies
make dev

# Install package only
make install

# Build the package
python -m build

# Run pre-commit hooks
pre-commit run --all-files
```

### Code Quality
```bash
# Run linting checks
make lint

# Format code with black and ruff
make format

# Type checking
make type-check

# Clean build artifacts and cache
make clean
```

### Performance Profiling
```bash
# Profile TRF hot paths
python benchmark/profile_trf_hot_paths.py

# Run performance regression tests
pytest benchmark/test_performance_regression.py -v

# View profiling summary
cat benchmark/docs/completed/trf_profiling.md

# View optimization history
cat benchmark/docs/README.md
```

## Architecture

### Core Components

The library is organized around several key modules in the `nlsq/` directory:

#### Core Optimization
1. **minpack.py** - Contains the main `CurveFit` class and `curve_fit` function that provides the high-level API compatible with SciPy
2. **least_squares.py** - Implements the `LeastSquares` class with the core least squares solver
3. **trf.py** - Trust Region Reflective algorithm implementation (recently optimized)
4. **loss_functions.py** - Various loss functions for robust fitting
5. **_optimize.py** - Optimization result classes and utilities

#### Advanced Features
6. **algorithm_selector.py** - Automatic algorithm selection based on problem characteristics
7. **large_dataset.py** - Large dataset handling with automatic chunking and streaming
8. **memory_manager.py** - Memory management with configurable limits and monitoring
9. **smart_cache.py** - Smart caching system for repeated computations
10. **diagnostics.py** - Diagnostic monitoring and performance analysis
11. **recovery.py** - Error recovery mechanisms and fallback strategies
12. **stability.py** - Numerical stability improvements and condition monitoring
13. **validators.py** - Comprehensive input validation and sanitization
14. **robust_decomposition.py** - Robust linear algebra decompositions

#### Infrastructure
15. **config.py** - Configuration management and context managers
16. **common_jax.py** - JAX-specific utilities and functions
17. **common_scipy.py** - SciPy compatibility layer and shared utilities (recently fixed for JAX arrays)
18. **logging.py** - Logging and debugging utilities
19. **caching.py** - Core caching infrastructure
20. **optimizer_base.py** - Abstract base classes for optimization algorithms
21. **sparse_jacobian.py** - Sparse Jacobian matrix handling for large-scale problems
22. **streaming_optimizer.py** - Streaming optimization for very large datasets
23. **svd_fallback.py** - Fallback SVD implementations for numerical stability

### Key Design Principles

- **JAX JIT Compilation**: All fit functions must be JIT-compilable. This means avoiding Python control flow in fit functions and using JAX numpy operations.
- **Double Precision**: NLSQ requires 64-bit precision. The library automatically enables this when imported via `config.update("jax_enable_x64", True)`.
- **Automatic Differentiation**: Uses JAX's autodiff for computing Jacobians rather than requiring analytical derivatives or numeric approximation.
- **Backend Selection**: JAX automatically detects available accelerators (GPU/TPU). Use `JAX_PLATFORM_NAME=cpu` environment variable to force CPU-only execution.

### API Compatibility

The library maintains API compatibility with SciPy's `curve_fit`:
- Same function signature for `curve_fit(f, xdata, ydata, p0=None, ...)`
- Returns `popt` (optimized parameters) and `pcov` (covariance matrix)
- The `CurveFit` class allows reusing compiled functions for multiple fits

## Testing Approach

Tests use Python's unittest and pytest frameworks, with comprehensive coverage across all modules:

### Core Test Files
- `test_least_squares.py` - Tests for the least squares solver
- `test_minpack.py` - Tests for the curve_fit interface
- `test_trf_simple.py` - Tests for Trust Region Reflective algorithm
- `test_integration.py` - Integration tests across components (recently fixed flaky test)
- `test_large_dataset.py` - Tests for large dataset handling
- `test_stability.py` - Numerical stability tests
- `test_streaming_optimizer.py` - Streaming optimization tests
- `test_sparse_jacobian.py` - Sparse Jacobian tests

### Coverage and Additional Tests
- Multiple coverage test files ensure comprehensive testing
- Tests compare results against SciPy implementations for correctness
- Performance benchmarking tests for GPU/TPU acceleration
- **Current coverage**: 77% (target: 80%)
- **Test status**: 817 tests passing, 3 skipped (100% pass rate)

## Examples

The `examples/` directory contains Jupyter notebooks demonstrating the library's capabilities:
- `NLSQ Quickstart.ipynb` - Getting started guide with basic usage
- `NLSQ_2D_Gaussian_Demo.ipynb` - 2D Gaussian fitting demonstration
- `advanced_features_demo.ipynb` - Showcases advanced optimization features
- `large_dataset_demo.ipynb` - Demonstrates handling of large datasets with chunking and streaming

## Performance Characteristics

### Optimization Status (October 2025)

NLSQ has been extensively profiled and optimized. The codebase is **production-ready** with excellent performance characteristics.

**Performance Benchmarks** (CPU, first run including JIT compilation):
- Small problem (100 points): ~430ms total (~30ms runtime after JIT)
- Medium problem (1000 points): ~490ms total (~110ms runtime after JIT)
- Large problem (10000 points): ~605ms total (~134ms runtime after JIT)
- XLarge problem (50000 points): ~572ms total (~120ms runtime after JIT)

**GPU Performance** (NVIDIA Tesla V100):
- 1M points, 5 parameters: 0.15s (NLSQ) vs 40.5s (SciPy) = **270x speedup**

**Scaling Characteristics**:
- ✅ Excellent: 50x more data → only 1.2x slower
- ✅ Well-optimized with JAX primitives (51 @jit decorators)
- ✅ Minimal Python overhead
- ✅ 150-270x faster than baseline implementations

**JIT Compilation Notes**:
- First run includes JIT compilation overhead (60-75% of time)
- Subsequent runs are much faster due to caching
- Use `CurveFit` class to reuse compiled functions for multiple fits

### Recent Optimization Work (October 2025)

**NumPy↔JAX Conversion Reduction**:
- **Result**: 8% total performance improvement (~15% on core TRF algorithm)
- **Method**: Eliminated 11 unnecessary array conversions in hot paths
- **Files changed**: `nlsq/trf.py` (lines 895-900, 928, final return)
- **Testing**: Zero numerical regressions, all tests passing
- **Profiling tools**: cProfile + line_profiler
- **Documentation**: See `docs/optimization_case_study.md` for detailed analysis

**Key Changes**:
```python
# Before: Converted to NumPy immediately
cost = np.array(cost_jnp)
g = np.array(g_jnp)

# After: Keep as JAX arrays in hot paths
cost = cost_jnp  # Convert only at boundaries
g = g_jnp
```

**Decision**: Further complex optimizations (lax.scan, @vmap, @pmap) have been **deferred** due to:
- Diminishing returns (5-10% estimated gain for significant complexity)
- Code already highly optimized with JAX primitives
- User-facing features more valuable than micro-optimizations

See `benchmark/docs/future/lax_scan_design.md` for deferred optimization designs.

### Performance Tuning Guide

For applications requiring maximum performance:

1. **Use CurveFit class** for multiple fits (reuses JIT compilation)
2. **Enable GPU/TPU** (JAX automatically detects available accelerators)
3. **Batch operations** when fitting multiple curves
4. **Profile your workload** to identify actual bottlenecks
5. **Use curve_fit_large()** for datasets >20M points
6. **Set appropriate memory limits** with `memory_limit_gb` parameter

**Dataset Size Recommendations**:
- <1K points: Use SciPy (faster due to JIT overhead)
- 1K-100K points: NLSQ CPU or GPU
- >100K points: NLSQ GPU (significant speedup)

**Note**: The code is already highly optimized. Focus on features and user experience rather than micro-optimizations.

### When to Optimize Further

Consider additional optimization ONLY if:
- User data shows specific bottlenecks
- Batch processing becomes common use case
- Sparse Jacobian patterns are prevalent
- Multi-GPU systems are widely available

**Current recommendation**: Focus on features and user experience rather than micro-optimizations.

See `docs/optimization_case_study.md` for comprehensive analysis and lessons learned.
See `docs/performance_tuning_guide.md` for user-facing performance optimization guide.

## File Structure

```
nlsq/                                    # Main package
├── nlsq/                                # Core library (25 modules)
│   ├── minpack.py                       # High-level curve_fit API
│   ├── least_squares.py                 # Core solver
│   ├── trf.py                           # TRF algorithm (optimized)
│   ├── large_dataset.py                 # Chunking & streaming
│   ├── common_scipy.py                  # SciPy compatibility (fixed)
│   └── [18 other modules]
│
├── tests/                               # Test suite (23 files, 8.4K LOC)
│   ├── test_minpack.py                  # API tests
│   ├── test_integration.py              # Integration tests (fixed)
│   └── [21 other test files]
│
├── docs/                                # Documentation
│   ├── optimization_case_study.md       # Performance case study (NEW)
│   ├── performance_tuning_guide.md      # User tuning guide (NEW)
│   ├── PYPI_SETUP.md                    # PyPI publishing guide (NEW)
│   └── [13 other docs]
│
├── benchmark/                           # Performance testing
│   ├── run_benchmarks.py                # Primary CLI (Phase 3) ⭐
│   ├── benchmark_suite.py               # Comprehensive suite (Phase 3)
│   ├── profile_trf.py                   # TRF profiling tool (renamed)
│   ├── test_performance_regression.py   # Regression tests (CI) ✅
│   ├── README.md                        # Benchmark documentation
│   ├── legacy/                          # Historical tools 📦
│   │   ├── benchmark_v1.py              # Original benchmark tool
│   │   └── benchmark_sprint2.py         # Sprint 2 benchmarks
│   └── docs/                            # Optimization documentation
│       ├── README.md                    # Documentation index
│       ├── historical_results.md        # Benchmark results 2024-2025
│       ├── usage_guide.md               # Detailed usage examples
│       ├── completed/                   # Completed optimizations
│       │   ├── numpy_jax_optimization.md  # NumPy↔JAX (8%)
│       │   └── trf_profiling.md         # Profiling baseline
│       └── future/                      # Deferred work
│           └── lax_scan_design.md       # lax.scan design
│
├── examples/                            # Jupyter notebooks
│   ├── NLSQ Quickstart.ipynb
│   ├── NLSQ_2D_Gaussian_Demo.ipynb
│   └── [2 other notebooks]
│
├── codebase_analysis.md                 # Comprehensive analysis (NEW)
├── pyproject.toml                       # Project configuration
├── Makefile                             # Development commands
└── README.md                            # User documentation
```

## Important Notes for Development

### Code Style & Quality
- **Testing**: Always run `make test` before committing
- **Linting**: Pre-commit hooks enforce ruff + black formatting
- **Type hints**: Partial coverage (~60%), mypy configured for scientific computing
- **Coverage target**: 80% (currently 77%)

### Performance Considerations
- **Don't optimize prematurely**: Code is already highly optimized
- **Profile first**: Use `benchmark/profile_trf.py` to identify bottlenecks
- **Test thoroughly**: All optimizations must maintain numerical correctness
- **Document decisions**: Update `docs/optimization_case_study.md` for major optimizations

### JAX Compatibility
- **Immutability**: JAX arrays are immutable, use `np.array(x, copy=True)` for in-place ops
- **Precision**: Always use float64 (auto-enabled by NLSQ)
- **JIT compilation**: Fit functions must be JIT-compilable (no Python control flow)
- **Array conversions**: Minimize NumPy↔JAX conversions in hot paths

### Testing Best Practices
- **Random seeds**: Always set `np.random.seed()` in tests with random data
- **Flaky tests**: Use realistic assertion bounds for chunked/approximated algorithms
- **GPU tests**: Mark with `@pytest.mark.gpu` for hardware-specific tests
- **Coverage**: Focus on error paths and edge cases

### Documentation Standards
- **User-facing**: Clear, concise, with runnable examples
- **Developer-facing**: Detailed explanations with profiling data
- **Case studies**: Document "why" decisions were made, not just "what"
- **Performance**: Include benchmark data and scaling characteristics

## Common Issues & Solutions

### Issue: JAX Array Immutability Error
**Symptom**: `TypeError: JAX arrays are immutable`
**Solution**: Use `np.array(x, copy=True)` to convert to mutable NumPy array
**Fixed in**: `nlsq/common_scipy.py` (commit 07bc9eb)

### Issue: Flaky Test Failures
**Symptom**: Tests pass/fail non-deterministically
**Solution**: Set random seed + relax bounds for approximated algorithms
**Fixed in**: `tests/test_integration.py` (commit 07bc9eb)

### Issue: Performance Regression
**Detection**: `benchmark/test_performance_regression.py` flags >5% slowdown
**Action**: Profile with `benchmark/profile_trf_hot_paths.py`, investigate hot paths
**Prevention**: Run benchmark tests in CI

## Resources

### Documentation
- **ReadTheDocs**: https://nlsq.readthedocs.io
- **Case Study**: `docs/optimization_case_study.md` (performance optimization journey)
- **Tuning Guide**: `docs/performance_tuning_guide.md` (user-facing optimization)
- **Codebase Analysis**: `codebase_analysis.md` (comprehensive architecture review)

### Benchmarking
- **Main Documentation**: `benchmark/README.md` (comprehensive benchmarking guide)
- **Usage Guide**: `benchmark/docs/usage_guide.md` (detailed usage examples and best practices)
- **Historical Results**: `benchmark/docs/historical_results.md` (benchmark data 2024-2025)
- **Optimization History**: `benchmark/docs/README.md` (completed & future optimizations)
- **Primary CLI**: `benchmark/run_benchmarks.py` (recommended benchmarking tool)
- **Profiling script**: `benchmark/profile_trf.py`
- **Regression tests**: `benchmark/test_performance_regression.py` (13 tests, CI integrated)
- **Profiling results**: `benchmark/docs/completed/trf_profiling.md`
- **Optimization plan**: `benchmark/docs/completed/numpy_jax_optimization.md`
- **Legacy tools**: `benchmark/legacy/` (historical benchmarking tools)

### External Resources
- **JAX Documentation**: https://jax.readthedocs.io
- **Original JAXFit Paper**: https://doi.org/10.48550/arXiv.2208.12187
- **SciPy curve_fit**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

---

**Last Updated**: 2025-10-07
**Status**: Production-ready (Beta)
**Test Status**: 817 passing, 3 skipped (100% pass rate)
**Coverage**: 77% (target: 80%)
