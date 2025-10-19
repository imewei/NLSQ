# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Reference

**Repository**: https://github.com/imewei/NLSQ
**Maintainer**: Wei Chen (Argonne National Laboratory)
**Status**: Production-ready (Beta) | **Python**: 3.12+ | **Tests**: 1235/1235 passing | **Coverage**: 80.90%

### Essential Commands
```bash
# Testing
make test              # Run all tests
make test-cov          # With coverage report
pytest -v tests/       # Verbose test output

# Code Quality
make format            # Format code (black + ruff)
make lint              # Run linters
pre-commit run --all-files

# Benchmarking
python benchmark/run_benchmarks.py --quick
pytest benchmark/test_performance_regression.py -v
```

---

## Overview

NLSQ is a **GPU/TPU-accelerated nonlinear least squares curve fitting library** that ports SciPy's `curve_fit` to JAX.

### Core Features
- 🚀 **Drop-in replacement** for `scipy.optimize.curve_fit`
- ⚡ **GPU/TPU acceleration** via JAX (150-270x speedup)
- 🔧 **JIT compilation** for performance
- 📊 **Large dataset support** (>1M points)
- 🎯 **NumPy 2.0+ compatible**

### Key Metrics (2025-10-19)
- **Performance**: 1.7-2.0ms (cached), 450-650ms (first run with JIT)
- **Test Suite**: 1235/1235 passing (100% success rate) ✅
- **Coverage**: 80.90% (**exceeds 80% target**) ✅
- **Platform Support**: Full Windows/macOS/Linux compatibility
- **CI/CD**: All platforms passing, 0 flaky tests

---

## Dependencies

### ⚠️ Important: NumPy 2.0+ Required

NLSQ requires **NumPy 2.0+** as of v0.1.1 (tested on 2.3.4). See [`REQUIREMENTS.md`](REQUIREMENTS.md) for:
- Complete dependency strategy
- Migration guide from NumPy 1.x
- Installation options and troubleshooting

### Core Requirements (Tested Versions)
```toml
numpy>=2.0.0      # Tested: 2.3.4
scipy>=1.14.0     # Tested: 1.16.2
jax>=0.6.0        # Tested: 0.8.0
jaxlib>=0.6.0     # Tested: 0.8.0
matplotlib>=3.9.0 # Tested: 3.10.7
```

### Installation
```bash
# Basic install
pip install nlsq

# With all features
pip install nlsq[all]

# Development environment (exact versions)
pip install -r requirements-dev.txt
```

See [`REQUIREMENTS.md`](REQUIREMENTS.md) for detailed dependency management strategy.

---

## Architecture

### Module Organization
```
nlsq/
├── Core API
│   ├── minpack.py           # Main curve_fit API (SciPy compatible)
│   ├── least_squares.py     # Optimization solver
│   └── trf.py               # Trust Region Reflective algorithm
├── Advanced Features
│   ├── algorithm_selector.py
│   ├── large_dataset.py
│   ├── memory_manager.py
│   └── validators.py
└── Infrastructure
    ├── config.py            # JAX configuration
    ├── common_jax.py        # JAX utilities
    ├── common_scipy.py      # SciPy compatibility
    └── loss_functions.py
```

### Design Principles

**1. JAX JIT Compilation**
- All fit functions must be JIT-compilable
- No Python control flow in hot paths
- Use JAX transformations (grad, vmap, etc.)

**2. Float64 Precision**
- Auto-enabled: `config.update("jax_enable_x64", True)`
- Critical for numerical accuracy

**3. SciPy Compatibility**
```python
# Same API as scipy.optimize.curve_fit
from nlsq import curve_fit

popt, pcov = curve_fit(f, xdata, ydata, p0=None, ...)

# For multiple fits, reuse JIT compilation
from nlsq import CurveFit

fitter = CurveFit(f)
popt1, pcov1 = fitter.fit(xdata1, ydata1)
popt2, pcov2 = fitter.fit(xdata2, ydata2)  # Reuses compiled function
```

---

## Performance Guide

### Benchmarks (Latest - 2025-10-08)

**CPU Performance:**
| Size | First Run (JIT) | Cached | SciPy | Speedup |
|------|----------------|--------|-------|---------|
| 100  | 450-520ms | 1.7-2.0ms | 10-16ms | 0.1x slower |
| 1K   | 520-570ms | 1.8-2.0ms | 8-60ms | Comparable |
| 10K  | 550-650ms | 1.8-2.0ms | 13-150ms | Faster |

**GPU Performance (NVIDIA V100):**
- 1M points: **0.15s** (NLSQ) vs 40.5s (SciPy) = **270x speedup**

### When to Use NLSQ vs SciPy

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| **< 1K points, CPU, one-off** | Use SciPy | JIT overhead not worth it |
| **> 1K points, CPU** | Use NLSQ | Comparable or faster |
| **Any size, GPU/TPU** | Use NLSQ | 150-270x faster |
| **Batch processing** | Use NLSQ + CurveFit | 60-80x faster (cached JIT) |

### Optimization Tips

1. **Reuse JIT compilation** with `CurveFit` class
2. **Enable GPU/TPU** (auto-detected by JAX)
3. **Profile before optimizing**: `python benchmark/profile_trf.py`
4. **Use `curve_fit_large()`** for datasets >20M points

**Note**: Code is already highly optimized. Further micro-optimizations deferred (diminishing returns).

---

## Development Guidelines

### Testing

**Framework**: pytest + unittest
**Coverage Target**: 80% (current: 77%)

```bash
# Run specific test
pytest tests/test_minpack.py::test_exponential_fit -v

# Fast tests only (exclude slow)
make test-fast

# With coverage
make test-cov
pytest --cov=nlsq --cov-report=html

# README examples validation (CI job)
pytest tests/test_readme_examples.py -v
```

**Best Practices:**
- ✅ Always set random seeds in tests with random data
- ✅ Use realistic tolerances for approximated algorithms
- ✅ Focus on error paths and edge cases
- ✅ Run `make test` before committing

### CI/CD Integration

**GitHub Actions Workflows** (`.github/workflows/`):

1. **Documentation Examples** (`readme-examples.yml`) - ✅ ENABLED
   - **Purpose**: Validate all README.md code examples
   - **Triggers**: Push to main, PRs, weekly schedule, manual
   - **Tests**: 12 examples via `tests/test_readme_examples.py`
   - **Duration**: ~1-2 minutes
   - **Status**: [![Examples Validated](https://img.shields.io/badge/examples-validated%202025--10--09-brightgreen?style=flat)](https://github.com/imewei/NLSQ/actions/workflows/readme-examples.yml)

2. **Main CI Workflow** (`ci.yml`) - ⏸️ DISABLED
   - **Status**: Moved to `.github/workflows.disabled/` (resource optimization)
   - **Re-enable**: Move back to `.github/workflows/` when ready
   - **Includes**: pre-commit, tests, coverage, docs build, package validation

**Documentation Validation**:
- Examples badge shows last validation date
- CI automatically tests all code examples in README
- Failures trigger PR comments with detailed results
- Manual timestamp update: Edit README badge after validation

### Code Quality

**Tools**: Black (25.x), Ruff (0.14.1), mypy (1.18.2), pre-commit (4.3.0)

```bash
# Format code
make format

# Run all linters
make lint

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

**Standards:**
- Type hints: ~60% coverage (pragmatic for scientific code)
- Complexity: Max cyclomatic complexity <10 (refactored from 23)
- Pre-commit: 24/24 hooks passing

### JAX Best Practices

**Immutability**:
```python
# ❌ Wrong - JAX arrays are immutable
x[0] = 1.0

# ✅ Correct - convert to mutable NumPy
x = np.array(x, copy=True)
x[0] = 1.0
```

**JIT Compilation**:
```python
# ✅ Good - static control flow
@jit
def f(x):
    return jnp.where(x > 0, x, 0)


# ❌ Bad - Python control flow breaks JIT
@jit
def f(x):
    if x > 0:  # Python if statement
        return x
    return 0
```

**Performance**:
- Minimize NumPy↔JAX conversions in hot paths
- Use JAX primitives (jnp.* instead of np.*)
- Profile before optimizing: `benchmark/profile_trf.py`

---

## Common Issues & Solutions

### 1. JAX Array Immutability
**Error**: `TypeError: JAX arrays are immutable`
**Fix**: `x = np.array(x, copy=True)` to convert to mutable NumPy array

### 2. NumPy Version Incompatibility
**Error**: Import errors or numerical issues
**Fix**: Upgrade to NumPy 2.x
```bash
pip install --upgrade "numpy>=2.0"
```
See [`REQUIREMENTS.md`](REQUIREMENTS.md) for migration guide.

### 3. Flaky Tests
**Error**: Non-deterministic pass/fail
**Fix**:
- Set random seed: `np.random.seed(42)`
- Relax tolerances for approximated algorithms
- Use `pytest --lf` to re-run last failures

### 4. Performance Regression
**Detection**: `pytest benchmark/test_performance_regression.py -v` (>5% slowdown alerts)
**Action**: Profile with `python benchmark/profile_trf.py`

### 5. JIT Compilation Timeout
**Error**: First run takes too long
**Fix**:
- Expected behavior (450-650ms first run)
- Use `CurveFit` class to cache compilation
- Consider `curve_fit_large()` for very large problems

### 6. Chunking Shape Mismatch (curve_fit_large)
**Error**: Model function shape mismatch during chunked processing
**Cause**: Model function returns fixed-size array instead of respecting xdata size
**Fix**: Make model function respect xdata size (see Large Dataset Features below)

---

## Large Dataset Features (v0.1.3+)

### Chunking-Compatible Model Functions

When using `curve_fit_large()` with datasets >1M points, the model function **must** respect the size of xdata:

**❌ INCORRECT - Returns fixed size:**
```python
def bad_model(xdata, a, b):
    # Always returns full array, ignoring xdata size
    t_full = jnp.arange(10_000_000)  # Fixed size!
    return a * jnp.exp(-b * t_full)  # Shape mismatch during chunking
```

**✅ CORRECT - Uses xdata as indices:**
```python
def good_model(xdata, a, b):
    # Uses xdata as indices to return only requested subset
    indices = xdata.astype(jnp.int32)
    y_full = a * jnp.exp(-b * jnp.arange(10_000_000))
    return y_full[indices]  # Shape matches xdata
```

**✅ CORRECT - Operates directly on xdata:**
```python
def direct_model(xdata, a, b):
    # Operates directly on xdata
    return a * jnp.exp(-b * xdata)  # Shape automatically matches
```

### Shape Validation

NLSQ automatically validates model functions before chunked processing:
- Tests with first 100 points to catch shape mismatches early
- Provides clear error messages with fix examples
- Prevents silent failures and invalid results
- Negligible overhead (~0.1s for multi-hour fits)

### Logger Integration

Connect NLSQ's internal logger to your application's logger for better diagnostics:

```python
import logging
from nlsq import LargeDatasetFitter

# Create application logger
app_logger = logging.getLogger("myapp")

# Use with NLSQ - chunk failures now appear in myapp's logs
fitter = LargeDatasetFitter(memory_limit_gb=8, logger=app_logger)
result = fitter.fit(model_func, xdata, ydata, p0=[1, 2])
```

### Failure Diagnostics

Enhanced failure tracking for post-mortem analysis:

```python
result = fitter.fit(model_func, xdata, ydata, p0=[1, 2])

# Check failure diagnostics
if result.failure_summary["total_failures"] > 0:
    print(f"Failed chunks: {result.failure_summary['failed_chunk_indices']}")
    print(f"Common errors: {result.failure_summary['common_errors']}")

    # Access detailed per-chunk diagnostics
    for chunk in result.chunk_results:
        if not chunk["success"]:
            print(f"Chunk {chunk['chunk_idx']}: {chunk['error_type']}")
            print(f"  Data stats: {chunk['data_stats']}")
            print(f"  Timestamp: {chunk['timestamp']}")
```

### Configurable Success Rate

Tune the minimum success rate threshold for chunked fitting:

```python
from nlsq import LDMemoryConfig, LargeDatasetFitter

# Default: require 50% of chunks to succeed
config = LDMemoryConfig(memory_limit_gb=8, min_success_rate=0.5)  # Default

# Stricter: require 80% success (good for clean data)
config_strict = LDMemoryConfig(memory_limit_gb=8, min_success_rate=0.8)

# More permissive: allow 30% failures (for very noisy data)
config_permissive = LDMemoryConfig(memory_limit_gb=8, min_success_rate=0.3)

fitter = LargeDatasetFitter(config=config_strict)
```

---

## Testing Strategy

### Test Organization
```
tests/
├── test_minpack.py              # Core API tests
├── test_least_squares.py        # Solver tests
├── test_trf_simple.py           # Algorithm tests
├── test_integration.py          # End-to-end tests
├── test_validators_comprehensive.py
└── benchmark/
    └── test_performance_regression.py  # CI/CD regression tests
```

### Coverage by Module
- Core API: ~85%
- Algorithms: ~75%
- Utilities: ~70%
- Overall: 77%

**Focus Areas** (to reach 80%):
- Error handling paths
- Edge cases (empty arrays, singular matrices)
- Large dataset code paths
- Recovery mechanisms

---

## Benchmarking

### Quick Start
```bash
# Standard benchmarks
python benchmark/run_benchmarks.py

# Quick mode (faster iteration)
python benchmark/run_benchmarks.py --quick

# Specific problems
python benchmark/run_benchmarks.py --problems exponential gaussian

# Skip SciPy comparison
python benchmark/run_benchmarks.py --no-scipy
```

### Performance Regression Tests
```bash
# Run regression tests
pytest benchmark/test_performance_regression.py --benchmark-only

# Save baseline
pytest benchmark/test_performance_regression.py --benchmark-save=baseline

# Compare against baseline
pytest benchmark/test_performance_regression.py --benchmark-compare=baseline
```

**See**: [`benchmark/README.md`](benchmark/README.md) for comprehensive benchmarking guide.

---

## File Structure

```
nlsq/
├── nlsq/                        # 25 core modules
├── tests/                       # 23 test files (1168 tests)
├── docs/                        # Sphinx documentation
│   ├── optimization_case_study.md
│   └── performance_tuning_guide.md
├── benchmark/                   # Profiling & regression tests
│   ├── run_benchmarks.py       # Main benchmark CLI
│   ├── profile_trf.py          # TRF profiler
│   └── test_performance_regression.py
├── examples/                    # Jupyter notebooks
├── pyproject.toml              # Package config (updated 2025-10-09)
├── requirements*.txt           # Dependency lock files
├── REQUIREMENTS.md             # Dependency strategy guide
├── CLAUDE.md                   # This file
└── README.md                   # User documentation
```

---

## Resources

### Documentation
- **ReadTheDocs**: https://nlsq.readthedocs.io
- **Dependencies**: [`REQUIREMENTS.md`](REQUIREMENTS.md)
- **Optimization**: [`docs/developer/optimization_case_study.md`](docs/developer/optimization_case_study.md)
- **Performance Tuning**: [`docs/developer/performance_tuning_guide.md`](docs/developer/performance_tuning_guide.md)
- **Benchmarking**: [`benchmark/README.md`](benchmark/README.md)

### External References
- **JAX Documentation**: https://jax.readthedocs.io
- **JAXFit Paper**: https://doi.org/10.48550/arXiv.2208.12187
- **SciPy curve_fit**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
- **NumPy 2.0 Migration**: https://numpy.org/devdocs/numpy_2_0_migration_guide.html

---

## Recent Updates (2025-10-19)

### v0.1.4 Release - Critical Bug Fixes

#### TRF Numerical Accuracy Fix
- **Fixed**: Critical bug where `res.fun` returned scaled instead of unscaled residuals when using loss functions
- **Impact**: HIGH - Silent data corruption affecting scientific results
- **Files**: `nlsq/trf.py` (6 lines modified)
- **Tests**: `test_least_squares.py::TestTRF::test_fun` now passing

#### Parameter Estimation Improvements
- **Fixed**: 5 test failures in automatic p0 estimation feature
- **Changes**: Array comparison bug, pattern detection reordering, sigmoid detection, VAR_POSITIONAL handling
- **Files**: `nlsq/parameter_estimation.py` (4 sections modified)
- **Tests**: All 25 parameter estimation tests now passing

#### Test & Coverage Improvements
- **Tests**: 1235/1235 passing (100% success rate)
- **Coverage**: 80.90% (exceeds 80% target for first time!)
- **Quality**: 24/24 pre-commit hooks passing

### Deprecations and Changes (v0.1.3+)
- ⚠️ **DEPRECATED**: Subsampling parameters in v0.1.3+
  - Parameters `enable_sampling`, `sampling_threshold`, `max_sampled_size` are deprecated
  - Deprecated parameters emit `DeprecationWarning` and are ignored
  - Subsampling implementation removed (~250 lines)
  - Replaced with streaming optimization for zero accuracy loss
  - **Migration**: Remove these parameters from your code (they have no effect)
- ✅ **Required Dependency**: h5py is now required (was optional)
  - All installations include streaming optimization capabilities
  - No separate `[streaming]` extra needed
  - Update installations: `pip install --upgrade nlsq`
- ✅ **Improvement**: Streaming optimization for all large datasets
  - Zero accuracy loss (processes 100% of data)
  - Better than subsampling (no data loss from random sampling)
  - Consistent, reproducible results
- ✅ **API Compatibility**: Graceful backward compatibility
  - Old code with deprecated params continues to work
  - Clear deprecation warnings guide migration

### Performance Optimizations (Phase 2)
- ✅ **Optimization #2: Parameter Unpacking Simplification** (Commit `574acea`)
  - Replaced 100-line if-elif chain with 5-line JAX solution
  - **95% code reduction** (100 lines → 5 lines) in least_squares.py
  - 5-10% faster for >10 parameters
  - Leverages JAX 0.8.0+ efficient tuple unpacking
  - All tests passing: 18/18 minpack + 14/14 TRF = 32/32 total
  - See [ADR-004](docs/architecture/adr/004-parameter-unpacking-simplification.md)

- ✅ **Optimization #4: JAX Autodiff for Streaming** (Commit `2ed084f`)
  - Replaced O(n_params) finite differences with O(1) JAX autodiff
  - **50-100x speedup** for gradient computation (>10 parameters)
  - Enables large-scale models with 100+ parameters
  - Exact gradients (no numerical approximation errors)
  - JIT-compiled and cached for performance
  - All tests passing: 21/21 streaming optimizer tests
  - See [ADR-005](docs/architecture/adr/005-jax-autodiff-gradients.md)

### Architecture Improvements (Phase 3)
- ✅ **Architecture Decision Records (ADRs)** (Commit `7ea5c34`)
  - Created `docs/architecture/adr/` directory with ADR template
  - Documented 3 key architectural decisions:
    - ADR-003: Streaming optimization over subsampling
    - ADR-004: Parameter unpacking simplification (Phase 2.2)
    - ADR-005: JAX autodiff for gradient computation (Phase 2.4)
  - Each ADR documents context, decision, consequences, and references
  - Helps future maintainers understand architectural choices

- ✅ **TRF Profiling Infrastructure** (Commit `b4a700f` - Phase 1.4)
  - Added TRFProfiler and NullProfiler classes (null object pattern)
  - Optional profiling with zero overhead when disabled
  - Cross-linked documentation between trf_no_bounds() and trf_no_bounds_timed()
  - Fixed benchmark/profile_trf.py bug (result format handling)
  - Infrastructure ready for future TRF consolidation

### Code Quality Refactoring (2025-10-18)

**Session 1** (Morning):
- ✅ **Task 1: LargeDatasetFitter._fit_chunked() Refactoring** (Commit `19b9245`)
  - **Complexity Reduction**: E(36) → C(14) = **61% reduction** (93% to B(8) target)
  - **Helper Methods Extracted** (2 new methods, 7 total):
    - `_initialize_chunked_fit_state()` - A(3) complexity
      - Handles progress reporter, parameters, tracking lists initialization
    - `_finalize_chunked_results()` - A(1) complexity
      - Assembles OptimizeResult with failure diagnostics
      - Computes covariance from parameter history
  - **Lines Reduced**: ~170 lines → ~140 lines (18% reduction)
  - **All Tests Passing**: 27/27 large dataset tests (100% success rate)
  - **Zero Regressions**: No performance degradation detected

- ✅ **Type Hints Foundation** (Commit `bb417b6`)
  - **Created nlsq/types.py** (160 lines) - Comprehensive type alias library
  - **Type Categories**:
    - Array types: `ArrayLike`, `FloatArray`, `JAXArray`
    - Function types: `ModelFunction`, `JacobianFunction`, `CallbackFunction`, `LossFunction`
    - Bounds/Results: `BoundsTuple`, `OptimizeResultDict`
    - Configuration: `MethodLiteral`, `SolverLiteral`
    - Protocols: `HasShape`, `SupportsFloat`
  - **Benefits**: IDE autocomplete, type documentation, mypy foundation
  - **Type Coverage**: +2% (63% → 65%)

**Session 2** (Evening):
- ✅ **Type Hints for Public API** (Commit `a247391`)
  - **Added comprehensive type hints to 3 core functions**:
    - `curve_fit()` in nlsq/minpack.py - Full parameter and return types
    - `least_squares()` in nlsq/least_squares.py - Complete type annotations
    - `curve_fit_large()` in nlsq/__init__.py - Full type signatures
  - **Validation**: Mypy passes with --check-untyped-defs (0 errors)
  - **Testing**: All 18 minpack tests passing, zero regressions
  - **Type Coverage**: +7% (63% → **~70%**)
  - **Task 6 Progress**: 20% → **60%**

- ✅ **Final _fit_chunked() Optimization** (Commit `36ea557`)
  - **Complexity Reduction**: C(14) → **B(9)** = **75% total reduction** from E(36)
  - **Helper Method Extracted** (1 new method, 8 total):
    - `_check_success_rate_and_create_result()` - B(6) complexity
      - Validates minimum success rate threshold
      - Creates failure result if too many chunks failed
      - Calls _finalize_chunked_results() on success
  - **All Tests Passing**: 27/27 large dataset tests (100% success rate)
  - **Zero Regressions**: No performance degradation
  - **Task 1 Progress**: 93% → **99%** (1 complexity point from B(8) target)

### Previous Updates (2025-10-17)

### Large Dataset Enhancements (v0.1.3)
- ✅ **Shape Validation**: Automatic validation of model functions for chunking compatibility
  - Tests with first 100 points before processing all chunks
  - Clear error messages with fix examples for shape mismatches
  - Prevents silent failures and invalid covariance matrices
  - Negligible overhead (~0.1s for multi-hour fits)
- ✅ **Logger Integration**: External logger support for application integration
  - Pass custom logger to `LargeDatasetFitter` for chunk failure visibility
  - Warnings and errors now appear in application logs
  - Better diagnostics for production deployments
- ✅ **Enhanced Failure Diagnostics**: Detailed per-chunk failure tracking
  - New `failure_summary` in `OptimizeResult` with error categorization
  - Per-chunk statistics (timestamps, data stats, error types)
  - Top-3 most common error types identified automatically
  - Easier post-mortem debugging when chunks fail
- ✅ **Configurable Success Rate**: Tunable success rate threshold
  - New `min_success_rate` parameter in `LDMemoryConfig` (default: 0.5)
  - Stricter thresholds (0.8) for clean data
  - More permissive thresholds (0.3) for noisy data
  - Better control over chunked fitting validation
- ✅ **Documentation**: Comprehensive chunking examples in docstrings
  - INCORRECT vs CORRECT model function examples
  - Clear guidance on using xdata as indices
  - Prevents common user mistakes

### Dependency Refresh
- ✅ **Major Updates Validated**: All dependencies updated to latest stable versions
  - JAX: 0.7.2 → 0.8.0 (1174 tests passing, fully compatible)
  - NumPy: 2.3.3 → 2.3.4 (patch update)
  - h5py: 3.14.0 → 3.15.1 (minor + patch)
  - ipykernel: 6.30.1 → 7.0.1 (major version, Jupyter support maintained)
  - Ruff: 0.14.0 → 0.14.1 (patch update)
  - hypothesis: 6.140.3 → 6.142.1 (patch updates)
  - pyupgrade: 3.20.0 → 3.21.0 (minor update)
  - setuptools-scm: 9.2.0 → 9.2.1 (patch update)
  - sphinx-autodoc-typehints: 3.5.1 → 3.5.2 (patch update)
- ✅ **Configuration Files Updated**: All package configs synchronized
  - requirements.txt, requirements-dev.txt, requirements-full.txt
  - pyproject.toml (tested version comments)
  - .pre-commit-config.yaml (hook versions)
- ✅ **100% Test Pass Rate**: All 1174 tests passing with new versions
- ✅ **78.99% Coverage**: Maintained coverage target progress

### Previous Updates (2025-10-09)

### Platform Stability & Bug Fixes
- ✅ **Windows Compatibility**: All Windows tests passing (100%)
  - Fixed file locking errors (PermissionError on file reads)
  - Fixed Unicode encoding errors (added UTF-8 encoding)
  - Fixed PowerShell line continuation errors in CI
- ✅ **Test Reliability**: Fixed flaky timing tests
  - Resolved macOS intermittent failures in test_compare_profiles
  - Improved timing variance from ±20% to ±2%
  - All platforms now passing consistently
- ✅ **Logging System**: Fixed invalid date format string
  - Removed unsupported %f from formatter (ValueError fix)
  - Logging now works correctly on all platforms
- ✅ **CI/CD**: All GitHub Actions passing
  - Ubuntu, macOS, Windows: 100% success rate
  - 0 flaky tests remaining
  - 70% faster execution from workflow optimizations

### Previous Updates (2025-10-08)

#### Dependency Management Overhaul
- ✅ **NumPy 2.0+ Required**: Updated to NumPy 2.3.4 (breaking change)
- ✅ **JAX 0.8.0**: Updated from 0.7.2 (validated with all tests passing)
- ✅ **Requirements Files**: Created lock files for reproducibility
  - `requirements.txt`: Runtime deps (exact versions)
  - `requirements-dev.txt`: Dev environment (exact versions)
  - `requirements-full.txt`: Complete pip freeze
- ✅ **REQUIREMENTS.md**: Comprehensive dependency strategy guide
- ✅ **Jupyter Support**: Added as optional `[jupyter]` extra

#### Code Quality (2025-10-07)
- ✅ **Performance**: 8% improvement via NumPy↔JAX optimization
- ✅ **Code Quality**: Sprint 3 refactoring (complexity 23→<10)
- ✅ **Documentation**: Sphinx warnings fixed (196 → 0)
- ✅ **Pre-commit**: 100% compliance (24/24 hooks)

### Test Status (Latest - 2025-10-19)
- **Passing**: 1235/1235 tests (**100% success rate**) ✅
- **Failing**: 0 tests
- **Skipped**: 6 tests
- **Coverage**: 80.90% (**exceeds 80% target**) ✅
- **Platforms**: Ubuntu ✅ | macOS ✅ | Windows ✅
- **CI Status**: All workflows passing
- **Regression**: 0 performance regressions detected

---

## Recent Bug Fixes (2025-10-19)

### ✅ FIXED: TRF Numerical Bug (2025-10-19)

**Test**: `test_least_squares.py::TestTRF::test_fun` - **NOW PASSING** ✅

**Root Cause**: When loss functions are applied, residuals are scaled for optimization but `res.fun` must contain **unscaled** residuals.

**Fix Applied**:
1. Added `f_true` tracking in `_initialize_trf_state()` to preserve original residuals
2. Added `f_true_new` in `_evaluate_step_acceptance()` to preserve unscaled residuals after each iteration
3. Updated `trf_no_bounds()` to return `f_true` (unscaled) instead of `f` (scaled)

**Files Modified**:
- `nlsq/trf.py`: Lines 1011, 1018, 1393, 1396, 1548, 1664

**Test Results**: All 31 TRF tests now passing (100% success rate)

---

### ✅ FIXED: Parameter Estimation (5 tests - 2025-10-19)

**Tests**: All in `test_parameter_estimation.py` - **NOW PASSING** ✅
1. `test_estimate_p0_error_no_signature` - ValueError not raised for *args functions
2. `test_estimate_p0_with_explicit_array` - Array truth value ambiguity
3. `test_detect_exponential_decay_pattern` - Returns 'linear' instead of 'exponential_decay'
4. `test_detect_sigmoid_pattern` - Returns 'linear' instead of 'sigmoid'
5. `test_unknown_pattern_fallback` - Wrong fallback behavior

**Root Causes**:
1. **Array comparison bug**: Line 149 used `p0 != "auto"` which fails for numpy arrays
2. **Pattern detection order**: Linear correlation check before exponential/sigmoid patterns
3. **Sigmoid vs exponential confusion**: Both are monotonic, needed inflection point detection
4. **Missing exception handling**: No check for VAR_POSITIONAL (*args) parameters
5. **Recursive call bug**: Unknown pattern fallback called itself with *args lambda

**Fixes Applied**:
1. **Line 149-152**: Check `isinstance(p0, str)` before comparing to "auto"
2. **Lines 304-359**: Reordered pattern detection:
   - Perfect linear (r > 0.99) checked first
   - Gaussian and sigmoid checked before monotonic patterns
   - Sigmoid detection uses second derivative to identify inflection points
   - Exponential patterns checked after sigmoid
   - General linear (r > 0.95) checked last
3. **Lines 166-196**: Added VAR_POSITIONAL/VAR_KEYWORD parameter detection
4. **Lines 485-514**: Replaced recursive call with direct generic estimation

**Files Modified**:
- `nlsq/parameter_estimation.py`: Lines 149-152, 166-196, 304-359, 485-514

**Test Results**: All 25 parameter estimation tests now passing (100% success rate)

---

### Historical Issues (Now Resolved)

**Tests**: All in `test_parameter_estimation.py`
1. `test_estimate_p0_error_no_signature` - ValueError not raised
2. `test_estimate_p0_with_explicit_array` - Array truth value ambiguity
3. `test_detect_exponential_decay_pattern` - Returns 'linear' instead of 'exponential_decay'
4. `test_detect_sigmoid_pattern` - Returns 'linear' instead of 'sigmoid'
5. `test_unknown_pattern_fallback` - Wrong fallback count

**Issue**: Automatic p0 estimation feature (`p0='auto'`) has broken pattern detection

**Impact**:
- **VERY LOW**: Feature is experimental/incomplete
- Test comment states: "p0='auto' may not be implemented yet in curve_fit"
- Core curve_fit functionality works perfectly (user provides explicit p0)

**Status**:
- Pattern detection logic in `parameter_estimation.py` returns 'linear' for all patterns
- Requires 2-3 hours to fix pattern recognition algorithms
- Not a regression (was already broken)

**Workaround**:
- Always provide explicit `p0` parameter (recommended best practice)
- Example: `curve_fit(model, x, y, p0=[1.0, 2.0])`

**Recommendation**: Mark feature as experimental or complete implementation in future release

---

**Last Updated**: 2025-10-19
**Version**: v0.1.4 (Bug Fix Release)
**Python**: 3.12.3
**Tested Configuration**: See [`REQUIREMENTS.md`](REQUIREMENTS.md)
