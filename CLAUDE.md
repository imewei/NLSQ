# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview
NLSQ is a GPU/TPU-accelerated nonlinear least squares curve fitting library that ports SciPy's `curve_fit` to JAX.

**Repository**: https://github.com/imewei/NLSQ
**Maintainer**: Wei Chen (Argonne National Laboratory)
**Status**: Production-ready (Beta), 77% test coverage, 817/820 tests passing

## Recent Updates (2025-10-07)

### Completed Work
- ✅ **Pre-commit compliance**: 100% (24/24 hooks)
- ✅ **Performance**: 8% improvement via NumPy↔JAX conversion reduction
- ✅ **Tests**: 100% pass rate (817 passing, 3 skipped)
- ✅ **Code quality**: Sprint 3 refactoring (18 complexity violations, 2 functions refactored)
- ✅ **Documentation**: Added case studies, tuning guides, CI/CD fixes

### Key Files Modified
- `nlsq/validators.py`, `algorithm_selector.py` - Refactored (complexity 23→<10, 20→<10)
- `nlsq/trf.py` - Performance optimization
- `tests/test_validators_comprehensive.py` - Fixed 10 API mismatch tests
- Config files - Updated test counts (743) and coverage (70%)

See `sprint3_completion_summary.md` and `docs/optimization_case_study.md` for details.

## Development Commands

### Testing
```bash
make test              # All tests
make test-fast         # Exclude slow tests
make test-cov          # With coverage
pytest tests/test_minpack.py::test_exponential_fit -v  # Specific test
```

### Code Quality
```bash
make lint              # Linting
make format            # Black + ruff formatting
pre-commit run --all-files
```

### Performance
```bash
python benchmark/profile_trf.py  # Profile TRF
pytest benchmark/test_performance_regression.py -v  # Regression tests
```

## Architecture

### Core Modules (nlsq/)
**Core**: `minpack.py` (API), `least_squares.py` (solver), `trf.py` (algorithm), `loss_functions.py`
**Advanced**: `algorithm_selector.py`, `large_dataset.py`, `memory_manager.py`, `validators.py`
**Infrastructure**: `config.py`, `common_jax.py`, `common_scipy.py`, `logging.py`, `caching.py`

### Key Design Principles
- **JAX JIT**: All fit functions must be JIT-compilable (no Python control flow)
- **Float64**: Auto-enabled via `config.update("jax_enable_x64", True)`
- **Autodiff**: JAX autodiff for Jacobians
- **SciPy Compatible**: Drop-in replacement for `scipy.optimize.curve_fit`

### API Compatibility
```python
# Same signature as SciPy
curve_fit(f, xdata, ydata, p0=None, ...)
# Returns (popt, pcov)
# Use CurveFit class to reuse compiled functions
```

## Testing

**Framework**: pytest + unittest
**Coverage**: 77% (target 80%)
**Key files**: `test_minpack.py`, `test_least_squares.py`, `test_trf_simple.py`, `test_integration.py`

## Performance

**Status**: Production-ready, extensively optimized

**Benchmarks** (CPU):
- Small (100 pts): ~430ms (30ms after JIT)
- Medium (1K pts): ~490ms (110ms after JIT)
- Large (10K pts): ~605ms (134ms after JIT)

**GPU** (NVIDIA V100):
- 1M points: 0.15s (NLSQ) vs 40.5s (SciPy) = **270x speedup**

**Optimization Status**:
- ✅ 8% improvement from NumPy↔JAX conversion reduction
- ✅ 51 @jit decorators, minimal Python overhead
- ⏸️ Further optimizations deferred (diminishing returns)

**Tuning Guide**:
1. Use `CurveFit` class for multiple fits (reuses JIT compilation)
2. Enable GPU/TPU (auto-detected by JAX)
3. Profile first: `benchmark/profile_trf.py`
4. Use `curve_fit_large()` for datasets >20M points

**Size Recommendations**:
- <1K: Use SciPy (JIT overhead)
- 1K-100K: NLSQ CPU/GPU
- >100K: NLSQ GPU

## File Structure
```
nlsq/
├── nlsq/               # 25 core modules
├── tests/              # 23 test files (817 tests)
├── docs/               # Case studies, guides
├── benchmark/          # Profiling, regression tests
├── examples/           # Jupyter notebooks
└── pyproject.toml      # Configuration
```

## Development Guidelines

### Code Quality
- **Testing**: Run `make test` before committing
- **Linting**: Pre-commit hooks (ruff + black)
- **Type hints**: ~60% coverage, mypy configured

### JAX Best Practices
- **Immutability**: Use `np.array(x, copy=True)` for in-place ops
- **JIT**: Fit functions must avoid Python control flow
- **Conversions**: Minimize NumPy↔JAX conversions in hot paths

### Testing Best Practices
- **Random seeds**: Always set in tests with random data
- **Bounds**: Use realistic tolerances for chunked/approximated algorithms
- **Coverage**: Focus on error paths and edge cases

### Performance
- **Don't optimize prematurely**: Code already highly optimized
- **Profile first**: Use `benchmark/profile_trf.py`
- **Document decisions**: Update `docs/optimization_case_study.md`

## Common Issues

### JAX Array Immutability
**Error**: `TypeError: JAX arrays are immutable`
**Fix**: `np.array(x, copy=True)` to convert to mutable NumPy array

### Flaky Tests
**Error**: Non-deterministic pass/fail
**Fix**: Set random seed + relax bounds for approximated algorithms

### Performance Regression
**Detection**: `benchmark/test_performance_regression.py` (>5% slowdown)
**Action**: Profile with `benchmark/profile_trf.py`

## Resources

**Documentation**:
- ReadTheDocs: https://nlsq.readthedocs.io
- Case Study: `docs/optimization_case_study.md`
- Tuning Guide: `docs/performance_tuning_guide.md`

**Benchmarking**:
- Main docs: `benchmark/README.md`
- CLI: `benchmark/run_benchmarks.py`
- Regression: `benchmark/test_performance_regression.py`

**External**:
- JAX: https://jax.readthedocs.io
- JAXFit Paper: https://doi.org/10.48550/arXiv.2208.12187
- SciPy curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

---

**Last Updated**: 2025-10-07 | **Status**: Production-ready (Beta) | **Tests**: 817/820 passing | **Coverage**: 77%
