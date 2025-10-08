# Legacy Benchmarking Tools

This directory contains historical benchmarking tools that have been superseded by newer implementations but are preserved for reference and reproducibility of historical results.

## Contents

### benchmark_v1.py (Original: benchmark.py)
**Lines**: 701 lines
**Date**: 2024-2025
**Status**: LEGACY - Superseded by benchmark_suite.py + run_benchmarks.py

**Purpose**: Original comprehensive benchmarking tool with multiple test suites.

**Features**:
- Multiple benchmark suites: basic, solver, large, advanced
- NLSQ vs SciPy comparisons
- Solver comparisons (SVD, CG, LSQR)
- Large dataset testing
- Memory efficiency testing
- Sparse Jacobian testing
- Streaming optimizer testing

**Usage**:
```bash
# Run all benchmarks
python benchmark/legacy/benchmark_v1.py --suite all

# Run specific suite
python benchmark/legacy/benchmark_v1.py --suite basic
python benchmark/legacy/benchmark_v1.py --suite solver
python benchmark/legacy/benchmark_v1.py --suite large

# Save results
python benchmark/legacy/benchmark_v1.py --suite all --save --output-dir results/
```

**Why Legacy**: Replaced by more modular `benchmark_suite.py` + `run_benchmarks.py` which provide:
- Better CLI interface
- Profiler integration (Phase 3 feature)
- Dashboard generation
- More flexible configuration
- Cleaner architecture

**When to Use**:
- Reproducing historical benchmark results
- Comparing with legacy data
- Reference implementation for new benchmarks

---

### benchmark_sprint2.py
**Lines**: 232 lines
**Date**: October 2025 (Sprint 2)
**Status**: HISTORICAL - Sprint 2 artifact

**Purpose**: Sprint 2 specific benchmarks for performance improvements including function signature caching, memory pool optimization, and NumPy/JAX array handling.

**Features**:
- Function signature caching benchmarks
- Memory pool optimization tests
- NumPy/JAX array handling comparisons
- Before/after Sprint 2 optimization comparisons

**Usage**:
```bash
python benchmark/legacy/benchmark_sprint2.py
```

**Sprint 2 Context**:
Sprint 2 (October 2025) focused on:
- Memory pool optimization for TRF algorithm
- Function signature caching for faster JIT compilation
- NumPy↔JAX conversion optimization (achieved 8% improvement)

**Why Legacy**: Sprint-specific benchmarks no longer needed for general benchmarking. Results documented in:
- `../docs/completed/numpy_jax_optimization.md`
- Main CHANGELOG.md

**When to Use**:
- Understanding Sprint 2 optimization methodology
- Validating Sprint 2 improvements
- Historical performance comparison

---

## Migration Guide

### Migrating from benchmark_v1.py

**Old**:
```bash
python benchmark.py --suite all --save --output-dir results/
```

**New** (Recommended):
```bash
python benchmark/run_benchmarks.py --output results/
```

### Feature Mapping

| benchmark_v1.py | New Tools |
|----------------|-----------|
| `--suite basic` | `run_benchmarks.py --problems exponential gaussian` |
| `--suite solver` | `run_benchmarks.py --methods trf lm dogbox` |
| `--suite large` | `run_benchmarks.py --sizes 10000 100000` |
| `--save` | Automatic with `run_benchmarks.py` |
| `--output-dir` | `run_benchmarks.py --output DIR` |

### Key Improvements in New Tools

1. **Better CLI**: More intuitive flags and help text
2. **Profiler Integration**: Automatic performance profiling
3. **Dashboard**: HTML dashboard with visualizations
4. **Flexible Config**: Easy to customize problem sizes, repeats, methods
5. **Quick Mode**: Fast benchmarks for development (`--quick`)

---

## Deprecation Notice

⚠️ **These tools may be removed in future versions** (v0.2.0+)

Legacy tools will be maintained for at least one major version (through v0.1.x) but may be removed in v0.2.0 or later.

**Timeline**:
- v0.1.x: Legacy tools preserved in `legacy/`
- v0.2.0: Consider removal if unused
- Migration path: Use `run_benchmarks.py` + `benchmark_suite.py`

---

## Historical Results

For historical benchmark results and performance evolution, see:
- [Historical Results](../docs/historical_results.md) - Benchmark data 2024-2025
- [Completed Optimizations](../docs/completed/) - Optimization history
- [Main CHANGELOG](../../CHANGELOG.md) - Performance improvements by version

---

## Questions?

- **New benchmarks**: Use `run_benchmarks.py` (see main [README](../README.md))
- **Historical data**: See [historical_results.md](../docs/historical_results.md)
- **Optimization history**: See [docs/completed/](../docs/completed/)
- **Issues**: https://github.com/imewei/NLSQ/issues

---

**Last Updated**: 2025-10-08
**Status**: LEGACY - Preserved for historical reference
