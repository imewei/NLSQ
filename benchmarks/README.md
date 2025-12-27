# NLSQ Benchmark Suite

Comprehensive benchmarking and performance analysis tools for NLSQ (Nonlinear Least Squares), a GPU/TPU-accelerated curve fitting library built on JAX.

## Quick Start

```bash
# Run standard benchmarks (exponential, gaussian, polynomial, sinusoidal)
python benchmarks/run_benchmarks.py

# Quick benchmarks (smaller sizes, fewer repeats)
python benchmarks/run_benchmarks.py --quick

# Benchmark specific problems
python benchmarks/run_benchmarks.py --problems exponential gaussian --sizes 100 1000

# Skip SciPy comparison (faster)
python benchmarks/run_benchmarks.py --no-scipy

# Custom output directory
python benchmarks/run_benchmarks.py --output ./my_results

# View help
python benchmarks/run_benchmarks.py --help
```

**Output**: Text reports, CSV data, HTML dashboard with visualizations

---

## Active Tools

### 1. run_benchmarks.py ⭐ (Primary CLI)
**Purpose**: Command-line interface for running standardized benchmarks

**Features**:
- NLSQ vs SciPy comparisons
- Multiple problem types (exponential, gaussian, polynomial, sinusoidal)
- Flexible configuration (sizes, repeats, methods)
- Automatic dashboard generation with plots
- Quick mode for fast iteration

**Usage**: See [Quick Start](#quick-start) above

**Documentation**: [Usage Guide](docs/usage_guide.md)

### 2. benchmark_suite.py (Backend)
**Purpose**: Comprehensive benchmarking infrastructure

**Features**:
- Configurable benchmark suite (BenchmarkConfig)
- Profiler integration (PerformanceProfiler)
- Dashboard generation (ProfilingDashboard)
- Result export (CSV, JSON, HTML)

**Usage**: Typically used via `run_benchmarks.py`, but can be imported:

```python
from benchmark_suite import BenchmarkConfig, BenchmarkRunner

config = BenchmarkConfig(problem_sizes=[100, 1000], n_repeats=5)
runner = BenchmarkRunner(config)
results = runner.run_all_benchmarks()
```

### 3. profile_trf.py (Profiling Tool)
**Purpose**: Profile TRF algorithm hot paths for optimization analysis

**Features**:
- JIT compilation vs runtime breakdown
- Hot path identification
- Automatic optimization recommendations
- Scaling analysis

**Usage**:
```bash
python benchmarks/profile_trf.py
```

**Output**: Timing breakdown, recommendations for optimization

### 4. test_performance_regression.py ✅ (CI/CD)
**Purpose**: Automated performance regression tests for CI/CD

**Features**:
- 13 pytest-benchmark tests across problem sizes
- Baseline comparison
- Automatic regression detection (<5% threshold)
- JSON reports for CI integration

**Usage**:
```bash
# Run all regression tests
pytest benchmarks/test_performance_regression.py --benchmark-only

# Save baseline
pytest benchmarks/test_performance_regression.py --benchmark-save=baseline

# Compare against baseline
pytest benchmarks/test_performance_regression.py --benchmark-compare=baseline

# Generate CI report
pytest benchmarks/test_performance_regression.py --benchmark-json=report.json
```

---

## Directory Structure

```
benchmarks/
├── README.md                          # This file
├── run_benchmarks.py                  # Primary CLI ⭐
├── benchmark_suite.py                 # Comprehensive suite
├── profile_trf.py                     # TRF profiling tool
├── test_performance_regression.py     # CI/CD regression tests ✅
├── legacy/                            # Historical tools 📦
│   ├── README.md                      # Legacy tools documentation
│   ├── benchmark_v1.py                # Original benchmark tool
│   └── benchmark_sprint2.py           # Sprint 2 benchmarks
└── docs/                              # Documentation 📁
    ├── README.md                      # Documentation index
    ├── historical_results.md          # Benchmark results 2024-2025
    ├── usage_guide.md                 # Detailed usage examples
    ├── completed/                     # Completed optimizations
    │   ├── numpy_jax_optimization.md  # NumPy↔JAX optimization (8%)
    │   └── trf_profiling.md           # TRF profiling baseline
    └── future/                        # Future optimizations
        └── lax_scan_design.md         # Deferred: lax.scan design
```

---

## Current Performance (v0.1.1)

### Quick Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **1D Fitting Speed** | 1.91-2.15ms | 150-270x vs initial version |
| **GPU Speedup** | 150-270x | vs SciPy on CPU |
| **Large Datasets** | 500K+ points | Sub-second performance |
| **JIT Overhead** | 60-75% (first run) | Use CurveFit class to cache |
| **Scaling** | Excellent | 50x more data → only 1.2x slower |

### Performance Regression Tests (CI)

**Status**: ✅ All 13 tests passing (October 2025)

| Problem Size | Time (First Run) | Time (Cached) | Status |
|-------------|------------------|---------------|--------|
| Small (100) | ~500ms | 8.6ms | ✅ Pass |
| Medium (1K) | ~600ms | ~10ms | ✅ Pass |
| Large (10K) | ~630ms | ~15ms | ✅ Pass |
| XLarge (50K) | ~580ms | ~20ms | ✅ Pass |

**Zero Regressions**: 8% improvement from NumPy↔JAX optimization (October 2025)

---

## Recommendations by Use Case

### For Development

**Quick benchmarks**:
```bash
python benchmarks/run_benchmarks.py --quick
```

**Profile before optimizing**:
```bash
python benchmarks/profile_trf.py
```

**Run regression tests**:
```bash
pytest benchmarks/test_performance_regression.py --benchmark-only
```

### For Production

**Full comparison**:
```bash
python benchmarks/run_benchmarks.py --sizes 100 1000 10000 --repeats 10
```

**Save baseline for future comparison**:
```bash
pytest benchmarks/test_performance_regression.py --benchmark-save=production
```

### For Research

**Custom configuration**:
```bash
python benchmarks/run_benchmarks.py \
  --problems exponential gaussian \
  --sizes 100 500 1000 5000 10000 \
  --repeats 10 \
  --output ./research_results
```

**View dashboard**:
```bash
open research_results/dashboard/dashboard.html
```

---

## Key Insights

### When to Use NLSQ vs SciPy

| Scenario | Recommendation | Speedup |
|----------|---------------|---------|
| < 1K points, CPU | Use SciPy | SciPy 10-20x faster |
| > 1K points, CPU | Use NLSQ | Comparable or faster |
| Any size, GPU/TPU | Use NLSQ | 150-270x faster |
| Batch processing | Use NLSQ + CurveFit class | 58x faster (cached JIT) |

### Configuration Trade-offs

| Configuration | Overhead | Use Case |
|--------------|----------|----------|
| Default | 0% | High-performance computing |
| + Stability | +25-30% | Production systems |
| + All features | +30-60% | Critical applications |

**All advanced features are opt-in** (OFF by default)

---

## Documentation

### Quick Links

- **[Usage Guide](docs/usage_guide.md)** - Detailed usage examples and best practices
- **[Historical Results](docs/historical_results.md)** - Benchmark data 2024-2025
- **[Optimization History](docs/completed/)** - Completed optimizations
- **[Future Work](docs/future/)** - Deferred optimizations
- **[Legacy Tools](legacy/README.md)** - Historical benchmarking tools

### Optimization History

**NumPy↔JAX Optimization** (October 2025):
- **Result**: 8% overall improvement (~15% on TRF algorithm)
- **Method**: Eliminated 11 unnecessary array conversions
- **Status**: ✅ COMPLETED
- **Details**: [docs/completed/numpy_jax_optimization.md](docs/completed/numpy_jax_optimization.md)

**TRF Profiling** (October 2025):
- **Purpose**: Baseline performance analysis
- **Key Finding**: Excellent scaling (50x data → 1.2x slower)
- **Status**: ✅ COMPLETED
- **Details**: [docs/completed/trf_profiling.md](docs/completed/trf_profiling.md)

**lax.scan Conversion** (Deferred):
- **Estimated Gain**: 5-10% (unverified)
- **Status**: ⚠️ DEFERRED (diminishing returns)
- **Details**: [docs/future/lax_scan_design.md](docs/future/lax_scan_design.md)

---

## Troubleshooting

### Slow Performance

**Issue**: Benchmarks taking too long
**Solutions**:
- Use `--quick` flag for fast iteration
- Use `--no-scipy` to skip SciPy comparison
- Reduce problem sizes: `--sizes 100 1000`

### Memory Errors

**Issue**: Out of memory on large datasets
**Solutions**:
- Use `curve_fit_large()` instead of `curve_fit()`
- Reduce problem sizes
- Enable dynamic sizing (automatic)

### Inconsistent Results

**Issue**: Results vary between runs
**Solutions**:
- Increase repeats: `--repeats 10`
- Use more warmup runs (edit config)
- Check system resource contention

See [Usage Guide](docs/usage_guide.md#troubleshooting) for detailed troubleshooting.

---

## Related NLSQ Documentation

- [Main README](../README.md) - Installation and quick start
- [CHANGELOG](../CHANGELOG.md) - Version history and changes
- [CLAUDE.md](../CLAUDE.md) - Developer guide and architecture
- [Examples](../examples/) - Example notebooks and scripts

---

## Status

✅ **Production Ready** (v0.1.1, October 2025)
- Comprehensive benchmark suite (Phase 3, Days 20-24)
- 13 regression tests integrated in CI/CD
- Zero performance regressions
- 8% improvement from optimization work

**Last Updated**: 2025-10-08
**Version**: v0.1.1
**Maintainer**: Wei Chen (Argonne National Laboratory)
