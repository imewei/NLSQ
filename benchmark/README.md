# NLSQ Benchmark Documentation

## Overview

This directory contains comprehensive benchmarks and performance analysis for NLSQ (Nonlinear Least Squares), a GPU/TPU-accelerated curve fitting library built on JAX. The benchmarks evaluate performance, memory efficiency, numerical stability, and scalability across various problem sizes and configurations.

## Directory Structure

```
benchmark/
├── README.md                          # This file - main benchmark documentation
├── benchmark.py                       # Main benchmarking tool (all suites)
├── test_performance_regression.py     # CI performance regression tests (13 tests)
├── benchmark_sprint2.py               # Sprint 2 benchmarks (memory pool, compilation cache)
├── profile_trf_hot_paths.py           # Profiling tool for TRF algorithm analysis
└── docs/                              # Historical & future optimization documentation
    ├── README.md                      # Documentation index
    ├── completed_optimizations/       # Implemented optimizations (8% NumPy↔JAX, profiling)
    └── future_optimizations/          # Deferred work (lax.scan design)
```

**Active Tools:**
- `benchmark.py` - Run comprehensive benchmarks (various suites: basic, solver, large, advanced)
- `test_performance_regression.py` - Automated regression tests integrated in CI/CD
- `profile_trf_hot_paths.py` - Profile TRF algorithm hot paths for optimization analysis
- `benchmark_sprint2.py` - Benchmark memory pool and compilation cache features

**Documentation:**
- See `docs/README.md` for historical optimization documentation and future work

## Table of Contents

- [Directory Structure](#directory-structure)
- [Performance Summary](#performance-summary)
- [Key Features & Optimizations](#key-features--optimizations)
- [Benchmark Results](#benchmark-results)
- [Stability Features Impact](#stability-features-impact)
- [Large Dataset Performance](#large-dataset-performance)
- [Memory Efficiency](#memory-efficiency)
- [Usage Recommendations](#usage-recommendations)
- [Running Benchmarks](#running-benchmarks)

## Performance Summary

### Current Performance (2024)

| Metric | Value | Notes |
|--------|-------|-------|
| **1D Fitting Speed** | 1.91-2.15ms | 150-270x improvement over initial version |
| **Stability Overhead** | 25-30% | When enabled (OFF by default) |
| **Memory Savings** | 98% | With dynamic sizing |
| **Large Dataset Support** | Up to 500K+ points | Sub-second performance |
| **vs SciPy (CPU)** | 10-20x slower | NLSQ optimized for GPU/TPU |

### Performance Evolution

- **Initial Benchmark**: 290-574ms for 1D fitting
- **Current Performance**: 1.91-2.15ms (150-270x faster)
- **Improvement Source**: Proper JAX JIT compilation, fixed recompilation issues

## Key Features & Optimizations

### 1. Numerical Stability System ✅

**Components Implemented:**
- **Numerical Stability Guard**: Overflow/underflow protection
- **Input Validation**: Comprehensive input checking with fast mode
- **Algorithm Selection**: Automatic parameter tuning
- **Memory Management**: Efficient array pooling
- **Robust Decomposition**: Multi-level fallback (GPU → CPU → NumPy)
- **Smart Caching**: LRU memory and disk caching
- **Diagnostics**: Convergence monitoring and pattern detection
- **Recovery System**: Automatic failure recovery

**Performance Impact:**
- Default (no features): Baseline performance
- With stability: +25-30% overhead
- With all features: +30-60% overhead
- Features are **OFF by default** (zero impact on existing users)

### 2. Iterative Solvers ✅

**Available Solvers:**
- `auto`: Automatic solver selection
- `svd`: Traditional SVD-based exact solver
- `cg`: Conjugate gradient (best for medium problems)
- `lsqr`: LSQR iterative solver (good for sparse problems)

**Performance (100x100 grid):**
- CG: 15.1ms ⭐ (fastest)
- LSQR: 15.5ms
- SVD: 17.8ms
- Auto: 28.3ms

### 3. Dynamic Sizing ✅

**Memory Savings:**
| Problem Size | Fixed Sizing | Dynamic Sizing | Savings |
|-------------|-------------|----------------|---------|
| 100x100 | 0.95 GB | 0.01 GB | 98% |
| 300x300 | 0.004 GB | 0.000 GB | 97% |

- Eliminates memory waste from padding
- Maintains identical numerical results
- Backward compatible

## Benchmark Results

### 1D Exponential Fitting

| Dataset Size | NLSQ Default | +Stability | SciPy | NLSQ Speedup |
|-------------|--------------|------------|-------|--------------|
| 50 points | 2.02ms | 2.60ms | 0.12ms | 0.06x |
| 100 points | 1.91ms | 2.50ms | 0.12ms | 0.06x |
| 200 points | 6.80ms | 8.80ms | 0.14ms | 0.02x |
| 500 points | 1.97ms | 2.56ms | 0.18ms | 0.09x |
| 1000 points | 2.15ms | 2.80ms | 0.24ms | 0.11x |

### 2D Gaussian Fitting

| Grid Size | NLSQ Default | +Stability | Overhead |
|-----------|--------------|------------|----------|
| 10x10 | 2.86ms | 3.17ms | +10.9% |
| 20x20 | 2.49ms | 3.97ms | +59.1% |
| 30x30 | 2.74ms | 5.46ms | +99.3% |

## Stability Features Impact

### Configuration Options

```python
# Maximum performance (default)
cf = CurveFit()  # No overhead

# Production systems (recommended)
cf = CurveFit(enable_stability=True)  # 25-30% overhead

# Numerical edge cases
cf = CurveFit(enable_stability=True, enable_overflow_check=True)  # 30% overhead

# Critical applications
cf = CurveFit(
    enable_stability=True, enable_recovery=True, enable_overflow_check=True
)  # 30-60% overhead
```

### Overhead Breakdown

For a typical 100-point 1D problem:
- **Baseline**: 1.91ms
- **Algorithm selection**: +0.16ms (7.2%)
- **Fast validation**: +0.02ms (1.1%)
- **Other (logging, etc)**: +0.40ms (18.5%)
- **Total overhead**: ~0.58ms (30%)

## Large Dataset Performance

### Scalability Test Results

| Dataset Size | Time (Default) | Time (+Stability) | Overhead | Memory |
|-------------|---------------|------------------|----------|---------|
| 1,000 | 0.266s | 0.269s | +1.3% | 574 MB |
| 10,000 | 0.262s | 0.322s | +23.0% | 591 MB |
| 50,000 | 0.328s | 0.372s | +13.5% | 612 MB |
| 100,000 | 0.320s | 0.367s | +14.7% | 618 MB |
| 500,000 | 0.432s | 0.457s | +5.7% | 799 MB |

**Key Findings:**
- Overhead decreases with larger datasets
- Sub-second performance even at 500K points
- Automatic switching to `curve_fit_large` for >100K points
- Memory impact is minimal

### Memory Requirements Estimation

| Dataset | Parameters | Base | Jacobian | Total |
|---------|------------|------|----------|-------|
| 1,000 | 3 | 0.000 GB | 0.000 GB | 0.000 GB |
| 100,000 | 5 | 0.002 GB | 0.004 GB | 0.018 GB |
| 10,000,000 | 10 | 0.224 GB | 0.745 GB | 2.924 GB |

## Memory Efficiency

### Dynamic Sizing Benefits

- **98% memory reduction** for large problems
- No performance degradation
- Automatic size detection
- Backward compatible

### Memory Overhead of Stability Features

| Dataset Size | Default | +Stability | Overhead |
|-------------|---------|------------|----------|
| 1,000 | 0.2 MB | 0.5 MB | +0.3 MB |
| 10,000 | 0.2 MB | 0.3 MB | +0.2 MB |
| 50,000 | 0.0 MB | 0.3 MB | +0.3 MB |

## Usage Recommendations

### By Use Case

| Use Case | Configuration | Performance Impact |
|----------|--------------|-------------------|
| **High Performance Computing** | `CurveFit()` | Baseline |
| **Production Systems** | `CurveFit(enable_stability=True)` | -25-30% |
| **Numerical Research** | `CurveFit(enable_stability=True, enable_overflow_check=True)` | -30% |
| **Critical Applications** | All features enabled | -30-60% |

### By Dataset Size

| Dataset Size | Recommendation |
|-------------|---------------|
| < 1K points | Use default `curve_fit()` |
| 1K - 100K points | Use `curve_fit()` with optional stability |
| > 100K points | Use `curve_fit_large()` (automatic chunking) |
| Numerical edge cases | Always enable overflow checking |

### Solver Selection

| Problem Type | Best Solver | Notes |
|-------------|------------|-------|
| Small (< 100 points) | `auto` or `svd` | Fast and accurate |
| Medium (100-10K) | `cg` | Best performance |
| Large (> 10K) | `cg` or `lsqr` | Memory efficient |
| Ill-conditioned | `svd` | Most stable |
| Sparse Jacobian | `lsqr` | Optimized for sparsity |

## Running Benchmarks

```bash
# Run all benchmarks
python benchmark.py --suite all

# Run specific suite
python benchmark.py --suite basic    # Basic performance tests
python benchmark.py --suite solver   # Solver comparison
python benchmark.py --suite large    # Large dataset tests
python benchmark.py --suite advanced # Advanced features

# Save results
python benchmark.py --suite all --save --output-dir results/

# Use GPU if available
python benchmark.py --suite all --gpu
```

## Conclusions

### Achievements

1. **Massive Performance Improvement**: 150-270x faster than initial implementation
2. **Comprehensive Stability System**: Optional features with acceptable overhead (25-30%)
3. **Memory Efficiency**: 98% reduction with dynamic sizing
4. **Large Dataset Support**: Handles 500K+ points efficiently
5. **Backward Compatibility**: Zero overhead when features disabled

### Trade-offs

- **CPU Performance**: SciPy still faster for small problems on CPU (10-20x)
- **Stability Overhead**: 25-30% when enabled (justified by reliability gains)
- **GPU Advantage**: NLSQ designed for GPU/TPU acceleration (not CPU)

### Status

✅ **Production Ready**: All optimizations complete and tested
- Stability features provide significant reliability improvements
- Performance overhead is acceptable and optional
- Suitable for both research and production use cases

## Related Documentation

### Benchmark Documentation
- [Optimization History](docs/README.md) - Historical and future optimization documentation
- [Completed Optimizations](docs/completed_optimizations/) - NumPy↔JAX optimization (8%), profiling results
- [Future Optimizations](docs/future_optimizations/) - Deferred work (lax.scan design)

### NLSQ Documentation
- [API Documentation](../docs/api.md)
- [Installation Guide](../README.md#installation)
- [Examples](../examples/)
- [Changelog](../CHANGELOG.md)
- [CLAUDE.md](../CLAUDE.md) - Developer guide and codebase information
