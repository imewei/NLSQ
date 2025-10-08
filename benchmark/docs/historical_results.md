# NLSQ Historical Benchmark Results (2024-2025)

This document preserves historical benchmark data and performance evolution of NLSQ through various optimization phases.

## Performance Evolution

### Timeline
- **Initial Benchmark** (2024): 290-574ms for 1D fitting
- **Post-JIT Optimization** (2024): 1.91-2.15ms (150-270x faster)
- **NumPy↔JAX Optimization** (October 2025): 8% additional improvement
- **Current Performance** (v0.1.1): Baseline established, zero regressions

### Improvement Sources
1. **Proper JAX JIT compilation**: Fixed recompilation issues
2. **NumPy↔JAX conversion elimination**: Removed 11 conversions in hot paths
3. **Excellent scaling characteristics**: 50x more data → only 1.2x slower

---

## Detailed Benchmark Results

### 1D Exponential Fitting (2024)

| Dataset Size | NLSQ Default | +Stability | SciPy | NLSQ vs SciPy |
|-------------|--------------|------------|-------|---------------|
| 50 points | 2.02ms | 2.60ms | 0.12ms | 0.06x (slower) |
| 100 points | 1.91ms | 2.50ms | 0.12ms | 0.06x (slower) |
| 200 points | 6.80ms | 8.80ms | 0.14ms | 0.02x (slower) |
| 500 points | 1.97ms | 2.56ms | 0.18ms | 0.09x (slower) |
| 1000 points | 2.15ms | 2.80ms | 0.24ms | 0.11x (slower) |

**Notes**:
- NLSQ optimized for GPU/TPU, not CPU
- SciPy faster for small problems on CPU (10-20x)
- Stability overhead: 25-30% when enabled
- All benchmarks on CPU (JAX CPU backend)

### 2D Gaussian Fitting (2024)

| Grid Size | NLSQ Default | +Stability | Overhead |
|-----------|--------------|------------|----------|
| 10x10 | 2.86ms | 3.17ms | +10.9% |
| 20x20 | 2.49ms | 3.97ms | +59.1% |
| 30x30 | 2.74ms | 5.46ms | +99.3% |

**Stability Features Tested**:
- Numerical overflow/underflow protection
- Input validation (fast mode)
- Automatic algorithm selection

---

## Stability Features Impact

### Performance Overhead (2024 Data)

For typical 100-point 1D problem:
- **Baseline**: 1.91ms (100%)
- **Algorithm selection**: +0.16ms (+7.2%)
- **Fast validation**: +0.02ms (+1.1%)
- **Other (logging)**: +0.40ms (+18.5%)
- **Total overhead**: ~0.58ms (+30%)

### Configuration Performance Matrix

| Configuration | Overhead | Use Case |
|--------------|----------|----------|
| `CurveFit()` | 0% (baseline) | High-performance computing |
| `enable_stability=True` | +25-30% | Production systems |
| `+overflow_check=True` | +30% | Numerical edge cases |
| All features enabled | +30-60% | Critical applications |

---

## Large Dataset Performance (2024)

### Scalability Test Results

| Dataset Size | Time (Default) | Time (+Stability) | Overhead | Memory |
|-------------|---------------|------------------|----------|---------|
| 1,000 | 0.266s | 0.269s | +1.3% | 574 MB |
| 10,000 | 0.262s | 0.322s | +23.0% | 591 MB |
| 50,000 | 0.328s | 0.372s | +13.5% | 612 MB |
| 100,000 | 0.320s | 0.367s | +14.7% | 618 MB |
| 500,000 | 0.432s | 0.457s | +5.7% | 799 MB |

**Key Findings**:
- ✅ Overhead decreases with larger datasets
- ✅ Sub-second performance even at 500K points
- ✅ Automatic switching to `curve_fit_large` for >100K points
- ✅ Memory impact is minimal

### Memory Requirements Estimation

| Dataset | Parameters | Base | Jacobian | Total |
|---------|------------|------|----------|-------|
| 1,000 | 3 | 0.000 GB | 0.000 GB | 0.000 GB |
| 100,000 | 5 | 0.002 GB | 0.004 GB | 0.018 GB |
| 10,000,000 | 10 | 0.224 GB | 0.745 GB | 2.924 GB |

---

## Memory Efficiency (2024)

### Dynamic Sizing Benefits

Comparison of fixed vs dynamic array sizing:

| Problem Size | Fixed Sizing | Dynamic Sizing | Savings |
|-------------|-------------|----------------|---------|
| 100x100 | 0.95 GB | 0.01 GB | 98% |
| 300x300 | 0.004 GB | 0.000 GB | 97% |

**Benefits**:
- ✅ 98% memory reduction for large problems
- ✅ No performance degradation
- ✅ Automatic size detection
- ✅ Backward compatible

### Memory Overhead of Stability Features

| Dataset Size | Default | +Stability | Overhead |
|-------------|---------|------------|----------|
| 1,000 | 0.2 MB | 0.5 MB | +0.3 MB |
| 10,000 | 0.2 MB | 0.3 MB | +0.2 MB |
| 50,000 | 0.0 MB | 0.3 MB | +0.3 MB |

---

## Iterative Solvers Performance (2024)

### Solver Comparison (100x100 grid)

| Solver | Time | Relative |
|--------|------|----------|
| `cg` | 15.1ms | ⭐ Fastest |
| `lsqr` | 15.5ms | 1.03x |
| `svd` | 17.8ms | 1.18x |
| `auto` | 28.3ms | 1.87x |

**Recommendations**:
- **Small problems** (< 100 points): `auto` or `svd`
- **Medium problems** (100-10K): `cg` ⭐
- **Large problems** (> 10K): `cg` or `lsqr`
- **Ill-conditioned**: `svd` (most stable)
- **Sparse Jacobian**: `lsqr` (optimized for sparsity)

---

## Current Performance Summary (v0.1.1, October 2025)

### Key Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **1D Fitting Speed** | 1.91-2.15ms | 150-270x vs initial |
| **Stability Overhead** | 25-30% | Optional, OFF by default |
| **Memory Savings** | 98% | With dynamic sizing |
| **Large Dataset** | 500K+ points | Sub-second performance |
| **vs SciPy (CPU)** | 10-20x slower | NLSQ optimized for GPU/TPU |

### Performance Regression Tests (CI/CD)

**Test Suite**: 13 performance regression tests
**Status**: All passing (v0.1.1)
**Sizes Tested**: 100, 1K, 10K, 50K points
**Threshold**: <5% slowdown triggers alert

**Results (October 2025)**:
- Small problems (100): ~500ms (with JIT)
- Medium problems (1K): ~600ms
- Large problems (10K): ~630ms
- XLarge problems (50K): ~580ms
- **CurveFit class** (cached): 8.6ms ⭐ (58x faster)

**Scaling**:
- ✅ Excellent: 50x more data → only 1.2x slower
- ✅ Well-optimized with JAX primitives (51 @jit decorators)
- ✅ Minimal Python overhead

---

## Optimization History

### Phase 1: JAX JIT Compilation (2024)
**Improvement**: 150-270x speedup
**Key Fix**: Proper JIT compilation, fixed recompilation issues
**Result**: Reduced 1D fitting from 290-574ms to 1.91-2.15ms

### Phase 2: NumPy↔JAX Optimization (October 2025)
**Improvement**: 8% overall (~15% on TRF algorithm)
**Method**: Eliminated 11 unnecessary array conversions
**Files Changed**: `nlsq/trf.py`
**Testing**: Zero numerical regressions
**Documentation**: `docs/completed/numpy_jax_optimization.md`

### Phase 3: Deferred Optimizations (October 2025)
**Candidates**: lax.scan inner loop, @vmap, @pmap
**Estimated Gain**: 5-10% (unverified)
**Status**: DEFERRED due to diminishing returns
**Rationale**: User-facing features more valuable than micro-optimizations
**Documentation**: `docs/future/lax_scan_design.md`

---

## Trade-offs and Conclusions

### Achievements ✅

1. **Massive Performance Improvement**: 150-270x faster than initial implementation
2. **Comprehensive Stability System**: Optional features with acceptable overhead (25-30%)
3. **Memory Efficiency**: 98% reduction with dynamic sizing
4. **Large Dataset Support**: Handles 500K+ points efficiently
5. **Backward Compatibility**: Zero overhead when features disabled

### Trade-offs ⚠️

1. **CPU Performance**: SciPy still faster for small problems on CPU (10-20x)
   - **Rationale**: NLSQ designed for GPU/TPU acceleration
   - **Recommendation**: Use SciPy for <1K points on CPU

2. **Stability Overhead**: 25-30% when enabled
   - **Rationale**: Justified by reliability gains
   - **Mitigation**: OFF by default, opt-in when needed

3. **GPU Advantage**: NLSQ excels on GPU/TPU, not CPU
   - **Example**: 1M points, 5 parameters: 0.15s (GPU) vs 40.5s (SciPy CPU) = **270x speedup**

### Status

✅ **Production Ready** (v0.1.1)
- All optimizations complete and tested
- Stability features provide significant reliability improvements
- Performance overhead is acceptable and optional
- Suitable for both research and production use cases

---

## References

### Optimization Documentation
- [NumPy↔JAX Optimization](completed/numpy_jax_optimization.md) - 8% improvement
- [TRF Profiling Results](completed/trf_profiling.md) - Baseline performance analysis
- [Future Optimizations](future/lax_scan_design.md) - Deferred work

### Related Documentation
- [Main Benchmark README](../README.md) - Current tools and quick start
- [Usage Guide](usage_guide.md) - Detailed usage examples
- [CHANGELOG](../../CHANGELOG.md) - Version history
- [CLAUDE.md](../../CLAUDE.md) - Developer guide

---

**Last Updated**: 2025-10-08
**Version**: v0.1.1
**Data Period**: 2024-2025
