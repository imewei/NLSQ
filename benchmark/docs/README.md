# Benchmark Documentation

This directory contains historical and future optimization documentation for NLSQ performance.

## Directory Structure

```
docs/
├── completed_optimizations/    # Historical: Implemented optimizations
└── future_optimizations/       # Deferred: Potential future work
```

## Completed Optimizations

Documentation of performance optimizations that have been successfully implemented.

### NumPy ↔ JAX Conversion Optimization
**File:** `completed_optimizations/numpy_jax_optimization_plan.md`
**Date:** October 2025
**Result:** 8% overall performance improvement (~15% on core TRF algorithm)
**Status:** ✅ COMPLETED

Eliminated 11 unnecessary array conversions between NumPy and JAX in hot paths (`trf.py`). Zero numerical regressions, all tests passing.

**Key Changes:**
- Used JAX `jnp.linalg.norm` instead of NumPy `norm` (eliminated 4 conversions)
- Kept scalar values in JAX until final return (eliminated 7 conversions)
- Maintained JAX arrays in hot paths for better performance

### TRF Algorithm Profiling
**File:** `completed_optimizations/trf_profiling_summary.md`
**Date:** October 6, 2025
**Purpose:** Baseline performance analysis
**Status:** ✅ COMPLETED

Comprehensive profiling of Trust Region Reflective algorithm across different problem sizes. Identified hot paths and validated excellent scaling characteristics (50x more data → only 1.2x slower).

**Key Findings:**
- JIT compilation overhead dominates first run (60-75% of time)
- TRF algorithm scales excellently with data size
- Most time spent in JAX-compiled operations (already optimized)
- Minimal Python loop overhead

## Future Optimizations

Design documents for potential future optimizations that have been deferred due to diminishing returns or complexity concerns.

### lax.scan Inner Loop Conversion
**File:** `future_optimizations/lax_scan_design.md`
**Date:** October 2025
**Estimated Gain:** 5-10% (unverified)
**Status:** ⚠️ DEFERRED

Design for converting TRF inner loop (step acceptance) from Python `while` loop to JAX `lax.scan` for potential performance improvement.

**Deferral Rationale:**
- Diminishing returns (5-10% estimated gain for significant complexity)
- Code already highly optimized with JAX primitives
- User-facing features more valuable than micro-optimizations
- May revisit if batch processing becomes common use case

---

## Usage

### When to Consult These Documents

**Completed Optimizations:**
- Understanding why certain code patterns exist
- Comparing current vs. historical performance
- Learning what optimization strategies were effective

**Future Optimizations:**
- Planning next performance improvements
- Understanding what was already considered
- Evaluating ROI for complex optimizations

### How to Add New Documentation

**New Completed Optimization:**
1. Create file in `completed_optimizations/` with descriptive name
2. Include: date, result, status, key changes, performance impact
3. Update this README with summary

**New Future Optimization:**
1. Create file in `future_optimizations/` with descriptive name
2. Include: estimated gain, complexity analysis, deferral rationale
3. Update this README with summary

---

**Last Updated:** 2025-10-06
**Maintainer:** Wei Chen (Argonne National Laboratory)
