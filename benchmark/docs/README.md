# Benchmark Documentation

This directory contains optimization history, performance analysis, and detailed usage guides for NLSQ benchmarking.

## Directory Structure

```
docs/
├── README.md                      # This file
├── historical_results.md          # Benchmark data 2024-2025
├── usage_guide.md                 # Detailed usage examples and best practices
├── completed/                     # Completed optimizations
│   ├── numpy_jax_optimization.md  # NumPy↔JAX conversion optimization (8%)
│   └── trf_profiling.md           # TRF algorithm profiling baseline
└── future/                        # Future optimizations (deferred)
    └── lax_scan_design.md         # lax.scan inner loop conversion design
```

---

## Quick Links

### User Documentation
- **[Usage Guide](usage_guide.md)** - Comprehensive guide for running benchmarks
  - Command-line options
  - Configuration recommendations
  - Interpreting results
  - Troubleshooting
  - Best practices

- **[Historical Results](historical_results.md)** - Performance data 2024-2025
  - Performance evolution timeline
  - Detailed benchmark results
  - Stability features impact
  - Large dataset performance
  - Memory efficiency analysis

### Optimization History
- **[Completed Optimizations](completed/)** - Implemented performance improvements
- **[Future Optimizations](future/)** - Deferred optimization designs

---

## Completed Optimizations

### NumPy ↔ JAX Conversion Optimization
**File**: [completed/numpy_jax_optimization.md](completed/numpy_jax_optimization.md)
**Date**: October 2025
**Result**: 8% overall performance improvement (~15% on core TRF algorithm)
**Status**: ✅ COMPLETED

Eliminated 11 unnecessary array conversions between NumPy and JAX in hot paths (`nlsq/trf.py`). Zero numerical regressions, all tests passing.

**Key Changes**:
- Used JAX `jnp.linalg.norm` instead of NumPy `norm` (eliminated 4 conversions)
- Kept scalar values in JAX until final return (eliminated 7 conversions)
- Maintained JAX arrays in hot paths for better performance

**Impact**:
- 8% overall performance improvement
- ~15% improvement on TRF algorithm specifically
- Zero numerical regressions (all 743 tests passing)
- Validated with comprehensive profiling

---

### TRF Algorithm Profiling
**File**: [completed/trf_profiling.md](completed/trf_profiling.md)
**Date**: October 6, 2025
**Purpose**: Baseline performance analysis and hot path identification
**Status**: ✅ COMPLETED

Comprehensive profiling of Trust Region Reflective algorithm across different problem sizes. Identified hot paths and validated excellent scaling characteristics.

**Key Findings**:
- JIT compilation overhead dominates first run (60-75% of time)
- TRF algorithm scales excellently: 50x more data → only 1.2x slower
- Most time spent in JAX-compiled operations (already optimized)
- Minimal Python loop overhead
- Code already highly optimized with JAX primitives

**Value**:
- Established performance baseline for future work
- Identified diminishing returns for further micro-optimizations
- Informed decision to defer complex optimizations (lax.scan)

---

## Future Optimizations

### lax.scan Inner Loop Conversion
**File**: [future/lax_scan_design.md](future/lax_scan_design.md)
**Date**: October 2025
**Estimated Gain**: 5-10% (unverified)
**Status**: ⚠️ DEFERRED

Design for converting TRF inner loop (step acceptance) from Python `while` loop to JAX `lax.scan` for potential performance improvement.

**Deferral Rationale**:
- Diminishing returns (5-10% estimated gain for significant complexity)
- Code already highly optimized with JAX primitives
- User-facing features (Phase 1-3) more valuable than micro-optimizations
- May revisit if batch processing becomes common use case
- Would require significant refactoring with moderate risk

**When to Reconsider**:
- Batch processing of multiple curves becomes common
- GPU/TPU parallelization needs increase
- Profiling shows this as primary bottleneck (currently not the case)

---

## Documentation Usage

### When to Consult These Documents

**Usage Guide** ([usage_guide.md](usage_guide.md)):
- Running benchmarks for the first time
- Customizing benchmark configuration
- Understanding results and outputs
- Troubleshooting issues
- Learning best practices

**Historical Results** ([historical_results.md](historical_results.md)):
- Understanding performance evolution
- Comparing current vs historical performance
- Evaluating trade-offs (CPU vs GPU, stability overhead)
- Validating optimization impact
- Researching NLSQ performance characteristics

**Completed Optimizations** ([completed/](completed/)):
- Understanding why certain code patterns exist
- Learning what optimization strategies were effective
- Evaluating ROI of similar optimizations
- Documenting optimization methodology

**Future Optimizations** ([future/](future/)):
- Planning next performance improvements
- Understanding what was already considered
- Avoiding re-evaluation of deferred work
- Learning from deferral decisions

---

## How to Add New Documentation

### New Completed Optimization

1. Create file in `completed/` with descriptive name (e.g., `feature_name_optimization.md`)
2. Include sections:
   - **Date**: When completed
   - **Result**: Performance impact (with numbers)
   - **Status**: ✅ COMPLETED
   - **Key Changes**: What was changed and why
   - **Impact**: Numerical results, test status
   - **Lessons Learned**: What worked, what didn't
3. Update this README with summary
4. Update [historical_results.md](historical_results.md) if benchmark data changes

### New Future Optimization

1. Create file in `future/` with descriptive name (e.g., `technique_name_design.md`)
2. Include sections:
   - **Date**: When evaluated
   - **Estimated Gain**: Performance impact (if known)
   - **Status**: ⚠️ DEFERRED
   - **Deferral Rationale**: Why deferred
   - **Complexity Analysis**: Implementation difficulty
   - **When to Reconsider**: Conditions for revisiting
3. Update this README with summary

### Update Usage Guide

Add new sections to [usage_guide.md](usage_guide.md) when:
- New benchmarking tools added
- New command-line options available
- Common issues discovered
- Best practices identified

### Update Historical Results

Add to [historical_results.md](historical_results.md) when:
- Major performance improvements
- New benchmark data available
- Optimization phases complete
- Performance regressions identified and fixed

---

## Related Documentation

### Benchmark Documentation
- [Main Benchmark README](../README.md) - Quick start and active tools
- [Legacy Tools](../legacy/README.md) - Historical benchmarking tools

### NLSQ Documentation
- [NLSQ Main README](../../README.md) - Installation and overview
- [CHANGELOG](../../CHANGELOG.md) - Version history
- [CLAUDE.md](../../CLAUDE.md) - Developer guide and architecture
- [Examples](../../examples/) - Example notebooks and scripts

---

**Last Updated**: 2025-10-08
**Version**: v0.1.1
**Maintainer**: Wei Chen (Argonne National Laboratory)
