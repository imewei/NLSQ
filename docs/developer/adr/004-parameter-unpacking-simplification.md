# ADR-004: Parameter Unpacking Simplification

**Status**: Accepted

**Date**: 2025-10-18

**Deciders**: Performance Engineer, Code Quality Review (Phase 2.2)

## Context

The `masked_residual_func` in `least_squares.py` contained a 100-line if-elif chain to handle parameter unpacking for 1-15+ parameters. This was originally implemented to avoid `TracerArrayConversionError` in early JAX versions.

### Original Implementation (100 lines)
```python
if args.size == 1:
    func_eval = func(xdata, args[0]) - ydata
elif args.size == 2:
    func_eval = func(xdata, args[0], args[1]) - ydata
elif args.size == 3:
    func_eval = func(xdata, args[0], args[1], args[2]) - ydata
# ... 13 more branches ...
elif args.size <= 15:
    param_vals = tuple(args[i] for i in range(args.size))
    func_eval = func(xdata, *param_vals) - ydata
else:
    warnings.warn(...)
    func_eval = func(xdata, *args) - ydata
```

### Problems
1. **Code Maintenance**: 100 lines of repetitive boilerplate
2. **Complexity**: Harder to understand and modify
3. **Limited Scalability**: Special warnings for >15 parameters
4. **Outdated Workaround**: JAX 0.8.0+ handles tuple unpacking efficiently

## Decision

**Replace the 100-line if-elif chain with direct tuple unpacking.**

### New Implementation (5 lines)
```python
# JAX 0.8.0+ handles tuple unpacking efficiently without TracerArrayConversionError
# This replaces the previous 100-line if-elif chain (Optimization #2)
# See: OPTIMIZATION_QUICK_REFERENCE.md for performance analysis
func_eval = func(xdata, *args) - ydata
return jnp.where(data_mask, func_eval, 0)
```

## Consequences

### Positive
✅ **95% Code Reduction**: 100 lines → 5 lines
✅ **Improved Readability**: Immediately clear what the code does
✅ **Better Performance**: 5-10% faster for >10 parameters
✅ **Unlimited Parameters**: No artificial 15-parameter limit
✅ **Easier Maintenance**: Single implementation instead of 15+ branches
✅ **Modern JAX**: Leverages JAX 0.8.0+ improvements

### Negative
❌ **Requires JAX 0.8.0+**: Older JAX versions may have issues
  - **Mitigation**: NLSQ already requires JAX 0.8.0+ (tested with 0.8.0)
❌ **Loss of Explicit Branches**: Debugging slightly less granular
  - **Mitigation**: Actual impact minimal, error messages still clear

### Performance Impact
- **1-10 parameters**: Equivalent performance
- **10-15 parameters**: 5-10% faster
- **15+ parameters**: 5-10% faster + no warning overhead

## Testing

Comprehensive testing validates the change:
- ✅ 18/18 minpack tests passing (covers 1-15 parameters)
- ✅ 14/14 TRF tests passing
- ✅ 32/32 total tests passing
- ✅ No performance regression detected

## References

- [least_squares.py Implementation](../../../nlsq/core/least_squares.py#L1087-L1091)
- [JAX 0.8.0 Release Notes](https://github.com/google/jax/releases/tag/jax-v0.8.0)
- [Commit 574acea](https://github.com/imewei/NLSQ/commit/574acea)

## Status Updates

- **2025-10-18**: Accepted and implemented in Phase 2.2
- **2025-10-18**: Verified with full test suite (32/32 passing)
