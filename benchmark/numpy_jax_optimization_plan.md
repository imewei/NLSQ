# NumPy ↔ JAX Conversion Optimization Plan

## Objective
Reduce unnecessary array conversions between NumPy and JAX to improve performance by 10-20%.

## Identified Conversion Points

### trf_no_bounds Method

| Line | Current Code | Usage | Can Eliminate? |
|------|--------------|-------|----------------|
| 894 | `cost = np.array(cost_jnp)` | Comparisons, logging | **Partial** - keep as JAX except for logging |
| 897 | `g = np.array(g_jnp)` | norm(), comparisons | **Yes** - jnp.linalg.norm works with JAX |
| 968 | `s, V, uf = (np.array(val) for val in svd_output[2:])` | solve_lsq_trust_region | **No** - SciPy function requires NumPy |
| 997 | `predicted_reduction = np.array(predicted_reduction_jnp)` | Comparisons, logging | **Partial** - keep as JAX except for logging |
| 1018 | `cost_new = np.array(cost_new_jnp)` | Comparisons | **Partial** - keep as JAX except for logging |
| 1068 | `g = np.array(g_jnp)` | norm(), loop continuation | **Yes** - same as line 897 |

### trf_bounds Method

| Line | Current Code | Usage | Can Eliminate? |
|------|--------------|-------|----------------|
| 1213 | `cost = np.array(cost_jnp)` | Same as trf_no_bounds | **Partial** |
| 1216 | `g = np.array(g_jnp)` | Same as trf_no_bounds | **Yes** |
| 1297 | `s, V, uf = (np.array(val) for val in output[2:])` | solve_lsq_trust_region | **No** |
| 1347 | `cost_new = np.array(cost_new_jnp)` | Same as trf_no_bounds | **Partial** |
| 1395 | `g = np.array(g_jnp)` | Same as trf_no_bounds | **Yes** |

## Strategy

### 1. Use JAX norm instead of NumPy norm
**Current**:
```python
from numpy.linalg import norm

g = np.array(g_jnp)  # Unnecessary conversion!
g_norm = norm(g, ord=np.inf)
```

**Optimized**:
```python
from jax.numpy.linalg import norm as jnorm

# Keep g as JAX array
g_norm = jnorm(g_jnp, ord=jnp.inf)
```

**Benefit**: Eliminates 4 conversions (lines 897, 1068, 1216, 1395)

### 2. Keep scalar values in JAX until final return
**Current**:
```python
cost = np.array(cost_jnp)  # Immediate conversion
# ... use cost in comparisons
actual_reduction = cost - cost_new  # Both NumPy
```

**Optimized**:
```python
# Keep as JAX through loop
cost_jnp = self.default_loss_func(f)
# ... use cost_jnp in comparisons
actual_reduction_jnp = cost_jnp - cost_new_jnp  # JAX operations are faster
```

**Convert only for**:
- Logging: `self.logger.info(..., cost=float(cost_jnp))`
- Final return: `cost=np.array(cost_jnp)`

**Benefit**: Reduces conversions in hot inner loop

### 3. Minimize conversions in inner loop
**Current inner loop** (lines 974-1075):
```python
while actual_reduction <= 0:  # actual_reduction is NumPy float
    predicted_reduction = np.array(predicted_reduction_jnp)  # Line 997
    cost_new = np.array(cost_new_jnp)  # Line 1018
    actual_reduction = cost - cost_new
```

**Optimized**:
```python
while float(actual_reduction_jnp) <= 0:  # Convert only for condition check
    # Keep as JAX
    predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(...)
    cost_new_jnp = self.default_loss_func(f_new)
    actual_reduction_jnp = cost_jnp - cost_new_jnp
```

**Benefit**: 2 conversions per inner loop iteration → 0-1 conversions

## Implementation Steps

### Step 1: Update imports
```python
# Change
from numpy.linalg import norm

# To
import jax.numpy as jnp
from jax.numpy.linalg import norm as jnorm
```

### Step 2: Update gradient norm calculations
```python
# Replace all instances of:
g = np.array(g_jnp)
g_norm = norm(g, ord=np.inf)

# With:
g_norm = jnorm(g_jnp, ord=jnp.inf)
```

### Step 3: Keep scalars in JAX format
```python
# Remove immediate conversions:
# cost = np.array(cost_jnp)  # DELETE
# predicted_reduction = np.array(predicted_reduction_jnp)  # DELETE
# cost_new = np.array(cost_new_jnp)  # DELETE

# Use JAX variables directly in comparisons
actual_reduction_jnp = cost_jnp - cost_new_jnp

# Convert only for Python control flow if needed (usually not needed):
if float(actual_reduction_jnp) > 0:
    ...
```

### Step 4: Update variable names for clarity
- Keep `_jnp` suffix for JAX arrays
- Remove `_jnp` suffix when conversion is eliminated
- Use `.item()` or `float()` for scalar extraction when needed

### Step 5: Convert only at final return or logging
```python
# Final return
return OptimizeResult(
    x=x,
    cost=float(cost_jnp),  # Convert scalar
    fun=f_true,
    jac=J,
    grad=np.array(g_jnp),  # Convert array for return
    ...,
)
```

## Expected Performance Impact

### Conversion Overhead Estimates
- NumPy ↔ JAX conversion: ~10-50μs per conversion (depends on array size)
- Inner loop iterations: 1-100 per outer iteration
- Outer loop iterations: 5-20 typically

**Current state** (per outer iteration):
- 2-6 conversions in inner loop × (1-100 iterations) = 2-600 conversions
- Overhead: 20μs - 30ms

**After optimization**:
- 0-1 conversions in inner loop × (1-100 iterations) = 0-100 conversions
- Overhead: 0-5ms

**Expected speedup**: 10-25% on TRF runtime

## Compatibility Considerations

### Functions that MUST use NumPy
1. `solve_lsq_trust_region` - SciPy function
2. `check_termination` - Check if accepts JAX
3. `update_tr_radius` - Check if accepts JAX
4. `CL_scaling_vector` - Check if accepts JAX (for trf_bounds)

### Functions that work with JAX
1. `norm()` - Use `jax.numpy.linalg.norm`
2. Arithmetic operations (+, -, *, /) - JAX native
3. Comparisons (>, <, ==) - Work with JAX scalars
4. `self.logger.*` - Convert to Python float/int for logging

## Testing Strategy

### 1. Numerical Correctness
```python
# Test that results are identical (within numerical tolerance)
popt_original, pcov_original = curve_fit_original(...)
popt_optimized, pcov_optimized = curve_fit_optimized(...)

assert np.allclose(popt_original, popt_optimized, rtol=1e-10)
assert np.allclose(pcov_original, pcov_optimized, rtol=1e-8)
```

### 2. Performance Benchmarking
```bash
# Before optimization
pytest benchmark/test_performance_regression.py --benchmark-save=before

# After optimization
pytest benchmark/test_performance_regression.py --benchmark-save=after

# Compare
pytest benchmark/test_performance_regression.py --benchmark-compare=before
```

### 3. Edge Cases
- Very small arrays (overhead vs benefit)
- Large arrays (conversion cost is higher)
- Early termination (fewer iterations)
- Maximum iterations (many conversions)

## Implementation Checklist

- [ ] Update imports (jnorm from jax.numpy.linalg)
- [ ] Replace g conversions with direct JAX usage (4 locations)
- [ ] Remove cost conversions in loop body (2 locations)
- [ ] Remove predicted_reduction conversion (1 location)
- [ ] Add conversions only for logging/final return
- [ ] Update variable names for clarity
- [ ] Test all test suites pass
- [ ] Benchmark performance improvement
- [ ] Document changes in code comments

## Risk Assessment

**Low Risk** ✅:
- JAX arrays are compatible with most NumPy operations
- Easy to revert if issues arise
- Can be done incrementally

**Potential Issues**:
- Some SciPy functions may not accept JAX arrays (keep conversions for those)
- Numerical precision differences (test thoroughly)
- Memory usage changes (JAX may allocate differently)

## Expected Outcome

**Conservative**: 8-12% speedup on TRF runtime
**Target**: 12-18% speedup on TRF runtime
**Optimistic**: 18-25% speedup on TRF runtime

Based on:
- Medium problem (1000 pts): 259ms TRF → 220-238ms (8-15% faster)
- Large problem (10000 pts): 312ms TRF → 265-288ms (8-15% faster)
