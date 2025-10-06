# TRF Algorithm Profiling Summary

**Date**: 2025-10-06
**Purpose**: Identify hot paths for lax.scan optimization

## Performance Breakdown by Problem Size

| Problem Size | Total Time | TRF Time | Compilation Time | TRF % of Total |
|--------------|-----------|----------|------------------|----------------|
| Small (100 pts) | 1598ms | ~600ms | ~900ms | 37.5% |
| Medium (1000 pts) | 511ms | 259ms | 383ms | 50.7% |
| Large (10000 pts) | 642ms | 312ms | 471ms | 48.6% |
| XLarge (50000 pts) | 609ms | 326ms | 452ms | 53.5% |

## Key Findings

### 1. JIT Compilation Overhead Dominates First Run
- **Compilation**: 383-471ms (60-75% of total time on first run)
- **Runtime**: 259-326ms (actual optimization)
- **Implication**: Subsequent runs will be much faster due to caching

### 2. TRF Algorithm Scales Well
- 50x more data (1000 → 50000 points): only 1.26x slower (259ms → 326ms)
- Good scalability suggests current implementation is already efficient
- Most time is in JAX-compiled operations, not Python loops

### 3. Hot Functions (from profiling)

#### Top 5 by Cumulative Time:
1. `backend_compile_and_load`: 383-470ms (JIT compilation)
2. `trf_no_bounds`: 259-326ms (main algorithm)
3. `evaluate_quadratic`: 66-73ms (quadratic model evaluation)
4. `_pjit_call_impl_python`: 398-536ms (JAX dispatch)
5. `compile`/`from_hlo`: 388-488ms (HLO compilation)

### 4. Loop Structure Analysis

#### Outer Loop (Main Optimization)
- **Location**: `trf.py:923` (trf_no_bounds), `trf.py:1247` (trf_bounds)
- **Pattern**: `while True` with dynamic termination
- **Iterations**: Variable (depends on convergence, typically 5-20)
- **State**: x, f, J, g, cost, Delta, alpha, nfev, njev

**Operations per iteration**:
- Gradient computation: `compute_grad(J, f)`
- SVD decomposition: `svd_no_bounds(J, d_jnp, f)`
- Quadratic evaluation: `evaluate_quadratic(J_h, g_h, step_h)`
- Function evaluation: `fun(x_new, ...)`
- Jacobian evaluation: `jac(x, ...)`

#### Inner Loop (Step Acceptance)
- **Location**: `trf.py:974` (trf_no_bounds), `trf.py:1305` (trf_bounds)
- **Pattern**: `while (actual_reduction <= 0 and nfev < max_nfev and count < 100)`
- **Iterations**: 1-100 (typically 1-5 for well-behaved problems)
- **State**: Delta, alpha, actual_reduction, inner_loop_count

**Operations per iteration**:
- Solve trust region subproblem: `solve_lsq_trust_region(...)`
- Evaluate quadratic: `evaluate_quadratic(...)`
- Function evaluation: `fun(x_new, ...)`
- Trust region update: `update_tr_radius(...)`

## Optimization Opportunities

### High Priority (Expected 1.5-2x Speedup)

#### 1. Convert Inner Loop to lax.scan
- **Target**: Inner loop (lines 974-1045 in trf_no_bounds)
- **Benefit**:
  - Eliminates Python-level control flow
  - Enables better JIT compilation
  - Vectorizes trust region updates
- **Complexity**: Medium
  - Fixed max iterations (100)
  - Needs lax.cond for conditional updates
  - Early termination when actual_reduction > 0

**Implementation approach**:
```python
def inner_loop_body(carry, _):
    x, Delta, alpha, nfev, actual_reduction = carry

    # Solve subproblem (conditionally)
    step_h, alpha = lax.cond(
        actual_reduction <= 0,
        lambda s: solve_subproblem(s),
        lambda s: s,  # No-op if converged
        (Delta, alpha),
    )

    # Evaluate and update
    f_new = fun(x + step)
    actual_reduction = cost - cost_new
    Delta_new = update_tr_radius(...)

    return (x, Delta_new, alpha, nfev + 1, actual_reduction), None


# Run scan
(x_final, Delta, alpha, nfev, reduction), _ = lax.scan(
    inner_loop_body,
    (x0, Delta0, alpha0, nfev0, -1.0),
    None,
    length=max_inner_iterations,
)
```

#### 2. Minimize NumPy ↔ JAX Conversions
- **Current**: Frequent `np.array(jax_result)` calls
- **Target**: Lines with `np.array(...)` after JAX operations
- **Benefit**: 10-20% speedup
- **Locations**:
  - Line 997: `predicted_reduction = np.array(predicted_reduction_jnp)`
  - Line 1018: `cost_new = np.array(cost_new_jnp)`
  - Line 1068: `g = np.array(g_jnp)`

**Implementation**: Keep data in JAX arrays until final return

### Medium Priority (Expected 1.2-1.5x Speedup)

#### 3. Vectorize Quadratic Evaluations
- **Target**: Multiple calls to `evaluate_quadratic`
- **Benefit**: Batch operations, reduce kernel launch overhead
- **Complexity**: Low

#### 4. Pre-allocate Iteration History
- **Current**: Lists for timing data (trf_no_bounds_timed)
- **Benefit**: Reduce allocation overhead
- **Complexity**: Low

### Lower Priority (Expected 1.1-1.2x Speedup)

#### 5. Convert Outer Loop to lax.while_loop
- **Target**: Main optimization loop (line 923)
- **Benefit**: Eliminate Python-level loop overhead
- **Complexity**: High
  - Dynamic termination conditions
  - Multiple exit criteria
  - Complex state management

**Note**: May not provide significant speedup since outer loop typically has few iterations (5-20)

## Expected Performance Improvements

### Conservative Estimates:
- **Inner loop lax.scan**: 1.3-1.5x speedup (13-50ms savings)
- **Reduced conversions**: 1.1-1.2x speedup (10-20ms savings)
- **Combined**: 1.5-1.8x total speedup

### Optimistic Estimates:
- **Inner loop lax.scan**: 1.5-2x speedup
- **Reduced conversions**: 1.2-1.3x speedup
- **Vectorization**: 1.1-1.2x speedup
- **Combined**: 2-3x total speedup

### Realistic Target:
- **Phase 1 (Inner loop + conversions)**: 1.5-2x speedup on TRF runtime
  - Medium problem: 259ms → 130-173ms
  - Large problem: 312ms → 156-208ms

## Implementation Plan

### Step 1: Inner Loop lax.scan (High Priority)
1. Extract inner loop body into pure function
2. Implement carry state (x, Delta, alpha, nfev, actual_reduction)
3. Use lax.cond for conditional operations
4. Handle early termination with masking
5. Test numerical correctness against original
6. Benchmark performance improvement

### Step 2: Reduce NumPy ↔ JAX Conversions
1. Identify all `np.array(jax_result)` calls
2. Keep data in JAX format through iterations
3. Convert only at final return
4. Verify no performance regression

### Step 3: Vectorization and Batching
1. Batch quadratic evaluations
2. Use vmap where appropriate
3. Profile and optimize

### Step 4: Optional - Outer Loop Optimization
1. Convert to lax.while_loop if inner loop shows good results
2. Complex state management required
3. Test extensively for numerical correctness

## Testing Strategy

### Numerical Correctness
- Compare results against SciPy's least_squares
- Tolerance: `np.allclose(popt_new, popt_old, atol=1e-8)`
- Test with all benchmark problems (small, medium, large, xlarge)
- Test edge cases (ill-conditioned, near-singular, bounds)

### Performance Validation
- Run pytest-benchmark suite before/after changes
- Target: 1.5-2x improvement on TRF runtime
- Ensure no regression on compilation time
- Test across problem sizes

### Edge Cases
- Early convergence (1 iteration)
- Maximum iterations reached
- Ill-conditioned problems
- Bounded vs unbounded optimization

## Risks and Mitigation

### Risk 1: Numerical Stability
- **Issue**: lax.scan may handle edge cases differently
- **Mitigation**: Extensive testing, keep NumPy fallback

### Risk 2: Compilation Overhead
- **Issue**: Complex lax.scan may increase JIT time
- **Mitigation**: Profile compilation time, simplify if needed

### Risk 3: Modest Performance Gains
- **Issue**: Current code is already well-optimized
- **Mitigation**: Set realistic expectations (1.5-2x, not 5x)

## Conclusion

The profiling confirms that:
1. TRF algorithm is already well-optimized (good scaling with problem size)
2. Inner loop is the primary candidate for lax.scan optimization
3. Realistic speedup target is 1.5-2x for TRF runtime (not total time)
4. NumPy ↔ JAX conversions offer additional 10-20% improvement
5. Combined optimizations could achieve 2-3x speedup on TRF runtime

**Recommended next step**: Implement inner loop lax.scan conversion as proof of concept.
