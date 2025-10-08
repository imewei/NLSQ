# lax.scan Inner Loop Design Document

## Objective
Convert the TRF inner loop (step acceptance loop) from Python while loop to JAX lax.scan for performance improvement.

## Current Inner Loop Structure

### Location
- `trf_no_bounds`: lines 974-1045
- `trf_bounds`: lines 1305-1377

### Loop Pattern
```python
actual_reduction = -1
inner_loop_count = 0
max_inner_iterations = 100

while (actual_reduction <= 0 and nfev < max_nfev and inner_loop_count < max_inner_iterations):
    inner_loop_count += 1

    # Solve subproblem (conditional on solver type)
    if solver == "cg":
        if inner_loop_count > 1:
            step_h = solve_tr_subproblem_cg(...)
    else:
        step_h, alpha, _n_iter = solve_lsq_trust_region(...)

    # Evaluate quadratic model
    predicted_reduction = evaluate_quadratic(...)

    # Evaluate objective function
    x_new = x + step
    f_new = fun(x_new, ...)
    nfev += 1

    # Check for numerical issues (early continue)
    if not check_isfinite(f_new):
        Delta = 0.25 * step_h_norm
        continue

    # Compute actual reduction
    cost_new = loss_func(f_new)
    actual_reduction = cost - cost_new

    # Update trust region
    Delta_new, ratio = update_tr_radius(...)

    # Check termination (early break)
    termination_status = check_termination(...)
    if termination_status is not None:
        break

    # Update state
    alpha *= Delta / Delta_new
    Delta = Delta_new

    # Exit if successful (early break)
    if actual_reduction > 0:
        break
```

## Challenges

### 1. Early Termination
- **Continue**: When f_new is not finite, reduce Delta and skip rest of iteration
- **Break**: When termination_status is not None OR actual_reduction > 0

### 2. Conditional Logic
- Solver type (cg vs exact)
- First iteration vs subsequent (inner_loop_count > 1)
- Finite check

### 3. State Management
Needs to carry:
- `Delta`: Trust region radius (updated)
- `alpha`: Levenberg-Marquardt parameter (updated)
- `nfev`: Function evaluations (incremented)
- `actual_reduction`: Cost reduction (updated)
- `termination_status`: Termination flag (updated)
- `step_h`: Current step (used outside loop)
- `x_new`, `f_new`, `cost_new`: Results (used outside loop)

## lax.scan Conversion Strategy

### Core Idea
Use lax.scan with fixed max iterations (100), but use masking to handle early termination:
- Continue running all 100 iterations
- Use `should_continue` mask to prevent updates after convergence
- Final state is extracted from the last valid iteration

### Carry State Structure
```python
InnerLoopState = {
    'Delta': float,
    'alpha': float,
    'nfev': int,
    'actual_reduction': float,
    'termination_status': int (or None=-1),
    'should_continue': bool,
    'x_new': array,
    'f_new': array,
    'cost_new': float,
    'step_h': array,
    'step_h_norm': float,
    'inner_count': int,
}
```

### Loop Body Pseudo-code
```python
def inner_loop_body(carry, iteration):
    # Extract carry state
    Delta, alpha, nfev, actual_reduction, termination_status, should_continue = carry[...]

    # Only update if should_continue is True, else return unchanged state
    def do_iteration(state):
        # Solve subproblem
        step_h, alpha_new = solve_subproblem(...)

        # Evaluate
        x_new = x + d * step_h
        f_new = fun(x_new, ...)
        cost_new = loss_func(f_new)
        actual_red = cost - cost_new

        # Check finiteness
        is_finite = check_isfinite(f_new)
        Delta_adj = jnp.where(is_finite, Delta, 0.25 * norm(step_h))

        # Update trust region (only if finite)
        Delta_new, ratio = jnp.where(
            is_finite,
            update_tr_radius(...),
            (Delta_adj, 0.0)
        )

        # Check termination
        term = check_termination(...)

        # Determine if should continue next iteration
        should_cont_next = (
            (actual_red <= 0) &  # Not successful yet
            (term is None) &      # No termination
            is_finite &           # Finite values
            (nfev + 1 < max_nfev)  # Not exceeded max evals
        )

        return {
            'Delta': Delta_new,
            'alpha': alpha_new,
            'nfev': nfev + 1,
            'actual_reduction': actual_red,
            'termination_status': term,
            'should_continue': should_cont_next,
            'x_new': x_new,
            'f_new': f_new,
            'cost_new': cost_new,
            'step_h': step_h,
        }

    def skip_iteration(state):
        # Return unchanged state
        return state

    # Use lax.cond to conditionally execute iteration
    new_state = lax.cond(
        should_continue,
        do_iteration,
        skip_iteration,
        carry
    )

    return new_state, None  # (carry, output)

# Run scan
initial_state = {
    'Delta': Delta,
    'alpha': alpha,
    'nfev': nfev,
    'actual_reduction': -1.0,
    'termination_status': -1,  # Use -1 for None
    'should_continue': True,
    'x_new': x,
    'f_new': f,
    'cost_new': cost,
    'step_h': jnp.zeros(n),
}

final_state, _ = lax.scan(
    inner_loop_body,
    initial_state,
    jnp.arange(max_inner_iterations)
)

# Extract results
Delta = final_state['Delta']
alpha = final_state['alpha']
nfev = final_state['nfev']
actual_reduction = final_state['actual_reduction']
...
```

## Implementation Phases

### Phase 1: Proof of Concept (trf_no_bounds only)
1. Create `_inner_loop_scan` method in TrustRegionReflective class
2. Implement for "exact" solver only (simpler than "cg")
3. Test numerical correctness against original
4. Benchmark performance

### Phase 2: Full Implementation
1. Add "cg" solver support
2. Implement for trf_bounds
3. Comprehensive testing
4. Performance benchmarking

### Phase 3: Optimization
1. Reduce NumPy ↔ JAX conversions
2. Minimize lax.cond overhead
3. Optimize state structure

## Testing Strategy

### Correctness Tests
1. **Exact match**: Compare results with original implementation
   - Same final parameters (within numerical tolerance)
   - Same number of function evaluations
   - Same termination status

2. **Edge cases**:
   - Early termination (1 iteration)
   - Maximum iterations (100 iterations)
   - Non-finite function values
   - Successful step on first iteration
   - Multiple failed steps before success

### Performance Tests
1. **Benchmark**: Use pytest-benchmark suite
2. **Baseline**: Current implementation
3. **Target**: 1.3-1.5x speedup on inner loop iterations
4. **Regression**: Ensure no slowdown on compilation

## Risks and Mitigation

### Risk 1: Increased Compilation Time
- **Mitigation**: Keep loop body simple, avoid complex control flow
- **Fallback**: Provide option to disable lax.scan (use_scan=False)

### Risk 2: Numerical Differences
- **Mitigation**: Extensive testing, use same numerical operations
- **Fallback**: Keep original implementation as default, make lax.scan opt-in

### Risk 3: Complex State Management
- **Mitigation**: Use NamedTuple or dataclass for state, clear documentation
- **Fallback**: Simplify by using separate arrays instead of dict

### Risk 4: Modest Performance Gains
- **Expectation**: Based on profiling, expect 1.3-1.5x (not 2-5x)
- **Mitigation**: Combine with other optimizations (reduce conversions)

## Expected Outcomes

### Performance
- **Conservative**: 1.2-1.3x speedup
- **Target**: 1.3-1.5x speedup
- **Optimistic**: 1.5-2x speedup

### Code Quality
- More functional style (JAX-idiomatic)
- Better JIT compilation
- Potential for further optimization (vectorization)

## Implementation Checklist

- [ ] Create `_inner_loop_scan_exact` method for "exact" solver
- [ ] Define state structure (NamedTuple)
- [ ] Implement loop body function
- [ ] Handle early termination with masking
- [ ] Test against original implementation
- [ ] Benchmark performance
- [ ] Add "cg" solver support
- [ ] Implement for trf_bounds
- [ ] Add use_scan flag for opt-in
- [ ] Update documentation
- [ ] Add tests to test suite

## Next Steps

1. Implement proof of concept for trf_no_bounds with "exact" solver
2. Test numerical correctness
3. Benchmark performance
4. Decide whether to proceed with full implementation based on results
