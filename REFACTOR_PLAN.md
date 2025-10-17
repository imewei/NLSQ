# TRF Refactoring Plan

## Current State
- `trf_no_bounds`: 354 LOC, Complexity 31
- Nested loops (outer optimization + inner step acceptance)
- Multiple responsibilities mixed together

## Refactoring Strategy

### Phase 1: Extract Helper Methods (This Session)
Create 5 helper methods to reduce complexity from 31 to ~12-15:

1. **`_initialize_trf_state()`** (~30 LOC, Complexity 3)
   - Initialize variables (x, f, J, cost, g, scale, Delta, etc.)
   - Set up loss function if provided
   - Return initial state dictionary

2. **`_check_convergence_criteria()`** (~15 LOC, Complexity 2)
   - Check gradient norm vs gtol
   - Check max function evaluations
   - Return termination status

3. **`_solve_trust_region_subproblem()`** (~30 LOC, Complexity 4)
   - Handle CG vs SVD solver selection
   - Compute scaled Jacobian and gradient
   - Solve subproblem
   - Return step_h, J_h, and solver outputs

4. **`_evaluate_step_acceptance()`** (~50 LOC, Complexity 8)
   - Evaluate new point
   - Compute actual vs predicted reduction
   - Update trust region radius
   - Check termination
   - Return acceptance decision and new state

5. **`_update_state_from_step()`** (~40 LOC, Complexity 5)
   - Update x, f, J, g
   - Handle loss function scaling
   - Update Jacobian scaling if needed
   - Return updated state

### Main Loop Structure (After Refactoring)
```python
def trf_no_bounds(self, ...):
    state = self._initialize_trf_state(...)

    while True:
        term_status = self._check_convergence_criteria(state)
        if term_status is not None or state['nfev'] >= max_nfev:
            break

        # Solve subproblem
        step_info = self._solve_trust_region_subproblem(state, ...)

        # Inner loop: Try to accept step
        for inner_iter in range(max_inner_iterations):
            accept_info = self._evaluate_step_acceptance(
                state, step_info, ...
            )

            if accept_info['termination_status'] is not None:
                term_status = accept_info['termination_status']
                break

            if accept_info['actual_reduction'] > 0:
                state = self._update_state_from_step(accept_info, ...)
                break
            else:
                # Adjust trust region and retry
                step_info = self._solve_trust_region_subproblem(
                    state, Delta=accept_info['Delta_new']
                )

        # Handle callback
        if callback is not None:
            ...

    return OptimizeResult(...)
```

## Complexity Reduction Estimate
- **Before**: Main function = 31
- **After**:
  - Main function = ~12 (outer loop + callback handling)
  - Helper 1 = 3
  - Helper 2 = 2
  - Helper 3 = 4
  - Helper 4 = 8
  - Helper 5 = 5
- **Maximum per function**: 12 (vs target <10, close!)

## Testing Strategy
1. Keep original function as `trf_no_bounds_original` temporarily
2. Create new `trf_no_bounds` with helpers
3. Add integration test comparing outputs
4. Run full test suite
5. Remove original after validation

## Phase 2: Full Class-Based Refactoring (Next Sprint)
- Convert to `TRFSolverNoBounds` class
- Use dataclass for optimization state
- Further extract to strategy pattern for solver selection
- Target complexity <8 per method

## Estimated Time
- Phase 1 (this session): 2-3 hours
- Phase 2 (next sprint): 4-6 hours
