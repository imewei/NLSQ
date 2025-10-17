# NLSQ Refactoring Progress Report

**Date**: 2025-10-17
**Session**: Sprint Planning & Analysis Complete
**Status**: Ready for Implementation

---

## Executive Summary

✅ **Completed**:
- Comprehensive code quality analysis
- Test suite validation (1213/1213 passing)
- Complexity analysis (identified 3 critical functions)
- Detailed refactoring plan with ROI analysis
- Implementation roadmap with time estimates

🔄 **In Progress**:
- Refactoring plan document created
- Helper method extraction pattern defined

⏳ **Remaining**:
- Implementation of helper methods
- Test validation
- Coverage improvement

---

## Analysis Results

### Test Suite Status
```
✅ Tests Passing: 1213/1213 (100%)
⚠️  Coverage: 77-79% (target: 80%)
⏱️  Test Duration: ~4 minutes
```

### Critical Complexity Issues

| File | Function | LOC | Complexity | Priority |
|------|----------|-----|------------|----------|
| nlsq/trf.py:811 | trf_no_bounds | 354 | 31 | 🔴 CRITICAL |
| nlsq/trf.py:1166 | trf_bounds | 361 | 28 | 🔴 CRITICAL |
| nlsq/trf.py:1646 | trf_no_bounds_timed | 427 | 28 | 🔴 CRITICAL |
| nlsq/large_dataset.py | _fit_chunked | ~150 | 18 | ⚠️ HIGH |
| nlsq/minpack.py | _run_optimization | ~100 | 15 | ⚠️ HIGH |

### File Size Issues

| File | Current LOC | Target | Action Required |
|------|-------------|--------|-----------------|
| trf.py | 2116 | <1000 | Split into package |
| minpack.py | 1772 | <1000 | Split into 2 files |
| large_dataset.py | 1553 | <1000 | Extract utilities |
| least_squares.py | 1344 | <1000 | Extract validators |
| validators.py | 1062 | <1000 | Split validation logic |

---

## Refactoring Strategy

### Phase 1: Complexity Reduction (Priority 1 - THIS SPRINT)

#### Target: `trf_no_bounds` (Complexity 31 → ~12)

**Current Structure** (nlsq/trf.py:811-1164):
```python
class TrustRegionReflective(TrustRegionJITFunctions, TrustRegionOptimizerBase):
    def trf_no_bounds(self, fun, xdata, ydata, jac, ...):
        # Lines 909-959: Initialization (~50 LOC, Complexity ~5)
        x = x0.copy()
        # Setup loss function, scaling, Delta, etc.

        # Lines 960-1146: Main optimization loop (~186 LOC, Complexity ~26)
        while True:
            # Check convergence
            # Lines 1007-1093: Inner loop for step acceptance (~86 LOC, Complexity ~12)
            while actual_reduction <= 0 and ...:
                # Solve subproblem
                # Evaluate step
                # Update trust region

            # Update state after successful step
            # Handle callback

        # Lines 1147-1164: Create and return result (~17 LOC, Complexity ~2)
        return OptimizeResult(...)
```

**Refactored Structure** (Target):
```python
class TrustRegionReflective:
    # NEW HELPER METHODS (to be added before trf_no_bounds)

    def _initialize_trf_state(self, x0, f, J, loss_function, x_scale, ...):
        """Initialize optimization state.

        Complexity: 3 | LOC: ~30

        Returns
        -------
        dict
            Initial state containing x, f, J, cost, g, scale, Delta, etc.
        """
        state = {
            'x': x0.copy(),
            'f': f,
            'J': J,
            'nfev': 1,
            'njev': 1,
            'm': J.shape[0],
            'n': J.shape[1],
        }

        # Apply loss function if provided
        if loss_function is not None:
            rho = loss_function(f, f_scale)
            state['cost'] = self.calculate_cost(rho, data_mask)
            state['J'], state['f'] = self.cJIT.scale_for_robust_loss_function(
                J, f, rho
            )
        else:
            state['cost'] = self.default_loss_func(f)

        # Compute gradient and scaling
        state['g'] = self.compute_grad(state['J'], state['f'])

        jac_scale = isinstance(x_scale, str) and x_scale == "jac"
        if jac_scale:
            state['scale'], state['scale_inv'] = self.cJIT.compute_jac_scale(J)
            state['jac_scale'] = True
        else:
            state['scale'], state['scale_inv'] = x_scale, 1 / x_scale
            state['jac_scale'] = False

        # Initialize trust region radius
        Delta = norm(x0 * state['scale_inv'])
        state['Delta'] = Delta if Delta > 0 else 1.0

        return state

    def _check_convergence_criteria(self, state, gtol):
        """Check if convergence criteria are met.

        Complexity: 2 | LOC: ~10

        Returns
        -------
        int or None
            Termination status (1 if converged, None otherwise)
        """
        g_norm = jnorm(state['g'], ord=jnp.inf)

        if g_norm < gtol:
            self.logger.debug(
                "Convergence: gradient tolerance satisfied",
                g_norm=g_norm,
                gtol=gtol,
            )
            return 1

        return None

    def _solve_trust_region_subproblem(self, state, Delta, alpha, solver):
        """Solve trust region subproblem.

        Complexity: 4 | LOC: ~25

        Returns
        -------
        dict
            Subproblem solution containing step_h, J_h, s, V, uf
        """
        d = state['scale']
        d_jnp = jnp.array(d)
        g_h_jnp = self.compute_grad_hat(state['g'], d_jnp)

        if solver == "cg":
            J_h = state['J'] * d_jnp
            step_h = self.solve_tr_subproblem_cg(
                state['J'], state['f'], d_jnp, Delta, alpha
            )
            return {
                'step_h': step_h,
                'J_h': J_h,
                'g_h': g_h_jnp,
                'd': d,
                'd_jnp': d_jnp,
                's': None,
                'V': None,
                'uf': None,
            }
        else:  # SVD solver
            svd_output = self.svd_no_bounds(state['J'], d_jnp, state['f'])
            J_h = svd_output[0]
            s, V, uf = (np.array(val) for val in svd_output[2:])

            return {
                'J_h': J_h,
                'g_h': g_h_jnp,
                'd': d,
                'd_jnp': d_jnp,
                's': s,
                'V': V,
                'uf': uf,
                'step_h': None,  # Will be computed in inner loop
            }

    def _evaluate_step_acceptance(
        self, state, subproblem, Delta, alpha, ftol, xtol,
        fun, xdata, ydata, data_mask, transform, loss_function, f_scale
    ):
        """Evaluate whether to accept a step and update trust region.

        Complexity: 8 | LOC: ~45

        Returns
        -------
        dict
            Step evaluation results including acceptance decision,
            new state, Delta_new, termination_status
        """
        # Compute step if not already done (SVD solver)
        if subproblem['step_h'] is None:
            step_h, alpha_new, _n_iter = solve_lsq_trust_region(
                state['n'], state['m'],
                subproblem['uf'], subproblem['s'], subproblem['V'],
                Delta, initial_alpha=alpha
            )
        else:
            step_h = subproblem['step_h']
            alpha_new = alpha

        # Compute predicted reduction
        predicted_reduction = -self.cJIT.evaluate_quadratic(
            subproblem['J_h'], subproblem['g_h'], step_h
        )

        # Transform step and evaluate objective
        step = subproblem['d'] * step_h
        x_new = state['x'] + step
        f_new = fun(x_new, xdata, ydata, data_mask, transform)
        step_h_norm = norm(step_h)

        # Check for numerical issues
        if not self.check_isfinite(f_new):
            return {
                'accepted': False,
                'Delta_new': TR_REDUCTION_FACTOR * step_h_norm,
                'alpha_new': alpha,
                'termination_status': None,
                'actual_reduction': -1,
            }

        # Compute actual reduction
        if loss_function is not None:
            cost_new = loss_function(f_new, f_scale, data_mask, cost_only=True)
        else:
            cost_new = self.default_loss_func(f_new)

        actual_reduction = state['cost'] - cost_new

        # Update trust region radius
        Delta_new, ratio = update_tr_radius(
            Delta, actual_reduction, predicted_reduction,
            step_h_norm, step_h_norm > TR_BOUNDARY_THRESHOLD * Delta
        )

        # Check termination criteria
        step_norm = norm(step)
        termination_status = check_termination(
            actual_reduction, state['cost'], step_norm, norm(state['x']),
            ratio, ftol, xtol
        )

        return {
            'accepted': actual_reduction > 0,
            'x_new': x_new,
            'f_new': f_new,
            'cost_new': cost_new,
            'actual_reduction': actual_reduction,
            'Delta_new': Delta_new,
            'alpha_new': alpha * Delta / Delta_new if termination_status is None else alpha,
            'termination_status': termination_status,
            'step_norm': step_norm,
        }

    def _update_state_from_step(self, state, step_result, jac, xdata, ydata,
                                  data_mask, transform, loss_function, f_scale):
        """Update optimization state after accepting a step.

        Complexity: 5 | LOC: ~35

        Returns
        -------
        dict
            Updated state
        """
        new_state = state.copy()
        new_state['x'] = step_result['x_new']
        new_state['f'] = step_result['f_new']
        new_state['cost'] = step_result['cost_new']
        new_state['nfev'] += 1

        # Compute new Jacobian
        new_state['J'] = jac(new_state['x'], xdata, ydata, data_mask, transform)
        new_state['njev'] += 1

        # Apply loss function if needed
        if loss_function is not None:
            rho = loss_function(new_state['f'], f_scale)
            new_state['J'], new_state['f'] = self.cJIT.scale_for_robust_loss_function(
                new_state['J'], new_state['f'], rho
            )

        # Update gradient
        new_state['g'] = self.compute_grad(new_state['J'], new_state['f'])

        # Update Jacobian scaling if needed
        if state['jac_scale']:
            new_state['scale'], new_state['scale_inv'] = self.cJIT.compute_jac_scale(
                new_state['J'], new_state['scale_inv']
            )

        return new_state

    # REFACTORED MAIN FUNCTION
    def trf_no_bounds(self, fun, xdata, ydata, jac, data_mask, transform,
                      x0, f, J, lb, ub, ftol, xtol, gtol, max_nfev,
                      f_scale, x_scale, loss_function, tr_options, verbose,
                      solver="exact", callback=None, **kwargs):
        """Unbounded trust-region reflective algorithm (REFACTORED).

        Complexity: ~12 (down from 31) | LOC: ~80 (down from 354)
        """
        # Initialize optimization state
        state = self._initialize_trf_state(
            x0, f, J, loss_function, x_scale, f_scale, data_mask
        )

        if max_nfev is None:
            max_nfev = x0.size * DEFAULT_MAX_NFEV_MULTIPLIER

        alpha = INITIAL_LEVENBERG_MARQUARDT_LAMBDA
        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None
        Delta = state['Delta']

        self.logger.info(
            "Starting TRF optimization (no bounds)",
            n_params=state['n'], n_residuals=state['m'], max_nfev=max_nfev
        )

        if verbose == 2:
            print_header_nonlinear()

        # Main optimization loop
        with self.logger.timer("optimization", log_result=False):
            while True:
                # Check convergence
                term_status = self._check_convergence_criteria(state, gtol)
                if term_status is not None:
                    termination_status = term_status

                if verbose == 2:
                    g_norm = jnorm(state['g'], ord=jnp.inf)
                    print_iteration_nonlinear(
                        iteration, state['nfev'], state['cost'],
                        actual_reduction, step_norm, g_norm
                    )

                if termination_status is not None or state['nfev'] >= max_nfev:
                    if state['nfev'] >= max_nfev:
                        self.logger.warning("Max function evaluations reached")
                    break

                self.logger.optimization_step(
                    iteration=iteration, cost=state['cost'],
                    gradient_norm=jnorm(state['g'], ord=jnp.inf),
                    step_size=Delta, nfev=state['nfev']
                )

                # Solve trust region subproblem
                subproblem = self._solve_trust_region_subproblem(
                    state, Delta, alpha, solver
                )

                # Inner loop: Try to accept step
                actual_reduction = -1
                for inner_iter in range(100):  # max_inner_iterations
                    step_result = self._evaluate_step_acceptance(
                        state, subproblem, Delta, alpha, ftol, xtol,
                        fun, xdata, ydata, data_mask, transform,
                        loss_function, f_scale
                    )

                    if step_result['termination_status'] is not None:
                        termination_status = step_result['termination_status']
                        break

                    alpha = step_result['alpha_new']
                    Delta = step_result['Delta_new']
                    actual_reduction = step_result['actual_reduction']

                    if actual_reduction > 0:
                        step_norm = step_result['step_norm']
                        break

                # Update state if step was accepted
                if actual_reduction > 0:
                    state = self._update_state_from_step(
                        state, step_result, jac, xdata, ydata,
                        data_mask, transform, loss_function, f_scale
                    )
                else:
                    step_norm = 0
                    actual_reduction = 0

                iteration += 1

                # Handle callback
                if callback is not None:
                    try:
                        callback(
                            iteration=iteration, cost=float(state['cost']),
                            params=np.array(state['x']),
                            info={'gradient_norm': float(jnorm(state['g'], ord=jnp.inf)),
                                  'nfev': state['nfev'], 'step_norm': float(step_norm)}
                        )
                    except StopOptimization:
                        termination_status = 2
                        break
                    except Exception as e:
                        warnings.warn(f"Callback error: {e}", RuntimeWarning)

        if termination_status is None:
            termination_status = 0

        return OptimizeResult(
            x=state['x'], cost=float(state['cost']), fun=state['f'],
            jac=state['J'], grad=np.array(state['g']),
            optimality=float(jnorm(state['g'], ord=jnp.inf)),
            active_mask=np.zeros_like(state['x']),
            nfev=state['nfev'], njev=state['njev'],
            status=termination_status
        )
```

---

## Implementation Checklist

### Step 1: Add Helper Methods to TrustRegionReflective Class
- [ ] Add `_initialize_trf_state()` at line ~800 (before trf_no_bounds)
- [ ] Add `_check_convergence_criteria()` after initialization
- [ ] Add `_solve_trust_region_subproblem()` after convergence check
- [ ] Add `_evaluate_step_acceptance()` after subproblem solver
- [ ] Add `_update_state_from_step()` after step evaluation

### Step 2: Refactor trf_no_bounds
- [ ] Replace initialization block with `_initialize_trf_state()` call
- [ ] Replace convergence check with `_check_convergence_criteria()` call
- [ ] Replace subproblem solving with `_solve_trust_region_subproblem()` call
- [ ] Replace inner loop with `_evaluate_step_acceptance()` calls
- [ ] Replace state update with `_update_state_from_step()` call

### Step 3: Testing
- [ ] Run full test suite: `pytest tests/ -v`
- [ ] Verify all 1213 tests still pass
- [ ] Check no performance regression
- [ ] Verify complexity reduced: `python -c "...complexity check..."`

### Step 4: Apply Same Pattern to trf_bounds
- [ ] Extract similar helpers for bounded version
- [ ] Refactor trf_bounds using same pattern
- [ ] Test and validate

### Step 5: Coverage Improvement
- [ ] Identify uncovered lines in validators.py
- [ ] Add tests for error handling paths
- [ ] Add tests for edge cases in large_dataset.py
- [ ] Target: Reach 80%+ coverage

---

## Metrics Projection

### Before Refactoring
```
Function: trf_no_bounds
├─ LOC: 354
├─ Complexity: 31
├─ Max nesting: 3 levels
├─ Number of responsibilities: 5+
└─ Testability: Low (monolithic)
```

### After Refactoring (Estimated)
```
Function: trf_no_bounds (refactored)
├─ LOC: ~80
├─ Complexity: ~12
├─ Max nesting: 2 levels
└─ Testability: High (uses tested helpers)

Helper Functions:
├─ _initialize_trf_state: LOC ~30, Complexity 3
├─ _check_convergence_criteria: LOC ~10, Complexity 2
├─ _solve_trust_region_subproblem: LOC ~25, Complexity 4
├─ _evaluate_step_acceptance: LOC ~45, Complexity 8
└─ _update_state_from_step: LOC ~35, Complexity 5

Total Improvement:
├─ Max complexity: 31 → 12 (-61%)
├─ Main function LOC: 354 → 80 (-77%)
├─ Maintainability: +300% (estimated)
└─ Testability: +500% (each helper unit-testable)
```

---

## Time Estimates

### This Refactoring (trf_no_bounds only)
- Helper method implementation: 2 hours
- Main function refactoring: 1 hour
- Testing and validation: 1 hour
- **Total: 4 hours**

### Complete Sprint (All 3 tasks)
- trf_no_bounds refactoring: 4 hours
- trf_bounds refactoring: 3 hours (pattern established)
- Coverage tests (+3%): 2 hours
- **Total: 9 hours (~1.2 days)**

### Full Refactoring (All recommendations)
- File splitting (trf.py → package): 2 hours
- File splitting (minpack.py → 2 files): 1.5 hours
- Class-based refactoring (Phase 2): 6 hours
- Additional coverage tests: 2 hours
- **Total Sprint + Phase 2: ~20 hours (2.5 days)**

---

## Next Steps

### Immediate (Next Work Session)
1. Implement the 5 helper methods shown above
2. Refactor trf_no_bounds to use helpers
3. Run test suite and verify
4. Commit with message: "refactor(trf): extract helper methods to reduce complexity from 31 to 12"

### This Sprint
1. Apply same pattern to trf_bounds
2. Add coverage tests for uncovered paths
3. Reach 80%+ coverage target

### Next Sprint
1. Split trf.py into package structure
2. Full class-based refactoring
3. Apply refactoring pattern to large_dataset.py

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Test failures after refactoring | Low | High | Keep original function temporarily, comprehensive testing |
| Performance regression | Very Low | Medium | Benchmark before/after, helpers are pure extraction |
| Breaking backward compatibility | None | High | Refactoring is internal only, API unchanged |
| Incomplete test coverage | Medium | Low | Add tests during refactoring, not after |

---

## Success Criteria

- [x] Analysis complete and documented
- [ ] Helper methods implemented and tested
- [ ] trf_no_bounds complexity < 15 (target: 12)
- [ ] All 1213 tests passing
- [ ] No performance regression (±5%)
- [ ] Coverage ≥ 80%
- [ ] Code review passed
- [ ] Documentation updated

---

## Files Modified

### Created (This Session)
- `REFACTOR_PLAN.md` - High-level refactoring strategy
- `REFACTORING_PROGRESS.md` - This document (detailed implementation guide)

### To Be Modified (Next Session)
- `nlsq/trf.py` - Add helper methods, refactor trf_no_bounds
- `tests/test_trf.py` - Add unit tests for new helpers
- `CLAUDE.md` - Update metrics after refactoring complete

---

**Status**: Ready for implementation
**Next Action**: Implement helper methods as specified above
**Estimated Completion**: 4 hours for trf_no_bounds
**Overall Progress**: Analysis 100%, Planning 100%, Implementation 0%
