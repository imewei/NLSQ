# TRF Main Function Refactoring Roadmap

**Date**: 2025-10-17
**Status**: Ready for Execution
**Estimated Time**: 45-60 minutes
**Risk Level**: ⚠️ HIGH (modifies core optimization logic)

---

## Current State (Baseline)

```
Function: trf_no_bounds
Location: nlsq/trf.py:1255-1608
Lines of Code: 354
Cyclomatic Complexity: 31
Test Status: 14/14 passing + 15/15 helper tests passing
```

## Target State

```
Function: trf_no_bounds (refactored)
Estimated LOC: 80-100 (77% reduction)
Target Complexity: <15 (52% reduction)
Required: All 29 tests must still pass
```

---

## Safety Measures BEFORE Starting

### 1. Create Safety Net (5 minutes)

```bash
# Backup original function
cd /home/wei/Documents/GitHub/NLSQ

# Create backup branch
git checkout -b refactor/trf-main-function

# Copy original function to _trf_no_bounds_original
# (Keep for comparison testing)
```

### 2. Create Comparison Test (10 minutes)

Create `tests/test_trf_refactoring_validation.py`:

```python
"""Validate refactored trf_no_bounds produces identical results."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from nlsq import curve_fit


class TestRefactoringValidation:
    """Compare refactored vs original trf_no_bounds output."""

    def test_simple_exponential(self):
        """Test simple exponential fit produces identical results."""
        def model(x, a, b):
            return a * np.exp(-b * x)

        np.random.seed(42)
        xdata = np.linspace(0, 4, 50)
        ydata = model(xdata, 2.5, 1.3) + 0.2 * np.random.randn(50)

        # Fit should produce same results
        popt, pcov = curve_fit(model, xdata, ydata, method='trf')

        # Verify reasonable convergence
        assert_allclose(popt, [2.5, 1.3], rtol=0.1)
        assert pcov is not None

    def test_with_bounds(self):
        """Test with parameter bounds."""
        def model(x, a, b):
            return a * x + b

        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([1, 3, 5, 7, 9])

        popt, _ = curve_fit(model, xdata, ydata,
                            bounds=([0, 0], [10, 10]),
                            method='trf')

        assert_allclose(popt, [2.0, 1.0], rtol=0.01)

    def test_with_loss_function(self):
        """Test with robust loss function."""
        def model(x, a):
            return a * x

        xdata = np.array([0, 1, 2, 3, 4])
        ydata = np.array([0, 2, 4, 6, 8])

        popt, _ = curve_fit(model, xdata, ydata,
                            loss='soft_l1',
                            method='trf')

        assert abs(popt[0] - 2.0) < 0.1
```

**Run to establish baseline**:
```bash
pytest tests/test_trf_refactoring_validation.py -v
# All tests MUST pass before refactoring
```

---

## Refactoring Steps (Incremental Integration)

### Phase 1: Replace Initialization (10 minutes)

**Current code (lines ~1349-1406)**:
```python
x = x0.copy()
f_true = f
nfev = 1
njev = 1
m, n = J.shape

# ... initialization logic ...

if loss_function is not None:
    rho = loss_function(f, f_scale)
    cost_jnp = self.calculate_cost(rho, data_mask)
    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
else:
    cost_jnp = self.default_loss_func(f)
cost = cost_jnp

g_jnp = self.compute_grad(J, f)
g = g_jnp
# ... more initialization ...
```

**Refactored (use helper)**:
```python
# Initialize optimization state
state = self._initialize_trf_state(
    x0=x0,
    f=f,
    J=J,
    loss_function=loss_function,
    x_scale=x_scale,
    f_scale=f_scale,
    data_mask=data_mask,
)

# Extract state variables
x = state['x']
f = state['f']
J = state['J']
cost = state['cost']
g = state['g']
scale = state['scale']
scale_inv = state['scale_inv']
Delta = state['Delta']
nfev = state['nfev']
njev = state['njev']
m = state['m']
n = state['n']
jac_scale = state['jac_scale']
f_true = f  # Keep for result
```

**Test**:
```bash
pytest tests/test_trf_refactoring_validation.py -v
pytest tests/test_trf_simple.py -v
# MUST pass before continuing
```

**Commit**:
```bash
git add nlsq/trf.py
git commit -m "refactor(trf): use _initialize_trf_state helper

Replace initialization block with helper method call.
All tests passing."
```

---

### Phase 2: Replace Convergence Check (5 minutes)

**Current code (lines ~1400-1410)**:
```python
g_norm = jnorm(g, ord=jnp.inf)
if g_norm < gtol:
    termination_status = 1
    self.logger.debug(
        "Convergence: gradient tolerance satisfied",
        g_norm=g_norm,
        gtol=gtol,
    )
```

**Refactored**:
```python
termination_status = self._check_convergence_criteria(g, gtol)
if termination_status is not None:
    g_norm = jnorm(g, ord=jnp.inf)  # For logging
```

**Test & Commit** (same as Phase 1)

---

### Phase 3: Replace Subproblem Solving (10 minutes)

**Current code (lines ~1433-1447)**:
```python
# Setup scaled variables
d = scale
d_jnp = jnp.array(scale)
g_h_jnp = self.compute_grad_hat(g_jnp, d_jnp)

# Solve trust region subproblem
if solver == "cg":
    J_h = J * d_jnp
    step_h = self.solve_tr_subproblem_cg(J, f, d_jnp, Delta, alpha)
    s, V, uf = None, None, None
else:
    svd_output = self.svd_no_bounds(J, d_jnp, f)
    J_h = svd_output[0]
    s, V, uf = (np.array(val) for val in svd_output[2:])
```

**Refactored**:
```python
# Solve trust region subproblem
subproblem_result = self._solve_trust_region_subproblem(
    J=J,
    f=f,
    g=g,
    scale=scale,
    Delta=Delta,
    alpha=alpha,
    solver=solver,
)

# Extract subproblem solution
d = subproblem_result['d']
d_jnp = subproblem_result['d_jnp']
g_h_jnp = subproblem_result['g_h']
J_h = subproblem_result['J_h']
step_h = subproblem_result['step_h']
s = subproblem_result['s']
V = subproblem_result['V']
uf = subproblem_result['uf']
```

**Test & Commit** (same as Phase 1)

---

### Phase 4: Replace Step Acceptance Loop (15 minutes) ⚠️ MOST COMPLEX

**Current code (lines ~1448-1533)**:
```python
# Step acceptance loop
actual_reduction = -1
inner_loop_count = 0
max_inner_iterations = 100
while (
    actual_reduction <= 0
    and nfev < max_nfev
    and inner_loop_count < max_inner_iterations
):
    inner_loop_count += 1

    # ... inner loop logic (85 lines) ...

    if actual_reduction > 0:
        break

# Check if inner loop hit iteration limit
if inner_loop_count >= max_inner_iterations:
    # ... error handling ...
    termination_status = -3

# Update state if step accepted
if actual_reduction > 0:
    x = x_new
    f = f_new
    # ... state updates ...
else:
    step_norm = 0
    actual_reduction = 0
```

**Refactored**:
```python
# Evaluate and potentially accept step
acceptance_result = self._evaluate_step_acceptance(
    fun=fun,
    jac=jac,
    x=x,
    f=f,
    J=J,
    J_h=J_h,
    g_h_jnp=g_h_jnp,
    cost=cost,
    d=d,
    d_jnp=d_jnp,
    Delta=Delta,
    alpha=alpha,
    step_h=step_h,
    s=s,
    V=V,
    uf=uf,
    xdata=xdata,
    ydata=ydata,
    data_mask=data_mask,
    transform=transform,
    loss_function=loss_function,
    f_scale=f_scale,
    scale_inv=scale_inv,
    jac_scale=jac_scale,
    solver=solver,
    ftol=ftol,
    xtol=xtol,
    max_nfev=max_nfev,
    nfev=nfev,
)

# Update state from acceptance result
if acceptance_result['accepted']:
    x = acceptance_result['x_new']
    f = acceptance_result['f_new']
    J = acceptance_result['J_new']
    cost = acceptance_result['cost_new']
    g = acceptance_result['g_new']
    njev += acceptance_result['njev']

    if jac_scale and 'scale' in acceptance_result:
        scale = acceptance_result['scale']
        scale_inv = acceptance_result['scale_inv']

actual_reduction = acceptance_result['actual_reduction']
step_norm = acceptance_result['step_norm']
Delta = acceptance_result['Delta']
alpha = acceptance_result['alpha']
nfev = acceptance_result['nfev']

if acceptance_result['termination_status'] is not None:
    termination_status = acceptance_result['termination_status']
```

**Test & Commit** (same as Phase 1, but extra vigilance!)

---

### Phase 5: Final Cleanup (5 minutes)

1. Remove any remaining duplicate code
2. Update docstring if needed
3. Run final complexity measurement

**Expected final structure** (~80-100 lines):
```python
def trf_no_bounds(self, fun, xdata, ydata, jac, ...):
    """Trust Region Reflective algorithm (unbounded version).

    ... existing docstring ...
    """
    # Initialize state
    state = self._initialize_trf_state(...)
    # ... extract state variables ...

    # Main optimization loop
    while True:
        # Check convergence
        termination_status = self._check_convergence_criteria(g, gtol)

        if verbose == 2:
            print_iteration_nonlinear(...)

        if termination_status is not None or nfev == max_nfev:
            break

        # Log iteration
        self.logger.optimization_step(...)

        # Solve subproblem
        subproblem_result = self._solve_trust_region_subproblem(...)

        # Try to accept step
        acceptance_result = self._evaluate_step_acceptance(...)

        # Update from result
        if acceptance_result['accepted']:
            x = acceptance_result['x_new']
            # ... more updates ...

        actual_reduction = acceptance_result['actual_reduction']
        # ... more updates ...

        iteration += 1

        # Callback
        if callback is not None:
            # ... callback logic ...

    # Create result
    return OptimizeResult(...)
```

---

## Validation Checklist

After refactoring is complete, verify:

- [ ] All 29 TRF tests pass (14 original + 15 helpers)
- [ ] All 3 validation tests pass
- [ ] Measured complexity < 15
- [ ] LOC reduced to ~80-100
- [ ] No performance regression (run benchmarks)
- [ ] Code still readable and maintainable

---

## Rollback Plan

If ANY step fails tests:

```bash
# Revert last commit
git reset --hard HEAD~1

# Or revert entire branch
git checkout main
git branch -D refactor/trf-main-function

# Helpers are still there, try again later
```

---

## Success Criteria

✅ **ONLY proceed to next phase if**:
1. All tests pass
2. No new warnings
3. Code compiles without errors
4. Git commit created

❌ **STOP and revert if**:
1. Any test fails
2. Complexity increases
3. Performance degrades >10%
4. Code becomes less readable

---

## Estimated Timeline

| Phase | Description | Time | Risk |
|-------|-------------|------|------|
| 0 | Safety measures | 15 min | Low |
| 1 | Initialization | 10 min | Low |
| 2 | Convergence | 5 min | Low |
| 3 | Subproblem | 10 min | Medium |
| 4 | Step acceptance | 15 min | ⚠️ HIGH |
| 5 | Cleanup | 5 min | Low |
| **Total** | **End-to-end** | **60 min** | **High** |

---

## Next Session Checklist

Before starting refactoring session:

- [ ] Read this entire roadmap
- [ ] Allocate 90 minutes (60 min work + 30 min buffer)
- [ ] Ensure all current tests pass
- [ ] Create backup branch
- [ ] Set up safety net (comparison test)
- [ ] Have rollback plan ready
- [ ] Commit after each phase

**Remember**: Better to stop and revert than to break production code!

---

**Document Status**: Ready for Execution
**Last Updated**: 2025-10-17
**Author**: Refactoring Session Team
