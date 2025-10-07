# 🔧 Critical Refactoring Implementation Plan

**Date**: 2025-10-07
**Based On**: COMPREHENSIVE_QUALITY_REPORT.md
**Total Estimated Effort**: 52 hours (6.5 days)
**Approach**: Phased implementation with quick wins first

---

## 📊 Scope Analysis

### Critical Issues to Address

| Issue | Complexity | Lines | Priority | Effort | Risk |
|-------|------------|-------|----------|--------|------|
| `validate_curve_fit_inputs()` | 73 | 305 | 🔴 HIGH | 8h | MEDIUM |
| `curve_fit()` | 68 | 600 | 🔴 HIGH | 12h | HIGH |
| `least_squares()` | 55 | 349 | 🔴 HIGH | 10h | HIGH |
| `update_function()` (depth 11) | 12 | 203 | 🔴 HIGH | 3h | MEDIUM |
| `masked_residual_func()` (depth 11) | 12 | 133 | 🔴 HIGH | 3h | MEDIUM |
| Security warnings (8) | - | - | 🟡 MED | 2h | LOW |
| Magic number constants | - | - | 🟡 MED | 4h | LOW |
| Edge case tests | - | - | 🔴 HIGH | 16h | LOW |
| **TOTAL** | - | - | - | **58h** | - |

---

## 🎯 Implementation Strategy

### Phase 1: Quick Wins (5 hours) ⚡ **START HERE**

**Objective**: Immediate improvements with low risk

#### 1.1 Fix Security Warnings (2 hours) 🔒
**Files**: recovery.py, streaming_optimizer.py, large_dataset.py, validators.py
**Issue**: 8 instances (6× S110 try-except-pass, 2× S101 assert)

**Implementation**:
```python
# S110: Add logging to try-except-pass blocks
# Before:
try:
    risky_operation()
except Exception:
    pass

# After:
from nlsq.logging import get_logger

logger = get_logger(__name__)

try:
    risky_operation()
except Exception as e:
    logger.debug(f"Non-critical error in {operation_name}: {e}")
    pass  # Intentional fallback

# S101: Replace assert with explicit validation
# Before:
assert x > 0, "x must be positive"

# After:
if x <= 0:
    raise ValueError("x must be positive")
```

**Files to modify**:
- [ ] `nlsq/recovery.py` (3 instances)
- [ ] `nlsq/streaming_optimizer.py` (2 instances)
- [ ] `nlsq/large_dataset.py` (1 instance)
- [ ] `nlsq/validators.py` (2 assert statements)

**Testing**: Run `pytest tests/ -v` after changes

---

#### 1.2 Extract Magic Number Constants (1-2 hours) 📏
**File**: `nlsq/trf.py` (primary), `nlsq/least_squares.py` (secondary)

**Implementation**:
```python
# Create constants module: nlsq/constants.py
"""
Constants for NLSQ optimization algorithms.

These values are derived from:
- SciPy's trust-region algorithms
- Numerical optimization best practices
- JAX performance characteristics
"""

# Trust Region Reflective (TRF) Algorithm Constants
DEFAULT_MAX_NFEV_MULTIPLIER = 100  # Max function evaluations per parameter
STEP_ACCEPTANCE_THRESHOLD = 0.5  # Trust region step acceptance ratio
STEP_QUALITY_EXCELLENT = 0.75  # Excellent step quality threshold
STEP_QUALITY_GOOD = 0.25  # Good step quality threshold

INITIAL_TRUST_RADIUS = 1.0  # Initial trust region radius
MAX_TRUST_RADIUS = 1000.0  # Maximum trust region radius
MIN_TRUST_RADIUS = 1e-10  # Minimum trust region radius

INITIAL_LEVENBERG_MARQUARDT_LAMBDA = 0.0  # Initial LM damping parameter

# Convergence tolerances
DEFAULT_FTOL = 1e-8  # Function tolerance
DEFAULT_XTOL = 1e-8  # Parameter tolerance
DEFAULT_GTOL = 1e-8  # Gradient tolerance

# Algorithm selection thresholds
SMALL_DATASET_THRESHOLD = 1000  # Switch to different algorithms
LARGE_DATASET_THRESHOLD = 1_000_000  # Use chunking/streaming

# Numerical stability
MIN_POSITIVE_VALUE = 1e-15  # Minimum positive value for numerical stability
MAX_CONDITION_NUMBER = 1e12  # Maximum matrix condition number
```

**Usage in trf.py**:
```python
# Before:
max_nfev = x0.size * 100
alpha = 0.0
step_threshold = 0.5

# After:
from nlsq.constants import (
    DEFAULT_MAX_NFEV_MULTIPLIER,
    INITIAL_LEVENBERG_MARQUARDT_LAMBDA,
    STEP_ACCEPTANCE_THRESHOLD,
)

max_nfev = x0.size * DEFAULT_MAX_NFEV_MULTIPLIER
alpha = INITIAL_LEVENBERG_MARQUARDT_LAMBDA
step_threshold = STEP_ACCEPTANCE_THRESHOLD
```

**Files to modify**:
- [ ] Create `nlsq/constants.py` (new file)
- [ ] Update `nlsq/trf.py` (import and use constants)
- [ ] Update `nlsq/least_squares.py` (import and use constants)
- [ ] Update `nlsq/__init__.py` (export constants for users)

**Testing**: Run `pytest tests/test_trf*.py tests/test_least_squares.py -v`

---

#### 1.3 Start Validator Refactoring (2 hours) 🔨
**File**: `nlsq/validators.py`
**Function**: `validate_curve_fit_inputs()` (complexity 73 → goal: <20)

**Strategy**: Extract type/shape validation into separate functions

**Implementation**:
```python
# Step 1: Extract type validation
def _validate_and_convert_arrays(
    xdata: Any, ydata: Any
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Validate and convert xdata/ydata to numpy arrays.

    Returns
    -------
    errors : list
        Error messages
    warnings : list
        Warning messages
    xdata_clean : np.ndarray or tuple
        Cleaned xdata
    ydata_clean : np.ndarray
        Cleaned ydata
    """
    errors = []
    warnings = []

    # Handle tuple xdata (for multi-dimensional fitting)
    if isinstance(xdata, tuple):
        try:
            n_points = len(xdata[0]) if len(xdata) > 0 else 0
            for i, x_arr in enumerate(xdata):
                if len(x_arr) != n_points:
                    errors.append("All arrays in xdata tuple must have same length")
                    break
            warnings.append(f"xdata is tuple with {len(xdata)} arrays")
        except Exception as e:
            errors.append(f"Invalid xdata tuple: {e}")
            return errors, warnings, xdata, ydata
    else:
        # Convert to numpy arrays
        try:
            if not isinstance(xdata, (np.ndarray, jnp.ndarray)):
                xdata = np.asarray(xdata)
                warnings.append("xdata converted to numpy array")
        except Exception as e:
            errors.append(f"Cannot convert xdata to array: {e}")
            return errors, warnings, xdata, ydata

        # Check dimensions
        if xdata.ndim == 0:
            errors.append("xdata must be at least 1-dimensional")

    # Convert ydata
    try:
        if not isinstance(ydata, (np.ndarray, jnp.ndarray)):
            ydata = np.asarray(ydata)
            warnings.append("ydata converted to numpy array")
    except Exception as e:
        errors.append(f"Cannot convert ydata to array: {e}")
        return errors, warnings, xdata, ydata

    if ydata.ndim == 0:
        errors.append("ydata must be at least 1-dimensional")

    return errors, warnings, xdata, ydata


# Step 2: Extract shape validation
def _validate_data_shapes(
    xdata: np.ndarray | tuple, ydata: np.ndarray
) -> tuple[list[str], list[str], int]:
    """Validate that xdata and ydata have compatible shapes.

    Returns
    -------
    errors : list
        Error messages
    warnings : list
        Warning messages
    n_points : int
        Number of data points
    """
    errors = []
    warnings = []

    # Get number of points from xdata
    if isinstance(xdata, tuple):
        n_points = len(xdata[0]) if len(xdata) > 0 else 0
    elif xdata.ndim == 2:
        n_points = xdata.shape[0]
        n_vars = xdata.shape[1]
        warnings.append(f"xdata has {n_vars} independent variables")
    else:
        n_points = len(xdata) if hasattr(xdata, "__len__") else 1

    # Check shapes match
    if len(ydata) != n_points:
        errors.append(
            f"xdata ({n_points} points) and ydata ({len(ydata)} points) "
            f"must have same length"
        )

    return errors, warnings, n_points


# Step 3: Updated main function (much simpler)
def validate_curve_fit_inputs(
    self,
    f: Callable,
    xdata: Any,
    ydata: Any,
    p0: Any | None = None,
    bounds: tuple | None = None,
    sigma: Any | None = None,
    absolute_sigma: bool = True,
    check_finite: bool = True,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Validate inputs for curve_fit function."""

    # Step 1: Type and conversion validation
    errors, warnings, xdata, ydata = _validate_and_convert_arrays(xdata, ydata)
    if errors:
        return errors, warnings, xdata, ydata

    # Step 2: Shape validation
    shape_errors, shape_warnings, n_points = _validate_data_shapes(xdata, ydata)
    errors.extend(shape_errors)
    warnings.extend(shape_warnings)
    if errors:
        return errors, warnings, xdata, ydata

    # Step 3: Finite value validation
    if check_finite:
        finite_errors, finite_warnings = _validate_finite_values(xdata, ydata)
        errors.extend(finite_errors)
        warnings.extend(finite_warnings)

    # ... continue with remaining validation logic
    # (bounds, sigma, p0, etc. - to be extracted in future phases)

    return errors, warnings, xdata, ydata
```

**Complexity Reduction**: 73 → ~35 (after Phase 1), → <20 (after full refactoring)

**Files to modify**:
- [ ] `nlsq/validators.py` (extract helper functions)

**Testing**: Run `pytest tests/test_validators.py tests/test_minpack.py -v`

---

### Phase 2: Core Function Refactoring (30 hours) 🏗️

**Objective**: Refactor the 3 most complex functions

#### 2.1 Complete validate_curve_fit_inputs() Refactoring (6 hours)
**Current**: Complexity 73, 305 lines
**Target**: Complexity <20, ~150 lines

**Remaining extractions after Phase 1**:
```python
# Extract bounds validation (complexity ~15)
def _validate_bounds(
    bounds: tuple | None, n_params: int
) -> tuple[list[str], list[str], tuple | None]:
    """Validate parameter bounds."""
    errors = []
    warnings = []

    if bounds is None:
        return errors, warnings, (-np.inf, np.inf)

    # Validate bounds structure
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        errors.append("bounds must be a tuple of (lower, upper)")
        return errors, warnings, bounds

    # ... detailed bounds validation logic

    return errors, warnings, bounds


# Extract sigma validation (complexity ~10)
def _validate_sigma(
    sigma: Any | None, n_points: int, ydata_shape: tuple
) -> tuple[list[str], list[str], np.ndarray | None]:
    """Validate uncertainty (sigma) parameters."""
    # ... sigma validation logic


# Extract p0 validation (complexity ~8)
def _validate_initial_guess(
    p0: Any | None, f: Callable, xdata: np.ndarray, n_params_expected: int | None = None
) -> tuple[list[str], list[str], np.ndarray | None]:
    """Validate initial parameter guess."""
    # ... p0 validation logic


# Extract finite value validation (complexity ~5)
def _validate_finite_values(
    xdata: np.ndarray, ydata: np.ndarray
) -> tuple[list[str], list[str]]:
    """Check for NaN/Inf values in data."""
    errors = []
    warnings = []

    if not np.all(np.isfinite(xdata)):
        errors.append("xdata contains NaN or Inf values")

    if not np.all(np.isfinite(ydata)):
        errors.append("ydata contains NaN or Inf values")

    return errors, warnings
```

**Result**: Complexity 73 → ~15-18 (5 focused helper functions)

---

#### 2.2 Refactor curve_fit() (12 hours)
**File**: `nlsq/minpack.py:394`
**Current**: Complexity 68, 600 lines
**Target**: Complexity <20, ~200 lines

**Strategy**: Extract algorithm selection, execution, and post-processing

```python
# Extract validation (complexity ~15)
def _validate_and_prepare_inputs(
    f: Callable, xdata: Any, ydata: Any, p0: Any, **kwargs
) -> ValidatedInputs:
    """Validate inputs and prepare for fitting."""
    # Use existing validators.validate_curve_fit_inputs()
    # Convert to JAX arrays
    # Set up defaults
    return ValidatedInputs(...)


# Extract algorithm selection (complexity ~12)
def _select_fitting_algorithm(
    inputs: ValidatedInputs, method: str, ftol: float, **kwargs
) -> AlgorithmConfig:
    """Select and configure fitting algorithm."""
    # Decide between: TRF, Levenberg-Marquardt, dogbox
    # Configure algorithm parameters
    # Set up loss function
    return AlgorithmConfig(...)


# Extract fit execution (complexity ~10)
def _execute_fit(
    f: Callable, inputs: ValidatedInputs, config: AlgorithmConfig
) -> FitResult:
    """Execute the actual curve fitting."""
    # Call appropriate algorithm
    # Handle convergence
    # Collect diagnostics
    return FitResult(...)


# Extract result processing (complexity ~8)
def _process_fit_result(
    result: FitResult, inputs: ValidatedInputs, full_output: bool
) -> tuple | FitResult:
    """Process and format fit results."""
    # Compute covariance matrix
    # Calculate uncertainties
    # Format output
    return popt, pcov, ...


# Main function (much simpler, complexity ~15)
def curve_fit(f: Callable, xdata: Any, ydata: Any, p0: Any = None, **kwargs) -> tuple:
    """Fit a curve to data using nonlinear least squares."""

    # Validate and prepare
    inputs = _validate_and_prepare_inputs(f, xdata, ydata, p0, **kwargs)

    # Select algorithm
    config = _select_fitting_algorithm(inputs, method, ftol, **kwargs)

    # Execute fit
    result = _execute_fit(f, inputs, config)

    # Process and return results
    return _process_fit_result(result, inputs, full_output)
```

**Result**: Complexity 68 → ~15 (4 focused helper functions + main coordinator)

---

#### 2.3 Refactor least_squares() (10 hours)
**File**: `nlsq/least_squares.py:333`
**Current**: Complexity 55, 349 lines
**Target**: Complexity <20, ~150 lines

**Strategy**: Similar to curve_fit() - extract validation, setup, execution, processing

```python
# Extract Jacobian setup (complexity ~12)
def _setup_jacobian(
    fun: Callable, jac: Any, x0: np.ndarray, **kwargs
) -> JacobianConfig:
    """Set up Jacobian computation strategy."""
    # Decide: analytical, finite-diff, or auto-diff
    # Configure sparsity if applicable
    # Set up caching
    return JacobianConfig(...)


# Extract loss function setup (complexity ~8)
def _setup_loss_function(loss: str, f_scale: float) -> LossFunction:
    """Configure robust loss function."""
    # Select loss function (linear, soft_l1, huber, etc.)
    # Set scaling parameters
    return LossFunction(...)


# Extract optimizer selection (complexity ~10)
def _select_optimizer(method: str, bounds: tuple, problem_size: int) -> Optimizer:
    """Select appropriate optimization algorithm."""
    # Choose: trf, dogbox, lm
    # Configure for problem characteristics
    return Optimizer(...)


# Main function (complexity ~20)
def least_squares(fun: Callable, x0: np.ndarray, **kwargs) -> OptimizeResult:
    """Solve nonlinear least squares problem."""

    # Validate inputs (use existing validator)
    inputs = _validate_least_squares_inputs(fun, x0, **kwargs)

    # Setup problem
    jac_config = _setup_jacobian(fun, jac, x0, **kwargs)
    loss_func = _setup_loss_function(loss, f_scale)
    optimizer = _select_optimizer(method, bounds, len(x0))

    # Execute optimization
    result = optimizer.run(fun, x0, jac_config, loss_func, **kwargs)

    # Post-process
    return _finalize_result(result, inputs)
```

**Result**: Complexity 55 → ~18-20

---

### Phase 3: Deep Nesting Reduction (6 hours) 🌳

**Objective**: Reduce nesting depth from 11 to <5

#### 3.1 Refactor update_function() (3 hours)
**File**: `nlsq/least_squares.py:700`
**Current**: Depth 11, Complexity 12, 203 lines
**Target**: Depth <5, Complexity <10

**Strategy**: Early returns + extract nested blocks

```python
# Before: Depth 11
def update_function(*args, **kwargs):
    if condition1:
        if condition2:
            if condition3:
                # ... 8 more levels of nesting
                return result
    return None


# After: Depth 3-4 with early returns
def update_function(*args, **kwargs):
    # Early exit for special cases
    if not condition1:
        return _handle_case_1(*args, **kwargs)

    if not condition2:
        return _handle_case_2(*args, **kwargs)

    if not condition3:
        return _handle_case_3(*args, **kwargs)

    # Main logic at reduced depth
    return _handle_normal_case(*args, **kwargs)


# Extract deeply nested blocks
def _handle_case_1(*args, **kwargs):
    """Handle special case 1."""
    # Logic that was at depth 5-11
    return result


def _handle_case_2(*args, **kwargs):
    """Handle special case 2."""
    return result


def _handle_normal_case(*args, **kwargs):
    """Handle normal optimization path."""
    # Main logic, now at depth 2-3
    return result
```

**Result**: Depth 11 → 3-4, Complexity 12 → 8-10

---

#### 3.2 Refactor masked_residual_func() (3 hours)
**File**: `nlsq/least_squares.py:720`
**Current**: Depth 11, Complexity 12, 133 lines
**Target**: Depth <5, Complexity <10

**Strategy**: Same approach as update_function()

---

### Phase 4: Test Coverage Expansion (16 hours) ✅

**Objective**: Increase coverage from 70% to 80%

#### 4.1 Recovery Module Edge Cases (4 hours)
**File**: `tests/test_recovery.py`

```python
# Add tests for edge cases
def test_recovery_with_nan_residuals():
    """Test recovery when residuals contain NaN."""
    # ... test implementation


def test_recovery_with_singular_jacobian():
    """Test recovery when Jacobian is singular."""
    # ... test implementation


def test_recovery_with_memory_exceeded():
    """Test recovery when memory limit is exceeded."""
    # ... test implementation


def test_recovery_chain_multiple_failures():
    """Test recovery with multiple sequential failures."""
    # ... test implementation
```

---

#### 4.2 Memory Manager Edge Cases (4 hours)
**File**: `tests/test_memory_manager.py`

```python
def test_memory_limit_exactly_at_boundary():
    """Test behavior when memory usage exactly equals limit."""
    # ... test implementation


def test_memory_limit_with_gradual_growth():
    """Test memory tracking with gradual memory growth."""
    # ... test implementation


def test_memory_limit_with_spike():
    """Test memory limit with sudden spike."""
    # ... test implementation
```

---

#### 4.3 Streaming Optimizer Edge Cases (4 hours)
**File**: `tests/test_streaming_optimizer.py`

```python
def test_streaming_with_corrupted_chunk():
    """Test streaming when one chunk is corrupted."""
    # ... test implementation


def test_streaming_with_unequal_chunks():
    """Test streaming with unequal chunk sizes."""
    # ... test implementation


def test_streaming_interruption_and_resume():
    """Test streaming interruption and resume capability."""
    # ... test implementation
```

---

#### 4.4 Sparse Jacobian Edge Cases (4 hours)
**File**: `tests/test_sparse_jacobian.py`

```python
def test_sparse_jacobian_all_zeros():
    """Test sparse Jacobian when all entries are zero."""
    # ... test implementation


def test_sparse_jacobian_pattern_mismatch():
    """Test when actual sparsity doesn't match pattern."""
    # ... test implementation


def test_sparse_jacobian_memory_efficiency():
    """Verify sparse storage is more memory-efficient."""
    # ... test implementation
```

---

## 📅 Recommended Implementation Timeline

### Week 1: Quick Wins + Start Core Refactoring
**Days 1-2** (8-10 hours):
- [ ] Complete Phase 1 (security warnings, constants, start validator refactoring)
- [ ] Run full test suite after each change
- [ ] Commit incrementally

**Days 3-5** (12-15 hours):
- [ ] Complete validator refactoring (Phase 2.1)
- [ ] Start curve_fit() refactoring (Phase 2.2 partial)
- [ ] Add some edge case tests (Phase 4 partial)

### Week 2: Core Function Refactoring
**Days 6-8** (15-18 hours):
- [ ] Complete curve_fit() refactoring
- [ ] Complete least_squares() refactoring
- [ ] Add more edge case tests

**Days 9-10** (10-12 hours):
- [ ] Fix deep nesting (Phase 3)
- [ ] Complete edge case tests (Phase 4)
- [ ] Final testing and documentation

---

## ✅ Success Criteria

| Metric | Before | Target | How to Measure |
|--------|--------|--------|----------------|
| Complexity (validate_curve_fit_inputs) | 73 | <20 | AST analysis script |
| Complexity (curve_fit) | 68 | <20 | AST analysis script |
| Complexity (least_squares) | 55 | <20 | AST analysis script |
| Nesting depth (update_function) | 11 | <5 | AST analysis script |
| Nesting depth (masked_residual_func) | 11 | <5 | AST analysis script |
| Security warnings | 8 | 0 | `ruff check --select S` |
| Magic numbers | ~45 | <10 | Manual review |
| Test coverage | 70% | 80% | `pytest --cov` |
| All tests passing | ✅ | ✅ | `pytest tests/` |

---

## 🔒 Safety Protocols

### Before Any Refactoring:
1. ✅ Commit current working state
2. ✅ Run full test suite (baseline)
3. ✅ Create feature branch: `git checkout -b refactor-critical-issues`

### During Refactoring:
1. ✅ Make small, incremental changes
2. ✅ Run relevant tests after each change
3. ✅ Commit frequently with clear messages
4. ✅ If tests fail, revert immediately

### After Each Phase:
1. ✅ Run full test suite
2. ✅ Check code coverage
3. ✅ Run performance benchmarks (if applicable)
4. ✅ Update documentation
5. ✅ Create PR for review

---

## 📊 Progress Tracking

### Phase 1: Quick Wins (5 hours) - **IN PROGRESS**
- [x] Analysis and planning
- [ ] Fix security warnings (2h)
- [ ] Extract magic constants (1-2h)
- [ ] Start validator refactoring (2h)

### Phase 2: Core Refactoring (30 hours)
- [ ] Complete validate_curve_fit_inputs() (6h)
- [ ] Refactor curve_fit() (12h)
- [ ] Refactor least_squares() (10h)

### Phase 3: Deep Nesting (6 hours)
- [ ] Fix update_function() (3h)
- [ ] Fix masked_residual_func() (3h)

### Phase 4: Test Coverage (16 hours)
- [ ] Recovery edge cases (4h)
- [ ] Memory manager edge cases (4h)
- [ ] Streaming optimizer edge cases (4h)
- [ ] Sparse Jacobian edge cases (4h)

---

## 🔄 Rollback Plan

If any phase causes issues:

1. **Immediate rollback**: `git reset --hard <last-good-commit>`
2. **Analyze failure**: Review test output, error messages
3. **Adjust approach**: Smaller changes, more testing
4. **Re-attempt**: Try alternative refactoring strategy

---

## 📝 Documentation Updates Needed

After completing refactoring:

- [ ] Update `CLAUDE.md` with new architecture
- [ ] Update docstrings for refactored functions
- [ ] Add migration guide if API changes
- [ ] Update `COMPREHENSIVE_QUALITY_REPORT.md` with new metrics
- [ ] Create `docs/refactoring_completed.md` with lessons learned

---

**Next Action**: Begin Phase 1 implementation (security warnings + magic constants)

**Estimated Completion**: 2 weeks with focused daily work

**Status**: 📋 **READY TO START**
