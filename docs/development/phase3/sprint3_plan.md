# Sprint 3: API Fixes and Complexity Reduction - Implementation Plan

## Status: **READY TO START**

**Branch**: `sprint3-api-fixes` (to be created)
**Duration**: Days 10-14 (estimated 20-25 hours)
**Based on**: Sprint 1 (820 tests) + Sprint 2 (3 functions refactored)

## Overview

Sprint 3 focuses on fixing API mismatches in test suite and addressing remaining high-priority complexity violations.

## Primary Goals

### Goal 1: Fix API Mismatches (Priority: HIGH)
**Target**: Fix 10 failing tests in `test_validators_comprehensive.py`
**Impact**: Achieve 100% test pass rate (820/820)

**Current Status**: 809/820 passing (98.7%), 10 failures, 1 skipped

**Failing Tests** (from Sprint 1/2):
1. `test_function_not_callable_raises` - TypeError expected but not raised
2. `test_xdata_ydata_shape_mismatch_raises` - ValueError expected but not raised
3. `test_sigma_shape_mismatch_raises` - ValueError expected but not raised
4. `test_sigma_negative_raises` - ValueError expected but not raised
5. `test_sigma_zero_raises` - ValueError expected but not raised
6. `test_bounds_lower_ge_upper_raises` - ValueError expected but not raised
7. `test_p0_outside_bounds_raises` - ValueError expected but not raised
8. `test_method_invalid_raises` - ValueError expected but not raised
9. `test_method_lm_with_bounds_raises` - ValueError expected but not raised
10. `test_bounds_shape_mismatch_raises` - ValueError expected but not raised

**Root Cause**: These tests call `validator.validate_curve_fit_inputs()` directly, but the actual validation happens through the `curve_fit` integration. The validator returns errors/warnings rather than raising exceptions.

**Solution Approach**:
1. **Option A (Recommended)**: Update tests to check returned errors/warnings instead of expecting exceptions
2. **Option B**: Add optional `strict=True` mode to validator to raise exceptions
3. **Option C**: Delete tests and rely on integration tests only

**Recommended**: Option A - Update tests to match actual API behavior.

### Goal 2: Reduce High-Priority Complexity Violations (Priority: MEDIUM)
**Target**: Reduce 6-8 highest complexity violations to <10

**Priority Targets** (based on usage frequency and maintainability impact):

#### Tier 1: Critical (High usage, maintainability impact)
1. **`validators.validate_least_squares_inputs`** (complexity 23)
   - Location: `nlsq/validators.py:673`
   - Lines: ~120
   - Impact: Core validation, frequently used
   - Estimated effort: 3-4 hours

2. **`trf.trf_no_bounds`** (complexity 24)
   - Location: `nlsq/trf.py:805`
   - Lines: ~250
   - Impact: Core optimization algorithm
   - Estimated effort: 4-5 hours

3. **`trf.trf_bounds`** (complexity 21)
   - Location: `nlsq/trf.py:1129`
   - Lines: ~300
   - Impact: Core optimization algorithm
   - Estimated effort: 4-5 hours

#### Tier 2: Important (Moderate usage, good refactoring candidates)
4. **`algorithm_selector.select_algorithm`** (complexity 20)
   - Location: `nlsq/algorithm_selector.py:306`
   - Lines: ~150
   - Impact: Algorithm selection logic
   - Estimated effort: 2-3 hours

5. **`__init__.curve_fit_large`** (complexity 18)
   - Location: `nlsq/__init__.py:173`
   - Lines: ~100
   - Impact: Large dataset API
   - Estimated effort: 2-3 hours

6. **`least_squares.update_function`** (complexity 16)
   - Location: `nlsq/least_squares.py:1026`
   - Lines: ~80
   - Impact: Function update logic
   - Estimated effort: 2-3 hours

#### Tier 3: Lower Priority (Deferred to future sprints)
- `trf.trf_no_bounds_timed` (21) - Similar to trf_no_bounds
- `minpack._run_optimization` (15) - Already partially refactored
- `streaming_optimizer.fit_streaming` (15) - Less frequently used
- `config._initialize_memory_config` (14) - Configuration, less critical
- `least_squares._setup_functions` (14) - Already partially refactored
- `large_dataset._fit_chunked` (13) - Less frequently used
- Others (11-12) - Minor violations

### Goal 3: Improve Test Coverage (Priority: LOW)
**Target**: Increase from 70% → 75%
**Approach**: Add tests during refactoring work
**Effort**: Incremental

## Sprint 3 Daily Breakdown

### Day 10: API Mismatch Fixes (4-5 hours)
**Objective**: Fix all 10 failing validator tests

**Tasks**:
1. Create `sprint3-api-fixes` branch
2. Analyze actual vs expected validator behavior
3. Update test expectations to match actual API
   - Change from `with pytest.raises(...)` to checking error lists
   - Update all 10 failing tests
4. Run full test suite to verify 820/820 passing
5. Commit: "test: fix API mismatches in validators comprehensive tests"

**Success Criteria**:
- ✅ 820/820 tests passing (100%)
- ✅ Zero test failures
- ✅ API documented correctly

### Day 11: Tier 1 Refactoring Part 1 (5-6 hours)
**Objective**: Refactor `validate_least_squares_inputs` (complexity 23 → <10)

**Tasks**:
1. Read and analyze `validators.validate_least_squares_inputs`
2. Identify helper method extraction opportunities
3. Create 5-7 focused validation helpers
4. Refactor main method to orchestrator pattern
5. Test with existing test suite
6. Commit: "refactor: break down validate_least_squares_inputs (complexity 23→<10)"

**Estimated Helpers**:
- `_validate_bounds_compatibility` - Check bounds format and compatibility
- `_validate_method_and_bounds` - Validate method selection with bounds
- `_validate_tolerances` - Check ftol, xtol, gtol values
- `_validate_loss_and_f_scale` - Validate loss function parameters
- `_validate_x_scale` - Validate parameter scaling
- `_check_jac_compatibility` - Validate Jacobian function

**Success Criteria**:
- ✅ Complexity reduced to <10
- ✅ All tests passing
- ✅ Clear helper method responsibilities

### Day 12: Tier 1 Refactoring Part 2 (4-5 hours)
**Objective**: Refactor `trf_no_bounds` (complexity 24 → <10)

**Tasks**:
1. Read and analyze `trf.trf_no_bounds`
2. Identify helper method extraction opportunities
3. Create 6-8 focused helpers for TRF algorithm steps
4. Refactor main method to orchestrator pattern
5. Test with TRF-specific tests
6. Commit: "refactor: break down trf_no_bounds (complexity 24→<10)"

**Estimated Helpers**:
- `_initialize_trf_state` - Initialize algorithm state
- `_compute_step` - Compute trust region step
- `_evaluate_step_quality` - Check step acceptance
- `_update_trust_radius` - Adjust trust region radius
- `_check_convergence` - Test convergence criteria
- `_prepare_next_iteration` - Setup for next iteration

**Success Criteria**:
- ✅ Complexity reduced to <10
- ✅ All TRF tests passing
- ✅ Zero performance regression

### Day 13: Tier 2 Refactoring (4-5 hours)
**Objective**: Refactor 2-3 Tier 2 functions

**Target Functions**:
1. `select_algorithm` (complexity 20 → <10)
2. `curve_fit_large` (complexity 18 → <10)
3. `update_function` (complexity 16 → <10) - if time permits

**Tasks**:
1. Refactor `select_algorithm` with decision tree helpers
2. Refactor `curve_fit_large` with chunking/streaming helpers
3. Test all refactorings
4. Commit each refactoring separately

**Success Criteria**:
- ✅ 2-3 functions refactored to <10
- ✅ All tests passing
- ✅ Documented helper methods

### Day 14: Validation and Documentation (3-4 hours)
**Objective**: Final validation and Sprint 3 wrap-up

**Tasks**:
1. Run full test suite (820 tests)
2. Verify all complexity targets met
3. Run pre-commit hooks
4. Performance regression check
5. Create Sprint 3 completion summary
6. Update CLAUDE.md with Sprint 3 notes
7. Merge to main
8. Push to remote

**Success Criteria**:
- ✅ 820/820 tests passing
- ✅ Target complexity violations fixed
- ✅ Pre-commit hooks passing
- ✅ Documentation complete

## Success Metrics

### Primary Metrics
| Metric | Before | Target | Stretch Goal |
|--------|--------|--------|--------------|
| **Test Pass Rate** | 809/820 (98.7%) | 820/820 (100%) | 820/820 (100%) |
| **Complexity >20** | 5 functions | 0 functions | 0 functions |
| **Complexity >15** | 10 functions | 3-5 functions | 0 functions |
| **Total C901 Violations** | 19 | 10-13 | <10 |

### Secondary Metrics
| Metric | Before | Target |
|--------|--------|--------|
| **Test Coverage** | 70% | 75% |
| **Helper Methods** | 21 (S2) | +15-20 |
| **Documentation** | Good | Excellent |

## Risk Assessment

### Low Risk
✅ API mismatch fixes - Well understood, straightforward
✅ `validate_least_squares_inputs` - Similar to Sprint 2 work
✅ Tier 2 refactorings - Lower complexity, less critical

### Medium Risk
⚠️ `trf_no_bounds` refactoring - Core algorithm, performance sensitive
⚠️ `trf_bounds` refactoring - Core algorithm, complex logic

### Mitigation Strategies
1. **Performance Testing**: Benchmark before/after TRF refactoring
2. **Comprehensive Testing**: Run full test suite after each refactoring
3. **Incremental Approach**: Commit after each successful refactoring
4. **Rollback Ready**: Keep each refactoring in separate commits

## Deferred Work (Future Sprints)

### Sprint 4 Candidates
1. Remaining Tier 3 complexity violations (11-15)
2. Configuration object introduction (reduce argument counts)
3. Coverage push to 80%
4. Performance optimization round 2
5. API documentation improvements

### Technical Debt
1. `trf.trf_no_bounds_timed` (21) - Timing wrapper, less critical
2. Streaming/chunking optimizations (moderate complexity)
3. Memory manager configuration (complexity 14)

## Tools and Resources

### Testing
- `pytest -v tests/test_validators_comprehensive.py` - Validator tests
- `pytest -n auto --tb=line -q` - Full suite parallel
- `pytest tests/test_trf*.py -v` - TRF-specific tests

### Complexity Analysis
- `ruff check nlsq/ --select C901` - Find all violations
- `ruff check <file> --select C901 --output-format=concise` - Check specific file

### Performance
- `benchmark/profile_trf_hot_paths.py` - Profile TRF performance
- `benchmark/test_performance_regression.py` - Regression tests

## Expected Outcomes

### Code Quality
- ✅ 100% test pass rate (820/820)
- ✅ 6-8 high-priority complexity violations fixed
- ✅ 15-20 new helper methods with single responsibility
- ✅ Improved code maintainability

### Technical
- ✅ Zero performance regressions
- ✅ All pre-commit hooks passing
- ✅ Comprehensive documentation

### Process
- ✅ Proven refactoring pattern (3rd sprint)
- ✅ Strong test safety net maintained
- ✅ Clean git history

## Timeline

**Total Estimated Effort**: 20-25 hours
**Target Completion**: 5 days (Days 10-14)
**Confidence Level**: 🟢 HIGH (based on Sprint 1 & 2 success)

---

**Sprint 3 Plan Status**: READY TO EXECUTE
**Created**: 2025-10-07
**Next Step**: Create `sprint3-api-fixes` branch and begin Day 10

🤖 Generated with [Claude Code](https://claude.com/claude-code)
