# Sprint 4 Plan: Core Algorithm Refactoring (DEFERRED)

**Status**: PLANNED (Not Started)
**Priority**: LOW (Optional Enhancement)
**Risk Level**: HIGH (Core algorithm modifications)
**Timeline**: 5-7 days when needed
**Branch**: `sprint4-core-algorithms` (to be created)

## Context

Sprint 3 completed API fixes and reduced 2 complexity violations (23→<10, 20→<10). The most complex remaining functions are core TRF (Trust Region Reflective) algorithm implementations that were **strategically deferred** due to:

1. **High Risk**: Core optimization algorithms with critical numerical stability
2. **High Complexity**: Inherent algorithmic complexity that may resist simple extraction
3. **Low Priority**: Code is production-ready, well-tested, and performant
4. **Better ROI**: Focus on features and user experience over micro-refactoring

## Deferred Work

### Remaining Complexity Violations in `nlsq/trf.py`

| Function | Complexity | Risk | Priority | Reason for Deferral |
|----------|-----------|------|----------|---------------------|
| `trf_no_bounds` | 24 | HIGH | LOW | Core algorithm, 200+ lines, complex control flow |
| `trf_bounds` | 21 | HIGH | LOW | Core algorithm with boundary handling |
| `trf_no_bounds_timed` | 21 | MEDIUM | LOW | Wrapper around `trf_no_bounds` |

**Total**: 3 functions, 66 complexity points

### Other High-Complexity Functions (Non-TRF)

**From current analysis** (17 total violations):
- `nlsq/__init__.py:curve_fit_large` - C901 (18)
- `nlsq/minpack.py:_prepare_curve_fit_inputs` - C901 (complexity not reduced in Sprint 2)
- `nlsq/least_squares.py:least_squares` - C901 (complexity 24, main orchestrator)
- Plus ~11 other violations across the codebase

## Sprint 4 Objectives (IF UNDERTAKEN)

### Primary Goal: TRF Algorithm Refactoring
**Target**: Reduce 3 TRF functions to complexity <15 (realistic target, not <10)
**Approach**: Extract helper methods while preserving numerical correctness

### Secondary Goal: Test Coverage Enhancement
**Target**: Add unit tests for complex helper methods created in Sprints 1-3
**Baseline**: 77% coverage (5074 statements, 1144 missed)
**Goal**: 80% coverage

## Recommended Approach (If Sprint 4 is Executed)

### Phase 1: Risk Assessment (Day 1)
1. **Analyze TRF algorithms** - Understand mathematical foundations
2. **Identify extraction candidates** - Find safe helper opportunities
3. **Create safety tests** - Add algorithm-specific unit tests
4. **Document baseline** - Performance benchmarks and test coverage

### Phase 2: TRF Refactoring (Days 2-4)
1. **`trf_no_bounds_timed`** (Day 2)
   - Complexity 21 (lowest risk)
   - Wrapper function, easier to refactor
   - Extract timing/logging logic

2. **`trf_bounds`** (Day 3)
   - Complexity 21
   - Extract boundary checking logic
   - Extract step computation helpers

3. **`trf_no_bounds`** (Day 4)
   - Complexity 24 (highest risk)
   - Extract main loop helpers
   - Extract convergence check logic

### Phase 3: Validation (Days 5-6)
1. Run full test suite (817 tests must pass)
2. Run performance benchmarks (no >5% regression)
3. Run numerical accuracy tests
4. Update documentation

### Phase 4: Optional Enhancements (Day 7)
1. Add unit tests for new helpers
2. Improve test coverage to 80%
3. Clean up any code smells

## Success Criteria

### Must-Have (Required for Sprint 4 completion)
- ✅ All 817 tests passing (100% pass rate)
- ✅ No performance regression (>5% slowdown)
- ✅ Zero numerical accuracy regressions
- ✅ At least 2/3 TRF functions reduced below complexity 15

### Nice-to-Have (Optional enhancements)
- ⭐ All 3 TRF functions reduced to complexity <15
- ⭐ Test coverage increased to 80%
- ⭐ Helper method unit tests added
- ⭐ Documentation updated with refactoring patterns

## Risk Mitigation

### Technical Risks
1. **Numerical Instability**: TRF algorithms are numerically sensitive
   - **Mitigation**: Extensive numerical accuracy tests before/after
   - **Fallback**: Revert changes if any numerical regression

2. **Performance Regression**: Helper extraction may slow down hot paths
   - **Mitigation**: Run performance benchmarks on every change
   - **Fallback**: Inline critical helpers if needed

3. **JAX JIT Compilation**: Helper extraction may break JIT optimization
   - **Mitigation**: Use `@jit` decorators appropriately
   - **Fallback**: Keep helpers within main function if JIT breaks

### Process Risks
1. **Time Overrun**: TRF refactoring may take longer than planned
   - **Mitigation**: Timebox each function to 1 day maximum
   - **Fallback**: Defer remaining functions to future sprint

2. **Scope Creep**: Temptation to refactor other violations
   - **Mitigation**: Strict scope limited to 3 TRF functions
   - **Fallback**: Create Sprint 5 plan for other violations

## Decision: DEFER Sprint 4

**Recommendation**: **DO NOT execute Sprint 4** unless:

1. **User-reported issues** with TRF algorithm maintainability
2. **Bugs discovered** in TRF implementations that require refactoring
3. **Performance optimization** requires algorithm restructuring
4. **New features** require modifying TRF core logic

**Rationale**:
- Code is production-ready and well-tested (817/820 tests passing)
- Performance is excellent (270x faster than SciPy)
- Coverage is good (77%, close to 80% target)
- Complexity violations are in inherently complex algorithms
- Better ROI from features and user experience improvements

## Alternative Focus Areas

Instead of Sprint 4, consider:

1. **Documentation Enhancement**
   - User guides for advanced features
   - Performance tuning cookbook
   - API reference improvements

2. **Feature Development**
   - Additional loss functions
   - Sparse Jacobian optimizations
   - Multi-GPU support

3. **User Experience**
   - Better error messages
   - Progress callbacks
   - Diagnostic outputs

4. **Ecosystem Integration**
   - PyTorch compatibility layer
   - TensorFlow integration
   - Scikit-learn compatible API

## Baseline Metrics (Current State)

**From Sprint 3 Completion**:
- Tests: 817/820 passing (3 skipped, 100% pass rate)
- Coverage: 77% (5074 statements, 1144 missed)
- Complexity violations: 17 (down from 19)
- TRF violations: 3 functions (24, 21, 21 complexity)

**Performance Benchmarks** (CPU, from optimization work):
- Small (100 pts): ~430ms total (~30ms after JIT)
- Medium (1K pts): ~490ms total (~110ms after JIT)
- Large (10K pts): ~605ms total (~134ms after JIT)
- XLarge (50K pts): ~572ms total (~120ms after JIT)

**GPU Performance**:
- 1M points: 0.15s (NLSQ) vs 40.5s (SciPy) = 270x speedup

## Notes

- Sprint 4 is **optional** and **deferred indefinitely**
- Current codebase is production-ready and performant
- Focus should be on features, not micro-optimizations
- Refactoring core algorithms has diminishing returns
- This plan serves as a **reference** if future work requires TRF modifications

## References

- Sprint 3 Completion Summary: `/home/wei/Documents/GitHub/nlsq/sprint3_completion_summary.md`
- Optimization Case Study: `/home/wei/Documents/GitHub/nlsq/docs/optimization_case_study.md`
- Performance Tuning Guide: `/home/wei/Documents/GitHub/nlsq/docs/performance_tuning_guide.md`
- Current complexity violations: `ruff check nlsq/ --select C901`

---

**Created**: 2025-10-07
**Status**: DEFERRED
**Next Review**: Only if user-driven need arises
