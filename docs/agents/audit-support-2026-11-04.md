# NLSQ support-module correctness audit — 2026-11-04

This pass inspected caching, results, diagnostics, configuration, callbacks,
validation, safe serialization, device detection, and profiling. It used the
existing graph and targeted source/test reads. The workspace already contained
substantial uncommitted work; this report describes additional fixes only.

## Confirmed findings

1. **Callable identity collision.** `nlsq/caching/_closure_serial.py` promised a
   unique serial per object but used `WeakKeyDictionary`, which compares
   referents by equality. Distinct frozen dataclass callables whose comparison
   excludes their scale shared an entry. The public `UnifiedCache` then returned
   `[2, 4]` when the second callable should return `[3, 6]`. An identity-indexed
   weakref registry fixes that layer. A targeted probe showed JAX also reused
   equal callable objects despite distinct NLSQ wrappers; compiling an ordinary
   forwarding function for callable objects isolates the JAX trace as well.
   Regression: `tests/caching/test_callable_identity.py`.

2. **Restored-result residual signs.** `CurveFitResult` documents observed minus
   predicted residuals, whereas optimizer `fun` stores predicted minus observed.
   The fallback without a model used the wrong signs for both predictions and
   residuals. Two failing tests demonstrated predictions `[4, 6, 8]` instead of
   `[2, 4, 6]` and residuals `[-1, -1, -1]` instead of `[1, 1, 1]`. Both fallback
   signs now follow the documented definitions. As before, raw residuals cannot
   be reconstructed from weighted `fun` alone without model/weight metadata.
   Regression: `tests/core/test_result_fallback_residuals.py`.

3. **Sensitivity dimensionality over/underflow.** The participation ratio squared
   unnormalized eigenvalues. Two equal finite eigenvalues at scales `1e-200`
   and `1e200` produced zero or NaN dimensionality rather than two, with an
   incorrect health classification. Normalize the spectrum before summation and
   squaring; the ratio is invariant under this scaling. Four regressions cover
   Jacobian and precomputed-FIM public analysis paths in
   `tests/diagnostics/test_sensitivity_scale_invariance.py`.

## Verification evidence

Each new regression was executed and observed failing before its corresponding
fix. Focused green runs (CPU, `-n 0 --reruns 0`):

- Cache identity plus unified cache and concurrency suites: **26 passed**.
- Result fallback plus enhanced result suite: **38 passed**.
- Scale invariance plus sensitivity suite: **69 passed**.

The observed NaNs in the diagnostic reproduction were arithmetic overflow,
not a diverging fit. The fixed regression reports finite dimensionality two
and healthy status at both scales. Package-wide integration results and quality
checks are recorded separately in the synthesis report.
