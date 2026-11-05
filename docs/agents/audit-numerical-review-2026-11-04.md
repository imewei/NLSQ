# Independent numerical patch review — 2026-11-04

Reviewed the dirty changes in `nlsq/streaming/adaptive_hybrid.py`,
`nlsq/streaming/phases/{gauss_newton,warmup}.py`,
`nlsq/stability/{fallback,recovery}.py`, and
`nlsq/core/{trf,trf_jit}.py`. Focus: bounds preservation, unsupported weighting,
accepted versus rejected steps, normalized parameter coordinates, reused
optimizer state, and numerical termination.

**Confirmed regression, fixed:** the new relative CG stopping threshold is
zero when the right-hand side is zero. The previous `rsold >= tol_sq`
condition therefore entered the iteration for an already solved system.
Float64 performed all 100 iterations unnecessarily. Float32 produced NaNs
because the new `1e-300` division floor rounded to zero. Nonzero float32
systems whose curvature product underflows also produced NaNs.

Concrete reproducer through the exported helper class:
`TrustRegionJITFunctions().conjugate_gradient_solve(J, f, d)`, with float32
`J=[[1e-10]]`, `d=[1]`, and either `f=[0]` or `f=[1e-5]`.
The corresponding zero-RHS float64 case returned iteration count 100.
Ordinary `LeastSquares` converts `x0` to float64, and TRF scaling commonly
promotes the inputs; this review did not reproduce a divergent outer fit.
The directly exposed JIT helper still accepts float32 arrays.

The first correction (strict stopping plus a dtype-positive divisor floor)
prevented NaNs but was insufficient: the nonzero float32 example returned
`-0.000429606` instead of `-100000`. Independent review caught this weakness
in the finite-only test. The strengthened regression asserts the hand-derived
solution, reconstructed linear residual, and normal-equation residual norm;
it failed before the subsequent correction (1.04 seconds).

The final correction also normalizes the RHS by its maximum absolute entry
before CG and restores the solution and residual norm afterward. This removes
the extra squared RHS magnitude from the curvature product without changing
the working dtype, matrix, damping, or relative tolerance. Here `J^T J=1e-20`
is representable in float32; matrix scaling is unnecessary for this failure.
Strict stopping and a dtype-positive divisor floor protect zero/breakdown
cases. The existing float64 floor remains unchanged.

Three permanent regressions in `tests/core/test_trf_jit.py` first failed
(1.20 seconds); the nonzero case was subsequently strengthened as described
above. Final verification: **48 passed**, `-n 0 --reruns 0`, 12.94 seconds,
including existing float64 scale-invariance and trust-region tests. Ruff and
format checks passed for the two edited files.

No other concrete severe regression was found in the reviewed changes.
Hybrid sigma rejection prevents silently unweighted output; recovery retains
original user bounds; the variance penalty's physical-coordinate chain rule
and warmup's parameter/loss pairing are consistent with their objectives.
State-history results are copied before subsequent fits clear internal
history. This was a focused patch review, not a new broad test-suite run.
