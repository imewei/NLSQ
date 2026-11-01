# Exact Streaming Solver — Design

**Date:** 2026-10-31 (revision 2: cross-model review round 1 applied)
**Status:** Draft, written without a question round (the requester gave the four stages and their acceptance conditions directly). Every choice marked **[my call]** is the author's, not the requester's. Choices marked **[requester]** were decided by the requester; choices marked **[review]** were settled by the three-model review (Claude, Codex, Agy) the requester commissioned.
**Source:** `three-brain-out/2026-10-31-nlsq-4m-1b-perf/consensus.md` (audit), `three-brain-out/2026-10-31-integrate-exact-lm/consensus.md` (decision: option B, opt-in path), `three-brain-out/2026-10-31-spec-review/` (review rounds), `benchmarks/baselines/2026-10-31-library-comparison-cpu-gpu-100M.md` (all measurements cited here), prototype `benchmarks/prototypes/exact_accumulated_lm.py`.

## Intent

For 4M–1B points, `nlsq.curve_fit_large` fits each chunk independently and merges the results by precision-weighted GLS. For nonlinear models that is an approximation, not a full-batch fit (`nlsq/streaming/large_dataset.py:1804-1811`, flagged by `equivalent_to_full_batch=False`), which breaks the project rule that streaming must equal full-batch. It is also slow: at 100M points on GPU it takes 9–18 s where a prototype that accumulates the full-data objective takes 0.66–2.8 s (JAXopt 0.8–5.1 s), and 45% of the time on exp_decay is a host-side shuffle that exists only to make per-chunk fits representative.

The aim is an **exact** large-N solver: one pass over the data per trial point accumulates the cost, `Jᵀr` and `JᵀJ` in float64 (O(p²) state), then a damped Gauss–Newton (Levenberg–Marquardt) step is taken on the full-data problem. "Exact" means every observation contributes to one objective, so the answer does not depend on chunking, ordering, or where the data lives, up to floating-point summation order. It ships as a **new opt-in strategy**; the default `curve_fit_large` behaviour does not change.

## Supported model contract

The method is exact only for models that are **pointwise and deterministic**: `f(x, θ)[i]` depends only on `x[i]` and `θ`, with no dependence on the other points in the batch (no normalization by batch statistics, no `x.max()`, no cumulative sums along the data axis, no randomness), and is JAX-traceable and differentiable in `θ`. Anything else changes with chunk boundaries and silently loses exactness. Enforcement **[review]**: at entry the solver evaluates the model on the first chunk and on a strict subset of it (the first half, and a reversed copy) and requires `f(x[:k]) == f(x)[:k]` and `f(x[::-1]) == f(x)[::-1]` to relative 1e-12; a violation raises `ValueError` naming the contract. The check cannot prove pointwise behaviour (a model could behave differently only on later chunks), so the docs state the contract and the check is a guard, not a proof.

## Goals

1. **Stage 0 — harden the prototype.** Fix defects 1–5 (below) in `benchmarks/prototypes/exact_accumulated_lm.py` and pin each with a regression test. **[requester]**
2. **Stage 1 — unbounded, unweighted engine in the library.** New module `nlsq/streaming/exact_lm.py`, reachable through `curve_fit_large`. **[requester]** Acceptance:
   - (a) on the **unbounded** benchmark problems (exp_decay, gaussian, two_exp) and on a seeded randomized set, parameters, cost and scaled gradient meet the tolerance table below against `scipy.optimize.least_squares`;
   - (b) results are chunk-size-independent to the table's tolerance for chunk sizes `{n, n/2, 512, 513, 7}`;
   - (c) the unweighted covariance meets the covariance-parity tolerance against `curve_fit` for `κ(J) ≤ 1e4`;
   - (d) 100M-point GPU exp_decay warm median ≤ 1.10 × the prototype's, with prototype and module measured back-to-back in one session under the same transfer policy (data resident on device, 3 warm repeats). The prototype's recorded 0.66 s is orientation only: this laptop's host-bound timings have drifted up to 1.5× between sessions.
3. **Stage 2 — weights, bounds, weighted covariance.** Add 1-D `sigma`, box bounds, and the weighted covariance. Active-set bounds are a different algorithm from the trust-region one, so they need KKT-condition tests. **[requester]** Acceptance: weighted fits match a weighted SciPy reference; bounded fits meet the tolerance table against SciPy TRF on the benchmark `bounded_exp` and on a randomized set where 0, 1 and all parameters sit on a bound.
4. **Stage 3 — 1B points and a conditioning study.** Accept out-of-memory sources and run a pre-registered study of normal equations versus a QR-based accumulation. **[requester]** Acceptance: the 1B-point correctness and memory gates below, and the study's decision rule applied and recorded.

## Non-goals

- Changing the default `curve_fit_large` path, its tests, or `equivalent_to_full_batch` semantics for the existing chunked path.
- Removing or merging `AdaptiveHybridStreamingOptimizer` / `GaussNewtonPhase`.
- Exposing the method through `nlsq.fit()`, the CLI or the GUI (decisions and triggers in "Resolved deferrals").
- Multi-GPU sharding, a fused hand-written kernel, mixed precision (NLSQ is float64 only), multi-start with the exact method, `callback`, correlated (2-D) `sigma`, multi-feature `xdata` (1-D `xdata` only; others raise `ValueError`) **[my call]**.
- Fixing the unrelated defects Codex found in the legacy chunked path (weighted retries dropping `sigma`, the 50% success threshold, dropped metadata in the returned tuple). They are recorded in the audit.

## Anti-goals (a failure even if the tests pass)

- A fit that stalls, runs out of budget, or hits a numerical floor without having made progress reports success.
- The opposite: a fit that has converged to floating-point precision is reported as a failure and raises.
- A result that depends on the chunk size, the order of the data, or whether the data was resident or streamed (beyond floating-point summation tolerance).
- Silent behaviour change for existing `curve_fit_large` or `fit()` callers, including the default method, positional arguments, return types and existing warnings.
- Silent subsampling, early exit before all points contribute, or a hidden `float32` fit.
- A benchmark claim made from numbers taken in different machine states.
- Presenting active-set bounds as equivalent to TRF, or normal-equation covariance as equivalent to the SVD covariance without the conditioning caveat.

## The algorithm (fixed by the prototype and the audit)

State per iterate: `θ` (p,), `F = ½ Σ r²`, `g = Jᵀr` (r = y − f(x, θ), J = ∂f/∂θ, so `∇F = −g`), `H = JᵀJ`.

- **One pass per trial.** A pass visits every chunk once and returns `(F, g, H)` at one parameter vector. An accepted trial's statistics are exactly the next iterate's, so an accepted step costs one pass and a rejected step costs one pass. No separate evaluation pass.
- **Step.** Solve `(H + λ·D) δ = g` on the free variables, trial point `θ_t = clip(θ + δ, lb, ub)`, **actual displacement** `s = θ_t − θ`. Predicted reduction uses `s`, not `δ`: `pred = sᵀg − ½ sᵀHs` **[review]**. Gain ratio `ρ = (F − F_t) / pred`. Nielsen λ update on acceptance (`λ ← λ·max(1/3, 1 − (2ρ−1)³)`, `ν = 2`), `λ ← λ·ν, ν ← 2ν` on rejection.
- **Acceptance.** A trial is accepted iff `F_t`, `g_t` and `H_t` are all finite, `F − F_t > 0` and `pred > 0`.
- **Damping scale** **[review]**: `D_ii = max(D_ii_prev, H_ii)` (Moré running maximum), floored at `ε_D · max_j D_jj` with `ε_D = 1e-10`; if every `H_ii` is zero, `D = I`. This keeps `H + λD` positive definite for any `λ > 0`, including a zero Jacobian column. Rank deficiency alone is not a failure.
- **Chunks.** Per chunk, `r` and `J` (forward-mode Jacobian) are computed on device and reduced to `(F, g, H)`; chunk results are summed in float64. The tail chunk is padded to the common shape (edge-copied values, so a padded row has the same finiteness as a real row) and excluded with `jnp.where` on a mask, not by multiplication **[my call]**.
- **Device placement.** If the data fits in a configured fraction of free device memory (default 50%) it is placed on the device once; otherwise chunks are transferred each pass **[my call]**. On the CPU backend chunks are passed as views and no whole-array copy is made; a test verifies it (Tests section).
- **Covariance for free.** The final `H` gives `pcov` without another pass.

### Termination

Defaults 1e-8. The stationarity checks run at **every iterate with finite statistics, including the initial point**, so a start that is already optimal ends with one pass and no step **[review]**. The step-based checks run after an accepted step:

- `gtol` (stationarity, free variables only): `‖g_free‖∞ < gtol`, absolute, as `nlsq/core/trf.py:754`. With bounds, `free` is the active-set definition of the Algorithm section (variables at a bound with `g` pointing outward are excluded), the analogue of the `g·v` scaling in the bounded TRF check (`trf.py:2240`), not identical to it; without bounds, `free` is everything. At large N the gradient noise floor is far above 1e-8, so this rarely terminates a large noisy fit; that matches TRF and SciPy and is intended.
- **Newton-decrement criterion (damping-independent)** **[review]**: work in **Jacobi-scaled coordinates** so the test is invariant to parameter units: over the columns with `H_ii > 0` (a zero column has `g_i = 0` exactly and is excluded), `Ĥ = D^{-1/2} H D^{-1/2}` with `D = diag(H)` (unit diagonal, `λ̂_max ≤ p`) and `ĝ = D^{-1/2} g`; `Ĥ = ÛΛ̂Ûᵀ` (host `eigh`, p ≤ ~20) and `δ_N = ½ ĝᵀ Ĥ⁻¹ ĝ`. If **all** of the following hold, terminate as `"ftol"` (success), because even the undamped Gauss–Newton step is predicted to improve the cost by less than `ftol` of itself: (i) `δ_N ≤ ftol·F`; (ii) **nothing is truncated**: `λ̂_min > p·ε·λ̂_max`. Reason for (ii): `H = JᵀJ` formed in float64 has lost the information needed to tell roundoff from signal in a direction below that cutoff (counterexample: `J = [[1e15, 1e15],[1, −1]]`, `r = (0, 1)` gives every entry of `H` equal to `1e30`, a hidden gradient of `1.4e-15` after scaling, `δ_N = 0`, yet the step `(0.5, −0.5)` removes the whole residual), so no certificate can be issued from `H` alone there. When a direction is truncated the solver relies on the step-based `ftol`/`xtol` criteria and the projected `gtol` check; if it cannot progress it ends `"stalled"` with the message "rank cannot be certified from normal equations", and the TSQR path of the Stage 3 study (which works from `R`, not `JᵀJ`) is the remedy. Scaling matters for (i): with unscaled `H`, a parameter rescaling such as `H = diag(1e30, 1)`, `g = (0, 1)`, `F = ½` would inflate any cutoff; scaled, that case has `Ĥ = I`, `δ_N = ½ > ftol·F`, and the criterion correctly does not fire. The one-pass start-at-optimum guarantee therefore holds for full-rank systems and for zero columns (excluded before scaling), not for nonzero collinear columns; that limit is documented, not hidden. Tests: the `diag(1e20,1)` and `diag(1e30,1)` rescalings (must not fire), the `1e15` near-collinear case above (must not fire, must not report success unless a step criterion fires), a well-conditioned start at the optimum (must fire after one pass), exactly collinear columns for p = 3 (converges through the step criteria), and the bounded counterexample below. If `F == 0` exactly the fit is exact: terminate `"gtol"` (success) before any division. (iii) **No active bound**: the criterion is applied only when every variable is free. With active bounds, the free-subspace decrement is not an upper bound on a coupled feasible step that releases an active variable (a 2-parameter counterexample has free decrement 2 against a feasible model decrease 26), so with bounds the stationarity certificate is the projected `gtol` check plus the step-based `ftol`/`xtol` criteria. The noise-floor risk this leaves for bounded fits is covered by the Stage 2 acceptance test "bounded start at the constrained optimum terminates successfully"; if that fails, the response is to certify with the *constrained* model decrease, computed by a small projected-Newton solve of the p-dimensional quadratic (no data pass), instead of the free-subspace formula **[my call]**. This criterion also covers the numerical floor: at the optimum, rounding noise makes steps get rejected while λ grows, and without a criterion that does not depend on λ the solver would report a false stall. A criterion based on the damped `pred` is **not** used, because `pred` shrinks as λ grows and would label a non-stationary point (or a model that is non-finite everywhere except `p0`) as converged.
- `ftol` (step-based): `ΔF < ftol·F` and `ρ > 0.25` after an accepted step (`nlsq/common_jax.py:780`).
- `xtol`: `‖s‖ < xtol·(xtol + ‖θ‖)` after an accepted step (`nlsq/common_jax.py:781`).
- **Budget precedence:** if a convergence criterion holds on the same pass that exhausts `max_nfev`, convergence wins; `"max_nfev"` is reported only when no criterion holds.
- **Stall** is declared when trials keep being rejected (invalid, non-finite, `F_t ≥ F`, or `pred ≤ 0`) until `λ` reaches `1e14`, or when the per-iteration solve-attempt cap (20) is exhausted, with no stationarity criterion satisfied at the current iterate. Termination reasons: `"gtol" | "ftol" | "xtol" | "max_nfev" | "stalled"`. `"stalled"` and `"max_nfev"` are failures.
- `max_nfev` counts full-data passes including the initial one; a safety cap, not a parity criterion with SciPy (SciPy counts model evaluations). Default `100·p` **[my call]**; the stall rule is the primary guard against runaway cost at 1B points.

## Acceptance tolerances (all stages) **[review]**

Termination reasons are never compared across solvers; stopping rules are tested in isolation (Stage 0). Comparisons against SciPy use both solvers at `ftol = xtol = gtol = 1e-12`, because the loose default stops early on flat valleys (seen in the prototype's self-test).

| Quantity | Tolerance |
|---|---|
| Parameters, well-determined optimum | `rtol = 1e-6`, `atol = 1e-8` |
| Parameters, flat valley (two_exp at small n) | compare cost, and parameters after sorting the term pairs by rate, at `rtol = 1e-4` |
| Cost `F` | relative `1e-9` unweighted/unbounded; relative `1e-6` against SciPy TRF for bounded problems (active-set and TRF may stop at different points of a flat constrained valley) |
| Scaled gradient (free variables) | `|g_i| / (√H_ii · √(2F)) ≤ 1e-6` (scale-invariant: the cosine between residual and Jacobian column `i`; independent of N). Defined as 0 when `H_ii = 0` (a zero column has zero gradient) or `F = 0` (exact fit, handled before any division, including all-zero `y`); for `0 < F ≤ 1e-24·Σy²` (near-exact regime) it is replaced coordinatewise by `|g_i| ≤ 1e-12·√(Σy²)·√H_ii` |
| Zero-reference convention | every relative comparison is `|a − b| ≤ rtol·|b| + atol`, with `atol = 1e-12·max(½Σy² + ½Σf(x,θ_ref)², 1e-300)` for costs (positive even if all `y = 0`) and `atol = 1e-12·max(1, ‖θ_ref‖∞)` for parameters |
| Chunk-size independence | `rtol = 1e-9` on parameters, `1e-12` relative on `F` |
| Covariance parity with `curve_fit` | `rtol = 1e-6` for `κ(J) ≤ 1e4`; `rtol = 1e-3` for `κ(J) ≤ 1e6`; beyond that flagged, not claimed |

## Stage 0 — defects to fix (all in `benchmarks/prototypes/exact_accumulated_lm.py`)

| # | Defect | Required behaviour | Regression tests |
|---|---|---|---|
| 1 | Hitting the damping ceiling sets `success=True` (`success = lam > 1e14`) | Termination as defined above: `success=True` only for `gtol`/`ftol`/`xtol` (including the numerical-floor rule); `"stalled"` and `"max_nfev"` set `success=False` | (a) a model whose value is non-finite at every trial point except `p0` ends `"stalled"`, `success=False`; (b) a budget of 1, 2, 3 passes ends `"max_nfev"` with exactly that many passes; (c) a noisy, **full-rank** (untruncated scaled system) problem started at its own optimum ends `"gtol"`/`"ftol"` after one pass with `success=True` (no false stall at the noise floor); the guarantee is restricted to untruncated systems: a noisy start at the optimum with nonzero *collinear* columns may end `success=True` through a step criterion or `"stalled"` with the rank message, and the test asserts exactly that disjunction (never a success label on a point whose stationarity could be disproved) **[review]**; (d) a nonstationary start where a test double for the pass function returns `F_t = 2F` at every trial ends `"stalled"`, not `"ftol"` |
| 2 | A `LinAlgError` or non-finite solve retries without consuming budget; `λ·diag(H)` cannot regularize a zero diagonal | Damping scale and acceptance rules above; a failed or non-finite solve counts against the per-iteration attempt cap (20) and then ends as `"stalled"`; each attempt that evaluates the data consumes a pass | a model with a zero Jacobian column **converges** (`success=True`), leaves that parameter at its start value, and returns a rank-deficient `pcov` (pseudo-inverse over retained directions, as `curve_fit`); NaN planted in `H` ends `"stalled"` within the cap; rank-2 `H` for p=3 converges |
| 3 | `gtol` is relative to the first gradient; `ftol` lacks the reduction-ratio gate | Termination rules above | gtol met at a known noiseless optimum; ftol not triggered by an accepted step with `ρ ≤ 0.25`; xtol triggered by a tiny step; each rule tested in isolation (no comparison of termination reasons with SciPy) |
| 4 | `max_nfev` is unspecified | As defined above | see 1(b) |
| 5 | No finite checks; `jax_enable_x64` does not cast float32 inputs | Inputs converted to float64 explicitly; non-finite `x`/`y` raise `ValueError` (the `check_finite` flag arrives with the library entry in Stage 1; the prototype always checks); a non-finite model value or gradient at the starting point raises `ValueError` (also when `check_finite=False`, since the cost itself is then non-finite); a non-finite value at a trial is a rejected step | float32 input gives the same result as float64 input; NaN in `y`; model returning inf at `p0` |

## Stage 1 — library module

- **Module:** `nlsq/streaming/exact_lm.py`. Contents: the chunk-statistics function, the pass driver, the LM loop, the model-contract probe, and a result builder. About the prototype's size plus input handling (Agy estimated ~350 LOC; unverified). Nothing else in `nlsq/` is rewritten.
- **Entry point:** `curve_fit_large(..., method="exact_streaming")` **[my call: name; follows the `method="hybrid_streaming"` precedent at `nlsq/__init__.py:897`]**. `curve_fit_large`'s `method` annotation becomes `MethodLiteral | Literal["exact_streaming"]`; the shared `MethodLiteral` (`nlsq/types.py:122`, used by `fit()` and `curve_fit`) is **not** widened **[review]**. Returns `(popt, pcov)` like the other `curve_fit_large` paths (`nlsq/__init__.py:963` hybrid, `:1110-1118` chunked); small-input `curve_fit` delegation returns `CurveFitResult`, which this method never uses. The module's own function returns `nlsq.result.optimize_result.OptimizeResult` with `x`, `cost`, `success`, `message`, `nfev` (passes), `termination`, `pcov`, and `equivalent_to_full_batch=True`.
- **Accepted solver keyword arguments (the `**kwargs` beyond `curve_fit_large`'s named parameters, which are covered in the next-but-one item):** exactly `ftol`, `xtol`, `gtol`, `max_nfev` (plus `n_points` from Stage 3, factories only). Anything else raises `TypeError` naming it, as the hybrid dispatch does (`nlsq/__init__.py:911`) **[review]**.
- **Failure reaches the caller as an exception [requester].** If `termination` is `"stalled"` or `"max_nfev"`, `curve_fit_large(method="exact_streaming")` raises `ExactSolverError`, a `RuntimeError` subclass defined in `exact_lm.py` (the `scipy.optimize.curve_fit` convention) carrying the full result as `.result`. The module-level function returns the result without raising; only the `curve_fit_large` entry converts a failed result into the exception. Success returns `(popt, pcov)`.
- **Dispatch placement:** the `exact_streaming` branch is hoisted above the `np.asarray` conversion and the shared length checks (`nlsq/__init__.py:878-893`), because those would load an `h5py.Dataset` or a Zarr array whole in Stage 3. The branch validates from `shape`/`len` without reading data (equal lengths, at least 2 points, 1-D) **[my call]**. The hybrid branch and everything below it are untouched.
- **Size gate:** an explicit `method="exact_streaming"` always runs the exact solver, regardless of `size_threshold` **[my call]**.
- **Non-finite data check:** `check_finite=True` (default) is fused into pass 1: each chunk also reduces `all(isfinite(x), isfinite(y))`, so it costs no extra pass and works for streamed sources. A non-finite value raises `ValueError` after pass 1 and before any step. `check_finite=False` skips it **[my call]**.
- **Other `curve_fit_large` arguments** **[my call]**: `chunk_size`, `check_finite`, `show_progress` (one log line per pass), `absolute_sigma`, and `memory_limit_gb` (budget for resident data plus chunk workspace: device memory on GPU, host memory on CPU) are honoured. `stability`, `rescale_data`, `max_jacobian_elements_for_svd`, `auto_size_detection` and `size_threshold` raise `ValueError` if set to anything other than their defaults, naming the argument. `multistart=True` raises `ValueError`; while `multistart` is `False`, its sub-options (`n_starts`, `sampler`, `center_on_p0`, `scale_factor`) are ignored, since `fit()` always forwards them. Nothing else is silently ignored.
- **Explicit rejection (Stage 1 only):** `sigma`, finite `bounds` (with `lb < ub` required in every stage, as `nlsq/core/least_squares.py:984` already requires), 2-D `xdata` raise `ValueError` naming the stage that adds them **[my call]**.
- **`p0=None`:** same parameter-count inference as the hybrid dispatch (`nlsq/__init__.py:923-932`); factor the snippet if it can be shared without changing the hybrid behaviour.
- **Chunk size:** `chunk_size` argument if given, else 4M points (the prototype's value), reduced when the estimated chunk workspace (`chunk × p × 8 B × a small factor`) would exceed the budget **[my call]**.
- **Covariance (unweighted)** follows the `curve_fit` contract (Stage 2 section below) and is part of Stage 1 acceptance (c).
- **Harness:** register `curve_fit_large` and the exact method as adapters in `benchmarks/compare/adapters.py` so the comparison harness measures them (the current NLSQ adapter calls only plain `curve_fit`, which is why the first 100M GPU runs showed NLSQ out of memory).

## Stage 2 — weights, bounds, covariance

- **1-D sigma:** residuals and Jacobian rows scaled by `1/σ` per chunk (`σ` arrives with each chunk). Wrong length, non-positive or non-finite `σ` → `ValueError`; the finite check is fused into pass 1. 2-D sigma → `ValueError` (as `nlsq/streaming/large_dataset.py:2689`).
- **Covariance contract** to reproduce from `_compute_covariance` (`nlsq/core/minpack.py:3419-3500`) **[review]**:
  - Singular values `s_i = √λ_i(H)`, kept if above the rank cutoff; `pcov` is the pseudo-inverse over the kept directions (a rank-deficient `H` still gives a finite `pcov`; dropped directions contribute zero).
  - All-`inf` `pcov` plus `OptimizeWarning` ("Covariance of the parameters could not be estimated") **only** when no direction survives or any value is non-finite.
  - `absolute_sigma=False`: scale by `Σr²/(n−p)` (p = total parameter count, as the legacy code); if `n ≤ p`, all-`inf` plus the warning. With `absolute_sigma=True` there is no scaling and no `n ≤ p` failure.
  - **Rank cutoff from `H`:** `λ_i > p·ε·λ_max`, independent of `n`, i.e. directions with `κ(J) = √(λ_max/λ_i)` above `κ_H = 1/√(p·ε)` are dropped (`3.4×10⁷` for p = 4). The legacy cutoff on singular values (`s > ε·max(n,p)·s_max`, `minpack.py:3463`) drops directions above `κ_legacy = 1/(ε·max(n,p))`: `4.5×10⁹` at n = 1e6, `4.5×10⁷` at 1e8, `4.5×10⁶` at 1e9. The `n`-scaled form `ε·max(n,p)·λ_max` applied to `H` would wrongly discard directions above `κ = 1/√(ε·max(n,p))` (about 6,700 at 1e8), which is why it is not used. Rank decisions of the two methods agree for `κ(J) < min(κ_H, κ_legacy)` (both keep) and `κ(J) > max(κ_H, κ_legacy)` (both drop) and may differ in between; the docs state both formulas.
  - A `UserWarning` is emitted when `κ(J) = √cond(H)` exceeds `1e4` (accuracy is then about `ε·κ²` or worse); Stage 3's study may change this.
- **Bounds (active set):** variables at a bound whose `g` points outward are held; the rest are solved as a reduced system and the trial is projected onto the box. Convergence test on the projected gradient. This is not Coleman–Li TRF and the docs say so. Dof for the covariance uses all `p` parameters, as the legacy code.
- **KKT tests** at the returned point, with the scaled gradient `ĝ_i = g_i / (√H_ii · √(2F))` (skip columns with `H_ii = 0`) and `τ = 1e-6` **[review]**: free variable → `|ĝ_i| ≤ τ`; at lower bound → `ĝ_i ≤ τ`; at upper bound → `ĝ_i ≥ −τ`; feasibility `lb ≤ θ ≤ ub` exactly. (Signs: `∇F = −g`, so a lower-bound multiplier is nonnegative when `g_i ≤ 0`.) Applied to: benchmark `bounded_exp`; a seeded randomized set in which 0, 1 and all parameters end up on a bound; a start already on a bound; a bounded start at the constrained optimum (must terminate with `success=True`, see the Newton-decrement note); a start outside the box (clipped); a bound that cuts through a flat valley. Equal bounds (`lb == ub`) are not tested: they are rejected by validation, as in the legacy code.
- **Stage 2 is two deliverables, 2a and 2b** **[my call]**. 2a: weights and the weighted covariance. 2b: bounds. If the randomized bounded set shows the active-set method stalling or missing SciPy's cost beyond the table's tolerance, the order of response is: (1) TRF-style scaling on the reduced system, (2) if that still fails, ship 2a only, with bounds raising `ValueError`, recorded as a follow-up. That is decided by the test results, not by asking again. Correctness tests run on CPU in CI; the GPU and 100M performance checks are manual benchmarks run before merging a stage and recorded in `benchmarks/baselines/`.
- **Stage 2b performance stop condition** (skipped if bounds are deferred): `bounded_exp` on CPU at 100M points is no slower than `curve_fit_large` by more than the 7% seen in the prototype; if it is, the write-up says why and 2b still passes only if the GPU result (prototype: 8× faster) holds.

## Stage 3 — out-of-memory sources and the conditioning study

### Sources **[review]**

`xdata`, `ydata` and (Stage 2) `sigma` may be:

- **(a) Array-likes** with `.shape`, `len` and slicing returning NumPy-convertible blocks: NumPy arrays, `np.memmap`, `h5py.Dataset`, `zarr.Array`, JAX arrays. Read chunk by chunk with no whole-array copy and no shuffle.
- **(b) A chunk factory.** Public invocation **[review]**: `curve_fit_large(f, factory, None, p0=..., method="exact_streaming", n_points=N_or_None, ...)`, i.e. the callable is passed as `xdata` and `ydata` is `None` (any other `ydata` raises `ValueError`). `factory(chunk_size) -> Iterator[tuple]` yields `(x, y)` or `(x, y, sigma)` blocks of at most `chunk_size` points (the last may be shorter). It must be **replayable**: each call starts a fresh pass over identical data in identical order. `n_points` (accepted only with a factory; added to the solver-keyword whitelist) is optional; if omitted the count is taken during pass 1 (the length is needed only for degrees of freedom and diagnostics, never for chunking). A one-shot generator cannot be detected reliably; a factory that yields different data on replay breaks exactness, and every later pass verifies the point count against pass 1 and raises on mismatch. Factories skip the `shape`/`len` validation used for arrays; instead each block is validated as it arrives (1-D, equal lengths among its parts, at most `chunk_size` points, finite unless `check_finite=False`). The first-chunk model-contract probe applies to factories as to arrays. Before Stage 3, a callable `xdata` raises `ValueError`.

This needs **no new dependency**: `h5py` is already a core dependency (`pyproject.toml:112`), memmap is NumPy, and Zarr is duck-typed with its test guarded by `pytest.importorskip` **[my call]**. A background thread reads the next chunk while the device processes the current one, and is kept only if per-pass wall time is at least 10% lower with it than without on the 1B-style configuration **[review]**.

### Gates for the 1B run (manual benchmark, not part of the test suite)

exp_decay at 1,000,000,000 points (16 GB of x and y in float64), written to an HDF5 or memmap file by a seeded chunked generator (per-chunk seeds, so the file is reproducible without ever holding 16 GB in memory). On the 16 GB GPU:

1. **Correctness, noiseless:** a noiseless copy recovers the generating parameters to `rtol = 1e-6` with cost below `1e-6 · n`. This proves exactness at scale without a reference solver.
2. **Correctness, noisy:** a noisy copy recovers the generating parameters within `5·√diag(pcov)`.
3. **Every observation contributes** **[review]** (parameter recovery alone cannot show this): (i) the solver counts points in pass 1 and the count equals `n` on every pass; (ii) after the fit, an independent **streaming** NumPy evaluation of the full cost at the returned parameters equals the solver's reported `cost` to relative `1e-9`; this evaluator is new test code that reads the source chunk by chunk (`benchmarks/compare/timing.py`'s `residual_cost` materializes full model and residual arrays and is not used at 1B); (iii) a sentinel block: the final 1,000 points of the last chunk get an offset `Δ` added to `y`, calibrated before the run so the *linearized* parameter displacement `H⁻¹ J_sᵀ Δ` (computed on host from that block at the generating parameters) has a relative size of `1e-3` (1000× the gate-1 tolerance `rtol = 1e-6` on parameters); the fitted parameters of the sentinel run must displace from the sentinel-free run by at least half of that prediction, so an implementation that skips the tail is caught.
4. **Memory:** peak host RSS below data size + 4 GB. RSS alone cannot prove the absence of a whole-array copy (a memmap legitimately maps up to the file size), so a smaller-scale test also instruments reads (below).
5. **Reporting, not gated:** wall-clock time, passes, and per-pass throughput (dependent on the OS page cache; the host has 62 GB of RAM).

### Memory instrumentation (automated, small scale)

A test fits a 50M-point memmap source with a 1M-point chunk under `tracemalloc` and a counting wrapper around the source, and asserts: no single read larger than the chunk size; total bytes read per pass equal to the data size (within one chunk); `tracemalloc` peak below `8×` the chunk bytes plus a fixed constant (NumPy registers its allocations with `tracemalloc`; XLA device allocations are not counted, which is why this checks host copies only). Marked `@pytest.mark.serial`.

### Conditioning study (pre-registered)

**Problems** with controlled `κ(J)`: two_exp with rate separation `δ ∈ {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}`; a degree-8 and a degree-12 polynomial on `[0, 1]` (Vandermonde); a sum of three close Gaussians. Seeds 0–4. `κ(J)` is measured by SVD of the Jacobian at the reference optimum.

**Two separate data sets, never mixed** **[review]**: (1) the *accuracy set*, `n = 2000` observations per problem and seed, on which the extended-precision reference and both candidates are run on **identical observations**; (2) the *scale set*, `n = 2×10⁴` (float64 candidates against the SciPy baseline only) and the benchmark problems at 100M (timing only). The decision rule is applied to the accuracy set; the scale set only informs the speed clause.

**Reference solution (accuracy set):** an extended-precision Gauss–Newton (`mpmath`, 50 digits) run to convergence on the same observations, rounded to float64 for comparison. It supplies parameters `θ_ref`, cost, covariance `Σ_ref` (pseudo-inverse over the directions its own singular values keep under the rank cutoff `λ > p·ε·λ_max`, evaluated in extended precision), and rank. Error norms: parameter error `max_i |θ_i − θ_ref,i| / max(|θ_ref,i|, 1e-12)`; covariance error `‖Σ − Σ_ref‖_F / ‖Σ_ref‖_F` over the retained directions; rank agreement is exact equality of the retained-direction count. `scipy.optimize.least_squares` at tight tolerances is a float64 baseline on both sets.

**Candidates:** normal equations (`JᵀJ`), and a TSQR accumulation that factors the **augmented** chunk matrix `[J | r]` as `Q·[[R, c],[0, ρ]]`, so each chunk contributes a `(p+1)×(p+1)` upper-triangular block, combined pairwise by QR of the stacked blocks. The current cost is `F = ½(‖c‖² + ρ²)` (`ρ²` alone is only the *minimized linearized* residual, which excludes the part of `r` lying in the Jacobian's column space), the gradient is `g = Rᵀc`, and the Gauss–Newton step solves `R δ = c` directly, which is what avoids squaring `κ(J)`. Because `R` can be rank-deficient (the zero-column and collinear-column cases Stage 0 requires to converge), the solve is rank-revealing (column-pivoted QR or SVD of the small `p×p` factor, truncated at the same cutoff as the covariance), and the Levenberg–Marquardt damping enters as augmented rows, `min ‖[R; √λ·D^{1/2}] δ − [c; 0]‖`, so no normal equations are reconstructed. The TSQR implementation lives in the study script and moves into the module only if the rule adopts it.

**Decision rule (fixed before running; correctness is mandatory, speed is not a veto)** **[review]**:

1. Normal equations **pass** if, on every accuracy-set problem and seed with `κ(J) ≤ 1e4`: parameter relative error versus the reference ≤ `1e-6`, covariance relative error ≤ `1e-3`, and the same rank decision as the reference.
2. If they pass: normal equations stay the default; the `κ(J) > 1e4` warning stays; TSQR is not added.
3. If they fail: TSQR becomes the default regardless of speed. If it is more than 1.5× slower than normal equations on the 100M benchmark problems, the cost is documented and normal equations may be offered as an explicit fast opt-in with the warning.

## Tests and process

- `tests/streaming/test_exact_lm.py` (small arrays; no marker needed) — Stage 0–2 behaviour: SciPy agreement, chunk invariance, the five defect regressions, weights, KKT, covariance parity, model-contract probe, `fit()` gate. Stage 0 writes it against the prototype (`pyproject.toml:513` puts the repo root on `sys.path`, as `tests/benchmarks/` already relies on); Stage 1 changes the import.
- `tests/streaming/test_exact_lm_large.py` — anything with arrays over 100K elements or subprocesses carries `@pytest.mark.serial`, and `@pytest.mark.slow` if over 1 s (project rule in `CLAUDE.md`). Includes the memory-instrumentation test.
- Legacy tests (`test_large_dataset_equivalence_flag.py`, `test_streaming_pr30_regressions.py`, `test_large_dataset.py`) must pass unchanged at every stage.
- Benchmarks: every performance claim is taken with prototype/legacy/new measured back-to-back in one session; the machine state is recorded (`free`, governor, GPU clocks) next to the numbers.
- Docs: CHANGELOG entry per stage that changes user-visible behaviour; an API reference entry for the new method. `CLAUDE.md` is gitignored and not updated (it is also stale: it lists `nlsq/interfaces/`, which no longer exists).
- Lint: `ruff check` and `ruff format` clean; the project's mypy configuration for new `nlsq/` modules.

## Risks and what could change the plan

- **Normal equations square `κ(J)`.** Mitigation is the Stage 3 study; until then the warning and documented limit apply.
- **Active-set bounds can stall or zigzag** where TRF would not. Stage 2's randomized bounded set exists to find this; the response order is fixed above.
- **GPU numbers depend on transfer policy and machine state.** Hence the back-to-back rule and the resident/streamed distinction in every reported number.
- **JAX async dispatch.** The host reads `(F, g, H)` after every pass (a tiny sync). Overlap of the next chunk's transfer with compute is measured against the 10% gate, not assumed.
- **Model contract.** A non-pointwise model passes the probe but is not exact; documented, cannot be fully enforced.

## Resolved deferrals (three-model discussion, round 1) **[review]**

These were left open by the requester for the reviewers to decide. Each is a decision for this spec, with the trigger for revisiting it.

| Item | Decision | Why (evidence) | Revisit when |
|---|---|---|---|
| 1. Promote to the default for `curve_fit_large` | **Later**, not in this spec | Legacy tests assert GLS behaviour and `equivalent_to_full_batch` means `n_chunks == 1` (`large_dataset.py:1803`); `fit()` also reaches `curve_fit_large` (`nlsq/__init__.py:671`), so a default change reaches `fit()` users | Stage 3 complete and the study resolved; weights and bounds parity recorded; the flag redefined; legacy GLS tests migrated; one release with the method opt-in and no open correctness bug |
| 2. Retire `AdaptiveHybridStreamingOptimizer` | **Later**, separate spec | Still used by `fit()` presets and workflow code and by `curve_fit` (`nlsq/core/minpack.py:782`, `:1082`, `:4446`); it provides checkpointing, multistart and callbacks that the exact method does not | The exact method covers those features, and item 1 is done |
| 3. Expose through `nlsq.fit()` | **No for now, with an explicit gate** | Codex and Agy disagreed (Agy: trivial wiring). The code decides: `fit()` forwards `method` to `curve_fit_large` only for large inputs (`nlsq/__init__.py:671`) and to `curve_fit` otherwise, which would reject the name; and the large route returns a tuple while `fit()`'s other routes return `CurveFitResult`. So `fit(method="exact_streaming")` raises `ValueError` pointing to `curve_fit_large`, tested | `fit()`'s return contract for large inputs is decided |
| 4. Expose through the CLI and GUI | **Later** | The CLI reaches the core through `WorkflowRunner → curve_fit` (`nlsq/cli/workflow_runner.py:407`) and the GUI through `FitWorker → fit_adapter → core.fit` (`nlsq/gui_qt/adapters/fit_adapter.py:386`); both need routing, progress and cancellation support | Item 3 is done and the method supports a progress/cancel hook |
| 5. Multi-GPU sharding | **No** for this spec | One GPU finishes 100M exp_decay in 0.66 s; the reduction is p×p so communication is cheap, but there is no second GPU to validate on and the single-device 1B profile is not yet known | Two or more GPUs are available and the single-GPU 1B run is compute-bound, not transfer-bound |
| 6. Correlated (2-D) `sigma` | **No** for this scope | Dense Σ cannot be streamed (O(N²)); ordinary fitting uses a full Cholesky and the chunked path already rejects it (`large_dataset.py:2689`). Structured covariances (banded, AR(1)) can be whitened chunk-wise and are a separate design | A concrete structured-covariance use case exists |

## Left to the builder

Internal structure of `exact_lm.py` (class versus functions), the exact Nielsen constants beyond those stated, how chunk size is derived from the budget, logging format, and the file layout of the new tests, provided the interfaces and acceptance conditions above hold.
