# Deep Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four duplicated pieces of knowledge in NLSQ with four deep modules, without changing numerical behavior.

**Architecture:** Four independent parts, each its own branch and PR, in this order: A `NormalEquations`, B `prepare_problem`, C covariance, D `build_large_strategy`. Every part starts with characterization tests that pass on current code, then moves code behind the new interface (never a second pipeline beside the first), then repoints callers and tests.

**Tech Stack:** Python 3.12+, JAX (float64), pytest (xdist; `@pytest.mark.serial`/`slow` rules from `CLAUDE.md`), ruff, mypy, `uv`/`make`.

**Spec:** `docs/superpowers/specs/2026-11-01-deep-modules-design.md` (approved; read it first, especially sections 3 and 9).

## Global Constraints

- Every part is **behavior-preserving**. Not in this work: hybrid sigma support; hybrid `pinv` becoming all-inf + `OptimizeWarning`; unifying rank cutoffs (`eps*max(m,p)*s0` on J vs `p*eps*lambda_max` on H stay as they are).
- One branch and one PR per part (A, B, C, D = spec opportunities 1, 3, 4, 2), **strictly sequential**: Parts B, C and D all edit `nlsq/core/minpack.py` and `nlsq/__init__.py`, so each part branches from `main` only after the previous part has merged (or is rebased onto it).
- `PassStats.cost = 0.5 * sum(r^2)`; hybrid `residual_sum_sq = sum(r^2)`. The factor-2 conversion happens in exactly one named place (Task A3).
- Covariance helpers take `cost = 0.5 * sum(r^2)` (`res.cost`); each applies the factor 2 once internally (Part C). `exact_lm.compute_covariance` keeps returning a bare `ndarray`.
- Hybrid keeps its existing Jacobian (`_jacobian_fn_compiled`, per-point `jacrev`) through the `jacobian=` hook; no Jacobian is replaced by `jacfwd`.
- Checkpoint format and field names (`phase2_JTJ_accumulator`, `phase2_JTr_accumulator`) are unchanged; old `.h5` checkpoints must load.
- Hybrid benchmark must stay within 5% wall time (CPU and GPU, sizes the hybrid bench already uses); exact-streaming 100M and 1B gates unchanged.
- Never delete code or tests on the strength of a search of `nlsq/` alone; sweep `tests/` and state a reason in the commit message.
- Tests allocating over 100K elements, or spawning JAX subprocesses, get `@pytest.mark.serial`. Model functions use `jnp`, not `np`.
- Per part: `make verify` clean, `graphify update .` after code changes, changelog entry, signed commits (GPG signing works only outside the sandbox in this environment).

## Review Focus

- Padded tail with a non-finite model value (e.g. `log` of a padded zero) must not leak into `J^T J` through the shared kernel; expect finite stats equal to the unpadded result (Task A3).
- Partially rank-deficient Jacobian (duplicate column): standard and exact paths return a **finite** pseudo-inverse `pcov` with zeroed dropped directions; all-`inf` plus one `OptimizeWarning` when no direction survives or `n_eff <= p` without `absolute_sigma`; non-finite input is handled differently by the J-based version (all-`inf` only for a non-finite singular value; finite `J` with a NaN cost gives NaN `pcov`, no warning) and the H-based version (non-finite `H` or final `pcov` gives all-`inf` plus warning), and both are preserved (Tasks C1, C2).
- Hybrid group-variance penalties enabled: production (`adaptive_hybrid.py` ~:3090) and `phases/gauss_newton.py` (~:557) scale penalty derivatives differently; each keeps its own behavior (Tasks A1, A3, A4).
- NaN rows omitted with `nan_policy="omit"` plus a `data_mask`: degrees of freedom use `sum(mask)`, not the padded length (Task B2).
- One-parameter model (`p=1`) and chunk size 1 or `n=1`: shapes stay `(1,)` / `(1,1)`, no squeeze errors (Task A2).
- Empty `x`/`y` raises `ValueError` from every path that goes through `NormalEquations` and `prepare_problem` (Tasks A2, B2).

---

# Part A: `NormalEquations` (PR 1, branch `refactor/normal-equations`)

### Task A1: Characterization tests (no source change)

**Files:**
- Create: `tests/streaming/test_normal_equations_characterization.py`
- Reference: `nlsq/streaming/adaptive_hybrid.py` (`AdaptiveHybridStreamingOptimizer.fit`, `_accumulate_jtj_jtr` at :2428), `nlsq/streaming/exact_lm.py` (`exact_lm_fit`, `run_pass`), `nlsq/streaming/phases/checkpoint.py`

**Interfaces:**
- Produces: golden values (recorded once from current code, committed as constants in the test file) used by Tasks A2-A4 to prove equivalence: for a fixed-seed exponential model `a*exp(-b*x)` with 20_000 points and `p0=[1.0, 0.5]`: exact_lm `x`, `cost`, `pcov`; hybrid Phase 2 `x`, `residual_sum_sq`, `pcov` for each of {group-variance penalties off, on} x {scan forced, loop forced}; one `_accumulate_jtj_jtr` call's `(JTJ, JTr, chunk_cost)` on a 1_000-point chunk, for both `adaptive_hybrid.py` and `phases/gauss_newton.py` (characterized independently).

- [ ] **Step 1:** Write `test_exact_lm_golden`, `test_hybrid_phase2_golden` (parametrized over penalties and scan/loop), `test_hybrid_accumulate_chunk_golden`, `test_phases_accumulate_chunk_golden` asserting `np.testing.assert_allclose(..., rtol=1e-8, atol=1e-10)` against constants captured by running current code once (print them, paste in). Force scan or loop through the same switch production uses (`adaptive_hybrid.py` ~:643).
- [ ] **Step 2:** Write `test_hybrid_checkpoint_roundtrip`: run `fit` with checkpointing to a tmp dir, load via `CheckpointManager`, and assert both `phase2_*_accumulator` fields equal the values saved (`assert_allclose`). Then write `test_hybrid_phase2_resume_continues_from_checkpoint`: on a fresh optimizer call `_setup_normalization`, `_load_checkpoint(path)`, then `_run_phase2_gauss_newton` directly (not `fit()`, which restarts normalization and warmup at ~:4345-4367), and assert the parameters at the start of that Phase 2 equal the restored checkpoint parameters and the continued final `x` matches the uninterrupted run within the golden tolerance.
- [ ] **Step 3:** Write `test_hybrid_padded_tail_finite`: a chunk whose padded tail makes the model non-finite (`jnp.log(x)` with padded zeros) through `phases/gauss_newton.py::_accumulate_jtj_jtr(..., valid_size=...)`; assert finite results equal to the unpadded call.
- [ ] **Step 4:** Run `uv run pytest tests/streaming/test_normal_equations_characterization.py -v`. Expected: all PASS on unmodified code (hybrid fit tests marked `slow`; the 20_000-point tests do not need `serial`).
- [ ] **Step 5:** Commit `test: characterize streaming normal-equation accumulation`.

### Task A2: Extract `normal_equations.py` (pure move)

**Files:**
- Create: `nlsq/streaming/normal_equations.py`
- Modify: `nlsq/streaming/exact_lm.py` (remove moved definitions, import them back)
- Test: `tests/streaming/test_normal_equations.py`; existing `test_exact_lm*.py` must pass untouched.

**Interfaces:**
- Produces (moved verbatim from `exact_lm.py`, same signatures): `PassStats`, `make_chunk_stats(model, weighted=False)`, `iter_blocks`, `FactorySource`, `is_chunk_factory`, `run_pass(stats, theta, source, chunk, *, check_finite=False, check_sigma=False) -> PassStats`. New thin class:
  `class NormalEquations: __init__(self, model: Callable[..., Any], *, weighted: bool = False, chunk: int, jacobian: Callable[[Any, Any], Any] | None = None)`; `accumulate(self, theta, source, *, check_finite=False, check_sigma=False) -> PassStats`; `chunk_stats(self, theta, x, y, mask, sigma=None) -> tuple[jax.Array, jax.Array, jax.Array]` (the jitted per-chunk `(F, g, H)`). `jacobian=None` keeps today's chunk-wide `jacfwd`; a callable `(theta, x_chunk) -> (n_chunk, p)` replaces it. No cost-only method (dropped in the spec).
- Consumes: nothing new. `exact_lm.py` keeps re-exporting every moved name so `from nlsq.streaming.exact_lm import run_pass` still works.

- [ ] **Step 1:** Write `test_normal_equations.py::test_accumulate_matches_run_pass` (same stats as `run_pass` on a 3-chunk source), `test_single_parameter_shapes` (`p=1`: `H.shape == (1,1)`, `g.shape == (1,)`), `test_empty_source_raises` (`ValueError`, "non-empty"), `test_chunk_size_one` (`n=5, chunk=1` equals `chunk=5`), `test_custom_jacobian_hook_used` (a reverse-only model via `jax.custom_vjp` fails with default `jacfwd` but works with `jacobian=` supplying the Jacobian, and the stats match an analytic `J^T J`).
- [ ] **Step 2:** Run them; expected FAIL (module missing).
- [ ] **Step 3:** Move the listed definitions into `normal_equations.py`, add `NormalEquations` as a thin wrapper holding the jitted kernel from `make_chunk_stats`; `exact_lm.py` imports from the new module. Do not edit any numerical expression.
- [ ] **Step 4:** Run `uv run pytest tests/streaming/test_normal_equations.py tests/streaming/test_exact_lm.py tests/streaming/test_exact_lm_sources.py tests/streaming/test_exact_lm_dispatch.py tests/streaming/test_normal_equations_characterization.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit `refactor(streaming): extract NormalEquations from exact_lm`.

### Task A3: Route hybrid `_accumulate_jtj_jtr` through the shared kernel

**Files:**
- Modify: `nlsq/streaming/adaptive_hybrid.py` (`_accumulate_jtj_jtr` :2428; consumers :3041-3141, :3379-3446, :3545-3689, :4439-4469)
- Test: `tests/streaming/test_normal_equations_characterization.py` (extend)

**Interfaces:**
- Consumes: `NormalEquations.chunk_stats` from A2, built once in the hybrid optimizer's setup from `self.normalized_model` (unweighted) with `jacobian=self._compute_jacobian_chunk`-equivalent `lambda theta, x: self._jacobian_fn_compiled(theta, x)` (falling back to the inline `vmap(jacrev)` when it is `None`), so the derivative code is unchanged.
- Produces: `_accumulate_jtj_jtr(x_chunk, y_chunk, params, JTJ_prev, JTr_prev) -> tuple[jnp.ndarray, jnp.ndarray, float]` with the **same signature and meaning as today**: third element is `sum(r^2)` for the chunk, obtained as `2.0 * F` from the kernel in this one function (add a comment: "kernel cost is 0.5*sum r^2"). It still sets `self.phase2_JTJ_accumulator` / `self.phase2_JTr_accumulator`.

- [ ] **Step 1:** Add `test_hybrid_chunk_cost_convention`: for a known chunk, `_accumulate_jtj_jtr` third return equals `float(jnp.sum((y - model)**2))` (not half).
- [ ] **Step 2:** Run the characterization suite; expected: golden tests PASS, new test PASS (it pins the existing convention before the change).
- [ ] **Step 3:** Replace the body of `_accumulate_jtj_jtr` with a call to `self._normal_eq.chunk_stats(params, x_chunk, y_chunk, mask)` where `mask` is all-True for unpadded chunks; add the accumulation `JTJ_prev + H`, `JTr_prev + g`. Keep the caller code unchanged in this task.
- [ ] **Step 4:** Run `uv run pytest tests/streaming -k "hybrid or phases or normal_equations" -n 2`. Expected: PASS within the golden tolerances (no Jacobian change, so differences come only from summation order; `rtol=1e-8` should hold, and any loosening needs a stated reason in the commit message).
- [ ] **Step 5:** Commit `refactor(streaming): hybrid accumulation uses shared chunk kernel`.

### Task A3b: Collapse hybrid's four redundant chunk loops

**Files:**
- Modify: `nlsq/streaming/adaptive_hybrid.py` (loops at ~:3039-3065, ~:3397-3410 with progress reporting, ~:3544-3570, ~:4448-4470; three of them have a scan/loop branch)
- Test: `tests/streaming/test_normal_equations_characterization.py` (goldens from A1 unchanged)

**Interfaces:**
- Consumes: A3's `_accumulate_jtj_jtr`.
- Produces: private `_accumulate_pass(self, params: jnp.ndarray, x_data: Any, y_data: Any, *, on_chunk: Callable[[int, int], None] | None = None) -> tuple[jnp.ndarray, jnp.ndarray, float]` returning `(JTJ, JTr, residual_sum_sq)` with `residual_sum_sq = sum(r^2)` (same convention as today), walking `x_data`/`y_data` in plain `self.config.chunk_size` slices exactly as the loops do now (no bucket padding: that exists only in `phases/gauss_newton.py`), and setting `self.phase2_JTJ_accumulator` / `self.phase2_JTr_accumulator` once at the end of the pass. `on_chunk(chunk_idx, n_chunks)` is the hook the ~:3397 caller uses for its every-10%/50-chunks progress reporting. The four call sites call it; the group-variance penalty terms stay at the call sites.

- [ ] **Step 1:** Run the A1 goldens; expected PASS (baseline).
- [ ] **Step 2:** Implement `_accumulate_pass` by lifting one existing loop body unchanged; keep the scan branch for now (removed only in A4 if the gate passes).
- [ ] **Step 3:** Replace the four loops with calls to it (the ~:3397 caller passes `on_chunk`). Run `uv run pytest tests/streaming -k "hybrid or normal_equations" -n 2`. Expected: PASS within golden tolerances; the checkpoint test (A1 step 2) still passes with values equal to the pass-end accumulators.
- [ ] **Step 4:** Commit `refactor(streaming): single hybrid accumulation pass helper`.

### Task A4: Remove duplicate accumulators; route `phases/gauss_newton.py`

**Files:**
- Modify: `nlsq/streaming/adaptive_hybrid.py` (delete `_accumulate_jtj_jtr_scan` :856 and the scan-vs-loop branches at :3039, :3544, :4448 if the Task A5 gate passes), `nlsq/streaming/phases/gauss_newton.py:630`
- Test: `tests/streaming/test_phases.py`, `tests/streaming/test_phases_extended.py` (unchanged), `tests/regression/test_phase2_memory_validation_optimizations.py`

**Interfaces:**
- Consumes: A3's adapter. `phases/gauss_newton.py::_accumulate_jtj_jtr(x_chunk, y_chunk, params, JTJ_prev, JTr_prev, valid_size=None)` keeps its signature; `valid_size` becomes the kernel `mask`.

- [ ] **Step 1:** Run the A5 benchmark first (Task A5 steps 1-2) with scan **forced** versus the migrated block loop **forced** (on GPU both `main` and the branch auto-select scan, so default runs would not time the loop); this task proceeds with deletion only if the loop is within 5% on CPU and GPU.
- [ ] **Step 2:** Route `phases/gauss_newton.py::_accumulate_jtj_jtr` through `NormalEquations.chunk_stats`, converting `valid_size` to a boolean mask; run `tests/streaming/test_phases*.py`. Expected: PASS.
- [ ] **Step 3:** If the gate passed, delete `_accumulate_jtj_jtr_scan` and remove the scan branch inside `_accumulate_pass` (its four call sites keep calling `_accumulate_pass`); otherwise keep scan as a private strategy inside `NormalEquations` (not a second public method). `grep -rn "_accumulate_jtj_jtr_scan" nlsq tests` must show no stale references (update tests that call it, stating the reason in the commit).
- [ ] **Step 4:** Run `uv run pytest tests/streaming tests/regression -n 2`. Expected: PASS.
- [ ] **Step 5:** Commit `refactor(streaming): drop duplicate J^T J accumulators`.

### Task A5: Benchmark gate and changelog

**Files:**
- Modify: `CHANGELOG.md`; record results in the PR description.
- Reference: the `bench` skill and `benchmarks/` scripts used for hybrid and the 100M/1B exact-streaming gates.

- [ ] **Step 1:** On `main`, run the hybrid benchmark (CPU and GPU) with the scan strategy forced and again with the loop strategy forced (same switch as production, `adaptive_hybrid.py` ~:643); record all four wall times.
- [ ] **Step 2:** On the branch, repeat the same four runs. Expected: the migrated loop is within 5% of scan, and neither strategy regresses more than 5% against its `main` time.
- [ ] **Step 3:** Re-run the 100M and 1B exact-streaming gates. Expected: unchanged pass.
- [ ] **Step 4:** Run `make verify`; run `graphify update .`; add changelog line "Internal: shared NormalEquations module for streaming solvers; no behavior change".
- [ ] **Step 5:** Commit, push, open a draft PR titled `refactor(streaming): shared NormalEquations module`.

---

# Part B: `prepare_problem` (PR 2, branch `refactor/prepare-problem`)

### Task B1: Characterization of input preparation order

**Files:**
- Create: `tests/core/test_prepare_problem_characterization.py`
- Reference: `nlsq/core/minpack.py` `_prepare_curve_fit_inputs` (:4097-4258), `_setup_data_mask_and_padding` (:3851), `_setup_sigma_transform` (:3326), `_curve_fit_hybrid_streaming` (:4291, duplicate at ~4374-4407)

**Interfaces:**
- Produces: tests that pin, through `curve_fit` and `curve_fit(method="hybrid_streaming")` (not `curve_fit_large`, which rejects `nan_policy` at `nlsq/__init__.py` ~:1116): NaN rows with `nan_policy="omit"` give the same `popt` as the pre-filtered data; `data_mask` stays aligned after omission; `pcov` degrees of freedom use `sum(mask)`; the hybrid path accepts `nan_policy` once and still raises `TypeError` for a genuinely unknown keyword.

- [ ] **Step 1:** Write `test_omit_matches_prefiltered`, `test_mask_alignment_after_omit`, `test_dof_uses_mask_sum_not_padded_length`, `test_hybrid_nan_policy_accepted_once`, `test_hybrid_unknown_kwarg_still_raises_typeerror`, `test_hybrid_data_mask_still_rejected` (`NotImplementedError`), `test_hybrid_does_not_pad`, `test_kwargs_leftovers_visible_to_caller` (keys consumed by preparation are gone, others remain).
- [ ] **Step 2:** Run `uv run pytest tests/core/test_prepare_problem_characterization.py -v`. Expected: PASS on current code.
- [ ] **Step 3:** Commit `test: characterize curve_fit input preparation`.

### Task B2: Introduce `prepare_problem` by moving the code

**Files:**
- Create: `nlsq/core/problem.py`
- Modify: `nlsq/core/minpack.py` (`_prepare_curve_fit_inputs` becomes a thin call; consumers at :5236, :5256, :5313-5319)
- Test: `tests/core/test_problem.py`; repoint `tests/core/test_minpack_prepare_inputs_comprehensive.py`, `tests/core/test_minpack_coverage.py:1139`, `tests/core/test_pr29_regressions.py:139`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class PreparedProblem: n: int; p0: np.ndarray; lb: np.ndarray; ub: np.ndarray; x: Any; y: Any; sigma: Any | None; mask: np.ndarray | None; method: str; m_real: int; n_effective: int; len_diff: int; should_pad: bool; none_mask: bool` and `def prepare_problem(f, x, y, p0, bounds, sigma, data_mask, check_finite, nan_policy, method, solver, batch_size, kwargs, ctx: PrepareContext = PrepareContext()) -> PreparedProblem`, where `@dataclass(frozen=True) class PrepareContext: flength: float | None = None; use_dynamic_sizing: bool = False; enable_stability: bool = False; validator: Any | None = None; logger: Any | None = None` carries the `CurveFit` state the steps read today (padding, `enable_stability` at `minpack.py` ~:3663 and ~:3739). `kwargs` is mutated in place exactly as today. `method`, `solver` and `batch_size` default to `curve_fit`'s own defaults (`method=None`, `batch_size=None`, `solver` as in `CurveFit.curve_fit`), so callers that do not use them (the hybrid path) pass nothing. Together the fields cover every element of the legacy 13-tuple (`n, p0, xdata, ydata, data_mask, method, lb, ub, m, len_diff, should_pad, none_mask, sigma`). Ordering is omission, then method selection, then padding (spec section 4). `n_effective = int(mask.sum())` when a mask exists, else `m_real`; callers use `n_effective` where they used `ysize`; `none_mask` is still consumed at `minpack.py` ~:5345.

- [ ] **Step 1:** Write `test_problem.py::test_n_effective_counts_unmasked_rows`, `test_empty_input_raises_value_error`, `test_omit_realigns_mask`, `test_len_diff_is_padding_length`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Move the body of `_prepare_curve_fit_inputs` and the mask/padding code into `prepare_problem` unchanged in order; `_prepare_curve_fit_inputs` returns what it did before by unpacking `PreparedProblem` (keeps the 13-tuple for any remaining caller until Step 4).
- [ ] **Step 4:** Repoint the three listed test modules at `prepare_problem`; run `uv run pytest tests/core -n 2`. Expected: PASS. Remove `_prepare_curve_fit_inputs` and the discarded `_should_pad` return only if `grep -rn "_prepare_curve_fit_inputs" nlsq tests` is empty.
- [ ] **Step 5:** Commit `refactor(core): extract prepare_problem`.

### Task B3: Hybrid path uses `prepare_problem`

**Files:**
- Modify: `nlsq/core/minpack.py` `_curve_fit_hybrid_streaming` (:4374-4407)
- Test: B1 tests

- [ ] **Step 1:** Replace the hybrid path's own `nan_policy`/`check_finite`/p0/bounds handling with one `prepare_problem(..., ctx=PrepareContext(flength=None, use_dynamic_sizing=False))` call, so no padding is introduced; the `data_mask` `NotImplementedError` (~:4367) stays before it. The hybrid path pops `nan_policy` (and anything `prepare_problem` consumes) **before** its leftover-keyword `TypeError` check (`minpack.py` ~:4438), so a bare `get` does not make `nan_policy` fail that check.
- [ ] **Step 2:** Run `uv run pytest tests/core tests/streaming -n 2` and B1 tests. Expected: PASS, including `test_hybrid_unknown_kwarg_still_raises_typeerror`.
- [ ] **Step 3:** `make verify`, `graphify update .`, changelog "Internal: prepare_problem; hybrid streaming no longer consumes nan_policy twice"; commit; draft PR.

---

# Part C: Covariance (PR 3, branch `refactor/covariance`)

### Task C1: Golden covariance tests per site (no source change)

**Files:**
- Create: `tests/core/test_covariance_characterization.py`
- Reference: `nlsq/core/minpack.py:3419` (`_compute_covariance`, SVD of J), `nlsq/streaming/exact_lm.py:748` (`compute_covariance`, eigh of H), `nlsq/streaming/adaptive_hybrid.py:3727` (bare `pinv`), `nlsq/streaming/large_dataset.py:2455` (GLS)

**Interfaces:**
- Produces: for a well-conditioned problem, a partially rank-deficient one (duplicate parameter column) and a fully degenerate one (all-zero Jacobian, or `n <= p` without `absolute_sigma`), recorded `pcov` golden values per site, plus the warning contract as it is today: standard and exact return a **finite pseudo-inverse `pcov` with zeroed dropped directions** for the duplicate-column case; they return all-`inf` plus exactly one `OptimizeWarning` only for the fully degenerate cases; exact additionally emits a `UserWarning` when `sqrt(cond(H)) > 1e4`; the two **differ on non-finite input and that is preserved**: standard with finite `J` and a NaN cost returns NaN `pcov` with no warning, exact with non-finite `H` or non-finite final `pcov` returns all-`inf` plus `OptimizeWarning`; **hybrid gives its current finite `pcov` (pinned deliberately)**.

- [ ] **Step 1:** Write `test_standard_golden`, `test_exact_golden`, `test_hybrid_golden`, `test_*_partial_rank_deficient_contract`, `test_*_fully_degenerate_contract`, `test_standard_nonfinite_cost_gives_nan_no_warning`, `test_exact_nonfinite_input_gives_inf_and_warns` (four sites incl. GLS golden on the well-conditioned problem).
- [ ] **Step 2:** Run; expected PASS on current code.
- [ ] **Step 3:** Commit `test: characterize covariance contracts`.

### Task C2: One covariance module for the J-based and H-based sites

**Files:**
- Create: `nlsq/core/covariance.py`
- Modify: `nlsq/core/minpack.py:3419`, `nlsq/streaming/exact_lm.py:748` (`compute_covariance` stays importable as an alias)
- Test: `tests/core/test_covariance.py`; `tests/core/test_shared_jit_kernels.py:46` (`covariance_svd`) must still pass

**Interfaces:**
- Produces: `def pcov_from_jacobian(J, cost: float, n_eff: int, p: int, absolute_sigma: bool, svd: Callable[[Any], tuple[Any, Any]]) -> tuple[np.ndarray, bool]` (cutoff `eps*max(n_eff,p)*s0`; `svd(J) -> (s, VT)` is injected because `CurveFit._compute_covariance` calls its instance hook `self.covariance_svd` at `minpack.py` ~:3449, which it forwards unchanged) and `def pcov_from_normal(H, cost: float, n_eff: int, p: int, absolute_sigma: bool) -> tuple[np.ndarray, bool]` (cutoff `p*eps*lambda_max`, eigh). Second element is `True` when the all-`inf` fallback fired (not for a partial-rank pseudo-inverse). `cost` is `0.5 * sum(r^2)`; each function applies the factor 2 once internally. Behavior is exactly today's (see C1). Hybrid's `pinv` and the GLS site are **not** routed here. `exact_lm.compute_covariance(H, cost, n, p, absolute_sigma) -> np.ndarray` stays as a wrapper returning only `pcov`; `CurveFit._compute_covariance(res, ysize, p0, absolute_sigma)` keeps its signature and delegates.

- [ ] **Step 1:** Write `test_covariance.py::test_jacobian_and_normal_agree_when_well_conditioned` (rtol 1e-8), `test_duplicate_column_returns_finite_pseudoinverse`, `test_fully_degenerate_returns_inf_and_warns_once` for both functions, `test_absolute_sigma_skips_scaling`, `test_exact_compute_covariance_still_returns_ndarray`, `test_compute_covariance_uses_instance_svd_hook` (replace `covariance_svd` on a `CurveFit` instance with a recording stub, call `_compute_covariance`, assert the stub was invoked and its result used).
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Move the two bodies unchanged into `covariance.py`; make the two sites call them; keep `exact_lm.compute_covariance` as the ndarray-returning wrapper.
- [ ] **Step 4:** Run `uv run pytest tests/core tests/streaming -n 2` plus C1. Expected: PASS with identical goldens (including hybrid finite `pcov`).
- [ ] **Step 5:** `make verify`, `graphify update .`, changelog "Internal: shared covariance helpers; no change to cutoffs"; commit; draft PR.

---

# Part D: `build_large_strategy` (PR 4, branch `refactor/large-strategy`)

### Task D1: Routing table test (no source change)

**Files:**
- Create: `tests/streaming/test_large_strategy_routing.py`
- Reference construction sites: `nlsq/__init__.py:651, 1146, 1253`; `nlsq/streaming/large_dataset.py:1900, 3053`; `nlsq/core/minpack.py:782, 812, 1030, 1091, 1764, 1829, 4446, 4658`

**Interfaces:**
- Produces: a table of `(entry, kwargs) -> (optimizer class used, return type)` for the 13 sites, asserted by monkeypatching the two classes to record construction and delegate to the real ones. Not every site is reachable through a public call: every `WORKFLOW_PRESETS` key is rejected before `_fit_with_preset` (`minpack.py` ~:528), so its legacy constructors cannot be hit publicly. The table therefore has two kinds of rows: public calls (`curve_fit`, `curve_fit_large`, `fit(preset=...)`) and **direct-helper calls** (e.g. `CurveFit._fit_with_preset`) for sites with no public path; the public rejection of those preset keys stays covered by its own test. Also records each site's return shape (tuple, `OptimizeResult`, dict or fitter result).

- [ ] **Step 1:** Write one parametrized test over the 13 rows asserting the recorded class and return type, plus `test_legacy_preset_keys_rejected_publicly`.
- [ ] **Step 2:** Run `uv run pytest tests/streaming/test_large_strategy_routing.py -v`. Expected: PASS on current code.
- [ ] **Step 3:** Commit `test: pin large-dataset routing table`.

### Task D2: Strategy builder and site migration

**Files:**
- Create: `nlsq/streaming/strategy.py`
- Modify: the 13 sites listed in D1
- Test: D1 table unchanged; `tests/streaming/test_strategy.py`

**Interfaces:**
- Produces: `class Strategy(Protocol): def fit(self, f, x, y, p0, **common) -> Any` and `def build_large_strategy(kind: Literal["hybrid", "chunked"], config, **common) -> Strategy`. `kind` is the path the caller already chose; the function never decides routing. Each `Strategy` owns the **input** kwargs translation (e.g. "has no gtol/ftol/xtol", `minpack.py` ~771) and returns **the underlying engine's native result unchanged** (dict from the hybrid optimizer, the fitter's result for chunked); callers keep their own unpacking, because the 13 sites consume different shapes. In scope: all 13 construction sites, including the two inside `_fit_global_multistart` (`minpack.py` ~:1764, ~:1829) as construction only. Out of scope: the global-optimization logic itself and `_curve_fit_exact_streaming`.

- [ ] **Step 1:** Write `test_strategy.py::test_hybrid_drops_unsupported_tolerances`, `test_chunked_returns_native_result_unchanged`, `test_unknown_kind_raises_value_error`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement the two strategies by moving each site's translation code; migrate sites one file at a time (`large_dataset.py`, then `__init__.py`, then `minpack.py`), running the D1 table after each file.
- [ ] **Step 4:** Run `uv run pytest tests/streaming tests/core -n 2` and D1. Expected: PASS, table unchanged.
- [ ] **Step 5:** `make verify`, `graphify update .`, changelog "Internal: build_large_strategy; routing unchanged"; commit; draft PR.

---

## Self-review notes

- Spec coverage: sections 3 (A1-A5, traps 1-5 -> A3 cost test, A1 checkpoint test, A1 goldens, A2 shapes plus A3 mask, A5 bench), 4 (B1-B3), 5 (D1-D2), 6 (C1-C2), 7 (global constraints), 8 deferred (not planned), 9 defaults (global constraints, A4 route-not-delete).
- Open for the implementer: golden constants come from running current code once (A1, B1, C1); tolerances start at `rtol=1e-8`; any loosening needs a stated reason.
- Review iteration 1 (Codex + Agy) changed: covariance contract (partial rank is finite), `jacobian=` hook and no `jacfwd` swap, dropped `cost()` method, `PreparedProblem` fields and `prepare_problem` parameters, `nan_policy` consumed before the unknown-keyword check, sequential PRs, forced scan-vs-loop benchmark, penalty-enabled goldens, new Task A3b, `Strategy.fit` returns native results, global-multistart construction sites in scope.
- Review iteration 2 changed: A3b covers four loops with plain slices and a progress hook (bucket padding is `phases/` only); `PrepareContext` carries `CurveFit` state and hybrid passes no padding; the J-based vs H-based non-finite behaviors are preserved and tested; A4 step 3 refers to `_accumulate_pass`; spec clarifies `chunk_stats` is the unified kernel.
- Risk: the hybrid chunk loops carry penalty terms and progress reporting at the call sites; A3b must lift an existing loop unchanged rather than rewrite it.
