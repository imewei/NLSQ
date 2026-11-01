# Deep Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace four duplicated pieces of knowledge in NLSQ with four deep modules, without changing numerical behavior.

**Architecture:** Four independent parts, each its own branch and PR, in this order: A `NormalEquations`, B `prepare_problem`, C covariance, D `build_large_strategy`. Every part starts with characterization tests that pass on current code, then moves code behind the new interface (never a second pipeline beside the first), then repoints callers and tests.

**Tech Stack:** Python 3.12+, JAX (float64), pytest (xdist; `@pytest.mark.serial`/`slow` rules from `CLAUDE.md`), ruff, mypy, `uv`/`make`.

**Spec:** `docs/superpowers/specs/2026-11-01-deep-modules-design.md` (approved; read it first, especially sections 3 and 9).

## Global Constraints

- Every part is **behavior-preserving**. Not in this work: hybrid sigma support; hybrid `pinv` becoming all-inf + `OptimizeWarning`; unifying rank cutoffs (`eps*max(m,p)*s0` on J vs `p*eps*lambda_max` on H stay as they are).
- One branch and one PR per part (A, B, C, D). Branch from `main`, not from each other, except where a task says otherwise.
- `PassStats.cost = 0.5 * sum(r^2)`; hybrid `residual_sum_sq = sum(r^2)`. The factor-2 conversion happens in exactly one named place (Task A3).
- Checkpoint format and field names (`phase2_JTJ_accumulator`, `phase2_JTr_accumulator`) are unchanged; old `.h5` checkpoints must load.
- Hybrid benchmark must stay within 5% wall time (CPU and GPU, sizes the hybrid bench already uses); exact-streaming 100M and 1B gates unchanged.
- Never delete code or tests on the strength of a search of `nlsq/` alone; sweep `tests/` and state a reason in the commit message.
- Tests allocating over 100K elements, or spawning JAX subprocesses, get `@pytest.mark.serial`. Model functions use `jnp`, not `np`.
- Per part: `make verify` clean, `graphify update .` after code changes, changelog entry, signed commits (GPG signing works only outside the sandbox in this environment).

## Review Focus

- Padded tail with a non-finite model value (e.g. `log` of a padded zero) must not leak into `J^T J` through the shared kernel; expect finite stats equal to the unpadded result (Task A3).
- Rank-deficient Jacobian (duplicate column): standard and exact paths return all-`inf` `pcov` plus one `OptimizeWarning`; hybrid still returns its current finite `pcov` (Task C2).
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
- Produces: golden values (recorded once from current code, committed as constants in the test file) used by Tasks A2-A4 to prove equivalence: for a fixed-seed exponential model `a*exp(-b*x)` with 20_000 points and `p0=[1.0, 0.5]`: exact_lm `x`, `cost`, `pcov`; hybrid Phase 2 `x`, `residual_sum_sq`, `pcov`; one `_accumulate_jtj_jtr` call's `(JTJ, JTr, chunk_cost)` on a 1_000-point chunk.

- [ ] **Step 1:** Write `test_exact_lm_golden`, `test_hybrid_phase2_golden`, `test_hybrid_accumulate_chunk_golden` asserting `np.testing.assert_allclose(..., rtol=1e-8, atol=1e-10)` against constants captured by running current code once (print them, paste in).
- [ ] **Step 2:** Write `test_hybrid_checkpoint_roundtrip`: run `fit` with checkpointing to a tmp dir, load via `CheckpointManager`, assert both `phase2_*_accumulator` fields are not `None` with shapes `(2,2)` and `(2,)`.
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
  `class NormalEquations: __init__(self, model: Callable[..., Any], *, weighted: bool = False, chunk: int)`; `accumulate(self, theta, source, *, check_finite=False, check_sigma=False) -> PassStats`; `chunk_stats(self, theta, x, y, mask, sigma=None) -> tuple[jax.Array, jax.Array, jax.Array]` (the jitted per-chunk `(F, g, H)`).
- Consumes: nothing new. `exact_lm.py` keeps re-exporting every moved name so `from nlsq.streaming.exact_lm import run_pass` still works.

- [ ] **Step 1:** Write `test_normal_equations.py::test_accumulate_matches_run_pass` (same stats as `run_pass` on a 3-chunk source), `test_single_parameter_shapes` (`p=1`: `H.shape == (1,1)`, `g.shape == (1,)`), `test_empty_source_raises` (`ValueError`, "non-empty"), `test_chunk_size_one` (`n=5, chunk=1` equals `chunk=5`).
- [ ] **Step 2:** Run them; expected FAIL (module missing).
- [ ] **Step 3:** Move the listed definitions into `normal_equations.py`, add `NormalEquations` as a thin wrapper holding the jitted kernel from `make_chunk_stats`; `exact_lm.py` imports from the new module. Do not edit any numerical expression.
- [ ] **Step 4:** Run `uv run pytest tests/streaming/test_normal_equations.py tests/streaming/test_exact_lm.py tests/streaming/test_exact_lm_sources.py tests/streaming/test_exact_lm_dispatch.py tests/streaming/test_normal_equations_characterization.py -v`. Expected: PASS.
- [ ] **Step 5:** Commit `refactor(streaming): extract NormalEquations from exact_lm`.

### Task A3: Route hybrid `_accumulate_jtj_jtr` through the shared kernel

**Files:**
- Modify: `nlsq/streaming/adaptive_hybrid.py` (`_accumulate_jtj_jtr` :2428; consumers :3041-3141, :3379-3446, :3545-3689, :4439-4469)
- Test: `tests/streaming/test_normal_equations_characterization.py` (extend)

**Interfaces:**
- Consumes: `NormalEquations.chunk_stats` from A2, built once in the hybrid optimizer's setup from `self.normalized_model` (unweighted).
- Produces: `_accumulate_jtj_jtr(x_chunk, y_chunk, params, JTJ_prev, JTr_prev) -> tuple[jnp.ndarray, jnp.ndarray, float]` with the **same signature and meaning as today**: third element is `sum(r^2)` for the chunk, obtained as `2.0 * F` from the kernel in this one function (add a comment: "kernel cost is 0.5*sum r^2"). It still sets `self.phase2_JTJ_accumulator` / `self.phase2_JTr_accumulator`.

- [ ] **Step 1:** Add `test_hybrid_chunk_cost_convention`: for a known chunk, `_accumulate_jtj_jtr` third return equals `float(jnp.sum((y - model)**2))` (not half).
- [ ] **Step 2:** Run the characterization suite; expected: golden tests PASS, new test PASS (it pins the existing convention before the change).
- [ ] **Step 3:** Replace the body of `_accumulate_jtj_jtr` with a call to `self._normal_eq.chunk_stats(params, x_chunk, y_chunk, mask)` where `mask` is all-True for unpadded chunks; add the accumulation `JTJ_prev + H`, `JTr_prev + g`. Keep the caller code unchanged in this task.
- [ ] **Step 4:** Run `uv run pytest tests/streaming -k "hybrid or phases or normal_equations" -n 2`. Expected: PASS within the golden tolerances. If a golden test fails by more than the tolerance because of `jacrev` vs `jacfwd`, loosen `rtol` to `1e-6` only for the Jacobian-dependent quantities and note it in the commit message.
- [ ] **Step 5:** Commit `refactor(streaming): hybrid accumulation uses shared chunk kernel`.

### Task A4: Remove duplicate accumulators; route `phases/gauss_newton.py`

**Files:**
- Modify: `nlsq/streaming/adaptive_hybrid.py` (delete `_accumulate_jtj_jtr_scan` :856 and the scan-vs-loop branches at :3039, :3544, :4448 if the Task A5 gate passes), `nlsq/streaming/phases/gauss_newton.py:630`
- Test: `tests/streaming/test_phases.py`, `tests/streaming/test_phases_extended.py` (unchanged), `tests/regression/test_phase2_memory_validation_optimizations.py`

**Interfaces:**
- Consumes: A3's adapter. `phases/gauss_newton.py::_accumulate_jtj_jtr(x_chunk, y_chunk, params, JTJ_prev, JTr_prev, valid_size=None)` keeps its signature; `valid_size` becomes the kernel `mask`.

- [ ] **Step 1:** Run the A5 benchmark first (Task A5 steps 1-2) on the scan path versus the block-loop path; this task proceeds with deletion only if loop is within 5%.
- [ ] **Step 2:** Route `phases/gauss_newton.py::_accumulate_jtj_jtr` through `NormalEquations.chunk_stats`, converting `valid_size` to a boolean mask; run `tests/streaming/test_phases*.py`. Expected: PASS.
- [ ] **Step 3:** If the gate passed, delete `_accumulate_jtj_jtr_scan` and make the three call sites use `_accumulate_jtj_jtr` only; otherwise keep scan as a private strategy inside `NormalEquations` (not a second public method). `grep -rn "_accumulate_jtj_jtr_scan" nlsq tests` must show no stale references (update tests that call it, stating the reason in the commit).
- [ ] **Step 4:** Run `uv run pytest tests/streaming tests/regression -n 2`. Expected: PASS.
- [ ] **Step 5:** Commit `refactor(streaming): drop duplicate J^T J accumulators`.

### Task A5: Benchmark gate and changelog

**Files:**
- Modify: `CHANGELOG.md`; record results in the PR description.
- Reference: the `bench` skill and `benchmarks/` scripts used for hybrid and the 100M/1B exact-streaming gates.

- [ ] **Step 1:** On `main`, run the hybrid benchmark (CPU and GPU) and record wall times.
- [ ] **Step 2:** On the branch, repeat. Expected: each within 5% of `main`.
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
- Produces: tests that pin, through `curve_fit` and `curve_fit_large(method="hybrid_streaming")` (public entry points): NaN rows with `nan_policy="omit"` give the same `popt` as the pre-filtered data; `data_mask` stays aligned after omission; `pcov` degrees of freedom use `sum(mask)`; the hybrid path accepts `nan_policy` given once.

- [ ] **Step 1:** Write `test_omit_matches_prefiltered`, `test_mask_alignment_after_omit`, `test_dof_uses_mask_sum_not_padded_length`, `test_hybrid_nan_policy_not_consumed_twice`.
- [ ] **Step 2:** Run `uv run pytest tests/core/test_prepare_problem_characterization.py -v`. Expected: PASS on current code, except record (as `xfail(strict=False)` with reason) if `test_hybrid_nan_policy_not_consumed_twice` shows the pop-vs-get difference.
- [ ] **Step 3:** Commit `test: characterize curve_fit input preparation`.

### Task B2: Introduce `prepare_problem` by moving the code

**Files:**
- Create: `nlsq/core/problem.py`
- Modify: `nlsq/core/minpack.py` (`_prepare_curve_fit_inputs` becomes a thin call; consumers at :5236, :5256, :5313-5319)
- Test: `tests/core/test_problem.py`; repoint `tests/core/test_minpack_prepare_inputs_comprehensive.py`, `tests/core/test_minpack_coverage.py:1139`, `tests/core/test_pr29_regressions.py:139`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class PreparedProblem: n: int; p0: np.ndarray; lb: np.ndarray; ub: np.ndarray; x: Any; y: Any; sigma: Any | None; mask: np.ndarray | None; m_real: int; n_effective: int; len_diff: int` and `def prepare_problem(f, x, y, p0, bounds, sigma, data_mask, check_finite, nan_policy) -> PreparedProblem`. Ordering inside it is the one documented in spec section 4. `n_effective = int(mask.sum())` when a mask exists, else `m_real`; callers use `n_effective` where they used `ysize`.

- [ ] **Step 1:** Write `test_problem.py::test_n_effective_counts_unmasked_rows`, `test_empty_input_raises_value_error`, `test_omit_realigns_mask`, `test_len_diff_is_padding_length`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Move the body of `_prepare_curve_fit_inputs` and the mask/padding code into `prepare_problem` unchanged in order; `_prepare_curve_fit_inputs` returns what it did before by unpacking `PreparedProblem` (keeps the 13-tuple for any remaining caller until Step 4).
- [ ] **Step 4:** Repoint the three listed test modules at `prepare_problem`; run `uv run pytest tests/core -n 2`. Expected: PASS. Remove `_prepare_curve_fit_inputs` and the discarded `_should_pad` return only if `grep -rn "_prepare_curve_fit_inputs" nlsq tests` is empty.
- [ ] **Step 5:** Commit `refactor(core): extract prepare_problem`.

### Task B3: Hybrid path uses `prepare_problem`

**Files:**
- Modify: `nlsq/core/minpack.py` `_curve_fit_hybrid_streaming` (:4374-4407)
- Test: B1 tests

- [ ] **Step 1:** Replace the hybrid path's own `nan_policy`/`check_finite`/p0/bounds handling with one `prepare_problem(...)` call; use `kwargs.get("nan_policy", ...)` semantics so the value is not consumed twice.
- [ ] **Step 2:** Run `uv run pytest tests/core tests/streaming -n 2` and B1 tests. Expected: PASS; flip the B1 `xfail` (if any) to a normal pass.
- [ ] **Step 3:** `make verify`, `graphify update .`, changelog "Internal: prepare_problem; hybrid streaming no longer consumes nan_policy twice"; commit; draft PR.

---

# Part C: Covariance (PR 3, branch `refactor/covariance`)

### Task C1: Golden covariance tests per site (no source change)

**Files:**
- Create: `tests/core/test_covariance_characterization.py`
- Reference: `nlsq/core/minpack.py:3419` (`_compute_covariance`, SVD of J), `nlsq/streaming/exact_lm.py:748` (`compute_covariance`, eigh of H), `nlsq/streaming/adaptive_hybrid.py:3727` (bare `pinv`), `nlsq/streaming/large_dataset.py:2455` (GLS)

**Interfaces:**
- Produces: for a well-conditioned problem and a rank-deficient one (duplicate parameter column), recorded `pcov` golden values per site, plus assertions on the warning contract: standard and exact give all-`inf` + exactly one `OptimizeWarning` when singular; **hybrid gives finite `pcov` (current behavior, pinned deliberately)**.

- [ ] **Step 1:** Write `test_standard_golden`, `test_exact_golden`, `test_hybrid_golden`, `test_*_rank_deficient_contract` (four sites incl. GLS golden on the well-conditioned problem).
- [ ] **Step 2:** Run; expected PASS on current code.
- [ ] **Step 3:** Commit `test: characterize covariance contracts`.

### Task C2: One covariance module for the J-based and H-based sites

**Files:**
- Create: `nlsq/core/covariance.py`
- Modify: `nlsq/core/minpack.py:3419`, `nlsq/streaming/exact_lm.py:748` (`compute_covariance` stays importable as an alias)
- Test: `tests/core/test_covariance.py`; `tests/core/test_shared_jit_kernels.py:46` (`covariance_svd`) must still pass

**Interfaces:**
- Produces: `def pcov_from_jacobian(J, cost: float, n_eff: int, p: int, absolute_sigma: bool) -> tuple[np.ndarray, bool]` (cutoff `eps*max(n_eff,p)*s0`, SVD) and `def pcov_from_normal(H, cost: float, n_eff: int, p: int, absolute_sigma: bool) -> tuple[np.ndarray, bool]` (cutoff `p*eps*lambda_max`, eigh). Second element is `True` when the singular fallback fired. Both return all-`inf` and emit one `OptimizeWarning` in that case, as today. Hybrid's `pinv` and the GLS site are **not** routed here.

- [ ] **Step 1:** Write `test_covariance.py::test_jacobian_and_normal_agree_when_well_conditioned` (rtol 1e-8), `test_singular_returns_inf_and_warns_once` for both functions, `test_absolute_sigma_skips_scaling`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Move the two bodies unchanged into `covariance.py`; make the two sites call them; keep `exact_lm.compute_covariance` as a re-export.
- [ ] **Step 4:** Run `uv run pytest tests/core tests/streaming -n 2` plus C1. Expected: PASS with identical goldens (including hybrid finite `pcov`).
- [ ] **Step 5:** `make verify`, `graphify update .`, changelog "Internal: shared covariance helpers; no change to cutoffs"; commit; draft PR.

---

# Part D: `build_large_strategy` (PR 4, branch `refactor/large-strategy`)

### Task D1: Routing table test (no source change)

**Files:**
- Create: `tests/streaming/test_large_strategy_routing.py`
- Reference construction sites: `nlsq/__init__.py:651, 1146, 1253`; `nlsq/streaming/large_dataset.py:1900, 3053`; `nlsq/core/minpack.py:782, 812, 1030, 1091, 1764, 1829, 4446, 4658`

**Interfaces:**
- Produces: a table of `(public call, kwargs) -> (optimizer class used, return type)` for the 13 sites, asserted by monkeypatching the two classes to record construction and delegate to the real ones. Also records each site's return shape (tuple, `OptimizeResult` or dict).

- [ ] **Step 1:** Write one parametrized test over the 13 calls asserting the recorded class and return type.
- [ ] **Step 2:** Run `uv run pytest tests/streaming/test_large_strategy_routing.py -v`. Expected: PASS on current code.
- [ ] **Step 3:** Commit `test: pin large-dataset routing table`.

### Task D2: Strategy builder and site migration

**Files:**
- Create: `nlsq/streaming/strategy.py`
- Modify: the 13 sites listed in D1
- Test: D1 table unchanged; `tests/streaming/test_strategy.py`

**Interfaces:**
- Produces: `class Strategy(Protocol): def fit(self, f, x, y, p0, **common) -> OptimizeResult` and `def build_large_strategy(kind: Literal["hybrid", "chunked"], config, **common) -> Strategy`. `kind` is the path the caller already chose; the function never decides routing. Each `Strategy` owns the kwargs translation (e.g. "has no gtol/ftol/xtol", `minpack.py` ~771) and adapts the site's return shape to `OptimizeResult`. Out of scope: global-optimization callers and `_curve_fit_exact_streaming`.

- [ ] **Step 1:** Write `test_strategy.py::test_hybrid_drops_unsupported_tolerances`, `test_chunked_returns_optimize_result`, `test_unknown_kind_raises_value_error`.
- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement the two strategies by moving each site's translation code; migrate sites one file at a time (`large_dataset.py`, then `__init__.py`, then `minpack.py`), running the D1 table after each file.
- [ ] **Step 4:** Run `uv run pytest tests/streaming tests/core -n 2` and D1. Expected: PASS, table unchanged.
- [ ] **Step 5:** `make verify`, `graphify update .`, changelog "Internal: build_large_strategy; routing unchanged"; commit; draft PR.

---

## Self-review notes

- Spec coverage: sections 3 (A1-A5, traps 1-5 -> A3 cost test, A1 checkpoint test, A1 goldens, A2 shapes plus A3 mask, A5 bench), 4 (B1-B3), 5 (D1-D2), 6 (C1-C2), 7 (global constraints), 8 deferred (not planned), 9 defaults (global constraints, A4 route-not-delete).
- Open for the implementer: golden constants come from running current code once (A1, B1, C1); tolerances start at `rtol=1e-8` and may loosen only where A3 step 4 says.
- Risk: Part A step A3 changes hybrid's Jacobian from per-point `jacrev` to chunk `jacfwd`; this is the one place goldens may need loosening.
