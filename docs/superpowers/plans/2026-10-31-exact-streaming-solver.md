# Exact Streaming Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in exact large-N least-squares solver, `curve_fit_large(..., method="exact_streaming")`, that accumulates the full-data cost, `Jᵀr` and `JᵀJ` chunk by chunk (Levenberg–Marquardt, O(p²) state), after hardening its prototype.

**Architecture:** A new module `nlsq/streaming/exact_lm.py` holds the chunk-statistics function, the pass driver, the LM loop with its termination rules, covariance, active-set bounds and the data-source layer. `curve_fit_large` dispatches to it above the `np.asarray` conversion; `fit()` rejects the method; the default chunked path is untouched. Work proceeds in four stages that each end in a working, tested deliverable.

**Tech Stack:** Python ≥3.12, JAX (float64, `jacfwd`), NumPy, SciPy (reference only), pytest (+`serial`/`slow` markers), h5py (already a core dependency), optional zarr, mpmath (conditioning study script only).

**Spec:** `docs/superpowers/specs/2026-10-31-exact-streaming-solver-design.md` (read it with this plan; the spec fixes every value, this plan fixes files, names and tests). Prototype: `benchmarks/prototypes/exact_accumulated_lm.py`.

## Global Constraints

- Float64 only: host inputs are converted explicitly per block with `np.asarray(block, dtype=np.float64)`; JAX arrays are cast on device with `jnp.asarray(..., dtype=jnp.float64)` and stay resident; a lazy source (memmap, HDF5, Zarr, factory) is never converted whole. No mixed precision, no silent subsampling, no early exit before all points contribute.
- Defaults: `ftol = xtol = gtol = 1e-8`; `max_nfev` default `100·p` (the prototype's 100 is replaced); chunk size default `1 << 22`; damping floor `ε_D = 1e-10`; λ ceiling `1e14`; per-iteration solve-attempt cap `20`; device placement fraction `50%` of free device memory.
- Model contract: `model(x, *params)` written with `jax.numpy`, pointwise and deterministic in `x`; padded tail rows are masked with `jnp.where`, never by multiplication.
- Termination reasons are exactly `"gtol" | "ftol" | "xtol" | "max_nfev" | "stalled"`; `success` is true only for the first three. `max_nfev` counts full-data passes including the initial one. Convergence wins over budget on the same pass.
- Tolerance table (all stages; SciPy comparisons run both solvers at `ftol = xtol = gtol = 1e-12`): parameters `rtol 1e-6, atol 1e-8` (flat valley, two_exp small n: compare cost, and parameters sorted by rate, at `rtol 1e-4`); cost relative `1e-9` unbounded, `1e-6` against SciPy TRF when bounded; scaled gradient `|g_i|/(√H_ii·√(2F)) ≤ 1e-6`; chunk-size independence `rtol 1e-9` on parameters and `1e-12` on `F`; covariance parity with `curve_fit` `rtol 1e-6` for `κ(J) ≤ 1e4`, `1e-3` for `κ(J) ≤ 1e6`. Every relative comparison is `|a−b| ≤ rtol·|b| + atol`.
- Legacy tests `tests/streaming/test_large_dataset_equivalence_flag.py`, `test_streaming_pr30_regressions.py`, `test_large_dataset.py` pass unchanged at every task that touches `nlsq/`.
- Project rules: any test with arrays over 100K elements, subprocesses or Qt widgets carries `@pytest.mark.serial` (and `@pytest.mark.slow` over 1 s); run serial tests with `-n 0`. `uv run ruff check . && uv run ruff format --check .` clean; new `nlsq/` module passes `uv run mypy nlsq/streaming/exact_lm.py`. `mpmath` is installed in `.venv` but is **not** a project dependency (absent from `pyproject.toml` and `uv.lock`): only the Task 14 study scripts may import it, behind a guard (`pytest.importorskip("mpmath")` in tests; a clear `SystemExit` message in the script).
- `NLSQ_FORCE_CPU=1` for correctness tests in CI; GPU and 100M numbers are manual benchmarks recorded in `benchmarks/baselines/`, always prototype/legacy/new measured back-to-back in one session with machine state noted.
- Nothing is committed unless a step says so; commit messages end with the repository's attribution line.

## Review Focus

- Non-pointwise model (e.g. `a*exp(-b*(x - x.mean()))`): must raise `ValueError` naming the contract, not return a chunk-dependent answer. (Task 4)
- `n` not a multiple of the chunk size, `n < chunk`, `n = 2`, and a one-parameter model (`p = 1`): same answer as one big chunk. (Tasks 4, 5)
- Non-float64 or list inputs (float32 arrays, Python lists, int `x`) and NaN/inf in `y` with `check_finite` on and off. (Tasks 2, 4)
- `p0` outside the box, `p0=None`, a start already on a bound, and `lb == ub`. (Tasks 6, 9)
- Failure must raise, success must not: a stalled or budget-limited fit raises `ExactSolverError`; a converged noisy fit never does. (Tasks 1, 6)

---

## Stage 0 — Harden the prototype

### Task 1: Prototype termination rules and honest budget (defects 1, 3, 4)

**Files:**
- Modify: `benchmarks/prototypes/exact_accumulated_lm.py`
- Create: `tests/streaming/test_exact_lm.py`

**Interfaces:**
- Produces: `fit(model, x, y, p0, bounds=(-inf, inf), chunk=CHUNK, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None) -> OptimizeResult` (`nlsq.result.optimize_result.OptimizeResult`) with `x, cost, pcov, nfev, success, message, termination`; `max_nfev=None` means `100·p`.
- Produces: `step_termination(dF, F, rho, s_norm, x_norm, ftol, xtol) -> str | None` (`"ftol"` if `dF < ftol·F and rho > 0.25`, else `"xtol"` if `s_norm < xtol·(xtol + x_norm)`, else `None`).
- Produces: `newton_certificate(g, H, F, ftol, bounds_active=False) -> bool` implementing the spec's Jacobi-scaled Newton-decrement test (over columns with `H_ii > 0`; true only if `δ_N ≤ ftol·F`, nothing truncated, i.e. `λ̂_min > p·ε·λ̂_max`, and no bound active; `F == 0` is true).
- Produces (tests): module-level helper `run(model, x, y, p0, **kw)` in the test file that calls the implementation under test (re-pointed in Task 3).

- [ ] **Step 1: Write failing tests in `tests/streaming/test_exact_lm.py`** (models via `jnp`; data from `benchmarks.compare.problems.PROBLEMS["exp_decay"].data(3000)`):
  - `test_budget_ends_max_nfev_with_exact_pass_count[1,2,3]`: default exp_decay `p0` (the baseline run needs 6 passes at default tolerances, so `k ≤ 3` cannot converge), `max_nfev=k` → `termination == "max_nfev"`, `nfev == k`, `success is False`.
  - `test_nonfinite_everywhere_but_p0_stalls`: model returns `nan` unless the first parameter equals its `p0` value → `termination == "stalled"`, `success is False` (not `"ftol"`).
  - `test_forced_rejections_stall`: `monkeypatch` the module's `_pass` so every trial returns `2×` the incumbent cost → `"stalled"`.
  - `test_full_rank_start_at_optimum_one_pass`: converge at 1e-12, rerun from the result → `nfev == 1`, `success`, `termination in ("gtol", "ftol")`; noiseless start at truth → `termination == "gtol"`, `nfev == 1` (F = 0 path).
  - `test_step_termination_gates`: `(dF=0.5e-8·F, rho=0.1)` → `None`; `rho=0.5` → `"ftol"`; `s_norm=1e-12·x_norm` → `"xtol"`.
  - `test_newton_certificate_rescaling`: `diag(1e20,1)` and `diag(1e30,1)` with `g=(0,1)`, `F=0.5` → `False`; `J=[[1e15,1e15],[1,-1]]`, `r=(0,1)` (build `H=JᵀJ`, `g=Jᵀr`, `F=0.5`) → `False`; `H=I`, `g=1e-9·(1,1)`, `F=1` → `True`; same with `bounds_active=True` → `False`.
  - `test_zero_target_exact_fit`: `y ≡ 0` with a model that is exactly 0 at the start (`F == 0`) → `termination == "gtol"`, `nfev == 1`, no NaN (zero-denominator convention); a near-exact fixture `y = f(x, θ*) + 1e-15·x` (assert `0 < F ≤ 1e-24·Σy²` first) uses the coordinatewise gradient test and does not divide by zero.
  - `test_convergence_beats_budget`: `max_nfev` equal to the pass count at which the converging run stops → `success` true.
- [ ] **Step 2: Run** `uv run pytest tests/streaming/test_exact_lm.py -v -n 0`; expected: FAIL (missing names / old behaviour).
- [ ] **Step 3: Implement** in the prototype: replace `success = lam > 1e14` and the relative `gtol` with the spec's Termination section (stationarity checks `‖g_free‖∞ < gtol` and `newton_certificate` at the initial point and every accepted iterate; `step_termination` after accepted steps; stall only when no stationarity criterion holds; `F == 0` → `"gtol"`). Return `OptimizeResult` with `termination`. Keep `_selftest`/`_bench` working (read `out["x"]` etc.).
- [ ] **Step 4: Run** the same command; expected: PASS. Run `uv run ruff check benchmarks tests/streaming/test_exact_lm.py`.
- [ ] **Step 5: Commit** `fix(prototype): honest termination reasons, absolute gtol, pass budget`.

### Task 2: Prototype damping, solve accounting and input hardening (defects 2, 5)

**Files:**
- Modify: `benchmarks/prototypes/exact_accumulated_lm.py`
- Modify: `tests/streaming/test_exact_lm.py`

**Interfaces:**
- Consumes: Task 1 `fit`, `_pass`, `run`.
- Consumes/produces note: the prototype's `inv(H)` covariance stays until Task 5. Produces: damping `D_ii = max(D_prev, H_ii)` floored at `1e-10·max_j D_jj` (`D = I` if all zero); each trial pass consumes budget; ≥20 consecutive failed attempts in one iteration → `"stalled"`.

- [ ] **Step 1: Write failing tests:**
  - `test_zero_jacobian_column_converges`: `model = a*exp(-b*x) + c + 0.0*d` → `success`, `d` unchanged from `p0` (covariance assertions for this case live in Task 5, which owns the pseudo-inverse).
  - `test_nan_in_H_stalls_within_attempt_cap`: `monkeypatch _pass` to return `H` containing `nan` after the first call → `"stalled"`, `nfev <= 1 + 20`.
  - `test_rank_two_h_for_p3_converges`: `(a1 + a2)*exp(-b*x)` (p = 3, columns 1 and 2 identical, so `H` has rank 2) → `success`.
  - `test_noisy_collinear_start_at_optimum_success_or_stall`: converge that model, restart from the result → `(success and termination in ("gtol","ftol","xtol")) or (termination == "stalled" and "rank" in message)`.
  - `test_float32_input_equals_float64_input`: `fit(x32, y32)` and `fit(x32.astype(f64), y32.astype(f64))` give `np.array_equal` parameters.
  - `test_nonfinite_data_raises`: NaN in `y` → `ValueError`; model returning `inf` at `p0` → `ValueError`.
- [ ] **Step 2: Run** as in Task 1; expected: FAIL.
- [ ] **Step 3: Implement** damping scale, attempt cap, budget accounting for every data-evaluating attempt, explicit float64 casts, finite checks (data checked after the first pass's fused reduction, `p0` cost/gradient/Hessian finite), without changing Task 1's behaviour.
- [ ] **Step 4: Run** all of `tests/streaming/test_exact_lm.py` plus `uv run python -m benchmarks.prototypes.exact_accumulated_lm --selftest`; expected: PASS, selftest prints `ok` for all problems.
- [ ] **Step 5: Commit** `fix(prototype): damping floor, attempt cap, float64 and finite checks`.

---

## Stage 1 — Library module (unbounded, unweighted)

### Task 3: Move the engine into `nlsq/streaming/exact_lm.py`

**Files:**
- Create: `nlsq/streaming/exact_lm.py`
- Modify: `tests/streaming/test_exact_lm.py` (`run` re-pointed to the module; the Stage 0 prototype tests are kept as `tests/streaming/test_exact_lm_prototype.py`, a copy still importing the prototype, so Stage 0 stays covered)

**Interfaces:**
- Produces: `class ExactSolverError(RuntimeError)` with `.result: OptimizeResult`.
- Produces: `exact_lm_fit(model, x, y, p0, *, sigma=None, bounds=(-inf, inf), chunk_size=None, ftol=1e-8, xtol=1e-8, gtol=1e-8, max_nfev=None, check_finite=True, absolute_sigma=False, memory_limit_gb=None, show_progress=False, n_points=None) -> OptimizeResult` with `x, cost, pcov, nfev, success, message, termination, n_points, equivalent_to_full_batch=True`. In this task `sigma` and finite `bounds` raise `ValueError` naming Stage 2.
- Produces: `derive_chunk_size(n, p, memory_limit_gb, chunk_size) -> int` (explicit `chunk_size` wins; else `1 << 22`, reduced so `chunk × p × 8 B × 4` fits the budget; precedence: an explicit positive `chunk_size` wins, clamped to `n`; otherwise `1 << 22` reduced to fit the budget with a soft floor of 1024 (when `n < 1024` the chunk is `n`); if even a 1024-point chunk's workspace exceeds `memory_limit_gb`, raise `ValueError` naming `memory_limit_gb`).
- Produces: `make_chunk_stats(model)` → jitted `(theta, x, y, mask) -> (F, g, H)`; `run_pass(stats, theta, source, chunk) -> PassStats(cost, g, H, n_seen, data_finite)`; `step_termination`, `newton_certificate` (moved from the prototype, same signatures).
- Consumes: Task 1–2 behaviour, verbatim.

- [ ] **Step 1: Re-point** `run` in `tests/streaming/test_exact_lm.py` to call `exact_lm_fit(..., chunk_size=chunk)` and import `step_termination`, `newton_certificate`, `_pass`-patch target `nlsq.streaming.exact_lm.run_pass`; adjust the two monkeypatch tests accordingly. Run; expected: FAIL (module missing).
- [ ] **Step 2: Implement** the module by moving the prototype's tested logic: stats with `jnp.where` masking, fused finite/count reduction in pass 1, device placement rule, `show_progress` logging. The prototype is **not** changed to delegate: it stays standalone, frozen at its hardened Stage 0 state, so it remains an independent baseline for the Stage 1 timing gate and for its own `--selftest` (which still runs the bounded problem).
- [ ] **Step 3: Run** `uv run pytest tests/streaming/test_exact_lm.py -v -n 0` and the three legacy test files; expected: PASS. `uv run mypy nlsq/streaming/exact_lm.py` clean.
- [ ] **Step 4: Commit** `feat(streaming): add exact_lm engine module (unbounded, unweighted)`.

### Task 4: SciPy parity, chunk invariance, model-contract probe, finite/count checks

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `tests/streaming/test_exact_lm.py`

**Interfaces:**
- Produces: `check_model_contract(model, theta, x_chunk) -> None` raising `ValueError` (message contains `"pointwise"`); called once on the first chunk (also for factories later). It evaluates the model on the chunk, its first half, and its reversed copy, requiring equality to relative `1e-12`.

- [ ] **Step 1: Write failing tests** (all `PROBLEMS` except `bounded_exp`, n = 3000, tolerances from the Global table, SciPy `least_squares` residual `y − model(np, …)` with `xtol=ftol=gtol=1e-12`; two_exp compared after sorting term pairs by rate):
  - `test_matches_scipy[exp_decay|gaussian|two_exp]`: parameters, cost, scaled gradient `≤ 1e-6` at the result.
  - `test_chunk_size_independence`: chunk sizes `{n, n//2, 512, 513, 7}` → parameters `rtol 1e-9`, `F` relative `1e-12`.
  - `test_randomized_problems_match_scipy`: seeded set (seeds 0–9, exp_decay with random `p_true`, noise 0.05).
  - `test_non_pointwise_model_raises`: model `a*exp(-b*(x - x.mean()))` → `ValueError` matching `"pointwise"`.
  - `test_check_finite_flag`: NaN in `y` → `ValueError` after pass 1 when `check_finite=True`; with `False` the call raises `ValueError` too (non-finite cost at start), documenting both paths.
  - `test_tiny_and_p1_cases`: `n=2` with p = 1 model, and `n < chunk`, match a one-chunk run.
  - `test_permutation_invariance`: shuffled `(x, y)` pairs give parameters within `rtol 1e-9` of the ordered run.
  - `test_derive_chunk_size_precedence`: `derive_chunk_size(10**8, 4, 0.01, None)` yields a chunk with workspace ≤ 0.01 GB and ≥ 1024; explicit `chunk_size=7` returns 7; `n=2` returns 2; `derive_chunk_size(10**8, 4, 1e-9, None)` raises `ValueError` matching `"memory_limit_gb"`.
  - `test_cpu_chunks_are_views`: `np.shares_memory(next(iter_blocks(x, y, None, 1000))[0], x)` is true for float64 `ndarray` input (direct sharing check, Task 11 provides `iter_blocks`; this test is added there and listed here for ownership); `test_cpu_fit_peak_memory` (`@pytest.mark.serial`, `@pytest.mark.slow`): under `tracemalloc`, fitting 20M-point `ndarray`s (320 MB of `x` and `y`) with `chunk_size=1_000_000` peaks below `8×` the chunk bytes + 64 MB (a whole-array copy would exceed it).
  - `test_list_and_int_inputs`: Python lists and int `x` equal the float64-array result.
- [ ] **Step 2: Run**; expected: FAIL only where behaviour is missing (probe, finite flag).
- [ ] **Step 3: Implement** `check_model_contract`, the `check_finite` flag, and any fix the parity tests expose.
- [ ] **Step 4: Run** the file; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm model-contract probe and SciPy parity tests`.

### Task 5: Unweighted covariance contract

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `tests/streaming/test_exact_lm.py`

**Interfaces:**
- Produces: `compute_covariance(H, cost, n, p, absolute_sigma) -> np.ndarray`: eigen-decomposition of `H`, keep `λ_i > p·ε·λ_max`, pseudo-inverse over kept directions, scale `2·cost/(n−p)` unless `absolute_sigma`, all-`inf` plus `OptimizeWarning` ("Covariance of the parameters could not be estimated") only if no direction survives or any value is non-finite or (`n ≤ p` and not `absolute_sigma`). `UserWarning` when `√cond(H) > 1e4`.

- [ ] **Step 1: Write failing tests:** `test_pcov_matches_curve_fit` (`nlsq.curve_fit` on 3000 points, `rtol 1e-6` for exp_decay and gaussian, both `absolute_sigma` values); `test_rank_deficient_pcov_finite_and_zero_on_dropped` (includes the Task 2 zero-column model: finite `pcov`, row and column 3 equal to 0); `test_pcov_parity_to_cond_1e6` (degree-7 polynomial on `[0,1]` at n = 3000; the test first measures `κ(J)` by SVD and asserts it lies in `[1e4, 1e6]`, then `rtol 1e-3` against `curve_fit`); `test_n_le_p_gives_inf_with_warning` (`pytest.warns(OptimizeWarning)`, only when `absolute_sigma=False`); `test_ill_conditioned_warns` (`pytest.warns(UserWarning)` for two_exp with rates `1.0` and `1.0001`).
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** `compute_covariance` and use it in `exact_lm_fit` (replacing the prototype's `inv`).
- [ ] **Step 4: Run** the file; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm covariance with curve_fit contract`.

### Task 6: `curve_fit_large(method="exact_streaming")`, `fit()` gate, docs

**Files:**
- Modify: `nlsq/__init__.py` (`curve_fit_large` signature at ~714, dispatch hoisted above the `np.asarray` conversion at ~876; `fit()` at ~426), `nlsq/streaming/__init__.py` (export `ExactSolverError`), `CHANGELOG.md`, `docs/` API reference entry
- Create: `tests/streaming/test_exact_lm_dispatch.py`

**Interfaces:**
- Consumes: `exact_lm_fit`, `ExactSolverError`.
- Produces: `curve_fit_large(..., method: MethodLiteral | Literal["exact_streaming"] | None = None, ...)` (the shared `MethodLiteral` in `nlsq/types.py` is **not** widened); `exact_streaming` returns `(popt, pcov)`; failure raises `ExactSolverError` carrying `.result`.

- [ ] **Step 1: Write failing tests:**
  - `test_dispatch_returns_popt_pcov_and_matches_module`.
  - `test_failure_raises_with_result`: a stalling model → `ExactSolverError` (is a `RuntimeError`), `e.result.termination == "stalled"`; a budget-limited fit (`max_nfev=2`) raises with `"max_nfev"`.
  - `test_unknown_kwarg_typeerror`: `curve_fit_large(..., method="exact_streaming", foo=1)` → `TypeError` naming `foo`.
  - `test_unsupported_arguments_valueerror`: non-default `stability`, `rescale_data`, `max_jacobian_elements_for_svd`, `auto_size_detection`, `size_threshold`; `multistart=True`; `sigma=`; finite `bounds`; 2-D `xdata` → `ValueError` naming the argument (sub-options `n_starts`, `sampler`, `center_on_p0`, `scale_factor` ignored while `multistart=False`).
  - `test_runs_regardless_of_size_threshold` (n = 100 000 vs default threshold 1 000 000 gives the exact solver).
  - `test_p0_none_infers_parameter_count` (same inference as the hybrid branch at `nlsq/__init__.py:923-932`).
  - `test_fit_rejects_exact_streaming`: `nlsq.fit(f, x, y, p0, method="exact_streaming")` → `ValueError` mentioning `curve_fit_large`, for both small and large `n`.
  - `test_default_chunked_path_unchanged` (existing equivalence-flag behaviour: `equivalent_to_full_batch` still `n_chunks == 1`).
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** the hoisted branch (validate lengths ≥ 2, equal, 1-D from `shape`/`len` without reading data), kwarg whitelist `{ftol, xtol, gtol, max_nfev}`, argument policy, `ExactSolverError` conversion, and the `fit()` guard. Add the CHANGELOG entry and an API-docs entry stating the model contract, the failure behaviour and the covariance caveat.
- [ ] **Step 4: Run** `uv run pytest tests/streaming/test_exact_lm_dispatch.py tests/streaming/test_exact_lm.py tests/streaming/test_large_dataset_equivalence_flag.py tests/streaming/test_streaming_pr30_regressions.py tests/streaming/test_large_dataset.py -n 0`; expected: PASS. Run `uv run ruff check .`.
- [ ] **Step 5: Commit** `feat(streaming): curve_fit_large method="exact_streaming"`.

### Task 7: Harness adapters and the Stage 1 performance gate (manual)

**Files:**
- Modify: `benchmarks/compare/adapters.py` (add `_NLSQLarge` named `"nlsq_large"` and `_NLSQExact` named `"nlsq_exact"` to `ADAPTERS`), `tests/benchmarks/test_compare_adapters.py` (registry set becomes seven names; `test_unbounded_libraries_decline_the_bounded_problem` tuple at ~line 52 gains `"nlsq_exact"` until Task 9), `benchmarks/prototypes/exact_accumulated_lm.py` (`--bench` gains `--impl prototype|module`)
- Create: `benchmarks/baselines/2026-10-31-exact-streaming-stage1.md`

**Interfaces:**
- Consumes: `curve_fit_large` with `method=None` / `"exact_streaming"`. Adapters follow the `_Base` pattern (`_fit(problem, x, y) -> FitOutcome`, `capabilities`, `dist = "nlsq"`).

- [ ] **Step 1: Update** the registry test to the seven names and add `"nlsq_exact"` to the unbounded-decline tuple (its `supports` returns `(False, "no bounds support")` for `bounded_exp`); confirm FAIL; run `uv run pytest tests/benchmarks/test_compare_adapters.py -n 0`.
- [ ] **Step 2: Implement** the two adapters; `nlsq_exact` declines bounded problems until Task 9.
- [ ] **Step 3: Run** the adapter tests; expected: PASS (they recover truth on exp_decay within 5%).
- [ ] **Step 4 (manual, GPU, outside the sandbox):** in the isolated CUDA venv, run prototype and module back-to-back with matched resident-data preparation (both receive the same `jnp` arrays, 3 warm repeats, exp_decay 100M): `python -m benchmarks.prototypes.exact_accumulated_lm --bench exp_decay 100000000 --impl prototype`, then `--impl module`. The harness adapters (end-to-end, include host-to-device transfer) are used only for the comparison tables. Gate: module warm median ≤ 1.10 × prototype. Record both numbers with `free`, CPU governor and GPU clocks in the baseline file; also record CPU numbers for all unbounded problems.
- [ ] **Step 5: Commit** `bench: nlsq_large and nlsq_exact adapters, Stage 1 baseline`.

---

## Stage 2 — Weights, bounds

### Task 8: 1-D `sigma` and weighted covariance

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `nlsq/__init__.py` (stop rejecting `sigma`), `tests/streaming/test_exact_lm.py`, `tests/streaming/test_exact_lm_dispatch.py`

**Interfaces:**
- Produces: `exact_lm_fit(..., sigma=1-D array)`: per-chunk `w = 1/σ` applied to residuals and Jacobian rows; `σ` validated (length equals `n`, positive, finite — finite check fused into pass 1) → `ValueError`; 2-D `sigma` → `ValueError`.

- [ ] **Step 1: Write failing tests:** `test_weighted_matches_scipy` (heteroscedastic `σ = 0.02 + 0.1·x`, residual `(y−f)/σ`); `test_weighted_chunk_invariance`; `test_weighted_pcov_matches_curve_fit` (both `absolute_sigma`); `test_bad_sigma_raises` (wrong length, zero, negative, NaN, 2-D); dispatch test updated so `sigma` is accepted.
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** weights in `make_chunk_stats` (signature gains `w`), sigma validation, weighted covariance (`Σ r²/(n−p)` on weighted residuals).
- [ ] **Step 4: Run** both test files plus the three legacy files; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm 1-D weights`.

### Task 9: Box bounds by active set, with KKT tests

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `nlsq/__init__.py` (stop rejecting finite `bounds`), `benchmarks/compare/adapters.py` (`nlsq_exact` accepts bounded problems), `tests/streaming/test_exact_lm.py`, `tests/streaming/test_exact_lm_dispatch.py`, `tests/benchmarks/test_compare_adapters.py`
- Create: `tests/streaming/exact_lm_kkt.py` (helper)

**Interfaces:**
- Produces: `assert_kkt(result, model, x, y, lb, ub, tau=1e-6)` in `tests/streaming/exact_lm_kkt.py`: recomputes `g, H, F` in float64 with NumPy; scaled `ĝ_i = g_i/(√H_ii·√(2F))` (0 if `H_ii = 0` or `F = 0`; coordinatewise fallback for `0 < F ≤ 1e-24·Σy²`); free → `|ĝ_i| ≤ τ`, at lower → `ĝ_i ≤ τ`, at upper → `ĝ_i ≥ −τ`; `lb ≤ x ≤ ub` exactly.
- Produces: bounds handling in `exact_lm_fit`: `lb < ub` required (`ValueError`), `p0` clipped, free set = `~((θ ≤ lb) & (g ≤ 0)) & ~((θ ≥ ub) & (g ≥ 0))`, i.e. a variable is held when it sits at a bound with the gradient pointing outward, reduced system on free variables, trial `clip(θ+δ)`, `pred` on the projected step, stationarity on the free gradient, no Newton certificate while any bound is active.

- [ ] **Step 1: Write failing tests:** `test_bounded_exp_matches_scipy_trf` (cost relative `1e-6`, parameters `rtol 1e-4`) with `assert_kkt`; `test_randomized_bounded_kkt` (seeded; configurations with 0, 1 and all parameters ending on a bound); `test_free_set_both_signs` (unit test of the free-set function with a variable at the lower bound and `g<0`, at the lower bound and `g>0`, at the upper bound and `g>0`, at the upper bound and `g<0`), `test_start_on_bound`, `test_start_outside_box_is_clipped`, `test_bound_cutting_flat_valley`; `test_bounded_start_at_constrained_optimum_succeeds` (spec: must terminate `success=True`); `test_lb_ge_ub_raises` (including `lb == ub`); adapter test: `nlsq_exact` solves `bounded_exp`.
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** the active-set logic. Response order if the randomized set stalls or misses SciPy's cost: (1) TRF-style scaling on the reduced system, (2) if still failing, keep bounds raising `ValueError` and record the follow-up in the baseline file (the spec's 2a/2b rule). If `test_bounded_start_at_constrained_optimum_succeeds` fails, certify with the constrained model decrease (small projected-Newton solve of the p-dimensional quadratic).
- [ ] **Step 4: Run** the three exact_lm test files and the three legacy files; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm active-set bounds with KKT tests`.
- **Deferred branch (only if Step 3's fallback order is exhausted):** keep finite `bounds` raising `ValueError` naming Stage 2b (`test_bounds_deferred_raises` replaces the bounded tests), leave `nlsq_exact` declining bounded problems (the Task 7 tuple entry stays), document "bounds not supported by `exact_streaming`" in the API entry and CHANGELOG, record the follow-up in the Stage 2 baseline file, commit as `docs: defer exact_streaming bounds`, and skip Task 10 Step 1.

### Task 10: Stage 2 benchmark record and changelog (manual)

**Files:**
- Create: `benchmarks/baselines/2026-10-31-exact-streaming-stage2.md`
- Modify: `CHANGELOG.md`, API docs entry

- [ ] **Step 1 (manual):** back-to-back at 100M on GPU and CPU: `bounded_exp` for prototype, module and `curve_fit_large`. Stop condition: CPU `bounded_exp` no slower than `curve_fit_large` by more than the prototype's 7%; otherwise write why, and pass only if the GPU result (prototype: 8× faster) holds. Skip this step in the deferred branch.
- [ ] **Step 2:** write the baseline file (commands, machine state, numbers) and the CHANGELOG/docs update for weights, bounds and the "active set, not Coleman–Li" note.
- [ ] **Step 3: Commit** `docs: Stage 2 baseline and changelog`.

---

## Stage 3 — Out-of-memory sources and the conditioning study

### Task 11: Array-like sources (memmap, HDF5, Zarr) and read instrumentation

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `nlsq/__init__.py` (hoisted branch accepts any object with `shape`/`len`/slicing)
- Create: `tests/streaming/test_exact_lm_sources.py`

**Interfaces:**
- Produces: `iter_blocks(source_x, source_y, source_sigma, chunk) -> Iterator[tuple[np.ndarray, ...]]` reading by slice, `np.asarray(block, dtype=np.float64)` for host sources (a float64 `ndarray` yields views) and `jnp.asarray(..., dtype=jnp.float64)` kept on device for JAX arrays, never copying the whole array and never shuffling.

- [ ] **Step 1: Write failing tests:** `test_invalid_final_block_raises` (NaN only in the last block → `ValueError` after pass 1, before any step); `test_cpu_chunks_are_views` (`np.shares_memory` of the first block with a float64 `ndarray` input); `test_jax_array_stays_resident` (`isinstance(block, jax.Array)` for a `jnp` input, run under `jax.transfer_guard_device_to_host("disallow")` so an intermediate host round trip fails the test); `test_memmap_h5py_match_in_memory` (500K points; `h5py.File` in `tmp_path`; `zarr` variant guarded by `pytest.importorskip("zarr")`); `test_reads_are_chunk_sized` using a counting wrapper (class with `shape`, `__len__`, `__getitem__` recording `(start, stop)`): no single read larger than `chunk_size`, bytes read per pass equal to the data size within one chunk; `test_tracemalloc_peak_bounded`: 50M-point memmap, 1M chunk, peak `< 8×` chunk bytes + 64 MB; the large tests carry `@pytest.mark.serial` and `@pytest.mark.slow`.
- [ ] **Step 2: Run** with `-n 0`; expected: FAIL.
- [ ] **Step 3: Implement** `iter_blocks` and route array-likes through it; the model-contract probe runs on the first block only, while the fused finite check covers every block of pass 1.
- [ ] **Step 4: Run** the sources tests and the exact_lm/dispatch/legacy files; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm out-of-memory array sources`.

### Task 12: Chunk factory and `n_points`

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `nlsq/__init__.py` (callable `xdata`, `n_points` added to the whitelist), `tests/streaming/test_exact_lm_sources.py`, `tests/streaming/test_exact_lm_dispatch.py`

**Interfaces:**
- Produces: `curve_fit_large(f, factory, None, p0=..., method="exact_streaming", n_points=N | None, ...)`; `factory(chunk_size) -> Iterator[(x, y) | (x, y, sigma)]`, replayable; per-block validation (1-D, equal lengths, at most `chunk_size` points, finite unless `check_finite=False`); every pass after the first checks the point count.

- [ ] **Step 1: Write failing tests:** `test_factory_equals_array_source`; `test_factory_with_and_without_n_points`; `test_replay_count_mismatch_raises` (factory that drops the last block on its second call → `ValueError` naming the count); `test_block_validation` (oversized block, mismatched lengths, NaN); `test_callable_xdata_with_ydata_raises` (`ydata` not `None`); `test_n_points_rejected_without_factory`.
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** factory sources and validation; `n_points` only accepted with a factory.
- [ ] **Step 4: Run** all exact_lm test files and legacy files; expected: PASS.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm replayable chunk factories`.

### Task 13: Prefetch thread, kept only if it pays (manual gate)

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`
- Create: `benchmarks/studies/exact_prefetch.py`

**Interfaces:**
- Produces: private `_prefetch(iterator, depth=1)` generator that reads the next block on a background thread; used by the pass driver only if the gate passes.

- [ ] **Step 1:** write `benchmarks/studies/exact_prefetch.py`: per-pass wall time with and without prefetch on a 1B-style configuration (HDF5 file source, 100M-point slice if 1B is not yet generated), 5 passes each, machine state recorded.
- [ ] **Step 2:** implement `_prefetch` and a failing-then-passing unit test in `tests/streaming/test_exact_lm_sources.py::test_prefetch_preserves_order_and_propagates_errors`.
- [ ] **Step 3 (manual):** run the study. Keep prefetch enabled only if per-pass time is at least 10% lower; otherwise delete `_prefetch`, the test and the study, and note the result in the Stage 3 baseline.
- [ ] **Step 4: Commit** `feat(streaming): exact_lm prefetch` (or `chore:` removal commit).

### Task 14: Conditioning-study helpers and reference

**Files:**
- Create: `benchmarks/studies/__init__.py`, `benchmarks/studies/conditioning_helpers.py`, `tests/benchmarks/test_conditioning_helpers.py`

**Interfaces:**
- Produces: `mp_reference(model_mp, x, y, p0, digits=50) -> tuple[np.ndarray, float, np.ndarray, int]` (parameters rounded to float64, cost, covariance over retained directions by the `λ > p·ε·λ_max` rule evaluated in extended precision, rank); `tsqr_solve(chunks, theta, model, lam, d) -> tuple[np.ndarray, float]` (`d` is the running-maximum damping-scale vector of the LM loop) returning the damped Gauss–Newton step from augmented `[J | r]` blocks (`F = ½(‖c‖²+ρ²)`, `g = Rᵀc`, rank-revealing solve, damping as augmented rows) and the cost; `error_norms(theta, theta_ref, cov, cov_ref, rank, rank_ref) -> dict` (parameter error `max_i |θ_i−θ_ref,i|/max(|θ_ref,i|,1e-12)`; covariance Frobenius error `‖Σ−Σ_ref‖_F / max(‖Σ_ref‖_F, 1e-300)`, defined as the absolute `‖Σ‖_F` when `Σ_ref` is exactly zero; rank equality from the supplied ranks); `verdict(rows) -> Literal["normal_equations_stay", "tsqr_default"]` implementing the pre-registered rule over accuracy-set rows with `κ(J) ≤ 1e4`.

- [ ] **Step 1: Write failing tests** (`pytest.importorskip("mpmath")` at module top): `test_mp_reference_matches_scipy_on_well_conditioned_problem` (exp_decay, n = 200, `rtol 1e-10` parameters); `test_tsqr_step_equals_normal_equation_step_when_well_conditioned` (`rtol 1e-8`); `test_tsqr_step_uses_supplied_damping` (two different `d` vectors, e.g. `d = H_ii` and a history-inflated `d`, give steps equal to the corresponding damped normal-equation solutions); `test_error_norms_zero_reference_covariance` (noiseless case: no division by zero, returns the absolute norm); `test_tsqr_cost_includes_column_space_residual` (a problem whose residual has a column-space part: `F` equals `½‖r‖²` to `1e-12`); `test_error_norms_values_on_synthetic_inputs`; `test_verdict_rule` (all rows pass → `"normal_equations_stay"`; one row with covariance error `2e-3` at `κ ≤ 1e4` → `"tsqr_default"`; rows above `κ = 1e4` ignored).
- [ ] **Step 2: Run** `uv run pytest tests/benchmarks/test_conditioning_helpers.py -n 0`; expected: FAIL.
- [ ] **Step 3: Implement** the four functions (mpmath only inside `mp_reference`).
- [ ] **Step 4: Run**; expected: PASS.
- [ ] **Step 5: Commit** `bench: conditioning-study helpers`.

### Task 14b: Run the conditioning study and record the decision

**Files:**
- Create: `benchmarks/studies/conditioning_study.py`, `benchmarks/baselines/2026-10-31-exact-streaming-conditioning.md`

**Interfaces:**
- Produces: CLI `python -m benchmarks.studies.conditioning_study --out <dir>` writing JSON rows (problem, seed, `κ(J)`, errors for both candidates and the SciPy baseline) and printing `verdict(rows)`.

- [ ] **Step 1: Write the script** over the spec's problems (two_exp with `δ ∈ {1e-1,…,1e-5}`, Vandermonde degree 8 and 12, three close Gaussians), seeds 0–4, **accuracy set** n = 2000 on identical observations for the mpmath reference and both candidates, **scale set** n = 2×10⁴ against SciPy only; `SystemExit("mpmath is required for the study")` if the import fails.
- [ ] **Step 2: Run** the study; record the rows and the verdict in the baseline file with the exact command and machine state.
- [ ] **Step 3:** if the verdict is `"normal_equations_stay"`, stop (the `κ(J) > 1e4` warning from Task 5 stays); otherwise execute Task 15.
- [ ] **Step 4: Commit** `bench: conditioning study and decision`.

### Task 15 (conditional on the Task 14b verdict being "tsqr_default"): TSQR accumulation in the module

**Files:**
- Modify: `nlsq/streaming/exact_lm.py`, `tests/streaming/test_exact_lm.py`

**Interfaces:**
- Produces: `exact_lm_fit(..., accumulate: Literal["tsqr", "normal"] = "tsqr")`; TSQR as default per the rule, `"normal"` an explicit fast opt-in that warns above `κ(J) = 1e4` (documented cost if TSQR is >1.5× slower on the 100M problems).

- [ ] **Step 1: Write failing tests:** the Stage 1/2 parity and KKT suites parametrized over `accumulate`; `test_tsqr_survives_ill_conditioned_case` using the case the study found failing (parameter and covariance errors within the table).
- [ ] **Step 2: Run**; expected: FAIL.
- [ ] **Step 3: Implement** the TSQR path (per-chunk QR of `[J | r]` → `(p+1)×(p+1)` blocks, pairwise combine, solve from `R`); covariance from `R`.
- [ ] **Step 4: Run** all exact_lm tests; re-run the 100M timing for the 1.5× clause and record it.
- [ ] **Step 5: Commit** `feat(streaming): exact_lm TSQR accumulation`.

### Task 16: 1B-point run, gates and final docs (manual)

**Files:**
- Create: `benchmarks/studies/exact_1b_run.py`, `benchmarks/baselines/2026-10-31-exact-streaming-1b.md`
- Modify: `CHANGELOG.md`, API docs entry (sources, factory, `n_points`, conditioning result)

**Interfaces:**
- Produces: script writing exp_decay at 1,000,000,000 points (16 GB x+y float64) to an HDF5 file with per-chunk seeds, in a noiseless and a noisy copy, and a streaming NumPy cost evaluator `streaming_cost(source, model_np, theta, chunk) -> float` (new, not `residual_cost`).

- [ ] **Step 1: Write** the script and a unit test `tests/benchmarks/test_exact_1b_helpers.py::test_streaming_cost_matches_residual_cost_small` (10⁵ points, equality to 1e-12 relative).
- [ ] **Step 2 (manual, 16 GB GPU):** run the gates: (1) noiseless recovery `rtol 1e-6` and cost `< 1e-6·n`; (2) noisy parameters within `5·√diag(pcov)`; (3) every observation contributes: point count equals `n` on every pass, `streaming_cost` equals the solver's `cost` to relative `1e-9`, and the sentinel block (last 1,000 points of the last chunk offset by `Δ` calibrated so the linearized displacement `H⁻¹J_sᵀΔ` is `1e-3` relative) displaces the fit by at least half the prediction; (4) peak RSS `< data size + 4 GB`; (5) report wall time, passes, per-pass throughput.
- [ ] **Step 3:** write the baseline file with commands, machine state and results; update CHANGELOG and docs.
- [ ] **Step 4: Run** the full suite once: `uv run pytest tests -m "not serial" -n 2 -q`, then `uv run pytest tests -m serial -n 0 -q` and `uv run ruff check .`; expected: all pass.
- [ ] **Step 5: Commit** `docs: Stage 3 results and exact_streaming documentation`.

---

## Self-Review (run by the plan author)

- **Spec coverage:** Stage 0 defects 1–5 → Tasks 1–2; Stage 1 module/dispatch/args/failure/covariance/harness/100M gate → Tasks 3–7; Stage 2 weights/bounds/KKT/2a-2b rule/benchmarks → Tasks 8–10; Stage 3 sources, factory, instrumentation, prefetch gate, conditioning study with decision rule, 1B gates → Tasks 11–16 (study helpers Task 14, study run Task 14b); resolved deferrals are implemented only where they bind code (`fit()` guard, Task 6); the rest are decisions with no task. Model-contract probe → Task 4; Newton-decrement criterion → Task 1.
- **Type consistency:** `exact_lm_fit`, `ExactSolverError`, `step_termination`, `newton_certificate`, `run_pass`, `make_chunk_stats`, `compute_covariance`, `iter_blocks`, `assert_kkt`, `streaming_cost` keep one name and signature across tasks; `make_chunk_stats` gains `w` in Task 8 and `run_pass` follows.
- **Review Focus:** each of the five lines has a named test in the task noted.
- **Known judgement calls left to the executor:** internal structure of `exact_lm.py`, logging format, chunk-size derivation from the memory budget (spec's "left to the builder").
