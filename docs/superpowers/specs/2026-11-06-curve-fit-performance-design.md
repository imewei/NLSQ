# Performance design: `curve_fit` (< 100k points) and `curve_fit_large` (>= 100k points)

Date: 2026-11-06. Status: draft for approval. Base: `main` @ 87d8912a.
Sources: Claude profiling (this session), Codex source review (read-only, via Herdr), benchmark baseline
`benchmarks/baselines/2026-11-06-library-comparison-curve_fit-vs-large.md`.

## Intent

Make the two default entry points faster on the sizes where users call them, without changing what they compute.

- `curve_fit` for n < 100k: today 4-8 ms on CPU and 23-51 ms on GPU at n = 1k (SciPy 1-3 ms; JAXopt on GPU 3-7 ms).
  Bounded fits cost 47 ms CPU / 170 ms GPU at 1k.
- `curve_fit_large` (default strategy, not `method="exact_streaming"`) for n >= 100k: 1.5-3.2 s on GPU at 10M,
  13.9-23 s at 100M, 41-49 s on CPU at 100M.

Why now: the 2026-11-06 rerun showed fixed per-call and per-chunk overhead, not arithmetic, dominates both ranges.

## Goals

1. Lower warm end-to-end time in both ranges (targets below).
2. Keep results inside the existing benchmark cost gate (final cost within rtol 1e-6 of the best library) and keep all
   existing tests passing (`make verify`).
3. No new public API, no new required arguments, no change to defaults.
4. No increase in peak host or device memory (B-side changes should reduce it).

Targets (my proposal, derived from the measurements below; the owner may change them). Harness numbers, warm median:

| Case | Today | Target | Phase 0 note (2026-11-06) |
|---|---|---|---|
| `curve_fit` 1k unbounded, GPU | 23-51 ms | <= 12 ms | **Implausible without A5.** Re-measured 23.7-25.5 ms. Measured removable time: A2 at most 5/6 x 0.59 ms x 4 acceptance calls, about 2 ms; A1 at most 0.92 ms (gate failed); A3 about 0.6 ms x 5 SVDs, about 3 ms (gate failed). All three together reach about 19-20 ms |
| `curve_fit` 1k unbounded, CPU | 4-8 ms | <= 3.5 ms | **Implausible from A1+A2.** Re-measured 4.7-6.4 ms. Removable: A1 <= 0.52 ms (gate failed), A2 about 0.13 ms; together at most 11-14%, a 3.5 ms target needs about 26% or more |
| `curve_fit` 1k bounded (`bounded_exp`), GPU / CPU | 170 / 47 ms | <= 125 ms / <= 38 ms (aim) | Re-measured 148 / 43 ms. `select_step` is 16% / 13% of the wall and its conversions 7-8% / 1.5%, so A4 as scoped cannot give more than about 15%; larger gains need the per-iteration eager dispatch and syncs of `trf_bounds` |
| `curve_fit` 100k bounded (`bounded_exp`), GPU / CPU | not stated | <= 200 ms / <= 200 ms (aim, only if Task 8 is retargeted) | Measured 387 / 324 ms; `select_step` shows 50% / 60% of the wall, mostly the first-sync wait on model and Jacobian compute and the host `np.dot` in `build_quadratic_1d` |
| `curve_fit_large` 10M exp_decay, GPU | 1.5 s | <= 1.0 s | Re-measured 1.28 s. Tight: B1 saves about 0.07 s, `pcov_from_jacobian` is 0.18 s of it |
| `curve_fit_large` 100M exp_decay, GPU | 14-20 s | <= 12 s | Re-measured 12.5-13.0 s this session, so the target is only about 7% below today and inside session drift; judge by interleaved A/B only. B1 saves about 0.45 s, `gc.collect` 0.62 s (4.8%), `pcov_from_jacobian` 1.76 s |

## Non-goals

- Matching JAXopt/Optimistix (no bounds, sigma or covariance; different estimator).
- `exact_streaming`, float32 / mixed precision, changing chunk sizes, replacing the GLS combination, removing the
  shuffle or stopping before all chunks.
- Persistent compilation-cache tuning (warm timings only).

## Anti-goals (a failure even if the benchmark improves)

- Any result drift beyond the 1e-6 cost gate, or changed covariance semantics (`absolute_sigma`, rank-deficient
  `inf` covariance).
- A speedup visible only in a profile but not in the harness's end-to-end number.
- Stale-model bugs: two functions with identical code but different closure state must never share traced wrappers.
- Silent behavior change on the callback / diagnostics / custom-Jacobian paths.

## Evidence

Profiles use `cProfile` on the isolated GPU venv (RTX 4090 Laptop, JAX 0.12 CUDA13). Profiling inflates absolute
times; shares are the useful number. Script and raw output live in the session scratchpad, not the repo.

**`curve_fit`, n = 1k, exp_decay, warm, 3-parameter model**

| Cost | CPU (3.5 ms/call) | GPU (18.5 ms/call) |
|---|---|---|
| `_trace_model` (model re-traced every call to build the wrapper-cache key, `least_squares.py:700`) | ~37% (2.6 ms profiled; real wall time per call is lower) | 12% (2.5 ms profiled) |
| TRF loop (`trf_no_bounds`) | 43% | 72% |
| of which `_evaluate_step_acceptance` (`trf.py:901`; includes model and Jacobian evaluation, not only host syncs; 80 calls / 20 fits) | 25% | 35% |
| of which trust-region SVD (`_svd_no_bounds`, `trf_jit.py:303`) | small | 25% (1.25 ms per call) |
| covariance (`pcov_from_jacobian`, SVD again) | small | 7% (1.5 ms per call) |

**SVD microbenchmark (median ms; n x 3 float64):**

| n | GPU `svd` | GPU `qr`+`svd(R)` | GPU gram+`eigh` | CPU `svd` |
|---|---|---|---|---|
| 1,000 | 1.51 | 1.87 | 0.72 | 0.026 |
| 100,000 | 2.12 | 2.96 | 0.73 | not measured |
| 1,000,000 | 14.97 | 14.84 | 0.96 | not measured |

GPU SVD of a tiny tall-skinny matrix is latency-bound (about 58x the CPU cost at 1k). QR does not help. Singular values
from `svd` and `qr`+`svd(R)` agree to 1e-15.

**`curve_fit_large`, n = 10M, exp_decay, GPU, warm (1.16 s wall, 0.955 s profiled)**

| Cost | Share |
|---|---|
| `_fit_chunked` self time (host shuffle / gather loop, `large_dataset.py:2679`) | 40% (0.385 s) |
| per-chunk `pcov_from_jacobian` (10 calls x 15 ms; matches the 1M x 3 GPU SVD above) | 16% (0.155 s) |
| device syncs (`array._value`, 238 calls) | 9% |
| `create_chunks` | 4% |
| `_chunk_ssr` (model re-evaluated per chunk for SSR) | 5% (0.044 s) |
| `gc.collect` (one call in the profiled run) | 4% |
| `_require_finite` | 3% |

The 100M case was not profiled; it runs the same per-chunk loop with ~10x more chunks, so the shares are assumed to
carry over (verify in phase 0).

Routing fact (re-checked, `nlsq/streaming/specs.py:243-247`): default `curve_fit_large` fits in memory below 1M and uses
chunked GLS from 1M. Improvements to the TRF loop (workstream A) therefore also speed up every chunk fit in workstream B.

## Synthesis of the two reviews

| Item | Claude (measured) | Codex (source review) | Decision |
|---|---|---|---|
| Per-iteration host syncs / Python-driven TRF loop | 35% of GPU 1k time in acceptance | Compile the whole unbounded iteration; 2-5x guess | Adopt in two stages: fuse syncs first (A2), compiled loop only if A1-A3 miss targets (A5) |
| Model re-trace per call | 12% GPU, ~37% CPU at 1k | Not found | Adopt (A1); Codex missed it |
| GPU SVD latency | 25% GPU 1k (+7% covariance) | Not found | Adopt (A3), measured |
| Bounded step selection | not profiled | 2-4x guess, `trf.py:2405` (Codex cited 2393) | Measure first (A4) |
| Gather permutation into chunk buffers | `_fit_chunked` self time is 40% at 10M | several-percent guess | Adopt as top B item (B1); measured gain potential is larger than Codex guessed |
| Reuse final residuals for SSR | 5% at 10M | 5-20% guess | Adopt, low priority (B3); Codex's range is too high at 10M |
| Periodic `gc.collect` | 4% | unknown | Measure only (B4), likely drop |
| Per-chunk covariance cost | 16% at 10M | Not found | Adopt behind decision D1 (B2) |
| Keep GLS, shuffle, chunk size, SVD-based exactness | agree | agree | Non-goals |

## Workstreams

### A. `curve_fit`, n < 100k (ordered by evidence over risk)

- **A1. Make the per-call model key cheaper.** `_trace_model` runs `make_jaxpr` (and the Jacobian trace when custom
  rules are present) on every call to compute the cache key. Constraint: the key must stay exact. The traced program
  captures values the model reads from module globals, captured objects and bound-method state; the existing comment
  records that identity-based `make_jaxpr` caching returned stale jaxprs after state changes, and v0.7.0 fixed a
  closure-collision bug in JIT cache keys. So a memo keyed on function code plus closure cells is not acceptable on its
  own (it would miss globals and `self.k`). Allowed: cheaper construction of the same exact key (for example avoiding
  the Jacobian trace, cheaper `_snapshot_jaxpr`, avoiding repeated `eval_shape`), or a memo restricted to functions
  proven free of free variables beyond verified constants, with the current path as fallback. If neither yields a
  measurable gain, drop A1 and say so. Phase 0 re-measures the real (unprofiled) share before work starts. An independent CPU 1k measurement found `_trace_model` at 0.96 ms
  of a 4.57 ms call, of which `make_jaxpr` (0.43 ms) must stay and the removable parts (`eval_shape` 0.14, config key 0.06,
  snapshot 0.02 ms; the Jacobian trace is already skipped without custom rules) are about 5%: A1 is expected to be dropped
  at its gate.
- **A2. Fuse per-attempt scalar syncs.** `_evaluate_step_acceptance` already syncs once per inner attempt (earlier
  perf work), but still reads several device scalars one by one (finiteness, termination code, `actual_reduction`,
  `Delta_new > 0`, `math.isfinite(alpha...)`). Pack finiteness, actual reduction, termination code, updated radius,
  updated alpha, `step_h_norm` and `step_norm` into one jitted call returning one vector, read once per attempt, with the
  same host-side ordering as today (non-finite retry path, termination before radius/alpha update). The 35% profile share
  above includes the model and Jacobian evaluations done inside that function, so the real gain is unknown: Phase 0
  measures the scalar-conversion share and A2 is dropped below 15% of the acceptance call on GPU 1k. The loss function
  (user callables are rebuilt as new closures per `curve_fit` call) stays outside the jitted helper; only scalar math is fused. Behavior-preserving;
  callbacks and diagnostics keep working.
- **A3. Small-matrix linear algebra on the CPU device when the default backend is GPU.** For the trust-region SVD
  (`trf_jit.py:303`, five-output contract `J_h, U, s, V, uf`) and covariance SVD (`minpack.py:3323`, plain `(s, VT)`) when
  n x p is below a threshold, run the same SVD on the CPU device. Placement: `J_h = J*d` stays on the original device;
  `J_h` and `f` go to the host, `U.T @ f` is computed there, and `s`, `V`, `uf` go back to the original device (committed
  arrays on two devices cannot meet in a jitted call). The covariance SVD becomes a Python dispatch around its jitted
  kernel. Same math (LAPACK instead of cuSOLVER); no conditioning change. The threshold is set by measurement including
  transfer cost: Phase 0 gives `HOST_SVD_MAX_ELEMENTS` = 30,000 (n x p; host incl. transfer faster at every measured point
  up to there except one tie, mixed to 54,000, slower from 60,000), but the 5x gate at 1k failed (2.1-2.6x), so A3 is dropped unless revived. No effect when the backend is
  already CPU. Singular vectors of degenerate subspaces are not unique, so verification compares reconstruction,
  projectors and covariance, never signs.
- **A4. Bounded path.** Profile `bounded_exp` at 1k and 100k first (47 ms CPU / 170 ms GPU are outliers vs 4-8 / 23-51
  ms unbounded). `select_step` (`trf.py:2405`) already uses the JAX helpers; the remaining host reads are the `float(...)`
  conversions on the out-of-bounds path. Deliverable of this item is a profile and a targeted fix whose scope the profile
  sets (if `select_step` is under 20% of the bounded wall time the item targets what dominates, or is dropped). The step
  choice uses strict comparisons with a fall-through to the anti-gradient candidate (`trf.py:2557-2561`); any device-side
  version keeps those predicates, it does not substitute `argmin`. Target is set after the profile.
- **A5. (gated) Compiled TRF loop.** Internal `lax.while_loop` over the unbounded iteration for configurations without
  callbacks or diagnostics, others keep the Python loop. Only started if A1-A3 leave the 1k GPU time above target,
  because rejected-step handling, tolerances, evaluation budgets, robust losses and failure fallback all have to be
  reproduced.

### B. `curve_fit_large` default path, n >= 100k

- **B1. Feed the existing chunk buffer pool from the permutation** (`large_dataset.py:2744-2748`, `DataChunker.create_chunks`
  `:815` already copies each chunk into pooled buffers): keep the seed-0 permutation, exact chunk boundaries and sigma
  alignment; pass the permutation to `create_chunks` as an `indices` argument (instead of its `np.arange`) so the three
  full shuffled copies (x, y, sigma) and the 800 MB `arange` at 100M disappear. Sigma of each chunk is gathered with the
  same permutation slice, which replaces the boundary guard by construction. Target: the host-side share (the gather
  share is measured in Phase 0), plus lower peak memory. B1 is a memory change by default: it moves the random-access
  gathers into the chunk loop rather than removing them (independent CPU measurement at 10M: 0.33 s today versus 0.29 s,
  about 4% of wall), and the seed-0 `rng.permutation` itself (0.33 s at 10M) must stay bit-identical. A timing claim needs
  a Phase 0 measurement of both variants. This lowers the expected benefit of workstream B: the 40% `_fit_chunked` self
  time above is mostly the permutation and the random gathers, not removable copies.
- **B2. Cheaper per-chunk covariance (gated by D1).** 15 ms per chunk spent in a GPU SVD of an n_chunk x p Jacobian.
  Options: Gram + `eigh` (0.96 ms at 1M x 3) with a guard and SVD fallback; or run the SVD of the p x p factor. The
  guard requires finite, strictly positive eigenvalues and `sqrt(w_max/w_min) <= cond_max` (<= 1e6 as an upper bound; the Gram error grows like eps*cond^2 and an independent
  measurement gave scale-aware pcov errors of 2.6e-8, 1.9e-6 and 2.3e-4 at cond 1e4, 1e5 and 1e6, so the sweep is expected
  to settle near 1e4; set by a sweep over
  conditioning, column scales and rotations; scale-aware error `|dC_ij|/sqrt(C_ii C_jj)` <= 1e-6), and the result must be
  sorted descending because `pcov_from_jacobian` truncates relative to `s[0]`. Phase 0 records the per-chunk `cond(J)` of the
  benchmark problems on the unchanged path; once the threshold is calibrated, B2 proceeds only if at least 90% of chunks
  pass the guard, otherwise it is dropped before any production change. The per-chunk covariance is
  not only reported: `_accum_information` (about `large_dataset.py:2063`) inverts it and uses it as the weight when chunk
  estimates are combined, so this changes the fitted parameters (by a tiny fraction of their standard error) and the
  reported chunked `pcov` (by ~1e-6 relative), not just a diagnostic. It is computed inside the ordinary per-chunk
  `curve_fit` call, so the change travels through the existing `covariance_svd` hook installed only for the chunk loop;
  `curve_fit`'s own covariance for every other caller stays as is. Needs D1 (given).
- **B3. Reuse chunk final cost for SSR** (`_chunk_ssr` at `large_dataset.py:2355`, call in the chunk loop): drop the second
  model evaluation and vector transfer per chunk by using `2 * result.cost` of the chunk fit. Eligible only when the loss
  is linear, `nan_policy` is not `"omit"`, no other row-dropping option is set, and the result still has all `valid_length`
  rows; otherwise the old computation runs.
- **B4. `gc.collect` policy** (`large_dataset.py:2919`): measure at 100M; change only if it exceeds 1% of wall time.
- **B5.** A1-A3 apply to each chunk fit and need no separate work.

Routing (verified, `nlsq/streaming/specs.py:243-247`): default `curve_fit_large` runs the in-memory strategy (the
`curve_fit` path) below 1,000,000 points and `_fit_chunked` from 1,000,000 up. So workstream A also covers
100k-1M through both entry points, and workstream B only applies from 1M.

## Constraints

- JAX-first, float64 throughout; minimize host-device transfers (project rule). A3 adds a bounded transfer by design and
  must show a net win in the harness, not only in a profile.
- Python >= 3.12, ruff clean, mypy clean, `make verify` green. Subprocess or 100k+ element tests need
  `@pytest.mark.serial`.
- Timing noise on this laptop is up to ~50% between sessions: each claim needs an interleaved before/after run in the
  same session, repeats as in the baseline; within a session a gain counts only outside the measured spread of the
  interleaved arms (the plan's `ab_nlsq` rule), across sessions only gaps above 1.5x are signal unless a profile attributes
  them to the removed cost.
- No runtime device-to-host counter exists in the repo (`tests/core/test_host_device_transfers.py` checks source text and
  `nlsq/utils/profiling.py` reports placeholder zeros): the A2 test brings its own counting instrument and must show a
  non-zero baseline before asserting a reduction.
- Test files to keep green (Codex's list, all verified to exist on 2026-11-06): `test_trf_step_rejection.py`, `test_trf_refactoring_validation.py`,
  `test_callbacks.py`, `test_trf_select_step.py`, `test_trf_additional_branches.py`, `test_curvefit_golden_master.py`,
  `test_large_dataset.py`, `test_review_2026_11_04c_chunked_fixes.py`, `test_covariance_characterization.py`.
- Environment: CPU-only project `.venv`; GPU runs need the isolated venv (`uv venv` +
  `uv pip install -e . "jax[cuda13]==0.12.0.1" jaxopt==0.8.5 optimistix==0.1.0 lmfit==1.3.4 scipy==1.18.1`) and the
  sandbox bypass for the GPU. `benchmarks/compare/run_library_comparison.py` is the measurement harness.

## Decisions

Made by me (change on request):

- Order: A1 -> A2 -> A3 -> B1 -> B3 -> A4 -> B2 -> A5. Cheap, safe, measured items first.
- Targets are aims, not pass/fail gates; if A1-A4 and B1-B3 land correct but miss them, I report the numbers and
  ask before starting A5 (this is D2/D3 below).
- Environment: GPU results come from the isolated JAX 0.12 / CUDA 13 venv; CPU results must also be re-measured in the
  project `.venv` (which currently holds JAX 0.12.0.1 with the cuda13 plugin; behavior on older JAX that `pyproject.toml`
  still allows is not tested by this work) with `JAX_PLATFORMS=cpu`, and must show no slowdown there. A3 only uses `jax.devices("cpu")` and
  `jax.device_put`, which exist in the supported range.
- Phase 0 is measurement only: profile 100k and 100M, CPU `svd` at 100k and 1M, `bounded_exp`, the scalar-conversion share
  of one step-acceptance call, `cond(J)` of the benchmark problems and the real chunk counts; the targets table is
  finalized from it, and each later item has a go/no-go gate recorded in the plan (Task 2, Step 3).
- A3 uses the same SVD algorithm on a different device, not a different factorization.
- Work happens on a feature branch off `main`, one change per commit so each is revertible; no commits until asked.

Answered by the owner (2026-11-06):

- **D1: yes.** Per-chunk covariance (B2) may use a Gram + `eigh` path guarded by a condition estimate (cond(J) below
  about 1e6 as an upper bound; measured error suggests about 1e4), with SVD fallback, as an internal-only option on the chunk path. Never for the trust-region step, and
  `curve_fit`'s covariance for other callers is unchanged.
- **D2/D3: yes.** Targets table accepted. Targets are aims; if A1-A4 and B1-B3 land correct but miss them, report the
  numbers and ask before starting A5.
