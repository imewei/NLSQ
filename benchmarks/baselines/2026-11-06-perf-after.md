# Performance results after the curve_fit small/large work, 2026-11-06

Branch `perf/curve-fit-small-and-large` (HEAD `24f73ba8`) against `main` at `87d8912a`. Same machine as `2026-11-06-library-comparison-curve_fit-vs-large.md`
(i9-13900H, RTX 4090 Laptop 16 GB, jax[cuda13] 0.12.0.1, CPU governor `powersave`). Raw harness JSON (local, gitignored):
`benchmarks/results/2026-11-06-perf-after/`. Phase 0 measurements and the per-task A/B addenda are in `2026-11-06-perf-phase0.md`.

## What landed, what was dropped

Landed (code on the branch):

- **A2** (commits `824a21ec`, `409d44f4`): the per-attempt scalar host syncs of TRF step acceptance are fused into one transfer (unbounded path, `fuse_scalars`). Kept by owner/controller ruling (Ruling 10), not by passing its gate: the plan's strict A/B rule (ratio <= 0.9 and outside the in-session spread) held in 6 of 9 higher-power GPU 1k case-runs (per-case medians 0.870 / 0.806 / 0.700); no CPU speedup is claimed.
- **B1** (`3c935ca2`, `4fcb809f`): `_fit_chunked` gathers each chunk from the permutation instead of copying shuffled arrays (memory change; chunk-span tripwire and upfront index validation).
- **B3 / B4** (`172a2ea1`, `24f73ba8`): the chunk SSR is reused from `2 * res.cost`; `gc.collect()` runs only on RSS growth (failure-safe RSS read).

Dropped by the plan's gates (no code on the branch):

| Item | What it was | Gate | Measured | Owner could revive |
|---|---|---|---|---|
| A1 (Task 3) | cut the 1k CPU model-trace / copy overhead | removable share of CPU 1k wall >= 10% | 9.4% median over 5 sessions (8.2-10.9%); removable <= 0.52 ms | gate is borderline; at most ~9% of CPU 1k |
| A3 (Task 5) | host SVD for tiny Jacobians on GPU | host SVD incl. transfers >= 5x faster than GPU SVD at 1k | 2.1-2.6x (SVD alone 7.5-12.3x, transfers eat the gain); expected net saving ~13% of GPU 1k, under the plan's 0.85 bound | `HOST_SVD_MAX_ELEMENTS = 30,000` was the recorded threshold; largest remaining GPU 1k lever, adds cross-device correctness risk |
| A4 (Task 8) | bounded `select_step` conversions | `select_step` share of bounded wall >= 20% (spec: below 20% the item is dropped) | `select_step` was 13-16% of the 1k wall and its conversions 1.5-8%; the bounded cost is spread over per-iteration eager dispatch and syncs in `trf_bounds` | needs a different scope (compiled/fused bounded iteration), not a `select_step` fix |
| B2 (Task 9) | Gram-based per-chunk covariance | 10M GPU ratio <= 0.9 and < 1 - max(spread) on all three problems | cond gate passed (max chunk cond 6.5 / 6.5 / 789 vs cap 1e4); ratios 0.947 / 0.884 / 0.835 vs required < 0.880 / 0.843 / 0.795 (inside the spread) | see design below |

B2 design summary (copied from the SDD ledger `task-9-report.md`, which is deleted with the workspace): (1) `nlsq/core/covariance.py` gets
`GRAM_COND_MAX = 1e4`, a jitted Gram product and `gram_covariance_svd(jac, cond_max)`; it forms `J^T J`, takes `eigh`, and returns `(s, VT)`
only if float64, 2-D, finite, all eigenvalues positive and `sqrt(w_max/w_min) <= cond_max`, else falls back to the jitted `_covariance_svd`
(the cap was chosen from a 486-case sweep: worst scale-aware pcov error 1.7e-8 at 1e4 and 1.3e-6 at 1e5). (2) `nlsq/streaming/large_dataset.py`
gets an install/restore helper (about 12 lines) that sets this as the per-chunk covariance hook before the outer `try` in `_fit_chunked` and
restores it in `finally` (needed to keep ruff C901 under the limit and to cover the tripwire, failure and `BaseException` paths).
(3) About 45 lines of production code in total, 13 tests (6 covariance, 7 streaming hook); the work was reverted, nothing is saved.
Because B2 was dropped, covariance is unchanged and chunked `pcov` is unchanged.

## Method

Interleaved A/B (`benchmarks/studies/ab_nlsq.py`): per round one fresh harness run of `main` (git worktree at `87d8912a`, removed afterwards) and one of the
branch, alternating, same harness (`benchmarks.compare.run_library_comparison`, fresh process per case, warm median, end-to-end). The runner checks
that `nlsq` imports from each checkout. Correctness gate (explicit, per round): both arms `ok`, same case set, final `cost` equal within rtol 1e-6.
All 32 cells passed the gate; the largest relative cost difference was 7.9e-12 (GPU two_exp 10M). Problems: exp_decay, gaussian, two_exp, bounded_exp.
Library `nlsq` at 1k and 1M (`curve_fit`), `nlsq_large` at 10M and 100M (`curve_fit_large`, default strategy), as in the 2026-11-06 baseline.
Repeats per harness run: 7 warm at 1k/1M, 3 at 10M, at 100M 3 (GPU) and 2 (CPU). Rounds: 5 at 1k/1M, 3 at 10M, 3 at 100M GPU, 2 at 100M CPU (plus a 3-round rerun of two cells, below).
`ratio` = median(branch) / median(main) over rounds; spread = (max - min) / median over rounds, per arm. CPU runs use `NLSQ_FORCE_CPU=1` and `JAX_PLATFORMS=cpu`.
The session drifts by up to ~50% (governor `powersave`); only the interleaved ratios are comparable, not absolute times across sessions.

## Results, same layout as the 2026-11-06 baseline (NLSQ column only, `main -> branch`)

Cells are `CPU | GPU`. SciPy, lmfit, JAXopt and Optimistix were not re-run (unchanged by this work).

| Case | NLSQ CPU (main -> branch) | NLSQ GPU (main -> branch) | ratio CPU | ratio GPU |
|---|---|---|---|---|
| exp_decay 1,000 | 4.2 ms -> 3.8 ms | 20.4 ms -> 17.2 ms | 0.907 | 0.845 |
| gaussian 1,000 | 5.1 ms -> 5.0 ms | 28.5 ms -> 24.3 ms | 0.970 | 0.853 |
| two_exp 1,000 | 7.7 ms -> 6.0 ms | 47.3 ms -> 39.4 ms | 0.779 | 0.832 |
| bounded_exp 1,000 | 42.7 ms -> 41.9 ms | 138.1 ms -> 134.1 ms | 0.982 | 0.971 |
| exp_decay 1,000,000 | 457.3 ms -> 434.5 ms | 52.5 ms -> 49.0 ms | 0.950 | 0.934 |
| gaussian 1,000,000 | 1.06 s -> 1.01 s | 70.9 ms -> 62.4 ms | 0.955 | 0.880 |
| two_exp 1,000,000 | 1.26 s -> 1.24 s | 64.8 ms -> 62.4 ms | 0.977 | 0.963 |
| bounded_exp 1,000,000 | 1.27 s -> 1.29 s | 312.6 ms -> 303.5 ms | 1.015 | 0.971 |
| exp_decay 10,000,000 | 2.85 s -> 2.43 s | 883.0 ms -> 851.8 ms | 0.851 | 0.965 |
| gaussian 10,000,000 | 3.45 s -> 3.02 s | 924.5 ms -> 803.2 ms | 0.876 | 0.869 |
| two_exp 10,000,000 | 7.01 s -> 6.43 s | 1.04 s -> 865.8 ms | 0.918 | 0.830 |
| bounded_exp 10,000,000 | 5.18 s -> 4.81 s | 1.77 s -> 1.56 s | 0.929 | 0.884 |
| exp_decay 100,000,000 | 30.68 s -> 28.73 s | 9.39 s -> 8.82 s | 0.936 | 0.940 |
| gaussian 100,000,000 | 33.15 s -> 29.40 s | 9.56 s -> 8.95 s | 0.887 | 0.937 |
| two_exp 100,000,000 | 55.31 s -> 82.04 s | 10.54 s -> 8.92 s | 1.483 | 0.846 |
| bounded_exp 100,000,000 | 57.51 s -> 64.96 s | 16.80 s -> 15.89 s | 1.129 | 0.945 |

Ratio below 1 means the branch is faster. The 2026-11-06 baseline doc gives main's numbers from an earlier session (for example exp_decay 1,000: 4.2 ms CPU, 22.9 ms GPU); this session's main column differs from it by session drift.

## Every cell: before / after / ratio / spread

| Backend | n | Case | main | branch | ratio | spread (main / branch) | rounds x repeats | max rel cost diff |
|---|---|---|---|---|---|---|---|---|
| CPU | 1,000 | exp_decay | 4.2 ms | 3.8 ms | 0.907 | 0.281 / 0.319 | 5 x 7 | 0.0e+00 |
| CPU | 1,000 | gaussian | 5.1 ms | 5.0 ms | 0.970 | 0.171 / 0.288 | 5 x 7 | 0.0e+00 |
| CPU | 1,000 | two_exp | 7.7 ms | 6.0 ms | 0.779 | 1.472 / 0.125 | 5 x 7 | 0.0e+00 |
| CPU | 1,000 | bounded_exp | 42.7 ms | 41.9 ms | 0.982 | 0.642 / 0.281 | 5 x 7 | 0.0e+00 |
| CPU | 1,000,000 | exp_decay | 457.3 ms | 434.5 ms | 0.950 | 0.084 / 0.181 | 5 x 7 | 0.0e+00 |
| CPU | 1,000,000 | gaussian | 1.06 s | 1.01 s | 0.955 | 0.048 / 0.102 | 5 x 7 | 0.0e+00 |
| CPU | 1,000,000 | two_exp | 1.26 s | 1.24 s | 0.977 | 0.039 / 0.062 | 5 x 7 | 0.0e+00 |
| CPU | 1,000,000 | bounded_exp | 1.27 s | 1.29 s | 1.015 | 0.070 / 0.063 | 5 x 7 | 0.0e+00 |
| CPU | 10,000,000 | exp_decay | 2.85 s | 2.43 s | 0.851 | 0.008 / 0.044 | 3 x 3 | 0.0e+00 |
| CPU | 10,000,000 | gaussian | 3.45 s | 3.02 s | 0.876 | 0.049 / 0.024 | 3 x 3 | 0.0e+00 |
| CPU | 10,000,000 | two_exp | 7.01 s | 6.43 s | 0.918 | 0.033 / 0.058 | 3 x 3 | 0.0e+00 |
| CPU | 10,000,000 | bounded_exp | 5.18 s | 4.81 s | 0.929 | 0.030 / 0.020 | 3 x 3 | 0.0e+00 |
| CPU | 100,000,000 | exp_decay | 30.68 s | 28.73 s | 0.936 | 0.278 / 0.418 | 2 x 2 | 0.0e+00 |
| CPU | 100,000,000 | gaussian | 33.15 s | 29.40 s | 0.887 | 0.358 / 0.419 | 2 x 2 | 0.0e+00 |
| CPU | 100,000,000 | two_exp | 55.31 s | 82.04 s | 1.483 | 0.040 / 0.140 | 2 x 2 | 0.0e+00 |
| CPU | 100,000,000 | bounded_exp | 57.51 s | 64.96 s | 1.129 | 0.407 / 0.041 | 2 x 2 | 0.0e+00 |
| GPU | 1,000 | exp_decay | 20.4 ms | 17.2 ms | 0.845 | 0.169 / 0.168 | 5 x 7 | 0.0e+00 |
| GPU | 1,000 | gaussian | 28.5 ms | 24.3 ms | 0.853 | 0.062 / 0.105 | 5 x 7 | 0.0e+00 |
| GPU | 1,000 | two_exp | 47.3 ms | 39.4 ms | 0.832 | 0.103 / 0.186 | 5 x 7 | 0.0e+00 |
| GPU | 1,000 | bounded_exp | 138.1 ms | 134.1 ms | 0.971 | 0.146 / 0.136 | 5 x 7 | 0.0e+00 |
| GPU | 1,000,000 | exp_decay | 52.5 ms | 49.0 ms | 0.934 | 0.241 / 0.157 | 5 x 7 | 0.0e+00 |
| GPU | 1,000,000 | gaussian | 70.9 ms | 62.4 ms | 0.880 | 0.233 / 0.148 | 5 x 7 | 0.0e+00 |
| GPU | 1,000,000 | two_exp | 64.8 ms | 62.4 ms | 0.963 | 0.079 / 0.084 | 5 x 7 | 0.0e+00 |
| GPU | 1,000,000 | bounded_exp | 312.6 ms | 303.5 ms | 0.971 | 0.080 / 0.073 | 5 x 7 | 0.0e+00 |
| GPU | 10,000,000 | exp_decay | 883.0 ms | 851.8 ms | 0.965 | 0.037 / 0.052 | 3 x 3 | 2.9e-16 |
| GPU | 10,000,000 | gaussian | 924.5 ms | 803.2 ms | 0.869 | 0.041 / 0.157 | 3 x 3 | 0.0e+00 |
| GPU | 10,000,000 | two_exp | 1.04 s | 865.8 ms | 0.830 | 0.042 / 0.045 | 3 x 3 | 7.9e-12 |
| GPU | 10,000,000 | bounded_exp | 1.77 s | 1.56 s | 0.884 | 0.053 / 0.050 | 3 x 3 | 0.0e+00 |
| GPU | 100,000,000 | exp_decay | 9.39 s | 8.82 s | 0.940 | 0.014 / 0.058 | 3 x 3 | 0.0e+00 |
| GPU | 100,000,000 | gaussian | 9.56 s | 8.95 s | 0.937 | 0.011 / 0.124 | 3 x 3 | 0.0e+00 |
| GPU | 100,000,000 | two_exp | 10.54 s | 8.92 s | 0.846 | 0.013 / 0.040 | 3 x 3 | 2.0e-14 |
| GPU | 100,000,000 | bounded_exp | 16.80 s | 15.89 s | 0.945 | 0.006 / 0.010 | 3 x 3 | 0.0e+00 |

Reading the ratios: a ratio only counts as a speedup when it is below `1 - max(spread)` (the plan's rule). That holds clearly only in some cells with small spreads
(GPU 10M/100M two_exp and others, CPU 10M). At 1k (CPU spread 12-147%, GPU 6-19%) and at 1M the ratios are point estimates inside noise; the repeated higher-power GPU 1k
A/B from Task 4 (`2026-11-06-perf-phase0.md`, addendum 2) is the stronger evidence for A2 (medians 0.87 / 0.81 / 0.70; the strict rule held in 6 of 9 case-runs, and A2 is kept by owner/controller ruling, Ruling 10, not by a gate pass).

### CPU 100M: two cells need a note

At 100M on CPU the first run (2 rounds x 2 repeats) gave two_exp 55.31 s -> 82.04 s (ratio 1.483) and bounded_exp 57.51 s -> 64.96 s (ratio 1.129). Per round
the branch's warm medians were 87.8 s and 76.3 s against main's 56.4 s and 54.2 s, but the branch's cold fit in round 0 took 53.0 s, which is faster than
its warm median, so this is a transient slowdown of the machine, not a stable regression. A 3-round rerun of the same two cells (`rerun_raw/`):

| Case (CPU, 100M) | main per round | branch per round | median main | median branch | ratio | spread (main / branch) |
|---|---|---|---|---|---|---|
| two_exp | 89.61, 53.82, 51.81 s | 83.72, 49.91, 46.62 s | 53.82 s | 49.91 s | 0.927 | 0.702 / 0.743 |
| bounded_exp | 68.78, 47.70, 45.64 s | 61.53, 45.98, 40.33 s | 47.70 s | 45.98 s | 0.964 | 0.485 / 0.461 |

Both runs started in a slow phase (round 0 of main took 89.6 s); the spreads are 46-74%. The two runs disagree in sign (1.48 / 1.13 first, 0.93 / 0.96 second), so
there is no evidence of a CPU 100M slowdown or speedup for these two problems; treat them as unresolved noise. exp_decay and gaussian at 100M CPU (ratios 0.936 and 0.887)
also carry spreads of 28-42%.

## Peak RSS at 100M (GPU, `nlsq_large` exp_decay, 1 repeat)

`/usr/bin/time -v` around one harness run per arm, three runs each, interleaved (maximum resident set size):

| Arm | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| main `87d8912a` | 6.081 GB | 6.085 GB | 6.083 GB |
| branch `24f73ba8` | 4.524 GB | 4.524 GB | 4.496 GB |

Peak RSS falls by about 1.57 GB (-26%). (The Task 6 report quotes 6.07 -> 3.72 GB with its own in-process probe; the whole-harness `time -v` figure here includes the
harness process and JAX runtime, so the absolute branch value is higher.)

## Performance targets (spec table; aims, not gates)

Judged on this session's interleaved numbers (branch column), warm median. "Pass" means at or below the target in this session.

| Case | Today (spec) | Target | main (this session) | branch (this session) | Result |
|---|---|---|---|---|---|
| `curve_fit` 1k unbounded, GPU | 23-51 ms | <= 12 ms | exp_decay 20.4 ms, gaussian 28.5 ms, two_exp 47.3 ms | 17.2 ms, 24.3 ms, 39.4 ms | MISS (all three; Phase 0 had called this implausible without A5) |
| `curve_fit` 1k unbounded, CPU | 4-8 ms | <= 3.5 ms | 4.2 ms, 5.1 ms, 7.7 ms | 3.8 ms, 5.0 ms, 6.0 ms | MISS (all three; exp_decay is closest at 3.8 ms) |
| `curve_fit` 1k bounded (`bounded_exp`), GPU / CPU | 170 / 47 ms | not set (A4 dropped) | 138.1 ms / 42.7 ms | 134.1 ms / 41.9 ms | not set (A4 dropped) |
| `curve_fit` 100k bounded (`bounded_exp`), GPU / CPU | not stated | not set (A4 dropped) | not measured | not measured | not set (A4 dropped) |
| `curve_fit_large` 10M exp_decay, GPU | 1.5 s | <= 1.0 s | 883.0 ms | 851.8 ms | pass (main already below target this session, ratio 0.965) |
| `curve_fit_large` 100M exp_decay, GPU | 14-20 s | <= 12 s | 9.39 s | 8.82 s | pass (main already below target this session, ratio 0.940) |

The two large-fit targets pass, but main passes them in this session too (it was 1.5 s and 19.6 s in the 2026-11-06 baseline session), so the pass reflects session
drift more than this branch; the branch's own contribution there is the 0.83-0.97 GPU ratios and the lower peak RSS. Task 4's higher-power 1k GPU A/B puts the A2 effect at
about 10-30% on GPU 1k (this session: 0.845 / 0.853 / 0.832), which is not enough to reach 12 ms.

## Targets missed - owner decision on A5 required

- `curve_fit` 1k unbounded, GPU, target <= 12 ms: branch 17.2 ms (exp_decay), 24.3 ms (gaussian), 39.4 ms (two_exp); main 20.4 ms / 28.5 ms / 47.3 ms.
- `curve_fit` 1k unbounded, CPU, target <= 3.5 ms: branch 3.8 ms (exp_decay), 5.0 ms (gaussian), 6.0 ms (two_exp); main 4.2 ms / 5.1 ms / 7.7 ms. CPU 1k spreads are 12-147%.

A5 (compiled TRF loop) was not started, per D2; it waits for the owner's decision.

## A3: small exact SVD on the host device under a GPU backend (kept by ruling)

A3 was dropped by its Task 2 gate (host SVD incl. transfers 2.1-2.6x vs the 5x gate), then revived at the owner's request and kept by controller ruling
(Ruling 13): gaussian 1k pooled 0.872 exceeds the plan's 0.85 bound and the per-session `1 - max(spread)` rule is met in only a few cells, but all
three 1k problems are below 1 in 87-98% of 45 paired rounds (same evidence class as A2). `HOST_SVD_MAX_ELEMENTS = 30,000` (phase0 item (c));
100k (n x p = 300,000) never takes the host path, so its ratios are noise around 1. Added temporary memory: two n x p host arrays (`J_h`, `U`;
<= 240 KB each); `s`, `V`, `uf` are returned to the original device, `U` stays on the host. The 0.76-0.87 range quoted in the CHANGELOG is the pooled medians; individual session medians span 0.689-0.945. GPU, `ab_nlsq`, interleaved, ratio = candidate / base `571d4597`; sessions 2 and 5 swapped the arm order (ratios inverted here).
s1: 5 rounds x 7 repeats; s2, s3: 8 x 9; s4-s6 (1k only): 8 x 31.

| case | s1 | s2 | s3 | s4 | s5 | s6 | pooled paired-round median | rounds < 1 |
|---|---|---|---|---|---|---|---|---|
| exp_decay 1k | 0.770 | 0.713 | 0.783 | 0.770 | 0.752 | 0.764 | 0.765 (n=45) | 93% |
| gaussian 1k | 0.945 | 0.831 | 0.846 | 0.911 | 0.917 | 0.857 | 0.872 (n=45) | 87% |
| two_exp 1k | 0.689 | 0.711 | 0.767 | 0.762 | 0.772 | 0.790 | 0.758 (n=45) | 98% |
| exp_decay 100k | 1.527 | 0.956 | 0.949 | - | - | - | 0.986 (n=21) | 62% |
| gaussian 100k | 0.938 | 0.918 | 1.008 | - | - | - | 0.969 (n=21) | 67% |
| two_exp 100k | 0.894 | 0.943 | 0.980 | - | - | - | 0.980 (n=21) | 57% |

Session spreads (range/median over rounds) are 5-200%; only in-session ratios are meaningful (powersave governor). The 1k gain is about 4-8 ms of a
17-45 ms fit and does not close the 12 ms GPU target.

## Caveats

- CPU 100M cells carry spreads of up to 74% with 2 rounds; see the note above. GPU 100M spreads are 1-12%.
- GPU 1k and 1M cells have 6-24% spreads over 5 rounds; ratios there are point estimates.
- Only NLSQ was re-run; no cross-library comparison was repeated.
- Test status of this tree is in the task report: 4,822 + 538 passed; the only failures were environmental (lmfit/jaxopt not installed in `.venv`, read-only `~/.nlsq`) and also occur on `main` or pass with a writable HOME.
