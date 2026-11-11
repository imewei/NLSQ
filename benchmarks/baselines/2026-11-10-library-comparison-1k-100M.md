# Library comparison, 1k / 10k / 100k / 1M / 10M / 100M points (2026-11-10)

Commit `ec9dd429` (branch `perf/close-jaxopt-gap`; default exact `curve_fit_large`, adaptive TSQR covariance, `compute_covariance`). Raw data (local, gitignored): `benchmarks/results/2026-11-10/`. Same machine and harness as `2026-11-09-library-comparison-1k-100M.md` (i9-13900H, RTX 4090 Laptop 16 GB), project `.venv`: jax/jaxlib 0.12.0.1 (CUDA), scipy 1.18.1, lmfit 1.3.4, jaxopt 0.8.5, optimistix 0.1.0. CPU runs use `JAX_PLATFORMS=cpu`. CPU governor `powersave` (unchanged). Harness `benchmarks/compare/run_library_comparison.py`, fresh process per case, `--no-batch`, warm median, end-to-end. Cells are `CPU | GPU`; SciPy and lmfit are CPU only.

Machine state: `uptime` load average was 8.05 / 6.98 / 7.54 before the A/B runs and 4.9 / 7.3 / 8.9 right after the comparison, with nothing of mine running, so another process was loading the host throughout (not visible from the sandbox). Laptop drift between sessions is up to ~50% or more (see Caveats).

API split as on 2026-11-09: `curve_fit` at 1k / 10k / 100k / 1M, `curve_fit_large` at 1M / 10M / 100M. `curve_fit_large` is now the **exact streaming** path (default above `size_threshold`); on 2026-11-09 it was the chunked GLS default. `exact_streaming` was not run separately (it is the same code).

Invocations (one `--out` each, flags as on 2026-11-09):

```bash
R() { .venv/bin/python -m benchmarks.compare.run_library_comparison --no-batch --out benchmarks/results/2026-11-10/$1 "${@:2}"; }
R gpu-small    --sizes 1000 10000 100000 1000000 --libraries nlsq jaxopt optimistix --repeats 7
R gpu-1M-large --sizes 1000000   --libraries nlsq_large --repeats 7
R gpu-10M      --sizes 10000000  --libraries nlsq_large jaxopt optimistix --repeats 3
R gpu-100M     --sizes 100000000 --libraries nlsq_large jaxopt optimistix --repeats 3
export JAX_PLATFORMS=cpu
R cpu-small    --sizes 1000 10000 100000 1000000 --libraries nlsq scipy lmfit jaxopt optimistix --repeats 7
R cpu-1M-large --sizes 1000000   --libraries nlsq_large --repeats 7
R cpu-10M      --sizes 10000000  --libraries nlsq_large scipy lmfit jaxopt optimistix --repeats 3
R cpu-100M-jax --sizes 100000000 --libraries nlsq_large jaxopt optimistix --repeats 2
R cpu-100M-scipy-lmfit --sizes 100000000 --libraries scipy lmfit --repeats 1
```

Repeats: 7 warm at 1k-1M, 3 at 10M, 100M: 3 (GPU), 2 (JAX-based CPU), 1 (SciPy, lmfit). No per-library cap was needed at 100M.

| Case | SciPy | lmfit | NLSQ `curve_fit` | NLSQ `curve_fit_large` | JAXopt | Optimistix |
|---|---|---|---|---|---|---|
| exp_decay 1,000 | 1.9 ms | 2.5 ms | 8.4 ms \| 14.7 ms | — | 0.5 ms \| 2.0 ms | 0.8 ms \| 2.6 ms |
| gaussian 1,000 | 7.0 ms | 8.4 ms | 12.6 ms \| 27.5 ms | — | 1.9 ms \| 6.0 ms | 1.6 ms \| 5.0 ms |
| two_exp 1,000 | 8.5 ms | 6.8 ms | 11.3 ms \| 58.7 ms | — | 1.2 ms \| 9.5 ms | 3.6 ms \| 25.6 ms |
| bounded_exp 1,000 | 8.9 ms | 15.4 ms | 230.6 ms \| 492.4 ms | — | n/a | n/a |
| exp_decay 10,000 | 11.7 ms | 5.3 ms | 22.0 ms \| 18.8 ms | — | 2.3 ms \| 4.1 ms | 2.8 ms \| 4.0 ms |
| gaussian 10,000 | 20.0 ms | 10.5 ms | 25.8 ms \| 29.7 ms | — | 7.0 ms \| 7.0 ms | 10.5 ms \| 7.0 ms |
| two_exp 10,000 | 24.3 ms | 9.0 ms | 15.0 ms \| 48.4 ms | — | 9.8 ms \| 9.7 ms | 8.7 ms \| 11.9 ms |
| bounded_exp 10,000 | 38.0 ms | 47.9 ms | 135.1 ms \| 422.7 ms | — | n/a | n/a |
| exp_decay 100,000 | 420.9 ms | 121.6 ms | 123.6 ms \| 21.3 ms | — | 50.1 ms \| 3.2 ms | 59.4 ms \| 10.3 ms |
| gaussian 100,000 | 376.2 ms | 107.2 ms | 246.4 ms \| 29.9 ms | — | 93.9 ms \| 11.1 ms | 162.2 ms \| 28.9 ms |
| two_exp 100,000 | 467.8 ms | 85.1 ms | 125.9 ms \| 63.9 ms | — | 48.7 ms \| 14.1 ms | 110.5 ms \| 45.6 ms |
| bounded_exp 100,000 | 1.31 s | 1.08 s | 622.4 ms \| 859.2 ms | — | n/a | n/a |
| exp_decay 1,000,000 | 1.78 s | 941.9 ms | 1.37 s \| 54.6 ms | 384.7 ms \| 490.9 ms (†123 ms) | 289.3 ms \| 13.1 ms | 698.2 ms \| 61.7 ms |
| gaussian 1,000,000 | 4.34 s | 1.48 s | 3.35 s \| 66.2 ms | 833.4 ms \| 723.4 ms (†183 ms) | 491.9 ms \| 40.2 ms | 1.17 s \| 184.0 ms |
| two_exp 1,000,000 | 5.12 s | 2.26 s | 2.60 s \| 146.6 ms | 1.50 s \| 405.6 ms (†172 ms) | 1.75 s \| 50.1 ms | 4.38 s \| 410.6 ms |
| bounded_exp 1,000,000 | 5.64 s | 6.73 s | 3.30 s \| 818.5 ms | 576.5 ms \| 416.7 ms (†157 ms) | n/a | n/a |
| exp_decay 10,000,000 | 14.88 s | 13.36 s | — | 1.66 s \| 237.3 ms (†207 ms) | 2.96 s \| 96.8 ms | 6.58 s \| 1.50 s |
| gaussian 10,000,000 | 29.26 s | 11.26 s | — | 10.15 s \| 975.6 ms (†355 ms) | 5.52 s \| 342.5 ms | 11.67 s \| 2.13 s |
| two_exp 10,000,000 | 29.73 s | 32.71 s | — | 9.98 s \| 1.78 s (†1.02 s) | 6.76 s \| 377.7 ms | 21.47 s \| 4.38 s |
| bounded_exp 10,000,000 | 47.80 s | 49.95 s | — | 5.74 s \| 749.6 ms (†336 ms) | n/a | n/a |
| exp_decay 100,000,000 | 49.36 s | 39.03 s | — | 12.87 s \| 1.11 s | 16.71 s \| 1.09 s | 41.74 s \| FAIL |
| gaussian 100,000,000 | 87.55 s | 54.18 s | — | 31.29 s \| 2.34 s | 45.56 s \| 3.29 s | 107.80 s \| FAIL |
| two_exp 100,000,000 | 126.44 s | 117.42 s | — | 39.59 s \| 3.30 s | 45.84 s \| 4.20 s | 82.26 s \| FAIL |
| bounded_exp 100,000,000 | 181.65 s | 176.50 s | — | 28.84 s \| 1.73 s | n/a | n/a |

`†` = GPU recheck, see Caveats: the `curve_fit_large` GPU cells at 1M and 10M of the first run were ~1.1-4x slower than an immediate rerun of the same cells and than the same-hour A/B, so the rerun values are shown beside them. JAXopt GPU in the rerun (1M: 11 / 37 / 43 ms; 10M: 82 / 311 / 354 ms) matches the first run, so the slowdown was in the NLSQ host loop, not the GPU. The headline cells keep the original single-run numbers.

`FAIL` = Optimistix GPU OOM / cuSolver `Dormqr` error at 100M (same three problems as 2026-11-09). `n/a` = bounds unsupported. No case timed out. No `DNRO`.

## Gate

**Every NLSQ cell is within cost `rtol 1e-6` of the best: PASS.** Cost gaps were recomputed against the best cost of each (problem, n) across all invocations and both devices. The largest NLSQ gap over all `nlsq` and `nlsq_large` cells is **6.9e-9** (two_exp 100M); `curve_fit_large` two_exp is 3.2e-9 at 1M, 6.0e-9 at 10M, 6.9e-9 at 100M, bounded_exp 2.0e-9 at 1M-100M. All other NLSQ cells are below 1e-9.

**The 1M two_exp DNRO is gone: PASS.** `curve_fit_large` two_exp 1M was 7.3e-5 above the best on 2026-11-09 (both devices) and is 3.2e-9 now (CPU 1.50 s, GPU 406 ms).

## Observations

- `curve_fit_large` at 100M: GPU 1.11 / 2.34 / 3.30 / 1.73 s against 12.8 / 12.4 / 13.4 / 25.4 s on 2026-11-09 (11x / 5.3x / 4.1x / 15x); CPU 12.9 / 31.3 / 39.6 / 28.8 s against 27.3 / 23.4 / 45.1 / 40.0 s (gaussian is 1.3x slower, inside this machine's drift). At 10M GPU: 237 / 976 / 1780 / 750 ms (rerun 207 / 355 / 1023 / 336 ms) against 1.18 / 1.21 / 1.31 / 2.87 s.
- Versus JAXopt on GPU: at 100M NLSQ is level on exp_decay (1.11 vs 1.09 s) and faster on gaussian (2.34 vs 3.29 s) and two_exp (3.30 vs 4.20 s); on 2026-11-09 JAXopt led by 3-19x. At 10M JAXopt still leads on exp_decay (82 ms vs 207 ms rerun) and two_exp (354 vs 1023 ms rerun), and is level on gaussian (311 vs 355 ms rerun); at 1M it leads by 3-9x. The gap is therefore closed at 100M, not at 10M and below.
- CPU at 100M `curve_fit_large` is the fastest library on every problem it shares: 12.9 / 31.3 / 39.6 s against JAXopt 16.7 / 45.6 / 45.8 s, lmfit 39.0 / 54.2 / 117 s, SciPy 49.4 / 87.6 / 126 s. bounded_exp (only SciPy, lmfit, NLSQ): 28.8 s against 177 and 182 s.
- 1M boundary: on GPU plain `curve_fit` (55 / 66 / 147 / 819 ms) is faster than `curve_fit_large` (rerun 123 / 183 / 172 / 157 ms) on the three unbounded problems and slower on bounded_exp; on CPU `curve_fit_large` is faster on all four (385 / 833 / 1500 / 577 ms against 1.37 / 3.35 / 2.60 / 3.30 s).
- Small sizes (`curve_fit`, 1k-100k) are unchanged in kind from 2026-11-09 (no code change there beyond the QR covariance gate); absolute times are higher this session (SciPy 1M exp_decay 1.78 s against 1.07 s).
- Optimistix on GPU still fails at 100M.

## Caveats

- Session drift: SciPy and lmfit CPU cells are 1.1-2.3x slower than on 2026-11-09 (SciPy 1M exp_decay 1.78 s against 1.07 s; 100M exp_decay 49.4 s against 43.1 s), JAXopt GPU 1M exp_decay 13.1 ms against 9.3 ms. Compare libraries inside this table, not across dates; only interleaved A/B ratios (`2026-11-10-default-exact-ab.md`) measure code.
- The first-run GPU `curve_fit_large` cells at 1M and 10M (cold start 5-8 s against 1.6-2.5 s on the rerun, warm 1.1-4x slower) coincided with host load. The rerun (`benchmarks/results/2026-11-10/recheck-gpu-{1000000,10000000}/`, `nlsq_large` and `jaxopt` only) is consistent with the A/B session (cand: 1M 0.14 / 0.20 / - / 0.18 s; 10M 0.19 / 0.35 / 1.02 / 0.37 s). Not investigated further.
- CPU 100M SciPy/lmfit are single runs, the JAX-based CPU 100M cells are the median of two runs. The CPU `curve_fit_large` exp_decay 100M cell (12.9 s) was 6.2 s in the A/B session.
- JAXopt/Optimistix have no bounds, sigma or covariance; JAXopt is unmaintained.
- Run on the branch tip `ec9dd429`, tree clean apart from documentation edits and untracked profiler output.
