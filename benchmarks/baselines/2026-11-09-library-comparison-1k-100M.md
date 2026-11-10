# Library comparison, 1k / 10k / 100k / 1M / 10M / 100M points (2026-11-09)

Commit `4642e940` (branch `fix/review-rounds-2026-11-07`, not `main`). Raw data (local, gitignored): `benchmarks/results/2026-11-09/`. Same machine as `2026-11-06-library-comparison-curve_fit-vs-large.md` (i9-13900H, RTX 4090 Laptop 16 GB), project `.venv`: jax/jaxlib 0.12.0.1 (CUDA), scipy 1.18.1, lmfit 1.3.4, jaxopt 0.8.5, optimistix 0.1.0. CPU runs use `JAX_PLATFORMS=cpu`. CPU governor `powersave` (unchanged). Same harness (`benchmarks/compare/run_library_comparison.py`, fresh process per case, `--no-batch`). Warm median, end-to-end. Cells are `CPU | GPU`; SciPy and lmfit are CPU only.

API split: `curve_fit` at 1k / 10k / 100k / 1M, `curve_fit_large` (default strategy) at 1M / 10M / 100M. 1M runs both because it sits on the boundary. `exact_streaming` not run.

Repeats: 7 warm at 1k-1M, 3 at 10M, 100M: 3 (GPU), 2 (JAX-based CPU), 1 (SciPy, lmfit).

| Case | SciPy | lmfit | NLSQ `curve_fit` | NLSQ `curve_fit_large` | JAXopt | Optimistix |
|---|---|---|---|---|---|---|
| exp_decay 1,000 | 1.0 ms | 1.9 ms | 5.0 ms \| 17.6 ms | — | 0.3 ms \| 2.1 ms | 1.0 ms \| 3.3 ms |
| gaussian 1,000 | 1.7 ms | 2.3 ms | 6.4 ms \| 26.3 ms | — | 0.8 ms \| 5.8 ms | 1.1 ms \| 5.4 ms |
| two_exp 1,000 | 3.2 ms | 3.9 ms | 10.8 ms \| 35.2 ms | — | 1.5 ms \| 6.8 ms | 4.3 ms \| 25.6 ms |
| bounded_exp 1,000 | 4.5 ms | 7.4 ms | 53.5 ms \| 198.1 ms | — | n/a | n/a |
| exp_decay 10,000 | 3.5 ms | 3.8 ms | 10.5 ms \| 14.9 ms | — | 2.1 ms \| 4.2 ms | 1.9 ms \| 4.3 ms |
| gaussian 10,000 | 6.2 ms | 5.5 ms | 14.2 ms \| 27.5 ms | — | 5.4 ms \| 6.8 ms | 3.3 ms \| 7.3 ms |
| two_exp 10,000 | 8.2 ms | 8.2 ms | 15.5 ms \| 20.6 ms | — | 8.3 ms \| 7.9 ms | 8.0 ms \| 11.4 ms |
| bounded_exp 10,000 | 12.6 ms | 15.3 ms | 60.3 ms \| 145.7 ms | — | n/a | n/a |
| exp_decay 100,000 | 96.5 ms | 26.1 ms | 50.9 ms \| 20.1 ms | — | 11.5 ms \| 3.4 ms | 20.5 ms \| 10.8 ms |
| gaussian 100,000 | 169.4 ms | 44.4 ms | 119.2 ms \| 31.5 ms | — | 38.2 ms \| 10.8 ms | 65.9 ms \| 27.8 ms |
| two_exp 100,000 | 415.4 ms | 74.8 ms | 112.4 ms \| 23.2 ms | — | 43.8 ms \| 12.9 ms | 97.8 ms \| 45.9 ms |
| bounded_exp 100,000 | 380.0 ms | 532.9 ms | 351.9 ms \| 492.9 ms | — | n/a | n/a |
| exp_decay 1,000,000 | 1.07 s | 444.6 ms | 663.5 ms \| 58.8 ms | 329.5 ms \| 208.1 ms | 115.3 ms \| 9.3 ms | 260.2 ms \| 58.5 ms |
| gaussian 1,000,000 | 1.86 s | 727.9 ms | 1.50 s \| 111.1 ms | 411.2 ms \| 224.9 ms | 443.7 ms \| 31.7 ms | 835.3 ms \| 177.1 ms |
| two_exp 1,000,000 | 2.24 s | 1.16 s | 1.74 s \| 80.7 ms | DNRO* | 712.3 ms \| 52.4 ms | 1.72 s \| 576.4 ms |
| bounded_exp 1,000,000 | 3.94 s | 3.83 s | 1.86 s \| 732.4 ms | 1.41 s \| 1.72 s | n/a | n/a |
| exp_decay 10,000,000 | 7.24 s | 5.26 s | — | 3.59 s \| 1.18 s | 1.23 s \| 66.3 ms | 2.97 s \| 639.1 ms |
| gaussian 10,000,000 | 13.40 s | 9.25 s | — | 4.85 s \| 1.21 s | 4.73 s \| 314.1 ms | 9.53 s \| 2.10 s |
| two_exp 10,000,000 | 19.93 s | 14.23 s | — | 10.50 s \| 1.31 s | 5.99 s \| 349.1 ms | 18.99 s \| 4.34 s |
| bounded_exp 10,000,000 | 28.24 s | 28.92 s | — | 7.11 s \| 2.87 s | n/a | n/a |
| exp_decay 100,000,000 | 43.06 s | 33.28 s | — | 27.33 s \| 12.80 s | 7.16 s \| 663.9 ms | 14.23 s \| FAIL |
| gaussian 100,000,000 | 74.14 s | 52.13 s | — | 23.38 s \| 12.44 s | 23.19 s \| 2.95 s | 45.81 s \| FAIL |
| two_exp 100,000,000 | 115.68 s | 88.48 s | — | 45.07 s \| 13.38 s | 38.89 s \| 4.76 s | 74.99 s \| FAIL |
| bounded_exp 100,000,000 | 157.84 s | 250.78 s | — | 40.03 s \| 25.37 s | n/a | n/a |

`DNRO*` = did not reach optimum: `curve_fit_large` two_exp 1M ended at relative cost gap 7.3e-5 above the best (gate 1e-6) on both CPU (1.36 s) and GPU (392 ms), the same gap as on 2026-11-01. The 1M `curve_fit_large` runs were a separate invocation; their gaps were recomputed against the best cost across all runs of the case. Every other ok cell is within 1e-6. `FAIL` = Optimistix GPU OOM / cuSolver `Dormqr` error at 100M. `n/a` = bounds unsupported. No case timed out.

## Observations

- Small (`curve_fit`, 1k-100k): on CPU NLSQ is 2-5x slower than SciPy at 1k-10k (bounded 5-12x). It overtakes SciPy at 100k on every problem (two_exp 112 vs 415 ms), but lmfit is still faster there on the unbounded problems. The GPU adds a fixed cost of 15-35 ms (bounded 150-200 ms) and becomes faster than NLSQ CPU only at 100k on the unbounded problems (20-32 ms). Bounded on GPU is slower than on CPU up to 100k (493 vs 352 ms).
- JAXopt is faster than NLSQ `curve_fit` at every small size on both backends (100k GPU 3.4 / 10.8 / 12.9 ms vs 20.1 / 31.5 / 23.2 ms). At 10k on CPU, JAXopt and Optimistix are 1-2x faster than SciPy, while NLSQ is 2-3x slower.
- 1M boundary: on GPU, plain `curve_fit` (59 / 111 / 81 / 732 ms) is faster than `curve_fit_large` (208 / 225 / DNRO / 1.72 s). On CPU, `curve_fit_large` is faster (330 / 411 / DNRO / 1.41 s vs 664 ms / 1.50 / 1.74 / 1.86 s). So on GPU the large path pays off only above 1M; the two_exp gap rules out `curve_fit_large` at 1M for that problem.
- Large (`curve_fit_large`): GPU 10M 1.18-2.87 s, 100M 12.4-25.4 s. JAXopt GPU is 3-19x faster at 10M-100M on the unbounded problems. On CPU at 100M NLSQ is 1.6-3.9x faster than SciPy and 1.2-6.3x faster than lmfit (largest gain on bounded_exp, which only SciPy, lmfit and NLSQ support). JAXopt CPU is level on gaussian and faster on exp_decay and two_exp. Optimistix fails on GPU at 100M.
- Versus 2026-11-06 (overlapping sizes): GPU `curve_fit_large` 100M 19.6 / 18.1 / 13.9 / 23.0 s -> 12.8 / 12.4 / 13.4 / 25.4 s; 10M 1.51 / 1.59 / 1.70 / 3.20 s -> 1.18 / 1.21 / 1.31 / 2.87 s. GPU `curve_fit` 1M bounded 483 -> 732 ms (1.5x, at the noise threshold). CPU 100M SciPy 63.6 / 110 / 173 / 245 s -> 43.1 / 74.1 / 116 / 158 s (single runs). Everything else is within the sub-1.5x noise band, so no regression or improvement is implied. 10k and 100k have no prior baseline.

## Caveats

- Laptop drift between sessions is up to ~50% (governor `powersave`); treat sub-1.5x gaps as noise. CPU 100M SciPy/lmfit are single runs, and the JAX-based CPU cells are the median of two runs (= their mean).
- Run on the feature branch head `4642e940` (clean tree apart from untracked profiler output), not on `main`.
- JAXopt/Optimistix have no bounds, sigma or covariance; JAXopt is unmaintained.

## Addendum (2026-11-10): slowdown analysis and size ranges

### Is NLSQ getting slower? No, the code has not regressed; the differences come from machine drift

- **nfev is identical** in every session from 2026-10-30 to 2026-11-09 for each `curve_fit` case (for example exp_decay 1M = 7, bounded_exp = 18). No fix changed the iteration path.
- **Interleaved A/B, `main` (`36d56ccc`, includes the perf PRs #61/#62) vs this branch (`4642e940`, +9 review-fix commits)**: 3 rounds per arm, same session, run with `benchmarks/studies/ab_nlsq.py`. Raw data is in `benchmarks/results/2026-11-10-ab-main-vs-head/`. Ratios are branch / main.

  | Group | ratios (exp_decay / gaussian / two_exp / bounded_exp) |
  |---|---|
  | GPU `curve_fit` 1k | 0.99 / 1.00 / 1.06 / 0.96 |
  | GPU `curve_fit` 1M | 1.01 / 1.00 / 1.01 / 1.10 (spread 0.43) |
  | GPU `curve_fit_large` 10M | 1.00 / 0.98 / 1.02 / 1.00 |
  | CPU `curve_fit` 1k | 1.07 / 1.08 / 0.97 / 1.05 (spread 0.10-0.23) |
  | CPU `curve_fit` 1M | 1.03 / 1.01 / 1.00 / 0.99 |
  | CPU `curve_fit_large` 10M | 1.01 / 1.00 / 1.01 / 1.01 |

  Every ratio is within the in-session spread, and the review fixes cost at most a few percent.
- **The same code at a different time of day**: in the A/B session (06:30) the branch ran GPU `curve_fit_large` 10M in 0.80 / 0.82 / 0.88 / 1.98 s. The 23:40 run in the table above gave 1.18 / 1.21 / 1.31 / 2.87 s, about 1.45x slower. The GPU 1k exp_decay cell was 13 ms vs 17.6 ms. The same pattern appears in the 2026-11-06 data: `main` at `87d8912a` ran GPU 10M exp_decay in 1.51 s in the baseline session and in 0.88 s in the perf A/B session.
- **Why NLSQ drifts and JAXopt does not**: JAXopt on GPU is steady in every session (10M: 66 / 299-318 / 346-366 ms). Its whole solve is one compiled XLA program. NLSQ's TRF loop and the `curve_fit_large` chunk loop are driven from Python on the host, with per-iteration dispatch and host syncs. Their wall time therefore follows the host CPU clock, which the `powersave` governor and laptop thermals move by up to ~50%. Only interleaved A/B ratios measure code changes; absolute times from different sessions measure the machine.

### Best size ranges (this machine, 3-4 parameters)

| Backend | `curve_fit` | `curve_fit_large` (default) | Evidence |
|---|---|---|---|
| GPU | ~10k to ~10M | only above ~10-30M, where `curve_fit` OOMs (100M on 16 GB) | 1M: `curve_fit` 59-111 ms vs large 208-392 ms (2026-11-09). 10M, same session 2026-11-01: 443 / 707 / 650 / 2368 ms vs 1.67 / 1.41 / 1.66 / 2.53 s |
| CPU | 1k to ~1M | from ~1M up | 1M (2026-11-09): large 329 / 411 / 1412 ms vs `curve_fit` 664 / 1501 / 1862 ms, but on 2026-11-01 exp_decay favoured `curve_fit` (457 vs 582 ms). 10M-100M: large equal or up to 2.4x faster |

- **Below ~10k, use the CPU**: GPU `curve_fit` has a fixed cost of 15-35 ms, while CPU takes 5-16 ms. With bounds the GPU is slower than the CPU up to about 100k (493 vs 352 ms at 100k), so on GPU bounded problems pay off only from ~1M.
- **`curve_fit_large` accuracy caveat**: on two_exp at 1M it stops 7.3e-5 above the optimum on both backends (2026-11-01 and 2026-11-09), so `curve_fit` is the safer choice at that size.
- **Above ~10M, `method="exact_streaming"` beats the default `curve_fit_large`**: on 2026-11-01 it took 0.26-0.51 s at 10M and 0.6-2.8 s at 100M on GPU, against 1.4-2.5 s and 13-23 s, and 8.6-46 s at 100M on CPU against 29-78 s. On GPU it also beats plain `curve_fit` at 10M on all four problems (258 / 419 / 514 / 419 ms). It has a fixed cost of 60-140 ms, so it does not pay off below ~100k.
