# Library comparison, 1k / 1M / 10M / 100M points (2026-11-01)

Commit `ba0fda45` (main). Raw data (local, gitignored): `benchmarks/results/2026-11-01/`.

Same machine and isolated venv as `2026-10-31-library-comparison-cpu-gpu-100M.md` (i9-13900H, RTX 4090 Laptop 16 GB, jax[cuda13] 0.12.0.1, scipy 1.18.1, lmfit 1.3.4, jaxopt 0.8.5, optimistix 0.1.0). Harness: `benchmarks/compare/run_library_comparison.py` (fresh process per case, ranking gated on cost within rtol 1e-6 of the best). Warm median, end-to-end mode. Cells are `CPU | GPU`; SciPy and lmfit run on CPU only (one number).

Repeats: 7 warm at 1k/1M, 3 at 10M, 100M: 1 (SciPy, lmfit), 2 (JAX-based, CPU), 3 (JAX-based, GPU). Single run per case, no batch case.

| Case | SciPy | lmfit | NLSQ curve_fit | NLSQ large (GLS) | NLSQ exact_streaming | JAXopt | Optimistix |
|---|---|---|---|---|---|---|---|
| exp_decay 1,000 | 1.0 ms | 1.8 ms | 5.4 ms \| 26.6 ms | 3.8 ms \| 20.1 ms | 60.4 ms \| 91.7 ms | 0.8 ms \| 2.2 ms | 1.1 ms \| 2.9 ms |
| gaussian 1,000 | 1.8 ms | 4.0 ms | 5.5 ms \| 46.7 ms | 6.4 ms \| 43.1 ms | 118.9 ms \| 143.0 ms | 0.7 ms \| 5.2 ms | 0.7 ms \| 5.7 ms |
| two_exp 1,000 | 3.1 ms | 3.5 ms | 9.7 ms \| 83.5 ms | 7.8 ms \| 47.1 ms | 81.9 ms \| 104.3 ms | 1.1 ms \| 7.7 ms | 3.8 ms \| 25.7 ms |
| bounded_exp 1,000 | 4.4 ms | 7.9 ms | 53.2 ms \| 169.1 ms | 102.0 ms \| 141.8 ms | 110.6 ms \| 111.6 ms | n/a \| n/a | n/a \| n/a |
| exp_decay 1,000,000 | 832.8 ms | 272.4 ms | 457.5 ms \| 53.3 ms | 582.2 ms \| 476.9 ms | 191.9 ms \| 451.0 ms | 87.1 ms \| 10.7 ms | 176.3 ms \| 60.0 ms |
| gaussian 1,000,000 | 1.82 s | 475.0 ms | 1.41 s \| 98.3 ms | 808.3 ms \| 688.9 ms | 462.6 ms \| 305.4 ms | 329.0 ms \| 37.8 ms | 629.7 ms \| 179.8 ms |
| two_exp 1,000,000 | 1.77 s | 1.14 s | 1.20 s \| 74.9 ms | DNRO* \| DNRO* | 459.9 ms \| 207.7 ms | 1.04 s \| 41.2 ms | 1.33 s \| 406.1 ms |
| bounded_exp 1,000,000 | 3.43 s | 3.22 s | 1.75 s \| 661.3 ms | 1.99 s \| 2.71 s | 608.8 ms \| 469.1 ms | n/a \| n/a | n/a \| n/a |
| exp_decay 10,000,000 | 5.36 s | 3.45 s | 3.09 s \| 442.8 ms | 3.06 s \| 1.67 s | 1.34 s \| 258.3 ms | 897.3 ms \| 83.4 ms | 1.96 s \| 1.49 s |
| gaussian 10,000,000 | 12.31 s | 5.97 s | 7.06 s \| 706.6 ms | 3.68 s \| 1.41 s | 3.94 s \| 418.8 ms | 3.27 s \| 335.6 ms | 5.85 s \| 2.11 s |
| two_exp 10,000,000 | 14.75 s | 9.94 s | 8.05 s \| 650.2 ms | 7.05 s \| 1.66 s | 6.13 s \| 514.0 ms | 4.82 s \| 375.4 ms | 11.26 s \| 4.35 s |
| bounded_exp 10,000,000 | 28.29 s | 29.76 s | 9.49 s \| 2.37 s | 7.61 s \| 2.53 s | 7.18 s \| 419.5 ms | n/a \| n/a | n/a \| n/a |
| exp_decay 100,000,000 | 100.06 s | 36.59 s | 29.74 s \| FAIL | 28.85 s \| 13.31 s | 8.65 s \| 596.1 ms | 7.33 s \| 743.5 ms | 12.50 s \| FAIL |
| gaussian 100,000,000 | 125.20 s | 55.31 s | 79.85 s \| FAIL | 33.12 s \| 13.59 s | 26.36 s \| 1.87 s | 24.79 s \| 3.05 s | 56.91 s \| FAIL |
| two_exp 100,000,000 | 168.00 s | 118.91 s | 67.06 s \| FAIL | 78.24 s \| 15.79 s | 45.72 s \| 2.76 s | 54.65 s \| 4.17 s | 124.95 s \| FAIL |
| bounded_exp 100,000,000 | 185.83 s | 184.75 s | 72.83 s \| FAIL | 49.83 s \| 23.44 s | 42.59 s \| 1.94 s | n/a \| n/a | n/a \| n/a |

`DNRO*` = did not reach optimum: `nlsq_large` two_exp 1M ended at cost gap 7.3e-5 (gate 1e-6) on both CPU and GPU. `FAIL` = GPU OOM / cuSolver error (plain `curve_fit` and Optimistix at 100M). `n/a` = bounds unsupported.

## Observations

- GPU, `exact_streaming` is the fastest NLSQ path from 10M up. At 100M it takes 0.60 / 1.87 / 2.76 / 1.94 s (exp_decay / gaussian / two_exp / bounded_exp), 6-22x faster than the default `curve_fit_large` (13.3-23.4 s).
- At 100M on GPU it also beats JAXopt on all three unbounded problems (0.60 vs 0.74, 1.87 vs 3.05, 2.76 vs 4.17 s). At 10M and below JAXopt is faster (10M exp_decay 83 vs 258 ms; 1M 11 vs 451 ms); on the unbounded problems plain `curve_fit` is faster than `exact_streaming` at 1M on GPU (exp_decay 53 vs 451 ms), but not on bounded_exp (661 vs 469 ms at 1M; 169 vs 112 ms at 1k).
- `exact_streaming` has a high fixed cost at small n: 1k takes 60-119 ms on CPU and 92-143 ms on GPU (plain `curve_fit` CPU: 5-10 ms, bounded 53 ms).
- Plain `curve_fit` runs out of GPU memory at 100M on all four problems (as on 2026-10-31). Optimistix OOMs or fails in cuSolver (`Dormqr`) at 100M.
- Accuracy: `exact_streaming` final costs are 2.5e-9 to 6.9e-9 above the best (two_exp, bounded_exp); GLS `curve_fit_large` two_exp is 4e-7 (100M), 7e-7 (10M) and 7e-5 at 1M, which fails the 1e-6 gate.
- CPU 100M: `exact_streaming` 8.7 / 26.4 / 45.7 / 42.6 s is 3.7-11.6x faster than SciPy and 2.1-4.3x faster than lmfit. JAXopt CPU is about level on exp_decay (7.3 s) and gaussian (24.8 s), and `exact_streaming` is faster on two_exp (45.7 vs 54.7 s); JAXopt has no bounds.

## Caveats

- Drift on this laptop (governor `powersave`) is up to ~50% between sessions: GPU `curve_fit_large` 100M exp_decay was 9.1 s on 2026-10-31 and 13.3 s now; SciPy exp_decay 100M 94.8 -> 100 s. Treat sub-1.5x gaps as noise.
- CPU 100M: SciPy/lmfit 1 repeat, JAX-based 2 (median of two is their mean).
- JAXopt/Optimistix have no bounds, sigma or covariance; JAXopt is unmaintained.
