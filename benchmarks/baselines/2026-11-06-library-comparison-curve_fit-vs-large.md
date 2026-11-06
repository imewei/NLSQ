# Library comparison, 1k / 1M / 10M / 100M points (2026-11-06)
Commit `87d8912a` (main). Raw data (local, gitignored): `benchmarks/results/2026-11-06/`. Machine and pins as in `2026-11-01-library-comparison-1k-1M-10M-100M.md` (i9-13900H, RTX 4090 Laptop 16 GB, jax[cuda13] 0.12.0.1, scipy 1.18.1, lmfit 1.3.4, jaxopt 0.8.5, optimistix 0.1.0), fresh isolated venv. Same harness (fresh process per case, ranking gated on cost within rtol 1e-6; no case missed the gate). Warm median, end-to-end.
NLSQ column: `curve_fit` at 1k and 1M, `curve_fit_large` (default strategy) at 10M and 100M. `exact_streaming` not run. Cells are `CPU | GPU`; SciPy and lmfit are CPU only.
Repeats: 7 warm at 1k/1M, 3 at 10M, 100M: 1 (SciPy, lmfit), 2 (JAX-based, CPU), 3 (JAX-based, GPU).

| Case | SciPy | lmfit | NLSQ | JAXopt | Optimistix |
|---|---|---|---|---|---|
| exp_decay 1,000 | 1.0 ms | 1.8 ms | 4.2 ms \| 22.9 ms | 0.8 ms \| 3.1 ms | 0.7 ms \| 3.1 ms |
| gaussian 1,000 | 1.7 ms | 2.2 ms | 5.8 ms \| 28.7 ms | 0.9 ms \| 4.9 ms | 1.2 ms \| 5.1 ms |
| two_exp 1,000 | 3.1 ms | 3.4 ms | 7.5 ms \| 50.6 ms | 1.0 ms \| 6.6 ms | 5.1 ms \| 26.8 ms |
| bounded_exp 1,000 | 4.3 ms | 7.2 ms | 47.3 ms \| 170.4 ms | n/a | n/a |
| exp_decay 1,000,000 | 736.4 ms | 266.6 ms | 394.4 ms \| 75.7 ms | 90.4 ms \| 12.7 ms | 169.3 ms \| 62.6 ms |
| gaussian 1,000,000 | 1.31 s | 425.2 ms | 863.3 ms \| 94.1 ms | 290.0 ms \| 40.2 ms | 542.5 ms \| 180.4 ms |
| two_exp 1,000,000 | 1.42 s | 721.9 ms | 945.3 ms \| 86.8 ms | 505.5 ms \| 44.1 ms | 964.5 ms \| 580.8 ms |
| bounded_exp 1,000,000 | 4.16 s | 4.15 s | 1.96 s \| 482.8 ms | n/a | n/a |
| exp_decay 10,000,000 | 7.64 s | 5.40 s | 4.40 s \| 1.51 s | 1.27 s \| 77.6 ms | 2.85 s \| 657.7 ms |
| gaussian 10,000,000 | 13.11 s | 8.98 s | 5.49 s \| 1.59 s | 4.83 s \| 331.9 ms | 9.21 s \| 2.11 s |
| two_exp 10,000,000 | 20.15 s | 14.70 s | 11.11 s \| 1.70 s | 6.14 s \| 447.8 ms | 20.66 s \| 4.35 s |
| bounded_exp 10,000,000 | 28.81 s | 29.01 s | 7.44 s \| 3.20 s | n/a | n/a |
| exp_decay 100,000,000 | 63.56 s | 36.00 s | 40.57 s \| 19.61 s | 12.15 s \| 881.7 ms | 24.15 s \| FAIL |
| gaussian 100,000,000 | 109.58 s | 85.16 s | 44.42 s \| 18.10 s | 41.60 s \| 3.16 s | 86.23 s \| FAIL |
| two_exp 100,000,000 | 172.58 s | 135.26 s | 49.16 s \| 13.91 s | 39.18 s \| 4.06 s | 75.73 s \| FAIL |
| bounded_exp 100,000,000 | 245.16 s | 272.56 s | 43.81 s \| 23.03 s | n/a | n/a |
`FAIL` = Optimistix GPU OOM / cuSolver `Dormqr` error at 100M. `n/a` = bounds unsupported.

## Observations
- Small (`curve_fit`): at 1k the NLSQ CPU path is 4-8 ms (bounded 47 ms), 2.4-4x slower than SciPy unbounded (bounded 11x); GPU adds 20-50 ms fixed cost (bounded 170 ms). At 1M, GPU `curve_fit` takes 76-94 ms unbounded and 483 ms bounded, 4-11x faster than NLSQ CPU and 3.5-16x faster than SciPy/lmfit. CPU NLSQ is 1.5-2.1x faster than SciPy and slower than lmfit on the unbounded problems at 1M.
- JAXopt is faster than NLSQ at every size up to 10M, on both backends (1M GPU 12.7 / 40.2 / 44.1 ms vs 75.7 / 94.1 / 86.8 ms). Optimistix is faster than NLSQ on CPU at 1M for exp_decay and gaussian (level on two_exp) and slower than NLSQ on GPU at 1M two_exp (581 vs 87 ms).
- Large (`curve_fit_large`, GLS default): GPU 10M 1.5-3.2 s, 100M 13.9-23.0 s. It is the only NLSQ path that runs at 100M on GPU here; JAXopt GPU takes 0.88 / 3.16 / 4.06 s at 100M (3.4-22x faster) and Optimistix fails.
- Large on CPU at 100M: 40.6 / 44.4 / 49.2 / 43.8 s vs SciPy 63.6-245 s and lmfit 36-273 s. On the bounded problem NLSQ is 5.6-6.2x faster than SciPy/lmfit; on unbounded problems lmfit beats it on exp_decay (36.0 vs 40.6 s) and loses on two_exp (135 vs 49 s). JAXopt CPU is faster on exp_decay, gaussian and two_exp.
- Versus 2026-11-01 (same machine): GPU `curve_fit_large` 100M exp_decay 13.3 -> 19.6 s, gaussian 13.6 -> 18.1 s, two_exp 15.8 -> 13.9 s, bounded 23.4 -> 23.0 s; SciPy exp_decay 100M 100 -> 63.6 s (1 repeat). Spreads of this size match the previous session's drift; no code change is implied.

## Caveats
- Laptop drift between sessions up to ~50% (governor `powersave`); treat sub-1.5x gaps as noise. CPU 100M SciPy/lmfit are single runs, JAX-based CPU two (median of two = mean).
- The 1M row uses plain `curve_fit` and the 10M row `curve_fit_large` by choice of the split; the previous report showed both at every size.
- JAXopt/Optimistix have no bounds, sigma or covariance; JAXopt is unmaintained.
