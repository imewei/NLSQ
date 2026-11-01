# Library comparison: CPU vs GPU, 1k to 100M points (2026-10-31)

Commit `6c0814b`. Raw data (local, gitignored): `benchmarks/results/2026-10-31/`.

## Setup

- CPU: i9-13900H (20 logical), 62 GB RAM. GPU: RTX 4090 Laptop, 16 GB, CUDA 13.4 driver.
- Isolated venv (project `.venv` untouched): `jax[cuda13]==0.12.0.1`, scipy 1.18.1, lmfit 1.3.4,
  jaxopt 0.8.5, optimistix 0.1.0, nlsq 0.7.7.dev3 (editable). CPU runs use `JAX_PLATFORMS=cpu`
  in the same venv.
- Harness: `benchmarks/compare/run_library_comparison.py` (fresh process per case; ranking gated
  on final cost within rtol 1e-6 of the best in the case). Warm = median of repeats.
- Sequential runs only; never CPU and GPU runs concurrently.

Commands:

```bash
# 1k / 100k / 1M, 7 warm repeats, all libraries (once with JAX_PLATFORMS=cpu, once on GPU)
python -m benchmarks.compare.run_library_comparison --sizes 1000 100000 1000000 --out <dir>
# 100M, no batch
JAX_PLATFORMS=cpu python -m benchmarks.compare.run_library_comparison --sizes 100000000 \
    --no-batch --libraries scipy lmfit --repeats 1 --out <dir>
JAX_PLATFORMS=cpu python -m benchmarks.compare.run_library_comparison --sizes 100000000 \
    --no-batch --libraries nlsq jaxopt optimistix --repeats 3 --out <dir>
python -m benchmarks.compare.run_library_comparison --sizes 100000000 --no-batch \
    --libraries nlsq jaxopt optimistix --repeats 3 --out <dir>          # GPU
# NLSQ large-dataset path (harness adapter only calls plain curve_fit)
PYTHONPATH=. python benchmarks/results/2026-10-31/nlsq_large.py chunked <problem> 100000000 3
```

## 100M points, warm median

| Problem | SciPy | lmfit | NLSQ `curve_fit` CPU | NLSQ `curve_fit_large` CPU | NLSQ `curve_fit_large` GPU | JAXopt CPU | JAXopt GPU | Optimistix CPU | Optimistix GPU |
|---|---|---|---|---|---|---|---|---|---|
| exp_decay | 94.8 s | 54.8 s | 27.3 s | 25.8 s | 9.1 s | 7.3 s | 0.78 s | 12.3 s | failed (cuSolver) |
| gaussian | 88.6 s | 54.8 s | 59.2 s | 27.4 s | 9.2 s | 22.6 s | 3.0 s | 41.6 s | OOM |
| two_exp | 142 s | 95.6 s | 60.8 s | 59.6 s | 10.2 s | 39.0 s | 4.9 s | 75.8 s | OOM |
| bounded_exp | 185 s | 212 s | 61.6 s | 45.1 s | 16.1 s | n/a | n/a | n/a | n/a |

- Plain `nlsq.curve_fit` ran out of GPU memory (one 3-4 GiB allocation on the 16 GB card) on
  all four problems at 100M.
- `curve_fit_large` (chunked, per-chunk fit combined by GLS) reached the best cost within
  1.5e-13 (exp_decay), 1.1e-14 (gaussian), 4.2e-7 (two_exp); bounded_exp was 2.6e-9 *below*
  the best other library.
- `method="hybrid_streaming"`: 87 s for exp_decay at 2M points on GPU (chunked: 0.32 s at 2M).
  Not run at 100M (extrapolates to over an hour per fit).
- Cold first call, `curve_fit_large`: 10-18 s GPU, 27-60 s CPU.

## 1k to 1M points (warm median, CPU | GPU)

| Case | SciPy | lmfit | NLSQ | JAXopt | Optimistix |
|---|---|---|---|---|---|
| exp_decay 1k | 1.0 \| 1.1 ms | 1.9 \| 1.8 ms | 5.4 \| 23.8 ms | 0.76 \| 2.4 ms | 0.97 \| 3.5 ms |
| exp_decay 100k | 101 \| 141 ms | 26 \| 26 ms | 56 \| 24 ms | 9.0 \| 3.9 ms | 23 \| 12 ms |
| exp_decay 1M | 922 \| 906 ms | 427 \| 409 ms | 667 \| 66 ms | 164 \| 12 ms | 420 \| 62 ms |
| gaussian 1M | 1.37 \| 1.55 s | 615 \| 647 ms | 1.82 s \| 86 ms | 762 \| 40 ms | 1.24 s \| 198 ms |
| two_exp 1M | 1.87 \| 1.63 s | 1.02 s \| 820 ms | 2.60 s \| 75 ms | 1.36 s \| 39 ms | 2.80 s \| 405 ms |
| bounded_exp 1M | 2.99 \| 3.03 s | 3.14 \| 3.17 s | 2.13 s \| 519 ms | n/a | n/a |
| exp_decay, 256 fits x 200 pts | 211 \| 204 ms | 423 \| 429 ms | 1.18 \| 6.21 s | 24 \| 3.8 ms | 86 \| 169 ms |

NLSQ on GPU is slower than on CPU for small n (1k: 5 -> 24 ms) and for the non-vmapped batch.

## Caveats

- SciPy/lmfit run identical code in the CPU and GPU runs yet differ by up to ~40% at 100k
  (exp_decay 101 vs 141 ms): treat differences under ~1.5x at 100k as noise.
- SciPy/lmfit used 1 warm repeat at 100M; JAX libraries and NLSQ used 3. Single run per case.
- Laptop GPU (thermal limits). Stopping criteria are analogous, not identical.
- JAXopt and Optimistix have no bounds, sigma or covariance support; JAXopt is unmaintained.

## Profile of `curve_fit_large` at 100M points (GPU, chunked path)

Script: `benchmarks/results/2026-10-31/prof_large.py <problem> 100000000` (stage timers around
`LargeDatasetFitter._fit_chunked` internals, median of 3 warm runs, plus one `cProfile` pass).
100 chunks of 1M points; about 2 TRF iterations per chunk (204 total for exp_decay).

**Machine drift:** the same uninstrumented run took 9.07 s when benchmarked and 13.8 s about
90 minutes later. The host shuffle alone went 4.2 s -> 6.26 s (about 1.5x), so the host-bound
path scales with host speed; the profile below is from the slow state. Absolute NLSQ
`curve_fit_large` times are therefore only reproducible to roughly +-50% on this laptop
(governor `powersave`, cause of the slowdown not identified); stage fractions are the stable
result.

exp_decay, warm median 13.98 s:

| Stage | Time | Share |
|---|---|---|
| Host shuffle (`permutation` + `x[idx]`, `y[idx]`, `large_dataset.py:2707-2709`) | 6.26 s | 45% |
| Per-chunk `curve_fit` calls (100) | 5.27 s | 38% |
| of which `_compute_covariance` (SVD, 17 ms per chunk) | about 1.7 s | 12% |
| of which TRF loop | about 1.6 s | 11% |
| of which initial residual + Jacobian | about 0.9 s | 6% |
| of which model tracing / wrapper binding | about 0.55 s | 4% |
| `_chunk_ssr` (re-evaluates the model with NumPy on 1M points, downloads predictions) | 1.07 s | 8% |
| `create_chunks` (copy into pooled buffers) | 0.76 s | 5% |
| `gc.collect` (10 calls) | 0.63 s | 4% |
| GLS update, validation, result assembly | under 0.05 s | 0% |

Other problems (same fixed cost, different per-chunk cost):

| Problem | Warm total | Shuffle + gc + loop | Per-chunk `curve_fit` | `chunk_ssr` |
|---|---|---|---|---|
| exp_decay | 13.98 s | 6.89 s | 5.27 s | 1.07 s |
| two_exp | 16.83 s | 6.98 s | 7.65 s | 1.36 s |
| bounded_exp | 25.29 s | 7.02 s | 16.39 s | 1.04 s |

Device-to-host syncs (`array.py:_value`, 2128 calls, 1.46 s) are where the GPU's work shows up;
the GPU itself is not the bottleneck.

Stages that exist only because of per-chunk fitting plus GLS (shuffle, per-chunk covariance,
`_chunk_ssr`, chunk copies, gc) add up to about 10.5 s of 13.98 s on exp_decay (75%). An exact
accumulated solver would not need them; the saving is not measured.

## Exact chunk-accumulated LM prototype vs `curve_fit_large` vs JAXopt (100M points)

Prototype: `benchmarks/prototypes/exact_accumulated_lm.py` (not in `nlsq/`). One pass over the data
per trial point accumulates (0.5*sum r^2, J^T r, J^T J) in float64; O(p^2) state; result is
independent of chunking (asserted against `scipy.optimize.least_squares` in `--selftest`).
All three measured in the same session, 3 warm repeats, median. Prototype data resident on the
GPU (excludes the one-off ~1.6 GB host-to-device copy; JAXopt's `end_to_end` includes it).

| Problem | Backend | Prototype | `curve_fit_large` | JAXopt | Prototype passes |
|---|---|---|---|---|---|
| exp_decay | GPU | 0.66 s | 9.29 s | 0.80 s | 6 |
| gaussian | GPU | 1.79 s | 9.44 s | 3.14 s | 14 |
| two_exp | GPU | 2.80 s | 10.38 s | 5.05 s | 19 |
| bounded_exp | GPU | 2.14 s | 17.67 s | n/a | 22 |
| exp_decay | CPU | 9.42 s | 26.1 s | 11.2 s | 6 |
| gaussian | CPU | 21.7 s | 27.8 s | 38.9 s | 14 |
| two_exp | CPU | 32.3 s | 60.2 s | 61.4 s | 19 |
| bounded_exp | CPU | 48.8 s | 45.4 s | n/a | 22 |

- Prototype with data streamed from host memory each pass (GPU, exp_decay): 2.07 s.
- Final cost vs best recorded: prototype within 3e-13 (unbounded), 2.9e-9 (bounded_exp);
  `curve_fit_large` 4.2e-7 on two_exp (per-chunk GLS combine is approximate for nonlinear models).
- JAXopt CPU was 1.5-1.6x slower than its first 100M run (7.3/22.6/39.0 s then); treat CPU gaps
  under ~1.5x as noise. `curve_fit_large` GPU returned to ~9.3 s, so the earlier 13.8 s was transient.
- Prototype limits: normal equations (squares cond(J)), no sigma/weights, bounds by active-set
  projection (not full TRF), no 1B-point run, host-streaming only tried on exp_decay.
