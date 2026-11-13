# Library comparison, 1k / 10k / 100k / 1M / 10M / 100M points (2026-11-13)

Commit `343af740` (branch `fix/open-items-2-3-6`, PR #70). Raw data (local, gitignored): `benchmarks/results/2026-11-13/`. Same machine and harness as the 2026-11-10 run (i9-13900H, RTX 4090 Laptop 16 GB), project `.venv`, `--no-batch`, fresh process per case, warm median, end-to-end. CPU runs use `JAX_PLATFORMS=cpu`; governor `powersave`. Cells are `CPU / GPU`; SciPy and lmfit are CPU only. API split as before: `curve_fit` at 1k-1M, `curve_fit_large` at 1M-100M (default exact streaming).

**Machine state: not quiet.** Load average was 9.9 / 8.3 / 7.5 at the start and 31 / 22 / 15 at the end. Right after the run, `ps` showed six Julia processes (not from this run) at 90-210% CPU each; what loaded the host earlier in the run was not identified. Host-driven loops (NLSQ) are sensitive to this, GPU-resident single-program libraries (JAXopt GPU, Optimistix GPU) much less.

Invocations (one `--out` each):

```bash
R() { .venv/bin/python -m benchmarks.compare.run_library_comparison --no-batch --out benchmarks/results/2026-11-13/$1 "${@:2}"; }
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

Repeats: 7 warm at 1k-1M, 3 at 10M, 100M: 3 (GPU), 2 (JAX-based CPU), 1 (SciPy, lmfit).

| Case | SciPy | lmfit | NLSQ curve_fit (CPU / GPU) | NLSQ curve_fit_large (CPU / GPU) | JAXopt (CPU / GPU) | Optimistix (CPU / GPU) |
|---|---|---|---|---|---|---|
| exp_decay 1,000 | 1.1 ms | 4.6 ms | 4.0 ms / 24.8 ms | — / — | 0.4 ms / 2.0 ms | 0.5 ms / 3.0 ms |
| gaussian 1,000 | 1.7 ms | 2.4 ms | 5.3 ms / 39.2 ms | — / — | 0.9 ms / 5.5 ms | 1.2 ms / 5.0 ms |
| two_exp 1,000 | 3.1 ms | 3.5 ms | 7.0 ms / 27.1 ms | — / — | 1.0 ms / 6.8 ms | 2.0 ms / 25.6 ms |
| bounded_exp 1,000 | 4.4 ms | 7.2 ms | 42.8 ms / 139.2 ms | — / — | — / — | — / — |
| exp_decay 10,000 | 3.9 ms | 3.8 ms | 11.8 ms / 30.9 ms | — / — | 1.4 ms / 5.0 ms | 3.8 ms / 3.9 ms |
| gaussian 10,000 | 6.3 ms | 5.3 ms | 10.8 ms / 24.8 ms | — / — | 4.0 ms / 6.1 ms | 6.3 ms / 7.1 ms |
| two_exp 10,000 | 8.3 ms | 7.9 ms | 15.1 ms / 20.5 ms | — / — | 4.5 ms / 8.4 ms | 4.8 ms / 11.8 ms |
| bounded_exp 10,000 | 12.1 ms | 14.6 ms | 51.2 ms / 111.6 ms | — / — | — / — | — / — |
| exp_decay 100,000 | 111.2 ms | 24.5 ms | 44.4 ms / 48.0 ms | — / — | 8.6 ms / 4.1 ms | 19.3 ms / 11.1 ms |
| gaussian 100,000 | 164.1 ms | 39.1 ms | 64.7 ms / 27.8 ms | — / — | 37.2 ms / 10.7 ms | 58.5 ms / 29.6 ms |
| two_exp 100,000 | 312.8 ms | 64.4 ms | 87.0 ms / 27.6 ms | — / — | 42.3 ms / 12.6 ms | 82.2 ms / 52.0 ms |
| bounded_exp 100,000 | 524.1 ms | 454.2 ms | 311.4 ms / 383.6 ms | — / — | — / — | — / — |
| exp_decay 1,000,000 | 1.14 s | 464.0 ms | 362.3 ms / 140.5 ms | 90.8 ms / 18.5 ms | 82.8 ms / 11.6 ms | 159.7 ms / 62.6 ms |
| gaussian 1,000,000 | 1.53 s | 432.4 ms | 562.0 ms / 165.8 ms | 252.5 ms / 43.5 ms | 320.9 ms / 36.7 ms | 577.0 ms / 196.1 ms |
| two_exp 1,000,000 | 1.56 s | 737.0 ms | 930.4 ms / 115.5 ms | 328.2 ms / 48.4 ms | 482.9 ms / 47.1 ms | 984.4 ms / 405.3 ms |
| bounded_exp 1,000,000 | 3.21 s | 3.18 s | 1.28 s / 538.6 ms | 293.3 ms / 49.0 ms | — / — | — / — |
| exp_decay 10,000,000 | 5.26 s | 4.45 s | — / — | 977.2 ms / 78.1 ms | 945.4 ms / 77.3 ms | 2.25 s / 657.0 ms |
| gaussian 10,000,000 | 9.05 s | 5.40 s | — / — | 3.83 s / 217.4 ms | 3.15 s / 334.9 ms | 7.67 s / 2.11 s |
| two_exp 10,000,000 | 14.20 s | 9.43 s | — / — | 4.26 s / 296.5 ms | 3.74 s / 437.9 ms | 11.12 s / 4.35 s |
| bounded_exp 10,000,000 | 19.43 s | 19.97 s | — / — | 3.92 s / 256.7 ms | — / — | — / — |
| exp_decay 100,000,000 | 65.94 s | 53.46 s | — / — | 6.08 s / 618.1 ms | 7.70 s / 801.0 ms | 12.91 s / FAIL |
| gaussian 100,000,000 | 120.34 s | 54.72 s | — / — | 23.61 s / 1.63 s | 24.41 s / 2.84 s | 59.69 s / FAIL |
| two_exp 100,000,000 | 139.54 s | 107.25 s | — / — | 28.22 s / 2.22 s | 41.55 s / 4.24 s | 159.60 s / FAIL |
| bounded_exp 100,000,000 | 170.56 s | 342.39 s | — / — | 34.09 s / 1.65 s | — / — | — / — |

Optimistix GPU at 100M fails as in every earlier run.

COST GATE (nlsq / nlsq_large vs best across all libs+devices, relative):
[(6.862857174961217e-09, ('two_exp', 100000000, 'nlsq_large', 'gpu')), (6.862857174961217e-09, ('two_exp', 100000000, 'nlsq_large', 'cpu')), (6.014447015969453e-09, ('two_exp', 10000000, 'nlsq_large', 'gpu')), (6.014447015969453e-09, ('two_exp', 10000000, 'nlsq_large', 'cpu')), (3.2038500303384622e-09, ('two_exp', 1000000, 'nlsq_large', 'gpu'))]

RATIO new/old (NLSQ only; old = 2026-11-11 NLSQ rerun), >1.25 or <0.8 flagged:
median ratio 0.9620580029646546 n 56
0.35 ('gaussian', 1000000, 'nlsq', 'cpu') new 562.0 ms old 1.61 s
0.40 ('two_exp', 1000, 'nlsq', 'cpu') new 7.0 ms old 17.5 ms
0.48 ('gaussian', 1000, 'nlsq', 'cpu') new 5.3 ms old 11.1 ms
0.49 ('exp_decay', 1000, 'nlsq', 'cpu') new 4.0 ms old 8.2 ms
0.59 ('gaussian', 100000, 'nlsq', 'cpu') new 64.7 ms old 109.7 ms
0.62 ('gaussian', 100000, 'nlsq', 'gpu') new 27.8 ms old 45.0 ms
0.62 ('gaussian', 10000, 'nlsq', 'cpu') new 10.8 ms old 17.5 ms
0.74 ('two_exp', 1000, 'nlsq', 'gpu') new 27.1 ms old 36.6 ms
0.77 ('exp_decay', 10000000, 'nlsq_large', 'cpu') new 977.2 ms old 1.27 s
0.78 ('bounded_exp', 1000000, 'nlsq', 'cpu') new 1.28 s old 1.63 s
0.80 ('bounded_exp', 1000000, 'nlsq', 'gpu') new 538.6 ms old 676.2 ms
1.26 ('two_exp', 100000, 'nlsq', 'gpu') new 27.6 ms old 21.9 ms
1.30 ('exp_decay', 10000, 'nlsq', 'gpu') new 30.9 ms old 23.7 ms
1.40 ('exp_decay', 10000, 'nlsq', 'cpu') new 11.8 ms old 8.4 ms
1.87 ('two_exp', 1000000, 'nlsq', 'gpu') new 115.5 ms old 61.7 ms
1.91 ('gaussian', 1000000, 'nlsq', 'gpu') new 165.8 ms old 86.6 ms
1.94 ('exp_decay', 100000, 'nlsq', 'gpu') new 48.0 ms old 24.7 ms
2.79 ('exp_decay', 1000000, 'nlsq', 'gpu') new 140.5 ms old 50.4 ms

RATIO new/old for NON-NLSQ libs (drift control):
scipy cpu median 0.4 min 0.25 max 1.37 24
lmfit cpu median 0.48 min 0.2 max 1.94 24
jaxopt cpu median 0.54 min 0.17 max 0.91 18
jaxopt gpu median 0.97 min 0.71 max 1.27 18
optimistix cpu median 0.56 min 0.22 max 1.94 18
optimistix gpu median 1.0 min 0.44 max 1.14 15
