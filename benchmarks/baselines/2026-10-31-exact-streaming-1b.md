# Exact streaming solver, 1B-point run (2026-11-01)

`curve_fit_large(method="exact_streaming")` / `exact_lm_fit` on exp_decay (3 parameters, true
`[2.5, 1.3, 0.5]`) with 1,000,000,000 points read from HDF5 files (16 GB of `x` and `y` each,
chunked storage, per-chunk `SeedSequence` seeds), one noiseless and one noisy copy (sigma 0.05).
Script: `benchmarks/studies/exact_1b_run.py`; raw results: `2026-10-31-exact-streaming-1b.json`
(and `-1e8-rehearsal.json` for the same command at 100M points, run first).

## Command

```bash
# CUDA venv (jax[cuda13]==0.12.0.1), outside the sandbox, from the repo root
PYTHONPATH=. python -m benchmarks.studies.exact_1b_run run --data-dir <dir> --out <json>            # n = 1e9
PYTHONPATH=. python -m benchmarks.studies.exact_1b_run run --data-dir <dir> --n 100000000 --out <json>  # rehearsal
```

## Machine state

ThinkPad P1: i9-13900H (20 logical), 62 GB RAM (48 GB available, 31 GB page cache at start),
governor `powersave`, RTX 4090 Laptop 16 GB (210 MHz idle, 54 C at start), CUDA 13.4 driver, jax 0.12.0.1,
numpy 2.5.4, Python 3.13.7. The HDF5 data is read on the host and moved to the device chunk by chunk
(a 16 GB file cannot be device-resident). Two other agents were reading the repository (not running
GPU work) during the run.

## Gates (spec "Gates for the 1B run")

| Gate | Result | Detail |
|---|---|---|
| 1 Noiseless recovery (`rtol 1e-6`, cost `< 1e-6·n`) | PASS | max relative error 6.7e-16, cost 2.3e-23 (limit 1e3), 7 passes |
| 2 Noisy within `5·√diag(pcov)` | PASS | `|z|` = 0.77, 0.99, 1.27 |
| 3(i) Point count equals `n` on every pass | PASS | noiseless 7, sentinel 5, noisy 5 passes (counted by wrapping `run_pass`; `y` rows read equal `n × passes`) |
| 3(ii) Streaming NumPy cost equals solver cost (`rtol 1e-9`) | PASS (noisy fit) | relative difference 1.9e-16 (cost 1,249,983.77). The noiseless fit is not gated: its cost is rounding noise (2.27e-23 host against 2.28e-23 device) |
| 3(iii) Sentinel block (last 1,000 points, offset `Δ = 157.3`, predicted relative displacement 1e-3) | PASS | actual / predicted displacement 0.9999, projection 1.0003 |
| 4 Peak host RSS `< data size + 4 GB` | PASS | 1.84 GiB peak (limit 18.9 GiB) |
| 5 Reporting | not gated | below |

## Timing (not gated)

| Fit | Passes | Wall | Median pass | Throughput |
|---|---|---|---|---|
| noiseless | 7 | 37.7 s | 4.22 s | 2.4e8 points/s |
| sentinel | 5 | 21.7 s | 4.08 s | 2.5e8 points/s |
| noisy | 5 | 49.0 s | 9.50 s | 1.1e8 points/s |

Total 153 s with generation excluded (noiseless file written in 21 s, noisy in 36 s). The noisy pass
is slower because its file had been evicted from the page cache by the time it was read (31 GB cache
against 32 GB of files); the noiseless and sentinel fits ran with the file freshly written and cached.
The 100M rehearsal (1.6 GB files, fully cached) had a median pass of 0.47 s (2.1e8 points/s), so a
cached pass is about 10 times the 100M pass, and a cold-storage pass about twice that. At 1B the
pass is bounded by the HDF5 read plus host-to-device copy, not by device compute (100M resident on the
GPU takes about 0.55 s per pass), consistent with the prefetch study's reading in
`2026-11-01-exact-streaming-stage3.md`.

## Not covered

One seed, one model (exp_decay), one HDF5 chunking; the rehearsal and the 1B run share the generator.
Zarr and memmap sources are covered by the small-scale tests, not at 1B.
