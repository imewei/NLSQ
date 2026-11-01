# Exact streaming solver, Stage 3 benchmark record (2026-11-01)

## Task 13: background prefetch of lazy-source blocks (gate: keep only if per-pass time >= 10% lower)

Study script (removed with the prefetch code, commit after `0d50e22`; recover from `0d50e22`):
`benchmarks/studies/exact_prefetch.py`. Source: HDF5 file, `x` and `y` float64, 100M points (1.6 GB),
`chunk_size = 1 << 22`, exp_decay data, GPU (RTX 4090 Laptop, jax 0.12.0.1), 5 timed passes per
setting, three runs, file resident in the OS page cache (about 29 GB cached, 16 GB available RAM;
i9-13900H, governor `powersave`).

| Run | Prefetch off (median pass) | Prefetch on (median pass) | Change |
|---|---|---|---|
| 1 | 0.615 s | 0.578 s | -6.1% |
| 2 | 0.622 s | 0.503 s | -19.1% |
| 3 | 0.634 s | 0.592 s | -6.7% |

Pooled median over the three runs: about -7%, below the 10% threshold, so prefetch was removed
(`_prefetch`, its tests and the study). Run 2 alone would pass; the spread between runs (6-19%) is the
laptop's drift, not a signal. A resident-in-page-cache HDF5 read costs about 0.06-0.1 s per pass against
about 0.5 s of GPU compute, which bounds the possible gain. The measurement does not cover a cold,
larger-than-RAM file; the 1B-point run (Task 16) reads from storage and is the place to notice an I/O
bound pass. Raw logs: the study printed per-pass times and machine state for each run.
