# Exact streaming solver, Stage 1 performance gate (2026-10-31)

Commit `3780631` (module `nlsq/streaming/exact_lm.py`, unbounded and unweighted).
Gate: module warm median <= 1.10 x prototype warm median, 100M-point exp_decay on the GPU,
matched resident-data preparation.

## Method

```bash
# GPU (isolated CUDA venv with jax[cuda13]==0.12.0.1; outside the sandbox)
PYTHONPATH=. python -m benchmarks.prototypes.exact_accumulated_lm --bench exp_decay 100000000 --impl prototype
PYTHONPATH=. python -m benchmarks.prototypes.exact_accumulated_lm --bench exp_decay 100000000 --impl module
# CPU (project .venv)
NLSQ_FORCE_CPU=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m benchmarks.prototypes.exact_accumulated_lm \
    --bench <problem> 100000000 --impl prototype|module
```

- Both implementations receive the same device-resident float64 `x`, `y`, the same `p0`,
  `chunk_size = 1 << 22`, tolerances 1e-8, 3 warm repeats, and report the median.
- The module runs with `check_finite=False` (the prototype has no finite-data pass), so these
  numbers time the solver alone. The default library path adds one fused finite check on pass 1.
- Prototype and module runs alternate within minutes of each other. This laptop's speed drifts
  by up to about 1.5x over an hour, so only same-session ratios are meaningful.

## Machine state

- i9-13900H (20 logical), 62 GB RAM (about 33-39 GB available during the runs), CPU governor
  `powersave`. RTX 4090 Laptop 16 GB, CUDA 13.4 driver, SM clock 1935 MHz and 69 C at the end
  of the second GPU series (3105 MHz maximum).
- Data resident on the device for GPU runs (excludes the one-off host-to-device copy).

## GPU result, exp_decay 100M (5 passes, identical cost 125003.6718261893)

| Series | Prototype warm | Module warm | Ratio |
|---|---|---|---|
| Before the probe fix, run 1 | 0.63 s | 0.75 s | 1.19 (fails) |
| Before the probe fix, run 2 | 0.62 s | 0.78 s | 1.26 (fails) |
| After, run 1 | 0.62 s | 0.55 s | 0.89 |
| After, run 2 | 0.61 s | 0.57 s | 0.93 |

The implementer's own three alternating pairs after the fix gave 0.90, 0.92 and 0.95.
**Gate passed** (module <= 1.10 x prototype in every post-fix pair).

Cause of the first failure: `check_model_contract` (once per fit) evaluated the model three
times on a 4M-point chunk, copied each result to the host and ran three `np.allclose` passes,
about 198 ms. It now compares on the device with one scalar sync (about 4.8 ms). The per-chunk
code (`jnp.where` masks, point count) was never the problem; per pass the module was about
10 ms faster than the prototype (80-83 vs 91-92 ms). The module cold time rose from about 0.85 s
to about 1.6 s on the GPU (op-by-op compile of the probe), which the gate does not cover.

## CPU result, 100M points (warm median, identical passes and cost)

| Problem | Prototype | Module | Ratio | Passes |
|---|---|---|---|---|
| exp_decay | 10.37 s | 10.67 s | 1.03 | 5 |
| gaussian | 30.03 s | 31.65 s | 1.05 | 15 |
| two_exp | 38.37 s | 36.14 s | 0.94 | 19 |

Differences under about 1.5x on CPU are inside this machine's noise (see the 2026-10-31 library
comparison caveats). Bounded problems arrive in Stage 2.
