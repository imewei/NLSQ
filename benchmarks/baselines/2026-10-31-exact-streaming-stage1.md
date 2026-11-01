# Exact streaming solver, Stage 1 performance gate (2026-10-31)

Commits `3780631` (first gate pass) and `7455b86` (final series, default finite check, 1-D weights
present in the module). Module `nlsq/streaming/exact_lm.py`.
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
- The module runs with its default `check_finite=True` (finite check fused into pass 1). The frozen
  prototype also checks finiteness, fused into every pass. An earlier series (below) ran the
  module with `check_finite=False` on the mistaken premise that the prototype has no such check;
  an independent review caught this and the final series replaces it.
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

Final series at `7455b86` (module default `check_finite=True`, includes the Task 8 weights code;
unweighted path), three alternating pairs, identical cost and 5 passes:

| Pair | Prototype warm | Module warm | Ratio |
|---|---|---|---|
| 1 | 0.57 s | 0.52 s | 0.91 |
| 2 | 0.57 s | 0.54 s | 0.95 |
| 3 | 0.58 s | 0.53 s | 0.91 |

GPU SM clock 1935 MHz, 65 C, 40 GB RAM available. **Gate passed** (module <= 1.10 x prototype in
every post-fix pair, with the default finite check).

Cause of the first failure: `check_model_contract` (once per fit) evaluated the model three
times on a 4M-point chunk, copied each result to the host and ran three `np.allclose` passes,
about 198 ms. It now compares on the device with one scalar sync (about 4.8 ms). The per-chunk
code (`jnp.where` masks, point count) was never the problem; per pass the module was about
10 ms faster than the prototype (80-83 vs 91-92 ms). The module cold time rose from about 0.85 s
to about 1.6 s on the GPU (op-by-op compile of the probe), which the gate does not cover.

## CPU result, 100M points (warm median, identical passes and cost; module default finite check)

Measured at `7455b86` while two review agents were running on the same machine, so treat as
indicative. The gate itself is the GPU measurement above.

| Problem | Prototype | Module | Ratio | Passes |
|---|---|---|---|---|
| exp_decay | 7.78 s | 8.67 s | 1.11 | 5 |
| gaussian | 25.00 s | 26.88 s | 1.08 | 15 |
| two_exp | 34.73 s | 34.79 s | 1.00 | 19 |

An earlier CPU series with `check_finite=False` gave 1.03 / 1.05 / 0.94 (10.37 / 30.03 / 38.37 s
prototype). The same-problem prototype time moved by up to 1.35x between the two series, which is
this laptop's drift (see the 2026-10-31 library comparison caveats); differences under about 1.5x
on CPU are inside that noise. Bounded problems arrive in Stage 2.
