# Exact streaming solver, Stage 2 benchmark record (2026-11-01)

Commit `bf95bb1`. Stage 2 adds 1-D weights and box bounds (active set with a constrained
model-decrease certificate) to `nlsq/streaming/exact_lm.py`. Gate (plan Task 10 Step 1): at 100M
points, CPU `bounded_exp` no slower than `curve_fit_large` by more than the prototype's 7%;
otherwise pass only if the GPU result holds.

## Method

```bash
# GPU (isolated CUDA venv, jax[cuda13]==0.12.0.1; outside the sandbox); prototype, module, large
PYTHONPATH=. python -m benchmarks.prototypes.exact_accumulated_lm --bench bounded_exp 100000000 --impl prototype
PYTHONPATH=. python -m benchmarks.prototypes.exact_accumulated_lm --bench bounded_exp 100000000 --impl module
PYTHONPATH=. python benchmarks/results/2026-10-31/nlsq_large.py chunked bounded_exp 100000000 3
# CPU: same three commands with NLSQ_FORCE_CPU=1 JAX_PLATFORMS=cpu and the project .venv python
```

- `--bench` runs use device-resident float64 data, `chunk_size = 1 << 22`, tolerances 1e-8, 3 warm
  repeats, median. `curve_fit_large` (default chunked path, per-chunk fits combined by GLS)
  runs through `benchmarks/results/2026-10-31/nlsq_large.py` (3 warm repeats, includes the host
  data path). Two alternating series per backend.
- The `--bench` module path uses the library default `check_finite=True`.
- Sequential runs only; GPU and CPU series did not overlap.

## Machine state

- i9-13900H (20 logical), 62 GB RAM (45-46 GB available), CPU governor `powersave`.
  RTX 4090 Laptop 16 GB, CUDA 13.4 driver; GPU idle at 210 MHz and 51 C before the series
  (1935 MHz under load in the preceding gate runs).

## Results, `bounded_exp`, 100M points (warm median)

| Backend | Prototype | Module | `curve_fit_large` | Module vs prototype | Module vs `curve_fit_large` |
|---|---|---|---|---|---|
| GPU, series 1 | 2.13 s | 1.90 s | 17.19 s | 0.89 | 9.0x faster |
| GPU, series 2 | 2.21 s | 1.96 s | 16.41 s | 0.89 | 8.4x faster |
| CPU, series 1 | 33.82 s | 35.18 s | 45.56 s | 1.04 | 1.29x faster |
| CPU, series 2 | 33.70 s | 32.74 s | 45.38 s | 0.97 | 1.39x faster |

- Passes: 22 for both the prototype and the module in every run. Final cost: prototype and module
  agree exactly (773279.8162408264); `curve_fit_large` reached 773279.8126 (4.7e-9 relative
  lower, inside the harness rtol 1e-6 gate; per-chunk GLS is approximate, see the 2026-10-31
  library comparison).
- **Stop condition passed.** On CPU the module takes 23-28% less time than `curve_fit_large`
  (1.29-1.39x faster), so it is not slower, let alone more than 7% slower. The earlier prototype
  record (48.8 s vs 45.4 s, 7% slower on CPU) was taken in a slower machine state; in this
  session the prototype itself runs in 33.7-33.8 s with the same 22 passes. The GPU result also
  holds (about 8-9x faster than `curve_fit_large`, 0.89x of the prototype time). Machine drift of
  up to about 1.5x between sessions is documented in the 2026-10-31 library comparison, so
  absolute times are not comparable across sessions; the same-session ratios are the result.

## Correctness evidence (CPU CI, not timing)

- `assert_kkt` (free |scaled gradient| <= 1e-6, lower/upper sign rules, feasibility exact) passes
  for the SciPy-TRF parity problem and 500 seeded random bounded fits at tolerance 1e-12 with 0
  failures; configurations ending with 0, 1, 2 and 3 parameters on a bound all occur.
- The rate parameter of the randomized generator is clamped to b >= 0.1: negative rates admit
  other valid local minima at 12-19x SciPy's cost (the solver still reaches a KKT point there;
  verified for 84 configurations without the clamp).
- Active set, not Coleman-Li: the iterate path differs from SciPy `trf`, the optimum (a KKT
  point of the box-constrained problem) is the same up to the local-minimum caveat.
