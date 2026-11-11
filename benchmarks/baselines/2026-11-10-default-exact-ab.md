# Default exact streaming in `curve_fit_large`: A/B gate result (2026-11-10)

Change: `curve_fit_large` runs the exact full-data streaming LM (`Auto.exact`) by default at or above `size_threshold`,
with the adaptive normal/TSQR covariance (Tasks 5-7 of the "close the JAXopt gap" plan).
Base: `a8fac156` (branch point, worktree). Candidate: `ec9dd429` (branch `perf/close-jaxopt-gap`, which adds only the
`--cost-gate` flag on top of the code change `3c91b009`).
Raw A/B JSON: `benchmarks/results/2026-11-10-default-exact-ab/` (gitignored).

Machine: i9-13900H, RTX 4090 Laptop 16 GB, project `.venv` (jax/jaxlib 0.12.0.1 CUDA), CPU governor `powersave`,
CPU runs use `JAX_PLATFORMS=cpu`. Laptop clock drift between sessions is up to ~50%, so only the interleaved ratios
below mean anything. Runs were back to back, one at a time (never two on the GPU).

**Verdict: Gate 1 (speed) FAILED at one cell, GPU two_exp 10M: the 5 x 3 run gave ratio 1.018 (inside the spread),
and a 10 x 5 re-measurement (R16, below) gave 1.162, a real ~16% slowdown. Every other cell passes; Gate 2 (no
slowdown beyond the spread) passes for the main table; Gate 3 (route) passes.**

## Command

```bash
# per stage (absolute paths; --cost-gate no-worse; base = worktree at a8fac156, cand = this repo)
.venv/bin/python -m benchmarks.studies.ab_nlsq --base <ab-base worktree> --cand /home/wei/Documents/GitHub/NLSQ \
  --cost-gate no-worse --libraries nlsq_large --problems exp_decay gaussian two_exp bounded_exp \
  --out-root <abs results dir>/<stage> --platform {gpu,cpu} --sizes <N> [--repeats 3] [--rounds 3 --repeats 2]
```

| stage | sizes | rounds x repeats |
|---|---|---|
| GPU / CPU | 1M | 5 x 7 |
| GPU / CPU | 10M | 5 x 3 |
| GPU / CPU | 100M | 3 x 2 |

`ratio = median(cand) / median(base)` over the per-round warm medians; `spr` is the per-arm
`(max - min) / median` over rounds.

## GPU

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay 1M | 0.2023 | 0.1435 | 0.709 | 0.345 | 0.212 |
| gaussian 1M | 0.2187 | 0.2020 | 0.924 | 0.284 | 0.200 |
| two_exp 1M | skipped (base missed the optimum) | | | | |
| bounded_exp 1M | 1.7337 | 0.1789 | 0.103 | 0.152 | 0.186 |
| exp_decay 10M | 0.9393 | 0.1874 | 0.199 | 0.407 | 0.509 |
| gaussian 10M | 0.9026 | 0.3512 | 0.389 | 0.504 | 0.432 |
| two_exp 10M | 0.9977 | 1.0161 | **1.018** | 0.761 | 0.394 |
| bounded_exp 10M | 2.3311 | 0.3744 | 0.161 | 0.793 | 0.210 |
| exp_decay 100M | 9.6390 | 0.7163 | 0.074 | 0.525 | 0.623 |
| gaussian 100M | 10.5634 | 1.7445 | 0.165 | 0.491 | 0.254 |
| two_exp 100M | 10.2607 | 3.0106 | 0.293 | 0.205 | 0.438 |
| bounded_exp 100M | 26.2533 | 1.7997 | 0.069 | 0.195 | 0.198 |

## CPU

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay 1M | 0.3612 | 0.2278 | 0.631 | 0.187 | 0.827 |
| gaussian 1M | 0.4382 | 0.4779 | 1.091 | 0.267 | 0.972 |
| two_exp 1M | skipped (base missed the optimum) | | | | |
| bounded_exp 1M | 1.6847 | 0.5094 | 0.302 | 0.382 | 1.047 |
| exp_decay 10M | 3.9519 | 1.4591 | 0.369 | 0.075 | 1.005 |
| gaussian 10M | 4.8670 | 4.8233 | 0.991 | 0.059 | 0.199 |
| two_exp 10M | 10.6486 | 5.7760 | 0.542 | 0.422 | 0.067 |
| bounded_exp 10M | 7.7672 | 5.2811 | 0.680 | 0.299 | 0.130 |
| exp_decay 100M | 25.4046 | 6.1652 | 0.243 | 0.553 | 0.079 |
| gaussian 100M | 36.5939 | 25.6820 | 0.702 | 0.831 | 0.443 |
| two_exp 100M | 65.3021 | 34.0322 | 0.521 | 0.772 | 0.624 |
| bounded_exp 100M | 45.0796 | 27.8561 | 0.618 | 0.036 | 0.148 |

## Gates

1. **Every candidate median at or below base on GPU at 10M and 100M: FAILED at one cell.**
   GPU two_exp 10M: base 0.9977 s, cand 1.0161 s, ratio 1.018 in the 5 x 3 run. The ratio was inside both arms'
   spread (0.761 and 0.394), but the 10 x 5 re-measurement above gives 1.162 with spreads 0.038 / 0.016: a real slowdown. The other seven GPU
   10M/100M cells are 0.07-0.39x. For the same problem, base and cand costs at 10M differ by 7.2e-7 (base higher),
   which is inside `COST_RTOL` (1e-6), so the case was timed rather than skipped.
2. **No other cell slower beyond the spread: PASS.** The only cell with ratio above 1 besides the one above is CPU
   gaussian 1M (1.091, spread 0.267 / 0.972). No cell has `ratio > 1 + max(spread)`.
3. **Task 5 Step 4b route table: PASS, all four problems `"normal"` (so no TSQR pass is paid).** Measured with
   `benchmarks/studies/exact_cov_route.py` at 10M on GPU (Task 5): exp_decay kappa 6.33, gaussian 6.45,
   two_exp 761.7 (narrow margin below `TSQR_KAPPA` = 1e3), bounded_exp 6.65. The TSQR pass would cost
   0.098 s at 10M and 0.768 s at 100M (exp_decay data, after the Jacobian kernel was cached per
   `(model, weighted)`; about 8x and 13x one LM data pass; see `2026-11-10-qr-covariance-ab.md`). The first
   measurement, 0.153 s / 0.823 s (about 14x), predates that fix.

## Gate 1 re-measurement: GPU two_exp 10M (R16)

The 5 x 3 cell was inside its spread, so it was re-measured with 10 rounds x 5 repeats (raw JSON in
`benchmarks/results/2026-11-10-default-exact-ab/gpu-10M-two_exp-r14`):

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| two_exp 10M (10 x 5) | 0.856 | 0.995 | **1.162** | 0.038 | 0.016 |

Median final cost: base 12487.60118, cand 12487.59225 (cand lower by 7.1e-7 relative). **Gate 1 FAILED at this
cell.** The ratio exceeds both spreads, so it is a real ~16% slowdown, while exact reaches a lower cost. The opt-out
is `strategy=Chunked()`.

## Cost gate (`--cost-gate no-worse`)

Median final cost, base vs cand (relative difference = (base - cand) / cand; positive = base costlier):

| case | rel. diff | outcome |
|---|---|---|
| two_exp 1M (GPU and CPU) | +7.34e-5 | base missed the optimum (the GLS DNRO): skipped, untimed |
| two_exp 10M | +7.15e-7 | within rtol: timed |
| two_exp 100M | +4.12e-7 | within rtol: timed |
| exp_decay, gaussian (all sizes) | -3.6e-13 .. -9e-11 | equal: timed |
| bounded_exp (all sizes) | -3.6e-9 .. -4.7e-9 | cand costlier by 4e-9 (< rtol 1e-6; the active-set exact path vs Coleman-Li TRF): timed |

Every case was status `ok` in both arms in every round. No violation.

## Notes

- Spreads are large (up to 1.0) at 10M-100M on this laptop; speedups of 0.07-0.4x are far outside them, the 0.9-1.1
  ratios are not distinguishable from noise.
- Base arm timing on CPU at 100M (25-65 s per fit) is of the same order as the 2026-11-09 default
  `curve_fit_large` CPU figures (23-45 s), within the machine's drift. Not investigated: why bounded_exp gains most on GPU
  (0.07-0.16).
