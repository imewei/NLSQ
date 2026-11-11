# Default exact streaming in `curve_fit_large`: A/B gate result (2026-11-10)

Change: `curve_fit_large` runs the exact full-data streaming LM (`Auto.exact`) by default at or above `size_threshold`,
with the adaptive normal/TSQR covariance (Tasks 5-7 of the "close the JAXopt gap" plan).
Base: `a8fac156` (branch point, worktree). Candidate: `4dcc65b6` (branch `perf/close-jaxopt-gap`, with the exact
chunk kernel reused across calls). The first run measured `ec9dd429`, whose exact path recompiled its chunk kernel on
every call (see "Superseded: R16" below); every GPU cell and the CPU 1M/10M cells were re-measured on `4dcc65b6`.
Raw A/B JSON: `benchmarks/results/2026-11-10-default-exact-ab-v2/` (re-run) and `.../2026-11-10-default-exact-ab/`
(first run), both gitignored. CPU 100M was re-measured on `501fbe2f` (adds only a 64 MiB captured-constant cap
to the kernel cache).

Machine: i9-13900H, RTX 4090 Laptop 16 GB, project `.venv` (jax/jaxlib 0.12.0.1 CUDA), CPU governor `powersave`,
CPU runs use `JAX_PLATFORMS=cpu`. Laptop clock drift between sessions is up to ~50%, so only the interleaved ratios
below mean anything. Runs were back to back, one at a time (never two on the GPU).

**Verdict: all gates PASS.** Gate 1: every GPU 10M/100M candidate is at or below base (ratio 0.06-0.33). Gate 2: no
cell has a ratio above 1. Gate 3 (route) passes. The earlier gate-1 failure at GPU two_exp 10M (ratio 1.16) was the
per-call kernel recompile, not the solver; it is fixed in `4dcc65b6` and that cell now measures 0.33.

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

Re-measured on `4dcc65b6`.

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay 1M | 0.2131 | 0.0214 | 0.100 | 0.107 | 0.064 |
| gaussian 1M | 0.2233 | 0.0513 | 0.230 | 0.091 | 0.175 |
| two_exp 1M | skipped (base missed the optimum) | | | | |
| bounded_exp 1M | 1.7456 | 0.0569 | 0.033 | 0.059 | 0.109 |
| exp_decay 10M | 1.1917 | 0.0804 | 0.067 | 0.370 | 0.105 |
| gaussian 10M | 1.2314 | 0.2237 | 0.182 | 0.398 | 0.074 |
| two_exp 10M | 1.2839 | 0.2979 | 0.232 | 0.507 | 0.078 |
| two_exp 10M (10 x 5) | 0.8722 | 0.2887 | 0.331 | 0.411 | 0.055 |
| bounded_exp 10M | 2.5053 | 0.2368 | 0.095 | 0.329 | 0.104 |
| exp_decay 100M | 8.2388 | 0.5178 | 0.063 | 0.011 | 0.007 |
| gaussian 100M | 8.2626 | 1.5544 | 0.188 | 0.024 | 0.005 |
| two_exp 100M | 8.7589 | 2.1225 | 0.242 | 0.013 | 0.013 |
| bounded_exp 100M | 19.0566 | 1.6420 | 0.086 | 0.026 | 0.014 |

## CPU

1M and 10M re-measured on `4dcc65b6`, 100M on `501fbe2f`.

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay 1M | 0.3466 | 0.1219 | 0.352 | 0.507 | 0.345 |
| gaussian 1M | 0.4106 | 0.3564 | 0.868 | 0.226 | 0.306 |
| two_exp 1M | skipped (base missed the optimum) | | | | |
| bounded_exp 1M | 1.5462 | 0.3959 | 0.256 | 0.255 | 0.302 |
| exp_decay 10M | 3.9042 | 1.2669 | 0.325 | 1.152 | 1.059 |
| gaussian 10M | 4.9263 | 4.4632 | 0.906 | 1.180 | 0.335 |
| two_exp 10M | 10.1171 | 6.2740 | 0.620 | 0.699 | 0.329 |
| bounded_exp 10M | 7.3088 | 6.4729 | 0.886 | 0.646 | 0.804 |
| exp_decay 100M | 22.2989 | 5.8168 | 0.261 | 0.521 | 0.048 |
| gaussian 100M | 23.4143 | 22.2778 | 0.951 | 0.494 | 0.120 |
| two_exp 100M | 51.8469 | 26.7447 | 0.516 | 0.155 | 0.129 |
| bounded_exp 100M | 39.9230 | 25.4554 | 0.638 | 0.230 | 0.669 |

## Gates

1. **Every candidate median at or below base on GPU at 10M and 100M: PASS.** Ratios 0.06-0.33 (two_exp 10M 0.232 in
   the 5 x 3 run, 0.331 in the 10 x 5 run). For two_exp, base and cand costs at 10M differ by 7.2e-7 (base higher),
   inside `COST_RTOL` (1e-6), so the case was timed rather than skipped.
2. **No cell slower beyond the spread: PASS.** No re-measured cell has a ratio above 1.
3. **Task 5 Step 4b route table: PASS, all four problems `"normal"` (so no TSQR pass is paid).** Measured with
   `benchmarks/studies/exact_cov_route.py` at 10M on GPU (Task 5): exp_decay kappa 6.33, gaussian 6.45,
   two_exp 761.7 (narrow margin below `TSQR_KAPPA` = 1e3), bounded_exp 6.65. The TSQR pass would cost
   0.098 s at 10M and 0.768 s at 100M (exp_decay data, after the Jacobian kernel was cached per
   `(model, weighted)`; about 8x and 13x one LM data pass; see `2026-11-10-qr-covariance-ab.md`). The first
   measurement, 0.153 s / 0.823 s (about 14x), predates that fix.

## Superseded: R16 (GPU two_exp 10M, ratio 1.16)

The first run's 10 x 5 re-measurement gave base 0.856 s, cand 0.995 s (ratio 1.162), recorded as a real slowdown. It
was not the solver. `exact_lm_fit` built a new `jax.jit` chunk kernel per call, so every warm call recompiled it, and
the harness disables the persistent JAX cache (`NLSQ_DISABLE_PERSISTENT_CACHE=1`), so the full compile was paid each
time: 0.74 s for the 4-parameter two_exp, against 0.27 s for its 19 data passes (0.10-0.13 s for the 3-parameter
problems, which is why only two_exp crossed 1). `4dcc65b6` keys the kernel by the exact jaxpr digest curve_fit already
uses, so warm calls compile nothing; the cell now measures 0.289 s vs 0.872 s (ratio 0.331). Final costs are unchanged
(cand 12487.59225).

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
  (0.03-0.10 after the re-run).
