# QR-then-SVD covariance kernel: gate result (2026-11-10)

Change: `minpack._covariance_svd_body` takes the SVD of the `R` factor of `qr(J, mode="r")`
instead of the SVD of `J` (commit 73764629). Base for the A/B: `a8fac156` (worktree). Candidate: `73764629`.
Script: `benchmarks/studies/bench_cov_kernel.py` (`--crossover`, `--accuracy`; default = kernel).
Raw A/B JSON: `benchmarks/results/2026-11-10-ab-qr-cov/` (gitignored).

**Verdict: KEPT. Kernel, GPU A/B, CPU A/B and accuracy gates pass. Covariance-only limit `HOST_COV_SVD_MAX_ELEMENTS = 150,000` applied; `HOST_SVD_MAX_ELEMENTS` (TRF) left at 30,000 (see Crossover).**

## Kernel microbenchmark (go/no-go)

RTX 4090 Laptop, float64, jit, 3 warmups, 20 interleaved rounds, `block_until_ready`. Gate: new median < old median - old IQR.

| shape | old median (IQR) | new median (IQR) | new/old | gate |
|---|---|---|---|---|
| 1,000,000 x 3 | 12.816 ms (0.122) | 8.171 ms (0.131) | 0.638 | PASS |
| 1,000,000 x 6 | 26.210 ms (0.098) | 16.196 ms (0.079) | 0.618 | PASS |

## End-to-end A/B (interleaved base/cand, `ratio = median(cand)/median(base)`, spread = per-arm spread)

Gate: no ratio above the spread (ratio < 1 + max(spread)). All 16 cases are below 1.

GPU, `curve_fit`, 1M points, 5 rounds x 7 repeats:

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay | 0.0590 | 0.0510 | 0.865 | 0.103 | 0.115 |
| gaussian | 0.0821 | 0.0764 | 0.931 | 0.160 | 0.153 |
| two_exp | 0.0736 | 0.0654 | 0.889 | 0.133 | 0.035 |
| bounded_exp | 0.6476 | 0.6401 | 0.988 | 0.135 | 0.116 |

GPU, `curve_fit_large`, 10M points, 5 rounds x 3 repeats:

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay | 1.2158 | 1.1623 | 0.956 | 0.069 | 0.034 |
| gaussian | 1.2455 | 1.2063 | 0.969 | 0.072 | 0.113 |
| two_exp | 1.3136 | 1.2525 | 0.954 | 0.053 | 0.065 |
| bounded_exp | 2.7585 | 2.5014 | 0.907 | 0.086 | 0.105 |

CPU (`JAX_PLATFORMS=cpu`), `curve_fit`, 5 rounds x 7 repeats:

| case | base_s | cand_s | ratio | spr_b | spr_c |
|---|---|---|---|---|---|
| exp_decay/100k | 0.0565 | 0.0493 | 0.873 | 0.121 | 0.101 |
| exp_decay/1M | 0.6295 | 0.5386 | 0.856 | 0.065 | 0.074 |
| gaussian/100k | 0.1202 | 0.1139 | 0.947 | 0.080 | 0.076 |
| gaussian/1M | 1.4608 | 1.3969 | 0.956 | 0.048 | 0.041 |
| two_exp/100k | 0.1129 | 0.1059 | 0.938 | 0.054 | 0.052 |
| two_exp/1M | 1.7485 | 1.6845 | 0.963 | 0.113 | 0.068 |
| bounded_exp/100k | 0.3537 | 0.3448 | 0.975 | 0.076 | 0.105 |
| bounded_exp/1M | 1.8075 | 1.7648 | 0.976 | 0.103 | 0.066 |

The 10M bounded_exp saving (ratio 0.907, about 0.26 s) is larger than the other 10M cases (0.954-0.969) and is
unexplained.

Most ratios are within the spread, so these are "no regression" results, not speedup claims; the ratios lie
below 1 in every case, consistent with the kernel result.
The A/B harness's own correctness gate (same case set, `ok`, cost agreement to rtol 1e-6) passed in all rounds.

## Host-SVD crossover (`_covariance_svd_jit`, new body; host = `device_put` to CPU + SVD, 15 rounds, medians)

| n | p | n*p | GPU ms | host ms (incl. transfer) | winner |
|---|---|---|---|---|---|
| 2,000 | 3 | 6,000 | 1.467 | 0.165 | host |
| 5,000 | 3 | 15,000 | 1.613 | 0.187 | host |
| 10,000 | 3 | 30,000 | 1.489 | 0.207 | host |
| 20,000 | 3 | 60,000 | 1.610 | 0.301 | host |
| 50,000 | 3 | 150,000 | 1.927 | 0.493 | host |
| 2,000 | 6 | 12,000 | 1.806 | 0.183 | host |
| 5,000 | 6 | 30,000 | 1.977 | 0.236 | host |
| 10,000 | 6 | 60,000 | 2.385 | 0.333 | host |
| 20,000 | 6 | 120,000 | 2.252 | 0.434 | host |
| 50,000 | 6 | 300,000 | 2.990 | 2.801 | host (near tie) |

The host path wins at every size in both sweeps, so the sweep does not bracket the crossover (p = 3 is
right-censored at 150,000 elements, p = 6 is a near tie at 300,000). By the plan's rule (largest n*p where the
host wins in every sweep) the figure is **150,000**, 5x the current 30,000.

**Applied as a covariance-only limit.** `HOST_SVD_MAX_ELEMENTS` is shared with the per-iteration TRF full SVD
(`nlsq/core/trf_jit.py:409`), which still uses a full SVD and was measured at 30,000
(`2026-11-06-perf-phase0.md` item (c)), so the shared constant is unchanged. A separate
`HOST_COV_SVD_MAX_ELEMENTS = 150_000` in `nlsq/core/device_policy.py` is used only by `minpack._covariance_svd`.
The 150,000 figure is right-censored at p = 3 (the host still won at the largest n*p measured); extending the
sweep past 50,000 rows could raise it further.

## Accuracy (`pcov_from_jacobian`, old body = SVD of J, new = `_covariance_svd_body`)

Jacobians from `jax.jacfwd` on the `benchmarks/compare` problems at the true parameters, 1,000,000 points,
float64, cost = 1.0, `absolute_sigma=False`. Metric: `max_ij |new-old| / sqrt(old_ii old_jj)`.

| problem | cond(J) | max rel diff | tol | gate |
|---|---|---|---|---|
| exp_decay | 6.3e0 | 9.29e-16 | 1e-12 | PASS |
| gaussian | 6.4e0 | 4.03e-16 | 1e-12 | PASS |
| two_exp | 7.6e2 | 9.85e-15 | 1e-12 | PASS |
| bounded_exp | 6.3e0 | 9.29e-16 | 1e-12 | PASS |
| two_exp, rates 1.0 / 1.05 | 5.0e5 | 3.79e-14 | 1e-9 | PASS |
| two_exp, rates 1.0 / 1.0001 | 6.0e13 | 2.67e-12 | 1e-9 | PASS |

The bounded_exp row equals exp_decay by construction: same model and Jacobian at the true parameters, bounds
inactive.

Rates 1.0 / 1.0001 give cond(J) = 6e13, not the plan's "around 1e6". The rate pair 1.0 / 1.05 gives about 5e5,
so both are recorded. At 6e13 `J` is past the rank cutoff (`eps * n * s[0]`, about cond 5e9), so both
bodies return a truncated pseudo-inverse with a rank-deficient warning and still agree to 2.7e-12.

## Machine state

- Host RAM: 62 GB total, 8 GB free, 45 GB cache (`free -g`).
- CPU governor: `powersave` (`cpupower frequency-info -p`; `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor`).
- GPU: NVIDIA GeForce RTX 4090 Laptop, driver 620.70.06, idle clocks at the time of query
  (graphics/SM 210 MHz, memory 405 MHz; boost clocks not pinned), 59.6 C, 11 W idle.
- JAX 0.9.x CUDA in the project `.venv`; GPU runs were outside the sandbox and never concurrent.

## Task 5: TSQR covariance on the exact default (2026-11-10)

`exact_lm_fit` computes `pcov` from `H` when `kappa = normal_condition(H) <= TSQR_KAPPA`
(`covariance_method="normal"`). Above it, one extra data pass at `popt` accumulates the TSQR factor
`R <- qr(vstack(R, J_chunk), mode="r")`, and `pcov = pcov_from_jacobian(R, ..., _covariance_svd)`
(`"tsqr"`).

### Gate G-cov (`tests/streaming/test_exact_cov_accuracy.py`, CPU, float64)

Each case compares the fit's `pcov` with `pcov_from_jacobian(full_jac(popt), ..., _covariance_svd)` at the
same `popt`. Metric: `max_ij |dC_ij| / sqrt(C_ii C_jj)`, gate 1e-8, and the same rank decision. That decision is the count of
directions the route actually retains: singular values of the TSQR R above `eps*max(n,p)*s[0]`, or eigenvalues
of the accumulated H above `p*eps*lambda_max`. It must equal both the full-J reference's count and the rank the
case was constructed with (fix round 1; it replaced a correlation-matrix proxy). `chunk_size=700` gives a padded tail on every case except column scales, which uses
`chunk_size=128`.

Controlled cases use `B = U diag(geomspace(kappa, 1, 4)) V^T` with n = 2000. `svd(B)` reproduces `s` to 1e-12.

| case | kappa (measured) | route | error |
|---|---|---|---|
| controlled kappa=1 | 1 | normal | 4.0e-16 |
| controlled kappa=10 | 10 | normal | 5.7e-15 |
| controlled kappa=100 | 100 | normal | 9.1e-13 |
| controlled kappa=500 | 500 | normal | 2.1e-11 |
| controlled kappa=900 | 900 | normal | 1.2e-10 |
| controlled kappa=1100 | 1100 | tsqr | 2.7e-14 |
| controlled kappa=2000 | 2000 | tsqr | 5.4e-14 |
| two_exp (harness model, n=20k, noise 0.05) | 666 | normal | 4.2e-11 |
| gaussian (harness model, n=20k) | 6.4 | normal | 6.3e-16 |
| correlated: quadratic on t in [1, 2] | 158 | normal | 6.4e-12 |
| column scales 1e-6..1e6 (n=300) | 7.5e11 | tsqr | 4.0e-9 |
| sigma, kappa=100, absolute_sigma False / True | 100.3 | normal | 1.0e-12 / 1.0e-12 |
| sigma, kappa=2000, absolute_sigma False / True | 2006 | tsqr | 8.1e-14 / 8.1e-14 |
| active upper bound on theta_0 (kappa=100) | 100 | normal | 9.1e-13 |
| rank-deficient, s = (10, 5, 1, 0) | 1.7e8 (H) | tsqr | 1.9e-14, rank 3 on both |

The worst normal-route error is 1.2e-10, at kappa=900. That is below `eps * kappa^2` = 1.8e-10, and below
the gate by 87x. No case fails, so `TSQR_KAPPA` stays at **1e3**. The column-scale case (cond 7.5e11) reaches
4.0e-9. That error is the SVD floor, about `eps * cond(J)`, which the reference shares. A graded `J` with
cond much above 1e12 would also miss 1e-8 against its reference, on either route.

### Route at 10M and the TSQR pass cost (GPU)

`benchmarks/studies/exact_cov_route.py` (`uv run python -m benchmarks.studies.exact_cov_route`) was run on an
RTX 4090 Laptop GPU with JAX CUDA, outside the sandbox. It uses the harness data with `seed=0` and default
`exact_lm_fit` options. kappa is `normal_condition(H)` at `popt`.

| problem | covariance_method | kappa | nfev | fit wall (s) |
|---|---|---|---|---|
| exp_decay | normal | 6.33 | 5 | 1.13 (incl. compile) |
| gaussian | normal | 6.45 | 15 | 0.65 |
| two_exp | normal | 761.7 | 19 | 0.41 |
| bounded_exp | normal | 6.65 | 22 | 0.35 |

All four take the `"normal"` route at 10M. two_exp sits at 762, below 1e3, matching the spec's 762 at the
true parameters.

TSQR pass (`tsqr_r_factor`) against one LM data pass (`run_pass`), exp_decay at `p_true`. Data is on the
device, chunk is 4,194,304, both are warmed up, and the times are wall clock to a NumPy result over 3 repeats.

| n | tsqr min / median (s) | data pass min / median (s) | ratio |
|---|---|---|---|
| 10M | 0.098 / 0.098 | 0.012 / 0.013 | ~8x |
| 100M | 0.765 / 0.768 | 0.057 / 0.063 | ~13x |

These times are from fix round 1, after the Jacobian kernel was cached per `(model, weighted)`. Before that,
every call re-traced the kernel, and the same measurement gave 0.151 s at 10M and 0.821 s at 100M.

The TSQR pass costs more than one LM iteration, which is one data pass: about 8-13x one pass. The marginal
slope is 7.4 ms per 1M rows ((0.765 - 0.098) s / 90M rows). The spec estimated about 15 ms per 1M rows (about 1 s at 100M); the measurement is under that. The
cost is reported, as the spec asks, and it does not block: the pass runs only when kappa > 1e3, and none of
the four benchmark problems takes that route.
