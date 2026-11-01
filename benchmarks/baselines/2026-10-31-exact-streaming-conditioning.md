# Exact streaming solver, conditioning study and decision (2026-11-01)

Decision: **`normal_equations_stay`.** Normal equations remain the accumulation; the `κ(J) > 1e4`
covariance warning stays; TSQR is not added to the module (plan Task 15 is skipped).
Evidence is thin, see "What the study does and does not show".

## Command and machine state

```bash
NLSQ_FORCE_CPU=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m benchmarks.studies.conditioning_study \
    --out <dir> --tol 1e-12      # and --tol 1e-15; both ran concurrently on the CPU
```

Seeds 0-4, accuracy set `n = 2000`, scale set `n = 2e4`, chunk 512, 50-digit mpmath reference.
i9-13900H (20 logical CPUs), governor `powersave`, no GPU used. Full printed output of both runs:
`benchmarks/baselines/2026-10-31-exact-streaming-conditioning-logs/tol1e-12.txt` and `tol1e-15.txt`.
Candidates: normal equations = the library path `exact_lm_fit`; TSQR = a loop in the study script that
mirrors the module's LM loop (same damping, Nielsen update, stopping), with `tsqr_solve` doing the
step from the augmented `[J | r]` R-factor. Both run on identical float64 observations and the same
`p0`; the reference starts from the normal-equations solution and is rounded to float64.

## Pre-registered rule applied

The candidates' stopping tolerance is not fixed by the spec's rule, and it decides the outcome, so both
values were run before any verdict was read:

| `ftol = xtol = gtol` | rows in the rule (`κ ≤ 1e4`, valid reference) | normal equations | TSQR | printed verdict |
|---|---|---|---|---|
| 1e-12 | 1 (two_exp_1e-1, seed 1, `κ = 8.5e3`) | param error 1.96e-6 (> 1e-6, fail), cov error 4.7e-6, rank equal | param error 1.96e-6 (fail), cov error 4.7e-6 | normal equations fail, but TSQR fails the same row: "stopping-limited / undecided" (the pre-registered function alone returns `tsqr_default`) |
| 1e-15 | the same row | param error 3.8e-8, cov error 5.8e-8, rank equal (pass) | 1.3e-8 / 3.0e-8 | `normal_equations_stay` |

Why the 1e-12 failure is not an accumulation failure: costs of the candidates and the reference agree
to 1e-15 relative (flat valley, the `ftol` stop fires early), the two candidates' parameter errors
agree to five digits, and evaluated at the same `θ_ref` the normal-equation step error is 1.6e-12 and
its covariance error 1.5e-8 (both far inside the thresholds). The one-step extended-precision check at
the normal-equation end point gives step errors 3.8e-13 (normal equations) and 1.1e-12 (TSQR).
Switching to TSQR would not change the 1e-12 outcome, so rule 3's premise (TSQR succeeds where normal
equations fail) does not hold. **Ruling:** apply the rule's intent (accumulation accuracy) and record
this as a documented deviation from a literal reading at tol 1e-12; the choice between 1e-12 and 1e-15
was not made on the verdict, both are reported.

## What the study does and does not show

- Only 1 of 40 accuracy rows falls in the rule's range `κ(J) ≤ 1e4`. `κ` at the reference optimum is
  2e4-5e4 for most two_exp_1e-1 seeds, 3e4 for gauss3, 7e5 for poly8, 7e8 for poly12.
- 20 of 40 accuracy rows have no valid reference: the undamped extended-precision Gauss-Newton does not
  converge on the near-degenerate two_exp problems (δ ≤ 1e-2 and part of 1e-4/1e-5; one gauss3 seed).
  The banner lists each with `κ` at the SciPy optimum, which is only a proxy (for two_exp_1e-1 seed 1 the
  SciPy optimum had `κ = 1.4e16` against the true 8.5e3). A failed row with a small true `κ` would be
  missed by the rule; the script blocks a verdict only when the proxy is `≤ 1e4`.
- Above the rule's range the expected `κ²`-scaling shows up in the normal-equation covariance:
  poly8 (`κ = 7e5`) covariance error about 5e-6 against 6e-12 for TSQR, same-theta; poly12
  (`κ = 7.5e8`) 5e-4 against 6e-5 to 2.1e-3 for TSQR (TSQR is not uniformly better there: one seed
  exceeds 1e-3). Normal equations stay under the 1e-3 threshold in every row, and the library already warns when `sqrt(cond(H)) > 1e4`. Parameter errors on poly12 for both candidates are
  dominated by the flat valley and the stopping rule (TSQR stops worse there: 1-4 relative error), not
  by accumulation.
- Scale-set rows (n = 2e4) compare the candidates with SciPy only; on the degenerate two_exp problems
  SciPy ends in other minima, so those errors measure the minimum reached. Candidate times in the study
  (host NumPy TSQR against the JAX module) are not a speed comparison. The speed clause of rule 3 was
  not needed.
- Rank agreement is vacuous at `κ ≤ 1e4` (`κ(H) ≤ 1e8` is far below `1/(p·eps)`); it is not evidence.

## Consequence

No code change. `CHANGELOG` and docs keep the `κ(J) > 1e4` covariance warning as the user-facing
mitigation. If a user needs accurate covariances for `κ(J) > 1e5` problems, the study's same-theta
rows show the TSQR route is a sound follow-up, outside this plan.
