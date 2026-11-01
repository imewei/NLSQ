# Exact streaming solver, conditioning study and decision (2026-11-01)

Decision: **normal equations stay** (plan Task 15, TSQR in the module, is skipped), with a documented
interpretation of the pre-registered rule and thin evidence. The literal rule output depends on a
stopping tolerance the spec does not fix: `tsqr_default` at `ftol = xtol = gtol = 1e-12`,
`normal_equations_stay` at `1e-15`. If the maintainers read the rule literally at 1e-12, the follow-up
is Task 15; nothing else in this record changes.

## Command and machine state

```bash
NLSQ_FORCE_CPU=1 JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python -m benchmarks.studies.conditioning_study \
    --out <dir> --tol 1e-12      # and --tol 1e-15; both ran concurrently, CPU, with the test suite
```

Seeds 0-4, accuracy set `n = 2000`, scale set `n = 2e4`, chunk 512, 50-digit mpmath reference.
i9-13900H (20 logical CPUs), governor `powersave`, no GPU. Full printed output of both runs:
`benchmarks/baselines/2026-10-31-exact-streaming-conditioning-logs/tol1e-12.txt` and `tol1e-15.txt`.
Candidates: normal equations = the library path `exact_lm_fit`; TSQR = a loop in the study script that
mirrors the module's LM loop (same damping, Nielsen update, representable-step prediction and stopping),
whose step comes from the augmented `[J | r]` R-factor (`tsqr_solve`). Both run on identical float64
observations and the same `p0`. The extended-precision reference is started from each distinct
candidate solution (normal equations, TSQR, SciPy) in turn and the lowest converged cost is kept, so
no candidate's basin defines the target; a conflict between converged references would make the verdict
undecided (none occurred).

An earlier run of this study (reference started from the normal-equations solution only, eligibility of
failed rows judged by `κ` at the SciPy optimum) gave the same outcome; an independent review pointed
out both weaknesses and this record is the rerun after the fixes.

## Pre-registered rule applied

| `ftol = xtol = gtol` | rows in the rule (`κ ≤ 1e4`, valid reference) | normal equations | TSQR | literal rule output |
|---|---|---|---|---|
| 1e-12 | 1 (two_exp_1e-1, seed 1, `κ = 8.5e3`) | parameter error 1.96e-6 (> 1e-6, fail), covariance error 4.7e-6, rank equal | 1.96e-6 (fail), 4.7e-6 | `tsqr_default` |
| 1e-15 | the same row | 3.8e-8, 5.8e-8, rank equal (pass) | 1.8e-7, 4.2e-7 (pass) | `normal_equations_stay` |

Interpretation, recorded as a ruling by the implementation controller: at 1e-12 the failure is a
stopping artifact shared with TSQR, not accumulation error. Evidence: the two candidates' parameter
errors agree to five digits and their costs agree with the reference to 1e-15 relative (a flat
valley, `ftol` fires early); evaluated at the same `θ_ref`, the normal-equation step error is 1.6e-12
and its covariance error 1.5e-8 (TSQR: 8e-15 and 2e-13), far inside the thresholds; the one-step
extended-precision check at the normal-equation end point gives step errors 3.8e-13 (normal equations)
and 1.1e-12 (TSQR); and tightening the tolerance to 1e-15 moves the normal-equation error from 2e-6 to
4e-8. The plan's own tolerance table already relaxes the parameter comparison to `rtol 1e-4` for the
flat-valley two_exp problem at 1e-12, and 1.96e-6 passes that. The choice is not a free one: the rule
as written (parameter error `≤ 1e-6`) is what fails at 1e-12, and a maintainer who insists on it must
build TSQR. The tolerance was not chosen on the verdict; both are reported.

## What the study does and does not show

- Only 1 of 40 accuracy rows falls in the rule's range `κ(J) ≤ 1e4`; `κ` at the reference optimum is
  2e4-5e4 for most two_exp_1e-1 seeds, 3e4 for gauss3, 7e5 for poly8, 7e8 for poly12.
- 18-19 of 40 accuracy rows have no valid reference in each run (the undamped extended-precision
  Gauss-Newton does not converge on the near-degenerate two_exp problems and one gauss3 seed). Each
  failed row's `κ(J)` was computed at the normal-equation, TSQR and SciPy solutions; all of them exceed
  1e4 (smallest 1.98e4, gauss3 seed 3), so none belongs to the rule's range. A `κ` evaluated at an
  approximate solution of a degenerate problem is itself uncertain (for the one row with a valid
  reference, SciPy's value was 1.4e16 against the true 8.5e3), which is why the rule is judged on one row.
- Above the rule's range the expected `κ²` scaling appears in the normal-equation covariance: poly8
  (`κ = 7e5`) covariance error 5.0e-6 against 6e-12 for TSQR at the same parameters; poly12
  (`κ = 7.5e8`) 5.0e-4 against 6e-5 to 2.1e-3 for TSQR (TSQR is not uniformly better there: one seed
  exceeds 1e-3). In those poly rows the normal-equation covariance error stays under 1e-3. It does not
  on the degenerate two_exp rows (two_exp_1e-4 seeds 0 and 2: covariance error 1.0 and 0.33, `κ` 6e4
  and 9e4; the candidates' end-to-end errors there are identical to TSQR's, and the reference start
  conflict check is clean), so the covariance warning is needed, not cosmetic.
- Parameter errors on poly12 for both candidates are dominated by the flat valley and the stopping
  rule (TSQR stops worse there: relative error 1-4), not by accumulation.
- Scale-set rows (n = 2e4) compare the candidates with SciPy only; on the degenerate two_exp problems
  SciPy ends in other minima, so those errors measure the minimum reached. Candidate times in the study
  (host NumPy TSQR against the JAX module) are not a speed comparison; the speed clause of rule 3 was
  not needed.
- Rank agreement is vacuous at `κ ≤ 1e4` (`κ(H) ≤ 1e8` is far below `1/(p·ε)`), so "rank equal" is not
  evidence.

## Consequence

No code change. `CHANGELOG` and the docs keep the `κ(J) > 1e4` covariance warning as the user-facing
mitigation. If accurate covariances for `κ(J) > 1e5` problems are needed, the same-theta rows show the
TSQR route is a sound follow-up (Task 15 of the plan), outside this decision.
