# Repeated codebase review, 2026-11-05

## Scope and stopping condition

The user requested parallel whole-codebase review and debugging, repeated until
no findings remain. The review starts at `6a6488850d79106076594826bd5ddc7e7b9be328`;
the changed-code base is `11b3259abc1c95b564625bf1b1f1fef9c71911cc` (three-dot diff).
Review also covers the working tree and unchanged production modules.

The executable workflow is [review-debug-until-clean](../../workflows/review-debug-until-clean.md).
Completion requires two consecutive complete review passes with no new
actionable findings, no unresolved confirmed bugs, and final verification.
This is a bounded review result, not proof that no possible bug exists.

Parallel ownership initially separates core/public API, numerical pipelines,
CLI/GUI, and shared support code. Reviewers then rotate domains. Every production
module receives structural review; changed code and suspected runtime seams
receive detailed inspection and deterministic regression tests. Fast-context
could not authenticate; graphify and local source inspection provide context.

## Standards

No non-tool-enforced hard standards violations were identified in the pinned
diff. Fowler smells were used as heuristics. Similar streaming preparation and
result-conversion paths were considered advisory duplication: they represent
distinct algorithms and do not establish a concrete correctness defect. Broad
refactoring was therefore not treated as a required fix.

Repository scheduling guidance is corrected with the xdist configuration:
`loadgroup` honors serial groups; the collection hook preserves module/class
grouping for ordinary tests. The two-worker limit remains.

## Spec and runtime findings

All entries below have deterministic failing regressions observed before their
fixes. API docstrings, GUI recovery documentation, and existing exact-streaming
designs supply behavioral contracts; no originating feature issue was supplied.

| ID | Defect and correction | Regression evidence |
| --- | --- | --- |
| 01 | Direct least-squares residual and Jacobian wrappers ignored `data_mask`; mask both paths and validate shape. | `test_direct_mask_regression.py` |
| 02 | SciPy `(k, M)` input counted samples as features in memory selection; use the feature axis. | `test_workflow_layout_regression.py` |
| 03 | Explicit workflows converted `p0="auto"` to a scalar string array; resolve before dispatch. | `test_workflow_layout_regression.py` |
| 04 | Explicit workflows leaked routing-only `memory_limit_gb` to the solver; consume it during dispatch. | `test_workflow_layout_regression.py` |
| 05 | CMA-ES full output unconditionally reported convergence; preserve unsuccessful status in `ier`. | `test_cmaes_output_status_regression.py` |
| 06 | CMA-ES public wrappers reconstructed failed results as successes; propagate explicit optimizer/refinement metadata. | `test_cmaes_output_status_regression.py` |
| 07 | Weighted robust fallback rescaled `f_scale` despite invariant whitened residuals; retain weighted units. | `test_review_weighted_rescaling_20261105.py` |
| 08 | Singleton factory chunks reduced the contract probe to one point; retain at least two probe points. | `test_review_factory_probe_20261105.py` |
| 09 | Chunk buffers used only `x` dtype, rejecting fractional `y` with integer inputs; promote both dtypes. | `test_review_mixed_dtype_20261105.py` |
| 10 | Standalone warmup regularized normalized rather than physical parameters; denormalize the regularization input. | `test_review_phase_consistency_20261105.py` |
| 11 | Shared best-result tracking omitted the initial GN point, allowing a rejected trial to win; seed the tracker. | `test_review_phase_consistency_20261105.py` |
| 12 | CMA-ES refinement forwarded tuple-format flags and discarded its successful result; keep formatting outside internal refinement. | `test_review_refinement_modes_20261105.py` |
| 13 | Hybrid Sobol/Halton sampling received an LHS-only keyword; dispatch compatible arguments. | `test_review_hybrid_samplers_20261105.py` |
| 14 | Standalone warmup cost guard divided by an exact zero initial loss; guard the zero case. | `test_review_phase_consistency_20261105.py` |
| 15 | Public streaming dropped explicit centering and sampling scale; forward both settings. | `test_review_public_streaming_options_20261105.py` |
| 16 | CLI converted an estimator's `None` fallback into an object array; preserve the sentinel. | `test_loop_interface_estimator_none.py` |
| 17 | GUI scalar bounds loaded but crashed during fit snapshotting; normalize both supported YAML forms. | `test_loop_interface_import_bounds.py` |
| 18 | GUI crash recovery omitted workflow settings and pre-fit edits, and controls overwrote recovery; persist safe settings and hydrate controls. Custom source still requires explicit Apply. | `test_loop_interface_autosave_settings.py` |
| 19 | Outstanding pooled allocations aliased each other; remove buffers while checked out and return them explicitly. | `test_pool_allocation_isolation.py` |
| 20 | Accepted complex array dtypes failed JSON encoding; encode real/imaginary parts with explicit markers. | `test_complex_serialization_roundtrip.py` |
| 21 | Scalar initial points crashed while collecting validation errors; stop after the dimension error. | `test_scalar_validation_errors.py` |
| 22 | A sub-row memory budget produced zero-sized chunks and division by zero; raise a deliberate memory error. | `test_chunk_budget_errors.py` |
| 23 | Configuration getters/setters exposed caller-owned mutable state; copy configuration at ownership boundaries. | `test_config_snapshot_isolation.py` |
| 24 | Reused diagnostics retained prior-fit convergence histories and peak memory; reset per-fit state. | `test_diagnostics_reuse_regression.py` |
| 25 | Valid JSON with a non-object root crashed Jacobian-mode lookup; validate the configuration shape and fall back. | `test_jacobian_config_root_regression.py` |
| 26 | `serial` grouping was ignored by `loadscope`; use group-aware scheduling while retaining ordinary scope groups. | `test_test_isolation.py` |
| 27 | Explicit streaming ignored `check_finite=True`; run the existing finite-input validation before engine dispatch. | `test_review_public_streaming_options_20261105.py` |
| 28 | Malformed arrays escaped validation as raw NumPy exceptions; validate dtypes and dimensions, and return shape errors before dependent reductions. | Validator dtype and shape regressions. |
| 29 | GUI automatic guesses silently erased explicit bounds; preserve bounds independently of initial-guess mode. | GUI import, parameter-control, and fit-snapshot regressions. |
| 30 | Multi-start's docstring list lacked an RST boundary, failing the warning-as-error documentation build; add the required blank line. | Sphinx HTML build with `-W --keep-going`. |
| 31 | NaN tolerances bypassed positive-value validation; require a positive comparison while preserving existing infinity behavior. | Tolerance validation regressions. |
| 32 | Collection-time NLSQ imports initialized a shared JAX cache before the fixture's environment update; isolate cache directories before imports and explicitly configure JAX. | `test_compilation_cache_isolation.py` and two-worker tests. |
| 33 | Result documentation described a nonexistent conversion method, incorrect status/mask semantics, and automatic array conversion; align documentation with the existing container and optimizer behavior. | Independent source review and warning-as-error Sphinx build; no API or runtime change. |

Existing pool benchmarks now return checked-out arrays before expecting reuse or
eviction. Existing scalar-validation and feature-axis tests are corrected to
assert the supported behavior instead of preserving the original defect.

## Refuted candidates and review limits

- A fixture-only cache probe passed, but a follow-up with realistic collection-time
  NLSQ imports failed. Finding 32 fixes the demonstrated isolation gap; this does
  not retrospectively prove the cause of every historical cache warning.
- Warmup MSE and GN sum-of-squares tracking already convert units at their shared
  boundary; the suspected tracker-scale mismatch was refuted.
- Legacy public streaming `p0="auto"` is not documented for that route; the
  supported workflow route is covered by the resolved-guess regression.
- Custom model AST validation is documented as a best-effort denylist, not a
  sandbox. Recovery does not execute recovered custom code automatically.

## Review rounds and final verification

Initial review and the first rotated review found the defects above. Each new
confirmed defect reset the consecutive-clean counter. After the final fixes,
each domain completed two review passes without a new actionable finding:

| Domain | Final review emphasis | Result |
| --- | --- | --- |
| Core and public API | Routing, covariance, masks, callbacks, budgets, result formatting; independent SciPy comparisons. | Two clean passes; source hash `bddbdbf91d6ce88eeaa594286b6a7e4e560570780735309696f9d6be635b07a7`. |
| Numerical pipelines | Whole-module structural sweeps, status propagation, checkpoint/replay ownership, normalization and units. | Two clean passes; independent exact-budget, replay, and normalization invariants passed. |
| Shared support | All support modules, malformed inputs, configuration ownership, cache identity, reuse, serializers, mathematical primitives. | Two clean passes; production hash `d4c8fab8528071bcb2e769c9e86c83790c957bd9e95af5fc1930d6b467a96630`. |
| CLI and GUI | Configuration import through worker/export, control signals, recovery, option and bound preservation. | Two clean passes after the automatic-guess/bounds correction. |

The collection-time cache fixture correction received an additional independent
review and a 51-case two-worker run. Independent final support verification ran
101 focused tests. Final whole-tree production/test fingerprint at full-suite
start: `1abef0e19695908918aa9c5915a32d9a4dbde98f729dfe40c5afc212b7a4a976`.

Verified gates: Ruff check, Ruff formatting, mypy (133 modules, with the existing
package exclusions), Bandit medium/high severity, and Sphinx HTML with warnings
as errors. A real CUDA smoke check passed a fitted linear model, finite parameter
covariance, and masked direct least-squares fitting. AST graph refresh completed.

The full 5,410-case suite ran with two workers, offscreen Qt and reruns disabled:
5,400 passed, nine failed, one optional Zarr test skipped, and five subtests passed
in 57 minutes 41 seconds. The only warnings were four expected deprecations from
tests deliberately covering the legacy streaming preset.

All nine failures were test-contract maintenance:

- One routing assertion expected the removed routing-only `memory_limit_gb=None`
  keyword in solver arguments. The constructor's 2 GB budget and config-identity
  checks remain intact.
- Four autosave tests used a partial `SimpleNamespace` instead of the declared
  `SessionState`. Their helper now constructs the real state object.
- Four pool tests expected outstanding allocations to remain available. They now
  return buffers before asserting reuse/order/eviction, retaining those assertions
  and adding identity and checked-out-buffer checks.

The affected routing, GUI recovery, memory regression, pool and benchmark suites
then passed **210/210** cases. Exact JUnit `(classname, name)` reconciliation
confirms **5,409 unique passed cases, one optional skip, zero unresolved cases**.
This is a complete full-suite run plus an affected-suite rerun, not a claim that
the initial full run was green. No runtime implementation changed afterward;
only these test corrections and result documentation were updated. Independent
review checked the final corrections again.

The raw start/end fingerprint differed solely because generated `_version.py`
reflected the cleanup commit `2e37b6a694050bb5af1dc1483b46e8eaf301b69b` made during
verification. Replacing only its `dev186`/`g2e37b6a69` metadata with the prior
`dev185`/`g6a6488850` values reproduces the exact starting fingerprint. All library
and test contents otherwise matched through the frozen run; the generated version
file remains untracked/ignored. The separate cleanup commit is preserved.

Durable evidence is under `/tmp/nlsq-loop-20261105/`: `pytest.log`, `junit.xml`,
`corrected.xml`, `summary.json`, incremental failure records and `reconciled.json`.
The final report has no unresolved confirmed finding. Zero findings is limited to
the recorded review passes and evidence, not a proof over every possible input.
