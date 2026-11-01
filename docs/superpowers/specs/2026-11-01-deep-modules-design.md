# Deepening four modules in NLSQ: design and plan

Date: 2026-11-01. Status: **approved 2026-11-01** by the owner, with all section 9
proposed defaults accepted (phases/ routed through `NormalEquations`, nothing deleted; no
`pinv` change; no hybrid sigma support; one PR per opportunity). No code has been changed.
Source: four read-only design reviews (orchestration/preprocessing, protocol seams,
streaming stack, streaming config) run 2026-11-01. Vocabulary: module, interface, depth,
seam, adapter, leverage, locality (see the `codebase-design` skill).

## 1. Intent

Four places hold the same knowledge in several shapes. Each change below puts that
knowledge behind one small interface so a fix is made once. All four are
**behavior-preserving refactors first**. Any change in behavior (new sigma support,
changed covariance fallback) is a separate, later step with its own changelog line.

| # | Module | Replaces | Why |
|---|---|---|---|
| 1 | `NormalEquations` | 3 copies of J^T J / J^T r accumulation | one deep kernel, two real adapters |
| 2 | `build_large_strategy` (builder, not router) | 13 construction sites with their own kwargs translation | routing knowledge in one place |
| 3 | `prepare_problem` | `_prepare_curve_fit_inputs` plus a partial copy in the hybrid path | step-order rules live in comments only |
| 4 | covariance module | 4 covariance implementations | one fallback and warning contract |

Order: 1 first (requested; it unblocks 4 and de-risks the numerics path), then 3, 4, 2.
Rationale for 3 and 4 before 2: they are smaller and independent, and 2 touches the most
call sites. Each opportunity ships as its **own branch and PR**.

## 2. Goals, non-goals, anti-goals

Goals: fewer places to fix a bug; test through the module interface; no user-visible
numerical change in the refactor steps beyond documented tolerances.

Non-goals: rewriting `exact_lm` (it is already deep); merging the two cost conventions
into one public one; unifying rank cutoffs (see 6); the deferred items in section 8.

Anti-goals (would make this a failure even if tests pass): a second parallel pipeline
next to the first (the previous orchestration package failed exactly this way and was
deleted in `e25d560d`); silent changes to `pcov`; a performance regression on the 100M
and 1B gates; deleting anything because only `nlsq/` was searched (always sweep `tests/`).

## 3. Opportunity 1: `NormalEquations` (do first)

### Interface

```text
# nlsq/streaming/normal_equations.py   (extracted from exact_lm.py, not rewritten)
class NormalEquations:
    def __init__(self, model, *, weighted=False, chunk, jacobian="fwd"|callable): ...
    def accumulate(self, theta, source, *, check_finite=False, check_sigma=False) -> PassStats
    def cost(self, theta, source) -> float          # cost-only pass, no Jacobian
```

`PassStats` and `run_pass`/`make_chunk_stats` move out of `exact_lm.py` unchanged;
`exact_lm` imports them. This is a move plus a thin class, not new numerics.

### Facts the design rests on (verified in code)

- `run_pass` (exact_lm.py:361) reduces everything on device and syncs once per pass.
  `make_chunk_stats` uses `jacfwd` over the whole chunk with a padding `mask`.
- Hybrid `_accumulate_jtj_jtr` (adaptive_hybrid.py:2428) syncs per chunk via `float(...)`,
  uses per-point `vmap(jacrev)` (via `_compute_jacobian_chunk`, prefers
  `_jacobian_fn_compiled`), and writes `self.phase2_JTJ_accumulator` as a side effect.
  `_accumulate_jtj_jtr_scan` (:856) is a `lax.scan` variant; `phases/gauss_newton.py:630`
  is a third copy.
- The hybrid code adds group-variance penalty terms to cost and J^T J around the
  accumulation (~:3084-3108). `NormalEquations` returns the **data term only**; the
  penalty stays in hybrid.

### Traps that must be handled (each has a test or an explicit decision)

1. **Cost convention.** `PassStats.cost = 0.5 * sum(r^2)`; hybrid `residual_sum_sq =
   sum(r^2)`. The hybrid adapter converts once, in one named place. Every consumer of
   `res_sq`, `total_cost`, `chunk_cost` (adaptive_hybrid.py ~3041-3141, 3379-3446,
   3545-3689, 4439-4469; the covariance sigma^2 at ~3819-3983) is listed in the PR and
   checked. A missed factor of 2 silently shifts GN convergence and `pcov`.
2. **Checkpoint state.** `phase2_JTJ_accumulator` / `phase2_JTr_accumulator` are written per
   chunk and saved/restored by `phases/checkpoint.py` and `_load_checkpoint` (~4129), but a
   grep of `nlsq/` and `tests/` finds **no production reader**: after restore the next
   accumulation overwrites them, and hybrid checkpoints only at Phase 1/2 iteration
   boundaries (~2314, ~3509). Mid-pass resume therefore does not exist today. Resolved
   (my call): keep the field populated from the pass result at pass boundaries and leave
   the checkpoint format and field names unchanged, so old `.h5` files still load and no
   versioning is needed. A test saves, loads and resumes a checkpoint.
3. **Characterization first.** Before touching anything, add tests that pin the current
   behavior: H, g, cost, and final theta of hybrid Phase 2 on fixed seeds, with
   tolerances (not bitwise; `jacrev` vs `jacfwd` and per-chunk host sums vs device sums
   differ in the last bits). These tests must pass before and after.
4. **Model contract.** The normalized hybrid model may assume single-point `x[None]`
   (fallback path in `_compute_jacobian_chunk`). Verify it accepts batched `x`, or use the
   `jacobian=callable` adapter hook, before routing hybrid through the shared kernel.
5. **Performance gate.** The `lax.scan` path may beat a Python block loop on GPU for
   in-memory data. Acceptance (my call on the number): `bench` before/after on the hybrid
   path must stay within 5% wall time on CPU and on GPU at the sizes the hybrid bench
   already uses, plus the existing 100M and 1B exact-streaming gates unchanged. If the
   block loop is slower than that, keep scan as an internal strategy of `NormalEquations`,
   not as a second public method; otherwise delete it.

### Steps (each independently green)

1. Characterization tests for hybrid Phase 2 and for `exact_lm` (no source change).
2. Extract `PassStats`, `make_chunk_stats`, `run_pass`, sources into `normal_equations.py`;
   `exact_lm` imports them; full suite green; `exact_lm` benchmarks unchanged.
3. Route hybrid `_accumulate_jtj_jtr` through `NormalEquations` with the cost-convention
   adapter; delete `_accumulate_jtj_jtr_scan` and the three scan/loop branches
   (:3039, :3544, :4448) once equivalence tests pass; handle checkpoint per trap 2.
4. Route `phases/gauss_newton.py` through it **or** delete that copy (section 9 decision on
   the `phases/` fork; it has 72 tests, so not deleted without a decision).
5. Bench gate (trap 5). Changelog.

Behavior changes, **separate follow-up PRs, not part of the refactor**:
hybrid Phase 2 gains sigma/residual-weight support (currently rejected in `fit`, ~:4280).

## 4. Opportunity 3: `prepare_problem`

`prepare_problem(f, x, y, p0, bounds, sigma, data_mask, check_finite, nan_policy)
-> PreparedProblem` (frozen dataclass: `n, p0, lb, ub, x, y, sigma, mask, m_real,
n_effective, len_diff`). It owns the ordering rules now recorded only in comments in
`minpack.py`: NaN removal before method selection (~4191-4195); `data_mask` realigned after
omitting rows (~4210-4215); `ysize = sum(mask)` not `m` (~5313-5319); `len_diff` feeding
`_setup_sigma_transform` (~5256). Both `_prepare_curve_fit_inputs` (4097-4258) and the
duplicate in `_curve_fit_hybrid_streaming` (4374-4407) call it. Move existing code; no
second pipeline. Existing tests that call `_prepare_curve_fit_inputs` and
`_setup_data_mask_and_padding` are repointed at `prepare_problem`; `create_sigma_transform_funcs`
and `create_covariance_svd` are kept (tests use them).

Known behavior difference to resolve deliberately: hybrid path uses
`kwargs.pop("nan_policy")`, standard path `kwargs.get` (~4202). Proposed: `get` semantics
everywhere (the value is not consumed twice); confirm with a test before changing.

## 5. Opportunity 2: `choose_large_strategy`

**Scope (my call, from the builder check):** this is a *builder*, not a router. The 13
sites sit inside entry points that already know their path (`_fit_with_preset`,
`_fit_with_auto_global`, `_fit_global_multistart`, `_curve_fit_hybrid_streaming`,
`_curve_fit_auto_memory`, plus `__init__.py` and `large_dataset.py`), so routing must not
change. Name it `build_large_strategy(kind, config, **common) -> Strategy` where `kind`
is the path the caller already chose (`"hybrid"` or `"chunked"`), and `Strategy.fit(f, x, y,
p0, **common) -> OptimizeResult` hides the kwargs translation. Out of scope: global
optimization callers (multistart, CMA-ES, HPC) and `_curve_fit_exact_streaming`, which is
not among the 13 sites. Each site's current return type (tuple, `OptimizeResult` or dict)
is recorded in the routing-table test and adapted inside the `Strategy`. Replaces the 7
`AdaptiveHybridStreamingOptimizer(...)` sites (`__init__.py` 651, 1146; `large_dataset.py`
1900; `minpack.py` 782, 1091, 1829, 4446) and 6 `LargeDatasetFitter(...)` sites
(`__init__.py` 1253; `large_dataset.py` 3053; `minpack.py` 812, 1030, 1764, 4658). Each
site's kwargs translation (e.g. "has no gtol/ftol/xtol", minpack.py ~771) moves into the
matching `Strategy`.

Safety net first: record a **routing table** (inputs -> path taken) for all 13 sites as a
test, run it before and after; it must be identical. Public signatures of `curve_fit_large`
and `fit(preset=...)` do not change.

## 6. Opportunity 4: covariance

Two functions, one contract: `pcov_from_jacobian(J, cost, n_eff, p, absolute_sigma)` and
`pcov_from_normal(H, cost, n_eff, p, absolute_sigma)`, both returning `(pcov, warned)` with
the existing contract: all-inf plus `OptimizeWarning` when singular.
Sites: `minpack.py:3419` (SVD of J), `exact_lm.py:748` (eigh of H), `adaptive_hybrid.py:3727`
(bare `pinv`, no fallback), `large_dataset.py:2455` (GLS from chunk history; stays its own
adapter).

**Rank cutoffs are not unified.** `eps*max(m,p)*s0` on J and `p*eps*lambda_max` on H are not
equivalent (lambda = sigma^2). Each adapter keeps its cutoff unless a decision says
otherwise. The refactor step changes no value.

Behavior change, separate PR, **needs a decision (section 9):** hybrid's bare `pinv` becomes
inf plus warning. Today users get finite `pcov` from rank-deficient hybrid fits, so this
is user-visible.

## 7. Verification, every PR

`make verify` clean; characterization tests from the relevant step pass before and after;
no test deleted without a stated reason; `graphify update .` after code changes; serial
marker on any test that allocates over 100K elements or spawns JAX subprocesses; bench
gates for PR 1 only (and a smoke bench for PR 2).

## 8. Deferred (not in this work)

Inert learning-rate fields in `HybridStreamingConfig` (7 fields validated, never read in
production, confirmed by grep); splitting the config into frozen sub-configs; resolving
the `phases/` fork beyond PR 1 step 4; small seam cleanups (`FittableFunction`,
`ProgressCallback` wrappers, `StrategyFactory`); `CLAUDE.md` package-structure section
is stale (gitignored, fix separately).

## 9. Decisions needed from the owner

1. **`phases/` (most important; sets the scope of PR 1 step 4).** `GaussNewtonPhase` and
   the orchestrator are used only by `tests/streaming/test_phases*.py` (72 tests), not by
   production code. Delete them (with a stated reason), route them through
   `NormalEquations` as a third adapter, or make them the real seam that hybrid calls?
   Proposed default: route through `NormalEquations`, delete nothing yet.
2. Hybrid `pinv` -> inf + `OptimizeWarning`: accept as a later behavior change?
   Proposed default: not in this work; keep current finite `pcov`.
3. Should hybrid Phase 2 gain sigma support (later PR)? Proposed default: not in this work.
4. One PR per opportunity (proposed) or a single branch?

Resolved by the builder check and the code (my calls, please skim): mid-pass checkpoint
resume does not exist, so checkpoint format is unchanged (section 3, trap 2); the
`choose_large_strategy` scope and name (section 5); the 5% hybrid benchmark threshold
(section 3, trap 5).

## 10. Left to the builder

File and class names inside the modules; whether `Strategy` is a Protocol or a small
class hierarchy; exact tolerances for the characterization tests (start from existing
test tolerances).
