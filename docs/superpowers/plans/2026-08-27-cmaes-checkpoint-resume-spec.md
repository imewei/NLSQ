# Spec: CMA-ES Checkpoint/Resume (Phase 1 of workflow='hpc' checkpointing)

**Status:** Draft
**Supersedes:** the 7-point proposal reviewed in
`three-brain-out/2026-08-27-hpc-checkpoint-design/consensus.md`; formalizes
Phase 1 of `three-brain-out/2026-08-27-hpc-checkpoint-design/revised-plan.md`.

## 1. Problem

`workflow='hpc'` (`nlsq/core/minpack.py:_fit_with_hpc`) accepts `checkpoint_dir`
and `checkpoint_interval` but silently discards them after a `UserWarning`
(`minpack.py:1371-1382`). No checkpoint/crash-recovery exists anywhere in the
CMA-ES (`CMAESOptimizer`) or Multi-Start (`MultiStartOrchestrator`) code
paths that `workflow='hpc'`/`workflow='auto_global'` use. A long-running fit
on a preemptible/walltime-limited HPC node currently loses all progress on
kill.

## 2. Scope

**In scope (v1):** `CMAESOptimizer` with `config.restart_strategy == "none"`
only (`nlsq/global_optimization/cmaes_optimizer.py:_run_cmaes_single`).

**Explicitly out of scope for v1** (each gets its own future spec):
- BIPOP restarts (`restart_strategy == "bipop"`, the *default* — see §7 for
  the required guard). Its restart-cycle state (`BIPOPRestarter`,
  alternating popsize choice, `original_solution`/explore-vs-exploit choice
  history) is materially more complex to resume correctly than a single run;
  bundling it into v1 risks shipping an unverified resume path for the
  common case.
- `MultiStartOrchestrator` (thread-pool dispatch model, different state shape).
- `LargeDatasetFitter` (chunked route) — needs separate scoping.
- The streaming route (`AdaptiveHybridStreamingOptimizer`) — separate,
  smaller, already-scoped task: wire up the existing but disabled
  `resume_from_checkpoint` (`adaptive_hybrid.py:4483`).
- Multi-device/distributed coordination.

## 3. Functional requirements

- **FR1 — Enable checkpointing.** `CMAESOptimizer(config=CMAESConfig(...))`
  gains new `CMAESConfig` fields: `checkpoint_dir: str | Path | None = None`,
  `checkpoint_interval: int = 10` (generations), `run_id: str | None = None`,
  `model_id: str | None = None`. Checkpointing is enabled iff
  `checkpoint_dir is not None`.
- **FR2 — Seed required.** If `checkpoint_dir` is set and `config.seed is
  None`, raise `ValueError` at `CMAESOptimizer.__init__` time (or first
  `.fit()` call) — never silently proceed with an OS-entropy seed that
  would make resume produce a different trajectory than an uninterrupted run.
- **FR3 — BIPOP guard.** If `checkpoint_dir` is set and
  `config.restart_strategy == "bipop"`, raise `NotImplementedError` with a
  message telling the caller to set `restart_strategy="none"` for now.
- **FR4 — model_id and run_id required.** If `checkpoint_dir` is set and
  either `model_id` or `run_id` is `None`, raise `ValueError`. The model
  function's closure cannot be safely fingerprinted (`co_code`/`co_consts`
  alone miss closure-cell values — confirmed in the three-brain review), so
  model identity is the caller's responsibility, not inferred. `run_id` is
  required for the same reason `model_id`/`seed` are: an unset `run_id`
  silently defaulting to a shared filename (e.g. `"default"`) would let
  unrelated runs collide on the same checkpoint file, which is exactly the
  class of silent-cross-run-contamination bug the other two required fields
  exist to prevent (confirmed in grilling review — the original draft
  defaulted `run_id`, inconsistently, to `"default"`).
- **FR5 — Periodic save.** Every `checkpoint_interval` completed generations
  (and unconditionally after the final generation, converged or not), the
  full state needed to resume is written to
  `{checkpoint_dir}/{run_id}.h5` (see §5 for state contents, §6 for file
  safety).
- **FR6 — Auto-resume by run_id.** On `.fit()` entry, if
  `{checkpoint_dir}/{run_id}.h5` exists and its fingerprint (§6) matches the
  current call's inputs, resume from it: skip fresh initialization, continue
  the generation loop from the saved `generation_counter`, and preserve
  `fitness_history` (append, don't reset). If it exists and the fingerprint
  does **not** match, raise `ValueError` — never silently start fresh in the
  same file.
- **FR7 — Preemption signal.** `SIGTERM` and `SIGUSR1` handlers set a
  `threading.Event` only (no I/O in the handler). The generation loop checks
  the event once per completed generation (never mid `ask`/`tell`); if set,
  it performs one final checkpoint save at that boundary, then raises
  `KeyboardInterrupt`-compatible exit path documented in §8, distinct from a
  normal `max_generations`/`xtol` stop. Handler *registration* only happens
  on the main thread (`signal.signal` raises `ValueError` off it) — guard
  with `threading.current_thread() is threading.main_thread()` and skip
  registration (log, don't raise) when called from a worker thread; periodic
  interval saves remain the crash-safety net in that case.
- **FR8 — Atomic writes.** Every checkpoint save writes to a temp file in the
  same directory, fsyncs the file, `os.replace`s over the previous
  `{run_id}.h5`, and then fsyncs the *containing directory* too (a bare file
  fsync does not guarantee the rename's directory-entry update survives a
  crash) . The immediately-prior good file is rotated to `{run_id}.h5.bak`
  before replacement. `load()` **must** fall back to `.bak` (not merely log
  a warning) whenever the primary file is missing its completion marker,
  fails to open, or fails HDF5 validation — this fallback is a hard
  requirement, not best-effort, since save() unconditionally rotates the
  prior good file specifically so this path is always available.

- **FR9 — `method` actually forces route selection.** `fit()`'s top-level
  `method` parameter is currently consumed by `fit()`'s own signature and
  never reaches `_fit_with_auto_global`, which hardcodes
  `requested_method="auto"` to `MethodSelector.select()` regardless of what
  was requested (confirmed via code review: `minpack.py` — the selector
  itself already honors an explicit `"cmaes"`/`"multi-start"` request; only
  the plumbing between `fit()` and the selector is missing). `method` must
  be threaded through `fit()` → `_fit_with_hpc`/`_fit_with_auto_global` →
  `method_selector.select(requested_method=method or "auto", ...)` for
  every `workflow='auto_global'`/`'hpc'` caller, not only when checkpointing
  is enabled — without this, `workflow='hpc'` checkpointing only works by
  accident of data scale ratio, not by request.

## 4. Non-functional requirements

- **NFR1 — Determinism.** Given the same `run_id`, `checkpoint_dir`, and
  `config.seed`, (fresh run to generation N+M) and (fresh run to generation
  N → process exit → new process → resume → generation N+M) must produce
  identical `best_solution`, `best_fitness`, and `generation_counter` to
  float64 exact-bit tolerance (`np.testing.assert_array_equal` on the raw
  arrays — CMA-ES/evosax and JAX PRNG splitting are both deterministic given
  a fixed key).
- **NFR2 — Save latency measured, not assumed.** Before this spec's
  acceptance, benchmark `HPCCheckpointManager.save()` wall time at
  `n_params` ∈ {10, 100, 1000} on the CI runner's filesystem; record the
  numbers in the plan's final task. Measured (mean / max per 5 runs):
  n_params=10: 6.09ms / 13.28ms, n_params=100: 8.99ms / 16.62ms, n_params=1000:
  16.27ms / 23.92ms. Well under a generation's compute time even at n_params=1000;
  no wall-clock-interval option needed for v1.

## 5. State model

`CMAESCheckpointState` (new dataclass, `nlsq/global_optimization/checkpoint.py`):

| Field | Type | Source |
|---|---|---|
| `generation_counter` | `int` | evosax `State.generation_counter` |
| `mean` | `jax.Array` (n_params,) | evosax `State.mean` |
| `std` | `jax.Array` (scalar) | evosax `State.std` |
| `p_std` | `jax.Array` (n_params,) | evosax `State.p_std` |
| `p_c` | `jax.Array` (n_params,) | evosax `State.p_c` |
| `C` | `jax.Array` (n_params, n_params) | evosax `State.C` |
| `B` | `jax.Array` (n_params, n_params) | evosax `State.B` |
| `D` | `jax.Array` (n_params,) | evosax `State.D` |
| `best_solution` | `jax.Array` (n_params,) | evosax `State.best_solution` |
| `best_fitness` | `float` | evosax `State.best_fitness` |
| `key_data` | `np.ndarray[uint32]` shape (2,) | `jax.random.key_data(key)` after the generation's final split |
| `fitness_history` | `list[float]` | `CMAESDiagnostics.fitness_history` (verified field, `cmaes_diagnostics.py`) |
| `popsize` | `int` | as passed to `_run_cmaes_single` |

Confirmed via direct introspection this session: `evosax.algorithms.CMA_ES.init(...)`
returns `evosax.algorithms.distribution_based.cma_es.State` with exactly
these fields: `best_solution, best_fitness, generation_counter, mean, std,
p_std, p_c, C, B, D`. `params` (`es.default_params`, holding `std_init` etc.)
is NOT checkpointed — it's rebuilt identically from `CMAESConfig` on resume
(pure function of config, not evolving state).

## 6. File format and fingerprint

HDF5, mirroring `nlsq/streaming/phases/checkpoint.py`'s conventions
(`f.attrs["version"]`, groups, `create_dataset`, `safe_dumps`/`safe_loads`
for non-array metadata via `np.void(bytes)`), but as a new, independent
module — not a subclass (confirmed not reusable: tightly coupled to
`HybridStreamingConfig`/Optax/tournament state, and has its own latent
normalizer-on-load bug, out of scope here).

```
{run_id}.h5
  attrs: version="1.0", completion_marker=True (written last)
  /state/        -- CMAESCheckpointState fields as datasets (arrays) +
                     safe_dumps blob for fitness_history (list)
  /fingerprint/
    model_id            (str, attr)
    data_hash            sha256 hex of (dtype, shape, bytes) of xdata||ydata||sigma
    n_params             (int, attr)
    bounds_lower/upper    (arrays)
    config_hash           sha256 hex of a canonical repr of the checkpoint-
                           relevant CMAESConfig fields (popsize, sigma,
                           tol_fun, tol_x, seed) -- NOT checkpoint_dir/
                           checkpoint_interval/run_id/model_id themselves
```

Fingerprint mismatch on load → `ValueError`, never a silent fresh start.

## 7. Acceptance criteria

- [ ] FR1-FR9 each have at least one passing test (see plan for exact tests).
- [ ] A corrupted/torn primary checkpoint file falls back to `.bak` on load
      (FR8) rather than raising or silently starting fresh.
- [ ] `fit(workflow='hpc', method="cmaes", ...)` actually selects the
      CMA-ES route regardless of data scale ratio (FR9).
- [ ] NFR1's determinism test passes bit-exact.
- [ ] NFR2's latency numbers are recorded (informational, no fixed gate yet).
- [ ] `restart_strategy="bipop"` + `checkpoint_dir` set raises
      `NotImplementedError` (does not attempt anything).
- [ ] Existing `workflow='hpc'` tests (`test_workflow_presets.py`,
      `test_pr19_review_fixes.py`) still pass unmodified — this spec adds
      capability, it does not change `_fit_with_hpc`'s current behavior for
      chunked/streaming/multistart routes (those keep warning, per Scope §2).
