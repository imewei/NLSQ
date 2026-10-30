# NLSQ Slimming and Small-Fit Overhead — Design

**Date:** 2026-10-30
**Status:** Draft, written without a question round (the requester opted out). Every choice marked **[my call]** is the author's, not the requester's. Revision 2 (same day) adds the four-baseline comparison (section B) at the requester's instruction.
**Source:** `three-brain-out/2026-10-30-nlsq-assessment/` (`consensus.md`, `codex-review.md`, `agy-analysis.md`). The assessment's measurements and verifications are the evidence base; its unverified claims are explicitly out of scope (see Deferred).

## Intent

The assessment found NLSQ's core fitting algorithm justified but its packaging and periphery over-built, and found it slower than `scipy.optimize.curve_fit` on CPU at every size tried (warm: 168 ms vs 0.7 ms at n=1e3; 225 ms vs 20 ms at n=1e5; 1.03 s vs 0.52 s at n=2e6; cold start 0.5–1.2 s; no GPU measured).

The aim is a lighter, more honest library: `pip install nlsq` installs what a fitter needs, small fits stop paying avoidable orchestration cost, orphaned code goes, and the one documented non-equivalence in the streaming path becomes machine-visible.

The comparison that motivated this work was NLSQ against SciPy on one CPU, one model. It is not enough to steer by. This revision adds a reproducible comparison of NLSQ against four baselines — **SciPy** (`curve_fit`/`least_squares`), **lmfit**, **JAXopt** (`LevenbergMarquardt`/`GaussNewton`) and **Optimistix** (`least_squares`) — and makes that harness the instrument for every performance claim and the before/after measurement for the overhead work.

## Goals

1. **Core install is lean.** `pip install nlsq` no longer pulls GUI toolkits or ~32 dev/test/docs/build tools. Contributors lose nothing: `uv sync` still yields a complete dev environment.
2. **A reproducible four-baseline comparison exists.** One command benchmarks NLSQ, SciPy, lmfit, JAXopt and Optimistix on the same problems, data, start points and stopping criteria, and writes JSON plus a Markdown report. Acceptance: a committed CPU report in which every baseline is either measured or listed as *unavailable* with the reason; the report also records the JAX backend/device and package versions, and the harness runs unchanged on a GPU host.
3. **Small-fit overhead is understood, then reduced.** Profile the warm n=1e3 call; remove the avoidable costs found. Acceptance: top cost centers are documented, the warm n=1e3 time falls at least 5× (168 ms → ≤ 34 ms) **[my call: target]**, or the write-up states why not, and the before/after comparison report shows NLSQ's ratio to each baseline.
4. **Orphaned code is removed or deprecated** without breaking the public API in one step.
5. **Chunked nonlinear fits say what they are.** A fit that took the multi-chunk `LargeDatasetFitter` path reports that it is not full-batch-equivalent in its result and in a log warning.

## Non-goals

- Changing any numerical result of any fit path.
- Consolidating the two streaming engines (`LargeDatasetFitter`, `AdaptiveHybridStreamingOptimizer`). Both are live in `fit()` and `curve_fit_large`; merging them needs its own design.
- Any GPU work (device-side TRF control flow, pure-JAX differentiable entry point, TSQR covariance). No GPU was available to measure; no change is justified without numbers.
- Moving `matplotlib`, `evosax`, `h5py`, `optax`, `optimistix` out of core **[my call]**: all are imported by non-GUI core paths or have no tested actionable-error path today.
- Changing the conda recipe **[my call]**: conda users keep the GUI via `pyside6`/`pyqtgraph` run deps; the lean-install goal is a PyPI concern.

## Anti-goals (would make this a failure even if tests pass)

- Repeating the v0.7.1 failure: the `[dev]` split was reverted because Makefile, `.readthedocs.yaml`, `docs/requirements.txt`, `CONTRIBUTING.md` and the installation tutorial still referenced the removed extra. Every reference to an install form must change in the same change set, and a test must catch a dangling reference.
- Adding upper version caps to any dependency (standing user preference).
- A user who runs `nlsq-gui` or `nlsq gui` without the GUI extra seeing a bare `ModuleNotFoundError`.
- Silencing diagnostics instead of removing their cost.
- A comparison that ranks a solver fastest because it stopped early, was handed looser tolerances, or crashed silently and dropped out of the table.
- Presenting JAXopt as a current alternative without saying its README states it "is no longer maintained nor developed".
- Treating lmfit as an independent algorithm: it wraps SciPy's solvers (default `leastsq`, MINPACK Levenberg–Marquardt), so its numbers measure SciPy plus lmfit's `Parameters`/`Model` layer, and the report must say so.

## Constraints

- Python ≥ 3.12; version floors in `pyproject.toml` stay as they are (copy, do not retighten). No new upper caps.
- `uv.lock` is the lockfile source of truth and must be regenerated and pass `uv lock --check`.
- Tests that construct Qt widgets keep `@pytest.mark.serial`; tests that spawn subprocesses importing JAX also need `@pytest.mark.serial` (see `CLAUDE.md`).
- `pytest` config ignores `DeprecationWarning` globally, so deprecation tests must use `pytest.warns`.
- Benchmark-only packages (`lmfit`, `jaxopt`) live in a separate `bench` dependency group, never in core or `dev`; Optimistix is already a core dependency. Floors only, no upper caps. If a baseline cannot be locked or imported with the project's JAX, the harness reports it as unavailable; the project must not pin JAX down or patch the baseline to make it work **[my call]**.
- No timing assertions in the test suite (flaky on shared runners). Tests cover the harness's correctness and schema; numbers come from running the harness.
- The comparison is a measurement tool, not a CI gate.
- Commits: conventional commits, atomic, no AI attribution.
- `make verify` (pre-commit, mypy, bandit, `pytest -m "not slow"`) must be clean at the end.

## Design

### A. Packaging split **[my call on all of A's mechanics]**

- `[project].dependencies` keeps only: numpy, scipy, jax, jaxlib, optax, optimistix, matplotlib, psutil, tqdm, h5py, xxhash, pyyaml, evosax.
- New extra `gui = ["PySide6>=6.10.0", "pyqtgraph>=0.14.0"]` (floors copied from today's list).
- New PEP 735 `[dependency-groups]` table with two groups. `dev` holds everything else now in `dependencies` (test, lint/type, docs, notebooks, build). `bench` holds `lmfit` and `jaxopt` (floors only) for the comparison harness; Optimistix needs no entry because it is core. `uv sync` installs the `dev` group by default, so CI jobs that run plain `uv sync` keep their tools. The dev group also pulls `nlsq[gui]` so GUI tests run; a scratch-project check on uv 0.11.28 confirmed that a self-referencing extra in a group resolves and installs. If `uv lock` rejects it in this repo, CI and the Makefile use `uv sync --extra gui` instead, and those edits land in the same task as the split.
- `docs/requirements.txt` lists the documentation packages explicitly plus `-e .` (ReadTheDocs cannot install dependency groups reliably). A test asserts the docs packages in the `dev` group are all present in that file.
- A guard test (`tests/packaging/`) asserts: no dev/gui package in `[project].dependencies`; every `nlsq[extra]` / `.[extra]` reference in `README.md`, `CONTRIBUTING.md`, `Makefile`, `.readthedocs.yaml`, `docs/requirements.txt`, `docs/tutorials/routine/getting_started/installation.rst` and `.github/workflows/*.yml` names an extra that exists in `[project.optional-dependencies]`.
- `nlsq.gui_qt.run_desktop` and the `nlsq gui` CLI command raise/print an actionable message naming `pip install "nlsq[gui]"` when PySide6 is missing.
- CI wheel-smoke job. The existing step runs `VIRTUAL_ENV=.venv-test uv run python -c …`, but `uv run` ignores a `VIRTUAL_ENV` that differs from the project's `.venv` (it warns and uses `.venv`), so that step tests the dev environment, not the wheel. Replace it with a check that calls `.venv-test/bin/python` directly and asserts a core-only install can `import nlsq`, run `curve_fit` and a genuinely chunked fit (at least two chunks; `curve_fit_large` sends fewer than 1 000 000 points to plain `curve_fit`, and the chunk size is floored at 1000, so the check forces chunking explicitly), call `CurveFitResult.plot`, and that PySide6 and pytest are not installed.

### B. Four-baseline comparison harness **[my call on all of B's mechanics]**

New package `benchmarks/compare/`, run as `uv run python -m benchmarks.compare.run_library_comparison`. It reuses nothing from `benchmarks/benchmark_suite.py` except its conventions, because that suite only knows NLSQ and SciPy.

**Baselines and how each is called** (signatures checked against the cited docs on 2026-10-30; re-verify against the installed versions during implementation):

| Library | Call | Notes |
|---|---|---|
| NLSQ | `nlsq.curve_fit(f, x, y, p0=…, bounds=…)` | model in `jax.numpy` |
| SciPy | `scipy.optimize.curve_fit(f, x, y, p0=…, bounds=…, method="trf")` | NumPy model; TRF so bounds work |
| lmfit | `lmfit.Model(f).fit(y, params, x=x, method="leastsq")` (`least_squares` for the bounded problem) | NumPy model; wraps SciPy; has named/bounded/expression-constrained `Parameters` |
| JAXopt | `jaxopt.LevenbergMarquardt(residual_fun=r, tol=…, maxiter=…).run(p0, x, y)` | JAX model; **README: "no longer maintained nor developed"**; documented NLS solvers are `GaussNewton` and `LevenbergMarquardt`; no bounds → bounded problem marked unsupported |
| Optimistix | `optimistix.least_squares(fn, optimistix.LevenbergMarquardt(rtol=…, atol=…), y0, args, max_steps=…, throw=False)` | JAX model; `fn(y, args)` returns residuals; default `max_steps=256`; no bounds → bounded problem marked unsupported |

**Fairness rules.**
- One `Problem` definition per scenario with the model written once as `model(xp, x, *params)` (`xp` is `numpy` or `jax.numpy`); same seeded data (seed 0, noise σ = 0.05), same `p0`, same iteration cap (1000) for every library.
- Stopping criteria mapped to a common 1e-8: `ftol = xtol = gtol = 1e-8` for NLSQ/SciPy/lmfit; `rtol = atol = 1e-8` for Optimistix; `tol = 1e-8` for JAXopt. Note in the report that the criteria are analogous, not identical.
- **Accuracy gate:** a run counts toward speed rankings only if its final cost is within 1e-6 relative of the best final cost any library reached on that case. Other runs are reported as `did_not_reach_optimum`, with their cost. A library's own `converged` flag is recorded but does not decide the status. For batch cases the cost is the mean per-fit cost and `converged` is true only if every fit converged.
- Timing: *cold* = first call in a **fresh worker process** (one per library × mode × problem × size), after imports and a trivial backend warm-up so device initialisation is not charged to any one library; persistent compile caches disabled (`NLSQ_DISABLE_PERSISTENT_CACHE=1`, JAX persistent cache off); *warm* = median and inter-quartile range of ≥ 7 further calls in that worker. JAX results are synchronised (`block_until_ready`) before the clock stops.
- JAX baselines get two modes: `end_to_end` (NumPy arrays in, NumPy params out — what NLSQ users do) and `jitted_core` (data already on device, solve pre-jitted — the ceiling for a JAX-native pipeline). NLSQ, SciPy and lmfit have `end_to_end` only. Headline tables use `end_to_end`.

**Scenarios.** Problems: `exp_decay` (3 params), `gaussian` (3), `two_exp` (4, ill-conditioned: rates 1.0 and 1.5), `bounded_exp` (3, bounds active). Sizes: 1e3, 1e5, 1e6 (`--quick`: 200 points, one repeat). Plus one batch scenario, `batch_small`: 256 independent fits of 200 points; JAXopt and Optimistix run it with `jax.vmap`, the others in a Python loop — the case where JAX-native solvers should win and where NLSQ's lack of composability would show.

**Output.** JSON records with: `library`, `mode`, `problem`, `n` (points per fit), `batch` (fits per call, `null` for single fits), `backend`, `device`, `status` (`ok` | `unsupported` | `unavailable` | `did_not_reach_optimum`), `reason`, `converged`, `cold_s`, `warm_median_s`, `warm_iqr_s`, `nfev` (when the library exposes it), `cost`, `cost_gap`, `max_param_error_vs_truth`. The Markdown report adds the package versions, CPU model and thread count, a capability matrix (bounds, weights, covariance, `vmap`-able, maintained) and the caveats above. Raw JSON goes to `benchmarks/results/` (git-ignored); the report for the committed run goes to `benchmarks/baselines/2026-10-30-library-comparison.md`.

**Feasibility gate.** Before building the harness around a baseline, confirm it imports and fits `exp_decay` in this environment (installed JAX is 0.12.x; JAXopt is unmaintained and may not import). A baseline that fails is recorded as `unavailable` with the error text and stays in the capability matrix.

### C. Small-fit overhead

1. Library logging is on by default (checked 2026-10-30, `NLSQ_DEBUG` unset): after `import nlsq` the root logger carries a stderr `StreamHandler` installed by an imported dependency, and NLSQ's loggers default to INFO and propagate to it, so about 11 INFO/PERFORMANCE lines print per fit in `logging`'s basic format (lmfit and JAXopt print none). A subprocess test pins "default import + fit emits no NLSQ log output on stdout or stderr; `NLSQ_DEBUG=1` still logs". Fix the NLSQ logger's default level, not its console handler.
2. Add `benchmarks/small_fit_overhead.py`: `cProfile` top-15 by cumulative time for one warm n=1e3 NLSQ call. Timing comparisons come from the harness in B, not from this script.
3. Run the harness (`exp_decay`, n = 1e3 and 1e5) before and after the fixes.
4. Remove the top avoidable costs the profile shows (candidates, unverified: input validation passes, sparsity/diagnostics checks, host↔device syncs per iteration, repeated config construction). Each removal keeps its safety behavior at trust boundaries (validation of shape/dtype/NaN stays; it may be done once instead of repeatedly).
5. Record the before/after tables in the CHANGELOG entry.

### D. Orphaned code

- Delete `nlsq/facades/` (389 lines; no production importer, verified by grep), `tests/facades/`, the one facade use in `tests/benchmarks/test_decomposition_memory.py`, and the docs pages and references that mention it (`docs/reference/facades.rst`, `docs/api/nlsq.facades.rst`, and the mentions in `docs/reference/index.rst`, `docs/api/modules.rst`, `docs/advanced_guide.rst`, `docs/developer/architecture.rst`, and the tutorials under `docs/tutorials/advanced/`).
- `nlsq/utils/async_logger.py` is not exported from `nlsq` or `nlsq.utils`: delete it and its tests. Those tests sit in a file that also holds unrelated profiling tests, so remove only the async-logger tests. Also delete the facade-only test `test_lazy_import_reduces_baseline_memory` and the facade mention in `README.md`.
- `PerformanceProfiler`, `ProfileMetrics`, `clear_profiling_data`, `get_global_profiler`, `ProfilerVisualization`, `ProfilingDashboard` are public (exported lazily from `nlsq/__init__.py`): keep them working, emit one `DeprecationWarning` per name on first attribute access, and state removal in the next minor release. They are not deleted now.

### E. Chunked-fit equivalence flag

In `nlsq/streaming/large_dataset.py`, every fit result gains `"equivalent_to_full_batch": bool`, true iff exactly one chunk was used. The fit routine `_fit_implementation` (line 1632) holds the single-chunk branch (near line 1716) and all multi-chunk returns: success via `_finalize_chunked_results` (line 2467), the low-success-rate failure (near line 2567) and the exception failure (near line 2846). The key and the warning apply to all of them, so a failed multi-chunk fit is also flagged. (`"processing_strategy"` at line 2888 belongs to the memory-recommendation dict, not a fit result.) When false, log one `logger.warning` per fit stating that chunk estimates were precision-weighted and are not equal to a full-batch nonlinear fit. No numerics change.

## Deferred (reopen only with the stated evidence)

| Item | Why deferred | Reopen when |
|---|---|---|
| Merge the two streaming engines | Both live; large risk | A separate design spec exists |
| QR/TSQR covariance in `adaptive_hybrid.py:956` | `JᵀJ` squares condition number, but impact unmeasured | A benchmark shows covariance error on an ill-conditioned problem |
| Device-side TRF control flow; pure-JAX entry point | GPU unmeasured | A GPU run of the comparison harness shows host sync dominating, or `batch_small` shows JAXopt/Optimistix beating NLSQ by a margin users would care about |
| Running the comparison on GPU | Not available in the authoring sandbox | Someone runs the harness on a GPU host and commits the report |
| `guard.py:365/467` Jacobian modification; `trf_jit.py:134` CG threshold | Codex claims, not verified | A failing reproduction exists |
| `unified_cache` vs JAX cache | Agy claim, not verified | Evidence it duplicates, not replaces, per-process closure keying |
| `ClusterDetector`, `DiagnosticPlugin` | Used by `_fit_with_hpc` / has 9 test files | Product decision to drop HPC/plugin features |

## Left to the builder

Exact profiling hypotheses and fixes in C; whether the dev group can self-reference `nlsq[gui]`; wording of the GUI-missing message (must contain `nlsq[gui]`); the harness's internal class layout (the fairness rules, scenarios and output fields in B are fixed); how each baseline's iteration count is read (leave `nfev` empty when a library does not expose it).
