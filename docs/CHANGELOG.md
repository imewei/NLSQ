# Documentation Changelog

## Unreleased
- Obsolete, unused and duplicate documentation removed (`/dev-suite:docs ... --full`). Candidates were found from
  an unmuted Sphinx build (`-D suppress_warnings=`), the toctree and cross-reference graph, a shingle-overlap
  scan for duplicate prose, and a sweep of the repository outside `docs/` for references.
  - Deleted: `tutorials/01_first_fit`, `02_understanding_results`, `03_fitting_with_bounds`,
    `05_large_datasets`, `06_gpu_acceleration` and `gui/user_guide`. Each was a seven-line "deprecated"
    redirect whose content lives in `tutorials/routine/getting_started/first_fit` and
    `tutorials/routine/gui_desktop/index`. `conf.py` (`REMOVED_PAGE_REDIRECTS`) writes meta-refresh pages at the six
    old URLs after an HTML build, so published links still resolve.
    The hidden "Legacy Redirects" toctree in `tutorials/index.rst` went with them, and
    `tutorials/04_multiple_parameters` is retitled "Multi-Parameter Models" with its prerequisite repointed.
  - Unsourced multipliers: `api/nlsq.validators.rst` "~30-50% faster" replaced by a measurement (one CPU, median of
    7 calls: 0.0002 s against 0.0019 s at 100,000 points, 0.0024 s against 0.0218 s at 1,000,000);
    `api/nlsq.unified_cache.rst` "2-5x" compile-time goal removed. The "3-25x" in `api/nlsq.svd_fallback.rst`
    matches its own table and is kept.
  - Removed 13 toctree entries and their "Chapter Overview" write-ups (with time estimates) for chapters that
    were never written, in `tutorials/advanced/{stability,performance,extension,custom_workflows}/index.rst`.
    `toc.not_readable` is no longer suppressed in `conf.py`, so a dangling toctree entry now warns.
  - `api/modules.rst`: removed 32 "NEW in v0.x" tags and the "Enhanced Features (v0.1.1)" heading; "New in
    version 0.x" lead-ins removed from eight API pages; "Task Group" markers removed from two; three
    `.. versionadded:: 1.2.0` directives removed (no such release; the latest tag is v0.7.6).
  - Duplicate: `api/nlsq.stability.rst` rendered `nlsq.stability.guard` a second time; it now links to
    `api/nlsq.stability.guard.rst`.
  - Stale values and paths: `api/nlsq.rst` "3819 tests" claim; `api/index.rst` benchmark pointer (now
    `api/performance_benchmarks`) and flat module names (`nlsq.minpack` and similar, now the real
    `nlsq.core.minpack`, `nlsq.streaming.hybrid_config`, ...); seven API pages that named a non-existent import path.
  - Linked from toctrees so they are no longer orphans: `developer/gui_development_guide`,
    `developer/svd_removal_technical_note` and ADRs 003-005.
  - README: broken `api.html` link and a Colab link to a notebook path that no longer exists; the GUI launch
    block duplicated under "With GUI Support"; "Levenberg-Marquardt" claim in the feature table; stale "v0.6.3"
    workflow wording; a redundant `Requires: pip install nlsq` line.
  - Kept after checking (not duplicates): the profiler, profiling and profiler_visualization pages (three
    different modules), `howto/index` (the only route to `howto/domain_applications/*`), `tutorials/index`.
    `api/index`, `howto/index` and `tutorials/index` are now in a hidden "Section Indexes" toctree in `index.rst`.
- Architecture documentation refreshed (`/dev-suite:docs update ... --full`), checked against the code.
  - `developer/architecture.rst`: package tree and layer diagram regenerated (equal-width rows); large-data
    story now starts from the routing (`curve_fit_large` spec, `Auto` at `size_threshold`, exact default,
    chunked fallback with a warning, `ValueError` for data reasons, `fit(workflow=...)` via
    `MemoryBudgetSelector`) with the hybrid 4-phase optimizer described as one opt-in engine; the two
    standard-optimization flows merged (`prepare_problem` includes `InputValidator`; the TRF loop is a
    Python loop over JIT kernels); `TrustRegionOptimizerBase` removed from the class hierarchy
    (`LeastSquares` accepts `trf` and `lm`, an alias); the fallback chain is Cholesky then eigendecomposition
    only (diagonal regularization belongs to the stability guard); the global-optimization "by dataset size"
    text replaced (selection is memory-based); `HealthReport` and `BoundInference` renamed
    `ModelHealthReport` and `BoundsInference`; CLI data formats (ASCII/CSV/NPZ/HDF5), exports (JSON/CSV) and
    the `gui` subcommand corrected; GUI documented as the optional `nlsq[gui]` extra with `param_config.py`;
    previously unnamed modules added; two environment variables added.
  - Line counts: ~78,000 became ~83,400; per-package totals are recursive and rounded to the nearest
    hundred, and the ~25 per-file counts were dropped because they drift.
  - `README.md` architecture diagram regenerated (equal-width rows, ~83,400 lines, routing, optional GUI).
  - `tutorials/advanced/architecture/index.rst` and `overview.rst`: module hierarchy, "Protocol-Based DI"
    claim (only five `Protocol` classes exist), dependency graph.
  - `developer/gui_development_guide.md`, `api/nlsq.gui_qt.rst`: removed the non-existent `export_adapter` and
    `base_plot`; `explanation/regularization.rst`: source paths under `nlsq/streaming/`.
  - Not changed: `CLAUDE.md` is gitignored (it still says JAX tested on 0.9.1 and that the GUI ships in the
    main dependencies); `optimistix` is declared in `pyproject.toml` but not imported under `nlsq/`.
- Performance and benchmark documentation rewritten (`/dev-suite:docs update --readme`, then the Sphinx pages).
  An earlier pass in this cycle compared NLSQ on GPU with SciPy on CPU; the earlier README table, the
  150-270x claim and the V100-era tables had no recorded source. Every comparison is now within one device
  (CPU against CPU, GPU against GPU, or NLSQ CPU against NLSQ GPU) and is generated from
  `benchmarks/baselines/2026-11-13-library-comparison-1k-100M.md`.
  - `api/performance_benchmarks.rst` is the single benchmark page: summary, methodology, CPU, GPU and
    device-scaling tables for four models, device guidance, limitations and reproduction commands. Its
    unsourced content (A100/EPYC tables, memory-savings, solver, chunking, sparse, streaming and "real-world"
    figures, JAX 0.4.35 setup) was removed.
  - README: headline, GPU-install note and Performance section (two same-device tables, caveats).
  - `howto/optimize_performance.rst` (comparison recipe with warm-up and `JAX_PLATFORMS=cpu`, CPU/GPU
    guidance, results section), `howto/migration.rst`, `explanation/gpu_architecture.rst` (the CPU-versus-GPU
    description now states that NLSQ uses autodiff and compiled code on both devices),
    `tutorials/routine/gpu_acceleration/index.rst`, `tutorials/advanced/performance/index.rst`,
    `tutorials/routine/data_handling/large_datasets.rst`, `index.rst`, `api/nlsq.rst`, `api/nlsq.device.rst`,
    `explanation/index.rst`, `explanation/jax_autodiff.rst` and `developer/performance_tuning_guide.md` no
    longer quote speedup multipliers of their own; they link to the benchmark page.
  - SciPy comparison: the harness runs SciPy with `method="trf"`. A spot check of SciPy's default `lm` on the
    exponential decay (`benchmarks/baselines/2026-11-14-scipy-lm-spot-check.md`) shows SciPy faster than NLSQ
    `curve_fit` up to 1M points; the pages state both and name the method.
  - Unverified claims removed: the "283x" expected output, batch-fitting "10x" and "50-100x", "10-100x"
    Jacobian and first-call figures, and the capacity bullets (near-linear scaling to 50M points, batch time,
    ~500 MB at 1M points). `developer/optimization_case_study.md` keeps its historical 150-270x claim.
    `nlsq/device.py` still prints "Enable 20-100x speedup:" (code, not changed here).
- `/dev-suite:docs update --full` pass after PR #68 (audit hardening): `CHANGELOG.md` Fixed gains the
  PR's user-visible fixes (only the export file mode and an earlier `NLSQ_SAVE_ITERATIONS` note existed); `reference/configuration.rst` names the
  `optimization_history_*.npz` file `NLSQ_SAVE_ITERATIONS` now writes per fit; `reference/cli.rst` lists
  the runtime-string attribute hooks the validator rejects; `api/nlsq.core.covariance.rst` and
  `developer/architecture.rst` describe the `pcov_from_normal` Jacobi equilibration; the GUI workflow
  tutorial states that exports use the fit-time settings and are written atomically. `CLAUDE.md` is
  gitignored and was not touched.
- `/dev-suite:docs update --full` pass for the exact default in `curve_fit_large`: README
  feature table, `api/index.rst` and the `curve_fit_large` section of
  `api/large_datasets_api.rst` describe exact full-data streaming as the default at or above
  `size_threshold` (chunked via `strategy=Chunked()`), the adaptive normal-equations/TSQR
  covariance, and the `strategy` and `compute_covariance` parameters. Sphinx build: no
  warnings.
- Large-fit strategy: new `api/nlsq.streaming.large_fit.rst` documents the five strategy
  types (`Auto`, `InMemory`, `Chunked`, `Hybrid`, `Exact`), the `Multistart` settings object
  nested in `InMemory`/`Chunked`, and `fit_with_strategy`, linked from `api/modules.rst`. Code changes the page describes:
  - `curve_fit_large(..., strategy=...)` keyword; legacy keywords build the same strategy.
    An explicit strategy combined with a non-default legacy path option (including any
    `np.ndarray` value) raises `TypeError`.
  - `InMemory.options` / `Chunked.options` keys that the strategy or the fit sets itself
    (for example `stability`, `multistart`, `sigma`, `chunk_size`) now raise `TypeError`
    instead of being silently overwritten.
  - `fit_with_strategy` keywords that the strategy owns (for example `stability` with `InMemory`,
    or `ftol` / `n_points` with `Exact`) now raise `TypeError` instead of being silently
    overwritten or bypassing a check.
  - A chunked fit that returns no `pcov` now raises `RuntimeError` instead of returning an
    identity matrix.
  - Error precedence for `method="hybrid_streaming"` changed: the unrecognized-keyword
    `TypeError` and hybrid config-value errors are now raised before data and source
    validation (empty data, chunk factory), not after it.
  - `curve_fit_large` return annotation corrected.
  - Known limitation: the chunked path ignores `stability`, `rescale_data`,
    `max_jacobian_elements_for_svd` and multistart.
- `/dev-suite:docs update --full` pass for the exact streaming solver and the 2026-11-01
  benchmark rerun: new `api/nlsq.streaming.exact_lm.rst` (`ExactSolverError`) linked from
  `api/modules.rst`; `exact-streaming` label on the reference section; the large-dataset
  tutorial gains an `exact_streaming` tip and its unmeasured "Typical performance" table
  (e.g. 100M points in about 10 minutes) is replaced by measured 1K/1M/10M/100M timings;
  `benchmarks/README.md` lists the `nlsq_large` and `nlsq_exact` adapters and the new
  baselines; README feature table mentions the exact streaming fit.
- `reference/large_data.rst`: new "Exact streaming" section for
  `curve_fit_large(method="exact_streaming")` (model contract, accepted and
  rejected arguments, 1-D `sigma` weights, box `bounds` by active set,
  `ExactSolverError`, covariance caveat, out-of-memory array-like sources such as
  memmap/HDF5/Zarr read chunk by chunk; chunk-factory sources: replayable
  `factory(chunk_size)`, optional `n_points`, per-pass count check).
  Added the 1B-point run result and the conditioning-study outcome to the covariance caveat.

## v0.7.6 (2026-09-04)
- No documentation content changes; version-output examples bumped to match
  the 0.7.6 release.

## v0.7.5 (2026-09-04)
- No documentation content changes; version-output examples bumped to match
  the 0.7.5 release.

## v0.7.4 (2026-08-30)
- No documentation content changes; version-output examples bumped to match
  the 0.7.4 release.

## v0.7.3 (2026-08-29)
- `/dev-suite:docs update --full` pass: AST-based public-API docstring scan
  (0 real gaps -- all 64 flagged symbols were nested closures, not public
  API) and an automodule/autoclass coverage cross-check against every
  `nlsq/` module found one real gap -- `CMAESPreempted` (exported from
  `nlsq.global_optimization` for the v0.7.3 checkpoint/resume feature) had
  no API reference entry. Added a Checkpoint/Resume subsection with a
  runnable example plus the `autoclass` entry to
  `docs/api/nlsq.global_optimization.rst`. README, CHANGELOG cross-refs,
  and the rest of the API reference were confirmed already in sync (no
  stale references to modules removed by the ponytail-audit PR); full
  `sphinx-build -W --keep-going` stays warning-free.
- Fixed a duplicated `[0.7.2]` heading in the root `CHANGELOG.md`, added
  missing entries for #22/#23, and fixed README's `workflow="hpc"` example
  (`checkpoint_dir` alone now raises -- `run_id`/`model_id`/`seed` are
  required together).
- Three-brain (Codex + Agy) review of `docs/`: fixed 26 stale/fictional API
  examples across 17 files.
- Removed ~27 API stub pages and two full tutorial sections documenting
  the `orchestration/`/`interfaces/`/`factories_di`/`smart_cache`/etc.
  modules deleted by the ponytail-audit dead-code removal (#32), and
  fixed every toctree/link that pointed at them (`sphinx-build -b html`
  clean, zero new warnings).
- Six-agent full-codebase review (PR #37) follow-up: dropped
  `docs/developer/architecture.rst`'s reference to the deleted
  `OptimizeResultV2` class (#37).

## v0.7.2 (2026-08-26)
- Synced the Optimization Pipeline and Package Overview docs with the current module layout
  after the three-brain review fix series.
- Fixed a malformed bullet list in `curve_fit_large()`'s docstring that made `make html
  SPHINXOPTS="-W --keep-going"` fail; full AST-based doc/API gap check found no other
  gaps (no public API surface changed since the last docs sync).

## v0.7.1 (2026-08-06)
- **Reverted the `[dev]`-extra packaging split.** An earlier iteration of v0.7.1 moved
  dev/docs/build tooling into a `[dev]` extra to slim down `pip install nlsq`. The split was
  rolled back within the release window because it left Makefile / `.readthedocs.yaml` /
  `docs/requirements.txt` / `CONTRIBUTING.md` / installation tutorial references pointing at a
  `[dev]` extra that no longer existed, breaking local dev installs and ReadTheDocs builds.
  Dev/docs/build/test/lint deps are once again in core `dependencies`; there is no `[dev]` extra.
  Stale `.[dev]` references were corrected to install the package itself.
- Added `graphify` knowledge graph output (`graphify-out/`) for codebase navigation.

## v0.7.0 (2026-06-24)
- **Python 3.14 support**: added the `Python :: 3.14` classifier and extended the CI test matrix to `3.12`/`3.13`/`3.14` across Ubuntu, macOS, and Windows; `requires-python` stays `>=3.12`. The full runtime dependency tree provides cp314 wheels and `curve_fit` converges correctly on 3.14.
- Correctness sweep (codex + agy review): fixed bounds validation in `curve_fit` input handling, covariance/SVD-rank edge cases, and several orchestration/result bugs.
- Fixed `common_jax` over-aggressive denominator guard that broke rank-deficient TRF solves.
- Fixed chunked-fit fallback path and made JIT cache keys include function code identity to prevent closure collisions across distinct models.
- Test/CI hygiene: relaxed the too-strict gradient-precision optimality threshold, warmed up before the memory-leak baseline benchmark, and silenced expected non-convergence warnings so `make verify` is clean.

## v0.6.12 (2026-05-09)
- Multi-subsystem RCA hardening across GUI, core, orchestration, and security/infra.
- **GUI (Qt/PySide6)**: fixed a `QThread` GC crash (strong refs in `_pending_threads`), replaced the shallow `FitWorker` snapshot with a deep copy, made autosave writes atomic (tmpfile + rename), corrected bounds/array serialisation, and fixed NaN/Inf JSON export via `safe_serialize`.
- **Core**: 11 `fit()`-workflow correctness fixes (dtype handling, CG bias, alpha overflow, theta bound, API compat).
- **Orchestration**: corrected `data_preprocessor` sigma handling, `optimization_selector` method selection, `covariance_computer` SVD rank, and `streaming_coordinator` memory budgeting.
- **Security / infra**: JIT-safety fix, `safe_serialize` data-integrity fix, tightened `model_validation` AST allowlist, GPU-fallback locale handling, and excluded `graphify-out/` from codespell/detect-secrets.

## v0.6.11 (2026-03-12)
- Skip sparsity detection when `tr_solver` is already set (7–8ms per-call savings)
- Fix Windows CI: catch `AttributeError` for POSIX-only `os.sysconf` in xdist worker cap hook
- Align `release.yml` test step with `ci.yml` to prevent timeout on publish-triggered test runs
- Replace `warnings.warn()` with `logging.debug()` in SmartCache for operational fallback paths
- Reduce JIT cache clearing interval and add `pytest_xdist_auto_num_workers` hook for OOM prevention
- Align config tested-on version comments with local environment
- Update `uv.lock` for black 26.3.1

## v0.6.10 (2026-03-09)
- Added `check_plugin_conflicts()` for CUDA plugin dual-install and version-mismatch detection in `device.py`
- Fixed misleading GPU warning header; expanded uninstall command to include CUDA plugin packages
- Promoted xxhash, pyyaml, evosax, psutil, and dev/test tools to core dependencies (single `uv sync`)
- Removed `extra_requirements: [docs]` from `.readthedocs.yaml`; docs deps covered by core
- Added `gpu-diagnose` Makefile target and 3-step CUDA 13 install with conflict verification
- Major `architecture.rst` overhaul reflecting v0.6.4+ orchestration components and facades
- Trimmed ~1500 lines of redundant inline code blocks from tutorials 01–06
- Refreshed 18 autodoc RST stubs and added 5 new API stubs (stability.guard, utils.diagnostics, callbacks, core.factories, interfaces)
- Clarified mixed-precision fallback log message; added streaming docstring note
- Moved deferred `scipy.stats` imports to module level in GUI export/results pages
- Security: cryptography 46.0.4 → 46.0.5 (CVE-2026-26007), pillow 12.1.0 → 12.1.1 (CVE-2026-25990)

## v0.6.9 (2026-02-14)
- Bump version to 0.6.9

## v0.6.8 (2026-01-28)
- Fixed critical GUI startup crashes on macOS (SIGBUS) caused by OpenGL/Metal conflicts
- Enhanced JAX platform guards and device detection reliability
- Unified logging handlers to prevent file lock issues on Windows
- Hardened test suite against environmental variability and timeouts
- Fixed GUI cache directory cleanup on shutdown


## v0.6.4 (2026-01-14)
- Updated version references from v0.7.0 to v0.6.4 across all orchestration documentation
- Added CurveFit decomposition architecture documentation (v0.6.4 feature)
- Updated package structure references to v0.6.4

## v0.6.0 (2026-01-06)
- Reorganized documentation into Routine User Guide and Advanced User Guide
- Consolidated 3 migration files into single comprehensive `howto/migration.rst`
- Removed stale deprecation notices from result module
- Removed 8 non-existent workflow presets from CLI documentation
- Fixed broken documentation links in README

## v0.5.4 (2026-01-05)
- Migrated to Furo theme for modern documentation styling
- Removed all deprecated functionality from NLSQ package
- Comprehensive documentation update with AST-based code analysis (94.9% coverage)
- Fixed pre-push hook CI interference
- Fixed Windows resource module import compatibility

## v0.5.3 (2026-01-04)
- Replaced pyqtdarktheme with Qt 6.5+ built-in color scheme (`setColorScheme()`)
- Updated dependency versions: JAX 0.8.2, NumPy 2.4.0, SciPy 1.16.3, ruff 0.14.10, mypy 1.19.1
- Added CHANGELOG entries for v0.5.0, v0.5.1, v0.5.2 release history
- Reorganized benchmark suite into ci/components/microbench directories

## v0.5.2 (2026-01-04)
- 15% performance improvement in TRF optimizer (10K point fit: 1.04s → 0.88s)
- Fixed JIT compatibility in TRF subproblem solvers
- Fixed enum identity issues in parallel test execution

## v0.5.1 (2026-01-03)
- Added `nlsq.gui_qt` API documentation to Sphinx
- Fixed README link to GUI user guide (now points to ReadTheDocs)
- Updated README GUI description to reflect pyqtgraph plots (was incorrectly referencing Plotly)
- Fixed pre-commit issues: refactored complex functions, added contextlib.suppress, created .secrets.baseline
- Added mypy overrides for PySide6, pyqtgraph, and qdarktheme in pyproject.toml

## v0.5.0 (2026-01-02)
- **BREAKING**: Removed Streamlit GUI in favor of native Qt desktop application
  - The `nlsq.gui` package has been removed entirely
  - Use `nlsq.gui_qt` and `nlsq-gui` command for the desktop GUI
  - Removed `gui` optional extra from pyproject.toml; use `gui_qt` instead
  - Install with: `pip install "nlsq[gui_qt]"`
- Removed legacy StreamingOptimizer/StreamingConfig and the Adam warmup path from docs; use AdaptiveHybridStreamingOptimizer with HybridStreamingConfig (L-BFGS warmup) for large datasets.
- Removed DataGenerator, create_hdf5_dataset, and fit_unlimited_data from examples/docs; use LargeDatasetFitter or AdaptiveHybridStreamingOptimizer instead.
- Updated streaming workflow guidance to reflect hybrid streaming only.
