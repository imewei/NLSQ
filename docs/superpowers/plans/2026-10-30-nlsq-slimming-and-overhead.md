# NLSQ Slimming, Baseline Comparison and Small-Fit Overhead Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pip install nlsq` lean, build a reproducible NLSQ-vs-SciPy/lmfit/JAXopt/Optimistix benchmark, use it to cut avoidable small-fit overhead, remove orphaned code, and flag non-full-batch chunked fits.

**Architecture:** Packaging moves the GUI to an `nlsq[gui]` extra and tooling to PEP 735 `dev`/`bench` groups, guarded by a test that catches dangling install references (the v0.7.1 failure). The comparison harness (`benchmarks/compare/`) wraps each library in an adapter behind one interface, computes cost uniformly from returned parameters, gates speed rankings on reaching the common optimum, and reports unavailable libraries instead of dropping them. Overhead work is profile-first and measured with the harness. No fit numerics change.

**Tech Stack:** Python ≥3.12, setuptools + setuptools-scm, uv, pytest (+xdist, `serial` marker), JAX, SciPy, lmfit, JAXopt, Optimistix, GitHub Actions, ReadTheDocs.

**Spec:** `docs/superpowers/specs/2026-10-30-nlsq-slimming-and-overhead-design.md` (evidence: `three-brain-out/2026-10-30-nlsq-assessment/`, git-ignored)

## Global Constraints

- Version floors in `pyproject.toml` are copied unchanged; add **no** upper caps (including for `lmfit`, `jaxopt`).
- `uv.lock` regenerated with `uv lock`; `uv lock --check` must pass.
- Core `[project].dependencies` is exactly: numpy, scipy, jax, jaxlib, optax, optimistix, matplotlib, psutil, tqdm, h5py, xxhash, pyyaml, evosax.
- Extra `gui = ["PySide6>=6.10.0", "pyqtgraph>=0.14.0"]`. Group `bench = ["lmfit", "jaxopt"]` (floors from the latest release at implementation time).
- Harness constants: stopping tolerance `1e-8` for every library, iteration cap `1000`, data seed `0`, noise σ `0.05`, accuracy gate `1e-6` relative cost gap, ≥ 7 warm repeats (non-quick). Cost is always recomputed by the harness from returned parameters as `0.5 * sum((y - model(np, x, *p))**2)`.
- Tests contain **no timing assertions**. Tests that spawn subprocesses importing JAX or build Qt widgets carry `@pytest.mark.serial`; run them with `-n 0`.
- `pyproject.toml` ignores `DeprecationWarning` globally: deprecation tests use `pytest.warns`.
- Environment: run tests with `HOME=$TMPDIR/fakehome NLSQ_SKIP_GPU_CHECK=1 uv run pytest …` (read-only HOME breaks custom-model tests).
- Commits: conventional, atomic, no AI attribution. Work on branch `chore/slim-install-and-overhead` (create in Task 1; never commit to `main`).
- No fit result may change: `tests/core/test_curvefit_golden_master.py` stays green in every task.

## Review Focus

- A user with only `pip install nlsq` runs `nlsq-gui` or `nlsq gui`: expects a one-line message naming `pip install "nlsq[gui]"`, not a `ModuleNotFoundError` traceback (Task 2).
- A core-only environment does `import nlsq`, `curve_fit`, `curve_fit_large` and `CurveFitResult.plot()`: all must work without PySide6 (Task 4).
- A baseline that cannot import or fit in this environment (JAXopt is unmaintained; installed JAX is 0.12.x) must appear in the report as `unavailable` with the error text — never crash the run or vanish from the table (Tasks 6, 7).
- A solver that stops early at a worse cost, or was given looser tolerances, must not be ranked fastest (Tasks 5, 7).
- Default import + fit with `NLSQ_DEBUG` and `NLSQ_VERBOSE` unset must write no NLSQ `INFO:`/`PERFORMANCE:` lines to stdout or stderr; `NLSQ_DEBUG=1` still must (Task 8).
- Cold-start numbers are comparable across libraries only if each is measured in a fresh process after the same backend warm-up (Task 7).
- A batch case's cost and convergence aggregate over every fit in the batch, and a zero best cost neither crashes the gate nor rejects an exact fit (Tasks 5, 7).

## Suggested PR boundaries

Packaging (Tasks 1–4) · harness (Tasks 5–7) · cleanup (Tasks 9–10) · overhead (Task 8, after the harness baseline is merged). Task 11 runs on whichever branch is last. Tasks 1–4 and 9–10 are independent of the harness.

## File Structure

- Modify `pyproject.toml`, `uv.lock`, `Makefile`, `.readthedocs.yaml`, `docs/requirements.txt`, `CONTRIBUTING.md`, `README.md`, `docs/tutorials/routine/getting_started/installation.rst`, `benchmarks/README.md`, `.github/workflows/*.yml` (install lines only where needed), `CHANGELOG.md`.
- Create `tests/packaging/{__init__,test_dependency_layout}.py`, `scripts/check_core_install.py`, `tests/cli/test_gui_missing_dependency.py`.
- Create `benchmarks/compare/{__init__,types,problems,timing,adapters,report,run_library_comparison}.py`, `tests/benchmarks/test_compare_{core,adapters,runner}.py`, `benchmarks/baselines/2026-10-30-library-comparison.md`, `benchmarks/small_fit_overhead.py`, `tests/core/test_default_logging.py`.
- Create `tests/utils/test_profiler_deprecation.py`, `tests/streaming/test_large_dataset_equivalence_flag.py`.
- Modify `nlsq/gui_qt/__init__.py`, the `gui` CLI handler (`grep -rn "def handle_gui" nlsq/cli`), `nlsq/__init__.py`, `nlsq/streaming/large_dataset.py`, plus whatever the profile implicates in Task 8.
- Delete `nlsq/facades/`, `tests/facades/`, `nlsq/utils/async_logger.py` and its test, facade docs pages.

---

### Task 1: Core / extra / dev + bench groups in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml` (`[project].dependencies`, `[project.optional-dependencies]`, new `[dependency-groups]`), `uv.lock`
- Create: `tests/packaging/__init__.py`, `tests/packaging/test_dependency_layout.py`

**Interfaces:**
- Produces: extra `gui`; groups `dev` and `bench`; test helpers `_load_pyproject() -> dict` and `_dist_name(requirement: str) -> str` (lower-cased, `_`→`-`, name only) reused by Task 3.

- [ ] **Step 1: Create branch** `git switch -c chore/slim-install-and-overhead`.
- [ ] **Step 2: Write failing tests** in `tests/packaging/test_dependency_layout.py` (parse with `tomllib`):
  - `test_core_dependencies_are_exactly_the_runtime_set`: `{_dist_name(r) for r in project["dependencies"]} == {"numpy","scipy","jax","jaxlib","optax","optimistix","matplotlib","psutil","tqdm","h5py","xxhash","pyyaml","evosax"}`.
  - `test_gui_extra_declares_toolkit`: names in `optional-dependencies["gui"]` equal `{"pyside6","pyqtgraph"}`, both with `>=` floors.
  - `test_dev_group_has_test_lint_docs_build_tools`: `{"pytest","ruff","mypy","sphinx","furo","build","twine","hypothesis"} ⊆` names in `dependency-groups["dev"]`.
  - `test_bench_group_has_comparison_libraries`: names in `dependency-groups["bench"]` equal `{"lmfit","jaxopt"}`.
  - `test_no_upper_caps`: no requirement string in dependencies, extras or groups contains `<` (ignoring environment markers after `;`).
- [ ] **Step 3: Run** `uv run pytest tests/packaging -v` → FAIL.
- [ ] **Step 4: Restructure `pyproject.toml`**: move every non-runtime requirement (testing, lint/type/hooks, docs, notebooks, build, PySide6/pyqtgraph) out of `dependencies`; add the `gui` extra; add `[dependency-groups]` with `dev = [...]` (moved requirements verbatim plus `"nlsq[gui]"`) and `bench = ["lmfit>=<latest>", "jaxopt>=<latest>"]`. Leave `gpu_cuda*` extras and `[tool.uv] conflicts` untouched.
- [ ] **Step 5: Lock and sync.** `uv lock`. A scratch-project check on uv 0.11.28 showed a group entry `"nlsq[gui]"` resolves and installs, so expect it to work. If `uv lock` rejects it here anyway, drop that entry and note in the commit body that CI/Makefile use `uv sync --extra gui` (Task 3 reads this). If it cannot resolve `jaxopt` against the project's JAX, remove `jaxopt` from `bench` **immediately**, adjust `test_bench_group_has_comparison_libraries` to `{"lmfit"}`, and record that the JAXopt adapter will report `unavailable` (Task 6). Then `uv sync --group bench` and `uv lock --check` → exit 0.
- [ ] **Step 6: Run** `uv run pytest tests/packaging tests/test_test_file_structure.py -v` → PASS.
- [ ] **Step 7: Commit** `build: split core, gui extra, and dev/bench dependency groups` (include `uv.lock`).

### Task 2: Actionable error when the GUI extra is missing

**Files:**
- Modify: `nlsq/gui_qt/__init__.py` (`run_desktop`), the `gui` CLI handler
- Test: `tests/cli/test_gui_missing_dependency.py`

**Interfaces:**
- Produces: `GUI_INSTALL_HINT = 'The NLSQ desktop app needs the GUI extra: pip install "nlsq[gui]"'` in `nlsq/gui_qt/__init__.py` (plain string; must not import PySide6), imported by the CLI handler.

- [ ] **Step 1: Write failing tests** (both `@pytest.mark.serial`, subprocess via `sys.executable -c` with `sys.modules["PySide6"] = None` set first):
  - `test_run_desktop_without_pyside6_exits_with_hint`: the subprocess code is `raise SystemExit(nlsq.gui_qt.run_desktop())` (`run_desktop` returns an `int`); exit code non-zero; stderr contains `nlsq[gui]`, not `Traceback`.
  - `test_cli_gui_command_without_pyside6_exits_with_hint`: subprocess code `raise SystemExit(nlsq.cli.main.main(["gui"]))` (`main` returns an `int` and `handle_gui` is `-> int`; neither calls `sys.exit` itself) under the same block; exit code non-zero with `nlsq[gui]` on stderr, no `Traceback`.
- [ ] **Step 2: Run** `uv run pytest tests/cli/test_gui_missing_dependency.py -n 0 -v` → FAIL.
- [ ] **Step 3: Implement.** In `run_desktop` (PySide6 is imported lazily near line 139) and `handle_gui` (`nlsq/cli/main.py:182`) import PySide6 inside `try/except ImportError`, write `GUI_INSTALL_HINT` to stderr and **return** 1 (or the CLI's existing error code); the entry points already turn the returned int into the exit status.
- [ ] **Step 4: Run** the same command → PASS; `uv run pytest tests/gui_qt -m "not slow" -n 0 -q` → PASS.
- [ ] **Step 5: Commit** `feat(gui): print install hint when the gui extra is missing`.

### Task 3: Update every install reference, CI and Makefile

**Files:**
- Modify: `Makefile` (`install-dev`, `install-all`), `.readthedocs.yaml`, `docs/requirements.txt`, `CONTRIBUTING.md`, `README.md` (line "GUI included"), `docs/tutorials/routine/getting_started/installation.rst`, `.github/workflows/*.yml` (add `--extra gui` to `uv sync` only in the Task 1 fallback case), `CHANGELOG.md`
- Test: `tests/packaging/test_dependency_layout.py` (two more tests)

**Interfaces:**
- Consumes: `_load_pyproject`, `_dist_name` (Task 1).

- [ ] **Step 1: Add failing tests:**
  - `test_install_references_name_existing_extras`: regex `(?:nlsq|\.)\[([A-Za-z][A-Za-z0-9_,.-]*)\]` (the extra must start with a letter, otherwise shell classes such as `[0-9]+\.[0-9]+` in `Makefile` and `release.yml` match as `0-9`) over `README.md`, `CONTRIBUTING.md`, `Makefile`, `.readthedocs.yaml`, `docs/requirements.txt`, `docs/tutorials/routine/getting_started/installation.rst`, `.github/workflows/*.yml`; every comma-separated extra, normalised with `_dist_name`, is among the normalised keys of `optional-dependencies`; `README.md` must contain `nlsq[gui]`.
  - `test_docs_requirements_cover_docs_packages`: `{"sphinx","furo","sphinx-copybutton","sphinx-autodoc-typehints","sphinx-design","myst-parser"}` ⊆ names listed in `docs/requirements.txt`, which also contains `.[gui]` (autodoc imports `nlsq.gui_qt`; `fail_on_warning: true` breaks without PySide6).
- [ ] **Step 2: Run** `uv run pytest tests/packaging -v` → the two new tests FAIL.
- [ ] **Step 3: Edit references.** `docs/requirements.txt`: the six docs packages (floors from `dev`) + `-e .[gui]`. `.readthedocs.yaml`: `python.install` = `requirements: docs/requirements.txt`, then `method: pip, path: .` with `extra_requirements: [gui]`; refresh the comment that says there is no extra. `Makefile`: `install` stays core-editable; `install-dev`/`install-all` run `uv sync --extra gui --group bench` when `PKG_MANAGER` is uv, else `pip install -e ".[gui]" --group dev --group bench` (pip ≥ 25.1; say so in the echo). README/installation.rst/CONTRIBUTING: `pip install nlsq` = core; `pip install "nlsq[gui]"` = with desktop app; contributors use `make dev`/`uv sync`. CHANGELOG `Unreleased` → **Changed (install footprint)**, with a migration line for GUI users.
- [ ] **Step 4: Run** `uv run pytest tests/packaging -v` → PASS; `make -n install-dev` prints the new commands.
- [ ] **Step 5: Commit** `docs(build): point install instructions at the gui extra and dependency groups`.

### Task 4: Core-only wheel smoke check in CI

**Files:**
- Create: `scripts/check_core_install.py`
- Modify: `.github/workflows/ci.yml` (the existing wheel smoke step in the `build` job: `uv venv .venv-test` / `VIRTUAL_ENV=.venv-test uv pip install dist/*.whl` / `VIRTUAL_ENV=.venv-test uv run python -c "import nlsq; …"`)

**Interfaces:**
- Produces: `main() -> int` in `scripts/check_core_install.py`; exit 0 only if (a) `importlib.util.find_spec` is `None` for `PySide6`, `pyqtgraph`, `pytest`, `ruff`, `sphinx`, `lmfit`, `jaxopt`; (b) `nlsq.curve_fit` on 200 seeded noisy points of `a*exp(-b*x)` recovers `a`, `b` within `1e-1` relative; (c) chunked path: 20 000 seeded points fitted by `nlsq.streaming.LargeDatasetFitter(config=LDMemoryConfig(min_chunk_size=1000, max_chunk_size=2000))` (`LDMemoryConfig` from `nlsq.streaming`, `large_dataset.py:275` — not `nlsq.config.MemoryConfig`, which the fitter does not accept) (`max(min_chunk_size, …)` at `large_dataset.py:654` floors the chunk size at 1000, and `curve_fit_large` sends fewer than 1 000 000 points to plain `curve_fit` unless `auto_size_detection=False`, so a tiny memory limit alone does not chunk); require `result["n_chunks"] >= 2` and finite `popt`; additionally call `curve_fit_large(..., auto_size_detection=False, chunk_size=2000)` and require a finite `popt`; (d) `result.plot` is callable with `MPLBACKEND=Agg` and does not raise.

- [ ] **Step 1: Write the script** with the contract above.
- [ ] **Step 2: Verify it fails in the full dev env:** `uv run python scripts/check_core_install.py` → exit 1 naming the importable packages.
- [ ] **Step 3: Verify it passes on a core-only wheel:** `uv build --wheel`, `uv venv $TMPDIR/core-venv`, `VIRTUAL_ENV=$TMPDIR/core-venv uv pip install dist/*.whl`, run the script with that interpreter → exit 0. If core code imports a non-core package, lazy-import it (or move the dependency back) and say so in the commit body.
- [ ] **Step 4: Wire into `ci.yml`.** The existing step runs through `uv run`, which ignores a `VIRTUAL_ENV` that differs from the project's `.venv` and so tests the dev environment, not the wheel. Replace it: after the wheel install, run `.venv-test/bin/python scripts/check_core_install.py` (the `build` job is ubuntu-only). Also check locally that running the script through `uv run` from the repo root fails (a) so the new step is not vacuous.
- [ ] **Step 5: Commit** `ci: smoke-test a core-only wheel install`.

### Task 5: Comparison harness core — problems, timing, accuracy gate

**Files:**
- Create: `benchmarks/compare/__init__.py`, `types.py`, `problems.py`, `timing.py`
- Test: `tests/benchmarks/test_compare_core.py`

**Interfaces:**
- Produces (`types.py`): `Status = Literal["ok","unsupported","unavailable","did_not_reach_optimum"]`; `@dataclass(frozen=True) FitOutcome(params: np.ndarray, converged: bool, nfev: int | None)`; `@dataclass(frozen=True) Timing(cold_s: float, warm_median_s: float, warm_iqr_s: float)`; `Outcomes = FitOutcome | list[FitOutcome]` (a single fit or a batch).
- Produces (`problems.py`): `@dataclass(frozen=True) Problem(name: str, p_true: tuple[float, ...], p0: tuple[float, ...], bounds: tuple[tuple[float, ...], tuple[float, ...]] | None, x_range: tuple[float, float], model: Callable[..., Any])` where `model(xp, x, *params)` takes the array module `xp` (`numpy` or `jax.numpy`); methods `data(self, n: int, seed: int = 0, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]` and `batch_data(self, n: int, batch: int, seed: int = 0, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]` (shared `x` of shape `(n,)`, `Y` of shape `(batch, n)` with independent noise draws); `PROBLEMS: dict[str, Problem]` with keys `exp_decay` (`a*exp(-b*x)+c`, truth `(2.5,1.3,0.5)`, p0 `(1,1,0)`, x∈[0,4]), `gaussian` (`a*exp(-(x-mu)**2/(2*s**2))`, truth `(3.0,1.0,0.5)`, p0 `(1,0,1)`, x∈[-3,5]), `two_exp` (`a1*exp(-k1*x)+a2*exp(-k2*x)`, truth `(2.0,1.0,1.5,1.5)`, p0 `(1,0.5,1,2)`, x∈[0,5]), `bounded_exp` (same model/truth/p0/x as `exp_decay`, bounds `((0,0,0),(2.0,3.0,1.0))` so the upper bound on `a` is active). Values are the author's call.
- Produces (`timing.py`): `time_fit(run: Callable[[], Outcomes], repeats: int = 7) -> tuple[Timing, Outcomes]` (first call = cold; then `repeats` warm calls; returns the last outcome; works unchanged for single and batch runs); `residual_cost(problem: Problem, x, y, params) -> float`; `case_cost(problem: Problem, x, Y, outcomes: Outcomes) -> float` (mean `residual_cost` over a batch, or the single cost); `all_converged(outcomes: Outcomes) -> bool`; `reaches_optimum(cost: float, best_cost: float, rtol: float = 1e-6) -> bool` (`cost <= best_cost + rtol * max(abs(best_cost), tiny)` with `tiny = 1e-300`, so a zero `best_cost` neither divides by zero nor rejects an exact fit); `cold_in_fresh_process(...)` is **not** a function here: cold timing in a fresh worker process is the runner's job (Task 7).

- [ ] **Step 1: Write failing tests** in `tests/benchmarks/test_compare_core.py` (import `nlsq` first so x64 is enabled):
  - `test_data_is_deterministic_for_seed`: two `data(1000, seed=0)` calls equal; `seed=1` differs.
  - `test_model_numpy_and_jax_agree`: for every problem, `max|model(np,x,*p_true) - model(jnp,x,*p_true)| < 1e-10`.
  - `test_bounded_problem_truth_violates_a_bound`: `p_true[0] > bounds[1][0]` for `bounded_exp`.
  - `test_batch_data_shapes_and_independent_noise`: shapes `(200,)`/`(4,200)`; rows differ.
  - `test_time_fit_calls_once_cold_then_repeats_warm`: a counting fake `run` is called `1 + repeats` times; all `Timing` fields ≥ 0.
  - `test_reaches_optimum_gate`: `reaches_optimum(1.0 + 1e-9, 1.0)` is True; `reaches_optimum(1.0 + 1e-3, 1.0)` is False; `reaches_optimum(0.0, 0.0)` is True and `reaches_optimum(1e-3, 0.0)` is False (no `ZeroDivisionError`).
  - `test_case_cost_averages_a_batch_and_all_converged_requires_every_fit`: a fake batch of two `FitOutcome`s at the truth, one with `converged=False` → `case_cost` equals the single cost and `all_converged` is False.
  - `test_residual_cost_is_zero_at_noiseless_truth`: `residual_cost` of noiseless data at `p_true` is `< 1e-20`.
- [ ] **Step 2: Run** `uv run pytest tests/benchmarks/test_compare_core.py -v` → FAIL (modules missing).
- [ ] **Step 3: Implement** the three modules per the interfaces. `time_fit` uses `time.perf_counter`; warm IQR from `np.percentile(…, [25, 75])`.
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit** `feat(bench): add comparison problems, timing and accuracy gate`.

### Task 6: Library adapters and the feasibility gate

**Files:**
- Create: `benchmarks/compare/adapters.py`
- Test: `tests/benchmarks/test_compare_adapters.py`

**Interfaces:**
- Consumes: `Problem`, `PROBLEMS`, `FitOutcome` (Task 5).
- Produces: `TOL = 1e-8`, `MAX_ITER = 1000`; `@dataclass(frozen=True) Capabilities(bounds: bool, sigma_weights: bool, covariance: bool, vmappable: bool, maintained: bool, note: str)`; `@dataclass(frozen=True) Availability(ok: bool, reason: str, version: str | None)`; `class Adapter(Protocol)` with `name: str`, `capabilities: Capabilities`, `modes: tuple[str, ...]` (`("end_to_end",)` or `("end_to_end", "jitted_core")`), `availability() -> Availability`, `supports(problem: Problem) -> tuple[bool, str]` (always the pair `(supported, reason)`; reason is `""` when supported), `prepare(problem: Problem, x: np.ndarray, y: np.ndarray, mode: str) -> Callable[[], FitOutcome]`, `prepare_batch(problem: Problem, X: np.ndarray, Y: np.ndarray, mode: str) -> Callable[[], list[FitOutcome]]`; `ADAPTERS: dict[str, Adapter]` with keys `nlsq`, `scipy`, `lmfit`, `jaxopt`, `optimistix`.

Call conventions (spec section B table; re-check each against the installed version): NLSQ `nlsq.curve_fit`; SciPy `curve_fit(method="trf")`; lmfit `Model(f).fit(y, params, x=x, method="leastsq", fit_kws={ftol,xtol,gtol})` (`"least_squares"` for the bounded problem); JAXopt `LevenbergMarquardt(residual_fun, tol=TOL, maxiter=MAX_ITER).run(p0, x, y)`; Optimistix `least_squares(fn, LevenbergMarquardt(rtol=TOL, atol=TOL), y0, args=(x, y), max_steps=MAX_ITER, throw=False)`, converged iff `sol.result == optx.RESULTS.successful`. Optimistix and JAXopt `supports()` returns `(False, "no bounds support")` for problems with bounds. `maintained=False` for JAXopt with note quoting its README.

- [ ] **Step 1: Feasibility gate (before writing adapters).** `uv sync --group bench`. For each of lmfit, JAXopt, Optimistix write a throwaway call that fits `exp_decay` n=1000 and prints version, params, wall time. Paste the three outcomes into the Task 6 commit body. A failure (e.g., JAXopt import error under JAX 0.12.x) is not fixed: it becomes that adapter's `availability()` result.
- [ ] **Step 2: Write failing tests** in `tests/benchmarks/test_compare_adapters.py`:
  - `test_registry_has_the_five_libraries`: `set(ADAPTERS) == {"nlsq","scipy","lmfit","jaxopt","optimistix"}`.
  - `test_every_adapter_reports_availability_with_reason`: `ok` implies non-empty `version`; `not ok` implies non-empty `reason`.
  - `test_available_adapters_recover_truth_on_exp_decay`: for each adapter and each of its `modes`, if `availability().ok`: fit `exp_decay` n=1000, `converged` True and max relative parameter error vs truth `< 0.05`; otherwise assert the reason is non-empty (no `skip`).
  - `test_unbounded_libraries_decline_the_bounded_problem`: `supports(PROBLEMS["bounded_exp"])` is `(False, <reason containing "bounds">)` for `jaxopt` and `optimistix`, and `(True, "")` for `nlsq`, `scipy`, `lmfit`.
  - `test_bounded_fit_respects_bounds`: every available bounds-capable adapter returns params inside the bounds (tolerance `1e-9`).
  - `test_batch_cost_matches_single_fit_cost`: for each available adapter, a batch of 4 fits of `exp_decay` (n=200) gives per-fit cost within `1e-6` relative of the same data fitted singly.
  - `test_jaxopt_capability_records_unmaintained`: `ADAPTERS["jaxopt"].capabilities.maintained is False`.
- [ ] **Step 3: Run** `uv run pytest tests/benchmarks/test_compare_adapters.py -v` → FAIL.
- [ ] **Step 4: Implement** `adapters.py`. `availability()` imports the library and does a tiny fit (`exp_decay`, n=50) in `try/except Exception`, cached with `functools.cache`; on failure `reason` is `f"{type(e).__name__}: {e}"`. JAX adapters `jit` once per `prepare`; `jitted_core` places data on device first and ends each call with `jax.block_until_ready`; `end_to_end` converts NumPy in and params to NumPy out. `prepare_batch` uses `jax.vmap` for JAXopt/Optimistix and a Python loop for the rest. Leave `nfev=None` where the library does not expose it.
- [ ] **Step 5: Run** the same command → PASS.
- [ ] **Step 6: Commit** `feat(bench): add SciPy, lmfit, JAXopt and Optimistix adapters`.

### Task 7: Runner, report, first committed CPU baseline

**Files:**
- Create: `benchmarks/compare/report.py`, `benchmarks/compare/run_library_comparison.py`, `benchmarks/baselines/2026-10-30-library-comparison.md`
- Modify: `benchmarks/README.md`, `CHANGELOG.md`
- Test: `tests/benchmarks/test_compare_runner.py`

**Interfaces:**
- Consumes: Tasks 5–6.
- Produces (`report.py`): `RECORD_FIELDS: tuple[str, ...] = ("library","mode","problem","n","batch","backend","device","status","reason","converged","cold_s","warm_median_s","warm_iqr_s","nfev","cost","cost_gap","max_param_error_vs_truth")`; `environment() -> dict` (versions via `importlib.metadata.version` — JAXopt has no `__version__` — of numpy, scipy, jax, jaxlib, nlsq, lmfit, jaxopt, optimistix; `jax.default_backend()`; device kinds; CPU model; `os.cpu_count()`); `write_json(records: list[dict], path: Path) -> None`; `rank_case(records: list[dict]) -> list[dict]` (only `status == "ok"`, sorted ascending by `warm_median_s`); `to_markdown(records: list[dict], env: dict) -> str` (per-case ranking tables, a capability matrix, an unavailable/unsupported section with reasons, package versions, and the caveats: criteria analogous not identical; lmfit wraps SciPy; JAXopt "no longer maintained nor developed"; GPU not measured unless `backend != "cpu"`).
- Produces (`run_library_comparison.py`): `run(problems: Sequence[str], sizes: Sequence[int], repeats: int, include_batch: bool, libraries: Sequence[str] | None = None) -> list[dict]`; a `--worker` mode (hidden flag) that, given one (library, mode, problem, n, batch) in argv, imports only what it needs, does a trivial backend warm-up (a tiny `jnp` op, so device initialisation is not charged to a library), runs `time_fit`, and prints one JSON record on stdout; `run` launches one such subprocess per case so **cold = first call in a fresh process** (spec section B), and the persistent-cache variables are set in the subprocess environment; `main(argv: Sequence[str] | None = None) -> int` with flags `--quick` (n=200, one repeat), `--problems`, `--sizes`, `--repeats`, `--no-batch`, `--libraries`, `--out DIR` (default `benchmarks/results`). `main` sets `NLSQ_DISABLE_PERSISTENT_CACHE=1` before importing adapters. The batch scenario is `batch_small`: 256 fits × 200 points on `exp_decay`.

- [ ] **Step 1: Write failing tests** in `tests/benchmarks/test_compare_runner.py`:
  - `test_rank_case_excludes_early_stoppers`: three records, the fastest with `status="did_not_reach_optimum"` → `rank_case` omits it and returns the others in ascending `warm_median_s`.
  - `test_markdown_lists_unavailable_library_with_reason`: a record `status="unavailable", reason="ImportError: boom"` → `to_markdown` output contains `boom` and the library name.
  - `test_markdown_carries_required_caveats`: output contains `no longer maintained`, `wraps SciPy`, and the backend string from `env`.
  - `test_quick_run_writes_valid_json_and_markdown` (`@pytest.mark.serial`): `main(["--quick","--no-batch","--out",str(tmp_path)])` returns 0; the JSON is a non-empty list; every record has exactly `RECORD_FIELDS`; every `status` is in the `Status` literal; the Markdown file names all five libraries.
- [ ] **Step 2: Run** `uv run pytest tests/benchmarks/test_compare_runner.py -n 0 -v` → FAIL.
- [ ] **Step 3: Implement** both modules. For each (problem, n): iterate adapters → one record per mode, or one `unavailable`/`unsupported` record with the reason; compute `cost` with `case_cost` (so batches use the mean per-fit cost); set `best_cost` = minimum across that case; `status` = `ok` if `reaches_optimum(cost, best_cost)` else `did_not_reach_optimum` (the library's own `converged`, via `all_converged`, is recorded but does not decide status); `cost_gap = (cost - best_cost)/max(abs(best_cost), 1e-300)`. A library raising during `prepare`/`time_fit` is caught and recorded as `unavailable` with the exception text.
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Run the real baseline:** `uv run python -m benchmarks.compare.run_library_comparison --sizes 1000 100000 1000000 --repeats 7 --out benchmarks/results` (CPU; run in background; expect minutes). Render the Markdown report to `benchmarks/baselines/2026-10-30-library-comparison.md`. Check by eye that every library is either measured or listed with a reason.
- [ ] **Step 6: Docs.** `benchmarks/README.md`: add a "Library comparison" section with the command, the fairness rules in one paragraph, and the GPU recipe `JAX_PLATFORMS=cuda uv run python -m benchmarks.compare.run_library_comparison --out …`. CHANGELOG `Unreleased` → **Added**: harness and baseline report.
- [ ] **Step 7: Commit** `feat(bench): add four-baseline library comparison and first CPU report`.

### Task 8: Small-fit overhead — measure with the harness, then remove avoidable cost

**Files:**
- Create: `tests/core/test_default_logging.py`, `benchmarks/small_fit_overhead.py`
- Modify: whichever modules the profile implicates (expected among: `nlsq/core/minpack.py`, `nlsq/core/least_squares.py`, `nlsq/utils/logging.py`, `nlsq/utils/validators.py`, `nlsq/stability/guard.py`), `CHANGELOG.md`

**Interfaces:**
- Produces: `benchmarks/small_fit_overhead.py` runnable as `uv run python benchmarks/small_fit_overhead.py`; prints a `cProfile` top-15 table (cumulative) for one warm NLSQ `exp_decay` fit at n=1000, using `PROBLEMS["exp_decay"]` and `ADAPTERS["nlsq"].prepare`. Timing comparisons come from the harness, not this script.

- [ ] **Step 1: Write `tests/core/test_default_logging.py`** (`@pytest.mark.serial`). Observed 2026-10-30 with `NLSQ_DEBUG` and `NLSQ_VERBOSE` unset: after `import nlsq` the root logger holds a stderr `StreamHandler` (its installer is not `nlsq/utils/logging.py`'s console handler), the `nlsq.*` loggers sit at INFO with `propagate=True`, and a fit prints about 11 `INFO:name:msg` / `PERFORMANCE:name:msg` lines to stderr. Tests run a subprocess with both variables removed from its environment: `test_default_fit_writes_no_log_lines` fits `exp_decay` once and asserts **neither stdout nor stderr** has a line starting with `INFO:`, `PERFORMANCE:` or `DEBUG:` (the formatted prefix is `LEVEL:name:`; also assert on the substring `nlsq.` plus a level word, to cover a differently formatted handler); `test_nlsq_debug_enables_log_lines` sets `NLSQ_DEBUG=1` and asserts at least one NLSQ log line appears on stdout or stderr.
- [ ] **Step 2: Find the source first.** Before changing anything, install a spy on `logging.Logger.addHandler` for the root logger, `import nlsq` and print the stack (the logging-trace helper from the review is a ten-line script), and `grep -n "setLevel\|basicConfig\|NLSQ_DEBUG" nlsq/utils/logging.py nlsq/__init__.py`. Record the culprit in the commit body.
- [ ] **Step 3: Run** `uv run pytest tests/core/test_default_logging.py -n 0 -v` → the first test FAILS. Fix at the source found in Step 2: the default level of the `nlsq` logger tree must be WARNING unless `NLSQ_DEBUG=1` (or the existing verbosity setting) raises it; do not remove the root handler of a dependency and do not change `propagate`. Re-run until both pass.
- [ ] **Step 4: "Before" measurement** (taken **after** the logging fix, so the profile starts from the fixed default; also record the pre-fix number once from the Task 7 baseline for the changelog). `uv run python -m benchmarks.compare.run_library_comparison --problems exp_decay --sizes 1000 100000 --repeats 7 --no-batch --out benchmarks/results`; keep the Markdown table for the commit body.
- [ ] **Step 5: Write the profile script** and run it; identify the top three avoidable cost centers (spec section C4's candidates are guesses: trust the profile).
- [ ] **Step 6: Remove them, one per commit.** Each fix keeps validation of shape, dtype and NaN at the trust boundary (do it once, not repeatedly) and every safety check's behavior. After each: `uv run pytest tests/core tests/stability -m "not slow" -n 2 -q` → PASS including `test_curvefit_golden_master.py`. Commit `perf(core): <what was removed>`.
- [ ] **Step 7: "After" measurement** with the Step 4 command. Acceptance: NLSQ warm n=1000 ≤ 34 ms **or** a CHANGELOG paragraph states the measured floor and why further reduction was not safe. Add both tables, with NLSQ's warm-time ratio to each available baseline, to CHANGELOG `Unreleased` → **Performance**.
- [ ] **Step 8: Commit** `docs(changelog): record small-fit overhead results`.

### Task 9: Remove orphaned code; deprecate exported profilers

**Files:**
- Delete: `nlsq/facades/`, `tests/facades/`, `docs/reference/facades.rst`, `docs/api/nlsq.facades.rst`, `nlsq/utils/async_logger.py`, and **only the async-logger tests** (`grep -rl async_logger tests`; they live in `tests/core/test_host_device_transfers.py`, which also holds unrelated tests — delete those test functions, not the file)
- Modify: `tests/benchmarks/test_decomposition_memory.py` (drop the `OptimizationFacade` use and delete `test_lazy_import_reduces_baseline_memory`), `README.md` (the facade mention near line 243), `docs/reference/index.rst`, `docs/api/modules.rst`, `docs/advanced_guide.rst`, `docs/developer/architecture.rst`, `docs/tutorials/advanced/{index.rst,architecture/index.rst,architecture/overview.rst}`, `nlsq/utils/__init__.py` (docstring line for `async_logger`), `nlsq/__init__.py`, `CHANGELOG.md`
- Test: `tests/utils/test_profiler_deprecation.py`

**Interfaces:**
- Produces: in `nlsq/__init__.py`, `_DEPRECATED_EXPORTS: dict[str, str]` mapping each of `PerformanceProfiler`, `ProfileMetrics`, `clear_profiling_data`, `get_global_profiler`, `ProfilerVisualization`, `ProfilingDashboard` to `"nlsq.<name> is deprecated and will be removed in the next minor release"`, and `_deprecation_seen: set[str]`; the module `__getattr__` warns once per name via `warnings.warn(..., DeprecationWarning, stacklevel=2)`, then resolves as before.

- [ ] **Step 1: Confirm nothing internal uses the names via the top-level package:** `grep -rn "from nlsq import.*\(Profiler\|Profil\|profiling_data\)" nlsq` → empty; `grep -rn "async_logger" nlsq` → only the file itself and the `utils/__init__.py` docstring.
- [ ] **Step 2: Write failing tests** in `tests/utils/test_profiler_deprecation.py`:
  - `test_deprecated_names_warn_once` (parametrised over the six names; a fixture clears `nlsq._deprecation_seen`): first `getattr(nlsq, name)` under `pytest.warns(DeprecationWarning, match="next minor release")`; the second access records zero `DeprecationWarning`.
  - `test_import_nlsq_emits_no_deprecation_warning` (`@pytest.mark.serial`): subprocess `python -W error::DeprecationWarning -c "import nlsq"` exits 0.
  - `test_facades_package_is_gone`: `importlib.util.find_spec("nlsq.facades") is None`.
- [ ] **Step 3: Run** `uv run pytest tests/utils/test_profiler_deprecation.py -n 0 -v` → FAIL.
- [ ] **Step 4: Implement** the warn-once logic, then delete the files above and fix the doc references; `git grep -n -i "facade" -- . ':!CHANGELOG.md' ':!docs/CHANGELOG.md' ':!graphify-out' ':!docs/superpowers' ':!three-brain-out' ':!tests/utils/test_profiler_deprecation.py'` must be empty (that test file names `nlsq.facades` on purpose).
- [ ] **Step 5: Run** `uv run pytest tests/utils tests/architecture tests/test_test_file_structure.py tests/benchmarks -m "not slow" -n 2 -q` → PASS; `uv run sphinx-build -W --keep-going -q docs $TMPDIR/docs-out` → exit 0.
- [ ] **Step 6: CHANGELOG** `Unreleased` → **Removed** (`nlsq.facades`, `nlsq.utils.async_logger`) and **Deprecated** (the six names).
- [ ] **Step 7: Commit** `refactor: remove unused facades and async logger; deprecate profilers` (two commits if the diff exceeds ~300 lines excluding deletions).

### Task 10: Flag chunked fits that are not full-batch-equivalent

**Files:**
- Modify: `nlsq/streaming/large_dataset.py` (`_fit_implementation`, line 1632, which contains the single-chunk branch at ~1716 and every multi-chunk return: success via `_finalize_chunked_results` at ~2467, low-success-rate result at ~2567, exception result at ~2846; **not** the `"processing_strategy"` dict near line 2888, which is a memory-recommendation return), `CHANGELOG.md`
- Test: `tests/streaming/test_large_dataset_equivalence_flag.py`

**Interfaces:**
- Produces: result key `"equivalent_to_full_batch": bool` on every result `_fit_implementation` returns — success, low-success-rate failure and exception failure alike — `True` iff exactly one chunk was used; one `logger.warning` containing `not equivalent to a full-batch fit` per multi-chunk fit, success or failure. Implement it once (a small helper that annotates the result and warns, called at each return), not per branch.

- [ ] **Step 1: Write failing tests.** Reuse the chunk-forcing pattern from existing `tests/streaming/test_large_dataset*.py`:
  - `test_single_chunk_fit_is_equivalent`: small linear dataset, large memory limit → `result["equivalent_to_full_batch"] is True`; `caplog` has no record containing `not equivalent to a full-batch fit`.
  - `test_multi_chunk_fit_is_flagged`: same data forced into ≥ 3 chunks → `False`, and exactly one such warning record.
  - `test_failed_multi_chunk_fit_is_flagged`: force the low-success-rate path (a model that raises or returns NaN on most chunks) → `success` is False, the key is `False`, and one warning record.
- [ ] **Step 2: Run** `uv run pytest tests/streaming/test_large_dataset_equivalence_flag.py -n 0 -v` → FAIL.
- [ ] **Step 3: Implement** in `large_dataset.py`; no numeric path changes. First `grep -rn "keys()\|== {" tests/streaming` for exact result-key assertions and note which will need the new key.
- [ ] **Step 4: Run** `uv run pytest tests/streaming -m "not slow" -n 2 -q` → PASS (update existing tests that assert exact result-dict keys for the new key; do not loosen others).
- [ ] **Step 5: CHANGELOG** `Unreleased` → **Added**: the key and warning; cite the docstring at `large_dataset.py:1946`.
- [ ] **Step 6: Commit** `feat(streaming): flag multi-chunk fits as not full-batch-equivalent`.

### Task 11: Whole-branch verification

- [ ] **Step 1:** `HOME=$TMPDIR/fakehome NLSQ_SKIP_GPU_CHECK=1 make verify` → pre-commit, mypy, bandit and `pytest -m "not slow"` all clean (≈ 40 min for pytest; run in background).
- [ ] **Step 2:** `uv lock --check` → exit 0; `git diff main -- pyproject.toml | grep '^+.*<'` shows no new upper caps.
- [ ] **Step 3:** Re-run the Task 4 core-wheel check and `uv run python -m benchmarks.compare.run_library_comparison --quick --no-batch`; paste both outputs into the PR description with the Task 8 before/after tables.
- [ ] **Step 4:** Stop. Opening the PR is the requester's call.

---

## Self-review notes

- Spec coverage: Goal 1 → Tasks 1–4; Goal 2 (four-baseline comparison) → Tasks 5–7; Goal 3 (overhead, with per-baseline ratios) → Task 8; Goal 4 → Task 9; Goal 5 → Task 10. Fairness rules (common tolerances, iteration cap, accuracy gate, cold/warm, two JAX modes, scenarios, batch case, output fields, feasibility gate) map to Tasks 5–7. Anti-goals: dangling references → Task 3; upper caps → Tasks 1/11; bare traceback → Task 2; early-stopper/looser-tolerance ranking → Tasks 5, 7; silent dropout → Tasks 6, 7; JAXopt maintenance status and lmfit-wraps-SciPy caveats → Tasks 6, 7 tests. Deferred items have no tasks by design; Task 7's GPU recipe makes their reopen criteria checkable.
- Types: `FitOutcome`, `Timing`, `Problem`, `PROBLEMS` (Task 5) are consumed unchanged in Tasks 6–8; `RECORD_FIELDS` equals the spec's field list; `Status` literal matches the spec's four statuses; `_DEPRECATED_EXPORTS`/`_deprecation_seen` (Task 9) and `"equivalent_to_full_batch"` (Task 10) are consistent within their tasks.
- Review Focus lines each map to a task's tests (2, 4, 6/7, 5/7, 8).
- Known open mechanics the builder settles: whether `dev` can self-reference `nlsq[gui]` and whether `jaxopt` locks (Task 1 Step 5); each baseline's exact keyword spelling against its installed version (Task 6); the Task 8 fixes (profile-driven).
