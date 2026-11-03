# Large-fit strategy seam Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the large-fit path a typed first-class choice (`strategy=`), put `exact` behind the same seam as hybrid and chunked, and remove the fabricated `pcov`. (The chunked path's silent drop of `stability` and multistart is kept and documented; see the spec.)

**Architecture:** Slice 1 (Tasks 1-4) changes internals only: an `exact` adapter in `nlsq/streaming/strategy.py`, per-adapter `unpack`, a shared p0/bounds helper, and documenting the chunked limitation. Slice 2 (Tasks 5-7) adds frozen strategy dataclasses, moves the four path bodies out of `curve_fit_large` into one module, and adds a `strategy=` keyword with a legacy-keyword shim. Task 8 closes out docs and verification.

**Tech Stack:** Python >=3.12, JAX, NumPy, pytest (+xdist), ruff, mypy. `uv run` for commands.

**Spec:** `docs/superpowers/specs/2026-11-03-large-fit-strategy-design.md` (read it first; decisions and "My calls" there are binding).

## Global Constraints

- `curve_fit_large(f, xdata, ydata, p0, ...)` keeps its signature (plus the new `strategy=` keyword; `ydata` stays a required positional). Chunked, hybrid and exact return `(popt, pcov)`; the in-memory path returns `curve_fit`'s native result unchanged (`CurveFitResult`, or the longer plain tuple of `timeit`/`return_eval`/`full_output`). `test_large_strategy_routing.py:728` pins the `CurveFitResult`.
- The chunked path keeps ignoring `stability`, `rescale_data`, `max_jacobian_elements_for_svd` and multistart (chunk fits run `CurveFit.curve_fit`, where those keys are inert `**kwargs`). No task may change this. The chunked adapter keeps forwarding `multistart, n_starts, sampler, center_on_p0, scale_factor` exactly as today (`test_large_strategy_routing.py:609-617, 662-673, 711-719` pin them).
- Legacy-keyword error messages stay word for word (`ValueError`/`TypeError` text pinned by `tests/streaming/test_exact_lm_dispatch.py`, `test_exact_lm_sources.py`, `test_large_strategy_routing.py`). `nlsq._EXACT_DEFAULTS` must stay importable from `nlsq` (asserted at `test_exact_lm_dispatch.py:219`).
- `Strategy.fit` returns the engine's native result unchanged; unpacking is a separate `unpack` step. Engine classes/functions are looked up when a strategy is built (not at import), so `mock.patch("nlsq.streaming.<module>.<Class>")` targets keep working.
- Exact must never call `np.asarray` on `xdata`/`ydata`/`sigma` (memmap, h5py, zarr, chunk factory).
- No new warnings: `make verify` stays at zero warnings. `import nlsq` time stays ~620 ms; engine imports stay inside functions.
- No `DeprecationWarning` on legacy keywords in this plan. Strategy types are exported from `nlsq.streaming`, not from top-level `nlsq`.
- Tests: run with `uv run pytest <path> -n 0 -q`. If custom-model tests fail on a read-only HOME, use `HOME=$TMPDIR/fakehome`. Any test with 100K+ arrays or subprocesses gets `@pytest.mark.serial`.
- Commits: GPG signing can fail transiently; retry a few times before escalating. Never bypass signing.
- Line numbers below are from spec time; re-check before editing.

## Review Focus

Failure modes the spec implies; each has a pinning test in the owning task.

1. `strategy=` given together with a non-default legacy option, `global_search=True`, or any `**kwargs` must raise `TypeError` naming the option, not silently pick one (Task 7).
2. Size routing edges: `n == size_threshold` goes chunked, `size_threshold - 1` goes in-memory, `auto_size_detection=False` goes chunked even for tiny data (Task 5).
3. Exact must not materialize lazy sources; a chunk factory with any non-`Exact` strategy raises `ValueError`; 2-D `xdata` with `Exact` raises `ValueError` (Tasks 2 and 6).
4. A chunked result with no `pcov` raises `RuntimeError`; a hybrid result with no `pcov` still gets the inf-filled `(p, p)` (Task 3).
5. Legacy `method="multi-start"` with `multistart=False, n_starts=3, sampler="sobol"` must still forward the sampling values to the in-memory `curve_fit` (Task 7); exact with `multistart=False` and non-default sub-options must still be accepted; exact with a complex list `p0` must still raise `TypeError` (Tasks 2 and 7).

---

## Slice 1: internals, no public change

### Task 1: `exact` adapter in `build_large_strategy`

**Files:**
- Modify: `nlsq/streaming/strategy.py`
- Test: `tests/streaming/test_strategy.py`

**Interfaces:**
- Consumes: `nlsq.streaming.exact_lm.exact_lm_fit(model, x, y, p0, *, sigma, bounds, chunk_size, ftol, xtol, gtol, max_nfev, check_finite, absolute_sigma, memory_limit_gb, show_progress, n_points) -> OptimizeResult`.
- Produces: `build_large_strategy(kind: Literal["hybrid", "chunked", "exact"], config: Any, **common: Any) -> Strategy` (Task 6 adds `"in_memory"`); for `"exact"`, `config` must be `None` and `common` empty, and `.fit(f, x, y, p0, /, **common)` forwards `common` to `exact_lm_fit` and returns its `OptimizeResult` unchanged.

- [ ] **Step 1: Create the branch and record the baseline.** `git switch -c refactor/large-fit-strategy`. Run `uv run pytest tests/streaming/test_strategy.py tests/streaming/test_exact_lm_dispatch.py tests/streaming/test_exact_lm_sources.py tests/streaming/test_large_strategy_routing.py -n 0 -q`; write the pass count in the task notes. Expected: all pass.
- [ ] **Step 2: Write failing tests** in `tests/streaming/test_strategy.py`:
  - `test_exact_build_looks_up_engine_at_build_time`: `monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", fake)` *before* `build_large_strategy("exact", None)`; `strategy.fit(f, x, y, p0, chunk_size=10)` returns the fake's return object (`is`) and the fake received `{"chunk_size": 10}` as keywords.
  - `test_exact_build_rejects_config`: `build_large_strategy("exact", object())` raises `TypeError`.
  - `test_exact_build_rejects_common_kwargs`: `build_large_strategy("exact", None, ftol=1e-3)` raises `TypeError`.
  - Update `test_unknown_kind_raises_value_error` (`test_strategy.py:131-133`): change its argument from `"exact"` (now valid) to `"bogus"`, keep `match="kind"`, and make the error message in `strategy.py` list all kinds.
- [ ] **Step 3: Run them; expect FAIL** (`ValueError: unknown large-strategy kind 'exact'`).
- [ ] **Step 4: Implement `_ExactStrategy`** in `strategy.py` (attribute `engine` holds the function, resolved via `from nlsq.streaming import exact_lm` inside `__init__`), extend the `Literal`, the dispatcher and the error message, and update the module/`build_large_strategy` docstrings.
- [ ] **Step 5: Run the Step 1 command; expect all pass.**
- [ ] **Step 6: Commit** (`feat(streaming): exact adapter behind build_large_strategy`).

### Task 2: shared p0/bounds helper; exact validation moves behind the seam; delete `_curve_fit_exact_streaming`

**Files:**
- Modify: `nlsq/streaming/strategy.py`, `nlsq/__init__.py` (`_default_p0` at ~721, `_curve_fit_exact_streaming` at ~742-849, hybrid branch at ~1107-1167, exact branch at ~1055-1077)
- Test: `tests/streaming/test_exact_lm_dispatch.py`, `tests/streaming/test_exact_lm_sources.py` (run unchanged), new `tests/streaming/test_strategy.py` cases

**Interfaces:**
- Produces in `strategy.py`:
  - `default_p0(f: Callable[..., Any]) -> np.ndarray` (the existing `_default_p0` logic, same `ValueError` text).
  - `normalize_p0_bounds(f, p0: ArrayLike | None, bounds: Any) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]`: `default_p0(f)` if `p0 is None` else `np.atleast_1d(p0)` (no float cast), then `nlsq.core.least_squares.prepare_bounds(bounds, len(p0_arr))`. Used by hybrid and exact only (spec "My calls").
  - `check_exact_sources(xdata, ydata, sigma, n_points) -> None`: the factory rules, 1-D shape checks, equal-length and `>= 2` checks currently at `__init__.py:782-824`, same messages, shape/len only. Raises on `n_points` with a non-factory source.
- Produces in `nlsq/__init__.py`: `_check_exact_legacy_options(unsupported: dict[str, Any], multistart: bool, solver_kwargs: dict[str, Any]) -> None` holding the `_EXACT_SOLVER_KWARGS` `TypeError`, the `_EXACT_DEFAULTS` `ValueError`s and the `multistart` `ValueError`, in that order, same messages. Only `multistart=True` is an error; `n_starts`/`sampler`/`center_on_p0`/`scale_factor` with `multistart=False` stay accepted and ignored (`test_exact_lm_dispatch.py:222-234`). `_EXACT_DEFAULTS`, `_EXACT_SOLVER_KWARGS`, `_NOT_EXACT` stay in `nlsq/__init__.py`.

- [ ] **Step 1: Write failing tests** in `test_strategy.py`: `test_check_exact_sources_rejects_2d_xdata` (`ValueError` matching `must be 1-D for method='exact_streaming'`), `test_check_exact_sources_never_converts_lazy_source` (a small class with `shape`, `__len__`, `__getitem__` whose `__array__` raises `AssertionError`; check passes), `test_normalize_p0_bounds_defaults_p0_from_signature` (`f(x, a, b)` with `p0=None` gives `np.ones(2)`).
- [ ] **Step 2: Run them; expect FAIL** (names not defined).
- [ ] **Step 3: Implement `default_p0`, `normalize_p0_bounds`, `check_exact_sources`** in `strategy.py`; implement `_check_exact_legacy_options` in `nlsq/__init__.py`.
- [ ] **Step 4: Rewrite the exact and hybrid branches of `curve_fit_large`** to use them. Exact branch order must stay: `_check_exact_legacy_options`, `check_exact_sources`, then p0: if given, `np.asarray(p0, dtype=float)` **before** `normalize_p0_bounds` (a complex *list* `p0` must still raise `TypeError`; a complex *ndarray* `p0` keeps today's cast-with-`ComplexWarning` behavior; add `test_exact_complex_list_p0_raises_type_error` to `test_exact_lm_dispatch.py`, using a list, and do not pin the ndarray case), `normalize_p0_bounds`, then `exact_lm.check_bounds`, then `build_large_strategy("exact", None).fit(...)` with the same keywords the old function passed to `exact_lm_fit`. Hybrid keeps its own infinite-bounds-to-`None` step. Delete `_curve_fit_exact_streaming` and `_default_p0` (grep showed no other users; confirm). The deleted helper also owned result handling (`ExactSolverError` on failure, `(x, pcov)` extraction, `__init__.py:~844`): keep that block inline in the exact branch of `curve_fit_large` for this task so the unchanged dispatch suites stay green; Task 3 moves it into `unpack`.
- [ ] **Step 5: Run** `uv run pytest tests/streaming tests/core/test_minpack_coverage.py tests/core/test_core_review_2026_10_29.py -n 0 -q`; expect the Task 1 baseline plus the new tests passing, no other change.
- [ ] **Step 6: Commit** (`refactor(streaming): exact validation and p0/bounds helper behind the strategy seam`).

### Task 3: per-adapter `unpack`; remove the `np.eye` fallback

**Files:**
- Modify: `nlsq/streaming/strategy.py`, `nlsq/__init__.py` (result handling at ~1163-1167, ~1306-1317, exact return)
- Test: `tests/streaming/test_strategy.py`, `tests/streaming/test_large_strategy_routing.py` (~line 681-684 and its driver/table row)

**Interfaces:**
- Produces: `Strategy.unpack(self, native: Any, /) -> Any` on the Protocol and all adapters. Hybrid, chunked and exact return `tuple[np.ndarray, np.ndarray]`; the `in_memory` adapter (Task 6) returns the native result unchanged, so the Protocol return type is `Any`.
  - Hybrid: `(native["x"], native.get("pcov", np.full((p, p), np.inf)))`, `p = len(native["x"])` (unchanged behavior).
  - Chunked: `(native.popt, native.pcov)` when both attributes exist; otherwise `RuntimeError` naming the result.
  - Exact: `ExactSolverError(f"exact_streaming fit failed ({native['termination']}): {native['message']}", native)` when `not native["success"]`, else `(native["x"], native["pcov"])`.

- [ ] **Step 1: Write failing tests** in `test_strategy.py`: `test_chunked_unpack_without_pcov_raises_runtime_error` (object with only `.x`), `test_hybrid_unpack_without_pcov_returns_inf_matrix` (shape `(2, 2)`, `np.isinf` everywhere), `test_exact_unpack_raises_exact_solver_error_with_result` (`excinfo.value.result is native`; message starts `exact_streaming fit failed (stalled)`), `test_exact_unpack_returns_x_and_pcov_on_success`.
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement `unpack`** on the three adapters and the Protocol. Replace the three hand-written result-handling blocks in `curve_fit_large` with `large_strategy.unpack(native)`. Do not touch the other callers of `build_large_strategy` (`fit()` and `core/minpack.py`); they keep their own unpacking.
- [ ] **Step 4: Flip the one pinned assertion.** `_check_clf_chunked_x_only` (~681) expects `np.eye(2)`. `test_routing_table` calls `row.drive(ctx)` before `row.check`, so an exception raised by the driver never reaches the check. Change that row's driver to catch `RuntimeError` and return the exception object, and have the check assert `isinstance(result, RuntimeError)` (leave other rows and the harness alone). Read `Row`, `_drive_clf_chunked` and the table loop (~1728-1736, ~2088) first.
- [ ] **Step 5: Run** `uv run pytest tests/streaming tests/core -n 0 -q`; expect all pass. `tests/streaming/test_large_dataset.py:879` must still pass untouched.
- [ ] **Step 6: Commit** (`fix(streaming): unpack per adapter; no fabricated pcov on the chunked path`).

### Task 4: document the chunked limitation (no behavior change)

**Files:**
- Modify: `nlsq/__init__.py` (`curve_fit_large` docstring only)
- Test: new `tests/streaming/test_chunked_ignores_stability_options.py`

**Interfaces:** none produced or consumed.

- [ ] **Step 1: Write the characterization test** `test_chunk_fits_receive_no_stability_keywords` (parametrized `show_progress` in `[False, True]`): spy on `nlsq.core.minpack.CurveFit.curve_fit` (wrap with `monkeypatch`, record kwargs, delegate to the real one), run `curve_fit_large(MODEL, x, y, p0=[...], auto_size_detection=False, chunk_size=<small>, stability="check", rescale_data=False, max_jacobian_elements_for_svd=123, show_progress=...)` on a small noisy exponential. Assert at least two recorded calls and that none has `stability`, `rescale_data` or `max_jacobian_elements_for_svd` among its kwargs. (Docstring of the test: pins the documented decision that chunked ignores these options.)
- [ ] **Step 2: Run it; expect PASS** (it documents current behavior, so it must already pass).
- [ ] **Step 3: Add a "Notes" paragraph to the `curve_fit_large` docstring:** the chunked path ignores `stability`, `rescale_data`, `max_jacobian_elements_for_svd`, and the multistart options; they only affect the in-memory path (`n < size_threshold`).
- [ ] **Step 4: Run** `uv run pytest tests/streaming/test_chunked_ignores_stability_options.py tests/streaming/test_large_strategy_routing.py -n 0 -q`; expect all pass with the routing table unchanged apart from Task 3's flipped row.
- [ ] **Step 5: Commit** (`docs(streaming): document that chunked ignores stability options`).

---

## Slice 2: public surface

### Task 5: strategy dataclasses

**Files:**
- Create: `nlsq/streaming/specs.py` (imports `dataclasses`, `typing` only; no engines)
- Modify: `nlsq/streaming/__init__.py` (export + `__all__`)
- Test: `tests/streaming/test_specs.py`

**Interfaces (all `@dataclass(frozen=True)`; `Mapping` defaults via `field(default_factory=dict)`):**
- `Multistart(n_starts: int = 10, sampler: Literal["lhs", "sobol", "halton"] = "lhs", center_on_p0: bool = True, scale_factor: float = 1.0, enabled: bool = True)`
- `InMemory(method: FitMethod | None = None, stability: Literal["auto", "check", False] = False, rescale_data: bool = True, max_jacobian_elements_for_svd: int = 10_000_000, multistart: Multistart = field(default_factory=lambda: Multistart(enabled=False)), options: Mapping[str, Any] = field(default_factory=dict))` (`FitMethod = Literal["trf", "lm", "auto", "cmaes", "multi-start"]` is defined in `specs.py` itself because `nlsq.types.MethodLiteral` omits `"lm"`; `hybrid_streaming`/`exact_streaming` never reach these types. Never write a mutable default inline)
- `Chunked(chunk_size: int | None = None, method: FitMethod | None = None, multistart: Multistart = <same default as InMemory>, options: Mapping[str, Any] = <default_factory=dict>)`. No stability fields.
- `Hybrid(config: HybridStreamingConfig | None = None, verbose: int = 1)`
- `Exact(chunk_size: int | None = None, ftol: float = 1e-8, xtol: float = 1e-8, gtol: float = 1e-8, max_nfev: int | None = None, n_points: int | None = None)`
- `Auto(size_threshold: int = 1_000_000, auto_size_detection: bool = True, in_memory: InMemory = field(default_factory=InMemory), chunked: Chunked = field(default_factory=Chunked))` with `resolve(self, n_points: int) -> InMemory | Chunked`
- `LargeStrategy = Auto | InMemory | Chunked | Hybrid | Exact`

- [ ] **Step 1: Write failing tests** in `test_specs.py`: `test_auto_resolves_below_threshold_to_in_memory` (`n = size_threshold - 1` returns `auto.in_memory`), `test_auto_resolves_at_threshold_to_chunked` (`n == size_threshold` returns `auto.chunked`), `test_auto_without_size_detection_is_always_chunked` (`n = 10`), `test_exact_has_no_loss_field` (`Exact(loss="huber")` raises `TypeError`), `test_strategies_are_frozen` (`dataclasses.FrozenInstanceError`), `test_multistart_defaults` (10, "lhs", True, 1.0, enabled True), `test_inmemory_and_chunked_default_multistart_disabled` (`InMemory().multistart.enabled is False`, same for `Chunked`), `test_chunked_has_no_stability_fields` (`Chunked(stability="auto")` raises `TypeError`), `test_streaming_exports` (`from nlsq.streaming import Auto, Chunked, Exact, Hybrid, InMemory, Multistart` works).
- [ ] **Step 2: Run; expect FAIL** (module missing).
- [ ] **Step 3: Implement `specs.py`** and the exports. `Hybrid.config` is typed with a `TYPE_CHECKING` import to keep the module engine-free.
- [ ] **Step 4: Run** `uv run pytest tests/streaming/test_specs.py -n 0 -q`; expect PASS. Check `python -X importtime -c "import nlsq"` does not show `nlsq.streaming.specs` (it must not load with top-level `nlsq`).
- [ ] **Step 5: Commit** (`feat(streaming): strategy dataclasses`).

### Task 6: `fit_with_strategy`: the four path bodies move out of `curve_fit_large`

**Files:**
- Create: `nlsq/streaming/large_fit.py`
- Modify: `nlsq/streaming/strategy.py` (add the in-memory adapter), `nlsq/__init__.py` (remove the bodies)
- Test: `tests/streaming/test_large_fit.py`

**Interfaces:**
- Produces: `fit_with_strategy(strategy: LargeStrategy, f, xdata, ydata, p0, *, sigma, absolute_sigma, bounds, check_finite, memory_limit_gb, show_progress, **kwargs) -> Any` (`CurveFitResult` or tuple for in-memory; `(popt, pcov)` otherwise). For non-`Exact` strategies it first validates sources (chunk-factory `ValueError`, `np.asarray`, empty/length/`>= 2` checks, exactly the current order and messages) to get `n_points`, *then* resolves `Auto` via `strategy.resolve(n_points)`; `Exact` skips all of that and never converts a source. It then runs one of the four paths exactly as `curve_fit_large` does today (input validation for non-exact paths including the empty/length/`>= 2` checks and the chunk-factory `ValueError`; the chunked memory-config and context setup; the hybrid config/bounds handling; the exact path from Task 2). `build_large_strategy` gains `"in_memory"` (fourth adapter) wrapping `curve_fit`, with `unpack` returning the native result unchanged (no tuple coercion). The in-memory and chunked adapters always forward `multistart=<enabled>, n_starts, sampler, center_on_p0, scale_factor` from the strategy's `Multistart`, even when `enabled` is `False`. Callers of the other chunked builds are `core/minpack.py:801, 1019, 1754, 3772` and `large_dataset.py:3054`; leave them untouched.
- The in-memory adapter must resolve `curve_fit` the same way `curve_fit_large` does today (look at how `__init__.py` ~1197 reaches it) so tests that patch `nlsq.curve_fit` or `nlsq.core.minpack.curve_fit` keep taking effect; grep `tests/` for such patches first and keep them green.

- [ ] **Step 1: Write failing tests** in `test_large_fit.py`: `test_contract_every_strategy_returns_popt_pcov_shapes` parametrized over `[InMemory(), Chunked(chunk_size=500), Hybrid(), Exact()]` on one small synthetic exponential (`popt, pcov = result`; shapes `(2,)`, `(2, 2)`; `popt` close to truth; `InMemory` result is a `CurveFitResult`); `test_exact_passes_memmaps_through_unconverted` (write `x`, `y`, `sigma` to `tmp_path` as `np.memmap`; `monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", spy)` before the call, the spy records its arguments and delegates; assert `spy` received the very same memmap objects for `x`, `y` and `sigma=` via `is`, and popt close to truth); `test_exact_accepts_chunk_factory` (callable `xdata`, `ydata=None`, `Exact(n_points=N)`); `test_chunk_factory_with_non_exact_strategy_raises_value_error` (each of the other three strategies); `test_exact_rejects_2d_xdata`; `test_in_memory_special_output_passes_through_unchanged` (`InMemory(options={"full_output": True})` returns the same longer tuple `curve_fit(..., full_output=True)` returns).
- [ ] **Step 2: Run; expect FAIL** (module missing).
- [ ] **Step 3: Implement `large_fit.py` and the in-memory adapter.** Move code; do not rewrite logic. Engine imports stay inside functions.
- [ ] **Step 4: Make `curve_fit_large` call `fit_with_strategy`** for now by building the strategy with a minimal inline translation of the legacy arguments (Task 7 replaces this with `_legacy_to_strategy`). The existing suites are the oracle: run `uv run pytest tests/streaming tests/core tests/integration tests/global_optimization tests/regression -n 0 -q`; expect the same results as after Task 4.
- [ ] **Step 5: Commit** (`refactor(streaming): move large-fit path bodies into fit_with_strategy`).

### Task 7: `strategy=` keyword and `_legacy_to_strategy`

**Files:**
- Modify: `nlsq/__init__.py` (`curve_fit_large` signature, docstring, body)
- Test: `tests/streaming/test_large_fit_legacy_shim.py`

**Interfaces:**
- Produces: `curve_fit_large(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, stability=False, rescale_data=True, max_jacobian_elements_for_svd=10_000_000, memory_limit_gb=None, auto_size_detection=True, size_threshold=1_000_000, show_progress=False, chunk_size=None, multistart=False, n_starts=10, global_search=False, sampler="lhs", center_on_p0=True, scale_factor=1.0, strategy: "LargeStrategy | None" = None, **kwargs) -> "CurveFitResult | tuple[Any, ...]"`. Both new annotations are strings with `TYPE_CHECKING` imports: `nlsq/__init__.py` has no `from __future__ import annotations` and `import nlsq` must not load `nlsq.streaming.specs`. (existing parameters and order unchanged; `strategy` is added after `scale_factor`).
- `_legacy_to_strategy(**legacy) -> "LargeStrategy"` (string annotation, `TYPE_CHECKING` import) in `nlsq/__init__.py`, the only place that knows the old parameter set:
  - `global_search=True` sets `multistart=True`, `n_starts=20` (as today); the removed-sampling-parameters `TypeError` is kept.
  - `method == "exact_streaming"`: run `_check_exact_legacy_options`, then `Exact(chunk_size, ftol, xtol, gtol, max_nfev, n_points taken from kwargs)`.
  - `method == "hybrid_streaming"`: split `verbose` and `HybridStreamingConfig` fields out of `kwargs`; leftovers raise the existing `TypeError`; result is `Hybrid(config, verbose)` (a `callback` keyword stays a `TypeError`).
  - otherwise `Auto(size_threshold, auto_size_detection, in_memory=InMemory(...), chunked=Chunked(...))`. Both carry `method`, `Multistart(n_starts, sampler, center_on_p0, scale_factor, enabled=multistart)` (always built from the four values, even when `multistart` is `False`) and `kwargs` as `options`; `InMemory` also gets the three stability options; `Chunked` also gets `chunk_size` and drops the three stability options (documented limitation).
  - Hybrid keyword validation now happens in the shim, before source validation: double-invalid input (empty data plus an unknown keyword) reports the keyword `TypeError` first. This is spec behavior change 2; pin it with one test.
- When `strategy is not None`: each of the 13 legacy path options (`method, stability, rescale_data, max_jacobian_elements_for_svd, auto_size_detection, size_threshold, chunk_size, multistart, n_starts, global_search, sampler, center_on_p0, scale_factor`) must equal its `inspect.signature(curve_fit_large)` default and `kwargs` must be empty, else `TypeError` naming the first offender. The top-level parameters (`f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, bounds, memory_limit_gb, show_progress`) are never compared. An explicitly passed default value is accepted.
- Resolved hazard: today the in-memory and chunked calls forward `n_starts`, `sampler`, `center_on_p0`, `scale_factor` even when `multistart=False` (pinned by routing rows and needed by `method="multi-start"`), which is why `Multistart` carries an `enabled` flag and `InMemory.multistart`/`Chunked.multistart` are never `None`. Do not drop these keys.

- [ ] **Step 1: Write failing tests** in `test_large_fit_legacy_shim.py`: mapping table cases `(legacy kwargs -> expected strategy)` for plain call, `chunk_size=500, multistart=True`, `global_search=True` (`n_starts == 20`), `method="hybrid_streaming", gauss_newton_tol=1e-6`, `method="exact_streaming", gtol=1e-10`; error cases keeping messages (`stability="auto"` with `exact_streaming` -> `ValueError`, `loss="huber"` with `exact_streaming` -> `TypeError`, `callback=...` with `hybrid_streaming` -> `TypeError`); `test_explicit_strategy_rejects_non_default_legacy_option` (`strategy=Chunked(), size_threshold=5`; `strategy=InMemory(), method="trf"`; `strategy=Exact(), ftol=1e-10`; `strategy=Chunked(), global_search=True`; each `TypeError` whose message contains the option name); `test_explicit_strategy_accepts_explicit_defaults`; `test_method_multi_start_keeps_sampling_values_when_flag_false` (`method="multi-start", multistart=False, n_starts=3, sampler="sobol"`: the spy on `nlsq.curve_fit` sees `n_starts=3, sampler="sobol"`); `test_every_curve_fit_method_translates` (each of `trf, lm, auto, cmaes, multi-start` yields a strategy carrying it); `test_exact_accepts_multistart_suboptions_when_flag_false`; `test_hybrid_double_invalid_reports_keyword_type_error`; `test_strategy_call_matches_legacy_call` (same data via `strategy=Chunked(chunk_size=500)` and via `chunk_size=500, auto_size_detection=False` give equal `popt`/`pcov`).
- [ ] **Step 2: Run; expect FAIL.**
- [ ] **Step 3: Implement `_legacy_to_strategy` and the `strategy=` handling**; `curve_fit_large` becomes: removed-param check, translate or validate, `fit_with_strategy(...)`. Correct the return annotation. Update the docstring: add a `strategy` section and keep the existing `method="exact_streaming"` text.
- [ ] **Step 4: Run the full oracle:** `uv run pytest tests -n 0 -q -m "not slow and not gpu"`; expect no failures. Compare against a fresh run of the same selection on `main` (stash or use a worktree), not the Task 1 four-file baseline. Also run `uv run python -c "import sys, nlsq; assert 'nlsq.streaming.specs' not in sys.modules"`; expect exit 0.
- [ ] **Step 5: Commit** (`feat(streaming): strategy= keyword for curve_fit_large`).

### Task 8: docs, changelog, verification

**Files:**
- Modify: the streaming API page under `docs/api/` (find it with `grep -rl ExactSolverError docs/api`), `docs/CHANGELOG.md`, `docs/conf.py` only if the new page needs it
- Test: none new

- [ ] **Step 1: Document** the six strategy types and `fit_with_strategy`'s relationship to `curve_fit_large` in the streaming API page. Docstring cross-references must resolve (the docs build fails on warnings).
- [ ] **Step 2: Add an Unreleased changelog entry** naming: the `strategy=` keyword; a chunked result with no `pcov` now raises `RuntimeError` instead of returning an identity matrix; error precedence for doubly-invalid hybrid input; corrected return annotation; and a "known limitation" line that the chunked path ignores `stability`, `rescale_data`, `max_jacobian_elements_for_svd` and multistart.
- [ ] **Step 3: Run** `make lint`, `make type-check`, and `make docs SPHINXOPTS="-W --keep-going"` (plain `make docs` does not fail on warnings; RTD does); expect zero errors and zero warnings.
- [ ] **Step 4: Run** `make verify`; expect clean with zero warnings.
- [ ] **Step 5: Run** `graphify update .`.
- [ ] **Step 6: Commit** (`docs: large-fit strategy`). Do not open a PR unless asked.

---

## Self-review

- **Spec coverage:** exact adapter (T1); shared helper where paths agree, `_curve_fit_exact_streaming` and `unsupported` dict deleted (T2); `unpack`, `np.eye` to `RuntimeError`, one test flipped (T3); documented chunked limitation (T4); dataclasses, exports from `nlsq.streaming` only, no `Hybrid.callback`, `Auto` with `in_memory`/`chunked` (T5); bodies out of `curve_fit_large`, four adapters (T6); `strategy=None` default, explicit-strategy `TypeError`, the 13 enumerated legacy options, shim keeps messages, `method="multi-start"` sampling values kept, no deprecation warning, corrected annotation (T7); docs/changelog/verify (T8). The spec's "audit chunked callers before moving `memory_context`" is satisfied by not moving it into the adapter: the chunked context setup moves to `large_fit.py` with the rest of the body, and `core/minpack.py:802, 1020` and `fit()` are untouched.
- **Type consistency:** `Strategy.unpack`, `build_large_strategy` kinds, `LargeStrategy`, `fit_with_strategy`, `_legacy_to_strategy` are defined once and referenced by the same names.
- **Proportion:** steps name signatures, tests and values; function bodies are left to the implementer.
