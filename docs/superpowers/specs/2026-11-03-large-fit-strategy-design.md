# Large-fit strategy seam: design

Status: draft for approval. Date: 2026-11-03.
Follows the deep-modules refactor (PRs #51-#56). Vocabulary: module, interface, seam, adapter, depth, leverage, locality.

## Intent

`curve_fit_large` (`nlsq/__init__.py:853-1318`, about 460 lines, 23 parameters) is a shallow interface over four fit paths that it dispatches by hand:

1. in-memory `curve_fit` (when `n < size_threshold`),
2. chunked `LargeDatasetFitter`,
3. `method="hybrid_streaming"` (`AdaptiveHybridStreamingOptimizer`),
4. `method="exact_streaming"` (`exact_lm_fit`, `_curve_fit_exact_streaming` at `:742-852`).

The problems, all verified by reading the code:

- Many parameters are valid on only some paths (`multistart=True` is invalid for exact (`n_starts`, `sampler`, `center_on_p0`, `scale_factor` are accepted and ignored while `multistart=False`); `stability`, `rescale_data`, `max_jacobian_elements_for_svd`, `auto_size_detection`, `size_threshold` must stay default for exact). A caller must know the path before the signature makes sense. Illegal combinations are caught at runtime by `ValueError`/`TypeError`.
- `method` carries two meanings: an algorithm (`trf`, `lm`) and a path (`hybrid_streaming`, `exact_streaming`).
- `**kwargs` validity differs per path (exact accepts only `ftol`, `xtol`, `gtol`, `max_nfev`, `n_points`).
- Each path repeats p0 defaulting, bounds normalization, sigma handling and result unpacking.
- `stability`, `rescale_data` and `max_jacobian_elements_for_svd` are honored in-memory but silently dropped on the chunked path. The chunked branch never forwards them (`:1187-1189` vs `:1268-1279`). It does forward `multistart`, `n_starts`, `sampler`, `center_on_p0` and `scale_factor`, but the chunk fits call `CurveFit.curve_fit`, which lets them fall into `**kwargs`: stability and multistart are handled only by the module-level `curve_fit` wrapper (`core/minpack.py:2867-2910`), and `LargeDatasetFitter` builds a plain `CurveFit()` (`large_dataset.py:1134`). So `multistart=True` on the chunked path is also silently ignored today. **Pre-existing limitation, kept** (see decisions).
- The chunked branch fabricates `np.eye(len(popt))` as `pcov` when the result has none (`:1313`). That breaks the project's no-silent-fabrication rule.
- The return annotation says `tuple | OptimizeResult`, but the in-memory path returns a `CurveFitResult` (pinned by `test_large_strategy_routing.py:728`), or a longer plain tuple when `timeit`/`return_eval`/`full_output` are passed. The chunked, hybrid and exact paths return `(popt, pcov)`. `CurveFitResult` unpacks as `popt, pcov`.
- `exact` sits outside the `Strategy` seam that PR #54 built for hybrid and chunked.

Deletion test: removing the dispatch would push its complexity into about 40 callers (tests, benchmarks, examples, docs), so the module earns its keep. What is shallow is its interface.

## Goals

- Make the path a first-class choice with typed, per-path options, so an illegal combination cannot be written down.
- Put `exact` behind the same seam as hybrid and chunked, and add the in-memory path as a fourth adapter (`in_memory`). Four adapters make it a real seam.
- Normalize p0 and bounds in one shared helper for the paths that already normalize them the same way (hybrid and exact); unpack each path's result in one place per adapter, where that path's integrity rules live.
- Remove the fabricated `pcov`.
- Make the chunked path's lack of `stability`/`rescale_data`/`max_jacobian_elements_for_svd`/multistart support explicit and documented (the `Chunked` type has no stability fields), without changing results.
- Leave every existing caller working, unchanged.

## Non-goals

- No engine registry, no user-supplied routing hook, no third-party engines. Nothing varies across them yet.
- No `Problem` dataclass or new port beyond what the adapters need.
- No new public function name. `curve_fit_large` keeps its name.
- No change to the numerical algorithms in any engine.
- No `DeprecationWarning` on the old keywords in this project (see decisions).
- No change to `fit()` or `curve_fit()` public behavior.

## Anti-goals (failure even if tests pass)

- Behavior change for a caller who sets none of the options listed under "Behavior changes".
- A silent change to numerical results on any path.
- A new warning in `make verify` (the repo is at zero warnings and RTD has `fail_on_warning`).
- Import-time regression: `import nlsq` is about 620 ms; engine imports stay lazy.
- Weakening `mock.patch("nlsq.streaming.<module>.<Class>")` targets that tests rely on.

## Constraints

1. `curve_fit_large(f, xdata, ydata, p0, ...)` keeps its signature (`ydata` stays a required positional; callers of a chunk factory pass `ydata=None` as today). It returns `(popt, pcov)` on the chunked, hybrid and exact paths and the engine's native return on the in-memory path (`CurveFitResult`, or the longer tuple of the special output modes).
2. Error behavior is preserved: `ExactSolverError` (with `.result`) on an exact stall; the existing `ValueError`/`TypeError` messages for unsupported combinations on the legacy keywords, word for word (`tests/streaming/test_exact_lm_dispatch.py`, `test_exact_lm_sources.py`, `test_large_strategy_routing.py` pin them).
3. PR #54 rule: `build_large_strategy` never decides routing, and `Strategy.fit` returns the engine's native result unchanged (hybrid: dict; chunked: `OptimizeResult`; exact: its native result). Unpacking to `(popt, pcov)` is a separate step.
4. Engine classes and functions are looked up when a strategy is built, not at import time.
5. Exact accepts out-of-memory sources (`np.memmap`, `h5py.Dataset`, `zarr.Array`) and a replayable chunk factory (callable `xdata`, `ydata=None`, optional `n_points`). It must not call `np.asarray` on them.
6. Models are pointwise `jax.numpy` functions.
7. All dependencies are in-process; a plain array stands in for any lazy source in tests. No ports or mocks beyond the existing engines.

## Decisions

Decided by the user:

- **Public surface.** Keep `curve_fit_large`; add one keyword `strategy=`. The most common call shape is the plain `(f, x, y[, p0])` form (26 of 33 example calls), so path knowledge must not leak into the default case.
- **Chunked options (revised after review).** The first decision was to pass `stability`, `rescale_data` and `max_jacobian_elements_for_svd` through to the chunked engine. Review showed that this would be a no-op: the chunk fits call `CurveFit.curve_fit` (`large_dataset.py:1830, 2285, 2782`), which treats them as `**kwargs`, and the stability logic lives only in the module-level `curve_fit`. The user then chose: **keep dropping them and document it.** `Chunked` has no stability fields; the shim accepts the legacy values on a chunked route and ignores them, as today. Real chunked support is out of scope and needs its own design (a `CurveFit(enable_stability=...)` inside the engine plus per-chunk rescale).
- **Internals.** `exact` becomes a third kind and `in_memory` a fourth in `build_large_strategy`; `_curve_fit_exact_streaming` and its `unsupported` dict are deleted; p0 and bounds normalization is shared where the paths already agree (see "My calls"); each adapter has its own `unpack`.
- **`np.eye` fallback** is replaced by `RuntimeError`.

My calls (marked so they can be checked):

- **Strategy types** are frozen dataclasses in a new module `nlsq/streaming/specs.py` that imports no engines:
  - `Auto(size_threshold=1_000_000, auto_size_detection=True, in_memory=InMemory(), chunked=Chunked())` (the two nested defaults use `field(default_factory=...)`). This is the default. It resolves to `in_memory` when `auto_size_detection and n < size_threshold`, else `chunked`, exactly as today. Carrying both option sets lets old calls such as `chunk_size=500, multistart=True` keep working under automatic routing.
  - `InMemory(method: FitMethod | None, stability, rescale_data, max_jacobian_elements_for_svd, multistart: Multistart, options: Mapping)`. `method` takes any value `curve_fit` accepts for these paths (`FitMethod = Literal["trf", "lm", "auto", "cmaes", "multi-start"]`, defined in `specs.py`; `types.MethodLiteral` omits `"lm"`), because the shim forwards it unchanged.
  - `Chunked(chunk_size, method: FitMethod | None, multistart: Multistart, options)`. No stability fields (see "Chunked options"). `method=None` becomes `"trf"` when forwarded, as today.
  - `Hybrid(config: HybridStreamingConfig | None, verbose=1)`. No `callback` field: today a `callback` keyword on this path raises `TypeError` because it is not a `HybridStreamingConfig` field (`__init__.py:1120-1125`), so the `callback=kwargs.get("callback")` at `:1159` is unreachable. The shim keeps that `TypeError`; a hybrid callback would be new capability and is out of scope.
  - `Exact(chunk_size, ftol, xtol, gtol, max_nfev, n_points)`; it has no `loss`, `stability` or `multistart` field, so passing them is a dataclass `TypeError`.
  - `Multistart(n_starts=10, sampler="lhs", center_on_p0=True, scale_factor=1.0, enabled=True)`; legacy `global_search=True` maps to `enabled=True, n_starts=20`. `InMemory.multistart` and `Chunked.multistart` default to `Multistart(enabled=False)` (always present, never `None`), so the adapters always forward `multistart=<enabled>, n_starts, sampler, center_on_p0, scale_factor` exactly as today. This matters: `curve_fit(method="multi-start")` activates multistart from `method` alone, using the sampling values even when the `multistart` flag is `False`, and `test_large_strategy_routing.py:609-617, 662-673, 711-719` pin the forwarded dicts.
- **`Chunked.multistart` keeps today's meaning, which is none.** The `Multistart` fields are forwarded as fit kwargs (`__init__.py:1275-1279`) and the chunk fits ignore them (see Intent). `LargeDatasetFitter`'s constructor-level multistart (`large_dataset.py:1067-1123`) is a different mechanism and is not exposed. Making `multistart=True` work on the chunked path would change results for existing callers (anti-goal) and is out of scope; it is recorded in Risks as a known limitation.
- **Shared p0/bounds helper covers only hybrid and exact.** Today the paths differ: in-memory passes p0 raw and drops infinite bounds; chunked sends `p0=None` to the fitter (which defaults it per chunk) and passes bounds raw; hybrid uses `_default_p0` + `atleast_1d` + `prepare_bounds`; exact uses `_default_p0` + float cast + the strict `lb < ub` check. Forcing one behavior on all four would change errors and possibly results, so in-memory and chunked keep owning their own defaulting. The anti-goal (no behavior change) wins over "one place".
- **Exports.** The strategy types are exported from `nlsq.streaming` (alongside `ExactSolverError`), not from the top-level `nlsq` namespace, whose names `Auto`, `Exact` etc. would be too generic. Add them to the API docs under `nlsq.streaming`; no change to `import nlsq` time.
- **`strategy=None` is the default** (equivalent to `Auto()` built by the shim from the legacy options). "Legacy path options" are exactly these 13 parameters: `method`, `stability`, `rescale_data`, `max_jacobian_elements_for_svd`, `auto_size_detection`, `size_threshold`, `chunk_size`, `multistart`, `n_starts`, `global_search`, `sampler`, `center_on_p0`, `scale_factor`. The other parameters (`f, xdata, ydata, p0, sigma, absolute_sigma, check_finite, bounds, memory_limit_gb, show_progress`) are top-level and never compared. When `strategy` is given explicitly, every legacy path option must still equal its default and `**kwargs` must be empty; otherwise `TypeError` naming the offending option. This covers `strategy=Chunked(), loss="huber"`, `strategy=Exact(), ftol=1e-10` and `strategy=InMemory(), method="trf"` alike: all rejected, because the strategy owns those fields. Detection is by comparing against the documented defaults, so an explicitly passed default value is accepted; that imprecision is accepted.
- **Top-level parameters** of `curve_fit_large`: `f, xdata, ydata, p0=None, sigma, absolute_sigma, bounds, check_finite, memory_limit_gb, show_progress, strategy=None` (existing order and optionality kept; `ydata` has no default). The new `strategy` annotation is a string (`"LargeStrategy | None"`, imported under `TYPE_CHECKING`) because `nlsq/__init__.py` does not use `from __future__ import annotations` and `import nlsq` must not load `nlsq.streaming.specs`. `memory_limit_gb` and `show_progress` are used by chunked and exact; hybrid and in-memory ignore them today, and this stays documented, not changed.
- **Legacy shim.** The existing 23-parameter signature remains and translates to a strategy in one function (`_legacy_to_strategy`). It is the only place that knows the old parameter set and keeps the exact-mode default checks and their messages. If both `strategy=` and a legacy path option are given, raise `TypeError`.
- **No `DeprecationWarning`** in this project. Adding it later belongs with migrating the roughly 40 test calls.
- **Staging** (below). `memory_context` / `large_dataset_context` setup stays in `curve_fit_large` until every caller of `build_large_strategy("chunked", ...)` is audited (see Risks).
- **Return type** annotation becomes `"CurveFitResult | tuple[Any, ...]"` (string; `OptimizeResult` is dropped because no path returns it). No behavior change.
- **Exact and multistart suboptions.** `n_starts`, `sampler`, `center_on_p0`, `scale_factor` are *accepted and ignored* on `exact_streaming` while `multistart=False`; only `multistart=True` is an error (`test_exact_lm_dispatch.py:222-234`). The shim keeps this.
- **Exact p0 conversion order.** Exact converts a non-`None` `p0` with `np.asarray(p0, dtype=float)` *before* `np.atleast_1d`/bounds handling, so a complex `p0` still raises `TypeError` and does not emit `ComplexWarning`.
- **Validation order.** Non-exact paths validate sources (`is_chunk_factory`, empty, length, `>= 2`) before `strategy.resolve(n_points)`.

## Behavior changes (all others must be identical)

1. A chunked result with no `pcov` raises `RuntimeError` instead of returning `np.eye`. The only test pinning this is `tests/streaming/test_large_strategy_routing.py:684`, which uses a fake result with `.x` only; it flips to expect `RuntimeError`. The real chunked engine always sets `pcov` (inf on failure; `large_dataset.py:1870, 1934, 1954`). `test_large_dataset.py:879` is unrelated and stays.
2. Error precedence for inputs that are invalid in two ways at once: a hybrid call with an unknown keyword *and* empty data now reports the keyword `TypeError` first (the shim runs before source validation); before, the empty-data `ValueError` came first. Each error is unchanged for inputs invalid in only one way.
3. The return annotation is corrected (no runtime effect).

Removed from the earlier draft: "chunked now honors `stability`/`rescale_data`/`max_jacobian_elements_for_svd`". It would have been a no-op (see Intent), so there is no such change.

## Staging

- **Slice 1, no public change.** Add the `exact` adapter to `build_large_strategy`; shared p0/bounds helper; per-adapter `unpack` (including `ExactSolverError`); remove `np.eye`; document the chunked limitation; delete `_curve_fit_exact_streaming`. Existing tests pass, with the single flipped assertion.
- **Slice 2, new public surface.** Add `specs.py`, the `in_memory` adapter, the `strategy=` keyword, `Auto` resolution and `_legacy_to_strategy`. Old calls route through the shim.
- **Slice 3, deferred.** Migrate tests to `strategy=`, add the `DeprecationWarning`, add a "choosing a strategy" doc page.

## Testing

The interface is the test surface.

- One parametrized contract test over `[InMemory(), Chunked(chunk_size=...), Hybrid(), Exact()]` on the same small synthetic fit: each returns `popt.shape == (p,)` and `pcov.shape == (p, p)`.
- Routing table tests on `Auto` resolution alone (threshold edges, `auto_size_detection=False`).
- Shim tests: old keywords in, expected strategy out, same error messages.
- A characterization test that chunk fits (with and without `show_progress`) receive no `stability`/`rescale_data`/`max_jacobian_elements_for_svd` keyword, documenting the decision.
- Exact on a memmap: assert the object reaching `exact_lm_fit` *is* the memmap (spy on the build-time-resolved function), for `xdata`, `ydata` and `sigma`.
- Shim cases: `method="multi-start"` with `multistart=False, n_starts=3, sampler="sobol"` forwards those values; exact with `multistart=False` and non-default sub-options is accepted; exact with a complex `p0` raises `TypeError`.
- Exact on a plain array, an `np.memmap`, and a chunk factory.
- Existing suites stay unchanged: `test_exact_lm_dispatch.py`, `test_exact_lm_sources.py`, `test_large_strategy_routing.py` (one assertion flipped), `tests/core/test_minpack_coverage.py` and `test_core_review_2026_10_29.py` for the `mock.patch` targets.
- Apply `@pytest.mark.serial` to anything with 100K+ arrays or subprocesses. `make verify` must stay clean with zero warnings.

## Risks and unknowns

- `build_large_strategy("chunked", ...)` is also called outside `curve_fit_large`: `core/minpack.py:801, 1019, 1754, 3772` and the `fit_large_dataset` helper (`large_dataset.py:3054`). Moving `memory_context` into the chunked adapter could change their behavior. Not done in slice 1; audit those callers first. (`nlsq/__init__.py:651` and `core/minpack.py:767, 1079` build `"hybrid"`, not chunked, and are out of that audit.) Line numbers were taken from grep at spec time; re-check before editing.
- Known limitation, not fixed here: `multistart=True` and the stability options are silently ignored on the chunked path (pre-existing). A follow-up could warn or implement them; the user chose to leave behavior unchanged.
- Call counts differ between my sources (48 files with calls vs 57 files mentioning the name); neither number is used for a decision.
- Whether any test other than `:684` depends on the `np.eye` fallback was searched by grep only.

## Left to the builder

File and class naming beyond what is stated; wording of new dataclass error messages; internal helper names; how `unpack` is represented (method on the adapter vs table); docstring layout.
