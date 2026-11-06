# NLSQ parallel audit and debugging synthesis

The review covered the current working tree and pinned commits
`ee247aad474b68aa9db3d81f4814680da1de3ca0...f4fdb15508a8c13f6698ee55456546e9b71e8f82`.
The workspace began with extensive uncommitted fixes; those were preserved.
The structural complexity inventory covered all 133 Python source modules,
with targeted correctness inspection and automated verification across the
package. This is an evidence-based audit, not proof that no other defects exist.

The requested systematic-debugging and diagnosing-bugs workflows governed
reproduction and fixes. Standards and documented-behavior reviews ran separately
in parallel. The Ponytail audit first produced a report; a separate synthesis
pass applied supported simplifications under the user's instruction to fix
findings. Fast-context was unavailable because its credentials were missing;
navigation used the existing graph, AST inventory, and targeted searches.

## Standards

The most severe confirmed standards issue was silent parameter-count inflation:
variadic and keyword-only options could become phantom fitted parameters and
produce an incorrect covariance dimension. Signature inference now counts only
positional parameters consistently across ordinary, exact-streaming, hybrid,
chunked, multi-start, CMA-ES, and parameter-analysis paths. Explicit initial
guesses continue to support variadic models.

See [standards evidence](audit-standards-2026-11-04.md).

## Spec

No originating product specification was supplied; this axis verifies documented
behavior and existing interfaces only. The most severe confirmed issue was an
exported session bundle fitting fabricated placeholder data for datasets above
20 points. Reproduction scripts now read the actual bundled CSV and uncertainty
from beside the script, including when launched from another directory.

Other fixes preserve YAML workflow and global settings, map the supported flat
CLI fitting schema, keep parameter-bound indices aligned, reject partial initial
vectors, and honor input-file and column declarations. CLI reference material
and all 11 YAML workflow examples now describe implemented settings and local
solver capabilities.

See [documented-contract evidence](audit-spec-2026-11-04.md).

## Additional correctness findings

- Distinct equal callable objects could share NLSQ and JAX compilation entries,
  returning another model's output. Identity weakrefs and a forwarding function
  isolate both caches. Independent review caught a GC reentrancy deadlock in the
  initial fix; an `RLock` and a subprocess regression protect cleanup.
- Restored results without a model used the wrong optimizer-residual signs for
  predictions and reported residuals. Both now follow their documented meanings.
- Sensitivity participation ratios overflowed or underflowed on equal finite
  eigenvalue scales. Normalization preserves the correct dimensionality and
  health classification.
- Custom model loading misidentified preceding one-argument math helpers as the
  first model, dropping bare initial-guess and bounds utilities from the real
  model. Candidate filtering now requires positional fit parameters or varargs;
  named and variadic models retain their helpers without leaking them to later
  models in the file.
- CG's float64-only division floor vanished for float32, while its relative
  stopping test still entered zero-RHS systems. Strict zero stopping and RHS
  normalization prevent zero division and underflow of the curvature product.
  A finite-only first fix was rejected: the regression now requires the correct
  `-100000` solution and near-zero reconstructed residual, not merely finite
  output. Ordinary least-squares paths generally promote to float64; the
  demonstrated float32 defect is reachable through the exported JIT helper.

See [support-module evidence](audit-support-2026-11-04.md) and
[independent cache review](audit-cache-review-2026-11-04.md).
The additional numerical review and its rejected hypothesis are recorded in
[numerical evidence](audit-numerical-review-2026-11-04.md).

## Complexity

The synthesis removed an unused chunk-preparation method and duplicate
benchmark bodies, and consolidated configuration field inventories, Make
recipes/exclusions, warmup helpers and plot downsampling: **262 net lines
removed**. Baseline tests also exposed six stale benchmark memory-pool key
probes; these now use normalized dtype keys. All eleven findings have a recorded
disposition. Directly tested capabilities and the previously specified
dependency policy were retained. QMC replacement was rejected because native
SciPy behavior differs on currently accepted skip values.

See [ranked complexity audit and dispositions](audit-complexity-2026-11-04.md).

## Verification

**All 4,999 current test cases are accounted for: 4,998 passed, one skipped,
zero failures, and zero uncovered cases.** This is segmented verification, not
a claim that one uninterrupted final suite invocation passed. The generated
[node-id reconciliation summary](audit-verification-2026-11-05.json) records the
counts and skipped case.

The broad run was interrupted at 87% to inspect twelve failures and five setup
errors after 4,384 passes and one skip. Fresh checks resolved every report:

| Reported issue | Disposition and fresh verification |
| --- | --- |
| Plot-helper import failures/setup errors | Long-lived workers retained the pre-refactor package while loading changed widget modules. Fresh imports and all **487 CLI/GUI tests passed**; no import cycle reproduced. |
| Five benchmark failures | Workers had loaded pre-edit scalar-class pool keys or a subsequently removed duplicate. Both current benchmark files passed **25 tests**. |
| Variadic parameter-count assertion | Corrected an obsolete expectation that counted `*args` as one parameter; explicit initial guesses and public fitting behavior are covered. All **112 precision tests passed**. |
| CMA-ES introspection assertion | Replaced an outdated mock binding with genuinely uninspectable callables. Both exception types and all **41 CMA-ES module tests passed**. |

The final collection contained 4,999 node IDs: 4,384 already had successful or
intentional-skip reports, and the other **615 passed in a fresh execution**.
A temporary plugin outside the repository recorded exact node IDs; reconciliation
asserted identical worker selections, complete coverage, and no failing outcomes.
The sole skip was `test_zarr_matches_in_memory` because optional `zarr` was not
installed. The final selection emitted one JAX persistent-cache decompression
warning for `jit_stats`; JAX treated the unreadable entry as a cache miss and the
test passed. The installed JAX cache-read handler confirms that fallback. The
cache-corruption cause was not established, so no speculative code change or
warning suppression was applied.

Additional green checks include the **118-test final regression selection**,
**879 caching/diagnostics/utils tests**, package-wide Ruff lint, formatting of
all **400 Python files**, mypy on **133 source files**, and `git diff --check`.
These subset counts overlap and must not be summed. Mypy retains the repository
exclusions documented in AGENTS.md. Bandit passed the medium/high severity gate;
the existing low-severity B112 memory-detection finding remains informational.
Sphinx built all **231 documents** with `-W --keep-going`. The AST graph was
refreshed without an LLM call.

Comprehensive tests used CPU JAX and offscreen Qt. A separate check passed on
the existing CUDA backend: linear fitting recovered `[2, 1]`, and the float32
CG case returned the expected `-100000` step with zero residual. This is GPU
smoke coverage, not a full GPU test matrix; no toolchain packages were changed.

The initially missing NPZ/HDF5 fixtures were generated in a temporary directory
with the provided sample-data generator. Both polynomial and decay workflows
completed through `WorkflowRunner.run` with finite parameters, configured
tolerances and measurement uncertainties; export/visualization was disabled for
that isolated check. Nine other examples passed data/model/settings extraction.

### Interrupted-run failure inventory

Every case below has a disposition in the table above and passed its fresh
current counterpart, except the retired duplicate hit-rate test, whose canonical
case passed. CMA-ES mock-correction evidence is in the
[test disposition report](audit-cmaes-test-disposition-2026-11-05.md).

```text
tests/gui_qt/test_review_2026_11_04_gui.py::test_results_page_handles_2d_surface_data
tests/gui_qt/test_review_2026_11_04_gui.py::test_results_page_model_failure_is_not_perfect_fit
tests/gui_qt/test_review_2026_11_04_gui.py::test_results_page_unexpected_error_shows_error_state
tests/gui_qt/test_nav_results_fixes.py::test_custom_model_unlocks_fitting_options
tests/gui_qt/test_nav_results_fixes.py::test_clearing_result_does_not_jump_to_results
tests/gui_qt/test_nav_results_fixes.py::test_results_page_cleared_when_result_invalidated
tests/gui_qt/test_nav_results_fixes.py::test_shortcut_navigation_respects_guards
tests/gui_qt/test_nav_results_fixes.py::test_ctrl_t_syncs_checkbox_and_state
tests/gui_qt/test_nav_results_fixes.py::test_confidence_band_large_parameters
tests/gui_qt/test_gui_startup.py::TestApplicationStartup::test_main_window_creation
tests/benchmarks/test_benchmark_memory.py::TestPhase2LRUMemoryPool::test_lru_pool_hit_rate
tests/benchmarks/test_benchmark_memory.py::TestPhase2LRUMemoryPool::test_lru_eviction_preserves_hot_arrays
tests/benchmarks/test_benchmark_memory.py::TestPhase2LRUMemoryPool::test_pool_efficiency_metric
tests/benchmarks/test_benchmark_cache.py::TestMemoryPoolEfficiency::test_lru_pool_hit_rate
tests/benchmarks/test_benchmark_cache.py::TestMemoryPoolEfficiency::test_pool_efficiency_metric
tests/precision/test_algorithm_selector.py::TestAlgorithmSelectorBasic::test_estimate_n_params_fallback
tests/global_optimization/test_cmaes_optimizer.py::TestCMAESBugFixRegressions::test_scalar_bounds_without_p0_or_introspectable_signature_raises
```
