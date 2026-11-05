# NLSQ documented-contract audit — 2026-11-04

No originating product specification was supplied. This audit verifies documented
behavior, existing public interfaces, and reproducibility promises rather than
asserting compliance with an unavailable specification. Preexisting edits were
preserved; this report describes the GUI/YAML/export and CLI documentation slice.

Coverage: Qt configuration adapters, data-loading and column-selection controls,
script/session-bundle export, CLI workflow extraction/model resolution, CLI
reference, and all 11 YAML examples under `10_cli-commands/workflows`. Existing
GUI review regressions also exercise polynomial coefficient order and effective
workflow export. Other agents cover numerical kernels and architecture.

Findings and fixes:

- YAML round trips discarded the selected workflow and nested global settings.
  Imports now retain workflow, start count, sampler, centering and scale.
- Current CLI model degree/custom paths and flat `fitting.p0`, `bounds`, loss,
  tolerances and `max_nfev` were skipped by the GUI adapter. These fields now map
  to session settings. One-sided bounds expand their unbounded side. Missing or
  null parameter bounds retain positional alignment; partially specified initial
  guesses raise `ValueError` instead of compacting parameters into wrong positions.
- GUI data import skipped `data.input_file` and configured column assignments.
  It now accepts that key and legacy `data.file`, resolves relative GUI imports
  against the YAML directory, selects numeric/named columns, and enables 2D
  coordinates when `z` is configured. CLI paths remain relative to the process CWD.
- Exported bundles above 20 points fitted fabricated placeholder data. Scripts
  now load bundled `data.csv` beside `__file__`, including measurement uncertainty.
- CLI examples advertised ignored nested termination/manual parameter keys and
  global solvers. Examples now use flat fitting settings; the historical
  `06_global_optimization.yaml` is explicitly a bounded local fit. The CLI
  reference and examples README direct global optimization to the Python API.

Verification: an isolated temporary pytest plugin loaded HEAD's configuration
adapter without modifying the working tree: all seven new adapter regressions
failed. The current implementation passed 28 GUI tests and 13 CLI integration/
template tests with CPU JAX, Qt offscreen, `-n 0 --reruns 0`; scoped Ruff passed.
Nine examples loaded their data, resolved models and extracted corrected settings.
Two require generated fixtures absent from this checkout (`polynomial_data.npz`,
`large_dataset.h5`), so their end-to-end execution was not verified. Infinity in
bound assertions denotes an intentionally unconstrained parameter, not divergence.

Follow-up CLI correctness pass reviewed dirty batch execution, data validation,
error logging, exit status, model loading and histogram changes. One new regression
was reproduced through public `ModelRegistry.get_model`: a public one-argument
math helper preceding the model stole ownership of bare initial-guess/bounds
helpers. `nlsq/cli/model_registry.py:574` now filters first-model candidates to
functions accepting positional x plus positional or variadic fit parameters;
later models still cannot inherit another model's defaults. The public file-load
regression at `tests/cli/test_model_registry.py:126` covers named and variadic
parameters, preceding math/keyword-only utilities, and later-model isolation.
Both new variants failed before the fix; the complete model-registry and existing
CLI-review regression modules then passed **46 tests** (`-n 0 --reruns 0`, CPU JAX).
Scoped Ruff lint/format checks passed. No additional severe regressions were
identified in the other reviewed CLI changes.

Fresh-process follow-up (2026-11-05): the interrupted broad run reported GUI
`ImportError` for `_downsample` after plot-helper consolidation occurred while
its workers were alive. Current `plots/__init__.py:33` defines the helper before
pyqtgraph configuration and lazy widget imports; no import cycle was found.
Fresh imports of the helper, fit/residual widgets and ResultsPage all succeeded.
The complete `tests/cli tests/gui_qt` selection then passed **487 tests in 115.44s**
with CPU JAX, Qt offscreen, `-n 0 --reruns 0`; JUnit evidence is
`/tmp/nlsq-audit-ui-post.xml`. The original import failures are consistent with a
cached pre-refactor package and later-loaded changed submodules, rather than a
reproducible failure in the current source. No source adjustment was made for
those historical failures.
