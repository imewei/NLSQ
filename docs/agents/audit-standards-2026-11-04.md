# NLSQ standards audit — 2026-11-04

Scope: parameter inference across the public fitting interfaces, global
optimization, streaming, and precision analysis. Reviewed the pinned change
range `ee247aad474b68aa9db3d81f4814680da1de3ca0..f4fdb15508a8c13f6698ee55456546e9b71e8f82`
and the current dirty tree. Existing unrelated edits remain intact. This is a
focused API-contract audit, not an exhaustive certification of every module.

Finding: `len(signature(f).parameters) - 1` counted `*args`, `**kwargs`, and
keyword-only options as fitted parameters. A model such as
`f(x, slope, *, offset=1)` consequently received two positional fit values,
causing a call error or an incorrect covariance dimension. Variadic models
could instead silently acquire unconstrained phantom parameters.

Applicable contract: `nlsq/core/minpack.py` advertises SciPy-compatible
`curve_fit`; SciPy's `curve_fit` documentation, parameter `p0`, requires an
all-ones initial guess when the parameter count is introspectable. The
installed SciPy implementation's
`scipy._lib._util.getfullargspec_no_self` clause constructs `args` exclusively
from `inspect.Parameter.POSITIONAL_ONLY` and `POSITIONAL_OR_KEYWORD` kinds.
These exact Python signature kinds distinguish fitted positional values
from variadics and keyword-only options.

Fixed: `nlsq/core/problem.py::_n_params_from_signature` supplies that positional
count to `nlsq/core/minpack.py`, `nlsq/streaming/strategy.py::default_p0`,
`nlsq/streaming/large_dataset.py`, and
`nlsq/global_optimization/cmaes_optimizer.py`. `nlsq/__init__.py` and
`nlsq/global_optimization/multi_start.py` reuse `default_p0`.
`nlsq/precision/parameter_estimation.py` applies the same signature-kind filter
so algorithm analysis agrees with fitting. Explicit `p0` remains available
for models whose fitted arity cannot be inferred.

Evidence: the original audit reproduced the inference failures before the
fix and recorded six core regressions plus 57 nearby tests passing afterward.
The resumed verification ran all four new
`tests/{core,global_optimization,streaming,precision}/test_audit_standards_2026_11_04.py`
files: **13 passed** with `-n 0 --reruns 0` (11.52 seconds). Coverage includes
ordinary, exact-streaming, streaming-preset, chunked, multi-start, CMA-ES, and
algorithm-analysis paths. Fits assert expected parameter/covariance dimensions;
CMA-ES additionally checks finite fitted values. No numerical divergence was
observed. Ruff passed for all eight scoped source files and four regression
files; the new tests were format-checked. `mypy nlsq` passed for 133 source
files, subject to the repository's documented module-level exclusions.

Legacy-test disposition (November 5):
`tests/precision/test_algorithm_selector.py::test_estimate_n_params_fallback`
incorrectly described a parsable variadic signature as unparsable and counted
`*args` as one fitted parameter. It reproduced `0 != 1` before correction
(0.69 seconds). The updated test expects zero declared fit values, checks
explicit `p0` arity and a named positional parameter alongside variadics.
A public regression verifies `curve_fit` rejects variadic-only inference and
fits correctly with explicit `p0`. Production inference remains unchanged.
The complete `tests/precision` run passed: **112 tests**, `-n 0 --reruns 0`,
22.61 seconds; Ruff and formatting passed for the modified test file.
