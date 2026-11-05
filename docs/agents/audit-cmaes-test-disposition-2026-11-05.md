# CMA-ES inference regression disposition — 2026-11-05

The existing
`tests/global_optimization/test_cmaes_optimizer.py::TestCMAESBugFixRegressions::test_scalar_bounds_without_p0_or_introspectable_signature_raises`
failed on the current tree: **DID NOT RAISE**, 15.50 seconds.

This was a stale test setup, not a production failure. It patched the global
`inspect.signature` attribute to simulate an uninspectable model. After
signature inference moved into `nlsq/core/problem.py`, that module's imported
`signature` binding was unaffected; the test passed a genuinely inspectable
two-parameter model and optimization correctly proceeded.

Replaced the global mock with a real callable whose `__signature__` property
raises. The test now covers both `TypeError` and `ValueError`, asserting that
scalar bounds without `p0` and without introspectable arity raise the public
parameter-count error. The callable's model implementation remains valid.
Production code and corrected positional parameter inference are unchanged.

Ruff and formatting pass for the modified test file. The complete CMA-ES
optimizer module is verified on CPU with `-n 0 --reruns 0`; JUnit output is
written to `/tmp/nlsq-audit-cmaes-post.xml`: **41 passed**, 72.38 seconds.
