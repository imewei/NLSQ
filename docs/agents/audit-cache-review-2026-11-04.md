# Independent fix review — 2026-11-04

Scope: current changes in `nlsq/caching/_closure_serial.py`,
`nlsq/caching/unified_cache.py`,
`nlsq/diagnostics/parameter_sensitivity.py`, and
`nlsq/result/curve_fit_result.py`, plus their three new regression files.

**P1, fixed:** the new identity weakref registry originally used a
non-reentrant `threading.Lock`. Python cyclic garbage collection can execute
the registry's `discard` callback synchronously while `closure_serial` holds
that same lock, blocking the calling thread permanently. A subprocess with
`gc.set_threshold(1, 1, 1)` and repeatedly registered self-referential callable
objects reproduced the hang. The permanent serial-marked regression
`tests/caching/test_callable_identity.py::test_callable_identity_cleanup_during_cyclic_gc_does_not_deadlock`
failed before the fix: its 10-second faulthandler deadline showed `discard`
waiting inside `closure_serial` (pytest: one failed, 11.55 seconds).

Changed the registry lock to `threading.RLock`, allowing synchronous GC
cleanup while preserving cross-thread exclusion and the callback's weakref
identity guard. The regression also checks stable live-object serials,
garbage collection of cyclic callables, and serial non-reuse. Its subprocess
has a 15-second timeout and `NLSQ_SKIP_GPU_CHECK=1`.

Afterward the identity, unified-cache, additional-cache, and thread-safety
suites passed: **27 passed**, `-n 0 --reruns 0`, 4.69 seconds. Ruff and format
checks passed for the two edited files.

No additional concrete regression was found in the remaining reviewed fixes.
A direct JAX check exercised the identity wrapper with a callable instance,
bound method, and `functools.partial`: positional `static_argnums=(1,)`
continued to recognize the parameter when subsequently passed by keyword.
The result fallback's sign agrees with `CurveFit.curve_fit`'s documented
minimized vector, `f(xdata, *popt) - ydata`; raw reported residuals use the
opposite sign. The sensitivity participation ratio now normalizes finite
positive eigenvalues before squaring, preserving dimensionality across
overall scales. Existing weighted fallback limitations remain documented.
