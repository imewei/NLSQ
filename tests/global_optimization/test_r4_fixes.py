"""R4-3 (context-aware warnings) and R4-5 (non-finite span centering)."""

import warnings

import jax.numpy as jnp
import numpy as np

import nlsq.global_optimization.multi_start as ms
from nlsq import OptimizeWarning
from nlsq.global_optimization.sampling import center_samples_around_p0


def test_suppress_optimize_warning_context_aware_mode_does_not_leak(monkeypatch):
    """R4-3: in context-aware-warnings mode the ignore filter must end with the helper."""
    monkeypatch.setattr(ms, "_context_aware_warnings", lambda: True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with ms._suppress_optimize_warning():
            warnings.warn("inside", OptimizeWarning, stacklevel=1)
        warnings.warn("after", OptimizeWarning, stacklevel=1)
    assert [str(w.message) for w in rec] == ["after"]


def test_center_samples_one_sided_bounds_finite():
    """R4-5: a non-finite span uses the p0-based half-width, never NaN/inf."""
    samples = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    p0 = jnp.array([2.0, -3.0, 0.0])
    lb = jnp.array([0.0, -jnp.inf, -jnp.inf])
    ub = jnp.array([jnp.inf, 0.0, jnp.inf])
    out = np.asarray(center_samples_around_p0(samples, p0, 0.5, lb, ub))
    assert np.all(np.isfinite(out))
    assert np.all(out >= np.asarray(lb)) and np.all(out <= np.asarray(ub))
    # centre sample is p0 clipped into the box; half-width = (|p0|+0.1)*sf
    np.testing.assert_allclose(out[1], [2.0, -3.0, 0.0])
    np.testing.assert_allclose(out[0, 0], max(2.0 - (2.0 + 0.1) * 0.5, 0.0))
    np.testing.assert_allclose(out[2, 2], 0.0 + (0.0 + 0.1) * 0.5)


def test_center_samples_finite_bounds_unchanged():
    samples = jnp.array([[0.0], [0.5], [1.0]])
    out = center_samples_around_p0(
        samples, jnp.array([5.0]), 0.5, jnp.array([0.0]), jnp.array([10.0])
    )
    np.testing.assert_allclose(np.asarray(out).ravel(), [2.5, 5.0, 7.5])


def test_context_aware_warnings_requires_thread_inherit(monkeypatch):
    class Flags:
        context_aware_warnings = 1
        thread_inherit_context = 0

    monkeypatch.setattr(ms.sys, "flags", Flags)
    assert ms._context_aware_warnings() is False
    Flags.thread_inherit_context = 1
    assert ms._context_aware_warnings() is True
