"""Regression tests for the 2026-11-04 core/stability/result review."""

import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.precision.bound_inference import infer_bounds_for_multistart, merge_bounds
from nlsq.stability.fallback import PerturbInitialGuessStrategy
from nlsq.stability.guard import NumericalStabilityGuard


def _exp(x, a, b):
    import jax.numpy as jnp

    return a * jnp.exp(-b * x)


def _data(n=50):
    x = np.linspace(0, 3, n)
    return x, 2.0 * np.exp(-1.3 * x)


def test_stability_auto_does_not_impute_nan():
    x, y = _data()
    y[5] = np.nan
    with pytest.raises(ValueError, match="infs or NaNs"):
        curve_fit(_exp, x, y, p0=[1.0, 1.0], stability="auto")


def test_result_supports_scipy_indexing():
    x, y = _data()
    res = curve_fit(_exp, x, y, p0=[1.0, 1.0])
    popt, pcov = res
    assert res[0] is res.popt or np.array_equal(res[0], popt)
    assert np.array_equal(res[1], pcov)
    assert np.array_equal(res["x"], res.popt)
    assert len(res[:1]) == 1


def test_confidence_band_large_parameter():
    x = np.linspace(0, 1, 20)

    def f(x, a, b):
        return a * x + b

    y = 1e9 * x + 3.0
    res = curve_fit(f, x, y, p0=[1e9, 3.0])
    _, jac = res._finite_diff_jacobian(x, np.array([1e9, 3.0]))
    np.testing.assert_allclose(jac[:, 0], x, rtol=1e-3, atol=1e-6)


def test_p0_auto_failure_falls_back_to_defaults():
    x = np.linspace(0, 1, 20)
    y = 2.0 * x

    def g(x, a):
        return a * x

    res = curve_fit(g, x, y, p0="auto", bounds=([0], [10]))
    assert res.popt[0] == pytest.approx(2.0, rel=1e-6)


def test_p0_auto_failure_varargs_raises_clear_error():
    x = np.linspace(0, 1, 20)

    def g(x, *p):
        return p[0] * x

    with pytest.raises(ValueError, match="parameter"):
        curve_fit(g, x, 2.0 * x, p0="auto")


def test_bounds_length_mismatch_raises_clear_error():
    x, y = _data()
    with pytest.raises(ValueError, match="bounds"):
        curve_fit(_exp, x, y, p0=[1.0, 1.0], bounds=([0, 0, 0], [5, 5, 5]))


def test_merge_bounds_never_crosses():
    lb, ub = merge_bounds(
        (np.array([0.0, 0.05]), np.array([20.0, 20.0])),
        (np.array([100.0, -np.inf]), np.array([np.inf, np.inf])),
    )
    assert np.all(lb < ub)
    assert lb[0] == 100.0 and np.isposinf(ub[0])


def test_infer_bounds_respects_user_lower_above_inferred_upper():
    x = np.linspace(0, 1, 20)
    y = 3 * np.exp(-2 * x)
    lb, ub = infer_bounds_for_multistart(
        x, y, [1.0, 1.0], user_bounds=([100, -np.inf], [np.inf, np.inf])
    )
    assert np.all(lb < ub)


def test_perturb_strategy_is_reproducible_and_leaves_global_rng_alone():
    kw = {"p0": np.array([1.0, 2.0])}
    a = PerturbInitialGuessStrategy().apply(kw)["p0"]
    b = PerturbInitialGuessStrategy().apply(kw)["p0"]
    np.testing.assert_array_equal(a, b)
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    PerturbInitialGuessStrategy().apply(kw)
    np.testing.assert_array_equal(before, np.random.get_state()[1])


def test_guard_does_not_regularize_well_conditioned_small_scale_jacobian():
    import jax.numpy as jnp

    x = np.linspace(0, 1, 20)
    J = jnp.asarray(1e-16 * np.stack([np.ones_like(x), x], axis=1))
    _, issues = NumericalStabilityGuard().check_and_fix_jacobian(J)
    assert not issues["regularized"]


def test_memory_availability_retry_sees_freed_memory(monkeypatch):
    from nlsq.caching.memory_manager import MemoryManager

    mm = MemoryManager(memory_cache_ttl=1000.0)
    vals = iter([100, 1_000_000])

    class VM:
        def __init__(self, a):
            self.available = a

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: VM(next(vals)))
    ok, _ = mm.check_memory_availability(1000)
    assert ok


def test_auto_bounds_user_lower_above_inferred_upper_fits():
    x = np.linspace(0, 0.1, 50)
    y = 3 * np.exp(-60 * x)
    res = curve_fit(
        _exp,
        x,
        y,
        p0=None,
        bounds=([-np.inf, 50], [np.inf, np.inf]),
        auto_bounds=True,
    )
    assert res.popt[1] == pytest.approx(60.0, rel=1e-3)


def test_curve_fit_large_result_indexing():
    from nlsq import curve_fit_large

    x, y = _data(200)
    res = curve_fit_large(_exp, x, y, p0=[1.0, 1.0])
    popt, pcov = res
    assert np.array_equal(res[0], popt) and np.array_equal(res[1], pcov)
