"""Hybrid streaming input preparation goes through ``prepare_problem``.

Pins what ``_curve_fit_hybrid_streaming`` does with ``nan_policy``,
``check_finite``, ``p0`` and ``bounds`` after it delegates to
``nlsq.core.problem.prepare_problem``, including the five deliberate behavior
differences (each was a crash or a silently wrong result before hybrid shared preparation).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit
from nlsq.core import minpack
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

P0 = [1.0, 1.0]
_FAST = {
    "method": "hybrid_streaming",
    "verbose": 0,
    "warmup_iterations": 2,
    "max_warmup_iterations": 4,
    "gauss_newton_max_iterations": 3,
}


class _Reached(Exception):
    """Raised by the spy so tests inspect the optimizer inputs without fitting."""


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=50):
    x = np.linspace(0.0, 4.0, n)
    y = 2.0 * np.exp(-1.3 * x) + 0.01 * np.random.default_rng(0).standard_normal(n)
    return x, y


@pytest.fixture
def reached(monkeypatch):
    """Stop at ``AdaptiveHybridStreamingOptimizer.fit``; record what it receives."""
    seen = {}

    def spy(self, data_source, *args, **kwargs):
        seen["xy"] = tuple(np.array(d) for d in data_source)
        seen["kwargs"] = kwargs
        raise _Reached

    monkeypatch.setattr(AdaptiveHybridStreamingOptimizer, "fit", spy)
    return seen


def _run(x, y, **kw):
    return CurveFit().curve_fit(_model, x, y, **{**_FAST, **kw})


def test_hybrid_kwargs_not_mutated_by_preparation(monkeypatch, reached):
    """Preparation adds no tolerance defaults to the hybrid's kwargs.

    With ``enable_stability=True`` and ``method=None`` the standard path would add
    ftol/xtol/gtol/max_nfev/x_scale; those would fail the hybrid's leftover-keyword
    TypeError check. Hybrid passes an explicit method and no kwargs to
    ``prepare_problem``, so auto-selection cannot add them.
    """
    x, y = _data()
    calls = []
    orig = minpack.prepare_problem

    def wrapped(*args, **kwargs):
        before = dict(kwargs.get("kwargs") or {})
        out = orig(*args, **kwargs)
        assert dict(kwargs.get("kwargs") or {}) == before
        calls.append(kwargs)
        return out

    monkeypatch.setattr(minpack, "prepare_problem", wrapped)
    with pytest.raises(_Reached):
        CurveFit(enable_stability=True).curve_fit(_model, x, y, p0=P0, **_FAST)

    assert len(calls) == 1
    assert calls[0].get("kwargs") is None
    assert calls[0].get("method") == "hybrid_streaming"
    assert calls[0]["ctx"].flength is None
    # The prepared p0/bounds/sigma reach the optimizer call.
    assert set(reached["kwargs"]) >= {"p0", "bounds", "sigma"}


def test_hybrid_check_finite_false_behaviour_unchanged(reached):
    x, y = _data()
    y_nan = y.copy()
    y_nan[3] = np.nan

    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        _run(x, y_nan, p0=P0)
    # An invalid nan_policy is now rejected up front (was silently accepted).
    with pytest.raises(ValueError, match="nan_policy must be one of"):
        _run(x, y_nan, p0=P0, nan_policy="bogus")

    with pytest.raises(_Reached):
        _run(x, y_nan, p0=P0, check_finite=False)
    assert np.isnan(reached["xy"][1][3]) and reached["xy"][1].shape == (50,)
    with pytest.raises(_Reached):
        _run(x, y_nan, p0=P0, nan_policy="propagate")
    assert reached["xy"][1].shape == (50,)
    with pytest.raises(_Reached):
        _run(x, y_nan, p0=P0, nan_policy="omit")
    assert reached["xy"][1].shape == (49,)


def test_hybrid_p0_none_and_bounds_validation_unchanged(reached):
    x, y = _data()

    # p0=None: n comes from the signature, p0 defaults to ones.
    with pytest.raises(_Reached):
        _run(x, y)
    np.testing.assert_array_equal(reached["kwargs"]["p0"], [1.0, 1.0])
    assert reached["kwargs"]["bounds"] is None

    # p0 is clipped into finite bounds; bounds are handed to the optimizer.
    with pytest.raises(_Reached):
        _run(x, y, p0=[10.0, 10.0], bounds=([0, 0], [5, 5]))
    np.testing.assert_array_equal(reached["kwargs"]["p0"], [5.0, 5.0])
    np.testing.assert_array_equal(reached["kwargs"]["bounds"][1], [5.0, 5.0])

    # Scalar bounds broadcast.
    with pytest.raises(_Reached):
        _run(x, y, p0=P0, bounds=(0, 5))
    np.testing.assert_array_equal(reached["kwargs"]["bounds"][0], [0.0, 0.0])

    # Wrong-length bounds/p0 raise a clear ValueError naming `bounds`.
    with pytest.raises(ValueError, match="`bounds` must have length"):
        _run(x, y, p0=P0, bounds=([0, 0, 0], [5, 5, 5]))
    with pytest.raises(ValueError, match="`bounds` must have length"):
        _run(x, y, p0=[1.0, 1.0, 1.0], bounds=([0, 0], [5, 5]))


def test_hybrid_sigma_passes_through_untouched(reached):
    x, y = _data()
    sigma = np.full(50, 0.1)
    with pytest.raises(_Reached):
        _run(x, y, p0=P0, sigma=sigma)
    np.testing.assert_array_equal(reached["kwargs"]["sigma"], sigma)

    # nan_policy="omit" drops the bad row from sigma as well as from x and y.
    y_nan = y.copy()
    y_nan[3] = np.nan
    sigma = np.linspace(0.1, 0.2, 50)
    with pytest.raises(_Reached):
        _run(x, y_nan, p0=P0, sigma=sigma, nan_policy="omit")
    np.testing.assert_array_equal(reached["kwargs"]["sigma"], np.delete(sigma, 3))
    assert reached["xy"][0].shape == (49,)


# --- The five accepted behavior differences ---------------------------------


def test_hybrid_p0_auto_now_estimates(reached, monkeypatch):
    """Was a UFuncTypeError (np.clip on the string 'auto')."""
    import nlsq.precision.parameter_estimation as pe

    calls = []
    estimate = np.array([1.234, 0.567])

    def fake_estimate(f, xdata, ydata, p0):
        calls.append(p0)
        return estimate

    monkeypatch.setattr(pe, "estimate_initial_parameters", fake_estimate)
    x, y = _data()
    with pytest.raises(_Reached):
        _run(x, y, p0="auto")
    assert calls == ["auto"]
    np.testing.assert_array_equal(reached["kwargs"]["p0"], estimate)


def test_hybrid_length_mismatch_raises_valueerror():
    """Was a TypeError (broadcast) deep in the fit."""
    x, y = _data()
    with pytest.raises(ValueError, match="X and Y data lengths dont match"):
        _run(x, y[:40], p0=P0)


def test_hybrid_empty_data_raises_valueerror():
    """Was success=True returning p0 unchanged."""
    with pytest.raises(ValueError, match="must not be empty"):
        _run(np.array([]), np.array([]), p0=P0)


def test_hybrid_nan_lower_bound_raises_valueerror():
    """Was success=True with x=[nan, 1]."""
    x, y = _data()
    with pytest.raises(ValueError, match="Lower bounds contain NaN"):
        _run(x, y, p0=P0, bounds=([np.nan, 0], [5, 5]))


def test_hybrid_nan_upper_bound_raises_valueerror():
    x, y = _data()
    with pytest.raises(ValueError, match="Upper bounds contain NaN"):
        _run(x, y, p0=P0, bounds=([0, 0], [np.nan, 5]))


def test_hybrid_non_array_x_raises_x_needs_arrays():
    """Was 'could not convert string to float'."""
    _, y = _data()
    with pytest.raises(ValueError, match="X needs arrays"):
        _run("abc", y, p0=P0)


# --- (n_points, n_features) x layout ----------------------------------------


def _model_2d(x, a, b):
    return a * x[:, 0] + b * x[:, 1]


@pytest.mark.slow
def test_hybrid_accepts_points_first_2d_x():
    """Hybrid takes x as (n_points, n_features); the standard path is feature-major."""
    from nlsq import curve_fit

    rng = np.random.default_rng(1)
    x = rng.uniform(0.5, 2.0, size=(200, 2))
    y = 1.5 * x[:, 0] + 0.7 * x[:, 1]
    res = curve_fit(_model_2d, x, y, p0=[1.0, 1.0], **_FAST)
    np.testing.assert_allclose(res.x, [1.5, 0.7], rtol=1e-3)


def test_prepare_problem_x_points_axis_last_flag():
    from nlsq.core.problem import PrepareContext, prepare_problem

    x = np.ones((200, 2))
    y = np.ones(200)
    with pytest.raises(ValueError, match="X and Y data lengths dont match"):
        prepare_problem(_model_2d, x, y, p0=[1.0, 1.0])
    out = prepare_problem(
        _model_2d, x, y, p0=[1.0, 1.0], ctx=PrepareContext(x_points_axis_last=False)
    )
    assert out.m_real == 200 and out.x.shape == (200, 2)
