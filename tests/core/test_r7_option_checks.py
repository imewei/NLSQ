"""R7: one rule per option across curve_fit, fit, curve_fit_large, exact and hybrid."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit, curve_fit_large
from nlsq.core._option_checks import (
    coerce_absolute_sigma,
    validate_max_nfev,
    validate_tolerance,
)
from nlsq.core.least_squares import check_tolerance
from nlsq.result import OptimizeWarning


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.fixture
def data():
    x = np.linspace(0, 1, 60)
    return x, 2.0 * np.exp(-3.0 * x) + 1e-3 * np.sin(40 * x)


GOOD = [True, False, np.True_, np.False_, 1, 0, np.int64(1)]
BAD = ["false", None, 2, -1, 0.5, 1.0, 0.0, [True]]


@pytest.mark.parametrize("good", GOOD)
def test_coerce_accepts_bool_likes(good):
    out = coerce_absolute_sigma(good)
    assert type(out) is bool and out == bool(good)


@pytest.mark.parametrize("bad", BAD)
def test_coerce_rejects_everything_else(bad):
    with pytest.raises(TypeError, match="absolute_sigma"):
        coerce_absolute_sigma(bad)


@pytest.mark.parametrize("good", [1, 0])
def test_absolute_sigma_ints_accepted_at_every_entry(data, good):
    x, y = data
    s = 0.5 * np.ones_like(y)
    assert curve_fit(_exp, x, y, p0=[1, 1], sigma=s, absolute_sigma=good).success
    for kw in ({}, {"method": "exact_streaming"}):
        popt, _ = curve_fit_large(
            _exp, x, y, p0=[1, 1], sigma=s, absolute_sigma=good, **kw
        )
        assert np.allclose(popt, [2.0, 3.0], atol=0.1)


@pytest.mark.parametrize("bad", ["false", None, 2, 0.5])
def test_absolute_sigma_bad_rejected_at_every_entry(data, bad):
    x, y = data
    with pytest.raises(TypeError, match="absolute_sigma"):
        curve_fit(_exp, x, y, p0=[1, 1], absolute_sigma=bad)
    with pytest.raises(TypeError, match="absolute_sigma"):
        curve_fit_large(_exp, x, y, p0=[1, 1], absolute_sigma=bad)
    with pytest.raises(TypeError, match="absolute_sigma"):
        curve_fit_large(
            _exp, x, y, p0=[1, 1], absolute_sigma=bad, method="exact_streaming"
        )


def test_absolute_sigma_bad_rejected_by_direct_hybrid(data):
    from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    x, y = data
    opt = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig())
    with pytest.raises(TypeError, match="absolute_sigma"):
        opt.fit((x, y), _exp, jnp.array([1.0, 1.0]), absolute_sigma="no", verbose=0)


@pytest.mark.parametrize("val", [True, 0, 2.5, -1, "3"])
def test_max_nfev_rule_rejects(val):
    with pytest.raises(ValueError, match="max_nfev"):
        validate_max_nfev(val)


def test_max_nfev_rule_accepts():
    assert validate_max_nfev(None) is None
    assert validate_max_nfev(7) == 7
    v = validate_max_nfev(7.0)
    assert v == 7 and type(v) is int
    assert type(validate_max_nfev(np.int64(7))) is int


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_exact_spec_and_fit_reject_bad_max_nfev(data, bad):
    from nlsq.streaming import Exact

    x, y = data
    with pytest.raises(ValueError, match="max_nfev"):
        Exact(max_nfev=bad)
    with pytest.raises(ValueError, match="max_nfev"):
        curve_fit_large(_exp, x, y, p0=[1, 1], method="exact_streaming", max_nfev=bad)
    if (
        bad == 2.5
    ):  # TRF path deliberately accepts fractional (pins in test_pr29_regressions)
        return
    with pytest.raises(ValueError, match="max_nfev"):
        curve_fit(_exp, x, y, p0=[1, 1], max_nfev=bad)


def test_least_squares_coerces_integral_float_max_nfev(data):
    x, y = data
    assert curve_fit(_exp, x, y, p0=[1, 1], max_nfev=50.0).success


@pytest.mark.parametrize("name", ["ftol", "xtol", "gtol"])
def test_nan_tolerance_is_value_error_even_as_0d_array_on_cmaes(data, name):
    x, y = data
    with pytest.raises(ValueError, match=name):
        curve_fit(
            _exp,
            x,
            y,
            p0=[1, 1],
            method="cmaes",
            bounds=([0, 0], [10, 10]),
            **{name: np.array(np.nan)},
        )


def test_tolerance_warning_category_and_attribution():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        validate_tolerance("ftol", np.inf)
    assert [w.category for w in rec] == [OptimizeWarning]
    assert rec[0].filename == __file__  # attributed to the caller, not nlsq/core


def test_check_tolerance_inf_warns_as_optimize_warning_at_user_frame():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        check_tolerance(np.inf, 1e-8, 1e-8, "trf")
    assert rec and rec[0].category is OptimizeWarning
    assert rec[0].filename == __file__


def test_dogbox_not_recommended_in_error_message():
    import inspect

    import nlsq.core.minpack as mp

    assert "'trf' or 'dogbox'" not in inspect.getsource(mp)


@pytest.mark.parametrize("kind", ["np0d", "jax0d", "npint"])
def test_zero_dim_array_max_nfev_and_absolute_sigma_accepted(data, kind):
    # HEAD and scipy accept 0-d arrays; the shared validators must unwrap them.
    import jax.numpy as jnp

    x, y = data
    n = {"np0d": np.array(50), "jax0d": jnp.asarray(50), "npint": np.int64(50)}[kind]
    flag = {"np0d": np.array(True), "jax0d": jnp.asarray(True), "npint": np.True_}[kind]
    assert curve_fit(_exp, x, y, p0=[1, 1], max_nfev=n, absolute_sigma=flag).success


def test_exact_streaming_accepts_none_tolerance(data):
    x, y = data
    res = curve_fit_large(_exp, x, y, p0=[1, 1], method="exact_streaming", ftol=None)
    popt = res[0] if isinstance(res, tuple) else res.x
    assert np.allclose(popt, [2.0, 3.0], atol=1e-2)
