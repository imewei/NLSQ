"""R7: non-finite data in hybrid streaming, exact-solver option rules, warning types."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq.streaming.exact_lm as lm
from nlsq import curve_fit
from nlsq.result import OptimizeWarning
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.validators import validate_non_negative_int, validate_positive_int


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.fixture
def data():
    x = np.linspace(0, 3, 200)
    return x, 2.0 * np.exp(-1.3 * x)


def test_hybrid_optimizer_raises_on_nan_y(data):
    x, y = data
    y = y.copy()
    y[17] = np.nan
    opt = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig())
    with pytest.raises(ValueError, match="not finite"):
        opt.fit((x, y), _exp, jnp.array([1.0, 1.0]), verbose=0)


def test_curve_fit_hybrid_raises_on_nan_even_without_check_finite(data):
    x, y = data
    y = y.copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        curve_fit(
            _exp, x, y, p0=[1.0, 1.0], method="hybrid_streaming", check_finite=False
        )


def test_exact_nan_tolerance_message_names_param(data):
    with pytest.raises(ValueError, match="gtol"):
        lm.exact_lm_fit(_exp, *data, [1.0, 1.0], gtol=float("nan"))


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_exact_lm_fit_rejects_bad_max_nfev(data, bad):
    with pytest.raises(ValueError, match="max_nfev"):
        lm.exact_lm_fit(_exp, *data, [1.0, 1.0], max_nfev=bad)


def test_exact_lm_fit_accepts_integral_float_max_nfev(data):
    assert lm.exact_lm_fit(_exp, *data, [1.0, 1.0], max_nfev=50.0).nfev <= 50


@pytest.mark.parametrize("bad", [True, 0, -3, 2.5, float("nan"), float("inf"), "4"])
def test_positive_int_validator_rejects(bad):
    with pytest.raises(ValueError):
        validate_positive_int(bad, "n")


def test_integer_validators_accept_integral_values():
    validate_positive_int(4, "n")
    validate_positive_int(4.0, "n")
    validate_positive_int(np.int64(4), "n")
    validate_non_negative_int(0, "n")
    with pytest.raises(ValueError):
        validate_non_negative_int(-1, "n")


def test_dropped_chunk_and_unusable_cov_warnings_are_optimize_warnings():
    import inspect

    from nlsq.streaming import large_dataset

    src = inspect.getsource(large_dataset)
    assert "chunks failed and were dropped" in src
    head = src.split("chunks failed and were dropped")[1][:400]
    assert "OptimizeWarning" in head and "UserWarning" not in head
    assert "numerically positive definite (singular, indefinite or" in src
