"""Regression tests for the 2026-11-04 streaming review findings."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming.validators import validate_residual_weighting_config


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _fitter():
    cfg = LDMemoryConfig(memory_limit_gb=1.0, min_chunk_size=500, max_chunk_size=500)
    return LargeDatasetFitter(config=cfg)


def _data(n=2000):
    x = np.linspace(0, 4, n)
    y = 2.0 * np.exp(-1.3 * x)
    return x, y


def test_hybrid_check_finite_rejects_nan():
    x, y = _data()
    y[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        curve_fit_large(
            _model, x, y, p0=[1.0, 1.0], method="hybrid_streaming", check_finite=True
        )


def test_hybrid_unpack_raises_on_failed_result():
    from nlsq.streaming.strategy import _HybridStrategy

    hybrid_strategy = _HybridStrategy.__new__(_HybridStrategy)
    native = {"x": np.ones(2), "success": False, "message": "boom"}
    with pytest.raises(RuntimeError, match="boom"):
        hybrid_strategy.unpack(native)


def test_chunked_check_finite_rejects_nan():
    x, y = _data()
    y[10] = np.nan
    fitter = _fitter()
    with pytest.raises(ValueError, match="finite"):
        fitter.fit(_model, x, y, p0=np.array([1.0, 1.0]), check_finite=True)


def test_chunked_dropped_chunks_warn():
    x, y = _data()
    y[10] = np.nan
    fitter = _fitter()
    with pytest.warns(UserWarning, match="dropped"):
        res = fitter.fit(_model, x, y, p0=np.array([1.0, 1.0]), check_finite=False)
    assert res.failure_summary["total_failures"] >= 1


def test_chunked_does_not_touch_global_numpy_rng():
    x, y = _data()
    y[10] = np.nan  # forces the perturbed-retry path
    np.random.seed(123)
    state = np.random.get_state()[1].copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fitter().fit(_model, x, y, p0=np.array([1.0, 1.0]))
    assert np.array_equal(np.random.get_state()[1], state)


def test_residual_weights_reject_inf():
    with pytest.raises(Exception, match=r"positive|finite"):
        validate_residual_weighting_config(True, np.array([1.0, np.inf]))
