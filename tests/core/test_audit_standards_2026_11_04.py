"""Public regression coverage for documented fit parameter inference."""

from functools import partial

import numpy as np
import pytest

from nlsq import curve_fit, curve_fit_large, fit


def _model_kwargs(x, slope, **kwargs):
    return slope * x


def _model_varargs(x, slope, *args):
    return slope * x


def _model_keyword_only(x, slope, *, offset=1.0):
    return slope * x + offset


@pytest.mark.parametrize("model", [_model_kwargs, _model_varargs, _model_keyword_only])
@pytest.mark.parametrize("route", ["curve_fit", "exact_streaming"])
def test_inferred_p0_counts_only_positional_fit_parameters(model, route):
    """Variadics and keyword-only options cannot create phantom fit parameters."""
    x = np.linspace(0.0, 1.0, 12)
    y = model(x, 2.0)
    if route == "curve_fit":
        result = curve_fit(model, x, y, stability=None)
    else:
        result = curve_fit_large(model, x, y, method=route, chunk_size=5)
    popt, pcov = result
    np.testing.assert_allclose(popt, [2.0], rtol=1e-6)
    assert pcov.shape == (1, 1)


@pytest.mark.parametrize("model", [_model_kwargs, _model_varargs, _model_keyword_only])
def test_streaming_preset_infers_only_positional_fit_parameters(model, monkeypatch):
    from nlsq.streaming.hybrid_config import HybridStreamingConfig

    # Keep the real engine and use a small batch: the preset's default 10K-row
    # warmup is unnecessary for a 12-point parameter-inference regression.
    monkeypatch.setattr(
        "nlsq.streaming.hybrid_config.HybridStreamingConfig",
        partial(
            HybridStreamingConfig,
            chunk_size=16,
            warmup_iterations=1,
            max_warmup_iterations=2,
            gauss_newton_max_iterations=2,
        ),
    )
    x = np.linspace(0.0, 1.0, 12)
    with pytest.warns(FutureWarning, match="preset"):
        result = fit(model, x, model(x, 1.0), preset="streaming", verbose=0)
    np.testing.assert_allclose(result.popt, [1.0], rtol=1e-5)
    assert result.pcov.shape == (1, 1)
