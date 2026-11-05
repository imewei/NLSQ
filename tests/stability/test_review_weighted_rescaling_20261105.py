"""Weighted robust objectives must survive fallback data rescaling."""

from types import SimpleNamespace

import numpy as np
import pytest

from nlsq.stability.fallback import _fit_rescaled


@pytest.mark.parametrize("sigma", [None, np.ones(4), np.eye(4)])
def test_rescaling_preserves_weighted_robust_loss_threshold(sigma):
    x = np.arange(4.0)
    y = np.array([0.0, 2.0, 4.0, 20.0])

    def model(x, slope):
        return slope * x

    calls = []

    def capture_fit(f, xc, yc, **kwargs):
        calls.append((f, xc, yc, kwargs))
        return SimpleNamespace(x=np.array([2.0]))

    _fit_rescaled(
        capture_fit,
        model,
        x,
        y,
        {"p0": [2.0], "sigma": sigma, "loss": "soft_l1", "f_scale": 2.0},
        {"_x_scale": 3.0, "_y_scale": 20.0, "_x_offset": 0.0, "_y_offset": 0.0},
    )
    f, xc, yc, kw = calls[0]
    residual = y - model(x, 2.0)
    scaled_residual = yc - f(xc, 2.0)
    if sigma is None:
        original_z = residual / 2.0
        scaled_z = scaled_residual / kw["f_scale"]
    elif sigma.ndim == 1:
        original_z = residual / sigma / 2.0
        scaled_z = scaled_residual / kw["sigma"] / kw["f_scale"]
    else:
        original_z = np.linalg.solve(np.linalg.cholesky(sigma), residual) / 2.0
        scaled_z = (
            np.linalg.solve(np.linalg.cholesky(kw["sigma"]), scaled_residual)
            / kw["f_scale"]
        )
    np.testing.assert_allclose(scaled_z, original_z)
