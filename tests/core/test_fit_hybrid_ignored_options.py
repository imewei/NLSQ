"""fit(workflow=HybridStreamingConfig) must be loud about options it cannot honor."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nlsq import OptimizeWarning, fit
from nlsq.streaming.hybrid_config import HybridStreamingConfig

# Fast real hybrid fit (the default 200 warmup iterations takes ~60 s).
FAST_HYBRID = {
    "warmup_iterations": 5,
    "max_warmup_iterations": 10,
    "gauss_newton_max_iterations": 5,
}


def _linear(x, a, b):
    return a * x + b


@pytest.fixture
def data():
    x = np.linspace(0, 10, 200)
    return x, 2.0 * x + 1.0


def test_robust_loss_raises(data):
    x, y = data
    with pytest.raises(ValueError, match="robust loss"):
        fit(
            _linear,
            x,
            y,
            p0=[1.0, 1.0],
            workflow=HybridStreamingConfig(**FAST_HYBRID),
            loss="cauchy",
        )


def test_linear_loss_is_accepted(data):
    x, y = data
    r = fit(
        _linear,
        x,
        y,
        p0=[1.0, 1.0],
        workflow=HybridStreamingConfig(**FAST_HYBRID),
        loss="linear",
    )
    assert np.all(np.isfinite(np.asarray(r["x"])))


def test_other_ignored_options_warn_once(data):
    x, y = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(
            _linear,
            x,
            y,
            p0=[1.0, 1.0],
            workflow=HybridStreamingConfig(**FAST_HYBRID),
            f_scale=0.1,
            max_nfev=5,
        )
    msgs = [w for w in caught if issubclass(w.category, OptimizeWarning)]
    assert len(msgs) == 1
    text = str(msgs[0].message)
    assert "f_scale" in text
    assert "max_nfev" in text


def test_goal_injected_tolerances_do_not_warn(data):
    x, y = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit(
            _linear,
            x,
            y,
            p0=[1.0, 1.0],
            workflow=HybridStreamingConfig(**FAST_HYBRID),
            goal="quality",
        )
    assert not [w for w in caught if issubclass(w.category, OptimizeWarning)]
