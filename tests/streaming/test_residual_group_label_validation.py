"""R3: residual-weighting group labels are validated on the host."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.warmup import _validate_group_labels


def _model(x, a, b):
    return a * x[:, 1] + b


def _data(labels):
    n = len(labels)
    x = np.stack([np.asarray(labels, dtype=float), np.linspace(0, 1, n)], axis=1)
    return x, 2.0 * x[:, 1] + 1.0


@pytest.mark.parametrize("bad", [99.0, -1.0, 1.9, np.nan, np.inf])
def test_helper_rejects_invalid_labels(bad):
    x, _ = _data([0.0, 1.0, bad, 0.0])
    with pytest.raises(ValueError, match="group labels"):
        _validate_group_labels(x, np.array([1.0, 10.0]))


def test_helper_accepts_valid_labels():
    x, _ = _data([0.0, 1.0, 1.0, 0.0])
    _validate_group_labels(x, np.array([1.0, 10.0]))


@pytest.mark.parametrize("bad", [99.0, -1.0, 1.9])
def test_phase1_warmup_rejects_invalid_labels(bad):
    config = HybridStreamingConfig(
        normalize=False,
        enable_residual_weighting=True,
        residual_weights=[1.0, 10.0],
        verbose=0,
    )
    opt = AdaptiveHybridStreamingOptimizer(config)
    x, y = _data([0.0, 1.0] * 10 + [bad])
    with pytest.raises(ValueError, match="group labels"):
        opt._run_phase1_warmup((x, y), _model, jnp.array([1.0, 1.0]), None)
