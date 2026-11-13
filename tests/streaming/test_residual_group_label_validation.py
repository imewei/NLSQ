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


class _DeviceLikeArray:
    """Stands in for a device array: slicing is cheap, converting it whole is not."""

    def __init__(self, values):
        self._values = values
        self.shape = values.shape
        self.sliced = []

    def __getitem__(self, key):
        self.sliced.append(key)
        return self._values[key]

    def __array__(self, *args, **kwargs):
        raise AssertionError("whole array copied to the host")


def test_helper_copies_only_the_label_column():
    x, _ = _data([0.0, 1.0, 1.0, 0.0])
    spy = _DeviceLikeArray(x)
    _validate_group_labels(spy, np.array([1.0, 10.0]))
    assert len(spy.sliced) == 1  # exactly one slice: column 0


def test_helper_still_rejects_bad_labels_via_column_slice():
    x, _ = _data([0.0, 7.0])
    with pytest.raises(ValueError, match="group labels"):
        _validate_group_labels(_DeviceLikeArray(x), np.array([1.0, 10.0]))


def test_helper_rejects_1d_without_converting():
    spy = _DeviceLikeArray(np.arange(4.0))
    with pytest.raises(ValueError, match=r"shaped \(N, k\)"):
        _validate_group_labels(spy, np.array([1.0, 10.0]))
    assert not spy.sliced


def test_helper_accepts_jax_array_and_plain_list():
    x, _ = _data([0.0, 1.0, 1.0, 0.0])
    _validate_group_labels(jnp.asarray(x), np.array([1.0, 10.0]))
    _validate_group_labels(x.tolist(), np.array([1.0, 10.0]))
