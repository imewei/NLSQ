"""R3-4: hybrid multistart window must shift (not clip) at bounds."""

import jax.numpy as jnp
import numpy as np

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _cands(p0, lb, ub, scale_factor=1.0, n=64):
    cfg = HybridStreamingConfig(
        enable_checkpoints=False,
        verbose=0,
        enable_multistart=True,
        n_starts=n,
        center_on_p0=True,
        scale_factor=scale_factor,
    )
    opt = AdaptiveHybridStreamingOptimizer(cfg)
    opt.original_p0 = jnp.asarray(p0, dtype=jnp.float64)
    opt._generate_multistart_candidates((jnp.asarray(lb), jnp.asarray(ub)))
    return np.asarray(opt.multistart_candidates)


def test_corner_p0_covers_full_box_at_scale_one():
    c = _cands([0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    assert np.all(np.ptp(c, axis=0) > 0.9)


def test_corner_p0_preserves_window_width():
    c = _cands([0.0, 1.0], [0.0, 0.0], [1.0, 1.0], scale_factor=0.5)
    np.testing.assert_allclose(np.ptp(c, axis=0), 0.5, atol=0.05)
    assert np.all((c >= 0) & (c <= 1))


def test_p0_outside_box_does_not_collapse():
    c = _cands([2.0, -1.0], [0.0, 0.0], [1.0, 1.0])
    assert np.all(np.ptp(c, axis=0) > 0.9)
    assert np.all((c >= 0) & (c <= 1))


def test_infinite_bounds_unchanged_and_finite():
    c = _cands([1.0, 1.0], [0.0, -np.inf], [np.inf, np.inf], scale_factor=0.5)
    assert np.all(np.isfinite(c))
    assert np.all(c[:, 0] >= 0)
