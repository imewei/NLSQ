"""Warmup loss evaluations at the dataset level (initial / final) run in
``config.chunk_size``-row pieces and match the single full-data call."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.phases.warmup import WarmupPhase, _chunked_loss

N, CHUNK = 50, 7  # 8 chunks, the last one partial


def _model(x, a, b):
    return a * x[:, 1] + b


def _data():
    rng = np.random.default_rng(0)
    group = np.arange(N) % 3
    t = np.linspace(0.0, 1.0, N)
    y = 2.0 * t + 1.0 + 0.05 * rng.standard_normal(N)
    return jnp.asarray(np.stack([group, t], axis=1)), jnp.asarray(y)


def _config(weighted: bool, regularized: bool, **kw):
    return HybridStreamingConfig(
        normalize=False,
        chunk_size=CHUNK,
        enable_residual_weighting=weighted,
        residual_weights=np.array([1.0, 3.0, 0.5]) if weighted else None,
        enable_group_variance_regularization=regularized,
        group_variance_indices=[(0, 2)] if regularized else None,
        group_variance_lambda=0.7,
        **kw,
    )


class _Recorder:
    """loss_fn stand-in that records the rows each chunk call receives."""

    def __init__(self, loss_fn):
        self.inner, self.rows = loss_fn, []
        self.penalty = loss_fn.penalty

    def base_sums(self, params, x, y):
        self.rows.append(len(x))
        return self.inner.base_sums(params, x, y)

    def __call__(self, *a):  # pragma: no cover - must not be used for chunking
        raise AssertionError("full-data loss_fn called")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("regularized", [False, True])
def test_chunked_loss_matches_single_call_adaptive(weighted, regularized):
    opt = AdaptiveHybridStreamingOptimizer(_config(weighted, regularized))
    p0 = jnp.array([1.0, 0.5])
    opt._setup_normalization(_model, p0, None)
    loss_fn = opt._create_warmup_loss_fn()
    x, y = _data()

    rec = _Recorder(loss_fn)
    chunked = _chunked_loss(rec, p0, x, y, CHUNK)

    assert chunked == pytest.approx(float(loss_fn(p0, x, y)), rel=1e-12)
    assert max(rec.rows) <= CHUNK and sum(rec.rows) == N and len(rec.rows) > 1


@pytest.mark.parametrize("weighted", [False, True])
def test_phase1_exit_evaluates_final_loss_in_chunks(weighted, monkeypatch):
    cfg = _config(weighted, True, warmup_iterations=1, max_warmup_iterations=1)
    opt = AdaptiveHybridStreamingOptimizer(cfg)
    p0 = jnp.array([10.0, 10.0])
    x, y = _data()

    seen = []
    import nlsq.streaming.adaptive_hybrid as mod

    orig = mod._chunked_loss

    def spy(loss_fn, params, xd, yd, chunk_size):
        rec = _Recorder(loss_fn)
        out = orig(rec, params, xd, yd, chunk_size)
        seen.append((rec.rows, out, float(loss_fn(params, xd, yd))))
        return out

    monkeypatch.setattr(mod, "_chunked_loss", spy)
    result = opt._run_phase1_warmup((x, y), _model, p0, None)

    assert seen, "dataset-level loss evaluation did not go through _chunked_loss"
    for rows, chunked, single in seen:
        assert max(rows) <= CHUNK
        assert chunked == pytest.approx(single, rel=1e-12)
    assert result["final_loss"] == pytest.approx(seen[-1][2], rel=1e-12)


@pytest.mark.parametrize("weighted", [False, True])
def test_warmup_phase_loss_chunked_matches_single_call(weighted):
    class Model:
        normalizer = None

        def __call__(self, x, a, b):
            return _model(x, a, b)

    phase = WarmupPhase(_config(weighted, True), Model())
    if weighted:
        phase.set_residual_weights(jnp.array([1.0, 3.0, 0.5]))
    loss_fn = phase._create_loss_fn()
    p0 = jnp.array([1.0, 0.5])
    x, y = _data()

    rec = _Recorder(loss_fn)
    assert _chunked_loss(rec, p0, x, y, CHUNK) == pytest.approx(
        float(loss_fn(p0, x, y)), rel=1e-12
    )
    assert max(rec.rows) <= CHUNK and len(rec.rows) > 1
