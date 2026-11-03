"""Unit tests for ``nlsq.streaming.strategy.build_large_strategy``.

The routing table (``test_large_strategy_routing.py``) pins every call site;
these tests pin the builder's own contract: input kwargs translation, native
result passthrough, and no mutation of the caller's config.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig
from nlsq.streaming.strategy import build_large_strategy

X = np.linspace(0.0, 1.0, 20)
Y = 2.0 * X
P0 = np.array([1.0])


def model(x, a):
    return a * jnp.asarray(x)


def _record(monkeypatch, cls, name):
    calls = []
    sentinel = object()

    def stub(self, *args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(cls, name, stub)
    return calls, sentinel


def test_hybrid_drops_unsupported_tolerances(monkeypatch):
    calls, sentinel = _record(monkeypatch, AdaptiveHybridStreamingOptimizer, "fit")
    cfg = HybridStreamingConfig()
    original_tol = cfg.gauss_newton_tol

    built = build_large_strategy("hybrid", cfg, gtol=1e-5)

    # gtol maps onto the config on a copy; the caller's config is untouched.
    assert built.engine.config is not cfg
    assert built.engine.config.gauss_newton_tol == 1e-5
    assert cfg.gauss_newton_tol == original_tol

    def cb(*_a, **_k):
        return None

    out = built.fit(
        model, X, Y, P0, gtol=1.0, ftol=2.0, xtol=3.0, callback=cb, verbose=0
    )
    assert out is sentinel
    ((args, kwargs),) = calls
    assert args == ()
    assert set(kwargs) == {"data_source", "func", "p0", "callback", "verbose"}
    assert kwargs["data_source"][0] is X
    assert kwargs["data_source"][1] is Y
    assert kwargs["func"] is model
    assert kwargs["p0"] is P0
    assert kwargs["callback"] is cb
    assert kwargs["verbose"] == 0


def test_hybrid_without_gtol_keeps_config_identity():
    cfg = HybridStreamingConfig()
    built = build_large_strategy("hybrid", cfg)
    assert built.engine.config is cfg


def test_hybrid_rejects_unknown_fit_kwarg():
    # Only the tolerances are translated; anything else reaches the engine's
    # strict signature and fails loudly, as it does when called directly.
    cfg = HybridStreamingConfig()
    built = build_large_strategy("hybrid", cfg)
    with pytest.raises(TypeError):
        built.fit(model, X, Y, P0, not_a_kwarg=1)


def test_chunked_returns_native_result_unchanged(monkeypatch):
    calls, sentinel = _record(monkeypatch, LargeDatasetFitter, "fit")
    built = build_large_strategy("chunked", None, memory_limit_gb=1.5)
    assert isinstance(built.engine, LargeDatasetFitter)
    assert built.fit(model, X, Y, P0) is sentinel
    assert len(calls) == 1


def test_chunked_show_progress_selects_fit_with_progress(monkeypatch):
    fit_calls, _ = _record(monkeypatch, LargeDatasetFitter, "fit")
    prog_calls, sentinel = _record(monkeypatch, LargeDatasetFitter, "fit_with_progress")
    built = build_large_strategy("chunked", None, show_progress=True)
    assert built.fit(model, X, Y, P0, method="trf") is sentinel
    assert fit_calls == []
    assert prog_calls[0][1] == {"p0": P0, "method": "trf"}


def test_chunked_forwards_common_kwargs_and_does_not_mutate_config(monkeypatch):
    calls, _ = _record(monkeypatch, LargeDatasetFitter, "fit")
    cfg = LDMemoryConfig(memory_limit_gb=2.0, min_chunk_size=100, max_chunk_size=500)
    before = dataclasses.asdict(cfg)

    built = build_large_strategy("chunked", cfg, memory_limit_gb=cfg.memory_limit_gb)
    assert built.engine.config is cfg

    # show_progress is only a build-time switch: as a fit kwarg it is forwarded.
    common = {"ftol": 1e-9, "gtol": 1e-7, "xtol": 1e-6, "show_progress": True}
    built.fit(model, X, Y, P0, **common)

    ((args, kwargs),) = calls
    assert args[0] is model
    assert args[1] is X
    assert args[2] is Y
    assert kwargs == {"p0": P0, **common}
    assert dataclasses.asdict(cfg) == before


@pytest.mark.parametrize("tol", ["ftol", "xtol"])
def test_hybrid_build_rejects_ftol_xtol(tol):
    # Only gtol has a build-time mapping (gauss_newton_tol).
    with pytest.raises(TypeError):
        build_large_strategy("hybrid", HybridStreamingConfig(), **{tol: 1e-6})


def test_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="kind"):
        build_large_strategy("bogus", None)  # type: ignore[arg-type]


def test_exact_build_looks_up_engine_at_build_time(monkeypatch):
    seen: dict = {}
    sentinel = object()

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", fake)
    strategy = build_large_strategy("exact", None)
    assert strategy.fit(model, X, Y, P0, chunk_size=10) is sentinel
    assert seen == {"chunk_size": 10}


def test_exact_build_rejects_config():
    with pytest.raises(TypeError):
        build_large_strategy("exact", object())


def test_exact_build_rejects_common_kwargs():
    with pytest.raises(TypeError):
        build_large_strategy("exact", None, ftol=1e-3)
