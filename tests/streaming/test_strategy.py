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


class _LazySource:
    """Array-like with shape/len/slicing whose conversion to ndarray is forbidden."""

    shape = (10,)

    def __len__(self):
        return 10

    def __getitem__(self, key):
        raise AssertionError("check_exact_sources must not read the source")

    def __array__(self, *args, **kwargs):
        raise AssertionError("check_exact_sources must not convert the source")


def test_check_exact_sources_rejects_2d_xdata():
    from nlsq.streaming.strategy import check_exact_sources

    with pytest.raises(ValueError, match="must be 1-D for method='exact_streaming'"):
        check_exact_sources(np.zeros((4, 2)), np.zeros(4), None, None)


def test_check_exact_sources_never_converts_lazy_source():
    from nlsq.streaming.strategy import check_exact_sources

    check_exact_sources(_LazySource(), _LazySource(), _LazySource(), None)


def test_normalize_p0_bounds_defaults_p0_from_signature():
    from nlsq.streaming.strategy import normalize_p0_bounds

    def f(x, a, b):
        return a * x + b

    p0, (lb, ub) = normalize_p0_bounds(f, None, (-np.inf, np.inf))
    np.testing.assert_array_equal(p0, np.ones(2))
    assert lb.shape == ub.shape == (2,)


def test_chunked_unpack_without_pcov_raises_runtime_error():
    class OnlyX:
        x = np.ones(2)

    strategy = build_large_strategy("chunked", LDMemoryConfig(memory_limit_gb=1.0))
    with pytest.raises(RuntimeError, match="Unexpected result format"):
        strategy.unpack(OnlyX())


def test_chunked_unpack_returns_popt_and_pcov():
    class Res:
        popt = np.ones(2)
        pcov = np.eye(2)

    strategy = build_large_strategy("chunked", LDMemoryConfig(memory_limit_gb=1.0))
    res = Res()
    popt, pcov = strategy.unpack(res)
    assert popt is res.popt
    assert pcov is res.pcov


def test_hybrid_unpack_without_pcov_returns_inf_matrix():
    strategy = build_large_strategy("hybrid", HybridStreamingConfig())
    x, pcov = strategy.unpack({"x": np.ones(2)})
    assert pcov.shape == (2, 2)
    assert np.isinf(pcov).all()
    np.testing.assert_array_equal(x, np.ones(2))


def test_exact_unpack_raises_exact_solver_error_with_result():
    from nlsq.streaming.exact_lm import ExactSolverError

    native = {"success": False, "termination": "stalled", "message": "no progress"}
    with pytest.raises(
        ExactSolverError, match=r"^exact_streaming fit failed \(stalled\)"
    ) as excinfo:
        build_large_strategy("exact", None).unpack(native)
    assert excinfo.value.result is native


def test_exact_unpack_returns_x_and_pcov_on_success():
    native = {"success": True, "x": np.ones(2), "pcov": np.eye(2)}
    x, pcov = build_large_strategy("exact", None).unpack(native)
    assert x is native["x"]
    assert pcov is native["pcov"]
