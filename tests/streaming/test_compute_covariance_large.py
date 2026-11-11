"""``compute_covariance=False`` on the curve_fit_large paths."""

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit_large
from nlsq.streaming import Auto, Chunked, Exact, Hybrid, InMemory
from nlsq.streaming.large_dataset import LargeDatasetFitter

EXP = PROBLEMS["exp_decay"]
N = 5_000


def model(x, a, b, c):
    return EXP.model(jnp, x, a, b, c)


@pytest.fixture(scope="module")
def data():
    return EXP.data(N)


def test_in_memory_pcov_is_none(data):
    popt, pcov = curve_fit_large(
        model, *data, p0=EXP.p0, strategy=InMemory(), compute_covariance=False
    )
    assert pcov is None
    assert np.all(np.isfinite(popt))


def test_chunked_pcov_is_none_popt_unchanged_chunk_covariances_kept(data, monkeypatch):
    strategy = Chunked(chunk_size=1000)
    on_popt, on_pcov = curve_fit_large(model, *data, p0=EXP.p0, strategy=strategy)
    assert on_pcov is not None

    recorded = []
    original = LargeDatasetFitter._chunk_precision

    def recording(pcov_chunk, n_params):
        recorded.append(pcov_chunk)
        return original(pcov_chunk, n_params)

    monkeypatch.setattr(LargeDatasetFitter, "_chunk_precision", staticmethod(recording))
    popt, pcov = curve_fit_large(
        model, *data, p0=EXP.p0, strategy=strategy, compute_covariance=False
    )
    assert pcov is None
    np.testing.assert_array_equal(popt, on_popt)
    assert recorded and all(c is not None for c in recorded)


def test_exact_pcov_is_none(data):
    popt, pcov = curve_fit_large(
        model,
        *data,
        p0=EXP.p0,
        strategy=Exact(chunk_size=1000),
        compute_covariance=False,
    )
    assert pcov is None
    assert np.all(np.isfinite(popt))


def test_exact_default_still_returns_pcov(data):
    _, pcov = curve_fit_large(model, *data, p0=EXP.p0, strategy=Exact(chunk_size=1000))
    assert pcov.shape == (3, 3)


@pytest.mark.parametrize("strategy_cls", [Chunked, InMemory])
def test_options_cannot_carry_compute_covariance(strategy_cls):
    with pytest.raises(TypeError):
        strategy_cls(options={"compute_covariance": False})


@pytest.mark.parametrize(
    "route",
    [{"method": "hybrid_streaming"}, {"strategy": Hybrid()}],
    ids=["method", "strategy"],
)
def test_hybrid_rejects_false(data, route):
    with pytest.raises(ValueError, match="compute_covariance=False is not supported"):
        curve_fit_large(model, *data, p0=EXP.p0, compute_covariance=False, **route)


def test_in_memory_hybrid_method_rejects_false(data):
    with pytest.raises(ValueError, match="compute_covariance"):
        curve_fit_large(
            model,
            *data,
            p0=EXP.p0,
            strategy=InMemory(method="hybrid_streaming"),
            compute_covariance=False,
        )


def test_auto_below_threshold_hybrid_method_rejects_false(data):
    auto = Auto(size_threshold=N + 1, in_memory=InMemory(method="hybrid_streaming"))
    with pytest.raises(ValueError, match="compute_covariance"):
        curve_fit_large(
            model, *data, p0=EXP.p0, strategy=auto, compute_covariance=False
        )


@pytest.mark.slow
def test_in_memory_cmaes_pcov_is_none_without_covariance_svd(monkeypatch):
    from nlsq.core import minpack
    from nlsq.global_optimization import CMAESConfig

    calls = []
    real = minpack._covariance_svd_jit
    monkeypatch.setattr(
        minpack, "_covariance_svd_jit", lambda J: calls.append(1) or real(J)
    )
    cma = InMemory(
        method="cmaes",
        options={"cmaes_config": CMAESConfig(max_generations=5, seed=0)},
    )
    kw = {
        "p0": EXP.p0,
        "bounds": ([0.0, 0.0, -10.0], [10.0, 10.0, 10.0]),
        "strategy": cma,
    }
    data = EXP.data(500)
    curve_fit_large(model, *data, **kw)  # control: the spy is live with the flag on
    assert calls
    calls.clear()
    popt, pcov = curve_fit_large(model, *data, compute_covariance=False, **kw)
    assert pcov is None and calls == []
    assert np.all(np.isfinite(popt))
