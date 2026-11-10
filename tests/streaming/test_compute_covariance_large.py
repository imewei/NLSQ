"""``compute_covariance=False`` on the curve_fit_large paths."""

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.compare.problems import PROBLEMS
from nlsq import curve_fit_large
from nlsq.streaming import Chunked, Exact, InMemory
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


def test_hybrid_rejects_false(data):
    with pytest.raises(ValueError, match="compute_covariance=False is not supported"):
        curve_fit_large(
            model,
            *data,
            p0=EXP.p0,
            method="hybrid_streaming",
            compute_covariance=False,
        )
