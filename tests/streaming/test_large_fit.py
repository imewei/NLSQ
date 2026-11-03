"""fit_with_strategy: one entry point over the four large-fit paths."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.result import CurveFitResult
from nlsq.streaming.large_fit import fit_with_strategy
from nlsq.streaming.specs import Chunked, Exact, Hybrid, InMemory

TRUE = np.array([2.0, 0.5])
N = 2000


def model(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    x = np.linspace(0.0, 5.0, N)
    y = TRUE[0] * np.exp(-TRUE[1] * x) + 0.001 * np.random.default_rng(0).normal(size=N)
    return x, y


def _call(strategy, x, y, p0=(1.5, 0.4), sigma=None, **kwargs):
    return fit_with_strategy(
        strategy,
        model,
        x,
        y,
        None if p0 is None else np.asarray(p0),
        sigma=sigma,
        absolute_sigma=False,
        bounds=(-np.inf, np.inf),
        check_finite=True,
        memory_limit_gb=None,
        show_progress=False,
        **kwargs,
    )


@pytest.mark.parametrize(
    "strategy",
    [InMemory(), Chunked(chunk_size=500), Hybrid(verbose=0), Exact()],
    ids=["in_memory", "chunked", "hybrid", "exact"],
)
def test_contract_every_strategy_returns_popt_pcov_shapes(strategy):
    x, y = _data()
    result = _call(strategy, x, y)
    popt, pcov = result
    assert np.shape(popt) == (2,)
    assert np.shape(pcov) == (2, 2)
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)
    if isinstance(strategy, InMemory):
        assert isinstance(result, CurveFitResult)


def test_exact_passes_memmaps_through_unconverted(tmp_path, monkeypatch):
    from nlsq.streaming import exact_lm

    x0, y0 = _data()
    maps = {}
    for name, arr in (("x", x0), ("y", y0), ("sigma", np.full(N, 0.01))):
        m = np.memmap(tmp_path / f"{name}.dat", dtype=float, mode="w+", shape=(N,))
        m[:] = arr
        m.flush()
        maps[name] = np.memmap(tmp_path / f"{name}.dat", dtype=float, mode="r")

    seen = []
    real = exact_lm.exact_lm_fit

    def spy(f, x, y, p0, **kwargs):
        seen.append((x, y, kwargs))
        return real(f, x, y, p0, **kwargs)

    monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", spy)
    popt, _ = _call(Exact(chunk_size=500), maps["x"], maps["y"], sigma=maps["sigma"])
    (got_x, got_y, kw) = seen[0]
    assert got_x is maps["x"]
    assert got_y is maps["y"]
    assert kw["sigma"] is maps["sigma"]
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)


def test_exact_accepts_chunk_factory():
    x0, y0 = _data()

    def factory(chunk_size):
        for i in range(0, N, chunk_size):
            yield x0[i : i + chunk_size], y0[i : i + chunk_size]

    popt, pcov = _call(Exact(chunk_size=500, n_points=N), factory, None)
    np.testing.assert_allclose(popt, TRUE, rtol=1e-2)
    assert np.shape(pcov) == (2, 2)


@pytest.mark.parametrize(
    "strategy",
    [InMemory(), Chunked(chunk_size=500), Hybrid(verbose=0)],
    ids=["in_memory", "chunked", "hybrid"],
)
def test_chunk_factory_with_non_exact_strategy_raises_value_error(strategy):
    def factory(chunk_size):
        yield np.zeros(2), np.zeros(2)

    with pytest.raises(ValueError, match="requires method='exact_streaming'"):
        _call(strategy, factory, None)


def test_exact_rejects_2d_xdata():
    x0, y0 = _data()
    with pytest.raises(ValueError, match="multi-feature"):
        _call(Exact(), np.vstack([x0, x0]), y0)


def test_in_memory_special_output_passes_through_unchanged():
    x, y = _data()
    p0 = np.array([1.5, 0.4])
    got = _call(InMemory(options={"full_output": True}), x, y, p0=p0)
    want = curve_fit(model, x, y, p0=p0, full_output=True)
    assert type(got) is tuple and len(got) == len(want)
    assert [type(g) for g in got] == [type(w) for w in want]
    np.testing.assert_allclose(got[0], want[0])
    np.testing.assert_allclose(got[1], want[1])
