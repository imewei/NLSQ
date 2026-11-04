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
    for g, w in zip(got, want, strict=True):
        _assert_same(g, w)


def _assert_same(got, want):
    """Deep-compare arrays, dicts (infodict), strings and ints."""
    if isinstance(want, dict):
        assert isinstance(got, dict) and got.keys() == want.keys()
        for key in want:
            _assert_same(got[key], want[key])
    elif isinstance(want, str | int | bool | type(None)):
        assert got == want
    else:
        np.testing.assert_allclose(np.asarray(got), np.asarray(want))


def test_hybrid_rejects_engine_kwargs():
    x, y = _data()
    with pytest.raises(TypeError, match="Unrecognized keyword argument"):
        _call(Hybrid(), x, y, verbose=0)


def test_exact_forwards_explicit_numpy_scalar_tolerance_verbatim(monkeypatch):
    # np.float32(1e-8) == 1e-8 is True, so a value-only != default check drops it
    from nlsq import curve_fit_large
    from nlsq.streaming import exact_lm

    seen = []
    real = exact_lm.exact_lm_fit

    def spy(f, x, y, p0, **kwargs):
        seen.append(kwargs)
        return real(f, x, y, p0, **kwargs)

    monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", spy)
    x, y = _data()
    tol = np.float32(1e-8)
    _call(Exact(gtol=tol), x, y)
    curve_fit_large(model, x, y, p0=[1.5, 0.4], method="exact_streaming", gtol=tol)
    assert len(seen) == 2
    for kw in seen:
        assert "gtol" in kw
        assert type(kw["gtol"]) is np.float32


def _spy_exact(monkeypatch):
    from nlsq.streaming import exact_lm

    seen = []
    real = exact_lm.exact_lm_fit

    def spy(f, x, y, p0, **kwargs):
        seen.append(kwargs)
        return real(f, x, y, p0, **kwargs)

    monkeypatch.setattr("nlsq.streaming.exact_lm.exact_lm_fit", spy)
    return seen


_SOLVER_KEYS = {"ftol", "xtol", "gtol", "max_nfev", "n_points"}


def test_default_exact_forwards_no_solver_kwargs(monkeypatch):
    seen = _spy_exact(monkeypatch)
    x, y = _data()
    _call(Exact(chunk_size=500), x, y)
    (kw,) = seen
    assert not _SOLVER_KEYS & set(kw)
    assert (
        kw["chunk_size"] == 500
    )  # engine argument, passed once, not as a solver kwarg


def test_exact_forwards_non_default_settings_and_kwargs_override(monkeypatch):
    seen = _spy_exact(monkeypatch)
    x, y = _data()
    _call(Exact(ftol=1e-6, xtol=1e-7, max_nfev=50), x, y)
    _call(Exact(ftol=1e-6), x, y, ftol=1e-9)
    kw, kw_override = seen
    assert (kw["ftol"], kw["xtol"], kw["max_nfev"]) == (1e-6, 1e-7, 50)
    assert "gtol" not in kw
    assert kw_override["ftol"] == 1e-9


def test_in_memory_kwargs_override_options(monkeypatch):
    import nlsq

    seen = {}

    def spy(f, x, y, **kwargs):
        seen.update(kwargs)
        return np.ones(2), np.eye(2)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    x, y = _data()
    _call(InMemory(options={"ftol": 1e-6, "xtol": 1e-7}), x, y, ftol=1e-9)
    assert seen["ftol"] == 1e-9
    assert seen["xtol"] == 1e-7


def test_legacy_tolerances_reach_curve_fit_in_memory(monkeypatch):
    import nlsq

    seen = {}

    def spy(f, x, y, **kwargs):
        seen.update(kwargs)
        return np.ones(2), np.eye(2)

    monkeypatch.setattr(nlsq, "curve_fit", spy)
    x, y = _data()
    nlsq.curve_fit_large(model, x, y, p0=[1.5, 0.4], ftol=1e-9, xtol=1e-8, gtol=1e-7)
    assert (seen["ftol"], seen["xtol"], seen["gtol"]) == (1e-9, 1e-8, 1e-7)


def test_chunked_kwargs_override_options(monkeypatch):
    from types import SimpleNamespace

    from nlsq.streaming.large_dataset import LargeDatasetFitter

    seen = {}

    def stub(self, f, x, y, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(popt=np.ones(2), pcov=np.eye(2))

    monkeypatch.setattr(LargeDatasetFitter, "fit", stub)
    x, y = _data()
    _call(Chunked(options={"ftol": 1e-6, "xtol": 1e-7}), x, y, ftol=1e-9)
    assert seen["ftol"] == 1e-9
    assert seen["xtol"] == 1e-7


@pytest.mark.parametrize(
    ("xdata_shape", "n_points", "threshold", "expected"),
    [
        ((49,), 49, 50, "in_memory"),
        ((50,), 50, 50, "chunked"),
        # 2-D xdata is (k, M): the sample count is shape[-1], not shape[0]
        ((2, 60), 60, 60, "chunked"),
        ((2, 60), 60, 61, "in_memory"),
    ],
)
def test_auto_dispatches_on_sample_count(
    monkeypatch, xdata_shape, n_points, threshold, expected
):
    from nlsq.streaming import large_fit
    from nlsq.streaming.specs import Auto

    called = []
    monkeypatch.setattr(
        large_fit, "_fit_in_memory", lambda *a, **k: called.append("in_memory")
    )
    monkeypatch.setattr(
        large_fit, "_fit_chunked", lambda *a, **k: called.append("chunked")
    )
    _call(
        Auto(size_threshold=threshold),
        np.zeros(xdata_shape),
        np.zeros(n_points),
    )
    assert called == [expected]
