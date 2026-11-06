"""B1: DataChunker.create_chunks(indices=...) gathers per chunk (no shuffled copies)."""

import dataclasses
import warnings

import numpy as np
import pytest

from nlsq.streaming.large_dataset import (
    ChunkBufferPool,
    DataChunker,
    LargeDatasetFitter,
)


def _lin(x, a, b):
    return a * x + b


def _make(shape, dtype, n):
    rng = np.random.default_rng(3)
    x = rng.standard_normal((n, *shape)).astype(dtype)
    y = rng.standard_normal(n)
    return x, y


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [((), np.float64), ((3,), np.float64), ((), np.float32)],
    ids=["1d", "2d", "float32"],
)
@pytest.mark.parametrize("use_pool", [False, True], ids=["nopool", "pool"])
@pytest.mark.parametrize("n, chunk_size", [(1000, 300), (1000, 250), (777, 512)])
def test_create_chunks_with_indices_equals_reference(
    shape, dtype, use_pool, n, chunk_size
):
    x, y = _make(shape, dtype, n)
    perm = np.random.default_rng(0).permutation(n)
    pool = (
        ChunkBufferPool(
            chunk_size=chunk_size,
            dtype=np.result_type(x.dtype, y.dtype),
            x_feature_shape=shape,
        )
        if use_pool
        else None
    )
    bounds = DataChunker.chunk_bounds(n, chunk_size)
    seen = 0
    for xc, yc, i, valid in DataChunker.create_chunks(
        x, y, chunk_size, buffer_pool=pool, indices=perm
    ):
        s, e = bounds[i]
        assert valid == e - s
        # padding rows (>= valid) gather x[0] and are deliberately not checked
        np.testing.assert_array_equal(xc[:valid], x[perm][s:e])
        np.testing.assert_array_equal(yc[:valid], y[perm][s:e])
        seen += valid
    assert seen == n


def test_same_permutation_as_before(monkeypatch):
    n = 6000
    captured = {}
    x = np.linspace(0, 1, n)
    y = 2 * x + 1
    real = DataChunker.create_chunks

    def spy(*a, **k):
        captured["indices"] = np.array(k["indices"])
        return real(*a, **k)

    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    stats = dataclasses.replace(
        fitter.estimate_requirements(n, 2), recommended_chunk_size=1000, n_chunks=6
    )
    monkeypatch.setattr(DataChunker, "create_chunks", staticmethod(spy))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter._fit_chunked(
            _lin,
            x,
            y,
            np.array([1.0, 1.0]),
            (-np.inf, np.inf),
            "trf",
            "auto",
            False,
            stats,
        )
    np.testing.assert_array_equal(
        captured["indices"], np.random.default_rng(0).permutation(n)
    )


def test_indices_and_shuffle_are_exclusive():
    x, y = _make((), np.float64, 100)
    with pytest.raises(ValueError, match="indices"):
        next(DataChunker.create_chunks(x, y, 50, shuffle=True, indices=np.arange(100)))


def _chunked_fixture():
    n = 6000
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, n)
    y = 2 * x + 1 + 0.1 * rng.standard_normal(n)
    sigma = 0.1 + 0.5 * x
    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    stats = dataclasses.replace(
        fitter.estimate_requirements(n, 2), recommended_chunk_size=1000, n_chunks=6
    )
    return fitter, stats, x, y, sigma


def _run(fitter, stats, x, y, sigma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter._fit_chunked(
            _lin,
            x,
            y,
            np.array([1.0, 1.0]),
            (-np.inf, np.inf),
            "trf",
            "auto",
            False,
            stats,
            sigma=sigma,
            absolute_sigma=True,
        )


def test_fit_chunked_sigma_follows_x(monkeypatch):
    fitter, stats, x, y, sigma = _chunked_fixture()
    calls = []
    real = fitter.curve_fit.curve_fit

    def spy(f, xc, yc, *a, **k):
        calls.append((np.array(xc), np.array(yc), np.array(k["sigma"])))
        return real(f, xc, yc, *a, **k)

    monkeypatch.setattr(fitter.curve_fit, "curve_fit", spy)
    _run(fitter, stats, x, y, sigma)
    perm = np.random.default_rng(0).permutation(len(x))
    assert len(calls) == 6
    for i, (xc, yc, sc) in enumerate(calls):
        s, e = i * 1000, (i + 1) * 1000
        np.testing.assert_array_equal(xc, x[perm][s:e])
        np.testing.assert_array_equal(yc, y[perm][s:e])
        np.testing.assert_array_equal(sc, 0.1 + 0.5 * xc)  # sigma == g(x)


_BAD_INDICES = {
    "bool_mask": np.ones(100, dtype=bool),
    "float": np.arange(100, dtype=np.float64),
    "2d": np.arange(100).reshape(10, 10),
    "wrong_length": np.arange(99),
    "negative": np.r_[-1, np.arange(1, 100)],
    "equals_n": np.r_[np.arange(99), 100],
}


@pytest.mark.parametrize("use_pool", [False, True], ids=["nopool", "pool"])
@pytest.mark.parametrize("name", list(_BAD_INDICES))
def test_invalid_indices_rejected_before_first_chunk(name, use_pool):
    x, y = _make((), np.float64, 100)
    pool = (
        ChunkBufferPool(chunk_size=50, dtype=np.float64, x_feature_shape=())
        if use_pool
        else None
    )
    gen = DataChunker.create_chunks(
        x, y, 50, buffer_pool=pool, indices=_BAD_INDICES[name]
    )
    with pytest.raises(ValueError):
        next(gen)  # nothing is yielded before the error


# Golden values recorded from BASE 409d44f4 (pre-B1, three shuffled full-size copies).
_GOLD_POPT = [2.009739160144365, 0.994933129932033]
_GOLD_PCOV = [
    [0.00018146726586760735, -4.172258101621211e-05],
    [-4.172258101621211e-05, 1.9589297726120452e-05],
]


def test_fit_chunked_result_unchanged():
    res = _run(*_chunked_fixture())
    np.testing.assert_allclose(res.x, _GOLD_POPT, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res["pcov"], _GOLD_PCOV, rtol=1e-12, atol=1e-12)


def test_fit_chunked_span_tripwire_returns_failed_result(monkeypatch):
    """A chunker/chunk_bounds disagreement is a RuntimeError caught by the outer
    handler of ``_fit_chunked``: the fit returns an unsuccessful result."""
    fitter, stats, x, y, sigma = _chunked_fixture()
    real = DataChunker.create_chunks

    def short(*a, **k):
        for xc, yc, i, valid in real(*a, **k):
            yield xc, yc, i, valid - 1

    monkeypatch.setattr(DataChunker, "create_chunks", staticmethod(short))
    res = _run(fitter, stats, x, y, sigma)
    assert res.success is False
    assert "chunk_bounds span" in res.message
    assert np.all(np.isinf(res["pcov"]))
