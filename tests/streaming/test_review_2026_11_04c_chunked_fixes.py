"""Regression tests for EXA1 / EXA2 / HYB3 (2026-11-04 streaming review)."""

import dataclasses
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.streaming import Chunked
from nlsq.streaming.large_dataset import LargeDatasetFitter


def _lin(x, a, b):
    return a * x + b


def _weighted_chunked(inject_failure):
    n = 6000
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, n)
    y = 2 * x + 1 + 100 * rng.standard_normal(n) * np.where(x > 0.5, 1, 0.01)
    sigma = np.full(n, 100.0)
    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    real = fitter.curve_fit.curve_fit
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        if inject_failure and calls["n"] == 2:
            raise RuntimeError("injected")
        return real(*a, **k)

    fitter.curve_fit.curve_fit = wrapped
    stats = dataclasses.replace(
        fitter.estimate_requirements(n, 2), recommended_chunk_size=1000, n_chunks=6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fitter._fit_chunked(
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
    return res.x, np.sqrt(np.diag(res["pcov"]))


def test_weighted_chunk_retry_keeps_sigma():
    """EXA1: a retried chunk must be fit (and its SSR scored) with sigma."""
    popt0, perr0 = _weighted_chunked(False)
    popt1, perr1 = _weighted_chunked(True)
    np.testing.assert_allclose(popt1, popt0, rtol=0.5, atol=0.5)
    np.testing.assert_allclose(perr1, perr0, rtol=0.2)


def test_curve_fit_large_chunked_failure_raises():
    """EXA2: a failed chunked fit raises instead of returning p0 + inf pcov."""

    def bad(x, a, b, c):
        return a * jnp.log(x - b) + c

    x = np.linspace(0, 1, 6000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="chunked fit failed"):
            curve_fit_large(
                bad, x + 1, 2 * x, p0=[1.0, 5.0, 0.0], strategy=Chunked(chunk_size=1000)
            )


@pytest.mark.slow
def test_hybrid_accepts_scipy_layout_2d_xdata():
    """HYB3: curve_fit_large takes (k, M) xdata; the hybrid engine wants (n, k)."""
    rng = np.random.default_rng(0)
    m = 4000
    xs = rng.uniform(0, 1, (2, m))

    def f(x, a, b):
        return a * x[0] + b * x[1]

    y = f(xs, 2.0, -3.0) + 0.01 * rng.normal(size=m)
    popt, _ = curve_fit_large(
        f,
        xs,
        y,
        p0=[1.0, 1.0],
        method="hybrid_streaming",
        verbose=0,
        warmup_iterations=5,
        max_warmup_iterations=10,
        gauss_newton_max_iterations=5,
    )
    np.testing.assert_allclose(popt, [2.0, -3.0], atol=0.05)


# --- STR1: 2-D (k, M) xdata must chunk along the points axis -----------------


def _lin2d(x, a, b):
    return a * x[0] + b * x[1]


def _data_2d(m=60000, seed=3):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1, (2, m))
    y = 2.0 * xs[0] + 3.0 * xs[1] + 0.01 * rng.normal(size=m)
    return xs, y


def _chunk_counter(fitter):
    seen = {"n_chunks": None, "sizes": []}
    real_est = fitter.estimate_requirements
    real_cf = fitter.curve_fit.curve_fit

    def est(n_points, n_params):
        stats = real_est(n_points, n_params)
        seen["n_points"], seen["n_chunks"] = n_points, stats.n_chunks
        return stats

    def cf(f, x, y, *a, **k):
        seen["sizes"].append(np.shape(x))
        return real_cf(f, x, y, *a, **k)

    fitter.estimate_requirements = est
    fitter.curve_fit.curve_fit = cf
    return seen


def test_chunked_2d_xdata_chunks_along_points_axis():
    xs, y = _data_2d()
    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    seen = _chunk_counter(fitter)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitter._fit_implementation(
            _lin2d, xs, y, None, (-np.inf, np.inf), "trf", "auto", False
        )
    assert seen["n_points"] == 60000
    # the per-chunk curve_fit still receives scipy layout (k, chunk)
    assert all(s[0] == 2 for s in seen["sizes"])


@pytest.mark.serial
def test_chunked_2d_matches_full_batch_with_chunk_size_and_sigma(monkeypatch):
    from nlsq import curve_fit

    n_chunks = []
    real_est = LargeDatasetFitter.estimate_requirements

    def est(self, n_points, n_params):
        stats = real_est(self, n_points, n_params)
        n_chunks.append(stats.n_chunks)
        return stats

    monkeypatch.setattr(LargeDatasetFitter, "estimate_requirements", est)

    xs, y = _data_2d()
    sig = np.full(y.size, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p2, c2 = curve_fit_large(
            _lin2d,
            xs,
            y,
            p0=[1.0, 1.0],
            sigma=sig,
            absolute_sigma=True,
            strategy=Chunked(chunk_size=10000),
        )
        p1, c1 = curve_fit(_lin2d, xs, y, p0=[1.0, 1.0], sigma=sig, absolute_sigma=True)
    assert n_chunks[-1] == 6
    np.testing.assert_allclose(p2, [2.0, 3.0], atol=0.01)
    np.testing.assert_allclose(p2, p1, rtol=1e-6)
    np.testing.assert_allclose(c2, c1, rtol=1e-6)


def test_chunked_2d_retry_path():
    xs, y = _data_2d(m=6000)
    fitter = LargeDatasetFitter(memory_limit_gb=1.0)
    real = fitter.curve_fit.curve_fit
    calls = {"n": 0}

    def wrapped(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected")
        return real(*a, **k)

    fitter.curve_fit.curve_fit = wrapped
    stats = dataclasses.replace(
        fitter.estimate_requirements(6000, 2), recommended_chunk_size=1000, n_chunks=6
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fitter._fit_chunked(
            _lin2d,
            xs.T,
            y,
            np.array([1.0, 1.0]),
            (-np.inf, np.inf),
            "trf",
            "auto",
            False,
            stats,
        )
    np.testing.assert_allclose(res.x, [2.0, 3.0], atol=0.05)
