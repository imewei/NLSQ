"""curve_fit_large must not build an invalid chunk range for small inputs or a
small explicit chunk_size (min_chunk_size used to be floored at 1000 regardless)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n):
    x = np.linspace(0, 5, n)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * np.random.default_rng(0).normal(size=n)
    return x, y


@pytest.mark.parametrize(
    ("n", "chunk_size"),
    [(2000, None), (20_000, 500)],
    ids=["small-dataset", "small-explicit-chunk"],
)
def test_forced_large_path_accepts_small_chunk_bounds(n, chunk_size):
    x, y = _data(n)
    popt, _ = curve_fit_large(
        _model,
        x,
        y,
        p0=[1.0, 1.0],
        auto_size_detection=False,
        chunk_size=chunk_size,
    )
    np.testing.assert_allclose(popt, [2.0, 0.7], atol=0.1)


class _Stop(Exception):
    """Raised by the spy so the test sees the config without running a fit."""


def _bounds_seen_by_fitter(monkeypatch, n, chunk_size):
    from nlsq.streaming.large_dataset import LargeDatasetFitter

    seen = []

    def spy(self, *args, **kwargs):
        seen.append((self.config.min_chunk_size, self.config.max_chunk_size))
        raise _Stop

    monkeypatch.setattr(LargeDatasetFitter, "fit", spy)
    x, y = _data(n)
    with pytest.raises(_Stop):
        curve_fit_large(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            auto_size_detection=False,
            chunk_size=chunk_size,
        )
    return seen[0]


@pytest.mark.parametrize("n", [2, 5, 9])
def test_tiny_inputs_get_one_point_chunks(monkeypatch, n):
    assert _bounds_seen_by_fitter(monkeypatch, n, None) == (1, 1)


def test_explicit_chunk_size_is_the_ceiling_and_caps_the_floor(monkeypatch):
    assert _bounds_seen_by_fitter(monkeypatch, 20_000, 500) == (500, 500)


def test_chunk_size_above_the_default_ceiling_is_kept(monkeypatch):
    assert _bounds_seen_by_fitter(monkeypatch, 20_000, 1_500_000) == (
        1000,
        1_500_000,
    )


@pytest.mark.parametrize("bad", [0, -5])
def test_non_positive_chunk_size_is_rejected_by_name(bad):
    x, y = _data(20_000)
    with pytest.raises(ValueError, match=r"^chunk_size must be positive"):
        curve_fit_large(
            _model,
            x,
            y,
            p0=[1.0, 1.0],
            auto_size_detection=False,
            chunk_size=bad,
        )
