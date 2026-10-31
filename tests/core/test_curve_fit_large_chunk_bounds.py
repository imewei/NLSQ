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
