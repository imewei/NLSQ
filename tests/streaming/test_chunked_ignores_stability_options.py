"""Characterization: the chunked path of ``curve_fit_large`` ignores stability options."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq.core import minpack
from nlsq.streaming.specs import Auto, Chunked, InMemory

N = 600
X = np.linspace(0.0, 4.0, N)
Y = 2.5 * np.exp(-1.3 * X) + 0.01 * np.random.default_rng(0).standard_normal(N)


def _model(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.mark.parametrize("show_progress", [False, True])
def test_chunk_fits_receive_no_stability_keywords(monkeypatch, show_progress) -> None:
    """Pins the documented decision that the chunked path ignores these options."""
    seen: list[dict] = []
    real = minpack.CurveFit.curve_fit

    def spy(self, *args, **kwargs):
        seen.append(kwargs)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(minpack.CurveFit, "curve_fit", spy)
    nlsq.curve_fit_large(
        _model,
        X,
        Y,
        p0=[2.0, 1.0],
        # exact=None: the default large path is exact streaming, not chunk fits
        strategy=Auto(
            auto_size_detection=False,
            in_memory=InMemory(
                stability="check",
                rescale_data=False,
                max_jacobian_elements_for_svd=123,
            ),
            chunked=Chunked(chunk_size=200),
            exact=None,
        ),
        show_progress=show_progress,
    )
    assert len(seen) >= 2
    for kwargs in seen:
        assert not {"stability", "rescale_data", "max_jacobian_elements_for_svd"} & set(
            kwargs
        )
