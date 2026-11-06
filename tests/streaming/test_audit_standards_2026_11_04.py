"""Chunked fit public contract regression coverage."""

import numpy as np

from nlsq import curve_fit_large
from nlsq.streaming import Chunked


def _slope(x, a, **options):
    return a * x


def test_chunked_infers_positional_parameters_with_kwargs():
    x = np.linspace(0, 1, 12)
    popt, pcov = curve_fit_large(_slope, x, 2 * x, strategy=Chunked(chunk_size=6))
    np.testing.assert_allclose(popt, [2.0], rtol=1e-6)
    assert pcov.shape == (1, 1)
