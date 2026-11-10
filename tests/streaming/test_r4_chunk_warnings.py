"""R4 SF-3: chunks with unusable covariance must be reported, not silently dropped."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit_large
from nlsq.result import OptimizeWarning
from nlsq.streaming.large_dataset import LargeDatasetFitter


def _fitter() -> LargeDatasetFitter:
    fitter = LargeDatasetFitter(memory_limit_gb=8.0)
    fitter._accum_information = None
    fitter._accum_info_vector = None
    fitter._n_unusable_cov_chunks = 0
    return fitter


def test_unusable_chunks_are_counted():
    fitter = _fitter()
    fitter._update_parameters_convergence(
        None, np.array([1.0, 2.0]), None, [], np.inf, 0, 2
    )
    fitter._update_parameters_convergence(
        np.array([1.0, 2.0]), np.array([1.1, 2.1]), None, [], np.inf, 1, 2
    )
    assert fitter._n_unusable_cov_chunks == 2


def test_usable_chunk_not_counted():
    fitter = _fitter()
    fitter._update_parameters_convergence(
        None, np.array([1.0, 2.0]), np.eye(2), [], np.inf, 0, 1
    )
    assert fitter._n_unusable_cov_chunks == 0


@pytest.mark.serial
def test_all_chunks_unusable_warns_end_to_end():
    """An unidentifiable parameter gives every chunk a zero-variance pcov."""

    def model(x, a, b, c):
        return a * jnp.exp(-b * x) + 0.0 * c

    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 40_000)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(x.size)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        curve_fit_large(
            model,
            x,
            y,
            p0=[1.0, 1.0, 0.5],
            size_threshold=1000,
            chunk_size=5000,
            show_progress=False,
        )
    msgs = [str(w.message) for w in rec if issubclass(w.category, OptimizeWarning)]
    assert any("unusable covariance" in m for m in msgs), msgs
