"""R5-01: a singular PSD chunk covariance must not become a garbage precision."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.result import OptimizeWarning
from nlsq.streaming.large_dataset import LargeDatasetFitter
from nlsq.streaming.large_dataset import LDMemoryConfig as _LDMemoryConfig


def test_singular_psd_pcov_is_unusable():
    # rank-1 PSD: np.linalg.inv does not raise but returns ~1e16+ indefinite junk
    v = np.array([1.0, 2.0])
    pcov = np.outer(v, v) * (1.0 + 1e-17)
    pcov[0, 0] -= 1e-16
    assert LargeDatasetFitter._chunk_precision(pcov, 2) is None


def test_exactly_singular_pcov_is_unusable():
    assert LargeDatasetFitter._chunk_precision(np.ones((2, 2)), 2) is None


def test_indefinite_pcov_is_unusable():
    assert LargeDatasetFitter._chunk_precision(np.diag([1.0, -1.0]), 2) is None


def test_well_conditioned_pcov_gives_inverse():
    pcov = np.array([[2.0, 0.3], [0.3, 1.0]])
    prec = LargeDatasetFitter._chunk_precision(pcov, 2)
    np.testing.assert_allclose(prec, np.linalg.inv(pcov))


@pytest.mark.serial
def test_unidentifiable_product_gives_sane_combined_result():
    """Model (a*b)*x: pcov is rank-1 in every chunk."""

    def model(x, a, b):
        return (a * b) * x

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 10_000)
    y = 3.0 * x + 0.1 * rng.standard_normal(x.size)
    fitter = LargeDatasetFitter(config=_LDMemoryConfig(max_chunk_size=1000))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fitter.fit(model, x, y, p0=[1.0, 2.0])
    msgs = [str(w.message) for w in rec if issubclass(w.category, OptimizeWarning)]
    assert any("unusable covariance" in m for m in msgs), msgs
    pcov = np.asarray(res.pcov)
    finite = np.isfinite(np.diag(pcov))
    assert np.all(np.diag(pcov)[finite] >= 0), pcov
    assert abs(res.popt[0] * res.popt[1] - 3.0) < 0.5, res.popt


@pytest.mark.serial
def test_normal_ten_chunk_fit_unchanged_and_quiet():
    def model(x, a, b):
        return a * jnp.exp(-b * x)

    rng = np.random.default_rng(1)
    x = np.linspace(0.0, 4.0, 10_000)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(x.size)
    fitter = LargeDatasetFitter(config=_LDMemoryConfig(max_chunk_size=1000))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fitter.fit(model, x, y, p0=[1.0, 1.0])
    assert not [w for w in rec if "unusable covariance" in str(w.message)]
    np.testing.assert_allclose(res.popt, [2.0, 0.7], rtol=0.02)
    assert np.all(np.diag(res.pcov) > 0)
