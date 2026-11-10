"""R6-01/03/04/06: round-5 regression fixes in the streaming package."""

import warnings

import numpy as np
import pytest

from nlsq.result import OptimizeWarning
from nlsq.streaming.large_dataset import LargeDatasetFitter, LDMemoryConfig


def lin(x, a, b):
    return a * x + b


def test_chunk_precision_is_scale_invariant():
    # diagonal, perfectly conditioned in correlation terms, 1e8 apart in scale
    prec = LargeDatasetFitter._chunk_precision(np.diag([1e-20, 1e-4]), 2)
    assert prec is not None
    np.testing.assert_allclose(np.diag(prec), [1e20, 1e4])


def test_chunk_precision_nonpositive_diagonal_is_unusable():
    assert LargeDatasetFitter._chunk_precision(np.diag([1.0, 0.0]), 2) is None


@pytest.mark.serial
def test_badly_scaled_parameters_use_all_chunks():
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1e9, 8000)
    y = 2e-9 * x + 1.0 + 0.01 * rng.standard_normal(x.size)
    fitter = LargeDatasetFitter(
        config=LDMemoryConfig(max_chunk_size=2000, min_chunk_size=1000)
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = fitter.fit(lin, x, y, p0=[1e-9, 0.5])
    msgs = [str(w.message) for w in rec if issubclass(w.category, OptimizeWarning)]
    assert not any("unusable covariance" in m for m in msgs), msgs
    # Compare with the truth: full-batch curve_fit itself stalls at this data
    # scale (R6-08, scipy-parity trust-region limitation).
    np.testing.assert_allclose(res.popt, [2e-9, 1.0], rtol=1e-2)


def test_streaming_hybrid_ints_validated_and_coerced():
    cfg = LDMemoryConfig(streaming_max_epochs=3.0, gc_chunk_interval=2.0)
    assert isinstance(cfg.streaming_max_epochs, int)
    assert isinstance(cfg.gc_chunk_interval, int)
    with pytest.raises(Exception, match="streaming_max_epochs"):
        LDMemoryConfig(streaming_max_epochs=3.5)
    with pytest.raises(Exception, match="gc_chunk_interval"):
        LDMemoryConfig(gc_chunk_interval=2.5)
    with pytest.raises(Exception, match="streaming_max_epochs"):
        LDMemoryConfig(streaming_max_epochs=0)


@pytest.mark.serial
def test_unusable_cov_warning_attributed_to_caller():
    def model(x, a, b):
        return (a * b) * x

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 4000)
    y = 3.0 * x + 0.1 * rng.standard_normal(x.size)
    fitter = LargeDatasetFitter(config=LDMemoryConfig(max_chunk_size=1000))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fitter.fit(model, x, y, p0=[1.0, 2.0])
    hits = [w for w in rec if "unusable covariance" in str(w.message)]
    assert hits
    assert all(
        not w.filename.replace("\\", "/").endswith("nlsq/streaming/large_dataset.py")
        for w in hits
    ), [w.filename for w in hits]


def test_hybrid_warning_prefix_is_package_dir_with_separator():
    import os

    from nlsq.core.covariance import _NLSQ_DIR

    assert _NLSQ_DIR[0].endswith(os.sep)


@pytest.mark.parametrize("bad", [0.0, np.nan, np.inf])
def test_chunked_large_rejects_bad_sigma_instead_of_dropping_chunk(bad):
    """A single bad sigma entry used to discard a whole chunk silently (R9)."""
    from nlsq import curve_fit_large

    rng = np.random.default_rng(0)
    n = 2001
    x = np.linspace(0, 1, n)
    y = 2 * x + 1 + 0.01 * rng.standard_normal(n)
    s = np.ones(n)
    s[0] = bad
    with pytest.raises(ValueError, match="sigma"):
        curve_fit_large(
            lambda x, a, b: a * x + b,
            x,
            y,
            p0=[1, 0],
            sigma=s,
            chunk_size=500,
            size_threshold=10,
        )
