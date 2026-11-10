import numpy as np
import pytest
from scipy.optimize import curve_fit

from nlsq import curve_fit_large
from nlsq.streaming import Chunked


@pytest.mark.parametrize("absolute_sigma", [False, True])
def test_chunked_scalar_sigma_matches_full_fit(absolute_sigma):
    x = np.arange(20.0)
    y = 2 * x + 1 + np.sin(x)
    model = lambda x, a, b: a * x + b
    expected, expected_cov = curve_fit(
        model, x, y, sigma=2.0, absolute_sigma=absolute_sigma
    )
    actual, actual_cov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.0, 0.0],
        sigma=2.0,
        absolute_sigma=absolute_sigma,
        strategy=Chunked(chunk_size=10),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5)


@pytest.mark.parametrize("sigma", [0.0, np.nan, np.inf])
def test_chunked_rejects_invalid_scalar_sigma(sigma):
    x = np.arange(20.0)
    with pytest.raises(ValueError, match=r"non-zero|finite"):
        curve_fit_large(
            lambda x, a: a * x,
            x,
            2 * x,
            p0=[1.0],
            sigma=sigma,
            strategy=Chunked(chunk_size=10),
        )


@pytest.mark.parametrize("weighted", [False, True])
def test_chunked_omit_uses_only_observed_rows_for_covariance(weighted):
    x = np.arange(20.0)
    y = 2 * x + 1 + np.sin(x)
    y[[1, 4, 9]] = np.nan
    sigma = None
    if weighted:
        x[2] = np.nan
        sigma = 1 + np.arange(20.0) / 20
        sigma[2] = 0
        sigma[3] = np.nan
    valid = np.isfinite(x) & np.isfinite(y)
    if sigma is not None:
        valid &= np.isfinite(sigma)
    model = lambda x, a, b: a * x + b
    expected, expected_cov = curve_fit(
        model, x[valid], y[valid], sigma=None if sigma is None else sigma[valid]
    )
    actual, actual_cov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.0, 0.0],
        sigma=sigma,
        strategy=Chunked(chunk_size=10, options={"nan_policy": "omit"}),
        check_finite=weighted,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5)


def test_chunked_skips_a_fully_masked_chunk():
    x = np.arange(20.0)
    y = 2 * x + 1 + np.sin(x)
    mask = np.zeros(20, dtype=bool)
    mask[np.random.default_rng(0).permutation(20)[:10]] = True
    model = lambda x, a, b: a * x + b
    expected, expected_cov = curve_fit(model, x[mask], y[mask])
    actual, actual_cov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.0, 0.0],
        strategy=Chunked(chunk_size=10, options={"data_mask": mask}),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5)


def test_chunked_omit_skips_chunks_without_finite_rows():
    x = np.arange(30.0)
    y = 2 * x + 1 + np.sin(x)
    y[np.random.default_rng(0).permutation(30)[:20]] = np.nan
    valid = np.isfinite(y)
    model = lambda x, a, b: a * x + b
    expected, expected_cov = curve_fit(model, x[valid], y[valid])
    actual, actual_cov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.0, 0.0],
        strategy=Chunked(chunk_size=10, options={"nan_policy": "omit"}),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5)


def test_chunked_omit_fails_when_no_observations_remain():
    x = np.arange(30.0)
    with pytest.raises(RuntimeError, match="No observations remain"):
        curve_fit_large(
            lambda x, a: a * x,
            x,
            np.full_like(x, np.nan),
            p0=[1.0],
            strategy=Chunked(chunk_size=10, options={"nan_policy": "omit"}),
        )


def test_chunked_omit_rejects_zero_sigma_on_observed_row():
    x = np.arange(20.0)
    sigma = np.ones(20)
    sigma[3] = 0
    with pytest.raises(ValueError, match="sigma values must be non-zero"):
        curve_fit_large(
            lambda x, a: a * x,
            x,
            2 * x,
            p0=[1.0],
            sigma=sigma,
            strategy=Chunked(chunk_size=10, options={"nan_policy": "omit"}),
        )


@pytest.mark.parametrize("omit", [False, True])
def test_chunked_mask_selects_the_same_rows_as_full_fit(omit):
    x = np.arange(20.0)
    y = 2 * x + 1 + np.sin(x)
    mask = x % 3 != 0
    sigma = 1 + x / 20
    if omit:
        y[1] = np.nan
        mask &= np.isfinite(y)
    model = lambda x, a, b: a * x + b
    expected, expected_cov = curve_fit(model, x[mask], y[mask], sigma=sigma[mask])
    actual, actual_cov = curve_fit_large(
        model,
        x,
        y,
        p0=[1.0, 0.0],
        sigma=sigma,
        strategy=Chunked(
            chunk_size=10,
            options={"data_mask": mask, "nan_policy": "omit" if omit else "raise"},
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5)
    np.testing.assert_allclose(actual_cov, expected_cov, rtol=1e-5)


@pytest.mark.parametrize(
    ("mask", "match"),
    [
        (np.ones(29, dtype=bool), "doesn't match"),
        (np.zeros(30, dtype=bool), "excludes all"),
    ],
    ids=["wrong_length", "all_excluded"],
)
def test_chunked_rejects_invalid_data_mask(mask, match):
    x = np.arange(30.0)
    y = 2 * x + 1
    with pytest.raises(ValueError, match=match):
        curve_fit_large(
            lambda x, a, b: a * x + b,
            x,
            y,
            p0=[1.0, 0.0],
            strategy=Chunked(chunk_size=10, options={"data_mask": mask}),
        )
