"""Direct CMAESOptimizer.fit must apply nan_policy to the global search too."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import CMAESConfig, CMAESOptimizer


def _f(x, a, b):
    return a * jnp.exp(-b * x)


def _data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 200)
    y = 2.0 * np.exp(-0.7 * x) + 0.01 * rng.standard_normal(200)
    return x, y


def _cfg(refine):
    return CMAESConfig(
        seed=3, max_generations=40, restart_strategy="none", refine_with_nlsq=refine
    )


@pytest.mark.parametrize("refine", [False, True])
def test_omit_drops_nan_rows_in_global_search(refine):
    x, y = _data()
    y_nan = y.copy()
    y_nan[[5, 77]] = np.nan
    keep = np.isfinite(y_nan)
    kw = {"p0": [1.0, 1.0], "bounds": ([0, 0], [10, 10])}
    got = CMAESOptimizer(_cfg(refine)).fit(_f, x, y_nan, nan_policy="omit", **kw)
    ref = CMAESOptimizer(_cfg(refine)).fit(_f, x[keep], y[keep], **kw)
    np.testing.assert_allclose(np.asarray(got["popt"]), np.asarray(ref["popt"]))
    assert np.all(np.isfinite(np.asarray(got["popt"])))


def test_default_policy_raises_on_nan():
    x, y = _data()
    y[3] = np.nan
    with pytest.raises(ValueError):
        CMAESOptimizer(_cfg(False)).fit(
            _f, x, y, p0=[1.0, 1.0], bounds=([0, 0], [10, 10])
        )


def test_omit_keeps_data_mask_aligned():
    x, y = _data()
    y[3] = np.nan
    mask = np.ones(200, bool)
    mask[100:] = False
    kw = {"p0": [1.0, 1.0], "bounds": ([0, 0], [10, 10])}
    got = CMAESOptimizer(_cfg(False)).fit(
        _f, x, y, nan_policy="omit", data_mask=mask, **kw
    )
    keep = mask & np.isfinite(y)
    ref = CMAESOptimizer(_cfg(False)).fit(_f, x[keep], y[keep], **kw)
    np.testing.assert_allclose(np.asarray(got["popt"]), np.asarray(ref["popt"]))


def test_direct_omit_handles_row_wise_multivariable_data():
    x = np.arange(40.0).reshape(20, 2)
    y = 2 * x[:, 0] + 3 * x[:, 1]
    y[2] = np.nan
    model = lambda x, a, b: a * x[:, 0] + b * x[:, 1]
    kw = {"p0": [1.0, 1.0], "bounds": ([0, 0], [10, 10])}
    got = CMAESOptimizer(_cfg(False)).fit(model, x, y, nan_policy="omit", **kw)
    keep = np.isfinite(y)
    ref = CMAESOptimizer(_cfg(False)).fit(model, x[keep], y[keep], **kw)
    np.testing.assert_allclose(got["popt"], ref["popt"])


def test_direct_omit_rejects_ambiguous_square_sample_axis():
    x = np.arange(16.0).reshape(4, 4)
    y = np.array([1.0, np.nan, 3.0, 4.0])
    with pytest.raises(ValueError, match="sample axis is ambiguous"):
        CMAESOptimizer(_cfg(False)).fit(
            _f, x, y, p0=[1.0, 1.0], bounds=([0, 0], [10, 10]), nan_policy="omit"
        )


def test_direct_omit_handles_higher_dimensional_samples():
    x = np.arange(120.0).reshape(20, 2, 3)
    y = x.sum(axis=(1, 2))
    y[2] = np.nan
    model = lambda x, a: a * x.sum(axis=(1, 2))
    kw = {"p0": [1.0], "bounds": ([0], [2])}
    got = CMAESOptimizer(_cfg(False)).fit(model, x, y, nan_policy="omit", **kw)
    keep = np.isfinite(y)
    ref = CMAESOptimizer(_cfg(False)).fit(model, x[keep], y[keep], **kw)
    np.testing.assert_allclose(got["popt"], ref["popt"])


@pytest.mark.parametrize("bad", [0.0, float("nan"), float("inf")])
def test_cmaes_rejects_bad_sigma_like_trf(bad):
    """curve_fit(method='cmaes') must not return p0/success on 0/NaN/inf sigma."""
    from nlsq import curve_fit

    x = np.linspace(0, 1, 30)
    y = 2.0 * x + 1.0
    sigma = np.ones_like(x)
    sigma[3] = bad
    with pytest.raises(ValueError, match=r"non-zero|finite"):
        curve_fit(
            lambda x, a, b: a * x + b,
            x,
            y,
            p0=[1.0, 0.0],
            sigma=sigma,
            bounds=([-10, -10], [10, 10]),
            method="cmaes",
            refine_with_nlsq=False,
        )
