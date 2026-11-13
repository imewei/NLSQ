"""Masked rows with sigma 0/NaN must not poison the fit."""

import numpy as np
import pytest


@pytest.mark.parametrize("bad", [0.0, np.nan])
def test_masked_rows_with_bad_sigma_are_excluded(bad):
    import jax.numpy as jnp

    from nlsq import curve_fit

    x = np.linspace(0, 4, 100)
    y = 2 * np.exp(-x) + 0.01 * np.random.default_rng(0).normal(size=100)
    mask = np.ones(100, bool)
    mask[[5, 6, 7]] = False
    sigma = np.full(100, 0.1)
    sigma[5] = bad
    popt, _ = curve_fit(
        lambda x, a, b: a * jnp.exp(-b * x),
        x,
        y,
        p0=[1, 1],
        sigma=sigma,
        data_mask=mask,
        check_finite=False,
    )
    ref, _ = curve_fit(
        lambda x, a, b: a * jnp.exp(-b * x),
        x[mask],
        y[mask],
        p0=[1, 1],
        sigma=np.full(97, 0.1),
    )
    np.testing.assert_allclose(popt, ref, rtol=1e-6)
