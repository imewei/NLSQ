"""Gap-closing tests from the r3 test-quality audit (mutation survivors)."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from nlsq import curve_fit
from nlsq.core.least_squares import LeastSquares


def test_full_covariance_sigma_matches_scipy():
    """2-D ``sigma`` (full covariance) must use the lower Cholesky factor."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 4, 30)
    # strongly correlated AR(1) noise covariance: off-diagonals matter
    idx = np.arange(x.size)
    cov = 0.05**2 * 0.9 ** np.abs(idx[:, None] - idx[None, :])
    y = (
        2.0 * np.exp(-1.1 * x)
        + 0.3
        + np.linalg.cholesky(cov) @ rng.standard_normal(x.size)
    )

    popt, pcov = curve_fit(
        lambda x, a, b, c: a * jnp.exp(-b * x) + c, x, y, p0=[1.0, 1.0, 0.0], sigma=cov
    )
    ref, rcov = scipy.optimize.curve_fit(
        lambda x, a, b, c: a * np.exp(-b * x) + c, x, y, p0=[1.0, 1.0, 0.0], sigma=cov
    )
    np.testing.assert_allclose(popt, ref, rtol=1e-6)
    np.testing.assert_allclose(pcov, rcov, rtol=1e-4)


def test_least_squares_rejects_equal_bounds():
    """``lb == ub`` is invalid; only ``lb < ub`` is accepted."""
    with pytest.raises(ValueError, match="strictly less"):
        LeastSquares().least_squares(
            lambda p: p - 1.0, jnp.array([1.0, 1.0]), bounds=([1.0, 0.0], [1.0, 5.0])
        )
