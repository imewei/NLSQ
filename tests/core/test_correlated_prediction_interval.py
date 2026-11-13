import jax.numpy as jnp
import numpy as np
from scipy.stats import t

from nlsq import curve_fit


def test_prediction_interval_uses_correlated_residual_chi_square():
    x = np.arange(4.0)
    y = np.array([0.0, 2.0, 1.0, 4.0])
    sigma = np.eye(4) * 0.5 + np.ones((4, 4)) * 0.5

    def model(x, a):
        return a + jnp.zeros_like(x)

    result = curve_fit(model, x, y, p0=[1.0], sigma=sigma)
    residual = y - result.popt[0]
    variance = residual @ np.linalg.solve(sigma, residual) / (len(y) - 1)
    expected_halfwidth = t.ppf(0.975, len(y) - 1) * np.sqrt(
        result.pcov[0, 0] + variance
    )
    interval = result.prediction_interval()
    np.testing.assert_allclose(interval[:, 1] - result.popt[0], expected_halfwidth)
