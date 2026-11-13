import jax.numpy as jnp
import numpy as np

from nlsq import curve_fit


def test_numeric_mask_has_same_covariance_as_boolean_mask():
    x = np.arange(4.0)
    y = np.array([1.0, 3.0, 8.0, 0.0])

    def model(x, a, b):
        return a * x + b + jnp.zeros_like(x)

    _, numeric = curve_fit(model, x, y, p0=[1, 1], data_mask=[1, 2, 1, 0])
    _, boolean = curve_fit(model, x, y, p0=[1, 1], data_mask=[True, True, True, False])
    np.testing.assert_allclose(numeric, boolean)
