import numpy as np
import pytest

from nlsq.core.functions import polynomial
from nlsq.core.minpack import fit


def test_polynomial_defaults_p0_inference():
    """Test that polynomial fit works without explicit p0 or bounds.

    This reproduces the 'Expected N coefficients, got 1' error where
    fit() incorrectly infers n_params=1 for *args signature.
    """
    # Degree 2 polynomial: y = c0*x^2 + c1*x + c2 (3 params)
    model = polynomial(2)

    x = np.linspace(-5, 5, 20)
    # y = 2*x^2 + 3*x + 1
    y = 2 * x**2 + 3 * x + 1 + np.random.normal(0, 0.1, 20)

    # Should work with p0='auto' logic which is default behavior if p0 is None?
    # Wait, fit() default is p0=None.

    try:
        # workflow='auto' is default
        result = fit(model, x, y, p0=None)

        # Check coefficients
        assert len(result.popt) == 3
        # Roughly Check values
        assert np.allclose(result.popt, [2, 3, 1], rtol=0.2, atol=0.5)

    except ValueError as e:
        pytest.fail(f"Polynomial fit failed with ValueError: {e}")
    except RuntimeError as e:
        # minpack might raise RuntimeError on failure
        pytest.fail(f"Polynomial fit failed with RuntimeError: {e}")
