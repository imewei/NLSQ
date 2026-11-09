import numpy as np
import pytest

from nlsq import CurveFit

pytestmark = pytest.mark.serial


def test_enable_stability_does_not_cap_iterations_or_loosen_tolerances():
    """Stability mode must not inject max_nfev/ftol/xtol the user did not set."""
    rng = np.random.default_rng(0)
    n = 150_000  # >100k points: auto-selection used to cap max_nfev at 100
    x = np.linspace(0, 10, n)
    y = 2.0 * (x + 1.0) ** -1.5 + 0.01 * rng.standard_normal(n)

    def model(x, a, b, c):
        return a * (x + c) ** b

    p0 = [1.0, 3.0, 0.2]
    plain = CurveFit(enable_stability=False).curve_fit(model, x, y, p0=p0)
    stab = CurveFit(enable_stability=True).curve_fit(model, x, y, p0=p0)
    np.testing.assert_allclose(stab.popt, plain.popt, rtol=1e-3)
