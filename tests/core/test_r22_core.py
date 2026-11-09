"""Round-22 core regressions."""

import numpy as np

from nlsq import curve_fit
from nlsq.core.functions import exponential_decay, sigmoid


def test_sigmoid_default_bounds_allow_falling_curve():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = 2.0 / (1 + np.exp(1.5 * (x - 5.0))) + 0.5 + 0.01 * rng.standard_normal(80)
    lb, ub = sigmoid.bounds()
    assert lb[2] == -np.inf
    popt, _ = curve_fit(sigmoid, x, y, p0=sigmoid.estimate_p0(x, y), bounds=(lb, ub))
    assert np.sum((np.asarray(sigmoid(x, *popt)) - y) ** 2) < 0.1


def test_exponential_p0_half_life_measured_from_decay_start():
    x = np.linspace(20, 30, 50)
    y = 3.0 * np.exp(-0.5 * (x - 20)) + 1.0
    b0 = exponential_decay.estimate_p0(x, y)[1]
    assert 0.3 < b0 < 0.8


def test_three_dim_xdata_points_last_axis_with_nan_omit():
    rng = np.random.default_rng(0)
    X = rng.random((2, 3, 30))

    def f(X, a, b):
        return a * X[0].sum(0) + b * X[1].sum(0)

    y = f(X, 1.5, -0.5) + 0.01 * rng.standard_normal(30)
    popt, _ = curve_fit(f, X, y, p0=[1, 1])
    np.testing.assert_allclose(popt, [1.5, -0.5], atol=0.1)
    y[4] = np.nan
    popt2, _ = curve_fit(f, X, y, p0=[1, 1], nan_policy="omit")
    np.testing.assert_allclose(popt2, [1.5, -0.5], atol=0.1)


def test_exponential_decay_auto_p0_handles_rising_curve():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 6, 80)
    y = -5.0 * np.exp(-0.7 * x) + 10.0 + 0.02 * rng.standard_normal(80)
    a0, b0, c0 = exponential_decay.estimate_p0(x, y)
    assert a0 < 0 and b0 > 0 and abs(c0 - 10) < 1
    popt, _ = curve_fit(exponential_decay, x, y, p0="auto")
    np.testing.assert_allclose(popt, [-5.0, 0.7, 10.0], atol=0.1)
