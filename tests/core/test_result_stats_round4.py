"""Round-4 regressions for CurveFitResult statistics.

* confidence_band / prediction_interval use the exact model Jacobian
  (jax.jacfwd) instead of a forward difference whose absolute step floor
  swamped tiny parameters.
* AIC/BIC use the sigma-weighted chi-square when sigma is supplied.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from nlsq import curve_fit


def _hand_band_halfwidth(model, x, res):
    J = np.asarray(
        jax.jacfwd(lambda q: model(jnp.asarray(x), *q))(jnp.asarray(res.popt))
    )
    dof = len(res.ydata) - len(res.popt)
    t = stats.t.ppf(0.975, dof)
    return t * np.sqrt(np.einsum("ij,jk,ik->i", J, res.pcov, J))


def _decay(x, a, b):
    return a * jnp.exp(-b * x)


def _rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def test_confidence_band_small_parameter_exact_jacobian():
    xb = np.linspace(0, 1e8, 50)
    yb = 2 * np.exp(-1e-8 * xb) + 0.01 * np.random.default_rng(5).standard_normal(50)
    res = curve_fit(_decay, xb, yb, p0=[1.0, 1.5e-8])
    assert abs(res.popt[1]) < 1e-7  # the regime the old abs-floor step broke
    hw = _hand_band_halfwidth(_decay, xb, res)
    f0 = np.asarray(_decay(xb, *res.popt))
    lo, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-8
    assert _rel(f0 - lo, hw) <= 1e-8


def test_confidence_band_ordinary_scale_exact_jacobian():
    x = np.linspace(0, 5, 60)
    y = 3 * np.exp(-0.7 * x) + 0.02 * np.random.default_rng(1).standard_normal(60)
    res = curve_fit(_decay, x, y, p0=[1.0, 1.0])
    hw = _hand_band_halfwidth(_decay, x, res)
    f0 = np.asarray(_decay(x, *res.popt))
    _, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-10
    # prediction interval shares the Jacobian
    s2 = np.sum((y - f0) ** 2) / (len(x) - 2)
    t = stats.t.ppf(0.975, len(x) - 2)
    hw_pi = np.sqrt((hw / t) ** 2 + s2) * t
    pi = res.prediction_interval()
    assert _rel(pi[:, 1] - f0, hw_pi) <= 1e-10


def test_band_falls_back_for_non_traceable_model():
    """A numpy-only model cannot be traced; central differences take over."""
    x = np.linspace(0, 5, 40)
    y = 3 * np.exp(-0.7 * x) + 0.02 * np.random.default_rng(2).standard_normal(40)
    res = curve_fit(_decay, x, y, p0=[1.0, 1.0])
    hw = _hand_band_halfwidth(_decay, x, res)
    f0 = np.asarray(_decay(x, *res.popt))
    res["model"] = lambda x, a, b: a * np.exp(-b * np.asarray(x))  # untraceable
    _, hi = res.confidence_band()
    assert _rel(hi - f0, hw) <= 1e-6


def _lin(x, a, b):
    return a * x + b


def _quad(x, a, b, c):
    return a * x**2 + b * x + c


def _het_data(seed):
    n = 200
    x = np.linspace(0, 10, n)
    sig = 0.05 * (1 + x**2)
    y = 1 + 0.5 * x + sig * np.random.default_rng(seed).standard_normal(n)
    return x, y, sig


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_aic_bic_use_weighted_chi2_with_sigma(absolute_sigma):
    x, y, sig = _het_data(0)
    res = curve_fit(_quad, x, y, p0=[0, 1, 1], sigma=sig, absolute_sigma=absolute_sigma)
    n, k = len(y), 3
    chi2 = np.sum(((y - np.asarray(_quad(x, *res.popt))) / sig) ** 2)
    base = chi2 if absolute_sigma else n * np.log(chi2 / n)
    np.testing.assert_allclose(res.aic, base + 2 * k, rtol=1e-10)
    np.testing.assert_allclose(res.bic, base + k * np.log(n), rtol=1e-10)


def test_aic_with_covariance_sigma_uses_whitened_chi2():
    x, y, sig = _het_data(1)
    C = np.diag(sig**2)
    C[0, 1] = C[1, 0] = 0.3 * sig[0] * sig[1]
    res = curve_fit(_lin, x, y, p0=[1, 1], sigma=C, absolute_sigma=True)
    r = y - np.asarray(_lin(x, *res.popt))
    chi2 = r @ np.linalg.solve(C, r)
    np.testing.assert_allclose(res.aic, chi2 + 4, rtol=1e-10)


def test_aic_bic_unweighted_unchanged():
    x, y, _ = _het_data(0)
    res = curve_fit(_lin, x, y, p0=[1, 1])
    n, k = len(y), 2
    rss = np.sum((y - np.asarray(_lin(x, *res.popt))) ** 2)
    assert res.aic == 2 * k + n * np.log(rss / n)
    assert res.bic == k * np.log(n) + n * np.log(rss / n)


@pytest.mark.parametrize("absolute_sigma", [True, False])
def test_heteroscedastic_aic_prefers_true_linear_model(absolute_sigma):
    # seed 3: unweighted-RSS AIC/BIC picked the quadratic; chi2 picks linear
    x, y, sig = _het_data(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = curve_fit(_lin, x, y, p0=[1, 1], sigma=sig, absolute_sigma=absolute_sigma)
        r2 = curve_fit(
            _quad, x, y, p0=[0, 1, 1], sigma=sig, absolute_sigma=absolute_sigma
        )
    assert r1.aic < r2.aic
    assert r1.bic < r2.bic
