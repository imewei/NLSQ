"""R5 regressions: sigmoid Jacobian, overflow-warning threshold, CMA-ES kwarg checks."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import nlsq
from nlsq import CurveFit, curve_fit
from nlsq.core.functions import sigmoid
from nlsq.result import OptimizeWarning


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


class TestSigmoidStability:
    def test_steep_jacobian_is_finite(self):
        x = jnp.linspace(-10, 10, 201)
        jac = jax.jacfwd(lambda p: sigmoid(x, *p))(jnp.array([1.0, 0.3, 90.0, 0.0]))
        assert jnp.all(jnp.isfinite(jac))

    def test_values_match_logistic_formula(self):
        x = np.linspace(-3, 3, 50)
        L, x0, k, b = 2.0, 0.4, 3.0, -0.5
        ref = L / (1 + np.exp(-k * (x - x0))) + b
        np.testing.assert_allclose(
            sigmoid(jnp.asarray(x), L, x0, k, b), ref, atol=1e-14
        )

    def test_steep_fit_converges(self):
        x = np.linspace(-10, 10, 201)
        rng = np.random.default_rng(0)
        y = 1 / (1 + np.exp(-80 * (x - 0.3))) + 0.01 * rng.standard_normal(x.size)
        popt, _ = curve_fit(sigmoid, x, y, p0=[1.0, 0.0, 10.0, 0.0])
        assert np.all(np.isfinite(popt))
        assert np.sum((sigmoid(x, *popt) - y) ** 2) < 2e-2


class TestOverflowWarningThreshold:
    def test_quiet_when_nothing_is_clipped(self):
        x = np.linspace(0, 1, 30)
        y = 1e9 * np.exp(-0.7 * x)  # 1e8 < max|f| <= 1e10: no clipping occurs
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            CurveFit(enable_overflow_check=True).curve_fit(_exp, x, y, p0=[1e9, 0.5])


class TestCmaesKwargValidation:
    @pytest.fixture
    def data(self):
        x = np.linspace(0, 1, 40)
        return x, 2.0 * np.exp(-3.0 * x)

    @pytest.mark.parametrize(
        ("kw", "exc"),
        [
            ({"ftol": float("nan")}, ValueError),
            ({"xtol": float("nan")}, ValueError),
            ({"f_scale": 0}, ValueError),
            ({"tr_solver": "bogus"}, ValueError),
            ({"max_nfev": True}, ValueError),
        ],
    )
    def test_bad_kwarg_raises_on_cmaes_path(self, data, kw, exc):
        x, y = data
        with pytest.raises(exc):
            curve_fit(
                _exp,
                x,
                y,
                p0=[1.0, 1.0],
                bounds=([0, 0], [10, 10]),
                method="cmaes",
                **kw,
            )


class TestCurveFitLargeMethodMessage:
    def test_message_lists_exact_streaming(self):
        x = np.linspace(0, 1, 40)
        with pytest.raises(ValueError, match="exact_streaming"):
            nlsq.curve_fit_large(
                _exp, x, 2 * np.exp(-3 * x), p0=[1, 1], method="exact_streamin"
            )


class TestCmaesOmitLayout:
    def test_layout_mismatch_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(200, 2))  # points-first: not the (k, n) layout
        y = X[:, 0] + X[:, 1]
        y[3] = np.nan

        def model(x, a, b):
            return a * x[0] + b * x[1]

        with pytest.raises(ValueError):
            curve_fit(
                model,
                X,
                y,
                p0=[1.0, 1.0],
                bounds=([0, 0], [10, 10]),
                method="cmaes",
                nan_policy="omit",
            )


class TestWarningAttribution:
    def test_rank_deficient_warning_points_at_caller(self):
        def f(x, a, b, c):
            return a * jnp.exp(-b * x) + 0.0 * c

        x = np.linspace(0, 1, 200)
        y = 2.0 * np.exp(-0.7 * x) + 1e-3 * np.cos(30 * x)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            curve_fit(f, x, y, p0=[1.0, 1.0, 0.5])
        hits = [w for w in rec if "rank-deficient" in str(w.message)]
        assert hits
        assert all(w.filename == __file__ for w in hits)

    def test_overflow_warning_points_at_caller(self):
        x = np.linspace(0, 1, 30)
        y = 1e12 * np.exp(-0.7 * x)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            CurveFit(enable_overflow_check=True).curve_fit(_exp, x, y, p0=[5e11, 0.5])
        hits = [w for w in rec if "clipped" in str(w.message)]
        assert hits
        assert all(w.filename == __file__ for w in hits)
