"""R4 regressions: API validation holes, CMA-ES NaN handling, silent-failure warnings."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.result import OptimizeWarning


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


@pytest.fixture
def data():
    x = np.linspace(0, 1, 40)
    y = 2.0 * np.exp(-3.0 * x) + 1e-3 * np.sin(40 * x)
    return x, y


def _fit(data, **kw):
    x, y = data
    return curve_fit(_exp, x, y, p0=[1.0, 1.0], **kw)


class TestOptionValidation:
    def test_tr_solver_typo(self, data):
        with pytest.raises(ValueError, match="tr_solver"):
            _fit(data, tr_solver="lsm")

    @pytest.mark.parametrize("solver", [None, "exact", "lsmr"])
    def test_tr_solver_valid(self, data, solver):
        assert _fit(data, tr_solver=solver).success

    @pytest.mark.parametrize("bad", [True, "Auto", "autto"])
    def test_stability_invalid(self, data, bad):
        with pytest.raises(ValueError, match="stability"):
            _fit(data, stability=bad)

    @pytest.mark.parametrize("ok", [False, "auto", "check"])
    def test_stability_valid(self, data, ok):
        assert _fit(data, stability=ok).success

    def test_nan_tolerance(self, data):
        for name in ("ftol", "xtol", "gtol"):
            with pytest.raises(ValueError, match=name):
                _fit(data, **{name: float("nan")})

    def test_inf_tolerance_warns(self, data):
        with pytest.warns(UserWarning, match="immediately"):
            _fit(data, gtol=float("inf"))

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_f_scale_invalid(self, data, bad):
        with pytest.raises(ValueError, match="f_scale"):
            _fit(data, loss="huber", f_scale=bad)

    def test_negative_n_starts(self, data):
        with pytest.raises(ValueError, match="n_starts"):
            _fit(data, multistart=True, n_starts=-1)

    def test_method_typo_suggests(self, data):
        with pytest.raises(ValueError, match="cmaes"):
            _fit(data, method="cmaess")
        with pytest.raises(ValueError, match="hybrid_streaming"):
            _fit(data, method="hybrid-streaming")

    @pytest.mark.parametrize("bad", ["False", 2, 0.5, None])
    def test_absolute_sigma_non_bool(self, data, bad):
        with pytest.raises(TypeError, match="absolute_sigma"):
            _fit(data, sigma=0.5 * np.ones(40), absolute_sigma=bad)

    def test_absolute_sigma_np_bool_ok(self, data):
        assert _fit(data, sigma=0.5 * np.ones(40), absolute_sigma=np.True_).success

    def test_max_nfev_bool(self, data):
        with pytest.raises(ValueError, match="max_nfev"):
            _fit(data, max_nfev=True)


def _logistic(x, L, k, x0):
    return L / (1 + jnp.exp(-k * (x - x0)))


class TestCmaesNonFinite:
    @staticmethod
    def _problem():
        rng = np.random.default_rng(0)
        x = np.linspace(-3, 5, 50)
        y = np.asarray(_logistic(x, 4.0, 2.0, 1.0)) + 0.02 * rng.standard_normal(50)
        y[::7] = np.nan
        bounds = ([0.1, 0.1, -5.0], [20.0, 10.0, 5.0])
        return x, y, bounds

    def test_default_raises(self):
        x, y, bounds = self._problem()
        with pytest.raises(ValueError):
            curve_fit(_logistic, x, y, p0=[3, 1, 0.5], bounds=bounds, method="cmaes")

    @pytest.mark.slow
    def test_omit_matches_clean_fit(self):
        x, y, bounds = self._problem()
        ok = np.isfinite(y)
        ref = curve_fit(_logistic, x[ok], y[ok], p0=[3, 1, 0.5], bounds=bounds)
        res = curve_fit(
            _logistic,
            x,
            y,
            p0=[3, 1, 0.5],
            bounds=bounds,
            method="cmaes",
            nan_policy="omit",
        )
        np.testing.assert_allclose(res.popt, ref.popt, rtol=5e-2)

    @pytest.mark.slow
    def test_omit_with_sigma(self):
        x, y, bounds = self._problem()
        ok = np.isfinite(y)
        sig = np.full(50, 0.02)
        ref = curve_fit(
            _logistic, x[ok], y[ok], p0=[3, 1, 0.5], bounds=bounds, sigma=sig[ok]
        )
        res = curve_fit(
            _logistic,
            x,
            y,
            p0=[3, 1, 0.5],
            bounds=bounds,
            sigma=sig,
            method="cmaes",
            nan_policy="omit",
        )
        np.testing.assert_allclose(res.popt, ref.popt, rtol=5e-2)
        assert np.all(np.isfinite(res.xdata)) and len(res.ydata) == ok.sum()


class TestSilentFailureWarnings:
    def test_overflow_check_quiet_when_small(self, data):
        x, y = data
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            CurveFit(enable_overflow_check=True).curve_fit(_exp, x, y, p0=[1.0, 1.0])

    def test_full_rank_no_warning(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            _fit(data)
