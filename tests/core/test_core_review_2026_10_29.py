"""Regression tests for the 2026-10-29 core review (minpack / least_squares)."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import curve_fit as scipy_curve_fit

from nlsq import CurveFit, curve_fit


def _model(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def _model_np(x, a, b, c):
    return a * np.exp(-b * x) + c


def _data(m=40, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, m)
    y = 2.5 * np.exp(-1.3 * x) + 0.5 + 0.05 * rng.standard_normal(m)
    return x, y


def _r2(y, yp):
    return 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)


class TestPaddedAndMaskedStatistics:
    def test_flength_padding_excluded_from_result(self):
        x, y = _data()
        ref = CurveFit().curve_fit(_model, x, y, p0=[1, 1, 0])
        res = CurveFit(flength=64).curve_fit(_model, x, y, p0=[1, 1, 0])
        assert res.xdata.shape == (40,)
        assert res.ydata.shape == (40,)
        assert res.fun.shape == (40,)
        assert res.jac.shape == (40, 3)
        assert res.r_squared == pytest.approx(ref.r_squared, abs=1e-6)
        assert res.rmse == pytest.approx(ref.rmse, rel=1e-4)
        assert res.residuals.shape == (40,)

    def test_flength_full_output_fvec_length(self):
        x, y = _data()
        out = CurveFit(flength=64).curve_fit(
            _model, x, y, p0=[1, 1, 0], full_output=True
        )
        assert np.asarray(out[2]["fvec"]).shape == (40,)

    def test_data_mask_statistics_match_subset(self):
        x, y = _data()
        y[[3, 17]] += 5.0
        mask = np.ones(40, bool)
        mask[[3, 17]] = False
        res = curve_fit(_model, x, y, p0=[1, 1, 0], data_mask=mask)
        popt = scipy_curve_fit(_model_np, x[mask], y[mask], p0=[1, 1, 0])[0]
        assert res.r_squared == pytest.approx(
            _r2(y[mask], _model_np(x[mask], *popt)), abs=1e-6
        )
        assert res.ydata.shape == (38,)

    def test_data_mask_2d_xdata_slices_last_axis(self):
        rng = np.random.default_rng(0)
        x = np.vstack([np.linspace(0, 1, 30), np.linspace(1, 2, 30)])
        y = 1.5 * x[0] + 0.7 * x[1] + 0.01 * rng.standard_normal(30)
        mask = np.ones(30, bool)
        mask[[4, 9]] = False
        res = curve_fit(
            lambda X, a, b: a * X[0] + b * X[1], x, y, p0=[1, 1], data_mask=mask
        )
        assert res.xdata.shape == (2, 28)
        assert res.ydata.shape == (28,)


class TestSigma2DWithMask:
    def test_full_covariance_respects_data_mask(self):
        rng = np.random.default_rng(3)
        x, y = _data(seed=3)
        y[[3, 17]] += 5.0
        mask = np.ones(40, bool)
        mask[[3, 17]] = False
        sg = 0.05 + 0.05 * rng.random(40)
        C = (
            np.diag(sg**2)
            + 0.3 * np.diag(sg[:-1] * sg[1:], 1)
            + 0.3 * np.diag(sg[:-1] * sg[1:], -1)
        )
        ref = scipy_curve_fit(
            _model_np, x[mask], y[mask], p0=[1, 1, 0], sigma=C[np.ix_(mask, mask)]
        )
        popt, pcov = tuple(
            curve_fit(_model, x, y, p0=[1, 1, 0], data_mask=mask, sigma=C)
        )
        np.testing.assert_allclose(popt, ref[0], atol=1e-5)
        np.testing.assert_allclose(pcov, ref[1], rtol=1e-3)

    def test_non_positive_definite_sigma_message(self):
        x = np.linspace(0, 1, 10)
        C = np.eye(10)
        C[0, 0] = -1.0
        with pytest.raises(ValueError, match="not positive definite"):
            curve_fit(lambda x, a, b: a * x + b, x, 2 * x + 1, p0=[1, 1], sigma=C)


class TestUserJacobianOrientation:
    X = np.linspace(0, 4, 50)
    Y = 2.5 * np.exp(-1.3 * np.linspace(0, 4, 50)) + 0.5

    @staticmethod
    def _cols(x, a, b, c):
        return [jnp.exp(-b * x), -a * x * jnp.exp(-b * x), jnp.ones_like(x)]

    def test_scipy_style_m_by_n_accepted(self):
        def jac(x, a, b, c):
            return jnp.stack(self._cols(x, a, b, c), axis=-1)  # (m, n)

        p, _ = curve_fit(_model, self.X, self.Y, p0=[1, 1, 0], jac=jac)
        np.testing.assert_allclose(p, [2.5, 1.3, 0.5], atol=1e-6)

    def test_native_n_by_m_still_accepted(self):
        def jac(x, a, b, c):
            return jnp.stack(self._cols(x, a, b, c), axis=0)  # (n, m)

        p, _ = curve_fit(_model, self.X, self.Y, p0=[1, 1, 0], jac=jac)
        np.testing.assert_allclose(p, [2.5, 1.3, 0.5], atol=1e-6)

    def test_wrong_shape_raises_clear_error(self):
        def jac(x, a, b, c):
            return jnp.ones((7, 5))

        with pytest.raises(ValueError, match="`jac` returned shape"):
            curve_fit(_model, self.X, self.Y, p0=[1, 1, 0], jac=jac)


class TestMethodArgumentForwarding:
    X = np.linspace(0, 4, 50)
    Y = 2.5 * np.exp(-1.3 * np.linspace(0, 4, 50)) + 0.5

    def test_auto_forwards_full_output(self):
        out = curve_fit(
            _model, self.X, self.Y, p0=[1, 1, 0], method="auto", full_output=True
        )
        assert isinstance(out, tuple)
        assert len(out) == 5

    def _fake_fit(self):
        return {
            "x": np.array([2.5, 1.3, 0.5]),
            "success": True,
            "message": "ok",
            "fun": np.zeros(50),
            "pcov": np.eye(3),
        }

    def test_hybrid_streaming_warns_on_unsupported_and_honours_outputs(self):
        target = "nlsq.streaming.adaptive_hybrid.AdaptiveHybridStreamingOptimizer.fit"
        with patch(target, side_effect=lambda *a, **k: self._fake_fit()):
            with pytest.warns(UserWarning, match=r"\['jac'\]"):
                curve_fit(
                    _model,
                    self.X,
                    self.Y,
                    p0=[1, 1, 0],
                    method="hybrid_streaming",
                    jac=lambda x, a, b, c: jnp.zeros((3, x.shape[0])),
                    verbose=0,
                )
            out = curve_fit(
                _model,
                self.X,
                self.Y,
                p0=[1, 1, 0],
                method="hybrid_streaming",
                full_output=True,
                verbose=0,
            )
            assert isinstance(out, tuple) and len(out) == 5 and out[4] == 1
            ev = curve_fit(
                _model,
                self.X,
                self.Y,
                p0=[1, 1, 0],
                method="hybrid_streaming",
                return_eval=True,
                verbose=0,
            )
            assert len(ev) == 3 and ev[2].shape == (50,)


class TestFunctionIdentityReuse:
    """Reusing one CurveFit must not keep a stale wrapper (same bytecode, new state)."""

    X = np.arange(1.0, 7.0)
    Y = 2 * np.arange(1.0, 7.0)

    def test_bound_methods_of_different_instances(self):
        class Model:
            def __init__(self, scale):
                self.scale = scale

            def f(self, x, a):
                return self.scale * a * x

        opt = CurveFit()
        opt.curve_fit(Model(1.0).f, self.X, self.Y, p0=[0.5])
        r = opt.curve_fit(Model(2.0).f, self.X, self.Y, p0=[0.5])
        np.testing.assert_allclose(r.popt, [1.0], rtol=1e-6)

    def test_different_default_arguments(self):
        def maker(s):
            def f(x, a, scale=s):
                return scale * a * x

            return f

        opt = CurveFit()
        opt.curve_fit(maker(1.0), self.X, self.Y, p0=[0.5])
        r = opt.curve_fit(maker(2.0), self.X, self.Y, p0=[0.5])
        np.testing.assert_allclose(r.popt, [1.0], rtol=1e-6)

    def test_same_function_and_same_bound_instance_still_hit_cache(self):
        from nlsq.core.least_squares import _closure_aware_changed

        class Model:
            def f(self, x, a, k=1.0):
                return k * a * x

        m = Model()
        assert not _closure_aware_changed(m.f, m.f)
        assert not _closure_aware_changed(Model.f, Model.f)
        assert _closure_aware_changed(Model().f, Model().f)
        # array-valued defaults cannot be compared with ==: fail safe (changed)
        arr = np.ones(2)

        def g(x, a, k=arr):
            return a * x

        arr2 = np.ones(2)

        def h(x, a, k=arr2):
            return a * x

        assert _closure_aware_changed(g, h)


class TestMaskedModelDomainReverseMode:
    """Excluded/padded rows must not poison reverse-mode derivatives."""

    @staticmethod
    def _sqrt_model(x, a):
        return jnp.sqrt(a * x - 1)

    @pytest.mark.parametrize("mode", ["fwd", "rev"])
    def test_padded_rows(self, mode):
        x = np.linspace(2, 4, 8)
        y = np.sqrt(3 * x - 1)
        r = CurveFit(flength=12).curve_fit(
            self._sqrt_model, x, y, p0=[2.0], jacobian_mode=mode
        )
        assert r.success
        np.testing.assert_allclose(r.popt, [3.0], rtol=1e-6)

    @pytest.mark.parametrize("mode", ["fwd", "rev"])
    def test_user_data_mask_excludes_out_of_domain_row(self, mode):
        x = np.array([2.0, 0.0, 3.0, 4.0])
        y = np.sqrt(3 * x - 1)
        y[1] = 0.0
        r = CurveFit().curve_fit(
            self._sqrt_model,
            x,
            y,
            p0=[2.0],
            data_mask=np.array([True, False, True, True]),
            jacobian_mode=mode,
        )
        assert r.success
        np.testing.assert_allclose(r.popt, [3.0], rtol=1e-6)


class TestFlengthSmallerThanData:
    """Observations exceeding flength are not padded; sigma must still validate."""

    @pytest.mark.parametrize("kind", ["1d", "2d"])
    def test_sigma_shape_with_unpadded_data(self, kind):
        x = np.arange(6.0)
        y = 2 * x + 1
        sigma = np.ones(6) if kind == "1d" else np.eye(6)
        r = CurveFit(flength=4).curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[1, 1], sigma=sigma
        )
        np.testing.assert_allclose(r.popt, [2.0, 1.0], rtol=1e-6)


class TestMultivariatePadding:
    def test_three_features_padded_to_flength(self):
        t = np.arange(1.0, 7.0)
        x = np.vstack([t, t**2, t**3])
        y = 2 * (x[0] + x[1] + x[2])
        r = CurveFit(flength=8).curve_fit(
            lambda X, a: a * (X[0] + X[1] + X[2]), x, y, p0=[1.0]
        )
        np.testing.assert_allclose(r.popt, [2.0], rtol=1e-6)
