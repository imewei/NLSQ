"""Regression tests for the 2026-11-11 round-7 review of the hybrid streaming
optimizer: zero-p0 scaling and covariance equilibration, the curve_fit callback
contract, max-iteration status, and nfev/nit/status reporting."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize as so

from nlsq import curve_fit, curve_fit_large
from nlsq.callbacks import EarlyStopping, StopOptimization
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer
from nlsq.streaming.hybrid_config import HybridStreamingConfig


def _expd(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def _expd_np(x, a, b, c):
    return a * np.exp(-b * x) + c


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 5, 5000)
    y = _expd_np(x, 3, 1.3, 0.5) + 0.05 * rng.standard_normal(5000)
    return x, y


def _hyb(x, y, p0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return curve_fit(_expd, x, y, p0=p0, method="hybrid_streaming", verbose=0, **kw)


class TestZeroP0ExtremeScale:
    @pytest.mark.parametrize("s", [1e8, 1e-8])
    def test_matches_scipy(self, data, s):
        x, y0 = data
        y = y0 * s
        p0 = [2 * s, 1.0, 0.0]
        ps, cs = so.curve_fit(_expd_np, x, y, p0=p0)
        r = _hyb(x, y, p0)
        cost = lambda p: np.sum((y - _expd_np(x, *p)) ** 2)
        assert cost(np.asarray(r.x)) <= cost(ps) * (1 + 1e-6)
        np.testing.assert_allclose(
            np.asarray(r["perr"]), np.sqrt(np.diag(cs)), rtol=1e-4
        )
        assert r.success

    def test_well_scaled_perr_unchanged(self, data):
        x, y = data
        popt, pcov = curve_fit(_expd, x, y, p0=[2, 1, 0.1])
        r = _hyb(x, y, [2, 1, 0.1])
        np.testing.assert_allclose(np.asarray(r.x), np.asarray(popt), rtol=1e-6)
        np.testing.assert_allclose(
            np.asarray(r["pcov"]), np.asarray(pcov), rtol=1e-6, atol=1e-14
        )


class TestCallbackContract:
    def test_curve_fit_signature_and_units(self, data):
        x, y = data
        calls = []

        def cb(iteration, cost, params, info):
            calls.append((iteration, cost, np.asarray(params), info))

        r = _hyb(x, y, [2, 1, 0], callback=cb, callback_frequency=1)
        its = [c[0] for c in calls]
        assert its[0] == 1 and bool(np.all(np.diff(its) > 0))
        assert all(isinstance(c[2], np.ndarray) and c[2].shape == (3,) for c in calls)
        assert all("gradient_norm" in c[3] and c[3]["phase"] in (1, 2) for c in calls)
        # cost is 0.5 * SSR in both phases (curve_fit/TRF convention)
        final_half_ssr = 0.5 * float(np.sum(np.asarray(r["fun"]) ** 2))
        assert calls[-1][1] == pytest.approx(final_half_ssr, rel=1e-6)
        p1 = [c[1] for c in calls if c[3]["phase"] == 1]
        p2 = [c[1] for c in calls if c[3]["phase"] == 2]
        assert p1 and p2
        assert p2[0] <= p1[-1] * 1.01  # no unit jump at the phase boundary

    def test_callback_classes_work(self, data):
        x, y = data
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            es = EarlyStopping(patience=1, min_delta=1e9)
            r = _hyb(x, y, [2, 1, 0], callback=es)
        assert r.success
        assert r["status"] == -2
        assert "StopOptimization" in r.message
        assert r["nit"] < 50

    def test_stop_in_phase2_keeps_covariance(self, data):
        x, y = data
        seen = []

        def cb(iteration, cost, params, info):
            seen.append(info["phase"])
            if info["phase"] == 2:
                raise StopOptimization

        r = _hyb(x, y, [2, 1, 0], callback=cb, callback_frequency=1)
        assert seen[-1] == 2 and seen.count(2) == 1
        assert r["status"] == -2 and r.success
        assert np.all(np.isfinite(np.asarray(r["pcov"])))

    def test_other_exceptions_not_fatal(self, data):
        x, y = data

        def cb(iteration, cost, params, info):
            raise RuntimeError("bug")

        r = _hyb(x, y, [2, 1, 0], callback=cb)
        assert r.success and r["status"] in (1, 2)

    def test_curve_fit_large_hybrid_callback_message(self, data):
        x, y = data
        with pytest.raises(TypeError, match="callback is not supported"):
            curve_fit_large(
                _expd, x, y, p0=[2, 1, 0], method="hybrid_streaming", callback=print
            )


class TestStatusAndCounts:
    def test_max_iterations_is_failure(self, data):
        x, y = data
        r = _hyb(
            x,
            y,
            [2, 1, 0],
            gauss_newton_max_iterations=1,
            max_warmup_iterations=6,
            warmup_iterations=5,
        )
        assert r.message == "Maximum iterations reached"
        assert not r.success and r["status"] == 0
        out = _hyb(
            x,
            y,
            [2, 1, 0],
            gauss_newton_max_iterations=1,
            max_warmup_iterations=6,
            warmup_iterations=5,
            full_output=True,
        )
        assert out[4] == 5

    def test_noise_free_converges(self):
        x = np.linspace(0, 5, 50000)
        y = _expd_np(x, 3, 1.3, 0.5)
        r = _hyb(x, y, [2, 1, 0])
        assert r.success and r.message != "Maximum iterations reached"
        np.testing.assert_allclose(np.asarray(r.x), [3, 1.3, 0.5], rtol=1e-8)

    def test_nfev_nit_status_populated(self, data):
        x, y = data
        r = _hyb(x, y, [2, 1, 0])
        it = r["streaming_diagnostics"]["phase_iterations"]
        assert r["nit"] == it["phase1"] + it["phase2"]
        assert r["nfev"] >= r["nit"]
        assert r["status"] in (1, 2)
        _, _, info, _, ier = _hyb(x, y, [2, 1, 0], full_output=True)
        assert info["nfev"] == r["nfev"]
        assert ier == r["status"]

    def test_direct_optimizer_status(self, data):
        x, y = data
        opt = AdaptiveHybridStreamingOptimizer(HybridStreamingConfig(verbose=0))
        r = opt.fit((x, y), _expd, np.array([2.0, 1.0, 0.1]), verbose=0)
        assert r["success"] and r["status"] in (1, 2) and r["nfev"] > 0
