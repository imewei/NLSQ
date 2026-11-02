"""Characterization of ``curve_fit`` input preparation (Part B, Task B1).

These tests pin the CURRENT observable behavior of input preparation
(``CurveFit._prepare_curve_fit_inputs`` and its helpers, plus the duplicate
preparation inside ``_curve_fit_hybrid_streaming``) through the public entry
points, so the extraction of ``prepare_problem`` into ``nlsq/core/problem.py``
can be shown to be behavior-preserving.

They deliberately assert what the code does today, including quirks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit
from nlsq.streaming.adaptive_hybrid import AdaptiveHybridStreamingOptimizer

N = 60
P0 = [1.0, 1.0]


def _model(x, a, b):
    return a * jnp.exp(-b * x)


def _data(n=N, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0, n)
    y = 2.0 * np.exp(-1.3 * x) + 0.01 * rng.standard_normal(n)
    return x, y


def _manual_pcov(x, y, popt):
    """pcov = (J^T J)^-1 * sum(r^2) / (m - p), with m the real point count."""
    a, b = popt
    e = np.exp(-b * x)
    jac = np.stack([e, -a * x * e], axis=1)
    r = y - a * e
    return np.linalg.inv(jac.T @ jac) * (r @ r) / (len(x) - len(popt))


# Fast hybrid configuration (default warmup takes about a minute on CPU).
_HYBRID_FAST = {
    "warmup_iterations": 2,
    "max_warmup_iterations": 4,
    "gauss_newton_max_iterations": 3,
    "verbose": 0,
}


@pytest.fixture
def hybrid_spy(monkeypatch):
    """Record the (x, y) shapes the hybrid optimizer is actually handed."""
    seen = []
    orig = AdaptiveHybridStreamingOptimizer.fit

    def spy(self, data_source, *args, **kwargs):
        seen.append(tuple(np.asarray(d).shape for d in data_source))
        return orig(self, data_source, *args, **kwargs)

    monkeypatch.setattr(AdaptiveHybridStreamingOptimizer, "fit", spy)
    return seen


class TestOmitAndMask:
    def test_omit_matches_prefiltered(self):
        x, y = _data()
        y_nan = y.copy()
        y_nan[[3, 10]] = np.nan
        keep = np.isfinite(y_nan)

        popt, pcov = curve_fit(_model, x, y_nan, p0=P0, nan_policy="omit")
        popt_ref, pcov_ref = curve_fit(_model, x[keep], y_nan[keep], p0=P0)

        np.testing.assert_allclose(popt, popt_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(pcov, pcov_ref, rtol=1e-8, atol=1e-12)

    def test_mask_alignment_after_omit(self):
        """A data_mask sized to the ORIGINAL data is realigned after omission."""
        x, y = _data()
        y_nan = y.copy()
        y_nan[[3, 10]] = np.nan
        mask = np.ones(N, dtype=bool)
        mask[[10, 50, 51]] = False  # index 10 is both NaN and masked out
        keep = np.isfinite(y_nan) & mask

        popt, pcov = curve_fit(
            _model, x, y_nan, p0=P0, nan_policy="omit", data_mask=mask
        )
        popt_ref, pcov_ref = curve_fit(_model, x[keep], y_nan[keep], p0=P0)

        np.testing.assert_allclose(popt, popt_ref, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(pcov, pcov_ref, rtol=1e-8, atol=1e-12)

    def test_dof_uses_mask_sum_not_padded_length(self):
        """pcov scales with sum(mask) - p, never with the padded length - p."""
        x, y = _data()

        # (a) padding: flength=100 pads 60 real rows with 40 masked-out rows.
        popt, pcov = curve_fit(_model, x, y, p0=P0, flength=100)
        manual = _manual_pcov(x, y, popt)
        np.testing.assert_allclose(pcov, manual, rtol=1e-6)
        wrong = manual * (N - 2) / (100 - 2)  # what the padded length would give
        assert not np.allclose(pcov, wrong, rtol=1e-2)

        # (b) caller mask: 10 real rows excluded -> dof = 50 - 2.
        mask = np.ones(N, dtype=bool)
        mask[::6] = False
        assert mask.sum() == 50
        popt_m, pcov_m = curve_fit(_model, x, y, p0=P0, data_mask=mask)
        np.testing.assert_allclose(
            pcov_m, _manual_pcov(x[mask], y[mask], popt_m), rtol=1e-6
        )


class TestHybridStreaming:
    def test_hybrid_nan_policy_accepted_once(self, hybrid_spy):
        x, y = _data(300)
        y[[3, 10]] = np.nan

        res = curve_fit(
            _model,
            x,
            y,
            p0=[1.9, 1.2],
            method="hybrid_streaming",
            nan_policy="omit",
            **_HYBRID_FAST,
        )

        assert np.all(np.isfinite(res.x))
        assert hybrid_spy == [((298,), (298,))]  # rows omitted exactly once

    def test_hybrid_unknown_kwarg_still_raises_typeerror(self):
        x, y = _data(300)
        with pytest.raises(TypeError, match="Unrecognized keyword argument"):
            curve_fit(
                _model,
                x,
                y,
                p0=P0,
                method="hybrid_streaming",
                not_a_real_option=1,
                **_HYBRID_FAST,
            )

    def test_hybrid_data_mask_still_rejected(self):
        x, y = _data(300)
        with pytest.raises(NotImplementedError, match="data_mask"):
            curve_fit(
                _model,
                x,
                y,
                p0=P0,
                method="hybrid_streaming",
                data_mask=np.ones(300, dtype=bool),
                **_HYBRID_FAST,
            )

    def test_hybrid_does_not_pad(self, hybrid_spy):
        """flength pads on the standard path; hybrid sees the unpadded data."""
        x, y = _data(300)

        res = curve_fit(
            _model,
            x,
            y,
            p0=[1.9, 1.2],
            method="hybrid_streaming",
            flength=400,
            **_HYBRID_FAST,
        )

        assert hybrid_spy == [((300,), (300,))]
        assert np.asarray(res["xdata"]).shape == (300,)


class TestKwargsLeftovers:
    def test_kwargs_leftovers_visible_to_caller(self):
        """Pin how preparation treats the caller's ``kwargs`` dict.

        Isolated on purpose: it calls the private ``_prepare_curve_fit_inputs``
        directly, so the extraction task (B2) repoints it to ``prepare_problem``.

        Current behavior (differs from the plan's wording): preparation CONSUMES
        nothing. ``nan_policy`` is read with ``kwargs.get`` and stays in the dict;
        unrelated keys stay; the only mutation is that auto-selection
        (``enable_stability=True`` and ``method=None``) ADDS tolerance keys the
        caller did not set, never overwriting caller-supplied ones.
        """
        x, y = _data()
        args = (
            np.array([1.0, 1.0]),
            (-np.inf, np.inf),
            "auto",
            None,  # batch_size
            None,  # method
            True,  # check_finite
            None,  # data_mask
        )

        kwargs = {"nan_policy": "omit", "unrelated": 1}
        CurveFit()._prepare_curve_fit_inputs(_model, x, y, *args, kwargs)
        assert kwargs == {"nan_policy": "omit", "unrelated": 1}

        kwargs = {"ftol": 1e-3}
        CurveFit(enable_stability=True)._prepare_curve_fit_inputs(
            _model, x, y, *args, kwargs
        )
        assert kwargs["ftol"] == 1e-3  # caller-supplied value is kept
        assert set(kwargs) == {"ftol", "xtol", "gtol", "max_nfev", "x_scale"}
