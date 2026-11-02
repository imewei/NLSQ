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
    """Record the (x, y) arrays (copies) the hybrid optimizer is actually handed."""
    seen = []
    orig = AdaptiveHybridStreamingOptimizer.fit

    def spy(self, data_source, *args, **kwargs):
        seen.append(tuple(np.array(d) for d in data_source))
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


class TestPreparationOrder:
    def test_preparation_order_omit_then_select_then_pad(self, monkeypatch):
        """Order is omit NaN rows -> auto-select method -> pad to flength.

        Seam: ``_select_optimization_method`` imports ``auto_select_algorithm``
        lazily from ``nlsq.precision.algorithm_selector`` at call time, so a
        spy there sees exactly the xdata/ydata the selector is given.
        """
        import nlsq.precision.algorithm_selector as selector

        seen = []
        orig = selector.auto_select_algorithm

        def spy(f, xdata, ydata, p0, bounds, *args, **kwargs):
            seen.append((np.array(xdata), np.array(ydata)))
            return orig(f, xdata, ydata, p0, bounds, *args, **kwargs)

        monkeypatch.setattr(selector, "auto_select_algorithm", spy)

        x, y = _data()
        y_nan = y.copy()
        y_nan[[3, 10]] = np.nan
        keep = np.isfinite(y_nan)
        n_clean = int(keep.sum())
        assert n_clean == N - 2

        out = CurveFit(enable_stability=True, flength=100)._prepare_curve_fit_inputs(
            _model,
            x,
            y_nan,
            np.array([1.0, 1.0]),
            (-np.inf, np.inf),
            "auto",
            None,
            None,  # method=None -> auto-selection runs
            True,
            None,
            {"nan_policy": "omit"},
        )
        _, _, xdata, ydata, data_mask, _, _, _, m, len_diff, should_pad, _, _ = out

        # Selector ran once, on the cleaned and still-unpadded observations.
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0][0], x[keep])
        np.testing.assert_array_equal(seen[0][1], y_nan[keep])
        # Padding came after: padded arrays, mask marks only the clean rows.
        assert m == n_clean
        assert should_pad
        assert len_diff == 100 - n_clean
        assert len(xdata) == len(ydata) == len(data_mask) == 100
        assert int(np.sum(data_mask)) == n_clean
        np.testing.assert_array_equal(np.asarray(xdata)[:n_clean], x[keep])
        np.testing.assert_array_equal(np.asarray(ydata)[:n_clean], y_nan[keep])


class TestHybridStreaming:
    def test_hybrid_nan_policy_accepted_once(self, hybrid_spy):
        """nan_policy is consumed exactly once on the hybrid path.

        A second consumption would leave the later stage with the default
        nan_policy='raise' and fail on the NaN rows (or an unconsumed key would
        hit the unknown-keyword TypeError). The optimizer must receive exactly
        the finite rows, and fit like a pre-filtered run.
        """
        x, y = _data(300)
        y_nan = y.copy()
        y_nan[[3, 10]] = np.nan
        keep = np.isfinite(y_nan)
        kw = {"p0": [1.9, 1.2], "method": "hybrid_streaming", **_HYBRID_FAST}

        res = curve_fit(_model, x, y_nan, nan_policy="omit", **kw)
        ref = curve_fit(_model, x[keep], y_nan[keep], **kw)

        assert res.success
        assert np.all(np.isfinite(res.x))
        assert len(hybrid_spy) == 2
        assert hybrid_spy[0][0].shape == (298,)  # rows omitted exactly once
        np.testing.assert_array_equal(hybrid_spy[0][0], x[keep])
        np.testing.assert_array_equal(hybrid_spy[0][1], y_nan[keep])
        # Same data, same deterministic CPU path -> same result.
        np.testing.assert_allclose(res.x, ref.x, rtol=1e-8, atol=1e-10)

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

        assert [tuple(a.shape for a in s) for s in hybrid_spy] == [((300,), (300,))]
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
