"""Regression tests for round-7 multistart findings (F1 tuple modes, F3 NaN, F5 p0)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit


def _expd(x, a, b):
    return a * jnp.exp(-b * x)


def _sine(x, a, w):
    return a * jnp.sin(w * x)


def _gauss(x, a, mu, s):
    return a * jnp.exp(-0.5 * ((x - mu) / s) ** 2)


@pytest.fixture
def expd_data():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 4, 50)
    return x, 2 * np.exp(-1.3 * x) + 0.01 * rng.standard_normal(50)


class TestTupleModes:
    """F1: full_output / return_eval / timeit must return the single-start tuple."""

    @pytest.mark.parametrize(
        "kw,length",
        [({"full_output": True}, 5), ({"return_eval": True}, 3), ({"timeit": True}, 5)],
    )
    def test_multistart_matches_single_start_tuple_shape(self, expd_data, kw, length):
        x, y = expd_data
        single = curve_fit(_expd, x, y, p0=[1, 1], **kw)
        multi = curve_fit(_expd, x, y, p0=[1, 1], multistart=True, n_starts=3, **kw)

        assert isinstance(multi, tuple) and not hasattr(multi, "popt")
        assert len(multi) == len(single) == length
        np.testing.assert_allclose(multi[0], single[0], rtol=1e-4)
        assert multi[1].shape == single[1].shape
        if "return_eval" in kw:
            assert multi[2].shape == single[2].shape
        if "full_output" in kw:
            assert multi[3] == single[3] or isinstance(multi[3], str)
            assert multi[4] in (1, 2, 3, 4)


class TestNanOmitWithoutP0:
    """F3: NaN in y + nan_policy='omit' + p0=None must not poison p0/bounds."""

    def test_multistart_fits(self):
        x = np.linspace(-5, 5, 50)
        rng = np.random.default_rng(0)
        y = np.asarray(_gauss(x, 3, 0.5, 1.2)) + 0.01 * rng.standard_normal(50)
        y[3] = np.nan
        res = curve_fit(_gauss, x, y, nan_policy="omit", multistart=True, n_starts=3)
        assert np.all(np.isfinite(res.popt))
        # The width only enters squared, so +s and -s are the same curve.
        popt = np.asarray(res.popt)
        np.testing.assert_allclose(
            [popt[0], popt[1], abs(popt[2])], [3, 0.5, 1.2], rtol=0.1
        )


class TestUserP0IsACandidate:
    """F5: multistart is never worse than the single start from the user's p0."""

    @pytest.mark.parametrize("seed", range(6))
    def test_not_worse_than_single_start(self, seed):
        rng = np.random.default_rng(seed)
        x = np.linspace(0, 10, 200)
        y = 1.5 * np.sin(2.0 * x) + 0.05 * rng.standard_normal(200)
        kw = {"p0": [1.4, 2.05], "bounds": ([0.1, 0.1], [5.0, 12.0])}

        def cost(p):
            return float(np.sum((y - np.asarray(_sine(x, *p))) ** 2))

        single = curve_fit(_sine, x, y, **kw)
        multi = curve_fit(_sine, x, y, multistart=True, n_starts=3, **kw)
        assert cost(multi.popt) <= cost(single.popt) * (1 + 1e-9)

    def test_n_starts_is_total_evaluated(self, expd_data):
        x, y = expd_data
        res = curve_fit(_expd, x, y, p0=[1, 1], multistart=True, n_starts=4)
        diag = res["multistart_diagnostics"]
        assert diag["n_starts_evaluated"] == 4
        assert diag["p0_included"] is True
