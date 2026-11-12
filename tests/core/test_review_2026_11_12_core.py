"""Regression tests for the 2026-11-12 core review (f9)."""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import CurveFit, curve_fit, fit
from nlsq.common_scipy import make_strictly_feasible


def _exp(x, a, b, c):
    return a * jnp.exp(-b * x) + c


def test_auto_global_cmaes_route_warns_about_ignored_sampler_options():
    x = np.linspace(0, 4, 50)
    y = 2 * np.exp(-1.3 * x) + 0.5
    with pytest.warns(UserWarning, match="sampler.*ignored"):
        fit(
            _exp,
            x,
            y,
            p0=[1, 1, 1],
            # scale ratio > 1000 -> MethodSelector picks CMA-ES
            bounds=([0, 0, 0], [1e6, 10, 10]),
            workflow="auto_global",
            sampler="sobol",
            center_on_p0=False,
            scale_factor=0.5,
        )


def test_lm_underdetermined_raises_same_error_when_padded():
    def lin(x, a, b, c):
        return a * x + b + c

    x, y = np.array([0.0, 1.0]), np.array([1.0, 2.0])
    with pytest.raises(TypeError, match="must not exceed"):
        curve_fit(lin, x, y, p0=[1, 1, 1], method="lm")
    with pytest.raises(TypeError, match="must not exceed"):
        CurveFit(flength=50).curve_fit(lin, x, y, p0=[1, 1, 1], method="lm")


def test_make_strictly_feasible_extreme_bounds_no_warning():
    big = np.finfo(float).max
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = make_strictly_feasible(np.array([1.0]), np.array([-big]), np.array([big]))
    np.testing.assert_array_equal(out, [1.0])


def test_curve_fit_extreme_bounds_no_runtime_warning():
    big = np.finfo(float).max
    x = np.linspace(0, 1, 20)
    y = 2 * x + 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        popt, _ = curve_fit(
            lambda x, a, b: a * x + b, x, y, p0=[1, 1], bounds=(-big, big)
        )
    np.testing.assert_allclose(popt, [2, 1], rtol=1e-6)
