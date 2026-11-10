"""Regression tests for the round-2 review (GD-1 sampling window, R2-3 BIPOP scale)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import (
    GlobalOptimizationConfig,
    MultiStartOrchestrator,
    center_samples_around_p0,
)
from nlsq.global_optimization.sampling import latin_hypercube_sample


def _lhs(n, d):
    import jax

    return latin_hypercube_sample(n, d, rng_key=jax.random.PRNGKey(0))


class TestCenterWindowShift:
    """GD-1: the window keeps its width and shifts inside the bounds."""

    def test_corner_p0_full_scale_covers_whole_box(self):
        lb, ub = jnp.array([0.0, 0.0]), jnp.array([10.0, 10.0])
        out = np.asarray(center_samples_around_p0(_lhs(1000, 2), lb, 1.0, lb, ub))
        assert out.min() >= 0.0 and out.max() <= 10.0
        assert out.max(axis=0) == pytest.approx([10.0, 10.0], abs=0.05)
        assert out.min(axis=0) == pytest.approx([0.0, 0.0], abs=0.05)

    def test_corner_p0_partial_scale_keeps_width(self):
        lb, ub = jnp.array([0.0]), jnp.array([10.0])
        out = np.asarray(center_samples_around_p0(_lhs(500, 1), lb, 0.4, lb, ub))
        # width 4 anchored at the lower bound (old code: only [0, 2])
        assert out.max() == pytest.approx(4.0, abs=0.05)
        assert out.min() >= 0.0

    def test_mid_box_p0_still_centered(self):
        lb, ub = jnp.array([0.0]), jnp.array([10.0])
        p0 = jnp.array([5.0])
        out = np.asarray(center_samples_around_p0(_lhs(500, 1), p0, 0.4, lb, ub))
        assert out.min() == pytest.approx(3.0, abs=0.05)
        assert out.max() == pytest.approx(7.0, abs=0.05)

    def test_finite_dim_unaffected_by_infinite_dim(self):
        lb, ub = jnp.array([0.0, 0.0]), jnp.array([10.0, jnp.inf])
        out = np.asarray(
            center_samples_around_p0(
                jnp.full((3, 2), 0.5), jnp.array([5.0, 1.0]), 0.4, lb, ub
            )
        )
        assert np.all(out[:, 0] == 5.0)


def _sin_problem():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 200)
    y = 2.0 * np.sin(3.3 * x) + rng.normal(0, 0.05, x.size)

    def model(x, a, w):
        return a * jnp.sin(w * x)

    return model, x, y


@pytest.mark.slow
class TestSinusoidReachesGlobalMinimum:
    def test_orchestrator(self):
        model, x, y = _sin_problem()
        res = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=12)).fit(
            model, x, y, bounds=([0.1, 0.1], [5.0, 5.0])
        )
        assert np.allclose(res.popt, [2.0, 3.3], atol=0.05)

    def test_curve_fit_multistart(self):
        from nlsq import curve_fit

        model, x, y = _sin_problem()
        popt, _ = curve_fit(
            model, x, y, bounds=([0.1, 0.1], [5.0, 5.0]), multistart=True, n_starts=12
        )
        assert np.allclose(popt, [2.0, 3.3], atol=0.05)
