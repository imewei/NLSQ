"""Regression tests for the 2026-11-18 three-brain review (fixer F2: stability/)."""

import warnings
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.stability.fallback import (
    AdjustTolerancesStrategy,
    FallbackOrchestrator,
    UseRobustLossStrategy,
    _is_noop,
)
from nlsq.stability.guard import NumericalStabilityGuard, solve_with_cholesky_fallback


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-20])
def test_c09_indefinite_system_solved_at_any_scale(scale):
    A = np.array([[2.0, 1.0], [1.0, -3.0]]) * scale
    b = np.array([1.0, 2.0])
    x, _ = solve_with_cholesky_fallback(jnp.asarray(A), jnp.asarray(b))
    np.testing.assert_allclose(np.asarray(x), np.linalg.solve(A, b), rtol=1e-8)


@pytest.mark.parametrize("scale", [1.0, 1e-12, 1e-20])
def test_c09_outlier_detection_is_scale_invariant(scale):
    r = np.random.default_rng(0).standard_normal(50)
    r[3] = 1000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, has_outliers = NumericalStabilityGuard().check_residuals(
            jnp.asarray(r * scale)
        )
    assert has_outliers


def test_c09_adjust_tolerances_skips_disabled_tolerance():
    out = AdjustTolerancesStrategy().apply({"ftol": None})
    assert out["ftol"] is None
    assert out["xtol"] > 1e-8 and out["gtol"] > 1e-8


def test_c09_robust_loss_matching_callers_loss_is_a_noop():
    kw = {"loss": "soft_l1", "p0": [1.0]}
    assert _is_noop(UseRobustLossStrategy().apply(kw), kw)


def test_p07_noop_proposal_does_not_end_progressive_sequence(monkeypatch):
    calls = []

    def fake_curve_fit(f, x, y, **kw):
        calls.append(kw.get("loss"))
        ok = kw.get("loss") == "huber"
        return SimpleNamespace(
            success=ok, x=np.ones(1), popt=np.ones(1), cost=1.0, fun=np.zeros(2)
        )

    monkeypatch.setattr("nlsq.core.minpack.curve_fit", fake_curve_fit)
    orch = FallbackOrchestrator(strategies=[UseRobustLossStrategy()], max_attempts=4)
    orch.fit_with_fallback(
        lambda x, a: a * x,
        np.ones(2),
        np.ones(2),
        p0=[1.0],
        loss="soft_l1",
        f_scale=1.0,
    )
    # The caller's own soft_l1 proposal is a no-op; huber must still be reached.
    assert "huber" in calls


def test_p07_safe_norm_negative_order_uses_smallest_entry():
    val = NumericalStabilityGuard().safe_norm(jnp.array([1e-150, 1e150]), ord=-2)
    assert val == pytest.approx(1e-150, rel=1e-12)
