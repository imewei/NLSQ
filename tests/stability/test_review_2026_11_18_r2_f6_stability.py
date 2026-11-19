"""Regression tests for the 2026-11-18 round-2 review (fixer F6: stability/)."""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.stability.guard import NumericalStabilityGuard, check_problem_stability
from nlsq.stability.recovery import OptimizationRecovery

G = NumericalStabilityGuard()


def test_p07_safe_norm_negative_order_does_not_underflow():
    assert G.safe_norm(jnp.array([1e-100, 1.0]), ord=-4) == pytest.approx(1e-100)


def test_p07_safe_norm_high_order_does_not_overflow():
    expected = float(np.linalg.norm(np.array([1e80, 1e79]) / 1e80, ord=6)) * 1e80
    assert G.safe_norm(jnp.array([1e80, 1e79]), ord=6) == pytest.approx(expected)


def test_p07_safe_norm_empty_vector_is_zero():
    assert G.safe_norm(jnp.array([])) == 0.0


@pytest.mark.parametrize("order", [None, "fro"])  # codespell:ignore fro
def test_c07_safe_norm_non_numeric_order_at_extreme_scale(order):
    x = jnp.full((2, 2), 1e200)
    assert G.safe_norm(x, order) == pytest.approx(2e200)


def test_p07_scalar_p0_preflight():
    assert check_problem_stability(np.arange(1, 9.0), np.arange(1, 9.0), 1.0)
    x = np.arange(1, 9.0)
    for mode in ("check", "auto"):
        popt, _ = curve_fit(lambda x, a: a * x, x, 2 * x, p0=1.0, stability=mode)
        assert popt[0] == pytest.approx(2.0)


def test_p07_check_residuals_accepts_integers():
    with pytest.warns(UserWarning, match="outliers"):
        _, has_outliers = G.check_residuals(jnp.array([0, 0, 0, 10]))
    assert has_outliers


def test_p07_recovery_bounded_scalar_p0():
    rec = OptimizationRecovery(max_retries=1, enable_diagnostics=False)
    out = rec._multi_start("convergence_failure", {"p0": 0.1, "bounds": (0.0, 1.0)}, 0)
    assert np.shape(out["p0"]) == (1,)


def test_p07_callback_stop_with_full_output_is_not_retried():
    calls = []

    def cb(*a, **k):
        calls.append(1)
        return True

    x = np.arange(1, 9.0)
    curve_fit(
        lambda x, a: a * x,
        x,
        2 * x,
        p0=[0.01],
        fallback=True,
        max_fallback_attempts=4,
        callback=cb,
        full_output=True,
    )
    assert len(calls) == 1
