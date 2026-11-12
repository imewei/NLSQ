"""Round-7 regressions: fallback ranking across losses, logger format, safe_power."""

import logging

import jax.numpy as jnp
import numpy as np

from nlsq.stability.fallback import FallbackOrchestrator
from nlsq.stability.guard import NumericalStabilityGuard


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


def _outlier_problem():
    x = np.linspace(0, 4, 60)
    y = 2 * np.exp(-1.3 * x)
    y[::7] += 3
    return x, y


def _ssr_half(r):
    return 0.5 * float(np.sum(np.asarray(r.fun) ** 2))


class TestFallbackRanking:
    def test_robust_loss_cost_does_not_win_ranking(self):
        x, y = _outlier_problem()
        orch = FallbackOrchestrator()
        r = orch.fit_with_fallback(_exp, x, y, p0=[1, 0.1], max_nfev=2)
        assert not r.success
        # Returned result is the best in the caller's (linear) domain: the
        # original attempt's residual cost is the baseline it must not exceed.
        orig = FallbackOrchestrator(max_attempts=1).fit_with_fallback(
            _exp, x, y, p0=[1, 0.1], max_nfev=2
        )
        assert _ssr_half(r) <= _ssr_half(orig) + 1e-9
        assert r.fallback_strategy_used != "robust_loss"

    def test_strategy_reported_for_failed_best(self):
        x, y = _outlier_problem()
        orch = FallbackOrchestrator(max_attempts=4)
        r = orch.fit_with_fallback(_exp, x, y, p0=[1, 0.1], max_nfev=2)
        assert not r.success
        # strategy_used names the attempt that produced r (None = original).
        if r.fallback_strategy_used is None:
            assert _ssr_half(r) == _ssr_half(
                FallbackOrchestrator(max_attempts=1).fit_with_fallback(
                    _exp, x, y, p0=[1, 0.1], max_nfev=2
                )
            )

    def test_failure_warning_formatted(self, caplog):
        x, y = _outlier_problem()
        orch = FallbackOrchestrator(max_attempts=3)
        with caplog.at_level(logging.WARNING):
            orch.fit_with_fallback(_exp, x, y, p0=[1, 0.1], max_nfev=2)
        text = " ".join(rec.getMessage() for rec in caplog.records)
        assert "%d" not in text
        assert "All 3 fallback attempts failed" in text
        assert not any(rec.exc_info for rec in caplog.records)


class TestSafePowerInteger:
    def test_int_base_negative_exponent(self):
        g = NumericalStabilityGuard()
        out = g.safe_power(jnp.array([2, 3]), -1)
        np.testing.assert_allclose(np.asarray(out), [0.5, 1 / 3])
