"""Round-6 review regressions: fallback, guard math, convergence monitor, memory."""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import curve_fit
from nlsq.caching.memory_manager import MemoryManager
from nlsq.stability.fallback import FallbackOrchestrator
from nlsq.stability.guard import NumericalStabilityGuard
from nlsq.utils.diagnostics import ConvergenceMonitor


def _exp(x, a, b):
    return a * jnp.exp(-b * x)


class TestFallbackOnUnsuccessfulFit:
    def test_success_false_triggers_strategies(self):
        x = np.linspace(0, 4, 30)
        y = 3 * np.exp(-0.7 * x)
        r = curve_fit(_exp, x, y, p0=[100, -5], max_nfev=2, fallback=True)
        assert r.fallback_attempts > 1  # not accepted at the first attempt

    def test_returns_best_unsuccessful_result_when_none_converge(self):
        x = np.linspace(0, 4, 30)
        y = 3 * np.exp(-0.7 * x)
        orch = FallbackOrchestrator(max_attempts=4)
        r = orch.fit_with_fallback(_exp, x, y, p0=[100, -5], max_nfev=1)
        assert r.fallback_strategy_used is None
        assert not r.success
        assert r.fallback_attempts == 4

    def test_all_raising_still_raises(self):
        def bad(x, a):
            raise ValueError("boom")

        orch = FallbackOrchestrator(max_attempts=4)
        with pytest.raises(RuntimeError, match="fallback attempts failed"):
            orch.fit_with_fallback(bad, np.arange(5.0), np.arange(5.0), p0=[1.0])


class TestFallbackProgressiveStrategies:
    def test_max_attempts_reached_for_always_raising_model(self):
        def bad(x, a):
            raise ValueError("boom")

        orch = FallbackOrchestrator(max_attempts=10)
        with pytest.raises(RuntimeError, match="All 10 fallback attempts failed"):
            orch.fit_with_fallback(bad, np.arange(5.0), np.arange(5.0), p0=[1.0])

    def test_progressive_strategies_exhaust(self):
        def bad(x, a):
            raise ValueError("boom")

        orch = FallbackOrchestrator(max_attempts=100)
        with pytest.raises(RuntimeError):
            orch.fit_with_fallback(bad, np.arange(5.0), np.arange(5.0), p0=[1.0])
        by_name = {
            s["name"]: s["attempts"] for s in orch.get_statistics()["strategies"]
        }
        assert by_name["perturb_p0"] == 3
        assert by_name["robust_loss"] == 3
        assert by_name["alternative_method"] == 1
        assert by_name["add_bounds"] == 1


class TestGuardMath:
    g = NumericalStabilityGuard()

    def test_safe_divide_keeps_sign(self):
        assert float(self.g.safe_divide(jnp.array([1.0]), jnp.array([-1e-300]))[0]) < 0
        assert float(self.g.safe_divide(jnp.array([1.0]), jnp.array([0.0]))[0]) > 0

    @pytest.mark.parametrize("ord_", [1, 2, np.inf])
    def test_safe_norm_inf(self, ord_):
        assert self.g.safe_norm(jnp.array([np.inf, 1.0]), ord=ord_) == np.inf

    def test_safe_norm_nan_stays_nan(self):
        assert np.isnan(self.g.safe_norm(jnp.array([np.nan, 1.0])))

    def test_safe_power_negative_base_fractional_is_nan(self):
        assert np.isnan(float(self.g.safe_power(jnp.array(-8.0), 1 / 3)))

    def test_safe_power_negative_base_integer(self):
        assert float(self.g.safe_power(jnp.array(-2.0), 3.0)) == -8.0

    def test_safe_power_negative_exponent_finite(self):
        out = float(self.g.safe_power(jnp.array(1e-200), -2.0))
        assert np.isfinite(out) and out > 1e300

    def test_safe_power_positive_overflow_finite(self):
        assert np.isfinite(float(self.g.safe_power(jnp.array(1e200), 2.0)))


class TestConvergenceMonitorThresholds:
    @staticmethod
    def _feed(m, costs):
        for c in costs:
            m.update(c, np.array([c]))

    @pytest.mark.parametrize("s", [0.5, 1.0, 2.0, 5.0])
    def test_falling_cost_never_diverging(self, s):
        m = ConvergenceMonitor(window_size=10, sensitivity=s)
        self._feed(m, [100 * 0.98**i for i in range(10)])
        flagged, score = m.detect_divergence()
        assert not flagged and score >= 0.0

    def test_divergence_threshold_monotonic(self):
        ts = [
            ConvergenceMonitor(sensitivity=s).divergence_threshold
            for s in (0.5, 1, 2, 4)
        ]
        assert all(a > b > 1.0 for a, b in itertools.pairwise(ts))

    @pytest.mark.parametrize("s", [0.3, 0.7, 1.0, 2.0])
    def test_perfect_alternation_flagged(self, s):
        m = ConvergenceMonitor(window_size=10, sensitivity=s)
        self._feed(m, [1.0 if i % 2 else 2.0 for i in range(10)])
        flagged, score = m.detect_oscillation()
        assert score == 1.0 and flagged

    @pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
    def test_rising_cost_diverging(self, s):
        m = ConvergenceMonitor(window_size=10, sensitivity=s)
        self._feed(m, [1.0 * 1.5**i for i in range(10)])
        assert m.detect_divergence()[0]


class TestAllocateArrayTelemetry:
    def test_allocate_array_does_not_decay_safety_factor(self):
        mm = MemoryManager(safety_factor=1.2, enable_adaptive_safety=True)
        for i in range(20):
            arr = mm.allocate_array((100 + i, 100))  # distinct keys: bypass pool
            del arr
        assert mm.safety_factor == 1.2
        assert mm.get_safety_telemetry()["telemetry_entries"] == 0
