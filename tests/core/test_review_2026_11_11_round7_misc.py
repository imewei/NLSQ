"""Round-7 review fixes: multi-scale feasibility, multistart log noise, memory_limit_gb."""

import logging

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.common_scipy import make_strictly_feasible


def _old_make_strictly_feasible(x, lb, ub, rstep=1e-10):
    """Pre-fix implementation (SciPy parity), for the ordinary-scale comparison."""
    from nlsq.common_scipy import find_active_constraints

    x_new = np.array(x, copy=True)
    active = find_active_constraints(x, lb, ub, rstep)
    lo, hi = active == -1, active == 1
    x_new[lo] = lb[lo] + rstep * np.maximum(1, np.abs(lb[lo]))
    x_new[hi] = ub[hi] - rstep * np.maximum(1, np.abs(ub[hi]))
    tight = (x_new < lb) | (x_new > ub)
    x_new[tight] = 0.5 * (lb[tight] + ub[tight])
    return x_new


class TestMultiScaleFeasibility:
    def test_tiny_interior_point_stays_put(self):
        lb = np.array([1e-12, 1e-5, 0.1])
        ub = np.array([1e-8, 1e-1, 2.0])
        x = np.array([2.23e-11, 1e-3, 0.5])
        np.testing.assert_array_equal(make_strictly_feasible(x, lb, ub), x)
        x2 = np.array([lb[0] + 1e-12, 1e-3, 0.5])
        np.testing.assert_array_equal(make_strictly_feasible(x2, lb, ub), x2)

    def test_point_on_tiny_bound_moves_relative_amount(self):
        lb = np.array([1e-12])
        ub = np.array([1e-8])
        out = make_strictly_feasible(np.array([1e-12]), lb, ub)
        assert lb[0] < out[0] < 1.01e-12
        out = make_strictly_feasible(np.array([1e-8]), lb, ub)
        assert 0.99e-8 < out[0] < ub[0]

    def test_ordinary_bounds_unchanged_vs_old(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            n = 5
            lb = rng.uniform(-10, 10, n)
            ub = lb + rng.uniform(1, 20, n)
            lb[rng.random(n) < 0.2] = -np.inf
            ub[rng.random(n) < 0.2] = np.inf
            lo = np.where(np.isfinite(lb), lb, -50.0)
            hi = np.where(np.isfinite(ub), ub, 50.0)
            x = rng.uniform(lo, hi)
            # Put some points exactly on / within 1e-11 of the bounds.
            pick = rng.random(n)
            x = np.where((pick < 0.2) & np.isfinite(lb), lb, x)
            x = np.where((pick > 0.8) & np.isfinite(ub), ub, x)
            x = np.where((pick > 0.4) & (pick < 0.5) & np.isfinite(lb), lb + 1e-11, x)
            np.testing.assert_array_equal(
                make_strictly_feasible(x, lb, ub),
                _old_make_strictly_feasible(x, lb, ub),
            )

    def test_tiny_scale_fit_converges(self):
        from nlsq import curve_fit

        def model(x, D0, g, n):
            return D0 * (1 + (x / g) ** n)

        x = np.logspace(-4, -1, 200)
        true = (2.23e-11, 1e-3, 0.5)
        y = np.asarray(model(jnp.asarray(x), *true))
        lb = [1e-12, 1e-5, 0.1]
        ub = [1e-8, 1e-1, 2.0]
        popt, _ = curve_fit(model, x, y, p0=list(true), bounds=(lb, ub))
        np.testing.assert_allclose(popt, true, rtol=1e-6)
        ssr = float(np.sum((np.asarray(model(jnp.asarray(x), *popt)) - y) ** 2))
        assert ssr < 1e-20


class TestMultistartLoserLogs:
    def test_loser_logger_warning_hidden_winner_kept(self, monkeypatch, caplog):
        from nlsq.global_optimization import multi_start as ms

        log = logging.getLogger("nlsq.curve_fit")

        def fake(f, xdata, ydata, p0, bounds, kwargs, cfk=None):
            loss = float(p0[0])
            log.warning(f"rank-deficient start p0={loss}")
            return (np.asarray(p0), loss, None)

        monkeypatch.setattr(ms, "_fit_single_start", fake)
        orch = ms.MultiStartOrchestrator()
        pts = np.array([[3.0], [1.0], [2.0]])
        with caplog.at_level(logging.WARNING):
            results, _ = orch.evaluate_starting_points(
                lambda x, a: a * x,
                np.arange(3.0),
                np.arange(3.0),
                pts,
                (np.array([0.0]), np.array([5.0])),
            )
        assert results[0][1] == 1.0
        msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("p0=1.0" in m for m in msgs)
        assert not any("p0=3.0" in m or "p0=2.0" in m for m in msgs)

    def test_warnings_outside_multistart_unaffected(self, caplog):
        log = logging.getLogger("nlsq.curve_fit")
        with caplog.at_level(logging.WARNING):
            log.warning("plain warning")
        assert any(r.getMessage() == "plain warning" for r in caplog.records)


class TestMemoryLimitValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
    def test_memory_budget_rejects_bad_limit(self, bad):
        from nlsq.core.workflow import MemoryBudget

        with pytest.raises(ValueError, match="memory_limit_gb"):
            MemoryBudget.compute(n_points=1000, n_params=3, memory_limit_gb=bad)

    def test_fit_auto_nan_memory_limit_clear_error(self):
        from nlsq import fit

        x = np.linspace(0, 1, 50)
        with pytest.raises(ValueError, match="memory_limit_gb"):
            fit(
                lambda x, a, b: a * x + b,
                x,
                2 * x + 1,
                p0=[1.0, 0.0],
                workflow="auto",
                memory_limit_gb=float("nan"),
            )
