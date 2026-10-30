"""Regression tests for the 2026-10-29 global_optimization review fixes.

Covers: inferred multistart bounds leaking into the fits, tournament scoring of
failed batches, per-start callback isolation, and recording the CMA-ES seed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.callbacks import EarlyStopping
from nlsq.global_optimization import (
    CMAESConfig,
    CMAESOptimizer,
    GlobalOptimizationConfig,
    MultiStartOrchestrator,
    TournamentSelector,
)


def _linear(x, a, c):
    return a * x + c


class TestMultistartDoesNotImposeInferredBounds:
    @pytest.mark.parametrize("bounds", [None, (-np.inf, np.inf)])
    def test_unbounded_problem_stays_unbounded(self, bounds):
        """Inferred (sampling-only) bounds force lb>=0 for positive p0; the
        fits must still reach the true optimum a=2, c=-5."""
        x = np.linspace(0, 10, 50)
        y = 2 * x - 5
        cfg = GlobalOptimizationConfig(n_starts=3, sampler="lhs")
        result = MultiStartOrchestrator(cfg).fit(
            _linear, x, y, p0=[1.0, 1.0], bounds=bounds
        )
        np.testing.assert_allclose(result.popt, [2.0, -5.0], atol=1e-5)

    def test_half_infinite_user_bounds_are_respected_exactly(self):
        """Only the user's own (partly infinite) bounds constrain the fits."""
        x = np.linspace(0, 10, 50)
        y = 2 * x - 5
        cfg = GlobalOptimizationConfig(n_starts=3, sampler="lhs")
        result = MultiStartOrchestrator(cfg).fit(
            _linear, x, y, p0=[1.0, 1.0], bounds=([0.0, -np.inf], [np.inf, np.inf])
        )
        np.testing.assert_allclose(result.popt, [2.0, -5.0], atol=1e-5)


class TestTournamentPenalisesFailedBatches:
    @staticmethod
    def _model(x, a):
        # a=0 gives NaN for x>=1; a=0.2 is finite everywhere
        return a * x + 0 * jnp.sqrt(a - 0.05 * jnp.floor(x))

    @staticmethod
    def _batches():
        for k in range(4):
            x = np.linspace(k, k + 0.9, 20)
            yield x, 0.15 * x

    @pytest.mark.parametrize("rounds", [0, 1])
    def test_candidate_failing_on_batches_does_not_win(self, rounds):
        cfg = GlobalOptimizationConfig(
            n_starts=2,
            elimination_rounds=rounds,
            elimination_fraction=0.5,
            batches_per_round=4,
        )
        selector = TournamentSelector(np.array([[0.0], [0.2]]), cfg)
        best = selector.run_tournament(self._batches(), self._model, top_m=1)
        assert best[0][0] == pytest.approx(0.2)


class TestCallbackIsolatedPerStart:
    def test_early_stopping_not_shared_across_starts(self):
        def m(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        x = np.linspace(0, 5, 60)
        y = 3 * np.exp(-0.7 * x) + 1
        bounds = ([0, 0, 0], [10, 5, 10])
        cfg = GlobalOptimizationConfig(n_starts=6, sampler="sobol", center_on_p0=False)

        base = MultiStartOrchestrator(cfg).fit(m, x, y, p0=[1, 1, 1], bounds=bounds)
        cb = EarlyStopping(patience=3, min_delta=0.0, verbose=False)
        with_cb = MultiStartOrchestrator(cfg).fit(
            m, x, y, p0=[1, 1, 1], bounds=bounds, callback=cb
        )

        # The user's own object is never run (copies are), so it stays pristine,
        # and every start converges as it does with no callback.
        assert cb.best_cost == float("inf")
        np.testing.assert_allclose(
            sorted(with_cb.multistart_diagnostics["all_losses"]),
            sorted(base.multistart_diagnostics["all_losses"]),
            rtol=1e-6,
            atol=1e-12,
        )

    def test_undeepcopyable_callback_falls_back_to_shared(self):
        class _NoCopy:
            def __call__(self, *a, **k):
                pass

            def __deepcopy__(self, memo):
                raise TypeError("no copy")

        x = np.linspace(0, 10, 30)
        y = 2 * x - 5
        cfg = GlobalOptimizationConfig(n_starts=2, sampler="lhs")
        result = MultiStartOrchestrator(cfg).fit(
            _linear,
            x,
            y,
            p0=[1.0, 1.0],
            bounds=([-10, -10], [10, 10]),
            callback=_NoCopy(),
        )
        np.testing.assert_allclose(result.popt, [2.0, -5.0], atol=1e-4)

    def test_undeepcopyable_callback_warns_once(self, caplog):
        class _NoCopy:
            def __call__(self, *a, **k):
                pass

            def __deepcopy__(self, memo):
                raise TypeError("no copy")

        x = np.linspace(0, 10, 30)
        y = 2 * x - 5
        cfg = GlobalOptimizationConfig(n_starts=4, sampler="lhs")
        with caplog.at_level("WARNING"):
            MultiStartOrchestrator(cfg).fit(
                _linear,
                x,
                y,
                p0=[1.0, 1.0],
                bounds=([-10, -10], [10, 10]),
                callback=_NoCopy(),
            )
        assert caplog.text.count("could not be deep-copied") == 1


class TestCMAESSeedRecorded:
    @staticmethod
    def _problem():
        def m(x, a, b):
            return a * jnp.exp(-b * x)

        x = np.linspace(0, 4, 30)
        return m, x, 2.5 * np.exp(-1.3 * x), ([0.0, 0.0], [10.0, 10.0])

    @pytest.mark.parametrize("strategy", ["none", "bipop"])
    def test_recorded_seed_reproduces_run(self, strategy):
        m, x, y, bounds = self._problem()
        kw = {
            "max_generations": 4,
            "restart_strategy": strategy,
            "refine_with_nlsq": False,
        }
        if strategy == "bipop":
            kw["max_restarts"] = 1
        first = CMAESOptimizer(CMAESConfig(**kw)).fit(m, x, y, bounds=bounds)
        seed = first["cmaes_diagnostics"]["seed"]
        assert isinstance(seed, int)

        replay = CMAESOptimizer(CMAESConfig(seed=seed, **kw)).fit(
            m, x, y, bounds=bounds
        )
        assert replay["cmaes_diagnostics"]["seed"] == seed
        np.testing.assert_array_equal(first["popt"], replay["popt"])
