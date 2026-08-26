"""Regression tests for the /review-pr findings fixed on top of PR #19.

Covers: MultiStartOrchestrator re-raising the winning start's covariance
warning, TournamentSelector raising instead of silently falling back on
total failure, TournamentSelector's per-round elimination staying above
top_m, sampling.center_samples_around_p0 clamping an out-of-bounds p0,
LargeDatasetFitter._fit_single_chunk reporting an all-inf (not identity)
pcov on failure, and CMAESOptimizer._nlsq_refinement propagating (not
swallowing) a NotImplementedError from the underlying curve_fit() call.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq import fit
from nlsq.global_optimization.cmaes_config import CMAESConfig
from nlsq.global_optimization.cmaes_optimizer import CMAESOptimizer
from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.multi_start import MultiStartOrchestrator
from nlsq.global_optimization.sampling import center_samples_around_p0
from nlsq.global_optimization.tournament import TournamentSelector
from nlsq.result.optimize_warning import OptimizeWarning
from nlsq.streaming.large_dataset import LargeDatasetFitter


def _exp_model(x, a, b):
    return a * jnp.exp(-b * x)


class TestAutoGlobalBoundsValidation:
    """Regression tests for _fit_with_auto_global's bounds-validation trio."""

    def _data(self):
        x = np.linspace(0, 4, 30)
        y = np.array(_exp_model(x, 2.5, 1.3))
        return x, y

    def test_partially_unbounded_bounds_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="finite lower AND upper bound"):
            fit(
                _exp_model,
                x,
                y,
                p0=[1, 1],
                workflow="auto_global",
                bounds=([0, -np.inf], [10, np.inf]),
            )

    def test_inverted_bounds_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="inverted"):
            fit(
                _exp_model,
                x,
                y,
                p0=[1, 1],
                workflow="auto_global",
                bounds=([10, 0], [0, 10]),
            )

    def test_overflowing_range_bounds_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="bound range"):
            fit(
                _exp_model,
                x,
                y,
                p0=[1, 1],
                workflow="auto_global",
                bounds=([-1e308, 0], [1e308, 10]),
            )

    def test_hpc_bounds_error_names_hpc_not_auto_global(self):
        x, y = self._data()
        with pytest.raises(ValueError, match=r"workflow='hpc'") as excinfo:
            fit(
                _exp_model,
                x,
                y,
                p0=[1, 1],
                workflow="hpc",
                bounds=([0, -np.inf], [10, np.inf]),
            )
        assert "auto_global" not in str(excinfo.value)


class TestMultiStartWinnerWarning:
    def test_winning_start_covariance_warning_is_not_swallowed(self):
        """A single-start fit with more parameters than data points can't
        have its covariance estimated (ysize <= p0.size) -- the resulting
        OptimizeWarning must reach the caller, not be silently dropped by
        the per-start suppression filter meant only for the N-1 losers."""

        def model(x, a, b, c):
            return a * x**2 + b * x + c

        x = np.array([0.0, 1.0])
        y = np.array([1.0, 2.0])

        config = GlobalOptimizationConfig(n_starts=1, sampler="lhs")
        orchestrator = MultiStartOrchestrator(config=config)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = orchestrator.fit(
                f=model,
                xdata=x,
                ydata=y,
                p0=np.array([1.0, 1.0, 1.0]),
                bounds=([-10, -10, -10], [10, 10, 10]),
            )

        assert result is not None
        assert any(issubclass(w.category, OptimizeWarning) for w in caught), (
            "winning start's OptimizeWarning (covariance could not be "
            "estimated) was not surfaced to the caller"
        )


class TestTournamentSelectorFailureHandling:
    def test_all_non_finite_losses_raises_instead_of_silent_fallback(self):
        candidates = [np.array([1.0]), np.array([2.0])]
        config = GlobalOptimizationConfig(elimination_rounds=1)
        selector = TournamentSelector(candidates=candidates, config=config)
        selector.survival_mask[:] = False

        with pytest.raises(RuntimeError, match="no valid survivors"):
            selector.get_top_candidates(top_m=1)

    def test_per_round_elimination_never_drops_below_top_m(self):
        n_candidates = 4
        top_m = 3
        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=1,
            elimination_fraction=0.5,
            batches_per_round=1,
        )
        candidates = [np.array([float(i)]) for i in range(n_candidates)]
        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a):
            return a * x

        def data_batch_generator():
            x_batch = np.linspace(0, 1, 20)
            y_batch = 2.0 * x_batch
            while True:
                yield x_batch, y_batch

        selector._run_single_round(
            data_batch_generator(),
            model,
            round_number=0,
            top_m=top_m,
        )

        assert selector.n_survivors >= top_m


class TestSamplingP0Clamping:
    def test_out_of_bounds_p0_is_clamped_before_centering(self):
        import jax.numpy as jnp

        samples = jnp.array([[0.0], [0.5], [1.0]])
        p0 = jnp.array([50.0])
        lb = jnp.array([0.0])
        ub = jnp.array([10.0])

        centered = center_samples_around_p0(samples, p0, 0.5, lb, ub)

        assert bool(jnp.all(centered >= lb))
        assert bool(jnp.all(centered <= ub))


class TestLargeDatasetFitterFailurePcov:
    def test_single_chunk_failure_reports_inf_pcov_not_identity(self):
        fitter = LargeDatasetFitter(memory_limit_gb=8.0)

        def bad_model(x, a, b):
            raise RuntimeError("deliberate failure")

        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        result = fitter._fit_single_chunk(
            bad_model,
            x,
            y,
            p0=np.array([1.0, 1.0]),
            bounds=(-np.inf, np.inf),
            method="trf",
            solver="auto",
        )

        assert result.success is False
        assert np.all(np.isinf(result.pcov)), (
            "a failed chunk fit must report pcov as all-inf (not "
            "estimated), not an identity matrix that reads as a "
            "legitimate unit-variance result"
        )


class TestCMAESRefinementNotImplementedPropagation:
    def test_not_implemented_error_from_curve_fit_is_not_swallowed(
        self,
        monkeypatch,
    ):
        import nlsq.core.minpack as minpack_mod

        def fake_curve_fit(*args, **kwargs):
            raise NotImplementedError("simulated unsupported combination")

        monkeypatch.setattr(minpack_mod, "curve_fit", fake_curve_fit)

        optimizer = CMAESOptimizer(config=CMAESConfig(max_generations=3))

        with pytest.raises(NotImplementedError):
            optimizer._nlsq_refinement(
                f=lambda x, a: a * x,
                xdata=np.linspace(0, 1, 10),
                ydata=np.linspace(0, 1, 10),
                p0=np.array([1.0]),
                bounds=(np.array([-10.0]), np.array([10.0])),
                sigma=None,
            )
