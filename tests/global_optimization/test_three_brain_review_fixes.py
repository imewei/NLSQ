"""Regression tests for the three-brain review fixes to global_optimization.

Covers: Sobol direction-number correctness, LHS reused-seed bug, NaN-loss
selection, ignored curve_fit_instance config (including cache_config),
tournament batch-count/top_m/empty-candidates edge cases, and config
validation of NaN/inf/non-int/bool values.
"""

import inspect

import numpy as np
import pytest

from nlsq.core.minpack import CurveFit
from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.multi_start import (
    MultiStartOrchestrator,
    _fit_single_start,
)
from nlsq.global_optimization.sampling import sobol_sample
from nlsq.global_optimization.tournament import TournamentSelector
from nlsq.result.curve_fit_result import CurveFitResult


class TestSobolCorrectness:
    def test_matches_reference_sobol_sequence(self):
        """First points must match the canonical Sobol/Antonov-Saleev sequence."""
        x = np.asarray(sobol_sample(8, 3))
        expected_dim0 = [0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125]
        np.testing.assert_allclose(x[:, 0], expected_dim0, atol=1e-9)
        # Dimension 2 (d=2 in the Joe-Kuo table) must not collapse to
        # duplicates of dimension 1, as it did with the fabricated table.
        assert not np.allclose(x[:, 1], x[:, 0])

    def test_no_duplicate_points_past_16_bit_boundary(self):
        """The old truncated (16-entry) table degenerated into duplicate
        points once the Gray-code index exceeded 2**16 -- n_samples must
        actually cross that boundary (bit position 16) or this test would
        pass against the pre-fix buggy table too."""
        n_samples = 70_000
        assert (n_samples - 1).bit_length() > 16
        x = np.asarray(sobol_sample(n_samples, 1))
        assert len(np.unique(x)) == len(x)

    def test_no_duplicate_points_multi_dim_polynomial_branch(self):
        """The 16-entry truncation applied to every dimension, not just
        dimension 1 -- verify the polynomial-recurrence branch (dims >= 2)
        also stays duplicate-free past the same 2**16 boundary."""
        n_samples = 70_000
        assert (n_samples - 1).bit_length() > 16
        x = np.asarray(sobol_sample(n_samples, 5))
        unique_rows = {tuple(row) for row in x}
        assert len(unique_rows) == len(x)

    def test_max_dims_still_supported(self):
        y = sobol_sample(4, 21)
        assert y.shape == (4, 21)
        with pytest.raises(ValueError):
            sobol_sample(4, 22)


class TestMultiStartRngVariation:
    def test_lhs_starts_vary_across_calls(self):
        """Repeated fit()-driven sample generation must not silently reuse
        the exact same LHS points every time."""
        orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=6))
        a = orch._generate_starting_points(2, np.zeros(2), np.ones(2))
        b = orch._generate_starting_points(2, np.zeros(2), np.ones(2))
        assert not np.array_equal(a, b)

    def test_explicit_rng_key_still_honored(self):
        """An explicitly passed rng_key must still be reproducible."""
        import jax

        orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=6))
        key = jax.random.PRNGKey(42)
        a = orch._generate_starting_points(2, np.zeros(2), np.ones(2), rng_key=key)
        b = orch._generate_starting_points(2, np.zeros(2), np.ones(2), rng_key=key)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


class TestCurveFitInstancePropagation:
    def test_custom_instance_config_propagates_to_workers(self):
        cf = CurveFit(enable_stability=True, flength=128)
        orch = MultiStartOrchestrator(
            GlobalOptimizationConfig(n_starts=2),
            curve_fit_instance=cf,
        )
        cfg = orch._curve_fit_config()
        assert cfg["enable_stability"] is True
        assert cfg["flength"] == 128

    def test_cache_config_propagates(self):
        """cache_config is a real CurveFit.__init__ param that CurveFit didn't
        even store on self -- it was structurally impossible to recover
        until CurveFit.__init__ started keeping self.cache_config."""
        cf = CurveFit(cache_config={"maxsize": 7})
        orch = MultiStartOrchestrator(
            GlobalOptimizationConfig(n_starts=2),
            curve_fit_instance=cf,
        )
        cfg = orch._curve_fit_config()
        assert cfg["cache_config"] == {"maxsize": 7}

    def test_curve_fit_config_keyset_matches_constructor_signature(self):
        """A hand-maintained dict silently drops new CurveFit.__init__
        params (as happened with cache_config) -- pin the keyset against
        the real signature so future drift is a loud test failure."""
        expected = set(inspect.signature(CurveFit.__init__).parameters) - {"self"}
        orch = MultiStartOrchestrator(GlobalOptimizationConfig(n_starts=2))
        assert set(orch._curve_fit_config().keys()) == expected

    @pytest.mark.parametrize("forced_workers", [1, 2])
    def test_worker_threads_actually_receive_custom_config(
        self,
        monkeypatch,
        forced_workers,
    ):
        """Regression for the actual bug: workers used to always build a
        bare default CurveFit(), silently ignoring curve_fit_instance.
        Exercises both the sequential (n_workers<=1) and parallel
        (ThreadPoolExecutor) code paths in evaluate_starting_points."""
        received_kwargs = []

        class _RecordingCurveFit(CurveFit):
            def __init__(self, **kwargs):
                received_kwargs.append(kwargs)
                super().__init__(**kwargs)

        monkeypatch.setattr("nlsq.core.minpack.CurveFit", _RecordingCurveFit)
        monkeypatch.setattr(
            "nlsq.global_optimization.multi_start._select_worker_count",
            lambda n_starts: forced_workers,
        )

        def model(x, a, b):
            return a * x + b

        x = np.linspace(0, 5, 20)
        y = 2 * x + 1
        starting_points = np.tile([1.0, 1.0], (2, 1))
        bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))

        cf = CurveFit(enable_stability=True)
        orch = MultiStartOrchestrator(
            GlobalOptimizationConfig(n_starts=2),
            curve_fit_instance=cf,
        )
        received_kwargs.clear()  # drop the constructor call for `cf` itself
        orch.evaluate_starting_points(model, x, y, starting_points, bounds)

        assert len(received_kwargs) == 2
        assert all(k.get("enable_stability") is True for k in received_kwargs)


class TestNaNLossClamp:
    def test_nan_cost_is_clamped_to_inf_not_left_as_nan(self, monkeypatch):
        """A NaN loss must not be able to sort ahead of a genuinely
        successful finite-loss fit (Python's sort gives no useful ordering
        for NaN)."""

        class _NaNCostCurveFit:
            def __init__(self, **kwargs):
                pass

            def curve_fit(self, f, xdata, ydata, p0, bounds, **kwargs):
                result = CurveFitResult()
                result.popt = np.asarray(p0)
                result.cost = float("nan")
                return result

        monkeypatch.setattr("nlsq.core.minpack.CurveFit", _NaNCostCurveFit)

        def model(x, a):
            return a * x

        _params, loss, result = _fit_single_start(
            model,
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([1.0]),
            (np.array([-10.0]), np.array([10.0])),
            {},
        )
        assert loss == float("inf")
        assert not np.isnan(loss)
        assert result is not None  # the fit "succeeded", just with NaN cost


class TestTournamentEdgeCases:
    def test_rejects_empty_candidates(self):
        with pytest.raises(ValueError):
            TournamentSelector(np.empty((0, 2)), GlobalOptimizationConfig(n_starts=0))

    def test_rejects_empty_1d_candidates(self):
        """A 1-D empty array (ndim==1) used to reshape to (1, 0) *before*
        the old post-reshape `shape[0] == 0` check ran, so it slipped
        through as a degenerate single zero-parameter candidate instead of
        raising."""
        with pytest.raises(ValueError):
            TournamentSelector(np.array([]), GlobalOptimizationConfig(n_starts=0))

    def test_rejects_non_positive_top_m(self):
        candidates = np.arange(6).reshape(3, 2)
        selector = TournamentSelector(
            candidates,
            GlobalOptimizationConfig(n_starts=3, elimination_rounds=0),
        )
        with pytest.raises(ValueError):
            selector.get_top_candidates(-1)
        with pytest.raises(ValueError):
            selector.get_top_candidates(0)

    def test_rejects_non_integer_top_m(self):
        """A fractional top_m used to slip past the `top_m < 1` check and
        die downstream with a confusing TypeError from slicing instead of
        the intended ValueError."""
        candidates = np.arange(6).reshape(3, 2)
        selector = TournamentSelector(
            candidates,
            GlobalOptimizationConfig(n_starts=3, elimination_rounds=0),
        )
        with pytest.raises(ValueError):
            selector.get_top_candidates(2.5)

    def test_batch_count_not_overcounted_on_partial_round(self):
        """The old `min(batch_idx + 1, batches_per_round)` overcounted by
        one whenever the data iterator was exhausted mid-round (the failed
        fetch attempt got counted as a batch)."""

        def model(x, a):
            return a * x

        def two_batch_iterator():
            yield np.array([1.0, 2.0]), np.array([1.0, 2.0])
            yield np.array([1.0, 2.0]), np.array([1.0, 2.0])
            # exhausted after 2 batches; batches_per_round asks for 5

        candidates = np.array([[1.0], [2.0], [3.0], [4.0]])
        config = GlobalOptimizationConfig(
            n_starts=4,
            elimination_rounds=1,
            elimination_fraction=0.5,
            batches_per_round=5,
        )
        selector = TournamentSelector(candidates, config)
        selector.run_tournament(two_batch_iterator(), model, top_m=1)
        assert selector.round_history[0]["batches_evaluated"] == 2


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"scale_factor": float("nan")},
            {"scale_factor": float("inf")},
            {"elimination_fraction": float("nan")},
        ],
    )
    def test_rejects_non_finite_floats(self, kwargs):
        with pytest.raises(ValueError):
            GlobalOptimizationConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_starts": 1.5},
            {"elimination_rounds": 1.5},
            {"batches_per_round": 1.5},
        ],
    )
    def test_rejects_non_integer_counts(self, kwargs):
        with pytest.raises(TypeError):
            GlobalOptimizationConfig(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_starts": True},
            {"elimination_rounds": False},
            {"batches_per_round": True},
        ],
    )
    def test_rejects_bool_for_int_fields(self, kwargs):
        """bool is an int subclass in Python -- True/False must not
        silently pass as valid n_starts/elimination_rounds/batches_per_round."""
        with pytest.raises(TypeError):
            GlobalOptimizationConfig(**kwargs)
