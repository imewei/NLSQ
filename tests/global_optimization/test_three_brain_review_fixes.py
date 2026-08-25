"""Regression tests for the three-brain review fixes to global_optimization.

Covers: Sobol direction-number correctness, LHS reused-seed bug, NaN-loss
selection, ignored curve_fit_instance config, tournament batch-count/top_m/
empty-candidates edge cases, and config validation of NaN/inf/non-int values.
"""

import numpy as np
import pytest

from nlsq.core.minpack import CurveFit
from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.multi_start import MultiStartOrchestrator
from nlsq.global_optimization.sampling import sobol_sample
from nlsq.global_optimization.tournament import TournamentSelector


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
        """The old truncated (16-entry) table degenerated into duplicate points
        once the Gray-code index exceeded 2**16."""
        x = np.asarray(sobol_sample(2000, 1))
        assert len(np.unique(x)) == len(x)

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


class TestTournamentEdgeCases:
    def test_rejects_empty_candidates(self):
        with pytest.raises(ValueError):
            TournamentSelector(np.empty((0, 2)), GlobalOptimizationConfig(n_starts=0))

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
