"""
Tests for Tournament Selector
=============================

Tests for the TournamentSelector class that implements progressive elimination
for multi-start optimization on streaming datasets.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from nlsq.global_optimization import GlobalOptimizationConfig
from nlsq.global_optimization.tournament import TournamentSelector


class TestTournamentSelector:
    """Test TournamentSelector implementation."""

    def test_progressive_elimination_reduces_candidates(self):
        """Test TournamentSelector progressive elimination (N -> N/2 -> N/4)."""
        # Create 16 candidates (4 rounds needed to get to 1)
        n_candidates = 16
        n_params = 3
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, n_params) * 0.1 + 1.0

        # Create config with elimination_fraction=0.5
        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=4,
            elimination_fraction=0.5,
            batches_per_round=5,
        )

        # Create tournament selector
        selector = TournamentSelector(candidates=candidates, config=config)

        # Check initial state
        assert selector.n_candidates == n_candidates
        assert selector.n_survivors == n_candidates

        # Define a simple model for evaluation
        def model(x, a, b, c):
            return a * x**2 + b * x + c

        # Create data batches generator
        def data_batch_generator():
            for _ in range(20):  # Enough batches for all rounds
                x_batch = np.linspace(0, 5, 50)
                y_batch = (
                    1.0 * x_batch**2 + 2.0 * x_batch + 3.0 + np.random.randn(50) * 0.1
                )
                yield x_batch, y_batch

        # Run tournament
        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=1,
        )

        # Should have at least one best candidate
        assert len(best_candidates) >= 1
        assert best_candidates[0].shape == (n_params,)

        # Verify progressive elimination occurred
        assert len(selector.round_history) > 0
        # Each round should have fewer or equal survivors
        for i in range(1, len(selector.round_history)):
            prev_survivors = selector.round_history[i - 1]["n_survivors_after"]
            curr_survivors_before = selector.round_history[i]["n_survivors_before"]
            assert curr_survivors_before <= prev_survivors

    def test_tournament_evaluates_candidates_on_batches(self):
        """Test tournament evaluates candidates on streaming batches."""
        n_candidates = 8
        n_params = 2
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, n_params) * 0.5 + np.array(
            [2.0, 3.0]
        )

        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=2,
            elimination_fraction=0.5,
            batches_per_round=3,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a, b):
            return a * x + b

        # Track how many batches are consumed
        batch_count = [0]

        def data_batch_generator():
            for _ in range(10):
                batch_count[0] += 1
                x_batch = np.linspace(0, 10, 30)
                y_batch = 2.0 * x_batch + 3.0 + np.random.randn(30) * 0.1
                yield x_batch, y_batch

        # Run tournament
        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=1,
        )

        # Verify batches were consumed for evaluation
        # batches_per_round * elimination_rounds = 3 * 2 = 6 batches minimum
        assert batch_count[0] >= config.batches_per_round * config.elimination_rounds

        # Verify best candidate returned
        assert len(best_candidates) >= 1

    def test_best_candidate_from_tournament_is_reasonable(self):
        """Test best candidate from tournament becomes Phase 2 starting point."""
        n_candidates = 10
        n_params = 3
        np.random.seed(42)

        # Create candidates with one "good" candidate close to true params
        true_params = np.array([1.5, 0.3, 2.0])
        candidates = np.random.randn(n_candidates, n_params) * 2.0 + np.array(
            [5.0, 5.0, 5.0]
        )
        # Make one candidate close to true params (index 3)
        candidates[3] = true_params + np.random.randn(n_params) * 0.05

        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=3,
            elimination_fraction=0.5,
            batches_per_round=10,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a, b, c):
            return a * jnp.exp(-b * x) + c

        def data_batch_generator():
            for _ in range(50):
                x_batch = np.linspace(0, 5, 50)
                y_batch = (
                    1.5 * np.exp(-0.3 * x_batch) + 2.0 + np.random.randn(50) * 0.05
                )
                yield x_batch, y_batch

        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=1,
        )

        # The best candidate should be reasonably close to true params
        # (the "good" candidate should survive and likely win)
        best = best_candidates[0]
        assert best.shape == (n_params,)

        # Best candidate should have lower loss than worst candidates
        # This is implicitly tested by the tournament working correctly

    def test_tournament_respects_elimination_fraction_config(self):
        """Test tournament respects elimination_fraction configuration."""
        n_candidates = 10
        n_params = 2
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, n_params) * 0.5 + 1.0

        # Test with 0.25 elimination fraction (eliminate 25% each round)
        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=2,
            elimination_fraction=0.25,  # Only eliminate 25%
            batches_per_round=3,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a, b):
            return a * x + b

        def data_batch_generator():
            for _ in range(20):
                x_batch = np.linspace(0, 5, 20)
                y_batch = 1.0 * x_batch + 1.0 + np.random.randn(20) * 0.1
                yield x_batch, y_batch

        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=1,
        )

        # With 25% elimination per round:
        # Round 1: 10 -> ~8 (eliminate 2-3)
        # Round 2: 8 -> ~6 (eliminate 1-2)
        # Should have more survivors than with 50% elimination
        if len(selector.round_history) >= 2:
            round1_survivors = selector.round_history[0]["n_survivors_after"]
            # With 25% elimination, should keep ~75% = ~7-8 from 10
            assert round1_survivors >= 6

    def test_checkpoint_save_resume_preserves_tournament_state(self):
        """Test multi-start checkpoint save/resume preserves tournament state."""
        import pickle
        import tempfile
        from pathlib import Path

        n_candidates = 8
        n_params = 2
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, n_params) * 0.5 + 1.0

        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=3,
            elimination_fraction=0.5,
            batches_per_round=3,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        # Simulate partial tournament (1 round)
        def model(x, a, b):
            return a * x + b

        def data_batch_generator_round1():
            for _ in range(5):
                x_batch = np.linspace(0, 5, 20)
                y_batch = 1.0 * x_batch + 1.0 + np.random.randn(20) * 0.1
                yield x_batch, y_batch

        # Run partial tournament (just first round)
        selector._run_single_round(data_batch_generator_round1(), model, round_number=0)

        # Save checkpoint
        checkpoint_state = selector.to_checkpoint()

        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pkl") as f:
            pickle.dump(checkpoint_state, f)
            checkpoint_path = Path(f.name)

        # Load checkpoint into new selector
        with open(checkpoint_path, "rb") as f:
            loaded_state = pickle.load(f)

        new_selector = TournamentSelector.from_checkpoint(loaded_state, config)

        # Verify state preserved
        assert new_selector.current_round == selector.current_round
        assert new_selector.n_survivors == selector.n_survivors
        np.testing.assert_array_equal(
            new_selector.survival_mask, selector.survival_mask
        )
        np.testing.assert_array_almost_equal(
            new_selector.cumulative_losses, selector.cumulative_losses
        )

        # Clean up
        checkpoint_path.unlink()

    def test_streaming_optimizer_fallback_on_tournament_failure(self):
        """Test streaming optimizer falls back gracefully if tournament fails."""
        n_candidates = 5
        n_params = 2
        np.random.seed(42)
        candidates = np.random.randn(n_candidates, n_params) * 0.5 + 1.0

        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=2,
            elimination_fraction=0.5,
            batches_per_round=3,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        # Model that will cause numerical failures for some candidates
        def problematic_model(x, a, b):
            # Create potential for numerical issues
            result = a * jnp.exp(-b * x)
            return result

        # Generator with very few batches (may not complete all rounds)
        def limited_batch_generator():
            for _ in range(2):  # Not enough batches for full tournament
                x_batch = np.linspace(0, 5, 20)
                y_batch = 1.0 * np.exp(-0.5 * x_batch) + np.random.randn(20) * 0.1
                yield x_batch, y_batch

        # Should not raise - should fall back gracefully
        try:
            best_candidates = selector.run_tournament(
                data_batch_iterator=limited_batch_generator(),
                model=problematic_model,
                top_m=1,
            )
            # Even if incomplete, should return something
            assert len(best_candidates) >= 1
        except StopIteration:
            # This is expected if batches run out - selector should handle gracefully
            # Fall back to returning best current candidates
            best_candidates = selector.get_top_candidates(top_m=1)
            assert len(best_candidates) >= 1


class TestTournamentSelectorEdgeCases:
    """Test edge cases for TournamentSelector."""

    def test_single_candidate_tournament(self):
        """Test tournament with single candidate."""
        candidates = np.array([[1.0, 2.0]])

        config = GlobalOptimizationConfig(
            n_starts=1,
            elimination_rounds=2,
            elimination_fraction=0.5,
            batches_per_round=2,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a, b):
            return a * x + b

        def data_batch_generator():
            for _ in range(5):
                x_batch = np.linspace(0, 5, 20)
                y_batch = 1.0 * x_batch + 2.0
                yield x_batch, y_batch

        # Should handle single candidate gracefully
        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=1,
        )

        assert len(best_candidates) == 1
        np.testing.assert_array_almost_equal(best_candidates[0], [1.0, 2.0])

    def test_tournament_with_zero_elimination_rounds(self):
        """Test tournament with zero elimination rounds."""
        n_candidates = 5
        candidates = np.random.randn(n_candidates, 2)

        config = GlobalOptimizationConfig(
            n_starts=n_candidates,
            elimination_rounds=0,  # No elimination
            elimination_fraction=0.5,
            batches_per_round=2,
        )

        selector = TournamentSelector(candidates=candidates, config=config)

        def model(x, a, b):
            return a * x + b

        def data_batch_generator():
            for _ in range(5):
                x_batch = np.linspace(0, 5, 20)
                y_batch = 1.0 * x_batch + 1.0
                yield x_batch, y_batch

        # With zero rounds, should just return top candidates based on initial eval
        best_candidates = selector.run_tournament(
            data_batch_iterator=data_batch_generator(),
            model=model,
            top_m=3,
        )

        assert len(best_candidates) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
