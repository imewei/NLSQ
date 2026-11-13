import numpy as np
import pytest

from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.tournament import TournamentSelector


def test_tournament_ranks_candidates_when_top_m_covers_all():
    candidates = np.array([[4.0], [1.0]])
    config = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=1, batches_per_round=1
    )
    selector = TournamentSelector(candidates, config)
    batch = iter([(np.ones(2), np.ones(2))])

    result = selector.run_tournament(batch, lambda x, a: a * x, top_m=2)

    np.testing.assert_array_equal(result, [[1.0], [4.0]])


def test_tournament_rejects_all_nonfinite_losses():
    config = GlobalOptimizationConfig(
        n_starts=2, elimination_rounds=1, batches_per_round=1
    )
    selector = TournamentSelector(np.array([[1.0], [2.0]]), config)
    batch = iter([(np.ones(2), np.ones(2))])

    with pytest.raises(RuntimeError, match="no valid survivors"):
        selector.run_tournament(batch, lambda x, a: x * np.nan, top_m=1)
