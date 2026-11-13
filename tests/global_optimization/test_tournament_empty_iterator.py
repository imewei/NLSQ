"""GD-5: an empty batch iterator must not be blamed on non-finite losses."""

import numpy as np
import pytest

from nlsq.global_optimization.config import GlobalOptimizationConfig
from nlsq.global_optimization.tournament import TournamentSelector


def test_empty_iterator_reports_no_batches_evaluated():
    selector = TournamentSelector(np.ones((4, 2)), GlobalOptimizationConfig(n_starts=4))
    with pytest.raises(RuntimeError, match="no data batches were evaluated"):
        selector.run_tournament(iter([]), lambda x, a, b: a * x + b)
